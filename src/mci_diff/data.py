"""Manifest, longitudinal records, split files and task masks."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


VISIT_MONTHS = (0, 6, 12, 18, 24, 36)
VISIT_TO_INDEX = {month: index for index, month in enumerate(VISIT_MONTHS)}
ALLOWED_DATASETS = ("ADNI1", "ADNI2", "AIBL")
REQUIRED_MANIFEST_COLUMNS = (
    "subject_id",
    "dataset",
    "month",
    "label",
    "feature_path",
)
OPTIONAL_PATH_COLUMNS = ("image_path",)
FEATURE_SUFFIXES = (".npy", ".npz")
SPLIT_SCHEMA_VERSION = 1


Scalar = Union[str, int, float, bool, None]


def _clean_text(value: Any, name: str, row_number: int) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{name} is empty on manifest row {row_number}")
    return text


def _resolve_path(value: Any, root: Path, name: str, row_number: int) -> Path:
    text = _clean_text(value, name, row_number)
    source = Path(text).expanduser()
    return (source if source.is_absolute() else root / source).resolve()


def _optional_path(value: Any, root: Path) -> Optional[Path]:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    source = Path(text).expanduser()
    return (source if source.is_absolute() else root / source).resolve()


def _demographic_value(value: Any) -> Scalar:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.casefold()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        raise ValueError(f"Demographic value must be finite, got {text!r}")
    return int(number) if number.is_integer() else number


def _parse_month(value: Any, row_number: int, visits: Sequence[int]) -> int:
    text = _clean_text(value, "month", row_number)
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(f"month must be numeric on manifest row {row_number}") from error
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"month must be an integer on manifest row {row_number}")
    month = int(number)
    if month not in visits:
        raise ValueError(
            f"month {month} on manifest row {row_number} is not in {list(visits)}"
        )
    return month


def _parse_label(value: Any, positive: str, negative: str, row_number: int) -> Tuple[int, str]:
    text = _clean_text(value, "label", row_number)
    folded = text.casefold()
    positive_values = {positive.casefold(), "1", "true"}
    negative_values = {negative.casefold(), "0", "false"}
    if folded in positive_values:
        return 1, positive
    if folded in negative_values:
        return 0, negative
    raise ValueError(
        f"label {text!r} on manifest row {row_number} is neither "
        f"{positive!r} nor {negative!r}"
    )


@dataclass(frozen=True)
class ScanRecord:
    subject_id: str
    dataset: str
    month: int
    label: int
    label_name: str
    feature_path: Path
    image_path: Optional[Path] = None
    demographics: Mapping[str, Scalar] = field(default_factory=dict)
    source_row: int = 0

    @property
    def visit_index(self) -> int:
        return VISIT_TO_INDEX[self.month]


# Kept as an alias for older data scripts.
ManifestRow = ScanRecord


def read_manifest(
    path: Union[str, Path],
    *,
    project_root: Optional[Union[str, Path]] = None,
    visit_months: Sequence[int] = VISIT_MONTHS,
    allowed_datasets: Sequence[str] = ALLOWED_DATASETS,
    positive_label: str = "pMCI",
    negative_label: str = "sMCI",
    check_paths: bool = False,
    require_image: bool = False,
    require_baseline: bool = True,
) -> List[ScanRecord]:
    """Read one scan per row. Labels are supplied by the cohort table."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    visits = tuple(int(month) for month in visit_months)
    if visits != VISIT_MONTHS:
        raise ValueError(f"visit_months must be exactly {list(VISIT_MONTHS)}")
    datasets = tuple(str(item).strip() for item in allowed_datasets)
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("allowed_datasets must be a non-empty unique sequence")
    if not positive_label.strip() or not negative_label.strip():
        raise ValueError("positive_label and negative_label cannot be empty")
    if positive_label.casefold() == negative_label.casefold():
        raise ValueError("positive_label and negative_label must differ")
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else manifest_path.parent
    )

    records: List[ScanRecord] = []
    scan_keys: set[Tuple[str, int]] = set()
    subject_labels: Dict[str, int] = {}
    subject_datasets: Dict[str, str] = {}
    path_owners: Dict[Path, Tuple[str, int, str]] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Manifest has no header")
        fieldnames = [str(name).strip() for name in reader.fieldnames]
        if len(set(fieldnames)) != len(fieldnames):
            raise ValueError("Manifest header contains duplicate column names")
        reader.fieldnames = fieldnames
        missing = sorted(set(REQUIRED_MANIFEST_COLUMNS) - set(fieldnames))
        if missing:
            raise ValueError(f"Manifest is missing columns: {missing}")
        reserved = {
            "subject_id",
            "dataset",
            "month",
            "image_path",
            "label",
            "feature_path",
        }
        for row_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError(f"Manifest row {row_number} has more fields than the header")
            subject_id = _clean_text(raw.get("subject_id"), "subject_id", row_number)
            dataset = _clean_text(raw.get("dataset"), "dataset", row_number)
            if dataset not in datasets:
                raise ValueError(
                    f"dataset {dataset!r} on manifest row {row_number} is not allowed"
                )
            month = _parse_month(raw.get("month"), row_number, visits)
            scan_key = (subject_id, month)
            if scan_key in scan_keys:
                raise ValueError(f"Duplicate subject-month entry: {subject_id} at month {month}")
            scan_keys.add(scan_key)
            label, label_name = _parse_label(
                raw.get("label"), positive_label, negative_label, row_number
            )
            old_label = subject_labels.setdefault(subject_id, label)
            if old_label != label:
                raise ValueError(f"Conflicting labels for subject {subject_id}")
            old_dataset = subject_datasets.setdefault(subject_id, dataset)
            if old_dataset != dataset:
                raise ValueError(
                    f"Subject {subject_id} appears in both {old_dataset} and {dataset}"
                )

            feature_path = _resolve_path(
                raw.get("feature_path"), root, "feature_path", row_number
            )
            image_path = _optional_path(raw.get("image_path"), root)
            if feature_path.suffix.casefold() not in FEATURE_SUFFIXES:
                raise ValueError(
                    f"feature_path must end in .npy or .npz on row {row_number}"
                )
            if require_image and image_path is None:
                raise ValueError(f"image_path is required on manifest row {row_number}")
            for source_path, path_name in (
                (feature_path, "feature_path"),
                (image_path, "image_path"),
            ):
                if source_path is None:
                    continue
                owner = path_owners.get(source_path)
                if owner is not None:
                    raise ValueError(
                        f"Resolved {path_name} is reused by {subject_id} month {month}; "
                        f"it already belongs to {owner[0]} month {owner[1]} as {owner[2]}: "
                        f"{source_path}"
                    )
                path_owners[source_path] = (subject_id, month, path_name)
                if check_paths and not source_path.is_file():
                    raise FileNotFoundError(
                        f"{path_name} does not exist for {subject_id} month {month}: "
                        f"{source_path}"
                    )
            demographics = {
                name: _demographic_value(raw.get(name))
                for name in fieldnames
                if name not in reserved
            }
            records.append(
                ScanRecord(
                    subject_id=subject_id,
                    dataset=dataset,
                    month=month,
                    label=label,
                    label_name=label_name,
                    feature_path=feature_path,
                    image_path=image_path,
                    demographics=demographics,
                    source_row=row_number,
                )
            )
    if not records:
        raise ValueError("Manifest contains no scans")
    if require_baseline:
        baseline_subjects = {record.subject_id for record in records if record.month == 0}
        missing_baseline = sorted(set(subject_labels) - baseline_subjects)
        if missing_baseline:
            raise ValueError(
                "Subjects without a baseline visit: " + ", ".join(missing_baseline[:10])
            )
    records.sort(key=lambda item: (item.dataset, item.subject_id, item.month))
    return records


def _check_file_size(path: Path, maximum_bytes: int, kind: str) -> None:
    if maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive")
    if not path.is_file():
        raise FileNotFoundError(f"{kind} file does not exist: {path}")
    if path.stat().st_size > maximum_bytes:
        raise ValueError(f"{kind} file is larger than the configured limit: {path}")


def load_feature_array(
    path: Union[str, Path],
    *,
    expected_dim: Optional[int] = None,
    array_key: Optional[str] = None,
    maximum_bytes: int = 512 * 1024 * 1024,
) -> np.ndarray:
    """Load a numeric vector without pickle deserialization."""

    source = Path(path).expanduser().resolve()
    _check_file_size(source, maximum_bytes, "Feature")
    suffix = source.suffix.casefold()
    if suffix == ".npy":
        array = np.load(source, allow_pickle=False)
    elif suffix == ".npz":
        if not zipfile.is_zipfile(source):
            raise ValueError(f"Invalid NPZ archive: {source}")
        with zipfile.ZipFile(source, "r") as zipped:
            expanded_bytes = sum(item.file_size for item in zipped.infolist())
        if expanded_bytes > maximum_bytes:
            raise ValueError(f"Expanded NPZ archive exceeds the size limit: {source}")
        with np.load(source, allow_pickle=False) as archive:
            keys = list(archive.files)
            if array_key is not None:
                if array_key not in archive:
                    raise KeyError(f"Feature key {array_key!r} not found in {source}")
                array = np.asarray(archive[array_key])
            elif "feature" in archive:
                array = np.asarray(archive["feature"])
            elif "features" in archive:
                array = np.asarray(archive["features"])
            elif len(keys) == 1:
                array = np.asarray(archive[keys[0]])
            else:
                raise ValueError(
                    f"NPZ feature file must contain feature/features or one array: {source}"
                )
    else:
        raise ValueError(f"Feature file must be .npy or .npz: {source}")
    array = np.asarray(array)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"Feature array must have shape [D], got {array.shape}: {source}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"Feature array must be real-valued numeric data: {source}")
    if array.nbytes > maximum_bytes:
        raise ValueError(f"Uncompressed feature array exceeds the size limit: {source}")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"Feature array contains NaN or infinity: {source}")
    if expected_dim is not None and array.shape != (int(expected_dim),):
        raise ValueError(
            f"Expected feature dimension {expected_dim}, got {array.shape[0]}: {source}"
        )
    return np.ascontiguousarray(array)


@dataclass
class SequenceRecord:
    subject_id: str
    dataset: str
    label: int
    features: np.ndarray
    observed: np.ndarray
    imputed: np.ndarray = field(default_factory=lambda: np.zeros(len(VISIT_MONTHS), dtype=bool))
    demographics: Tuple[Mapping[str, Scalar], ...] = field(
        default_factory=lambda: tuple({} for _ in VISIT_MONTHS)
    )
    image_paths: Tuple[Optional[Path], ...] = field(
        default_factory=lambda: tuple(None for _ in VISIT_MONTHS)
    )
    feature_paths: Tuple[Optional[Path], ...] = field(
        default_factory=lambda: tuple(None for _ in VISIT_MONTHS)
    )

    def __post_init__(self) -> None:
        self.subject_id = str(self.subject_id).strip()
        self.dataset = str(self.dataset).strip()
        self.label = int(self.label)
        self.features = np.ascontiguousarray(np.asarray(self.features, dtype=np.float32))
        self.observed = np.asarray(self.observed, dtype=bool).copy()
        self.imputed = np.asarray(self.imputed, dtype=bool).copy()
        self.demographics = tuple(copy.deepcopy(dict(item)) for item in self.demographics)
        self.image_paths = tuple(self.image_paths)
        self.feature_paths = tuple(self.feature_paths)
        self.validate()

    @property
    def available(self) -> np.ndarray:
        return self.observed | self.imputed

    @property
    def complete(self) -> bool:
        return bool(self.observed.all())

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def validate(self) -> None:
        if not self.subject_id:
            raise ValueError("SequenceRecord.subject_id cannot be empty")
        if self.dataset not in ALLOWED_DATASETS:
            raise ValueError(f"Unsupported dataset {self.dataset!r}")
        if self.label not in (0, 1):
            raise ValueError("SequenceRecord.label must be 0 or 1")
        if self.features.ndim != 2 or self.features.shape[0] != len(VISIT_MONTHS):
            raise ValueError("features must have shape [6, D]")
        if self.features.shape[1] < 1:
            raise ValueError("features must have a positive feature dimension")
        if self.observed.shape != (len(VISIT_MONTHS),):
            raise ValueError("observed must have shape [6]")
        if self.imputed.shape != (len(VISIT_MONTHS),):
            raise ValueError("imputed must have shape [6]")
        if np.any(self.observed & self.imputed):
            raise ValueError("A visit cannot be both observed and imputed")
        if not np.isfinite(self.features).all():
            raise ValueError("Sequence features contain NaN or infinity")
        for name, values in (
            ("demographics", self.demographics),
            ("image_paths", self.image_paths),
            ("feature_paths", self.feature_paths),
        ):
            if len(values) != len(VISIT_MONTHS):
                raise ValueError(f"{name} must contain six visit entries")

    def clone(self) -> "SequenceRecord":
        return SequenceRecord(
            subject_id=self.subject_id,
            dataset=self.dataset,
            label=self.label,
            features=self.features.copy(),
            observed=self.observed.copy(),
            imputed=self.imputed.copy(),
            demographics=self.demographics,
            image_paths=self.image_paths,
            feature_paths=self.feature_paths,
        )

    def with_imputations(
        self,
        features: np.ndarray,
        mask: Sequence[bool],
    ) -> "SequenceRecord":
        return clone_with_imputations(self, features, mask)


def assemble_sequences(
    scans: Sequence[ScanRecord],
    *,
    feature_dim: int,
    feature_array_key: Optional[str] = None,
) -> List[SequenceRecord]:
    if int(feature_dim) < 1:
        raise ValueError("feature_dim must be positive")
    grouped: Dict[str, List[ScanRecord]] = defaultdict(list)
    for scan in scans:
        grouped[scan.subject_id].append(scan)
    sequences: List[SequenceRecord] = []
    for subject_id in sorted(grouped):
        subject_scans = sorted(grouped[subject_id], key=lambda item: item.month)
        datasets = {item.dataset for item in subject_scans}
        labels = {item.label for item in subject_scans}
        months = [item.month for item in subject_scans]
        if len(datasets) != 1 or len(labels) != 1:
            raise ValueError(f"Inconsistent cohort or label for subject {subject_id}")
        if len(months) != len(set(months)):
            raise ValueError(f"Duplicate visits for subject {subject_id}")
        features = np.zeros((len(VISIT_MONTHS), int(feature_dim)), dtype=np.float32)
        observed = np.zeros(len(VISIT_MONTHS), dtype=bool)
        demographics: List[Mapping[str, Scalar]] = [{} for _ in VISIT_MONTHS]
        image_paths: List[Optional[Path]] = [None for _ in VISIT_MONTHS]
        feature_paths: List[Optional[Path]] = [None for _ in VISIT_MONTHS]
        for scan in subject_scans:
            index = VISIT_TO_INDEX[scan.month]
            features[index] = load_feature_array(
                scan.feature_path,
                expected_dim=feature_dim,
                array_key=feature_array_key,
            )
            observed[index] = True
            demographics[index] = scan.demographics
            image_paths[index] = scan.image_path
            feature_paths[index] = scan.feature_path
        sequences.append(
            SequenceRecord(
                subject_id=subject_id,
                dataset=next(iter(datasets)),
                label=next(iter(labels)),
                features=features,
                observed=observed,
                demographics=tuple(demographics),
                image_paths=tuple(image_paths),
                feature_paths=tuple(feature_paths),
            )
        )
    if not sequences:
        raise ValueError("No sequences could be assembled")
    return sequences


# Common alias in early experiment scripts.
build_sequences = assemble_sequences


@dataclass
class FeatureNormalizer:
    mean: np.ndarray
    scale: np.ndarray
    epsilon: float = 1.0e-6
    fitted_partition: str = "train"
    count: int = 0

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float32).copy()
        self.scale = np.asarray(self.scale, dtype=np.float32).copy()
        self.epsilon = float(self.epsilon)
        self.count = int(self.count)
        if self.mean.ndim != 1 or self.scale.shape != self.mean.shape:
            raise ValueError("Normalizer mean and scale must have matching [D] shapes")
        if self.mean.size == 0 or not np.isfinite(self.mean).all():
            raise ValueError("Normalizer mean is empty or non-finite")
        if not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("Normalizer scale must be finite and positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Normalizer epsilon must be finite and positive")
        if self.fitted_partition != "train":
            raise ValueError("FeatureNormalizer may only be fitted on the train partition")

    @classmethod
    def fit(
        cls,
        records: Sequence[SequenceRecord],
        *,
        partition: str = "train",
        epsilon: float = 1.0e-6,
        include_imputed: bool = False,
    ) -> "FeatureNormalizer":
        if partition != "train":
            raise ValueError("Normalizer fitting is restricted to partition='train'")
        if not records:
            raise ValueError("Cannot fit a normalizer on an empty training partition")
        feature_dim = records[0].feature_dim
        total = np.zeros(feature_dim, dtype=np.float64)
        total_square = np.zeros(feature_dim, dtype=np.float64)
        count = 0
        for record in records:
            if record.feature_dim != feature_dim:
                raise ValueError("Training records have inconsistent feature dimensions")
            mask = record.available if include_imputed else record.observed
            batch = np.asarray(record.features[mask], dtype=np.float64)
            if batch.size == 0:
                continue
            total += batch.sum(axis=0)
            total_square += np.square(batch).sum(axis=0)
            count += int(batch.shape[0])
        if count == 0:
            raise ValueError("Training partition contains no observed feature vectors")
        mean = total / count
        variance = np.maximum(total_square / count - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[scale < float(epsilon)] = 1.0
        return cls(
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
            epsilon=epsilon,
            fitted_partition=partition,
            count=count,
        )

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != self.mean.shape[0]:
            raise ValueError("Feature array dimension does not match the normalizer")
        result = (array - self.mean) / self.scale
        if not np.isfinite(result).all():
            raise ValueError("Normalization produced a non-finite value")
        return np.asarray(result, dtype=np.float32)

    def inverse_transform_array(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape[-1] != self.mean.shape[0]:
            raise ValueError("Feature array dimension does not match the normalizer")
        return np.asarray(array * self.scale + self.mean, dtype=np.float32)

    def transform(self, record: SequenceRecord) -> SequenceRecord:
        transformed = record.clone()
        mask = transformed.available
        transformed.features[mask] = self.transform_array(transformed.features[mask])
        transformed.features[~mask] = 0.0
        return transformed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "epsilon": self.epsilon,
            "fitted_partition": self.fitted_partition,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureNormalizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            epsilon=float(payload.get("epsilon", 1.0e-6)),
            fitted_partition=str(payload.get("fitted_partition", "train")),
            count=int(payload.get("count", 0)),
        )


def fit_normalizer(
    records: Sequence[SequenceRecord],
    *,
    partition: str = "train",
    epsilon: float = 1.0e-6,
) -> FeatureNormalizer:
    return FeatureNormalizer.fit(records, partition=partition, epsilon=epsilon)


def partition_complete_sequences(
    records: Sequence[SequenceRecord],
) -> Tuple[List[SequenceRecord], List[SequenceRecord]]:
    complete = [record for record in records if record.complete]
    incomplete = [record for record in records if not record.complete]
    return complete, incomplete


def clone_with_imputations(
    record: SequenceRecord,
    features: np.ndarray,
    mask: Sequence[bool],
    *,
    allow_replace_imputed: bool = True,
) -> SequenceRecord:
    """Copy a sequence and add generated visits without changing observed data."""

    target_mask = np.asarray(mask, dtype=bool)
    if target_mask.shape != (len(VISIT_MONTHS),) or not target_mask.any():
        raise ValueError("Imputation mask must have shape [6] and select at least one visit")
    if np.any(target_mask & record.observed):
        raise ValueError("Imputation cannot overwrite a genuinely observed visit")
    if not allow_replace_imputed and np.any(target_mask & record.imputed):
        raise ValueError("Imputation mask selects an already imputed visit")

    selected = int(target_mask.sum())
    feature_values = np.asarray(features, dtype=np.float32)
    if feature_values.shape == record.features.shape:
        feature_values = feature_values[target_mask]
    if feature_values.shape != (selected, record.feature_dim):
        raise ValueError(
            f"Imputed features must have shape [{selected}, {record.feature_dim}] "
            f"or {record.features.shape}"
        )
    if not np.isfinite(feature_values).all():
        raise ValueError("Imputed features contain NaN or infinity")
    result = record.clone()
    result.features[target_mask] = feature_values
    result.imputed[target_mask] = True
    result.validate()
    return result


@dataclass(frozen=True)
class SplitDefinition:
    name: str
    protocol: str
    fold: int
    train: Tuple[str, ...]
    validation: Tuple[str, ...]
    test: Tuple[str, ...]
    external: Tuple[str, ...] = ()
    seed: int = 2026

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "protocol", str(self.protocol).strip())
        object.__setattr__(self, "fold", int(self.fold))
        object.__setattr__(self, "seed", int(self.seed))
        for partition in ("train", "validation", "test", "external"):
            identifiers = tuple(str(item).strip() for item in getattr(self, partition))
            object.__setattr__(self, partition, identifiers)
        self.validate()

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Split name cannot be empty")
        if self.protocol not in {"adni_transfer", "adni_5fold"}:
            raise ValueError(f"Unsupported split protocol {self.protocol!r}")
        if self.fold < 0 or self.seed < 0:
            raise ValueError("Split fold and seed must be non-negative")
        if not self.train:
            raise ValueError("A split must contain training subjects")
        if not self.validation:
            raise ValueError("A split must contain validation subjects")
        if not self.test:
            raise ValueError("A split must contain test subjects")
        owner: Dict[str, str] = {}
        for partition in ("train", "validation", "test", "external"):
            identifiers = getattr(self, partition)
            if any(not identifier for identifier in identifiers):
                raise ValueError(f"Split partition {partition} has an empty subject ID")
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"Split partition {partition} contains duplicate IDs")
            for identifier in identifiers:
                previous = owner.setdefault(identifier, partition)
                if previous != partition:
                    raise ValueError(
                        f"Subject {identifier} occurs in both {previous} and {partition}"
                    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "protocol": self.protocol,
            "fold": self.fold,
            "seed": self.seed,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
            "external": list(self.external),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitDefinition":
        return cls(
            name=str(payload["name"]),
            protocol=str(payload["protocol"]),
            fold=int(payload.get("fold", 0)),
            seed=int(payload.get("seed", 2026)),
            train=tuple(payload.get("train", ())),
            validation=tuple(payload.get("validation", ())),
            test=tuple(payload.get("test", ())),
            external=tuple(payload.get("external", ())),
        )


FoldSplit = SplitDefinition


def _subject_table(records: Sequence[Any]) -> Dict[str, Tuple[str, int]]:
    table: Dict[str, Tuple[str, int]] = {}
    for record in records:
        try:
            subject_id = str(record.subject_id).strip()
            dataset = str(record.dataset).strip()
            label = int(record.label)
        except AttributeError as error:
            raise TypeError("Split records need subject_id, dataset, and label fields") from error
        if not subject_id:
            raise ValueError("Cannot split an empty subject ID")
        if dataset not in ALLOWED_DATASETS or label not in (0, 1):
            raise ValueError(f"Invalid cohort or label for subject {subject_id}")
        previous = table.setdefault(subject_id, (dataset, label))
        if previous != (dataset, label):
            raise ValueError(f"Conflicting cohort or label for subject {subject_id}")
    if not table:
        raise ValueError("Cannot make a split from an empty cohort")
    return table


def _stratified_holdout(
    subject_ids: Sequence[str],
    labels: Mapping[str, int],
    fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    if not 0.0 <= float(fraction) < 1.0:
        raise ValueError("validation_fraction must lie in [0, 1)")
    grouped: Dict[int, List[str]] = defaultdict(list)
    for subject_id in subject_ids:
        grouped[int(labels[subject_id])].append(subject_id)
    generator = random.Random(int(seed))
    training: List[str] = []
    validation: List[str] = []
    for label in sorted(grouped):
        members = sorted(grouped[label])
        generator.shuffle(members)
        if fraction == 0 or len(members) < 2:
            count = 0
        else:
            count = max(1, int(round(len(members) * fraction)))
            count = min(count, len(members) - 1)
        validation.extend(members[:count])
        training.extend(members[count:])
    return sorted(training), sorted(validation)


def _validate_split_against_records(
    split: SplitDefinition,
    records: Sequence[Any],
    *,
    require_coverage: bool = True,
) -> None:
    table = _subject_table(records)
    assigned = set(split.train + split.validation + split.test + split.external)
    unknown = sorted(assigned - set(table))
    if unknown:
        raise ValueError(f"Split contains unknown subjects: {unknown[:10]}")
    if require_coverage:
        missing = sorted(set(table) - assigned)
        if missing:
            raise ValueError(f"Split omits subjects: {missing[:10]}")
    for subject_id in split.train + split.validation:
        dataset = table[subject_id][0]
        if split.protocol == "adni_transfer" and dataset != "ADNI1":
            raise ValueError("adni_transfer training/validation subjects must come from ADNI1")
        if split.protocol == "adni_5fold" and dataset not in {"ADNI1", "ADNI2"}:
            raise ValueError("adni_5fold training/validation subjects must be ADNI subjects")
    for subject_id in split.test:
        dataset = table[subject_id][0]
        if split.protocol == "adni_transfer" and dataset != "ADNI2":
            raise ValueError("adni_transfer test subjects must come from ADNI2")
        if split.protocol == "adni_5fold" and dataset not in {"ADNI1", "ADNI2"}:
            raise ValueError("adni_5fold test subjects must be ADNI subjects")
    if any(table[subject_id][0] != "AIBL" for subject_id in split.external):
        raise ValueError("The external partition is reserved for AIBL")
    if {table[subject_id][1] for subject_id in split.train} != {0, 1}:
        raise ValueError("The training partition must contain both conversion classes")


def make_adni_transfer_split(
    records: Sequence[Any],
    *,
    validation_fraction: float = 0.10,
    seed: int = 2026,
) -> SplitDefinition:
    table = _subject_table(records)
    labels = {subject_id: value[1] for subject_id, value in table.items()}
    adni1 = [subject_id for subject_id, value in table.items() if value[0] == "ADNI1"]
    adni2 = [subject_id for subject_id, value in table.items() if value[0] == "ADNI2"]
    aibl = [subject_id for subject_id, value in table.items() if value[0] == "AIBL"]
    if not adni1 or not adni2:
        raise ValueError("adni_transfer requires both ADNI1 and ADNI2 subjects")
    training, validation = _stratified_holdout(
        adni1, labels, validation_fraction, seed
    )
    split = SplitDefinition(
        name="adni1_to_adni2",
        protocol="adni_transfer",
        fold=0,
        train=tuple(training),
        validation=tuple(validation),
        test=tuple(sorted(adni2)),
        external=tuple(sorted(aibl)),
        seed=seed,
    )
    _validate_split_against_records(split, records)
    return split


def make_adni_5fold_splits(
    records: Sequence[Any],
    *,
    n_folds: int = 5,
    validation_fraction: float = 0.10,
    seed: int = 2026,
) -> List[SplitDefinition]:
    if int(n_folds) != 5:
        raise ValueError("adni_5fold uses exactly five folds")
    table = _subject_table(records)
    adni = {
        subject_id: value[1]
        for subject_id, value in table.items()
        if value[0] in {"ADNI1", "ADNI2"}
    }
    aibl = sorted(
        subject_id for subject_id, value in table.items() if value[0] == "AIBL"
    )
    if not adni:
        raise ValueError("adni_5fold requires ADNI1 or ADNI2 subjects")
    grouped: Dict[int, List[str]] = defaultdict(list)
    for subject_id, label in adni.items():
        grouped[label].append(subject_id)
    for label, members in grouped.items():
        if len(members) < n_folds:
            raise ValueError(
                f"Class {label} has {len(members)} subjects, fewer than {n_folds} folds"
            )
    generator = random.Random(int(seed))
    test_folds: List[List[str]] = [[] for _ in range(n_folds)]
    for label in sorted(grouped):
        members = sorted(grouped[label])
        generator.shuffle(members)
        for offset, subject_id in enumerate(members):
            test_folds[offset % n_folds].append(subject_id)

    splits: List[SplitDefinition] = []
    all_adni = set(adni)
    for fold, test_ids in enumerate(test_folds):
        remaining = sorted(all_adni - set(test_ids))
        training, validation = _stratified_holdout(
            remaining,
            adni,
            validation_fraction,
            seed + fold + 1,
        )
        split = SplitDefinition(
            name=f"adni_5fold_{fold}",
            protocol="adni_5fold",
            fold=fold,
            train=tuple(training),
            validation=tuple(validation),
            test=tuple(sorted(test_ids)),
            external=tuple(aibl),
            seed=seed,
        )
        _validate_split_against_records(split, records)
        splits.append(split)
    test_counts: Dict[str, int] = defaultdict(int)
    for split in splits:
        for subject_id in split.test:
            test_counts[subject_id] += 1
    if set(test_counts) != all_adni or any(count != 1 for count in test_counts.values()):
        raise RuntimeError("Five-fold construction did not assign every ADNI subject once")
    return splits


def make_protocol_splits(
    records: Sequence[Any],
    protocol: str,
    *,
    n_folds: int = 5,
    validation_fraction: float = 0.10,
    seed: int = 2026,
) -> List[SplitDefinition]:
    if protocol == "adni_transfer":
        return [
            make_adni_transfer_split(
                records,
                validation_fraction=validation_fraction,
                seed=seed,
            )
        ]
    if protocol == "adni_5fold":
        return make_adni_5fold_splits(
            records,
            n_folds=n_folds,
            validation_fraction=validation_fraction,
            seed=seed,
        )
    raise ValueError(f"Unsupported protocol {protocol!r}")


def save_splits(splits: Sequence[SplitDefinition], path: Union[str, Path]) -> None:
    definitions = list(splits)
    if not definitions:
        raise ValueError("Cannot save an empty split collection")
    for split in definitions:
        split.validate()
    payload = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "splits": [split.to_dict() for split in definitions],
    }
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=False)
            stream.write("\n")
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_splits(
    path: Union[str, Path],
    *,
    records: Optional[Sequence[Any]] = None,
    expected_protocol: Optional[str] = None,
) -> List[SplitDefinition]:
    source = Path(path).expanduser().resolve()
    _check_file_size(source, 16 * 1024 * 1024, "Split")
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("Split JSON must contain an object")
    if payload.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported split schema {payload.get('schema_version')!r}; "
            f"expected {SPLIT_SCHEMA_VERSION}"
        )
    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, list) or not raw_splits:
        raise ValueError("Split JSON must contain a non-empty splits list")
    splits = [SplitDefinition.from_dict(item) for item in raw_splits]
    names = [split.name for split in splits]
    if len(set(names)) != len(names):
        raise ValueError("Split JSON contains duplicate split names")
    if expected_protocol is not None and any(
        split.protocol != expected_protocol for split in splits
    ):
        raise ValueError(f"Split JSON does not match protocol {expected_protocol!r}")
    if records is not None:
        for split in splits:
            _validate_split_against_records(split, records)
    return splits


def partition_records(
    records: Sequence[SequenceRecord],
    split: SplitDefinition,
) -> Dict[str, List[SequenceRecord]]:
    by_subject: Dict[str, SequenceRecord] = {}
    for record in records:
        if record.subject_id in by_subject:
            raise ValueError(f"Duplicate sequence for subject {record.subject_id}")
        by_subject[record.subject_id] = record
    _validate_split_against_records(split, records)
    return {
        partition: [by_subject[subject_id] for subject_id in getattr(split, partition)]
        for partition in ("train", "validation", "test", "external")
    }


@dataclass(frozen=True)
class TaskMask:
    task: Literal["interpolation", "extrapolation"]
    difficulty: int
    input_mask: np.ndarray
    target_mask: np.ndarray

    def __post_init__(self) -> None:
        input_mask = np.asarray(self.input_mask, dtype=bool).copy()
        target_mask = np.asarray(self.target_mask, dtype=bool).copy()
        input_mask.setflags(write=False)
        target_mask.setflags(write=False)
        object.__setattr__(self, "input_mask", input_mask)
        object.__setattr__(self, "target_mask", target_mask)
        object.__setattr__(self, "difficulty", int(self.difficulty))
        self.validate()

    @property
    def target_indices(self) -> Tuple[int, ...]:
        return tuple(int(index) for index in np.flatnonzero(self.target_mask))

    def validate(self) -> None:
        if self.task not in {"interpolation", "extrapolation"}:
            raise ValueError(f"Unsupported task {self.task!r}")
        if self.input_mask.shape != (len(VISIT_MONTHS),):
            raise ValueError("Task input_mask must have shape [6]")
        if self.target_mask.shape != (len(VISIT_MONTHS),):
            raise ValueError("Task target_mask must have shape [6]")
        if np.any(self.input_mask & self.target_mask):
            raise ValueError("Task input and target masks overlap")
        indices = self.target_indices
        if not indices or len(indices) != self.difficulty:
            raise ValueError("Task difficulty must equal the number of target visits")
        if not 1 <= self.difficulty <= 4:
            raise ValueError("Task difficulty must lie in [1, 4]")
        allowed = range(1, 5) if self.task == "interpolation" else range(1, 6)
        if any(index not in allowed for index in indices):
            endpoint = "1..4" if self.task == "interpolation" else "1..5"
            raise ValueError(f"{self.task} target indices must lie in {endpoint}")
        if self.task == "extrapolation" and tuple(
            range(indices[0], indices[-1] + 1)
        ) != indices:
            raise ValueError("Task target visits must form one contiguous block")
        if not self.input_mask.any():
            raise ValueError("Task has no conditioning visit")
        if self.task == "interpolation":
            if not self.input_mask[: indices[0]].any() or not self.input_mask[indices[-1] + 1 :].any():
                raise ValueError("Interpolation needs context before and after its targets")
        elif self.input_mask[indices[0] :].any():
            raise ValueError("Extrapolation input cannot include the target or a later visit")


def _visit_mask(values: Sequence[bool], name: str) -> np.ndarray:
    mask = np.asarray(values, dtype=bool)
    if mask.shape != (len(VISIT_MONTHS),):
        raise ValueError(f"{name} must have shape [6]")
    return mask


def make_interpolation_masks(
    available: Sequence[bool],
    *,
    max_difficulty: int = 4,
) -> List[TaskMask]:
    source = _visit_mask(available, "available")
    if not 1 <= int(max_difficulty) <= 4:
        raise ValueError("max_difficulty must lie in [1, 4]")
    masks: List[TaskMask] = []
    for difficulty in range(1, int(max_difficulty) + 1):
        candidates = [index for index in range(1, 5) if source[index]]
        for indices in combinations(candidates, difficulty):
            target = np.zeros(len(VISIT_MONTHS), dtype=bool)
            target[list(indices)] = True
            inputs = source & ~target
            if inputs[: indices[0]].any() and inputs[indices[-1] + 1 :].any():
                masks.append(TaskMask("interpolation", difficulty, inputs, target))
    return masks


def make_extrapolation_masks(
    available: Sequence[bool],
    *,
    max_difficulty: int = 4,
) -> List[TaskMask]:
    source = _visit_mask(available, "available")
    if not 1 <= int(max_difficulty) <= 4:
        raise ValueError("max_difficulty must lie in [1, 4]")
    masks: List[TaskMask] = []
    for difficulty in range(1, int(max_difficulty) + 1):
        for start in range(1, 7 - difficulty):
            stop = start + difficulty
            target = np.zeros(len(VISIT_MONTHS), dtype=bool)
            target[start:stop] = True
            if not source[target].all():
                continue
            inputs = source.copy()
            inputs[start:] = False
            if inputs.any():
                masks.append(TaskMask("extrapolation", difficulty, inputs, target))
    return masks


def make_missing_masks(
    record: SequenceRecord,
    task: Literal["interpolation", "extrapolation"],
    *,
    max_difficulty: int = 4,
) -> List[TaskMask]:
    """Build inference masks for genuinely missing, not held-out, visits."""

    if not 1 <= int(max_difficulty) <= 4:
        raise ValueError("max_difficulty must lie in [1, 4]")
    available = record.available
    missing = ~available
    masks: List[TaskMask] = []
    if task == "interpolation":
        index = 1
        while index <= 4:
            if not missing[index]:
                index += 1
                continue
            stop = index
            while stop <= 4 and missing[stop] and stop - index < max_difficulty:
                stop += 1
            target = np.zeros(len(VISIT_MONTHS), dtype=bool)
            target[index:stop] = True
            if available[:index].any() and available[stop:].any():
                masks.append(TaskMask("interpolation", stop - index, available, target))
            index = stop
        return masks
    if task == "extrapolation":
        observed_indices = np.flatnonzero(available)
        if not len(observed_indices):
            return []
        start = int(observed_indices[-1]) + 1
        while start <= 5:
            stop = min(start + int(max_difficulty), 6)
            target_indices = [index for index in range(start, stop) if missing[index]]
            if not target_indices:
                start = stop
                continue
            # A gap inside a chunk is split, keeping TaskMask targets contiguous.
            run_start = target_indices[0]
            run_stop = run_start + 1
            for index in target_indices[1:] + [10]:
                if index == run_stop:
                    run_stop += 1
                    continue
                target = np.zeros(len(VISIT_MONTHS), dtype=bool)
                target[run_start:run_stop] = True
                inputs = available.copy()
                inputs[run_start:] = False
                masks.append(
                    TaskMask("extrapolation", run_stop - run_start, inputs, target)
                )
                run_start, run_stop = index, index + 1
            start = stop
        return masks
    raise ValueError(f"Unsupported task {task!r}")


class SequenceDataset(Dataset):
    def __init__(self, records: Sequence[SequenceRecord]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        available = record.available
        return {
            "subject_id": record.subject_id,
            "dataset": record.dataset,
            "label": torch.tensor(record.label, dtype=torch.long),
            "months": torch.tensor(VISIT_MONTHS, dtype=torch.long),
            "features": torch.from_numpy(record.features.copy()).float(),
            "observed_mask": torch.from_numpy(record.observed.copy()),
            "imputed_mask": torch.from_numpy(record.imputed.copy()),
            "available_mask": torch.from_numpy(available.copy()),
        }


class LongitudinalTaskDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SequenceRecord],
        *,
        tasks: Union[str, Sequence[str]] = ("interpolation", "extrapolation"),
        max_difficulty: int = 4,
        training: bool = True,
        include_imputed_context: bool = True,
    ) -> None:
        self.records = list(records)
        task_names = (tasks,) if isinstance(tasks, str) else tuple(tasks)
        if not task_names or any(
            task not in {"interpolation", "extrapolation"} for task in task_names
        ):
            raise ValueError("tasks must contain interpolation and/or extrapolation")
        self.training = bool(training)
        self.items: List[Tuple[int, TaskMask]] = []
        for record_index, record in enumerate(self.records):
            available = record.available if include_imputed_context else record.observed
            for task in task_names:
                if training and task == "interpolation":
                    masks = make_interpolation_masks(
                        available, max_difficulty=max_difficulty
                    )
                elif training and task == "extrapolation":
                    masks = make_extrapolation_masks(
                        available, max_difficulty=max_difficulty
                    )
                else:
                    masks = make_missing_masks(
                        record,
                        task,  # type: ignore[arg-type]
                        max_difficulty=max_difficulty,
                    )
                self.items.extend((record_index, mask) for mask in masks)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record_index, task_mask = self.items[index]
        record = self.records[record_index]
        input_features = np.zeros_like(record.features)
        input_features[task_mask.input_mask] = record.features[task_mask.input_mask]
        target_features = np.zeros_like(record.features)
        if self.training:
            target_features[task_mask.target_mask] = record.features[task_mask.target_mask]
        return {
            "subject_id": record.subject_id,
            "dataset": record.dataset,
            "label": torch.tensor(record.label, dtype=torch.long),
            "task": task_mask.task,
            "task_id": torch.tensor(
                0 if task_mask.task == "interpolation" else 1, dtype=torch.long
            ),
            "difficulty": torch.tensor(task_mask.difficulty, dtype=torch.long),
            "months": torch.tensor(VISIT_MONTHS, dtype=torch.long),
            "features": torch.from_numpy(record.features.copy()).float(),
            "input_features": torch.from_numpy(input_features).float(),
            "target_features": torch.from_numpy(target_features).float(),
            "observed_mask": torch.from_numpy(record.observed.copy()),
            "imputed_mask": torch.from_numpy(record.imputed.copy()),
            "available_mask": torch.from_numpy(record.available.copy()),
            "input_mask": torch.from_numpy(task_mask.input_mask.copy()),
            "target_mask": torch.from_numpy(task_mask.target_mask.copy()),
            "target_indices": torch.tensor(task_mask.target_indices, dtype=torch.long),
        }


ProgressiveDataset = LongitudinalTaskDataset


def longitudinal_collate(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    keys = set(samples[0])
    if any(set(sample) != keys for sample in samples):
        raise ValueError("Batch samples have different keys")
    result: Dict[str, Any] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        if key in {"subject_id", "dataset", "task"}:
            result[key] = values
        elif key == "target_indices":
            result[key] = values
        elif all(isinstance(value, torch.Tensor) for value in values):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


sequence_collate = longitudinal_collate


__all__ = [
    "ALLOWED_DATASETS",
    "FEATURE_SUFFIXES",
    "OPTIONAL_PATH_COLUMNS",
    "REQUIRED_MANIFEST_COLUMNS",
    "SPLIT_SCHEMA_VERSION",
    "VISIT_MONTHS",
    "VISIT_TO_INDEX",
    "FeatureNormalizer",
    "FoldSplit",
    "LongitudinalTaskDataset",
    "ManifestRow",
    "ProgressiveDataset",
    "ScanRecord",
    "SequenceDataset",
    "SequenceRecord",
    "SplitDefinition",
    "TaskMask",
    "assemble_sequences",
    "build_sequences",
    "clone_with_imputations",
    "fit_normalizer",
    "load_feature_array",
    "load_splits",
    "longitudinal_collate",
    "make_adni_5fold_splits",
    "make_adni_transfer_split",
    "make_extrapolation_masks",
    "make_interpolation_masks",
    "make_missing_masks",
    "make_protocol_splits",
    "partition_complete_sequences",
    "partition_records",
    "read_manifest",
    "save_splits",
    "sequence_collate",
]
