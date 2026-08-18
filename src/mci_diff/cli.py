"""Command line entry points for the experiments."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mci_diff.adapters import TorchScriptFeatureAdapter
from mci_diff.config import Config, dump_config, load_config
from mci_diff.data import (
    FeatureNormalizer,
    SequenceRecord,
    assemble_sequences,
    load_splits,
    make_protocol_splits,
    partition_records,
    read_manifest,
    save_splits,
)
from mci_diff.engine import (
    ClassifierTrainer,
    DiffusionTrainer,
    ProgressiveTrainingPool,
    aggregate_fold_metrics,
    atomic_json_dump,
    atomic_torch_save,
    balanced_class_weights,
    binary_metrics,
    bootstrap_binary_metrics,
    generate_trajectories,
    load_model_checkpoint,
    mark_stage_complete,
    resolve_device,
    run_progressive_training,
    seed_everything,
    stage_is_complete,
    torch_load,
    write_prediction_csv,
)
from mci_diff.models import (
    ConditionalFeatureDDPM,
    ConditionalTransformerDenoiser,
    DiffusionSchedule,
    TrajectoryClassifier,
)


LOGGER = logging.getLogger("mci_diff")


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _path(config: Config, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        if config.project_root is None:
            raise RuntimeError("configuration has no project root")
        path = config.project_root / path
    return path.resolve()


def _fold_directory(config: Config, fold: int) -> Path:
    return _path(config, config.experiment.output_dir) / ("fold_%d" % int(fold))


def _read_scans(config: Config, check_paths: bool = True, require_image: bool = False):
    return read_manifest(
        _path(config, config.data.manifest),
        project_root=config.project_root,
        visit_months=list(config.data.visit_months),
        allowed_datasets=list(config.data.allowed_datasets),
        positive_label=str(config.data.positive_label),
        negative_label=str(config.data.negative_label),
        check_paths=check_paths,
        require_image=require_image,
    )


def _read_sequences(config: Config) -> List[SequenceRecord]:
    return assemble_sequences(
        _read_scans(config, check_paths=True),
        feature_dim=int(config.data.feature_dim),
    )


def _select_split(config: Config, records: Sequence[SequenceRecord], fold: int):
    definitions = load_splits(
        _path(config, config.split.file),
        records=records,
        expected_protocol=str(config.split.protocol),
    )
    matches = [item for item in definitions if int(item.fold) == int(fold)]
    if len(matches) != 1:
        raise ValueError("split file has no unique definition for fold %d" % int(fold))
    return matches[0]


def _write_normalizer(normalizer: FeatureNormalizer, path: Path) -> None:
    atomic_json_dump(normalizer.to_dict(), path)


def _read_normalizer(path: Path) -> FeatureNormalizer:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError("normalizer JSON must contain an object")
    return FeatureNormalizer.from_dict(payload)


@dataclass
class FoldContext:
    config: Config
    fold: int
    directory: Path
    split: Any
    partitions: Dict[str, List[SequenceRecord]]
    normalizer: Optional[FeatureNormalizer]
    device: torch.device


def _prepare_fold(config: Config, fold: int) -> FoldContext:
    seed_everything(
        int(config.experiment.seed) + int(fold),
        bool(config.experiment.deterministic),
    )
    records = _read_sequences(config)
    split = _select_split(config, records, fold)
    partitions = partition_records(records, split)
    directory = _fold_directory(config, fold)
    directory.mkdir(parents=True, exist_ok=True)
    dump_config(config, directory / "resolved_config.yaml")

    normalizer = None
    if bool(config.normalization.enabled):
        normalizer_path = directory / "normalizer.json"
        if normalizer_path.is_file():
            normalizer = _read_normalizer(normalizer_path)
        else:
            normalizer = FeatureNormalizer.fit(
                partitions["train"],
                partition="train",
                epsilon=float(config.normalization.epsilon),
            )
            _write_normalizer(normalizer, normalizer_path)
        partitions = {
            name: [normalizer.transform(record) for record in values]
            for name, values in partitions.items()
        }
    device = resolve_device(str(config.experiment.device))
    return FoldContext(config, int(fold), directory, split, partitions, normalizer, device)


def _build_diffusion(config: Config) -> ConditionalFeatureDDPM:
    hidden = int(config.diffusion.hidden_dim)
    feedforward = int(round(hidden * float(config.diffusion.feedforward_multiplier)))
    denoiser = ConditionalTransformerDenoiser(
        feature_dim=int(config.data.feature_dim),
        model_dim=hidden,
        num_visits=len(config.data.visit_months),
        num_layers=int(config.diffusion.denoiser_layers),
        num_heads=int(config.diffusion.attention_heads),
        feedforward_dim=feedforward,
        dropout=float(config.diffusion.dropout),
        time_dim=int(config.diffusion.time_embedding_dim),
    )
    schedule = DiffusionSchedule(
        timesteps=int(config.diffusion.num_steps),
        name=str(config.diffusion.schedule),
        beta_start=float(config.diffusion.beta_start),
        beta_end=float(config.diffusion.beta_end),
    )
    return ConditionalFeatureDDPM(denoiser, schedule)


def _build_classifier(config: Config) -> TrajectoryClassifier:
    hidden = int(config.classifier.hidden_dim)
    return TrajectoryClassifier(
        feature_dim=int(config.data.feature_dim),
        model_dim=hidden,
        num_layers=int(config.classifier.num_layers),
        num_heads=int(config.classifier.attention_heads),
        feedforward_dim=int(round(hidden * float(config.classifier.feedforward_multiplier))),
        dropout=float(config.classifier.dropout),
        num_classes=2,
        num_visits=len(config.data.visit_months),
        architecture=str(config.classifier.architecture),
        pooling=str(config.classifier.pooling),
    )


def _records_to_pool(records: Sequence[SequenceRecord]) -> ProgressiveTrainingPool:
    if not records:
        raise ValueError("cannot make a training pool from an empty partition")
    return ProgressiveTrainingPool(
        features=torch.from_numpy(np.stack([item.features for item in records])).float(),
        observed=torch.from_numpy(np.stack([item.available for item in records])).bool(),
        labels=torch.tensor([item.label for item in records], dtype=torch.long),
        patient_ids=[item.subject_id for item in records],
    )


def command_validate(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.set)
    records = _read_sequences(config)
    summary = {
        "subjects": len(records),
        "scans": int(sum(item.observed.sum() for item in records)),
        "datasets": {
            dataset: {
                "subjects": sum(item.dataset == dataset for item in records),
                "pMCI": sum(item.dataset == dataset and item.label == 1 for item in records),
                "sMCI": sum(item.dataset == dataset and item.label == 0 for item in records),
            }
            for dataset in config.data.allowed_datasets
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def command_split(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.set)
    destination = _path(config, config.split.file)
    if destination.exists() and not args.force:
        raise FileExistsError("split file already exists; pass --force or choose another path")
    scans = _read_scans(config, check_paths=False)
    definitions = make_protocol_splits(
        scans,
        str(config.split.protocol),
        n_folds=int(config.split.n_folds),
        validation_fraction=float(config.split.validation_fraction),
        seed=int(config.experiment.seed),
    )
    save_splits(definitions, destination)
    LOGGER.info("wrote %d split definition(s) to %s", len(definitions), destination)


def _load_python_factory(specification: str) -> Any:
    if ":" not in specification:
        raise ValueError("factory must be written as module.path:function")
    module_name, attribute = specification.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory() if callable(factory) else factory


def _candidate_scorer(config: Config) -> Any:
    specification = config.sampling.scorer_factory
    if specification is None or not str(specification).strip():
        return None
    return _load_python_factory(str(specification))


def command_extract(args: argparse.Namespace) -> None:
    config = load_config(args.config, args.set)
    try:
        import nibabel as nib
    except ImportError as error:
        raise ImportError("install mci-diff[medical] to read NIfTI volumes") from error

    configured_backend = str(config.data.encoder.backend)
    encoder_path = args.encoder
    factory_spec = args.factory
    if encoder_path is None and configured_backend == "torchscript":
        encoder_path = config.data.encoder.checkpoint
    if factory_spec is None and configured_backend == "python_factory":
        factory_spec = config.data.encoder.python_factory
    if encoder_path and factory_spec:
        raise ValueError("choose one encoder source: TorchScript or Python factory")

    if encoder_path:
        encoder = TorchScriptFeatureAdapter(
            _path(config, encoder_path),
            device=str(resolve_device(str(config.experiment.device))),
            output_key=str(config.data.encoder.output_key),
        )
    elif factory_spec:
        from mci_diff.adapters import HFCNFeatureAdapter

        encoder = HFCNFeatureAdapter(
            _load_python_factory(str(factory_spec)),
            device=str(resolve_device(str(config.experiment.device))),
            output_key=str(config.data.encoder.output_key),
        )
    else:
        raise ValueError("extract needs --encoder or --factory")

    manifest = _path(config, config.data.manifest)
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "image_path" not in rows[0] or "feature_path" not in rows[0]:
        raise ValueError("manifest needs image_path and feature_path columns")
    written = 0
    for row_number, row in enumerate(rows, 2):
        image_text = str(row.get("image_path", "")).strip()
        feature_text = str(row.get("feature_path", "")).strip()
        if not image_text or not feature_text:
            raise ValueError("missing image/feature path on row %d" % row_number)
        image_path = _path(config, image_text)
        feature_path = _path(config, feature_text)
        if feature_path.exists() and not args.overwrite:
            continue
        volume = np.asarray(nib.load(str(image_path)).get_fdata(dtype=np.float32))
        if volume.ndim != 3 or not np.isfinite(volume).all():
            raise ValueError("NIfTI volume must be finite and 3-D: %s" % image_path)
        tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        feature = encoder.encode(tensor)[0].numpy()
        if feature.shape != (int(config.data.feature_dim),):
            raise ValueError(
                "encoder returned %s, expected [%d]" % (feature.shape, int(config.data.feature_dim))
            )
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        feature = feature.astype(np.float32, copy=False)
        with feature_path.open("wb") as stream:
            if feature_path.suffix.casefold() == ".npz":
                np.savez_compressed(stream, feature=feature)
            else:
                np.save(stream, feature, allow_pickle=False)
        written += 1
    LOGGER.info("wrote %d feature files", written)


def _train_diffusion(config: Config, fold: int) -> Path:
    context = _prepare_fold(config, fold)
    final_checkpoint = context.directory / "diffusion" / "final.pt"
    history_path = context.directory / "diffusion" / "progressive_history.json"
    pool_path = context.directory / "diffusion" / "training_pool_final.pt"
    if bool(config.experiment.resume) and stage_is_complete(
        context.directory, "diffusion", [final_checkpoint, history_path, pool_path]
    ):
        LOGGER.info("fold %d diffusion is complete", fold)
        return final_checkpoint

    model = _build_diffusion(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.diffusion.learning_rate),
        weight_decay=float(config.training.diffusion.weight_decay),
    )
    trainer = DiffusionTrainer(
        model,
        optimizer,
        device=context.device,
        precision=str(config.experiment.precision),
        ema_decay=float(config.training.ema_decay),
        gradient_clip_norm=float(config.training.gradient_clip_norm),
    )
    run_progressive_training(
        trainer,
        _records_to_pool(context.partitions["train"]),
        config,
        context.directory / "diffusion",
        device=context.device,
        resume=bool(config.experiment.resume),
        validation_records=_records_to_pool(context.partitions["validation"]),
        generator=torch.Generator().manual_seed(int(config.experiment.seed) + fold),
    )
    # The phase trainer has already restored the best model from the final phase.
    atomic_torch_save({"model": trainer.model.state_dict()}, final_checkpoint)
    mark_stage_complete(
        context.directory,
        "diffusion",
        [final_checkpoint, history_path, pool_path],
        {"fold": fold},
    )
    return final_checkpoint


def command_train_diffusion(args: argparse.Namespace) -> None:
    _train_diffusion(load_config(args.config, args.set), int(args.fold))


def _generate(config: Config, fold: int) -> Path:
    context = _prepare_fold(config, fold)
    directory = context.directory / "generated"
    index_path = directory / "index.json"
    partition_paths = [
        directory / (name + ".pt")
        for name in ("train", "validation", "test", "external")
        if context.partitions[name]
    ]
    if bool(config.experiment.resume) and stage_is_complete(
        context.directory, "generate", [index_path] + partition_paths
    ):
        return index_path

    checkpoint = context.directory / "diffusion" / "final.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("diffusion checkpoint is missing; run train-diffusion")
    diffusion = _build_diffusion(config)
    load_model_checkpoint(checkpoint, diffusion, context.device)
    scorer = _candidate_scorer(config)
    directory.mkdir(parents=True, exist_ok=True)
    index: Dict[str, Any] = {"schema_version": 1, "fold": fold, "partitions": {}}
    for name in ("train", "validation", "test", "external"):
        records = context.partitions[name]
        if not records:
            continue
        source = torch.from_numpy(np.stack([item.features for item in records])).float()
        observed = torch.zeros(source.shape[:2], dtype=torch.bool)
        observed[:, 0] = True
        source[:, 1:] = 0.0
        path = directory / (name + ".pt")
        generate_trajectories(
            diffusion,
            source,
            config=config,
            scorer=scorer,
            observed=observed,
            patient_ids=[item.subject_id for item in records],
            num_candidates=int(config.sampling.num_candidates),
            candidate_batch_size=int(config.sampling.candidate_batch_size),
            target_indices=range(1, len(config.data.visit_months)),
            device=context.device,
            use_guidance=scorer is not None,
            fallback=str(config.sampling.fallback),
            tie_break=str(config.sampling.tie_break),
            cache_path=path,
            resume=False,
            save_all_candidates=bool(config.sampling.save_all_candidates),
            generator=torch.Generator().manual_seed(
                int(config.experiment.seed) + fold * 1000 + len(index["partitions"])
            ),
            clip_x0=config.diffusion.clip_x0,
        )
        generated_payload = torch_load(path, "cpu")
        if not isinstance(generated_payload, Mapping):
            raise ValueError("generated trajectory cache is not a mapping")
        generated_payload = dict(generated_payload)
        generated_payload.update(
            {
                "schema_version": 1,
                "partition": name,
                "labels": torch.tensor([item.label for item in records], dtype=torch.long),
                "datasets": [item.dataset for item in records],
            }
        )
        atomic_torch_save(generated_payload, path)
        index["partitions"][name] = {
            "path": str(path),
            "subjects": len(records),
        }
    atomic_json_dump(index, index_path)
    mark_stage_complete(context.directory, "generate", [index_path] + partition_paths)
    return index_path


def command_generate(args: argparse.Namespace) -> None:
    _generate(load_config(args.config, args.set), int(args.fold))


class GeneratedTrajectoryDataset(Dataset):
    def __init__(self, payload: Mapping[str, Any]) -> None:
        required = ("trajectories", "observed", "labels", "patient_ids", "datasets")
        if any(name not in payload for name in required):
            raise ValueError("generated partition is incomplete")
        self.features = torch.as_tensor(payload["trajectories"]).float()
        self.observed = torch.as_tensor(payload["observed"]).bool()
        self.labels = torch.as_tensor(payload["labels"]).long()
        self.patient_ids = [str(item) for item in payload["patient_ids"]]
        self.datasets = [str(item) for item in payload["datasets"]]
        size = len(self.features)
        if self.observed.shape != self.features.shape[:2] or len(self.labels) != size:
            raise ValueError("generated tensors have incompatible shapes")
        if len(self.patient_ids) != size or len(self.datasets) != size:
            raise ValueError("generated metadata has the wrong length")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "features": self.features[index],
            "available_mask": self.observed[index],
            "label": self.labels[index],
            "subject_id": self.patient_ids[index],
            "dataset": self.datasets[index],
        }


def _generated_partition(index_path: Path, name: str) -> GeneratedTrajectoryDataset:
    with index_path.open("r", encoding="utf-8") as stream:
        index = json.load(stream)
    entry = index.get("partitions", {}).get(name)
    if not isinstance(entry, Mapping):
        raise ValueError("generated index has no %s partition" % name)
    payload = torch_load(entry["path"], "cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("generated partition must contain a mapping")
    return GeneratedTrajectoryDataset(payload)


def _trajectory_loader(
    dataset: GeneratedTrajectoryDataset,
    batch_size: int,
    shuffle: bool,
    config: Config,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(config.training.num_workers),
        pin_memory=bool(config.training.pin_memory),
    )


def _train_classifier(config: Config, fold: int) -> Path:
    context = _prepare_fold(config, fold)
    checkpoint = context.directory / "classifier" / "checkpoints" / "best.pt"
    if bool(config.experiment.resume) and stage_is_complete(
        context.directory, "classifier", [checkpoint]
    ):
        return checkpoint
    index_path = context.directory / "generated" / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError("generated trajectories are missing; run generate")
    train_data = _generated_partition(index_path, "train")
    validation_data = _generated_partition(index_path, "validation")
    model = _build_classifier(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.classifier.learning_rate),
        weight_decay=float(config.training.classifier.weight_decay),
    )
    weights = None
    if str(config.classifier.class_weight) == "balanced":
        weights = balanced_class_weights(train_data.labels)
    trainer = ClassifierTrainer(
        model,
        optimizer,
        context.device,
        str(config.experiment.precision),
        float(config.training.gradient_clip_norm),
        weights,
    )
    result = trainer.fit(
        _trajectory_loader(
            train_data, int(config.training.classifier.batch_size), True, config
        ),
        _trajectory_loader(
            validation_data, int(config.training.classifier.batch_size), False, config
        ),
        epochs=int(config.training.classifier.epochs),
        patience=int(config.training.classifier.patience),
        checkpoint_dir=checkpoint.parent,
        warmup_epochs=int(config.training.classifier.warmup_epochs),
        checkpoint_every=int(config.training.checkpoint_every),
        resume=bool(config.experiment.resume),
    )
    atomic_json_dump(result.history, context.directory / "classifier" / "history.json")
    mark_stage_complete(context.directory, "classifier", [checkpoint])
    return checkpoint


def command_train_classifier(args: argparse.Namespace) -> None:
    _train_classifier(load_config(args.config, args.set), int(args.fold))


def _metric_payload(
    result: Any,
    config: Config,
    fold: int,
    partition: str,
) -> Dict[str, Any]:
    metrics = binary_metrics(
        result.labels,
        result.probabilities,
        float(config.evaluation.threshold),
    )
    payload: Dict[str, Any] = {
        "fold": fold,
        "partition": partition,
        "subjects": len(result.labels),
        **metrics,
    }
    if int(config.evaluation.bootstrap_samples) > 0:
        estimates = bootstrap_binary_metrics(
            result.labels,
            result.probabilities,
            threshold=float(config.evaluation.threshold),
            samples=int(config.evaluation.bootstrap_samples),
            confidence=float(config.evaluation.bootstrap_confidence),
            seed=int(config.experiment.seed) + fold,
        )
        payload["bootstrap"] = {
            name: {
                "estimate": value.estimate,
                "lower": value.lower,
                "upper": value.upper,
                "successful_samples": value.successful_samples,
            }
            for name, value in estimates.items()
        }
    return payload


def _evaluate(config: Config, fold: int) -> Path:
    context = _prepare_fold(config, fold)
    output = context.directory / "evaluation" / "test_metrics.json"
    if bool(config.experiment.resume) and stage_is_complete(
        context.directory, "evaluate", [output]
    ):
        return output
    index_path = context.directory / "generated" / "index.json"
    checkpoint = context.directory / "classifier" / "checkpoints" / "best.pt"
    model = _build_classifier(config)
    load_model_checkpoint(checkpoint, model, context.device)
    trainer = ClassifierTrainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1.0e-8),
        context.device,
        str(config.experiment.precision),
    )
    evaluation_dir = output.parent
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    metrics_paths: List[Path] = []
    for partition in ("test", "external"):
        try:
            dataset = _generated_partition(index_path, partition)
        except ValueError:
            continue
        result = trainer.predict(
            _trajectory_loader(
                dataset, int(config.training.classifier.batch_size), False, config
            ),
            threshold=float(config.evaluation.threshold),
        )
        metric_path = evaluation_dir / (partition + "_metrics.json")
        atomic_json_dump(_metric_payload(result, config, fold, partition), metric_path)
        if bool(config.evaluation.save_predictions):
            write_prediction_csv(
                result,
                evaluation_dir / (partition + "_predictions.csv"),
                fold=fold,
                datasets=dataset.datasets,
            )
        metrics_paths.append(metric_path)
    if not output.exists():
        raise RuntimeError("test metrics were not produced")
    mark_stage_complete(context.directory, "evaluate", metrics_paths)
    return output


def command_evaluate(args: argparse.Namespace) -> None:
    _evaluate(load_config(args.config, args.set), int(args.fold))


def _ensure_split(config: Config) -> None:
    path = _path(config, config.split.file)
    if path.is_file():
        return
    scans = _read_scans(config, check_paths=False)
    save_splits(
        make_protocol_splits(
            scans,
            str(config.split.protocol),
            n_folds=int(config.split.n_folds),
            validation_fraction=float(config.split.validation_fraction),
            seed=int(config.experiment.seed),
        ),
        path,
    )


def _folds_for(config: Config, requested: Optional[Sequence[int]]) -> List[int]:
    available = [0] if str(config.split.protocol) == "adni_transfer" else list(
        range(int(config.split.n_folds))
    )
    folds = available if requested is None else [int(item) for item in requested]
    if len(folds) != len(set(folds)):
        raise ValueError("fold list contains duplicates")
    if any(fold not in available for fold in folds):
        raise ValueError("requested fold is outside the configured protocol")
    return folds


def _run(config: Config, folds: Optional[Sequence[int]]) -> None:
    _ensure_split(config)
    selected = _folds_for(config, folds)
    metric_paths: List[Path] = []
    for fold in selected:
        _train_diffusion(config, fold)
        _generate(config, fold)
        _train_classifier(config, fold)
        metric_paths.append(_evaluate(config, fold))
    if str(config.split.protocol) == "adni_5fold" and len(selected) == int(config.split.n_folds):
        aggregate = aggregate_fold_metrics(metric_paths)
        atomic_json_dump(
            aggregate,
            _path(config, config.experiment.output_dir) / "cross_validation_metrics.json",
        )


def command_run(args: argparse.Namespace) -> None:
    _run(load_config(args.config, args.set), args.folds)


def _ablation_values(name: str) -> Dict[str, bool]:
    values = {
        "interpolation_task": True,
        "interpolation_augmentation": True,
        "extrapolation_task": True,
        "extrapolation_augmentation": True,
    }
    mapping = {
        "no_interpolation_task": "interpolation_task",
        "no_interpolation_augmentation": "interpolation_augmentation",
        "no_extrapolation_task": "extrapolation_task",
        "no_extrapolation_augmentation": "extrapolation_augmentation",
    }
    if name != "full":
        if name not in mapping:
            raise ValueError("unknown ablation %s" % name)
        values[mapping[name]] = False
    return values


def _matrix_plan(config: Config) -> List[Tuple[str, List[str]]]:
    plan: List[Tuple[str, List[str]]] = []
    for protocol in config.matrix.protocols:
        for ablation in config.matrix.ablations:
            overrides = ["split.protocol=%s" % protocol]
            if protocol == "adni_transfer":
                overrides.append("split.file=data/splits/adni_transfer.json")
            else:
                overrides.append("split.file=data/splits/adni_5fold.json")
            for key, value in _ablation_values(str(ablation)).items():
                overrides.append("ablation.%s=%s" % (key, str(value).lower()))
            name = "%s_ablation_%s" % (protocol, ablation)
            overrides.append("experiment.name=%s" % name)
            overrides.append("experiment.output_dir=outputs/paper/%s" % name)
            plan.append((name, overrides))
    sensitivity = (
        ("steps", "diffusion.num_steps", config.matrix.diffusion_steps),
        ("difficulty", "progressive.max_difficulty", config.matrix.max_difficulties),
        ("candidates", "sampling.num_candidates", config.matrix.candidate_sizes),
        ("layers", "diffusion.denoiser_layers", config.matrix.denoiser_layers),
    )
    for label, key, values in sensitivity:
        for value in values:
            name = "adni_transfer_%s_%s" % (label, value)
            overrides = [
                "split.protocol=adni_transfer",
                "split.file=data/splits/adni_transfer.json",
                "%s=%s" % (key, value),
                "experiment.name=%s" % name,
                "experiment.output_dir=outputs/paper/%s" % name,
            ]
            if key == "sampling.num_candidates":
                overrides.append(
                    "sampling.candidate_batch_size=%s" % min(
                        int(value), int(config.sampling.candidate_batch_size)
                    )
                )
            plan.append((name, overrides))
    return plan


def command_matrix(args: argparse.Namespace) -> None:
    base = load_config(args.experiments, args.set)
    if not bool(base.matrix.enabled):
        raise ValueError("matrix.enabled=false in the experiment config")
    plan = _matrix_plan(base)
    payload = [{"name": name, "overrides": overrides} for name, overrides in plan]
    destination = _path(base, base.experiment.output_dir) / "matrix_plan.json"
    atomic_json_dump(payload, destination)
    print(json.dumps(payload, indent=2))
    if not args.execute:
        return
    for name, overrides in plan:
        LOGGER.info("matrix experiment %s", name)
        experiment = load_config(args.experiments, list(args.set or []) + overrides)
        _run(experiment, None)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mci-diff", description="MCI-Diff experiments")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="check manifest and feature files")
    _add_common(validate)
    validate.set_defaults(func=command_validate)

    split = commands.add_parser("split", help="write the configured patient split")
    _add_common(split)
    split.add_argument("--force", action="store_true")
    split.set_defaults(func=command_split)

    extract = commands.add_parser("extract", help="run a wrapped external sMRI encoder")
    _add_common(extract)
    extract.add_argument("--encoder", help="TorchScript HFCN-compatible model")
    extract.add_argument("--factory", help="Python module:function returning a model")
    extract.add_argument("--overwrite", action="store_true")
    extract.set_defaults(func=command_extract)

    stages = (
        ("train-diffusion", command_train_diffusion),
        ("generate", command_generate),
        ("train-classifier", command_train_classifier),
        ("evaluate", command_evaluate),
    )
    for name, function in stages:
        subparser = commands.add_parser(name)
        _add_common(subparser)
        subparser.add_argument("--fold", type=int, default=0)
        subparser.set_defaults(func=function)

    run = commands.add_parser("run", help="run the stage-wise pipeline")
    _add_common(run)
    run.add_argument("--folds", nargs="*", type=int)
    run.set_defaults(func=command_run)

    matrix = commands.add_parser("matrix", help="prepare or execute paper experiments")
    matrix.add_argument("--experiments", default="configs/paper_experiments.yaml")
    matrix.add_argument("--set", action="append", default=[])
    matrix.add_argument("--execute", action="store_true")
    matrix.set_defaults(func=command_matrix)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(bool(args.verbose))
    args.func(args)


if __name__ == "__main__":
    main()
