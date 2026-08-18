import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mci_diff.config import load_config
from mci_diff.data import (
    FeatureNormalizer,
    SequenceRecord,
    VISIT_MONTHS,
    assemble_sequences,
    make_adni_transfer_split,
    partition_records,
    read_manifest,
)
from mci_diff.engine import (
    binary_metrics,
    mark_stage_complete,
    stage_is_complete,
)


def _sequence(subject_id, dataset, label, offset=0.0):
    features = np.arange(18, dtype=np.float32).reshape(6, 3) + float(offset)
    return SequenceRecord(
        subject_id=subject_id,
        dataset=dataset,
        label=label,
        features=features,
        observed=np.ones(6, dtype=bool),
    )


def test_config_override_rejects_unknown_key():
    default_config = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    with pytest.raises(KeyError, match="Unknown override key"):
        load_config(default_config, overrides=["sampling.num_canddates=10"])


def test_manifest_six_visits_and_train_only_normalization(tmp_path):
    fieldnames = [
        "subject_id",
        "dataset",
        "month",
        "label",
        "feature_path",
    ]
    rows = []
    subject_visits = {
        "train_001": ("ADNI1", "pMCI", VISIT_MONTHS),
        "heldout_001": ("ADNI2", "sMCI", (0,)),
    }
    for subject_id, (dataset, label, months) in subject_visits.items():
        for month in months:
            visit = VISIT_MONTHS.index(month)
            if subject_id.startswith("train"):
                feature = np.asarray([visit, 2 * visit + 1], dtype=np.float32)
            else:
                feature = np.asarray([100.0, 200.0], dtype=np.float32)
            feature_path = tmp_path / f"{subject_id}_{month}.npy"
            np.save(feature_path, feature)
            rows.append(
                {
                    "subject_id": subject_id,
                    "dataset": dataset,
                    "month": month,
                    "label": label,
                    "feature_path": feature_path.name,
                }
            )

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    scans = read_manifest(
        manifest,
        project_root=tmp_path,
        check_paths=True,
        require_baseline=True,
    )
    records = assemble_sequences(
        scans,
        feature_dim=2,
    )
    by_id = {record.subject_id: record for record in records}
    training = by_id["train_001"]
    heldout = by_id["heldout_001"]

    assert training.observed.tolist() == [True] * 6
    assert heldout.observed.tolist() == [True, False, False, False, False, False]
    assert np.allclose(training.features[5], [5.0, 11.0])

    normalizer = FeatureNormalizer.fit([training], partition="train")
    assert normalizer.count == 6
    assert np.allclose(normalizer.mean, [2.5, 6.0])
    normalized_training = normalizer.transform(training)
    normalized_heldout = normalizer.transform(heldout)
    assert np.allclose(normalized_training.features.mean(axis=0), 0.0, atol=1.0e-6)
    assert normalized_heldout.features[0, 0] > 20.0
    assert np.equal(normalized_heldout.features[1:], 0.0).all()
    with pytest.raises(ValueError, match="train"):
        FeatureNormalizer.fit([heldout], partition="validation")


def test_transfer_split_keeps_subjects_in_one_partition():
    records = [
        _sequence("a1_neg_1", "ADNI1", 0, 0),
        _sequence("a1_neg_2", "ADNI1", 0, 10),
        _sequence("a1_pos_1", "ADNI1", 1, 20),
        _sequence("a1_pos_2", "ADNI1", 1, 30),
        _sequence("a2_neg", "ADNI2", 0, 40),
        _sequence("a2_pos", "ADNI2", 1, 50),
        _sequence("aibl_pos", "AIBL", 1, 60),
    ]
    split = make_adni_transfer_split(records, validation_fraction=0.5, seed=9)
    partitions = partition_records(records, split)

    id_sets = {
        name: {record.subject_id for record in values}
        for name, values in partitions.items()
    }
    names = tuple(id_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            assert id_sets[left].isdisjoint(id_sets[right])
    assert {record.dataset for record in partitions["train"]} == {"ADNI1"}
    assert {record.dataset for record in partitions["validation"]} == {"ADNI1"}
    assert {record.dataset for record in partitions["test"]} == {"ADNI2"}
    assert {record.dataset for record in partitions["external"]} == {"AIBL"}


def test_stage_marker_checks_declared_outputs(tmp_path):
    output = tmp_path / "diffusion" / "final.pt"
    output.parent.mkdir()
    output.write_bytes(b"checkpoint")
    unrelated = tmp_path / "metrics.json"
    unrelated.write_text("{}", encoding="utf-8")

    marker = mark_stage_complete(
        tmp_path,
        "diffusion",
        [output],
        metadata={"fold": 0},
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))

    assert payload["stage"] == "diffusion"
    assert payload["outputs"] == [str(output.resolve())]
    assert stage_is_complete(tmp_path, "diffusion")
    assert stage_is_complete(tmp_path, "diffusion", [output])
    assert not stage_is_complete(tmp_path, "diffusion", [unrelated])


def test_binary_metrics_use_pmci_as_positive_class():
    metrics = binary_metrics(
        labels=[0, 0, 1, 1],
        probabilities=[0.1, 0.8, 0.4, 0.9],
        threshold=0.5,
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["sensitivity"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["auc"] == pytest.approx(0.75)
