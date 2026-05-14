"""Tests for tactical asset restore validators.

These tests verify that validators correctly detect existing/missing assets
using only pickle-bytecode analysis (no torch import required).
"""

from __future__ import annotations

import csv
import io
import pickle
import zipfile
from pathlib import Path

from scripts.validate_tactical_checkpoint import validate as validate_checkpoint
from scripts.validate_tactical_dataset import validate as validate_dataset


def _make_zip_pt(data: dict, path: Path) -> None:
    """Create a minimal PyTorch zip-format .pt file with pickle."""
    pkl_data = pickle.dumps(data)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("archive/data.pkl", pkl_data)
        zf.writestr("archive/version", "3\n")
        zf.writestr("archive/.format_version", "1\n")
        zf.writestr("archive/byteorder", "little")
        zf.writestr("archive/.storage_alignment", "8\n")
    with open(path, "wb") as f:
        f.write(buf.getvalue())


def _make_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


# ---- Checkpoint validator tests ----


def test_checkpoint_validator_detects_existing_tactical_checkpoint(tmp_path: Path) -> None:
    """A .pt with all expected keys is COMPATIBLE."""
    data = {
        "model_state": {
            "model.0.weight": [[0.1] * 32] * 8,
            "model.0.bias": [0.0] * 8,
            "model.3.weight": [[0.1] * 8] * 4,
            "model.3.bias": [0.0] * 4,
            "model.5.weight": [[0.1] * 4],
            "model.5.bias": [0.0],
        },
        "input_dim": 32,
    }
    checkpoint = tmp_path / "tactical_model.pt"
    _make_zip_pt(data, checkpoint)

    report = validate_checkpoint(checkpoint)
    assert report["status"] == "TACTICAL_CHECKPOINT_COMPATIBLE"


def test_missing_checkpoint_produces_fail_closed_error(tmp_path: Path) -> None:
    report = validate_checkpoint(tmp_path / "missing.pt")
    assert report["status"] == "TACTICAL_CHECKPOINT_MISSING"
    assert report["errors"]


def test_checkpoint_without_model_state_is_incompatible(tmp_path: Path) -> None:
    data = {"not_model_state": {}}
    checkpoint = tmp_path / "bad.pt"
    _make_zip_pt(data, checkpoint)

    report = validate_checkpoint(checkpoint)
    assert report["status"] == "TACTICAL_CHECKPOINT_PRESENT_BUT_INCOMPATIBLE"


def test_checkpoint_missing_keys_is_incompatible(tmp_path: Path) -> None:
    data = {
        "model_state": {
            "model.0.weight": [[0.1] * 32] * 8,
            # missing bias and all other keys
        },
    }
    checkpoint = tmp_path / "partial.pt"
    _make_zip_pt(data, checkpoint)

    report = validate_checkpoint(checkpoint)
    assert report["status"] == "TACTICAL_CHECKPOINT_PRESENT_BUT_INCOMPATIBLE"
    assert "model.0.bias" in str(report["errors"])


# ---- Dataset validator tests ----


def test_dataset_validator_detects_dataset_schema(tmp_path: Path) -> None:
    dataset = tmp_path / "Wargame_Feature_Dataset.csv"
    _make_csv(dataset, [
        {"ID": "1", "开始点": "101", "目标点": "201", "样本标签": "1", "其他特征": "0.5"},
        {"ID": "2", "开始点": "102", "目标点": "202", "样本标签": "0", "其他特征": "0.3"},
    ])

    report = validate_dataset(dataset)
    assert report["status"] == "TACTICAL_DATASET_COMPATIBLE"
    assert report["label_column_found"] is True
    assert report["feature_dim"] == 4  # 5 total - 1 label


def test_missing_dataset_produces_fail_closed_error(tmp_path: Path) -> None:
    report = validate_dataset(tmp_path / "missing.csv")
    assert report["status"] == "TACTICAL_DATASET_MISSING"
    assert report["errors"]


def test_dataset_missing_label_column(tmp_path: Path) -> None:
    dataset = tmp_path / "no_label.csv"
    _make_csv(dataset, [
        {"ID": "1", "特征1": "0.5", "特征2": "0.3"},
    ])

    report = validate_dataset(dataset)
    assert report["status"] == "TACTICAL_DATASET_PRESENT_BUT_SCHEMA_MISMATCH"
    assert not report["label_column_found"]
