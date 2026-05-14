from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from ai.models import TacticalScorer
from scripts.validate_tactical_checkpoint import validate as validate_checkpoint
from scripts.validate_tactical_dataset import validate as validate_dataset


def test_checkpoint_validator_detects_existing_tactical_checkpoint(tmp_path: Path) -> None:
    model = TacticalScorer(input_dim=32, hidden_dim=8)
    checkpoint = tmp_path / "tactical_model.pt"
    torch.save({"model_state": model.state_dict(), "input_dim": 32}, checkpoint)

    report = validate_checkpoint(checkpoint)

    assert report["status"] == "TACTICAL_CHECKPOINT_COMPATIBLE"
    assert report["tactical_model_class"] == "ai.models.TacticalScorer"


def test_missing_checkpoint_produces_fail_closed_error(tmp_path: Path) -> None:
    report = validate_checkpoint(tmp_path / "missing.pt")

    assert report["status"] == "TACTICAL_CHECKPOINT_MISSING"
    assert report["errors"]


def test_dataset_validator_detects_dataset_schema(tmp_path: Path) -> None:
    dataset = tmp_path / "Wargame_Feature_Dataset.csv"
    frame = pd.DataFrame(
        {
            "ID": [1, 2],
            "开始点": [101, 102],
            "目标点": [201, 202],
            "样本标签": [1, 0],
        }
    )
    frame.to_csv(dataset, index=False)

    report = validate_dataset(dataset, sample_rows=2)

    assert report["status"] == "TACTICAL_DATASET_COMPATIBLE"
    assert report["label_column_present"] is True
    assert report["feature_matrix_shape_sample"][1] == 32


def test_missing_dataset_produces_fail_closed_error(tmp_path: Path) -> None:
    report = validate_dataset(tmp_path / "missing.csv")

    assert report["status"] == "TACTICAL_DATASET_MISSING"
    assert report["errors"]
