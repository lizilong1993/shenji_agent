from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ai.features import FEATURE_DIM, TACTICAL_FEATURE_DIM, action_feature_vector, find_operator, operator_to_feature
from .common import discover_dir, repo_root
from .maps import infer_map_id_from_scenario_id


def discover_dataset_paths(root: Optional[Path] = None) -> Dict[str, Path]:
    base = root or repo_root()
    datasets_root = base / "datasets"
    intent_dir = discover_dir(datasets_root, contains="WargameData_mini01", required_file="train.json")
    state_dir = discover_dir(datasets_root, contains="WG-StateGraph", required_file="label.csv")
    tactical_dir = discover_dir(datasets_root, required_file="Wargame_Feature_Dataset.csv")
    return {
        "intent": intent_dir,
        "state_graph": state_dir,
        "tactical": tactical_dir,
    }


def _read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk"]
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except Exception as exc:  # pragma: no cover
            last_error = exc
    raise last_error  # type: ignore[misc]


def _parse_list_like(value: object) -> List[float]:
    if isinstance(value, list):
        return [float(x) for x in value if isinstance(x, (int, float))]
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [float(x) for x in parsed if isinstance(x, (int, float))]
        except Exception:
            return []
    if isinstance(value, (int, float)):
        return [float(value)]
    return []


def _map_bucket_from_frame(frame: Dict[str, object], map_bucket_lookup: Dict[int, str]) -> str:
    filename = str(frame.get("filename", ""))
    map_id = infer_map_id_from_scenario_id(filename, map_bucket_lookup.keys())
    if map_id is None:
        return "medium"
    return map_bucket_lookup.get(map_id, "medium")


@dataclass
class IntentSample:
    sequence: np.ndarray
    target: np.ndarray
    bucket: str
    enemy_id: int
    enemy_pos: int


class IntentSequenceDataset(Dataset):
    def __init__(
        self,
        json_path: Path,
        *,
        history_len: int,
        horizons: int,
        map_bucket_lookup: Dict[int, str],
        max_samples: Optional[int] = None,
    ) -> None:
        self.samples: List[IntentSample] = []
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for stream in data:
            frames = [frame for frame in stream if isinstance(frame, dict)]
            for idx in range(history_len - 1, len(frames)):
                frame = frames[idx]
                for entry in frame.get("tank_enemy", []):
                    if len(entry) < 2 + horizons:
                        continue
                    enemy_id = int(entry[0])
                    enemy_pos = int(entry[1])
                    history = frames[idx - history_len + 1 : idx + 1]
                    features = []
                    for history_frame in history:
                        operator = find_operator(history_frame, enemy_id)
                        features.append(operator_to_feature(operator, frame=history_frame, fallback_pos=enemy_pos))
                    target = np.array([int(np.argmax(step)) for step in entry[2 : 2 + horizons]], dtype=np.int64)
                    self.samples.append(
                        IntentSample(
                            sequence=np.stack(features),
                            target=target,
                            bucket=_map_bucket_from_frame(frame, map_bucket_lookup),
                            enemy_id=enemy_id,
                            enemy_pos=enemy_pos,
                        )
                    )
        if max_samples and len(self.samples) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(self.samples), size=max_samples, replace=False)
            self.samples = [self.samples[int(idx)] for idx in sorted(indices.tolist())]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        sample = self.samples[idx]
        return {
            "sequence": torch.tensor(sample.sequence, dtype=torch.float32),
            "target": torch.tensor(sample.target, dtype=torch.long),
            "bucket": sample.bucket,
            "enemy_id": sample.enemy_id,
            "enemy_pos": sample.enemy_pos,
        }


class TacticalPositionDataset(Dataset):
    def __init__(self, csv_path: Path, *, max_rows: Optional[int] = None) -> None:
        self.features: np.ndarray
        self.labels: np.ndarray
        frame = _read_csv_with_fallback(csv_path, nrows=max_rows)
        label_column = next((col for col in frame.columns if col.lower() == "label"), frame.columns[-1])
        numeric_columns = []
        derived_rows: List[np.ndarray] = []
        for column in frame.columns:
            if column == label_column:
                continue
            try:
                series = pd.to_numeric(frame[column], errors="coerce")
                if series.notna().mean() > 0.8:
                    numeric_columns.append(series.fillna(0.0).to_numpy(dtype=np.float32))
                    continue
            except Exception:
                pass
            derived_rows.append(
                frame[column]
                .map(lambda item: float(len(_parse_list_like(item))) if isinstance(item, str) or isinstance(item, list) else 0.0)
                .to_numpy(dtype=np.float32)
            )

        stacked = numeric_columns + derived_rows
        if stacked:
            features = np.stack(stacked, axis=1)
        else:
            features = np.zeros((len(frame), TACTICAL_FEATURE_DIM), dtype=np.float32)
        if features.shape[1] < TACTICAL_FEATURE_DIM:
            features = np.pad(features, ((0, 0), (0, TACTICAL_FEATURE_DIM - features.shape[1])))
        elif features.shape[1] > TACTICAL_FEATURE_DIM:
            features = features[:, :TACTICAL_FEATURE_DIM]
        labels = pd.to_numeric(frame[label_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


class StateGraphDataset(Dataset):
    def __init__(self, root_dir: Path, *, map_bucket_lookup: Dict[int, str]) -> None:
        labels = _read_csv_with_fallback(root_dir / "label.csv")
        self.label_lookup = {row["filename"]: float(row["label"]) for _, row in labels.iterrows()}
        self.files = sorted((root_dir / "encoded_nodes-features").glob("*.csv"))
        self.map_bucket_lookup = map_bucket_lookup

    def __len__(self) -> int:
        return len(self.files)

    def _load_graph(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        frame = _read_csv_with_fallback(path, nrows=1)
        parsed_columns: List[List[float]] = []
        for column in frame.columns:
            parsed_columns.append(_parse_list_like(frame.iloc[0][column]))
        num_nodes = max((len(values) for values in parsed_columns), default=1)
        features = np.zeros((num_nodes, FEATURE_DIM), dtype=np.float32)
        for feat_idx, values in enumerate(parsed_columns[:FEATURE_DIM]):
            if not values:
                continue
            if len(values) == 1:
                features[:, feat_idx] = values[0]
            else:
                fill_values = values[:num_nodes] + [0.0] * max(0, num_nodes - len(values))
                features[:, feat_idx] = np.array(fill_values[:num_nodes], dtype=np.float32)
        similarity = features @ features.T
        np.fill_diagonal(similarity, 0.0)
        adjacency = np.eye(num_nodes, dtype=np.float32)
        if num_nodes > 1:
            top_k = min(4, num_nodes - 1)
            for row_idx in range(num_nodes):
                neighbor_idx = np.argsort(similarity[row_idx])[-top_k:]
                adjacency[row_idx, neighbor_idx] = 1.0
        return features, adjacency

    def __getitem__(self, idx: int) -> Dict[str, object]:
        path = self.files[idx]
        filename = path.name.replace("encoded_", "").replace("_nodes_features.csv", ".json")
        map_id = infer_map_id_from_scenario_id(filename, self.map_bucket_lookup.keys())
        bucket = self.map_bucket_lookup.get(map_id, "medium") if map_id is not None else "medium"
        features, adjacency = self._load_graph(path)
        label = self.label_lookup.get(filename, 0.0)
        return {
            "x": torch.tensor(features, dtype=torch.float32),
            "adj": torch.tensor(adjacency, dtype=torch.float32),
            "y": torch.tensor([label], dtype=torch.float32),
            "bucket": bucket,
        }


def discover_training_files(root: Optional[Path] = None) -> Dict[str, Path]:
    paths = discover_dataset_paths(root)
    return {
        "intent_train": paths["intent"] / "train.json",
        "intent_val": paths["intent"] / "test.json",
        "tactical_csv": paths["tactical"] / "Wargame_Feature_Dataset.csv",
        "state_graph_root": paths["state_graph"],
    }
