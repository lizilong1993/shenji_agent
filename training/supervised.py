from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from ai.models import GraphSituationModel, MultiHorizonIntentionModel, TacticalScorer
from .common import copy_publish_artifacts, dump_yaml, ensure_dir, load_config, now_tag, repo_root, safe_device, save_json, set_seed
from .data import IntentSequenceDataset, StateGraphDataset, TacticalPositionDataset, discover_training_files
from .maps import build_map_catalog, map_bucket_lookup


def _bucket_metrics(records: Iterable[Tuple[str, bool]]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for bucket, hit in records:
        grouped[bucket].append(float(hit))
    return {bucket: float(np.mean(values)) for bucket, values in grouped.items()}


def _prepare_run_dir(output_root: str) -> Path:
    path = ensure_dir(repo_root() / output_root / now_tag())
    return path


def train_intent_model(
    config: Dict[str, object],
    training_files: Dict[str, Path],
    bucket_lookup: Dict[int, str],
    device: torch.device,
    run_dir: Path,
) -> Dict[str, object]:
    history_len = int(config["history_len"])
    horizons = int(config["horizons"])
    train_dataset = IntentSequenceDataset(
        training_files["intent_train"],
        history_len=history_len,
        horizons=horizons,
        map_bucket_lookup=bucket_lookup,
        max_samples=config.get("max_train_samples"),
    )
    val_dataset = IntentSequenceDataset(
        training_files["intent_val"],
        history_len=history_len,
        horizons=horizons,
        map_bucket_lookup=bucket_lookup,
        max_samples=config.get("max_val_samples"),
    )
    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False)

    model = MultiHorizonIntentionModel(
        input_dim=108,
        hidden_dim=int(config["hidden_dim"]),
        horizons=horizons,
        num_classes=7,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    train_losses: List[float] = []
    for _ in range(int(config["epochs"])):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["sequence"].to(device))
            loss = 0.0
            target = batch["target"].to(device)
            for horizon_idx in range(horizons):
                loss = loss + criterion(logits[:, horizon_idx, :], target[:, horizon_idx])
            loss = loss / horizons
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        train_losses.append(epoch_loss / max(batches, 1))

    model.eval()
    all_hits: List[Tuple[str, bool]] = []
    per_horizon_hits: List[List[float]] = [[] for _ in range(horizons)]
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["sequence"].to(device))
            preds = logits.argmax(dim=-1).cpu()
            target = batch["target"]
            buckets = batch["bucket"]
            for row_idx in range(preds.shape[0]):
                immediate_hit = int(preds[row_idx, 0].item() == target[row_idx, 0].item())
                all_hits.append((buckets[row_idx], bool(immediate_hit)))
                for horizon_idx in range(horizons):
                    per_horizon_hits[horizon_idx].append(float(preds[row_idx, horizon_idx].item() == target[row_idx, horizon_idx].item()))

    weight_path = run_dir / "intent_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "history_len": history_len,
            "horizons": horizons,
            "feature_dim": 108,
            "num_classes": 7,
        },
        weight_path,
    )
    return {
        "train_loss": train_losses[-1] if train_losses else None,
        "val_immediate_accuracy": float(np.mean([hit for _, hit in all_hits])) if all_hits else 0.0,
        "bucket_accuracy": _bucket_metrics(all_hits),
        "horizon_accuracy": {f"h{idx+1}": float(np.mean(values)) for idx, values in enumerate(per_horizon_hits)},
        "samples": {
            "train": len(train_dataset),
            "val": len(val_dataset),
        },
        "artifact": weight_path.name,
    }


def train_tactical_model(
    config: Dict[str, object],
    training_files: Dict[str, Path],
    device: torch.device,
    run_dir: Path,
) -> Dict[str, object]:
    dataset = TacticalPositionDataset(training_files["tactical_csv"], max_rows=config.get("max_rows"))
    indices = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=int(config["batch_size"]), shuffle=False)

    model = TacticalScorer(input_dim=32, hidden_dim=int(config["hidden_dim"])).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    losses = []
    for _ in range(int(config["epochs"])):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features.to(device)).squeeze(-1)
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        losses.append(epoch_loss / max(batches, 1))

    model.eval()
    preds = []
    labels_all = []
    with torch.no_grad():
        for features, labels in val_loader:
            logits = model(features.to(device)).squeeze(-1)
            preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy().tolist())
            labels_all.extend(labels.numpy().tolist())
    accuracy = float(np.mean([int(pred == label) for pred, label in zip(preds, labels_all)])) if labels_all else 0.0
    weight_path = run_dir / "tactical_model.pt"
    torch.save({"model_state": model.state_dict(), "input_dim": 32}, weight_path)
    return {
        "train_loss": losses[-1] if losses else None,
        "val_accuracy": accuracy,
        "samples": len(dataset),
        "artifact": weight_path.name,
    }


def train_situation_model(
    config: Dict[str, object],
    training_files: Dict[str, Path],
    bucket_lookup: Dict[int, str],
    device: torch.device,
    run_dir: Path,
) -> Dict[str, object]:
    dataset = StateGraphDataset(training_files["state_graph_root"], map_bucket_lookup=bucket_lookup)
    indices = np.arange(len(dataset))
    if len(indices) < 2:
        return {"skipped": True, "reason": "not_enough_graph_samples"}
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    model = GraphSituationModel(num_features=108, hidden_dim=int(config["hidden_dim"])).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    train_losses = []
    for _ in range(int(config["epochs"])):
        model.train()
        epoch_loss = 0.0
        for idx in train_idx.tolist():
            sample = dataset[idx]
            optimizer.zero_grad()
            pred = model(sample["x"].to(device), sample["adj"].to(device))
            loss = criterion(pred.view(-1), sample["y"].to(device).view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        train_losses.append(epoch_loss / max(len(train_idx), 1))

    model.eval()
    hits: List[Tuple[str, bool]] = []
    with torch.no_grad():
        for idx in val_idx.tolist():
            sample = dataset[idx]
            pred = model(sample["x"].to(device), sample["adj"].to(device)).view(-1).item()
            label = sample["y"].view(-1).item()
            hits.append((sample["bucket"], (pred >= 0.5) == (label >= 0.5)))

    weight_path = run_dir / "situation_model.pt"
    torch.save({"model_state": model.state_dict(), "input_dim": 108}, weight_path)
    return {
        "train_loss": train_losses[-1] if train_losses else None,
        "val_accuracy": float(np.mean([hit for _, hit in hits])) if hits else 0.0,
        "bucket_accuracy": _bucket_metrics(hits),
        "samples": len(dataset),
        "artifact": weight_path.name,
    }


def build_summary(metrics: Dict[str, object], run_dir: Path) -> None:
    lines = [
        "# Shenji Supervised Training Summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- intent_immediate_accuracy: `{metrics['intent']['val_immediate_accuracy']:.4f}`",
        f"- tactical_accuracy: `{metrics['tactical'].get('val_accuracy', 0.0):.4f}`",
        f"- situation_accuracy: `{metrics['situation'].get('val_accuracy', 0.0):.4f}`",
        f"- complex_bucket_intent_accuracy: `{metrics['intent']['bucket_accuracy'].get('complex', 0.0):.4f}`",
        f"- medium_bucket_intent_accuracy: `{metrics['intent']['bucket_accuracy'].get('medium', 0.0):.4f}`",
        f"- simple_bucket_intent_accuracy: `{metrics['intent']['bucket_accuracy'].get('simple', 0.0):.4f}`",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supervised training for Shenji.")
    parser.add_argument("--config", default="configs/supervised_baseline.yaml")
    args = parser.parse_args()

    config = load_config(repo_root() / args.config)
    set_seed(int(config["seed"]))
    run_dir = _prepare_run_dir(str(config["output_root"]))
    device = safe_device()

    maps_root = repo_root() / "land_wargame_sdk" / "Data" / "Data" / "maps"
    catalog_path = repo_root() / "training" / "cache" / "map_catalog.json"
    records = build_map_catalog(maps_root, catalog_path)
    bucket_lookup = map_bucket_lookup(records)
    training_files = discover_training_files()

    metrics = {
        "device": str(device),
        "map_catalog": [record.__dict__ for record in records],
        "intent": train_intent_model(config["intent"], training_files, bucket_lookup, device, run_dir),
        "tactical": train_tactical_model(config["tactical"], training_files, device, run_dir),
        "situation": train_situation_model(config["situation"], training_files, bucket_lookup, device, run_dir),
    }

    save_json(metrics, run_dir / "metrics.json")
    save_json(
        {
            "intent_bucket_accuracy": metrics["intent"]["bucket_accuracy"],
            "situation_bucket_accuracy": metrics["situation"].get("bucket_accuracy", {}),
        },
        run_dir / "slice_report.json",
    )
    dump_yaml(config, run_dir / "config.yaml")
    build_summary(metrics, run_dir)

    if config.get("publish_to_ai_weights", False):
        manifest = {
            "generated_at": now_tag(),
            "history_len": int(config["intent"]["history_len"]),
            "horizons": int(config["intent"]["horizons"]),
            "intent_model": "intent_model.pt",
            "tactical_model": "tactical_model.pt",
            "situation_model": "situation_model.pt",
        }
        copy_publish_artifacts(run_dir, repo_root() / "ai" / "weights", manifest)


if __name__ == "__main__":
    main()
