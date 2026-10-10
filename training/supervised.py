from __future__ import annotations

import argparse
import json
from collections import Counter
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
from .common import copy_publish_artifacts, dump_yaml, ensure_dir, git_head_sha, load_config, now_tag, repo_root, safe_device, save_json, set_seed
from .data import IntentSequenceDataset, StateGraphDataset, TacticalPositionDataset, discover_training_files
from .maps import build_map_catalog, map_bucket_lookup


def _fit_feature_normalizer(feature_batches: Iterable[np.ndarray]) -> Dict[str, object]:
    stacked = np.concatenate([np.asarray(batch, dtype=np.float32) for batch in feature_batches], axis=0)
    feat_mean = stacked.mean(axis=0).astype(np.float32)
    raw_std = stacked.std(axis=0).astype(np.float32)
    zero_std_mask = raw_std < 1e-8
    safe_std = np.where(zero_std_mask, 1.0, raw_std).astype(np.float32)
    return {
        "feature_mean": feat_mean.tolist(),
        "feature_std": safe_std.tolist(),
        "feature_std_raw": raw_std.tolist(),
        "feature_dim": int(stacked.shape[1]),
        "num_samples": int(stacked.shape[0]),
        "zero_std_count": int(np.sum(zero_std_mask)),
        "zero_std_features": [int(idx) for idx in np.where(zero_std_mask)[0].tolist()],
        "fitted_on": "situation_train_split",
        "normalizer_kind": "per_dimension_zscore",
    }


def _apply_feature_normalizer(features: torch.Tensor, normalizer: Dict[str, object]) -> torch.Tensor:
    mean = torch.tensor(normalizer["feature_mean"], dtype=features.dtype, device=features.device)
    std = torch.tensor(normalizer["feature_std"], dtype=features.dtype, device=features.device)
    if mean.numel() != features.shape[1] or std.numel() != features.shape[1]:
        raise ValueError(
            f"Normalizer feature_dim mismatch: features={features.shape[1]} "
            f"mean={mean.numel()} std={std.numel()}"
        )
    return (features - mean) / std


def _bucket_metrics(records: Iterable[Tuple[str, bool]]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for bucket, hit in records:
        grouped[bucket].append(float(hit))
    return {bucket: float(np.mean(values)) for bucket, values in grouped.items()}


def _counter_to_dict(counter: Counter[str]) -> Dict[str, int]:
    return {key: int(value) for key, value in sorted(counter.items())}


def _prepare_run_dir(output_root: str) -> Path:
    path = ensure_dir(repo_root() / output_root / now_tag())
    return path


def train_intent_model(
    config: Dict[str, object],
    training_files: Dict[str, Path],
    bucket_lookup: Dict[int, str],
    dataset_contract: Dict[str, object],
    device: torch.device,
    run_dir: Path,
) -> Dict[str, object]:
    history_len = int(config["history_len"])
    horizons = int(config["horizons"])
    map_id_strategy = str(dataset_contract.get("map_id_strategy", "suffix"))
    filename_segment_index = dataset_contract.get("filename_segment_index")
    train_dataset = IntentSequenceDataset(
        training_files["intent_train"],
        history_len=history_len,
        horizons=horizons,
        map_bucket_lookup=bucket_lookup,
        map_id_strategy=map_id_strategy,
        filename_segment_index=int(filename_segment_index) if filename_segment_index is not None else None,
        max_samples=config.get("max_train_samples"),
    )
    val_dataset = IntentSequenceDataset(
        training_files["intent_val"],
        history_len=history_len,
        horizons=horizons,
        map_bucket_lookup=bucket_lookup,
        map_id_strategy=map_id_strategy,
        filename_segment_index=int(filename_segment_index) if filename_segment_index is not None else None,
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

    weight_path = run_dir / "intent_lstm.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "history_len": history_len,
            "horizons": horizons,
            "feature_dim": 108,
            "num_classes": 7,
            "hidden_dim": int(config["hidden_dim"]),
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
        "bucket_counts": {
            "train": _counter_to_dict(train_dataset.bucket_counts),
            "val": _counter_to_dict(val_dataset.bucket_counts),
        },
        "schema": {
            "map_id_strategy": map_id_strategy,
            "filename_segment_index": int(filename_segment_index) if filename_segment_index is not None else None,
            "history_len": history_len,
            "horizons": horizons,
        },
        "artifact": weight_path.name,
    }


def train_tactical_model(
    config: Dict[str, object],
    training_files: Dict[str, Path],
    dataset_contract: Dict[str, object],
    device: torch.device,
    run_dir: Path,
) -> Dict[str, object]:
    label_column = str(dataset_contract.get("label_col", "样本标签"))
    dataset = TacticalPositionDataset(
        training_files["tactical_csv"],
        label_column=label_column,
        max_rows=config.get("max_rows"),
    )
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
        "schema": {
            "label_column": dataset.label_column,
            "feature_columns": dataset.feature_columns,
        },
        "label_distribution": dataset.label_distribution,
        "artifact": weight_path.name,
    }


def train_situation_model(
    config: Dict[str, object],
    training_files: Dict[str, Path],
    bucket_lookup: Dict[int, str],
    dataset_contract: Dict[str, object],
    device: torch.device,
    run_dir: Path,
) -> Dict[str, object]:
    map_id_strategy = str(dataset_contract.get("map_id_strategy", "prefix"))
    filename_segment_index = dataset_contract.get("filename_segment_index")
    dataset = StateGraphDataset(
        training_files["state_graph_root"],
        map_bucket_lookup=bucket_lookup,
        map_id_strategy=map_id_strategy,
        filename_segment_index=int(filename_segment_index) if filename_segment_index is not None else None,
    )
    indices = np.arange(len(dataset))
    if len(indices) < 2:
        return {"skipped": True, "reason": "not_enough_graph_samples"}
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

    # Fit on train split only. The same transform is applied to train, val, and
    # later inference through the published normalizer artifact.
    train_feature_batches = []
    for idx in train_idx.tolist():
        sample = dataset[idx]
        train_feature_batches.append(sample["x"].numpy())
    normalizer = _fit_feature_normalizer(train_feature_batches)
    normalizer_path = run_dir / "situation_feature_normalizer.json"
    save_json(normalizer, normalizer_path)

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
            x = _apply_feature_normalizer(sample["x"].to(device), normalizer)
            pred = model(x, sample["adj"].to(device))
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
            x = _apply_feature_normalizer(sample["x"].to(device), normalizer)
            pred = model(x, sample["adj"].to(device)).view(-1).item()
            label = sample["y"].view(-1).item()
            hits.append((sample["bucket"], (pred >= 0.5) == (label >= 0.5)))

    weight_path = run_dir / "situation_gcn.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": 108,
            "hidden_dim": int(config["hidden_dim"]),
            "training_applied_normalizer": True,
            "normalizer_kind": normalizer["normalizer_kind"],
        },
        weight_path,
    )
    return {
        "train_loss": train_losses[-1] if train_losses else None,
        "val_accuracy": float(np.mean([hit for _, hit in hits])) if hits else 0.0,
        "bucket_accuracy": _bucket_metrics(hits),
        "samples": len(dataset),
        "bucket_counts": _counter_to_dict(dataset.bucket_counts),
        "schema": {
            "map_id_strategy": map_id_strategy,
            "filename_segment_index": int(filename_segment_index) if filename_segment_index is not None else None,
        },
        "artifact": weight_path.name,
        "normalizer": {
            "path": normalizer_path.name,
            "kind": normalizer["normalizer_kind"],
            "fitted_on": normalizer["fitted_on"],
            "feature_dim": int(normalizer["feature_dim"]),
            "num_samples": int(normalizer["num_samples"]),
            "zero_std_count": int(normalizer["zero_std_count"]),
            "training_applied_normalizer": True,
        },
    }


# ── I06 步骤 3：objective 模型（由引擎裁决标签真实训练）────────────────────
#
# 数据来源：`tools/build_objective_labels.py` 产出的 `objective_labels.jsonl`。
# 第 51 轮修好标签器后，高置信标签（`damage_kill_delta` / `occupy_success`）达 3181 条。
#
# 特征口径：**全部取自标签既有字段，不自造**。
# 目标：`reward_proxy`（由 `compute_reward_proxy()` 依据同一批事件算出，已含尺度归一）。
OBJECTIVE_FEATURE_NAMES = [
    "damage_dealt", "damage_taken", "kill_dealt", "kill_taken",
    "occupy_success", "distance_delta", "city_value", "side",
    "has_damage", "has_kill",
]
OBJECTIVE_FEATURE_DIM = len(OBJECTIVE_FEATURE_NAMES)


def _objective_feature_vector(label: Dict[str, object]) -> "np.ndarray":
    def g(key: str) -> float:
        value = label.get(key)
        if value is None or value is False:
            return 0.0
        if value is True:
            return 1.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    damage_dealt = g("damage_dealt")
    damage_taken = g("damage_taken")
    kill_dealt = g("kill_dealt")
    kill_taken = g("kill_taken")
    return np.asarray([
        damage_dealt, damage_taken, kill_dealt, kill_taken,
        g("occupy_success"), g("distance_delta"), g("city_value"), g("side"),
        1.0 if (damage_dealt or damage_taken) else 0.0,
        1.0 if (kill_dealt or kill_taken) else 0.0,
    ], dtype=np.float32)


def load_objective_labels(path: Path) -> List[Dict[str, object]]:
    """读 `objective_labels.jsonl`；坏行跳过而不抛（沿用既有风格）。"""
    out: List[Dict[str, object]] = []
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(record, dict) and record.get("reward_proxy") is not None:
                out.append(record)
    return out


def train_objective_model(config, labels_path, device, run_dir):
    """训练 objective 模型，返回写进 manifest 的指标。

    仅在**高置信**标签上训练：特征全为 0、目标恒为 0 的低置信标签
    只会教模型"输出 0"，故显式排除并写明数量。
    """
    labels = load_objective_labels(labels_path)
    high = [l for l in labels if str(l.get("label_confidence")) == "high"]
    if len(high) < 4:
        return {
            "model_trained": False,
            "objective_mode": "heuristic_policy_score",
            "reason": "not_enough_high_confidence_labels",
            "label_total": len(labels),
            "high_confidence": len(high),
        }

    from ai.models import ObjectiveModel  # noqa: PLC0415

    x = np.stack([_objective_feature_vector(l) for l in high])
    y = np.asarray([float(l["reward_proxy"]) for l in high], dtype=np.float32)
    train_idx, val_idx = train_test_split(
        np.arange(len(high)), test_size=0.2, random_state=42, shuffle=True)

    model = ObjectiveModel(num_features=OBJECTIVE_FEATURE_DIM,
                           hidden_dim=int(config.get("hidden_dim", 64))).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)

    train_losses = []
    for _ in range(int(config.get("epochs", 20))):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_t[train_idx]), y_t[train_idx])
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        pred_val = model(x_t[val_idx]).cpu().numpy()
    y_val = y[val_idx]
    mae = float(np.mean(np.abs(pred_val - y_val)))
    sign_agree = float(np.mean(np.sign(pred_val) == np.sign(y_val)))
    # 朴素基线：恒输出**训练集均值** —— 用来判断"训练了"是否等于"学到东西"
    train_mean = float(np.mean(y[train_idx]))
    baseline_mae = float(np.mean(np.abs(np.full_like(y_val, train_mean) - y_val)))

    weight_path = run_dir / "objective_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": OBJECTIVE_FEATURE_DIM,
            "hidden_dim": int(config.get("hidden_dim", 64)),
            "feature_names": OBJECTIVE_FEATURE_NAMES,
            "target": "reward_proxy",
        },
        weight_path,
    )
    return {
        "model_trained": True,
        "objective_mode": "trained_objective_model",
        "train_loss": train_losses[-1] if train_losses else None,
        "val_mae": mae,
        "val_sign_agreement": sign_agree,
        "baseline_mae": baseline_mae,
        "improves_over_baseline": bool(mae < baseline_mae),
        "label_total": len(labels),
        "high_confidence": len(high),
        "excluded_non_high": len(labels) - len(high),
        "feature_names": OBJECTIVE_FEATURE_NAMES,
        "artifact": weight_path.name,
    }


def _objective_label_counts(objective_metrics):
    """整理成 manifest 的 `objective_label_counts`。

    `high_confidence` 是 `I06` **判据 1 的取证字段**，故单独列出、不藏在别处。
    """
    return {
        "total": int(objective_metrics.get("label_total") or 0),
        "high_confidence": int(objective_metrics.get("high_confidence") or 0),
        "medium_confidence": 0,
        "low_confidence": int(objective_metrics.get("excluded_non_high") or 0),
    }


def build_summary(metrics: Dict[str, object], run_dir: Path) -> None:
    tactical = metrics.get("tactical", {})
    tactical_line = (
        f"- tactical_accuracy: `{tactical.get('val_accuracy', 0.0):.4f}`"
        if not tactical.get("skipped")
        else f"- tactical_accuracy: `skipped ({tactical.get('reason', 'disabled')})`"
    )
    lines = [
        "# Shenji Supervised Training Summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- intent_immediate_accuracy: `{metrics['intent']['val_immediate_accuracy']:.4f}`",
        tactical_line,
        f"- situation_accuracy: `{metrics['situation'].get('val_accuracy', 0.0):.4f}`",
        f"- complex_bucket_intent_accuracy: `{metrics['intent']['bucket_accuracy'].get('complex', 0.0):.4f}`",
        f"- medium_bucket_intent_accuracy: `{metrics['intent']['bucket_accuracy'].get('medium', 0.0):.4f}`",
        f"- simple_bucket_intent_accuracy: `{metrics['intent']['bucket_accuracy'].get('simple', 0.0):.4f}`",
        f"- intent_bucket_counts_train: `{metrics['intent']['bucket_counts']['train']}`",
        f"- intent_bucket_counts_val: `{metrics['intent']['bucket_counts']['val']}`",
        f"- situation_bucket_counts: `{metrics['situation'].get('bucket_counts', {})}`",
        f"- situation_normalizer: `{metrics['situation'].get('normalizer', {})}`",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _experiment_id(generated_at: str) -> str:
    """实验 ID：`<generated_at>-<commit 前 8 位>`（I07 步骤 1 的命名规范）。

      - 前缀是**训练产出时刻**（`now_tag()` 的 `YYYYmmdd_HHMMSS`），可按时间排序；
      - 后缀是**代码状态**，把「这一次实验」与「产生它的提交」绑定；
      - **完全确定**：同一时刻 + 同一提交 -> 同一 ID，不含随机数，
        故第三方复跑时能判断「是不是同一次实验」。
    """
    sha = git_head_sha(repo_root()) or ""
    return "%s-%s" % (generated_at, sha[:8]) if sha else generated_at


def _requirements_lock_sha() -> "Optional[str]":
    """依赖锁文件哈希。优先 `requirements.lock`，退回 `requirements.txt`。

    找不到时返回 `None`（**不编造**）—— 验证脚本据此把依赖一项标为
    「历史产物、无法核验」，而不是假装核过。
    """
    root = repo_root()
    for name in ("requirements.lock", "requirements.txt"):
        p = root / name
        if p.is_file():
            import hashlib

            return hashlib.sha256(p.read_bytes()).hexdigest()
    return None


def _artifact_sha256(names: "Sequence[str]", root: Path) -> Dict[str, str]:
    """对 `root` 下**实际存在**的 `names` 计算 sha256（I07 步骤 3 要求「与声明的哈希一致」）。

    **没有这张表，"逐字节核对"就无从谈起**：校验脚本只能打印实算值，
    回答不了「这份权重是不是 manifest 说的那一份」。同形状的另一次训练产物、
    甚至把 tactical 权重改名顶替 intent，都会判通过 —— 那正是评审实测出来的缺陷。

    ⚠ 调用点必须传**本次产物所在目录**（发布时为发布目录），
    **不能**默认读 `ai/weights/`：那会把上一次训练的旧哈希写进本次 manifest。
    读不到就返回空 dict，由校验脚本标注「未核验哈希」，而不是编一个值。
    """
    import hashlib

    out: Dict[str, str] = {}
    if not Path(root).is_dir():
        return out
    for name in names:
        f = Path(root) / name
        if f.is_file():
            out[name] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def _build_environment_record(experiment_id: str, generated_at: str) -> Dict[str, object]:
    """环境记录（I07 步骤 2）。

    必须**整块是 dict**，不能是「顶层几个标量」：`copy_publish_artifacts()`
    会把 manifest 里**任何字符串值**当成文件路径去 `run_dir / rel_path` 复制
    （只跳过 `generated_at` 与 `managed_files`）。
    收进一个 dict 可让该函数原样继续工作，**不必改它**。
    """
    import platform

    env: Dict[str, object] = {
        "experiment_id": experiment_id,
        "generated_at": generated_at,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "requirements_lock_sha256": _requirements_lock_sha(),
        "naming_rule": "checkpoint_policy/1: <generated_at>-<commit8>",
        "torch_version": None,
        # `artifact_sha256` **不在此处填**：本函数只知道 run_dir，而 manifest 会随产物
        # 一起被复制到发布位。哈希由 `copy_publish_artifacts()` 在**复制完成后**按
        # 发布目录的**实际字节**填入 —— 这样声明的一定是随本次产物发布的那一份。
        # 放这里填会描述"发布位上已有的旧文件"，正是评审点出的静默漏发布隐患。
        "artifact_sha256": {},
    }
    try:
        import torch  # noqa: PLC0415 - 缺 torch 也不该让清单构造失败

        env["torch_version"] = getattr(torch, "__version__", None)
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001 - 环境记录尽力而为，不得阻断训练
        env["torch_version"] = None
    return env


def _build_model_manifest(config: Dict[str, object], metrics: Dict[str, object], *, tactical_skipped: bool) -> Dict[str, object]:
    managed_files = [
        "intent_lstm.pt",
        "situation_gcn.pt",
        "situation_feature_normalizer.json",
        "model_manifest.json",
    ]
    if not tactical_skipped:
        managed_files.append("tactical_model.pt")
    if (metrics.get("objective") or {}).get("model_trained"):
        managed_files.append("objective_model.pt")
    situation_normalizer = metrics["situation"].get("normalizer", {})
    # 先取出 generated_at：experiment_id 由它派生，两处必须用**同一个**值，
    # 否则 now_tag() 被调两次会得到相差一秒的 ID 与时间戳。
    generated_at = now_tag()
    return {
        "generated_at": generated_at,
        "source_commit_sha": git_head_sha(repo_root()),
        "history_len": int(config["intent"]["history_len"]),
        "horizons": int(config["intent"]["horizons"]),
        # 声明的文件名 = **本次训练在 run_dir 里写出的**规范名（见 supervised.py 的落盘处）。
        # ⚠ 不能按发布位 `ai/weights/` 的现有内容取名：那会让"上一次训练留下的旧文件"
        # 决定本次 manifest 的声明，于是 `copy_publish_artifacts` 会去找
        # `run_dir/<旧名>` 而找不到，**新权重被静默漏发布**，且声明的哈希描述的是旧产物。
        # 命名解析（.pt vs 历史 .pth）放在**发布时**处理，不放在 manifest 构造时。
        "intent_lstm": "intent_lstm.pt",
        "situation_gcn": "situation_gcn.pt",
        "situation_feature_normalizer": "situation_feature_normalizer.json",
        "situation_normalizer_kind": situation_normalizer.get("kind"),
        "situation_normalizer_fitted_on": situation_normalizer.get("fitted_on"),
        "situation_normalizer_feature_dim": situation_normalizer.get("feature_dim"),
        "situation_normalizer_zero_std_count": situation_normalizer.get("zero_std_count"),
        "situation_training_applied_normalizer": situation_normalizer.get("training_applied_normalizer"),
        "tactical_model": "tactical_model.pt" if not tactical_skipped else None,
          # ── I06 步骤 3：objective 模型（**纯加法**，不动既有键/既有类型）──
          # 判据 1 的取证出口就在这里：
          #   objective_metrics.model_trained 与 objective_label_counts.high_confidence
          "objective_enabled": bool((metrics.get("objective") or {}).get("model_trained")),
          "objective_model": ("objective_model.pt"
                              if (metrics.get("objective") or {}).get("model_trained")
                              else None),
          "objective_mode": (metrics.get("objective") or {}).get(
              "objective_mode", "heuristic_policy_score"),
          "objective_metrics": metrics.get("objective") or {},
          "objective_label_counts": _objective_label_counts(metrics.get("objective") or {}),
          # **相对仓库**路径，不是绝对路径 —— 与 manifest 里其他"产物名"字段区分开。
          # 它是**出处元数据**（标签从哪来），不是 run_dir 下的产物，
          # 故同时登记进 MANIFEST_NON_PATH_KEYS，避免 copy_publish_artifacts 误当产物复制。
          "objective_label_source": (
              "experiments/shenji_agent/objective_labels/latest/objective_labels.jsonl"
          ),
        "managed_files": managed_files,
        # I07 新增：实验 ID / 依赖版本 / 环境记录（**整块 dict**，理由见 _build_environment_record）。
        # 与 I06 的耦合点：两者都可能改本函数。本扩展是**纯加法**
        # （不动既有键、不改既有值的类型），新信息集中在单独一个 dict，
        # 故与 I06 的改动可机械合并 —— 这正是前置条件要求的那句。
        "environment": _build_environment_record(_experiment_id(generated_at), generated_at),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supervised training for Shenji.")
    parser.add_argument("--config", default="configs/supervised_baseline.yaml")
    args = parser.parse_args()

    config = load_config(repo_root() / args.config)
    set_seed(int(config["seed"]))
    run_dir = _prepare_run_dir(str(config["output_root"]))
    device = safe_device()
    dataset_contract = config.get("dataset_contract", {})

    maps_root = repo_root() / "land_wargame_sdk" / "Data" / "Data" / "maps"
    catalog_path = repo_root() / "training" / "cache" / "map_catalog.json"
    records = build_map_catalog(maps_root, catalog_path)
    bucket_lookup = map_bucket_lookup(records)
    training_files = discover_training_files()

    tactical_config = config.get("tactical", {})
    tactical_metrics: Dict[str, object]
    if tactical_config.get("enabled", False):
        tactical_metrics = train_tactical_model(
            tactical_config,
            training_files,
            dataset_contract.get("tactical", {}),
            device,
            run_dir,
        )
    else:
        tactical_metrics = {"skipped": True, "reason": "disabled_by_config"}

    metrics = {
        "device": str(device),
        "map_catalog": [record.__dict__ for record in records],
        "dataset_contract": dataset_contract,
        "intent": train_intent_model(
            config["intent"],
            training_files,
            bucket_lookup,
            dataset_contract.get("intent", {}),
            device,
            run_dir,
        ),
        "tactical": tactical_metrics,
        "situation": train_situation_model(
            config["situation"],
            training_files,
            bucket_lookup,
            dataset_contract.get("situation", {}),
            device,
            run_dir,
        ),
        # I06 步骤 3：objective 模型由引擎裁决标签真实训练
        "objective": train_objective_model(
            config.get("objective", {}),
            repo_root() / "experiments" / "shenji_agent" / "objective_labels"
            / "latest" / "objective_labels.jsonl",
            device,
            run_dir,
        ),
    }

    save_json(metrics, run_dir / "metrics.json")
    save_json(
        {
            "intent_bucket_accuracy": metrics["intent"]["bucket_accuracy"],
            "intent_bucket_counts": metrics["intent"]["bucket_counts"],
            "situation_bucket_accuracy": metrics["situation"].get("bucket_accuracy", {}),
            "situation_bucket_counts": metrics["situation"].get("bucket_counts", {}),
        },
        run_dir / "slice_report.json",
    )
    dump_yaml(config, run_dir / "config.yaml")
    build_summary(metrics, run_dir)

    if config.get("publish_to_ai_weights", False):
        tactical_skipped = tactical_metrics.get("skipped", False)
        manifest = _build_model_manifest(config, metrics, tactical_skipped=tactical_skipped)
        copy_publish_artifacts(run_dir, repo_root() / "ai" / "weights", manifest)


if __name__ == "__main__":
    main()
