#!/usr/bin/env python3
"""Diagnose learned model quality without running full self-play.

Usage:
    python scripts/diagnose_model_quality.py \
        --artifacts ai/weights \
        --datasets datasets \
        --output experiments/shenji_agent/model_quality_diagnosis_latest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.artifacts import missing_required_assets, resolution_to_dict, resolve_manifest_assets

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def _repo_root() -> Path:
    return ROOT


def _git_head() -> str:
    import subprocess
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(_repo_root()),
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _check_file(path: Path) -> bool:
    return path.exists() and path.is_file()


def _load_situation_model(weights_path: Path, device: str = "cpu") -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    if not TORCH_AVAILABLE:
        return None, None
    sys.path.insert(0, str(_repo_root()))
    try:
        from ai.models import GraphSituationModel
    finally:
        sys.path.pop(0)

    if not weights_path.exists():
        return None, None
    try:
        payload = torch.load(weights_path, map_location="cpu")
        # ── C4 修正（situation 路径）：与 `ai/policy.py::_load_situation_model` 对齐 ──
        # 与 intent 路径**同一个 bug**：原实现只支持包装型
        # （`payload["model_state"]` + `backbone.` 前缀），而仓库实际的
        # `ai/weights/situation_gcn.pth` 是**裸 state_dict**、键为 legacy 形式
        # `sage1.linear.weight`（**无 `backbone.` 前缀**）。
        # 于是同样 KeyError → 被吞 → 报 "Situation model: MISSING"，
        # 并驱动 `train_situation_model` 这条**基于误判**的建议。
        # 主链路的处置是：**把 legacy 键加 `backbone.` 前缀包一层**再 load。
        state_dict = payload["model_state"] if ("model_state" in payload) else payload
        sage_key = None
        for candidate_key in ("backbone.sage1.linear.weight", "sage1.linear.weight"):
            if candidate_key in state_dict:
                sage_key = candidate_key
                break
        if sage_key is not None:
            hidden_dim = int(state_dict[sage_key].shape[0])
        else:
            hidden_dim = int(payload.get("hidden_dim", 64))
        has_backbone_prefix = any(k.startswith("backbone.") for k in state_dict)
        if not has_backbone_prefix and "sage1.linear.weight" in state_dict:
            num_features = int(state_dict["sage1.linear.weight"].shape[1] // 2)
            model = GraphSituationModel(num_features=num_features, hidden_dim=hidden_dim)
            model.load_state_dict({f"backbone.{k}": v for k, v in state_dict.items()})
        else:
            input_dim = int(payload.get("input_dim", 108))
            model = GraphSituationModel(num_features=input_dim, hidden_dim=hidden_dim)
            model.load_state_dict(state_dict)
        model.eval()
        if device != "cpu":
            model = model.to(device)
        if isinstance(payload, dict):
            payload.setdefault("input_dim", int(model.backbone.sage1.linear.weight.shape[1] // 2))
            payload["hidden_dim"] = hidden_dim
            payload["_resolved_state_dict"] = True
            payload.pop("model_state", None)
        return model, payload
    except Exception as exc:
        return None, {"load_error": str(exc)}


def _load_intent_model(weights_path: Path, device: str = "cpu") -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    if not TORCH_AVAILABLE:
        return None, None
    sys.path.insert(0, str(_repo_root()))
    try:
        from ai.models import MultiHorizonIntentionModel
    finally:
        sys.path.pop(0)

    if not weights_path.exists():
        return None, None
    try:
        payload = torch.load(weights_path, map_location="cpu")
        # ── C4 修正：与主链路 `ai/policy.py::_load_intent_model` 对齐 ─────────
        # 原先这里**只**支持 `{"model_state": {...}, "horizons":..., "num_classes":...}`
        # 这种"包装型"检查点，并把 horizons/num_classes 写死为 5/7。
        # 但仓库里实际的检查点 `ai/weights/intention_lstm.pth` 是**裸 state_dict**、
        # 且是 legacy `fc_intent` 头、**8 个类别**。于是：
        #   1) `payload["model_state"]` 抛 KeyError → 被下面 except 吞掉；
        #   2) 诊断把**明明存在且主链路能正常加载**的模型报成 "Intent model: MISSING"；
        #   3) 于是 `move_bias_estimate` 从未真正计算过，而它却驱动了
        #      `address_intent_move_bias` 这条处置建议。
        # 下面按主链路同样的规则解析：支持裸 state_dict、支持 legacy 头、
        # 维度与类别数从权重形状**推断**而非写死。
        state_dict = payload["model_state"] if ("model_state" in payload) else payload
        lstm_key = "lstm.weight_ih_l0"
        if lstm_key in state_dict:
            hidden_dim = int(state_dict[lstm_key].shape[0] // 4)
            feature_dim = int(state_dict[lstm_key].shape[1])
        else:
            hidden_dim = int(payload.get("hidden_dim", 128))
            feature_dim = int(payload.get("feature_dim", 108))
        horizons = int(payload.get("horizons", 5))
        if any(k.startswith("head.") for k in state_dict):
            num_classes = int(payload.get("num_classes", 7))
            model = MultiHorizonIntentionModel(
                input_dim=feature_dim, hidden_dim=hidden_dim, horizons=horizons, num_classes=num_classes
            )
            model.load_state_dict(state_dict)
        elif "fc_intent.weight" in state_dict:
            # legacy IntentionLSTM：输出是 [batch, num_classes]（**2 维，没有 horizons 维**）
            from ai.models import IntentionLSTM

            num_classes = int(state_dict["fc_intent.weight"].shape[0])
            model = IntentionLSTM(input_dim=feature_dim, hidden_dim=hidden_dim, num_classes=num_classes)
            model.load_state_dict(state_dict)
        else:
            num_classes = int(payload.get("num_classes", 7))
            model = MultiHorizonIntentionModel(
                input_dim=feature_dim, hidden_dim=hidden_dim, horizons=horizons, num_classes=num_classes
            )
            model.load_state_dict(state_dict)
        model.eval()
        if device != "cpu":
            model = model.to(device)
        # 把推断出的维度回传，供下面算分布时使用（避免再写死 7）
        if isinstance(payload, dict):
            payload.setdefault("feature_dim", feature_dim)
            payload.setdefault("hidden_dim", hidden_dim)
            payload["num_classes"] = num_classes
            payload["_resolved_state_dict"] = True
            payload.pop("model_state", None)
        return model, payload
    except Exception as exc:
        return None, {"load_error": str(exc)}


def _load_tactical_model(weights_path: Path, device: str = "cpu") -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    if not TORCH_AVAILABLE:
        return None, None
    sys.path.insert(0, str(_repo_root()))
    try:
        from ai.models import TacticalScorer
    finally:
        sys.path.pop(0)

    if not weights_path.exists():
        return None, None
    try:
        payload = torch.load(weights_path, map_location="cpu")
        input_dim = int(payload.get("input_dim", 32))
        hidden_dim = int(payload["model_state"]["model.0.weight"].shape[0])
        model = TacticalScorer(input_dim=input_dim, hidden_dim=hidden_dim)
        model.load_state_dict(payload["model_state"])
        model.eval()
        if device != "cpu":
            model = model.to(device)
        return model, payload
    except Exception as exc:
        return None, {"load_error": str(exc)}


def _load_normalizer(artifact_dir: Path) -> Optional[Dict[str, Any]]:
    for name in ("situation_feature_normalizer.json", "feature_normalizer.json", "normalizer.json"):
        path = artifact_dir / name
        if path.exists():
            return _load_json(path)
    return None


def _safe_to_device(value: Any, device: str) -> Any:
    if device == "cpu":
        return value
    to_method = getattr(value, "to", None)
    if callable(to_method):
        return to_method(device)
    return value


def diagnose(artifact_dir: Path, datasets_dir: Optional[Path] = None) -> Dict[str, Any]:
    device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

    manifest_path = artifact_dir / "model_manifest.json"
    manifest = _load_json(manifest_path)

    resolutions = resolve_manifest_assets(artifact_dir, manifest or {})
    intent_resolution = resolutions.get("intent_lstm") or resolutions.get("intent_model")
    situation_resolution = resolutions.get("situation_gcn") or resolutions.get("situation_model")
    tactical_resolution = resolutions.get("tactical_model")
    intent_path = intent_resolution.path if intent_resolution else artifact_dir / "intent_lstm.pt"
    situation_path = situation_resolution.path if situation_resolution else artifact_dir / "situation_gcn.pt"
    tactical_path = tactical_resolution.path if tactical_resolution else artifact_dir / "tactical_model.pt"

    intent_model, intent_payload = _load_intent_model(intent_path, device)
    situation_model, situation_payload = _load_situation_model(situation_path, device)
    tactical_model, tactical_payload = _load_tactical_model(tactical_path, device)

    normalizer = _load_normalizer(artifact_dir)

    result: Dict[str, Any] = {
        "git_head": _git_head(),
        "manifest_path": str(manifest_path),
        "manifest_loaded": manifest is not None,
        "intent_model_path": str(intent_path) if intent_path.exists() else None,
        "situation_model_path": str(situation_path) if situation_path.exists() else None,
        "tactical_model_path": str(tactical_path) if tactical_path.exists() else None,
        "history_len": manifest.get("history_len") if manifest else None,
        "horizons": manifest.get("horizons") if manifest else None,
        "feature_dim": 108,
        "situation_checkpoint_loaded": situation_model is not None,
        "intent_checkpoint_loaded": intent_model is not None,
        "tactical_checkpoint_loaded": tactical_model is not None,
        "normalizer_available": normalizer is not None,
        "normalizer_loaded": False,
        "asset_resolutions": {key: resolution_to_dict(value) for key, value in resolutions.items()},
        "missing_required_assets": {key: resolution_to_dict(value) for key, value in missing_required_assets(resolutions).items()},
        "warnings": [],
        "recommended_next_actions": [],
    }

    if manifest:
        result["manifest_generated_at"] = manifest.get("generated_at")
        result["manifest_source_commit"] = manifest.get("source_commit_sha")
        result["manifest_managed_files"] = manifest.get("managed_files", [])
        result["normalizer_applied_in_training"] = bool(manifest.get("situation_training_applied_normalizer", False))
        result["normalizer_manifest_feature_dim"] = manifest.get("situation_normalizer_feature_dim")
        result["normalizer_manifest_kind"] = manifest.get("situation_normalizer_kind")
    else:
        result["normalizer_applied_in_training"] = False

    # Check managed files
    managed = manifest.get("managed_files", []) if manifest else []
    for fname in managed:
        if not (artifact_dir / fname).exists():
            result["warnings"].append(f"manifest_managed_file_missing: {fname}")

    # Situation model diagnostics
    if situation_model is not None:
        try:
            feat_dim = int(situation_payload.get("input_dim", 108)) if situation_payload else 108
            result["situation_checkpoint_training_applied_normalizer"] = bool(
                situation_payload.get("training_applied_normalizer", False)
            ) if situation_payload else False
            num_nodes_list = [4, 8, 16]
            pre_sigmoid_vals = []
            sigmoid_vals = []
            per_node_variances = []

            for num_nodes in num_nodes_list:
                x = _safe_to_device(torch.randn(num_nodes, feat_dim), device)
                adj = _safe_to_device(torch.eye(num_nodes), device)
                adj = adj + _safe_to_device(torch.bernoulli(torch.full((num_nodes, num_nodes), 0.3)), device)
                adj = adj.clamp(0, 1)

                with torch.no_grad():
                    backbone_out = situation_model.backbone(x, adj)
                    pre_sigmoid = backbone_out.detach().cpu().numpy()
                    mean_out = float(situation_model(x, adj).item())

                pre_sigmoid_vals.extend(pre_sigmoid.flatten().tolist())
                sigmoid_vals.append(mean_out)
                per_node_variances.append(float(np.var(pre_sigmoid)))

            pre_arr = np.array(pre_sigmoid_vals)
            result["situation_pre_sigmoid_mean"] = float(np.mean(pre_arr))
            result["situation_pre_sigmoid_std"] = float(np.std(pre_arr))
            result["situation_pre_sigmoid_min"] = float(np.min(pre_arr))
            result["situation_pre_sigmoid_max"] = float(np.max(pre_arr))
            result["situation_output_mean"] = float(np.mean(sigmoid_vals))
            result["situation_output_std"] = float(np.std(sigmoid_vals))
            result["situation_per_node_variance_mean"] = float(np.mean(per_node_variances))

            if np.std(pre_arr) < 0.01:
                result["warnings"].append("situation_pre_sigmoid_near_zero_variance")
            if np.mean(sigmoid_vals) < 0.01 or np.mean(sigmoid_vals) > 0.99:
                result["warnings"].append("situation_output_saturated")
            if float(np.mean(per_node_variances)) < 1e-6:
                result["warnings"].append("situation_per_node_zero_variance")
            if abs(float(np.mean(pre_arr))) > 20:
                result["warnings"].append("situation_pre_sigmoid_extreme_magnitude")
        except Exception as exc:
            result["warnings"].append(f"situation_diagnosis_error: {exc}")

    # Intent model diagnostics
    if intent_model is not None:
        try:
            feat_dim = int(intent_payload.get("feature_dim", 108)) if intent_payload else 108
            history_len = int(manifest.get("history_len", 8)) if manifest else 8
            horizons = int(intent_payload.get("horizons", 5)) if intent_payload else 5
            num_classes = int(intent_payload.get("num_classes", 7)) if intent_payload else 7

            seq = _safe_to_device(torch.randn(4, history_len, feat_dim), device)
            with torch.no_grad():
                logits = intent_model(seq)
                # ── C4 修正：按输出维度取"第一批次、第 0 个 horizon" ──────────
                # 主链路 `ai/policy.py::_intent_distribution` 就是这样做的（3 维取
                # `output[0, 0]`，2 维取 `output[0]`）。
                # 原实现无条件写 `logits[:, 0, :]`：当模型是 legacy IntentionLSTM
                # （输出 **2 维** `[batch, num_classes]`）时，这会切出
                # `[batch, 第0个logit]`——把"horizon 0 的第一个 logit"当成整个类别分布，
                # 即**量错了对象**（测的是模型内部标量，不是类别概率）。
                if logits.dim() == 3:
                    logits0 = logits[:, 0, :]
                else:
                    logits0 = logits
                probs = torch.softmax(logits0, dim=-1).detach().cpu().numpy()

            class_means = probs.mean(axis=0).tolist()
            result["intent_output_distribution"] = {f"class_{i}": float(v) for i, v in enumerate(class_means)}
            result["intent_num_classes"] = int(probs.shape[1])
            # Horizon-0：class-0（通常为"原地/不动"）对 classes 1..n-1（移动方向）
            #
            # 原实现把除数写死为 7.0。本仓实际模型是 **8 类**（fc_intent.weight 形状
            # (8,512)），故按**实际类别数**归一，否则偏差量纲随类别数漂移。
            _n = int(probs.shape[1])
            move_bias_estimate = (
                float(np.sum(probs[:, 1:])) / float(max(1, _n - 1)) - float(np.mean(probs[:, 0]))
            )
            result["move_bias_estimate"] = move_bias_estimate

            if move_bias_estimate < -0.1:
                result["warnings"].append("intent_negative_move_bias")
            if np.max(class_means) > 0.9:
                result["warnings"].append("intent_overconfident_class")
        except Exception as exc:
            result["warnings"].append(f"intent_diagnosis_error: {exc}")

    # Tactical model diagnostics
    if tactical_model is not None:
        result["tactical_available"] = True
        try:
            feat_dim = int(tactical_payload.get("input_dim", 32)) if tactical_payload else 32
            x = _safe_to_device(torch.randn(16, feat_dim), device)
            with torch.no_grad():
                out = tactical_model(x)
                sigmoid_out = torch.sigmoid(out).detach().cpu().numpy()
            result["tactical_output_mean"] = float(np.mean(sigmoid_out))
            result["tactical_output_std"] = float(np.std(sigmoid_out))
        except Exception as exc:
            result["warnings"].append(f"tactical_diagnosis_error: {exc}")
    else:
        result["tactical_available"] = False
        result["warnings"].append("tactical_model_missing")

    # Normalizer check
    if normalizer is not None:
        result["normalizer_loaded"] = True
        result["normalizer_keys"] = list(normalizer.keys())
        for key in ("feature_mean", "mean"):
            if key in normalizer:
                result["normalizer_feature_count"] = len(normalizer[key])
                break
        result["normalizer_zero_std_count"] = int(normalizer.get("zero_std_count", 0))
        expected_dim = int(situation_payload.get("input_dim", 108)) if situation_payload else 108
        result["normalizer_loaded_in_inference"] = result.get("normalizer_feature_count") == expected_dim
        if not result["normalizer_loaded_in_inference"]:
            result["warnings"].append("normalizer_feature_dim_mismatch")
    else:
        result["normalizer_loaded_in_inference"] = False
        result["warnings"].append("normalizer_missing")
    if not result.get("normalizer_applied_in_training", False):
        result["warnings"].append("normalizer_not_applied_in_training")

    # Datasets availability
    if datasets_dir is not None and datasets_dir.exists():
        result["datasets_available"] = True
        # Check for training feature stats from situation graphs
        state_graph_dir = None
        for candidate_name in ["WG-StateGraph", "分队级陆战兵棋态势表征图数据集 WG-StateGraph"]:
            for parent in datasets_dir.iterdir():
                if parent.is_dir() and candidate_name in parent.name:
                    state_graph_dir = parent
                    break
            if state_graph_dir:
                break

        if state_graph_dir:
            encoded_dir = state_graph_dir / "encoded_nodes-features"
            if encoded_dir.exists():
                csv_files = list(encoded_dir.glob("*.csv"))
                if csv_files:
                    result["datasets_state_graph_csv_count"] = len(csv_files)
                    # Sample first file to get feature stats
                    try:
                        sample = _load_state_graph_csv(csv_files[0])
                        if sample is not None:
                            result["train_feature_stats_sample_shape"] = list(sample.shape)
                            result["train_feature_stats_available"] = True
                            feat_mean = sample.mean(axis=0).tolist()
                            feat_std = sample.std(axis=0).tolist()
                            result["train_feature_mean_abs_max"] = float(max(abs(v) for v in feat_mean))
                            result["train_feature_std_max"] = float(max(feat_std))
                            result["train_feature_std_min"] = float(min(feat_std))
                    except Exception:
                        pass

        intent_dir = None
        for candidate_name in ["WargameData_mini01", "意图识别研究数据集WargameData_mini01"]:
            for parent in datasets_dir.iterdir():
                if parent.is_dir() and candidate_name in parent.name:
                    intent_dir = parent
                    break
            if intent_dir:
                break

        if intent_dir:
            result["datasets_intent_available"] = True
            train_json = intent_dir / "train.json"
            if train_json.exists():
                result["datasets_intent_train_size"] = train_json.stat().st_size

        tactical_csv = None
        for candidate_name in ["陆战兵棋态势特征数据集"]:
            for parent in datasets_dir.iterdir():
                if parent.is_dir() and candidate_name in parent.name:
                    tactical_csv = parent / "Wargame_Feature_Dataset.csv"
                    break
            if tactical_csv:
                break

        if tactical_csv and tactical_csv.exists():
            result["datasets_tactical_csv_size"] = tactical_csv.stat().st_size

    # Generate recommendations
    recs = result["recommended_next_actions"]
    if not result["situation_checkpoint_loaded"]:
        recs.append("train_situation_model")
    if not result["intent_checkpoint_loaded"]:
        recs.append("train_intent_model")
    if not result["tactical_checkpoint_loaded"] and not result.get("tactical_available"):
        recs.append("train_tactical_model")
    if not result["normalizer_loaded"]:
        recs.append("save_feature_normalizer")
    if not result.get("normalizer_applied_in_training", False):
        recs.append("apply_normalizer_during_training")
    if result["warnings"]:
        if "situation_per_node_zero_variance" in result["warnings"]:
            recs.append("retrain_situation_with_normalized_features")
        if "intent_negative_move_bias" in result["warnings"]:
            recs.append("address_intent_move_bias")
        if "normalizer_missing" in result["warnings"]:
            recs.append("compute_and_save_feature_normalizer")
        if "tactical_model_missing" in result["warnings"]:
            recs.append("enable_and_train_tactical_model")

    return result


def _load_state_graph_csv(csv_path: Path) -> Optional[np.ndarray]:
    """Parse first row of state graph CSV into features matrix."""
    import ast
    import pandas as pd
    try:
        frame = pd.read_csv(csv_path, nrows=1)
    except Exception:
        return None
    parsed_columns = []
    for column in frame.columns:
        value = frame.iloc[0][column]
        if isinstance(value, str) and value.startswith("["):
            try:
                parsed_columns.append([float(x) for x in ast.literal_eval(value) if isinstance(x, (int, float))])
            except Exception:
                parsed_columns.append([])
        else:
            try:
                parsed_columns.append([float(value)])
            except Exception:
                parsed_columns.append([])
    num_nodes = max((len(v) for v in parsed_columns), default=1)
    features = np.zeros((num_nodes, 108), dtype=np.float32)
    for feat_idx, values in enumerate(parsed_columns[:108]):
        if not values:
            continue
        fill_values = values[:num_nodes] + [0.0] * max(0, num_nodes - len(values))
        features[:, feat_idx] = np.array(fill_values[:num_nodes], dtype=np.float32)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose learned model quality")
    parser.add_argument("--artifacts", default="ai/weights", help="Path to model artifacts directory")
    parser.add_argument("--datasets", default=None, help="Path to datasets directory")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    artifact_dir = (Path(args.artifacts) if Path(args.artifacts).is_absolute()
                    else _repo_root() / args.artifacts)
    datasets_dir = (Path(args.datasets) if args.datasets and Path(args.datasets).is_absolute()
                    else _repo_root() / args.datasets) if args.datasets else None

    result = diagnose(artifact_dir, datasets_dir)

    if args.output:
        output_path = Path(args.output) if Path(args.output).is_absolute() else _repo_root() / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Diagnosis written to {output_path}")

    # Print summary to stdout
    print(f"\n=== Model Quality Diagnosis ===")
    print(f"Git HEAD: {result['git_head']}")
    print(f"Manifest loaded: {result['manifest_loaded']}")
    print(f"Situation model: {'LOADED' if result['situation_checkpoint_loaded'] else 'MISSING'}")
    print(f"Intent model: {'LOADED' if result['intent_checkpoint_loaded'] else 'MISSING'}")
    print(f"Tactical model: {'LOADED' if result['tactical_checkpoint_loaded'] else 'MISSING'}")
    print(f"Normalizer: {'LOADED' if result['normalizer_loaded'] else 'MISSING'}")

    if "situation_pre_sigmoid_mean" in result:
        print(f"\nSituation model stats:")
        print(f"  pre-sigmoid mean: {result['situation_pre_sigmoid_mean']:.6f}")
        print(f"  pre-sigmoid std: {result['situation_pre_sigmoid_std']:.6f}")
        print(f"  output mean: {result['situation_output_mean']:.6f}")
        print(f"  per-node variance mean: {result.get('situation_per_node_variance_mean', 'N/A')}")

    if "move_bias_estimate" in result:
        print(f"\nIntent model stats:")
        print(f"  move bias estimate: {result['move_bias_estimate']:.4f}")
        if "intent_output_distribution" in result:
            print(f"  class distribution: {result['intent_output_distribution']}")

    if result["warnings"]:
        print(f"\nWarnings ({len(result['warnings'])}):")
        for w in result["warnings"]:
            print(f"  - {w}")

    if result["recommended_next_actions"]:
        print(f"\nRecommended actions:")
        for a in result["recommended_next_actions"]:
            print(f"  - {a}")

    if result["warnings"]:
        print(f"\nWARNING: {len(result['warnings'])} issue(s) detected.")


if __name__ == "__main__":
    main()
