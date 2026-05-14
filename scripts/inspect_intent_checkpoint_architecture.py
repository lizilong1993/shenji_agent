#!/usr/bin/env python3
"""Inspect ai/weights/intention_lstm.pth and compare with current model architecture."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ai.models import MultiHorizonIntentionModel, IntentionLSTM


def _classify_keys(state_dict):
    """Classify state_dict keys as old fc_intent, current head, or unknown."""
    fc_intent_keys = [k for k in state_dict if "fc_intent" in k]
    head_keys = [k for k in state_dict if k.startswith("head.")]
    lstm_keys = [k for k in state_dict if "lstm" in k]
    attention_keys = [k for k in state_dict if "attention" in k]
    other_keys = [k for k in state_dict if k not in fc_intent_keys + head_keys + lstm_keys + attention_keys]

    # Detect which architecture based on distinguishing keys
    has_fc_intent = len(fc_intent_keys) > 0
    has_head = len(head_keys) > 0

    if has_fc_intent and not has_head:
        arch = "old_fc_intent"
    elif has_head and not has_fc_intent:
        arch = "current_head"
    elif has_head and has_fc_intent:
        arch = "ambiguous_both"
    else:
        arch = "unknown_neither"

    return {
        "total_keys": len(state_dict),
        "fc_intent_keys": sorted(fc_intent_keys),
        "head_keys": sorted(head_keys),
        "lstm_keys": sorted(lstm_keys),
        "attention_keys": sorted(attention_keys),
        "other_keys": sorted(other_keys),
        "classified_architecture": arch,
    }


def _compare_shapes(checkpoint_state, model, model_name="MultiHorizonIntentionModel"):
    """Compare checkpoint key shapes with model expected shapes."""
    model_state = model.state_dict()
    checkpoint_keys = set(checkpoint_state.keys())
    model_keys = set(model_state.keys())

    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)

    shape_mismatches = []
    for key in model_keys & checkpoint_keys:
        ckpt_shape = tuple(checkpoint_state[key].shape)
        model_shape = tuple(model_state[key].shape)
        if ckpt_shape != model_shape:
            shape_mismatches.append({
                "key": key,
                "checkpoint_shape": ckpt_shape,
                "model_shape": model_shape,
            })

    # Check if renaming fc_intent -> head could work
    rename_possible = False
    rename_issues = []
    if any("fc_intent" in k for k in checkpoint_keys):
        renamed = {}
        for k, v in checkpoint_state.items():
            new_k = k.replace("fc_intent", "head.3")  # head.3 is the final Linear
            renamed[new_k] = v
        # Check if renamed keys match model
        renamed_possible_keys = {k for k in renamed if k in model_keys}
        if "head.3.weight" in renamed_possible_keys and "head.3.bias" in renamed_possible_keys:
            # Check shape compatibility
            ckpt_shape = tuple(renamed["head.3.weight"].shape)
            model_shape = tuple(model_state["head.3.weight"].shape)
            if ckpt_shape == model_shape:
                rename_possible = True
            else:
                rename_issues.append(f"head.3 weight shape {ckpt_shape} != expected {model_shape}")
        else:
            rename_issues.append("Renamed fc_intent keys not in model (head.3)")

    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "simple_rename_possible": rename_possible,
        "rename_issues": rename_issues,
    }


def inspect(checkpoint_path: Path) -> dict:
    result = {"checkpoint_path": str(checkpoint_path)}

    # 1. Load checkpoint
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as e:
        result["readable"] = False
        result["error"] = f"Failed to load checkpoint: {e}"
        return result

    result["readable"] = True
    result["checkpoint_top_level_type"] = type(payload).__name__
    result["is_dict"] = isinstance(payload, dict)

    if isinstance(payload, dict):
        state_key_candidates = ["model_state", "state_dict", "model"]
        state = None
        for sk in state_key_candidates:
            if sk in payload and isinstance(payload[sk], dict):
                state = payload[sk]
                result["state_key_used"] = sk
                break
        if state is None:
            # Check if the payload itself is a state_dict (all keys are strings containing .)
            if all(isinstance(k, str) and "." in k for k in payload):
                state = payload
                result["state_key_used"] = "payload_is_state_dict"
            else:
                # Last resort: look for any dict value
                for k, v in payload.items():
                    if isinstance(v, dict) and len(v) > 5:
                        state = v
                        result["state_key_used"] = f"auto:{k}"
                        break
        if state is None:
            result["state_dict_found"] = False
            result["error"] = "Cannot locate state_dict in checkpoint"
            return result

        result["state_dict_found"] = True
    else:
        result["state_dict_found"] = False
        result["error"] = f"Checkpoint is not a dict (type={type(payload).__name__})"
        return result

    # 2. Classify keys
    classification = _classify_keys(state)
    result.update({f"key_{k}": v for k, v in classification.items()})

    # 3. Attempt to instantiate current model and compare
    result["current_model_instantiable"] = False
    result["current_model_comparison"] = None
    result["old_model_instantiable"] = False
    result["old_model_comparison"] = None

    try:
        # Infer dimensions from checkpoint
        lstm_weight_key = [k for k in state if "lstm.weight_ih_l0" in k]
        if lstm_weight_key:
            hidden_dim = state[lstm_weight_key[0]].shape[0] // 4
        elif "lstm.weight_ih_l0" in state:
            hidden_dim = state["lstm.weight_ih_l0"].shape[0] // 4
        else:
            hidden_dim = 256  # default guess

        current_model = MultiHorizonIntentionModel(
            input_dim=108, hidden_dim=hidden_dim, horizons=5, num_classes=7
        )
        result["current_model_instantiable"] = True
        result["current_model_hidden_dim"] = hidden_dim
        result["current_model_comparison"] = _compare_shapes(state, current_model, "MultiHorizonIntentionModel")
    except Exception as e:
        result["current_model_instantiable"] = False
        result["current_model_error"] = str(e)

    # Also try old IntentionLSTM for reference
    try:
        old_model = IntentionLSTM(input_dim=108, hidden_dim=256, num_classes=4)
        # Check how old model keys would look
        old_model_state = old_model.state_dict()
        result["old_model_expected_keys"] = sorted(old_model_state.keys())
        result["old_model_instantiable"] = True
        result["old_model_comparison"] = _compare_shapes(state, old_model, "IntentionLSTM")
    except Exception as e:
        result["old_model_instantiable"] = False
        result["old_model_error"] = str(e)

    # 4. Compatibility assessment
    if result.get("current_model_comparison"):
        cmp = result["current_model_comparison"]
        missing = cmp.get("missing_keys", [])
        unexpected = cmp.get("unexpected_keys", [])
        shape_mismatches = cmp.get("shape_mismatches", [])

        if not missing and not unexpected and not shape_mismatches:
            result["compatibility_assessment"] = "direct_load_possible"
        elif cmp.get("simple_rename_possible") and not shape_mismatches:
            result["compatibility_assessment"] = "simple_rename_possible"
        else:
            result["compatibility_assessment"] = "incompatible"
            result["compatibility_details"] = {
                "missing": missing,
                "unexpected": unexpected,
                "shape_mismatches": shape_mismatches,
            }
    else:
        result["compatibility_assessment"] = "unknown"

    # 5. Summary conclusions
    arch = classification["classified_architecture"]
    ca = result["compatibility_assessment"]

    if ca == "direct_load_possible":
        result["conclusions"] = {
            "is_readable": True,
            "old_fc_intent_style": arch == "old_fc_intent",
            "current_head_style": arch == "current_head",
            "can_load_directly": True,
            "can_map_by_key_renaming": True,
            "tensor_shapes_compatible": True,
            "legacy_class_needed": False,
            "retraining_required": False,
        }
    elif ca == "simple_rename_possible":
        result["conclusions"] = {
            "is_readable": True,
            "old_fc_intent_style": arch == "old_fc_intent",
            "current_head_style": arch == "current_head",
            "can_load_directly": False,
            "can_map_by_key_renaming": True,
            "tensor_shapes_compatible": len(result.get("current_model_comparison", {}).get("shape_mismatches", [])) == 0,
            "legacy_class_needed": False,
            "retraining_required": False,
        }
    elif ca == "incompatible":
        # Check if old model can load it
        old_ok = False
        if result.get("old_model_comparison"):
            old_cmp = result["old_model_comparison"]
            if not old_cmp.get("missing_keys") and not old_cmp.get("unexpected_keys") and not old_cmp.get("shape_mismatches"):
                old_ok = True

        if old_ok:
            legacy_needed = True
            retrain = False
        else:
            legacy_needed = False
            retrain = True

        result["conclusions"] = {
            "is_readable": True,
            "old_fc_intent_style": arch == "old_fc_intent",
            "current_head_style": arch == "current_head",
            "can_load_directly": False,
            "can_map_by_key_renaming": False,
            "tensor_shapes_compatible": False,
            "legacy_class_needed": legacy_needed,
            "retraining_required": retrain,
        }
    else:
        result["conclusions"] = {
            "is_readable": result.get("readable", False),
            "old_fc_intent_style": None,
            "current_head_style": None,
            "can_load_directly": False,
            "can_map_by_key_renaming": False,
            "tensor_shapes_compatible": False,
            "legacy_class_needed": True,
            "retraining_required": True,
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Inspect intent checkpoint architecture")
    parser.add_argument("--checkpoint", default="ai/weights/intention_lstm.pth", help="Path to checkpoint")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        print(json.dumps({"error": f"Checkpoint not found: {ckpt_path}", "readable": False}, indent=2))
        sys.exit(1)

    result = inspect(ckpt_path)

    output = json.dumps(result, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(output)
        print(f"Written to {args.out}")
    else:
        print(output)

    # Exit code: 0 if readable, 1 if not
    if result.get("readable"):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
