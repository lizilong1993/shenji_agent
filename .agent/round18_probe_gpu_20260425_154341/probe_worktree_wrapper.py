#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]
PROBE_PATH = REPO_ROOT / ".agent" / "round18_coverage_probe" / "probe.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL)
        .strip()
    )


def _git_dirty_paths() -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--short", "ai/policy.py", "ai/agent.py", "training/selfplay.py", ".agent/round18_coverage_probe/probe.py"],
        cwd=str(REPO_ROOT),
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return [line.rstrip() for line in output.splitlines() if line.strip()]


def _first_param_device(model: Any) -> str | None:
    if model is None:
        return None
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    try:
        first = next(parameters())
    except StopIteration:
        return None
    return str(getattr(first, "device", None))


def _policy_snapshot(label: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    from ai.policy import LearnedPolicy

    policy = LearnedPolicy()
    faction = 0 if label == "candidate" else 1
    policy.configure(
        faction=faction,
        artifact_dir=str(payload["artifacts"]),
        policy_profile=str(payload["profile"]),
    )
    diagnostics = policy.get_diagnostics()
    return {
        "artifacts": str(payload["artifacts"]),
        "profile": str(payload["profile"]),
        "policy_device": str(getattr(policy, "device", "unknown")),
        "requested_device": str(getattr(policy, "requested_device", "unknown")),
        "loaded_from": diagnostics.get("loaded_from"),
        "intent_model_loaded": bool(diagnostics.get("intent_model_loaded")),
        "situation_model_loaded": bool(diagnostics.get("situation_model_loaded")),
        "intent_param_device": _first_param_device(policy.intent_model),
        "situation_param_device": _first_param_device(policy.situation_model),
    }


def _load_probe_module() -> Any:
    spec = importlib.util.spec_from_file_location("round18_coverage_probe_worktree", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load probe module from {PROBE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_participant(label: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    from ai.agent import Agent

    return {
        "label": label,
        "artifacts": str(payload["artifacts"]),
        "profile": str(payload["profile"]),
        "user_name": "CurrentAI" if label == "candidate" else "IncumbentAI",
        "source_commit_sha": str(payload["source_commit_sha"]),
        "code_origin": str(payload["code_origin"]),
        "agent_class": Agent,
    }


def _preflight(participants_path: Path) -> Dict[str, Any]:
    import ai.agent as agent_module
    import ai.policy as policy_module

    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    from training.selfplay import _train_env_runtime

    participants = _read_json(participants_path)
    _, env_backend = _train_env_runtime()

    cuda_available = bool(torch is not None and torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_current_device = int(torch.cuda.current_device()) if cuda_available else None
    cuda_device_name = str(torch.cuda.get_device_name(cuda_current_device)) if cuda_available else None

    payload = {
        "event": "probe_preflight",
        "repo_root": str(REPO_ROOT),
        "run_dir": str(RUN_DIR),
        "probe_path": str(PROBE_PATH),
        "participants_path": str(participants_path.resolve()),
        "git_head": _git_head(),
        "git_dirty_paths": _git_dirty_paths(),
        "env_backend": env_backend,
        "requested_device_env": os.environ.get("SHENJI_POLICY_DEVICE", ""),
        "ai_policy_file": str(Path(policy_module.__file__).resolve()),
        "ai_agent_file": str(Path(agent_module.__file__).resolve()),
        "torch_available": torch is not None,
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": cuda_device_count,
        "torch_cuda_current_device": cuda_current_device,
        "torch_cuda_device_name": cuda_device_name,
        "participants": {
            "candidate": _policy_snapshot("candidate", participants["candidate"]),
            "incumbent": _policy_snapshot("incumbent", participants["incumbent"]),
        },
    }
    _write_json(RUN_DIR / "preflight.json", payload)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True), flush=True)

    if env_backend == "mock_train_env.MockTrainEnv":
        raise SystemExit("Real probe blocked: resolved mock_train_env.MockTrainEnv instead of a real TrainEnv backend.")

    requested = (os.environ.get("SHENJI_POLICY_DEVICE") or "").strip().lower()
    if requested.startswith("cuda"):
        if not cuda_available:
            raise SystemExit("GPU preflight failed: torch.cuda.is_available() is False under requested CUDA probe path.")
        for label, snapshot in payload["participants"].items():
            policy_device = str(snapshot["policy_device"])
            if not policy_device.startswith("cuda"):
                raise SystemExit(f"GPU preflight failed: {label} policy configured device={policy_device!r} under CUDA request.")
            for key in ("intent_param_device", "situation_param_device"):
                value = snapshot.get(key)
                if value is not None and not str(value).startswith("cuda"):
                    raise SystemExit(
                        f"GPU preflight failed: {label} {key}={value!r} under CUDA request."
                    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Round 18 coverage probe against current worktree ai modules.")
    parser.add_argument("--participants", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    participants_path = Path(args.participants).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    invocation = {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "output_dir": str(output_dir),
        "participants_path": str(participants_path),
        "requested_device_env": os.environ.get("SHENJI_POLICY_DEVICE", ""),
    }
    _write_json(RUN_DIR / "wrapper_invocation.json", invocation)

    _preflight(participants_path)
    if args.preflight_only:
        return 0

    probe = _load_probe_module()
    probe.build_participant = _build_participant
    sys.argv = [
        str(PROBE_PATH),
        "--participants",
        str(participants_path),
        "--output-dir",
        str(output_dir),
        "--max-steps",
        str(args.max_steps),
    ]
    probe.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
