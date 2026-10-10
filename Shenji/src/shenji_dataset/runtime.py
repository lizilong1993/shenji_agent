from __future__ import annotations

import importlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

from .jobs import CollectionJob


def import_symbol(specification: str) -> type:
    """Import a class from a module:attribute specification."""

    if ":" not in specification:
        raise ValueError(f"import specification must use module:attribute: {specification}")
    module_name, attribute_name = specification.split(":", 1)
    module = importlib.import_module(module_name)
    value = getattr(module, attribute_name)
    if not isinstance(value, type):
        raise TypeError(f"imported symbol is not a class: {specification}")
    return value


def _prepend_sys_path(path: Path) -> None:
    text = str(path.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)


def resolve_miaosuan_env_class(project_root: Path | str) -> type:
    """Resolve the real MiaoSuan TrainEnv class without a mock fallback."""

    project = Path(project_root).resolve()
    _prepend_sys_path(project)
    errors: list[str] = []
    try:
        module = importlib.import_module("land_wargame_train_env")
        value = getattr(module, "TrainEnv")
        if not isinstance(value, type):
            raise TypeError("land_wargame_train_env.TrainEnv is not a class")
        return value
    except (ImportError, AttributeError, TypeError) as exc:
        errors.append(f"land_wargame_train_env: {exc}")

    sdk_dir = project / "land_wargame_sdk"
    _prepend_sys_path(sdk_dir)
    try:
        module = importlib.import_module("train_env")
        value = getattr(module, "TrainEnv")
        if not isinstance(value, type):
            raise TypeError("train_env.TrainEnv is not a class")
        return value
    except (ImportError, AttributeError, TypeError) as exc:
        errors.append(f"train_env: {exc}")
    detail = " | ".join(errors)
    raise RuntimeError(
        "MiaoSuan TrainEnv is unavailable. Use Linux, WSL, or a Linux container with the supplied manylinux wheel. "
        + detail
    )


def resolve_agent_class(specification: str, project_root: Path | str) -> type:
    _prepend_sys_path(Path(project_root))
    return import_symbol(specification)


def resolve_miaosuan_runtime(
    project_root: Path | str, agent_specification: str = "ai.agent:Agent"
) -> tuple[type, type]:
    """Resolve the real environment and one agent class for simple callers."""

    return (
        resolve_miaosuan_env_class(project_root),
        resolve_agent_class(agent_specification, project_root),
    )


def load_miaosuan_assets(
    job: CollectionJob, data_root: Path | str
) -> dict[str, Any]:
    """Load one scenario and its map assets from the MiaoSuan SDK data tree."""

    root = Path(data_root).resolve()
    paths = {
        "scenario": root / job.scenario_path,
        "basic": root / job.basic_path,
        "cost": root / job.cost_path,
        "see": root / job.see_path,
    }
    missing = [f"{name}:{path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing MiaoSuan assets: " + ", ".join(missing))
    with paths["scenario"].open("r", encoding="utf-8") as stream:
        scenario_data = json.load(stream)
    with paths["basic"].open("r", encoding="utf-8") as stream:
        basic_data = json.load(stream)
    with paths["cost"].open("rb") as stream:
        cost_data = pickle.load(stream)
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("loading MiaoSuan visibility assets requires NumPy") from exc
    with np.load(paths["see"]) as archive:
        if job.see_key not in archive:
            raise KeyError(
                f"visibility archive {paths['see']} has no key {job.see_key!r}"
            )
        see_data = archive[job.see_key]
    return {
        "scenario_data": scenario_data,
        "basic_data": basic_data,
        "cost_data": cost_data,
        "see_data": see_data,
    }
