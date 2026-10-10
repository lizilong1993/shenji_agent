from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


_MISSING = object()


def _view(state_by_side: dict[Any, Any], numeric_key: int, name: str) -> dict[str, Any]:
    value = state_by_side.get(numeric_key, _MISSING)
    if value is _MISSING:
        value = state_by_side.get(name, _MISSING)
    if value is _MISSING:
        raise ValueError(f"state_by_side is missing required {name} observation")
    if not isinstance(value, dict):
        raise TypeError(f"{name} observation must be a mapping")
    return value


def _step_from_observation(observation: dict[str, Any], fallback: int) -> int:
    time_value = observation.get("time", {})
    if isinstance(time_value, dict) and "cur_step" in time_value:
        return int(time_value["cur_step"])
    if "cur_step" in observation:
        return int(observation["cur_step"])
    return fallback


class EpisodeRecorder:
    """Capture synchronized truth and partial MiaoSuan observation streams."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        metadata_copy = copy.deepcopy(metadata)
        if metadata_copy.get("dataset", "Shenji") != "Shenji":
            raise ValueError("dataset must be Shenji")
        required = {"episode_id", "scenario_id", "map_id"}
        missing = sorted(required - set(metadata_copy))
        if missing:
            raise ValueError(f"metadata is missing required fields: {missing}")
        metadata_copy["dataset"] = "Shenji"
        metadata_copy.setdefault("schema_version", "shenji_schema_v1")
        metadata_copy.setdefault("simulation_backend", "MiaoSuan")
        self._metadata = metadata_copy
        self._observations: dict[str, list[dict[str, Any]]] = {
            "truth": [],
            "red": [],
            "blue": [],
        }
        self._actions: list[dict[str, Any]] = []

    def record(
        self,
        state_by_side: dict[Any, Any],
        actions: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record one synchronized environment step and its submitted actions."""

        truth = _view(state_by_side, -1, "truth")
        red = _view(state_by_side, 0, "red")
        blue = _view(state_by_side, 1, "blue")
        step = _step_from_observation(truth, len(self._observations["truth"]))
        self._observations["truth"].append(copy.deepcopy(truth))
        self._observations["red"].append(copy.deepcopy(red))
        self._observations["blue"].append(copy.deepcopy(blue))
        for action in actions or []:
            if not isinstance(action, dict):
                raise TypeError("each action must be a mapping")
            action_copy = copy.deepcopy(action)
            action_copy.setdefault("cur_step", step)
            self._actions.append(action_copy)

    def to_raw_episode(self) -> dict[str, Any]:
        """Return an isolated JSON-serializable raw episode document."""

        return {
            "metadata": copy.deepcopy(self._metadata),
            "observations": copy.deepcopy(self._observations),
            "actions": copy.deepcopy(self._actions),
        }

    def save(self, path: Path | str) -> Path:
        """Write the raw episode deterministically as UTF-8 JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_raw_episode(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        output_path.write_text(payload + "\n", encoding="utf-8")
        return output_path
