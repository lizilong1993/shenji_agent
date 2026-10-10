from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .jobs import CollectionJob
from .labeling import derive_grade_a_intent_plan, derive_grade_b_intent_plan, derive_grade_c_intent_plan
from .normalizer import normalize_episode
from .recorder import EpisodeRecorder
from .release import write_release


@dataclass(frozen=True)
class CollectionDependencies:
    env_factory: Callable[[], Any]
    red_agent_factory: Callable[[], Any]
    blue_agent_factory: Callable[[], Any]
    asset_loader: Callable[[CollectionJob], dict[str, Any]]


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _agent_setup(
    *,
    job: CollectionJob,
    side: str,
    state: dict[Any, Any],
    assets: dict[str, Any],
) -> dict[str, Any]:
    is_red = side == "red"
    return {
        "scenario": assets["scenario_data"],
        "basic_data": assets["basic_data"],
        "cost_data": assets["cost_data"],
        "see_data": assets["see_data"],
        "seat": 1 if is_red else 11,
        "faction": 0 if is_red else 1,
        "role": 1,
        "user_name": "ShenjiRed" if is_red else "ShenjiBlue",
        "user_id": 1 if is_red else 11,
        "policy_profile": job.red_policy_profile if is_red else job.blue_policy_profile,
        "policy_artifacts_dir": (
            job.red_artifacts_dir if is_red else job.blue_artifacts_dir
        ),
        "state": state,
    }


def _final_reward(state: dict[Any, Any] | None, side: int) -> float | int | None:
    if not isinstance(state, dict):
        return None
    observation = state.get(side)
    if not isinstance(observation, dict):
        return None
    reward = observation.get("reward")
    return reward if isinstance(reward, (int, float)) and not isinstance(reward, bool) else None


def _safe_reset(value: Any) -> None:
    if value is None:
        return
    reset = getattr(value, "reset", None)
    if callable(reset):
        reset()


def run_collection_job(
    job: CollectionJob,
    dependencies: CollectionDependencies,
    output_root: Path | str,
    canonical_root: Path | str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run or resume one collection job and persist all episode artifacts."""

    raw_root = Path(output_root)
    canonical_base = (
        Path(canonical_root) if canonical_root is not None else raw_root.parent / "canonical"
    )
    episode_dir = raw_root / job.episode_id
    release_dir = canonical_base / job.episode_id
    status_path = episode_dir / "status.json"
    job_path = episode_dir / "job.json"
    raw_path = episode_dir / "raw_episode.json"
    partial_path = episode_dir / "partial_raw_episode.json"
    intent_path = episode_dir / "intent_plan.json"
    episode_dir.mkdir(parents=True, exist_ok=True)

    if status_path.is_file():
        existing_status = _load_json(status_path)
        if resume and existing_status.get("state") == "completed":
            return {**existing_status, "skipped": True}
    if job_path.is_file():
        existing_job = _load_json(job_path)
        if existing_job != job.to_dict():
            raise ValueError(
                f"episode directory {episode_dir} belongs to a different collection job"
            )
    else:
        _atomic_write_json(job_path, job.to_dict())

    status: dict[str, Any] = {
        "dataset": "Shenji",
        "episode_id": job.episode_id,
        "state": "running",
        "step_count": 0,
        "max_steps": job.max_steps,
        "skipped": False,
    }
    _atomic_write_json(status_path, status)
    env = None
    red_agent = None
    blue_agent = None
    recorder: EpisodeRecorder | None = None
    state: dict[Any, Any] | None = None
    done = False
    step_count = 0
    try:
        if raw_path.is_file():
            raw_episode = _load_json(raw_path)
            step_count = max(
                0, len(raw_episode.get("observations", {}).get("truth", [])) - 1
            )
            done = bool(raw_episode.get("metadata", {}).get("environment_done", False))
        else:
            _seed_everything(job.seed)
            assets = dependencies.asset_loader(job)
            env = dependencies.env_factory()
            red_agent = dependencies.red_agent_factory()
            blue_agent = dependencies.blue_agent_factory()
            player_info = [
                {
                    "seat": 1,
                    "faction": 0,
                    "role": 1,
                    "user_name": "ShenjiRed",
                    "user_id": 1,
                },
                {
                    "seat": 11,
                    "faction": 1,
                    "role": 1,
                    "user_name": "ShenjiBlue",
                    "user_id": 11,
                },
            ]
            state = env.setup({**assets, "player_info": player_info})
            recorder = EpisodeRecorder(
                {
                    **job.to_metadata(),
                    "max_steps": job.max_steps,
                    "collection_protocol": "shenji_pilot_v1",
                }
            )
            recorder.record(state, actions=[])
            red_agent.setup(
                _agent_setup(job=job, side="red", state=state, assets=assets)
            )
            blue_agent.setup(
                _agent_setup(job=job, side="blue", state=state, assets=assets)
            )
            # Apply pre-set missions (intent-conditioned controllers)
            initial_mission_actions: list[dict[str, Any]] = []
            if job.red_pre_mission:
                red_agent.set_mission(
                    job.red_pre_mission.get("mission_type", 0),
                    job.red_pre_mission.get("target_pos"),
                )
                initial_mission_actions.append({
                    "actor": 1,
                    "type": 200,
                    "info": {
                        "mission_type": job.red_pre_mission["mission_type"],
                        "company_id": 1,
                        "start_time": 0,
                        "end_time": job.max_steps,
                    },
                    "cur_step": 0,
                })
            if job.blue_pre_mission:
                blue_agent.set_mission(
                    job.blue_pre_mission.get("mission_type", 0),
                    job.blue_pre_mission.get("target_pos"),
                )
            if initial_mission_actions:
                # Re-record step 0 with mission actions included
                recorder._actions = initial_mission_actions
            while not done and step_count < job.max_steps:
                red_actions = red_agent.step(state[0]) or []
                blue_actions = blue_agent.step(state[1]) or []
                if not isinstance(red_actions, list) or not isinstance(blue_actions, list):
                    raise TypeError("agent step methods must return action lists")
                submitted_actions = [*red_actions, *blue_actions]
                state, done = env.step(submitted_actions)
                recorder.record(state, submitted_actions)
                step_count += 1
                if step_count % 100 == 0:
                    _atomic_write_json(
                        status_path,
                        {**status, "state": "running", "step_count": step_count},
                    )
            raw_episode = recorder.to_raw_episode()
            raw_episode["metadata"]["environment_done"] = bool(done)
            raw_episode["metadata"]["termination_reason"] = (
                "environment_done" if done else "max_steps"
            )
            raw_episode["metadata"]["step_count"] = step_count
            raw_episode["metadata"]["red_final_reward"] = _final_reward(state, 0)
            raw_episode["metadata"]["blue_final_reward"] = _final_reward(state, 1)
            temporary_raw = raw_path.with_name(".raw_episode.json.tmp")
            temporary_raw.write_text(
                json.dumps(
                    raw_episode, ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_raw, raw_path)
            if partial_path.exists():
                partial_path.unlink()

        intent_plan = derive_grade_a_intent_plan(raw_episode)
        grade_b_plan = derive_grade_b_intent_plan(raw_episode)
        grade_c_plan = derive_grade_c_intent_plan(raw_episode)
        # Merge all grade segments
        seen_ids: set[str] = set()
        merged_segments: list[dict[str, Any]] = []
        for plan in (intent_plan, grade_b_plan, grade_c_plan):
            for seg in plan.get("segments", []):
                seg_id = seg.get("segment_id", "")
                if seg_id not in seen_ids:
                    seen_ids.add(seg_id)
                    merged_segments.append(seg)
        merged_segments.sort(key=lambda s: (s.get("start_step", 0), s.get("subject_id", ""), s.get("primary_intent", ""), s.get("segment_id", "")))
        intent_plan["segments"] = merged_segments
        intent_plan["generator"] = "shenji_all_grade_labeler_v1"
        _atomic_write_json(intent_path, intent_plan)
        tables = normalize_episode(raw_episode, intent_plan)
        manifest = write_release(tables, release_dir, formats=("jsonl",))
        status = {
            "dataset": "Shenji",
            "episode_id": job.episode_id,
            "state": "completed",
            "step_count": step_count,
            "max_steps": job.max_steps,
            "environment_done": bool(done),
            "termination_reason": "environment_done" if done else "max_steps",
            "red_final_reward": raw_episode["metadata"].get("red_final_reward"),
            "blue_final_reward": raw_episode["metadata"].get("blue_final_reward"),
            "intent_segment_count": len(intent_plan["segments"]),
            "artifact_count": len(manifest["artifacts"]),
            "raw_episode": str(raw_path),
            "canonical_release": str(release_dir),
            "skipped": False,
        }
        _atomic_write_json(status_path, status)
        return status
    except Exception as exc:
        if recorder is not None and not raw_path.exists():
            _atomic_write_json(partial_path, recorder.to_raw_episode())
        status = {
            "dataset": "Shenji",
            "episode_id": job.episode_id,
            "state": "failed",
            "step_count": step_count,
            "max_steps": job.max_steps,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "skipped": False,
        }
        _atomic_write_json(status_path, status)
        return status
    finally:
        for value in (red_agent, blue_agent, env):
            try:
                _safe_reset(value)
            except Exception:
                pass
