from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _slug(value: Any) -> str:
    slug = _SLUG_PATTERN.sub("_", str(value).lower()).strip("_")
    return slug or "unknown"


@dataclass(frozen=True)
class CollectionJob:
    episode_id: str
    scenario_id: str
    map_id: str
    scenario_path: str
    basic_path: str
    cost_path: str
    see_path: str
    see_key: str
    seed: int
    red_policy_code: str
    red_policy_profile: str
    red_agent_class: str
    red_artifacts_dir: str | None
    blue_policy_code: str
    blue_policy_profile: str
    blue_agent_class: str
    blue_artifacts_dir: str | None
    max_steps: int
    red_pre_mission: dict[str, Any] | None = None
    blue_pre_mission: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "dataset": "Shenji",
            "schema_version": "shenji_schema_v1",
            "episode_id": self.episode_id,
            "scenario_id": self.scenario_id,
            "map_id": self.map_id,
            "seed": self.seed,
            "simulation_backend": "MiaoSuan",
            "policy_red": self.red_policy_code,
            "policy_blue": self.blue_policy_code,
            "red_policy_profile": self.red_policy_profile,
            "blue_policy_profile": self.blue_policy_profile,
        }


def load_collection_manifest(path: Path | str) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("collection manifest root must be an object")
    return manifest


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("dataset") != "Shenji":
        raise ValueError("collection manifest dataset must be Shenji")
    if manifest.get("simulation_backend") != "MiaoSuan":
        raise ValueError("collection manifest simulation_backend must be MiaoSuan")
    max_steps = manifest.get("max_steps")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
        raise ValueError("collection manifest max_steps must be a positive integer")

    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("collection manifest scenarios must be a non-empty list")
    scenario_required = {
        "scenario_id",
        "map_id",
        "scenario_path",
        "basic_path",
        "cost_path",
        "see_path",
        "see_key",
    }
    scenario_ids: list[str] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("each scenario must be an object")
        missing = sorted(scenario_required - set(scenario))
        if missing:
            raise ValueError(f"scenario is missing required fields: {missing}")
        scenario_ids.append(str(scenario["scenario_id"]))
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("scenario_id values must be unique")

    policies = manifest.get("policies")
    if not isinstance(policies, list) or not policies:
        raise ValueError("collection manifest policies must be a non-empty list")
    policy_required = {"code", "profile", "agent_class"}
    policy_codes: list[str] = []
    for policy in policies:
        if not isinstance(policy, dict):
            raise ValueError("each policy must be an object")
        missing = sorted(policy_required - set(policy))
        if missing:
            raise ValueError(f"policy is missing required fields: {missing}")
        policy_codes.append(str(policy["code"]))
    if len(policy_codes) != len(set(policy_codes)):
        raise ValueError("policy codes must be unique")
    policy_code_set = set(policy_codes)

    matchups = manifest.get("matchups")
    if not isinstance(matchups, list) or not matchups:
        raise ValueError("collection manifest matchups must be a non-empty list")
    for matchup in matchups:
        if not isinstance(matchup, dict) or "red" not in matchup or "blue" not in matchup:
            raise ValueError("each matchup must define red and blue policy codes")
        if str(matchup["red"]) not in policy_code_set:
            raise ValueError(f"matchup references unknown red policy: {matchup['red']}")
        if str(matchup["blue"]) not in policy_code_set:
            raise ValueError(f"matchup references unknown blue policy: {matchup['blue']}")

    seeds = manifest.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("collection manifest seeds must be a non-empty list")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("every seed must be an integer")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")


def _episode_id(
    scenario_id: str,
    map_id: str,
    seed: int,
    red_policy_code: str,
    blue_policy_code: str,
) -> str:
    identity = {
        "scenario_id": scenario_id,
        "map_id": map_id,
        "seed": seed,
        "red_policy": red_policy_code,
        "blue_policy": blue_policy_code,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
    return (
        f"shenji_{_slug(scenario_id)}_{seed}_"
        f"{_slug(red_policy_code)}_vs_{_slug(blue_policy_code)}_{digest}"
    )


def expand_jobs(manifest: dict[str, Any]) -> list[CollectionJob]:
    """Expand a collection manifest into a stable, deterministic job list."""

    _validate_manifest(manifest)
    policies = {str(item["code"]): item for item in manifest["policies"]}
    jobs: list[CollectionJob] = []
    for scenario in manifest["scenarios"]:
        scenario_id = str(scenario["scenario_id"])
        map_id = str(scenario["map_id"])
        for seed in manifest["seeds"]:
            for matchup in manifest["matchups"]:
                red_code = str(matchup["red"])
                blue_code = str(matchup["blue"])
                red = policies[red_code]
                blue = policies[blue_code]
                jobs.append(
                    CollectionJob(
                        episode_id=_episode_id(
                            scenario_id, map_id, seed, red_code, blue_code
                        ),
                        scenario_id=scenario_id,
                        map_id=map_id,
                        scenario_path=str(scenario["scenario_path"]),
                        basic_path=str(scenario["basic_path"]),
                        cost_path=str(scenario["cost_path"]),
                        see_path=str(scenario["see_path"]),
                        see_key=str(scenario["see_key"]),
                        seed=seed,
                        red_policy_code=red_code,
                        red_policy_profile=str(red["profile"]),
                        red_agent_class=str(red["agent_class"]),
                        red_artifacts_dir=(
                            str(red["artifacts_dir"])
                            if red.get("artifacts_dir") is not None
                            else None
                        ),
                        red_pre_mission=red.get("pre_mission"),
                        blue_policy_code=blue_code,
                        blue_policy_profile=str(blue["profile"]),
                        blue_agent_class=str(blue["agent_class"]),
                        blue_artifacts_dir=(
                            str(blue["artifacts_dir"])
                            if blue.get("artifacts_dir") is not None
                            else None
                        ),
                        blue_pre_mission=blue.get("pre_mission"),
                        max_steps=int(manifest["max_steps"]),
                    )
                )
    jobs.sort(key=lambda job: job.episode_id)
    return jobs
