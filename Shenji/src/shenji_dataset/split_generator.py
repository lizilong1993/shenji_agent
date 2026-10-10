"""Split generator for Shenji dataset v2.

Generates 5 split tracks (IID, cross_map, cross_policy,
cross_scenario_family, visibility_stratified) with zero-leakage
verification.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ZERO_LEAKAGE_HASH = ""


def discover_episodes(canonical_root: str) -> list[dict[str, Any]]:
    """Scan canonical directory for all episodes and their metadata."""
    episodes = []
    for ep_dir in os.listdir(canonical_root):
        ep_path = Path(canonical_root) / ep_dir
        if not ep_path.is_dir() or not ep_dir.startswith("shenji_"):
            continue
        manifest_path = ep_path / "manifest.json"
        ep_jsonl = ep_path / "episodes.jsonl"
        if not manifest_path.is_file() or not ep_jsonl.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with open(ep_jsonl) as f:
            for line in f:
                if line.strip():
                    ep_meta = json.loads(line)
        episodes.append({
            "episode_id": ep_dir,
            "scenario_id": ep_meta.get("scenario_id", "?"),
            "map_id": ep_meta.get("map_id", "?"),
            "policy_red": ep_meta.get("policy_red", "?"),
            "policy_blue": ep_meta.get("policy_blue", "?"),
            "seed": ep_meta.get("seed", 0),
            "simulation_backend": ep_meta.get("simulation_backend", "MiaoSuan"),
            "manifest": manifest,
        })
    return episodes


def _scenario_family(scenario_id: str) -> str:
    """Extract scenario family from scenario_id."""
    return scenario_id.rstrip("0123456789") or scenario_id


def _policy_family(policy_code: str) -> str:
    """Map individual policy code to implementation family."""
    if "intent_" in policy_code or "challenger" in policy_code:
        return "learned_policy"
    if "baseline" in policy_code or "rule" in policy_code:
        return "rule_based"
    return policy_code


def generate_splits(
    episodes: list[dict[str, Any]],
    output_path: str | None = None,
    seed: int = 42,
    held_out_map_families: int = 2,
    held_out_policy_families: int = 1,
    held_out_scenario_families: int = 1,
) -> dict[str, dict[str, list[str]]]:
    """Generate 5 split tracks with zero-leakage guarantees."""
    rng = random.Random(seed)

    # Group episodes
    map_groups: dict[str, list[str]] = defaultdict(list)
    policy_groups: dict[str, list[str]] = defaultdict(list)
    scenario_groups: dict[str, list[str]] = defaultdict(list)
    all_ids = []

    for ep in episodes:
        eid = ep["episode_id"]
        all_ids.append(eid)
        map_groups[ep["map_id"]].append(eid)
        pfam = _policy_family(ep["policy_red"])
        policy_groups[pfam].append(eid)
        sfam = _scenario_family(ep["scenario_id"])
        scenario_groups[sfam].append(eid)

    rng.shuffle(all_ids)

    # IID split
    n = len(all_ids)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)
    splits: dict[str, dict[str, list[str]]] = {
        "iid": {
            "train": sorted(all_ids[:train_end]),
            "val": sorted(all_ids[train_end:val_end]),
            "test": sorted(all_ids[val_end:]),
        },
    }

    # Cross-map: hold out map families
    map_families = sorted(set(ep["map_id"] for ep in episodes))
    rng.shuffle(map_families)
    held_out_maps = set(map_families[:held_out_map_families])
    train_ids = [ep["episode_id"] for ep in episodes if ep["map_id"] not in held_out_maps]
    test_ids = [ep["episode_id"] for ep in episodes if ep["map_id"] in held_out_maps]
    splits["cross_map"] = {
        "train": sorted(train_ids),
        "test": sorted(test_ids),
        "held_out_maps": sorted(held_out_maps),
    }

    # Cross-policy: hold out policy implementation families
    policy_families = sorted(policy_groups.keys())
    rng.shuffle(policy_families)
    held_out_pfams = set(policy_families[:held_out_policy_families])
    train_ids = [ep["episode_id"] for ep in episodes if _policy_family(ep["policy_red"]) not in held_out_pfams]
    test_ids = [ep["episode_id"] for ep in episodes if _policy_family(ep["policy_red"]) in held_out_pfams]
    splits["cross_policy"] = {
        "train": sorted(train_ids),
        "test": sorted(test_ids),
        "held_out_families": sorted(held_out_pfams),
    }

    # Cross-scenario-family
    scenario_families = sorted(scenario_groups.keys())
    rng.shuffle(scenario_families)
    held_out_sfams = set(scenario_families[:held_out_scenario_families])
    train_ids = [ep["episode_id"] for ep in episodes if _scenario_family(ep["scenario_id"]) not in held_out_sfams]
    test_ids = [ep["episode_id"] for ep in episodes if _scenario_family(ep["scenario_id"]) in held_out_sfams]
    splits["cross_scenario_family"] = {
        "train": sorted(train_ids),
        "test": sorted(test_ids),
        "held_out_families": sorted(held_out_sfams),
    }

    # Visibility-stratified (placeholder - uses IID split with bin annotation)
    splits["visibility_stratified"] = {
        "train": splits["iid"]["train"],
        "val": splits["iid"]["val"],
        "test": splits["iid"]["test"],
        "note": "Visibility stratification bins to be computed from observation visibility metadata",
    }

    # Zero-leakage verification
    verification = verify_zero_leakage(splits, episodes)

    result = {
        "dataset": "Shenji",
        "version": "2.0.0",
        "generated_at": __import__("time").strftime("%Y%m%d_%H%M%S"),
        "seed": seed,
        "episode_count": len(all_ids),
        "splits": splits,
        "zero_leakage_verification": verification,
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Splits written to {path}")
        print(f"Zero-leakage verification: {verification}")

    return splits


def verify_zero_leakage(
    splits: dict[str, dict[str, list[str]]],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify no episode/group appears in multiple splits."""
    results = {}
    for track_name, track_data in splits.items():
        issues = []
        for split_name, eids in track_data.items():
            if not isinstance(eids, list):
                continue
            for other_name, other_eids in track_data.items():
                if not isinstance(other_eids, list) or other_name <= split_name:
                    continue
                overlap = set(eids) & set(other_eids)
                if overlap:
                    issues.append(f"Overlap between {split_name} and {other_name}: {len(overlap)} episodes")
        results[track_name] = {"pass": len(issues) == 0, "issues": issues}
    all_pass = all(r.get("pass", True) for r in results.values())
    return {"all_pass": all_pass, "tracks": results}


if __name__ == "__main__":
    import sys
    canon_root = sys.argv[1] if len(sys.argv) > 1 else "/remote-home/lizilong/shenji_agent/Shenji/raw/canonical_v3"
    output = sys.argv[2] if len(sys.argv) > 2 else "/remote-home/lizilong/shenji_agent/Shenji/benchmarks/splits_v2.json"
    eps = discover_episodes(canon_root)
    print(f"Discovered {len(eps)} episodes")
    generate_splits(eps, output_path=output)
