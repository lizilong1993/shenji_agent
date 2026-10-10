#!/usr/bin/env python3
"""Run all pilot collection jobs sequentially on server-154."""
import sys, json, time
sys.path.insert(0, "/remote-home/lizilong/shenji_agent")
sys.path.insert(0, "/remote-home/lizilong/shenji_agent/Shenji/src")

from shenji_dataset.collection import run_collection_job, CollectionDependencies
from shenji_dataset.jobs import load_collection_manifest, expand_jobs
from shenji_dataset.runtime import resolve_miaosuan_env_class, resolve_agent_class, load_miaosuan_assets

manifest = load_collection_manifest("/remote-home/lizilong/shenji_agent/Shenji/configs/pilot_manifest_v1.json")
jobs = expand_jobs(manifest)
print(f"Expanded {len(jobs)} jobs")

env_class = resolve_miaosuan_env_class(project_root="/remote-home/lizilong/shenji_agent")
agent_class = resolve_agent_class("ai.agent:Agent", project_root="/remote-home/lizilong/shenji_agent")
data_root = "/remote-home/lizilong/shenji_agent/land_wargame_sdk/Data/Data"
output_root = "/remote-home/lizilong/shenji_agent/Shenji/raw/pilot_v1"

deps = CollectionDependencies(
    env_factory=lambda: env_class(),
    red_agent_factory=lambda: agent_class(),
    blue_agent_factory=lambda: agent_class(),
    asset_loader=lambda j: load_miaosuan_assets(j, data_root),
)

results = []
for i, job in enumerate(jobs):
    t0 = time.time()
    print(f"[{i+1}/{len(jobs)}] {job.episode_id} starting...", flush=True)
    result = run_collection_job(
        job=job, dependencies=deps, output_root=output_root, resume=True
    )
    elapsed = time.time() - t0
    result["_elapsed"] = round(elapsed, 1)
    status = result.get("state", "?")
    print(f"[{i+1}/{len(jobs)}] -> {status} ({elapsed:.1f}s)", flush=True)
    results.append(result)

summary = "/remote-home/lizilong/shenji_agent/Shenji/raw/pilot_v1/_pilot_summary.json"
with open(summary, "w") as f:
    json.dump({"results": results, "completed_at": time.strftime("%Y%m%d_%H%M%S")}, f, indent=2)
print(f"Summary saved to {summary}", flush=True)

ok = all(r.get("state") in ("completed", "skipped") for r in results)
print(f"ALL JOBS DONE. Overall: {'SUCCESS' if ok else 'PARTIAL FAILURE'}", flush=True)
sys.exit(0 if ok else 1)
