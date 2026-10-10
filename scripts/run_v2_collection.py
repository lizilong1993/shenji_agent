import sys, json, os, time
sys.path.insert(0, "/remote-home/lizilong/shenji_agent")
sys.path.insert(0, "/remote-home/lizilong/shenji_agent/Shenji/src")

from shenji_dataset.collection import run_collection_job, CollectionDependencies
from shenji_dataset.jobs import load_collection_manifest, expand_jobs
from shenji_dataset.runtime import resolve_miaosuan_env_class, resolve_agent_class, load_miaosuan_assets

manifest = load_collection_manifest("/remote-home/lizilong/shenji_agent/Shenji/configs/pilot_v2_manifest.json")
jobs = expand_jobs(manifest)
print("Expanded {} jobs".format(len(jobs)))

env_class = resolve_miaosuan_env_class(project_root="/remote-home/lizilong/shenji_agent")
agent_class = resolve_agent_class("ai.agent:Agent", project_root="/remote-home/lizilong/shenji_agent")
data_root = "/remote-home/lizilong/shenji_agent/land_wargame_sdk/Data/Data"
output_root = "/remote-home/lizilong/shenji_agent/Shenji/raw/pilot_v2"

deps = CollectionDependencies(
    env_factory=lambda: env_class(),
    red_agent_factory=lambda: agent_class(),
    blue_agent_factory=lambda: agent_class(),
    asset_loader=lambda j: load_miaosuan_assets(j, data_root),
)

results = []
for i, job in enumerate(jobs):
    t0 = time.time()
    print("[{}/{}] {} starting...".format(i+1, len(jobs), job.episode_id), flush=True)
    result = run_collection_job(
        job=job, dependencies=deps, output_root=output_root, resume=True
    )
    elapsed = time.time() - t0
    result["_elapsed"] = round(elapsed, 1)
    status = result.get("state", "?")
    print("  -> {} ({:.1f}s)".format(status, elapsed), flush=True)
    results.append(result)

summary = "/remote-home/lizilong/shenji_agent/Shenji/raw/pilot_v2/_summary.json"
with open(summary, "w") as f:
    json.dump({"results": results, "completed_at": time.strftime("%Y%m%d_%H%M%S")}, f, indent=2)
print("Summary saved", flush=True)

ok = all(r.get("state") in ("completed", "skipped") for r in results)
print("ALL DONE. Overall: {}".format("SUCCESS" if ok else "PARTIAL FAILURE"), flush=True)
