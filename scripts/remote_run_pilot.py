#!/usr/bin/env python3
"""Remote pilot collection runner - sends script to server-154 and runs it."""
import sys, subprocess, shlex

REMOTE = "root@100.73.109.52"
PROXY = '"C:\\Program Files\\Tailscale\\tailscale.exe" nc %h %p'
ID = "C:\\Users\\lizilong\\.ssh\\id_ed25519_tailscale"
PYTHON = "/remote-home/lizilong/venvs/shenji/bin/python3.10"
SHENJI = "/remote-home/lizilong/shenji_agent/Shenji"

SCRIPT = f"""#!/usr/bin/env python3
\"\"\"Run all pilot collection jobs sequentially.\"\"\"
import sys, json, time
sys.path.insert(0, "/remote-home/lizilong/shenji_agent")
sys.path.insert(0, "{SHENJI}/src")

from shenji_dataset.collection import run_collection_job, CollectionDependencies
from shenji_dataset.jobs import load_collection_manifest, expand_jobs
from shenji_dataset.runtime import resolve_miaosuan_env_class, resolve_agent_class, load_miaosuan_assets

manifest = load_collection_manifest("{SHENJI}/configs/pilot_manifest_v1.json")
jobs = expand_jobs(manifest)
print(f"Expanded {{len(jobs)}} jobs")

env_class = resolve_miaosuan_env_class(project_root="/remote-home/lizilong/shenji_agent")
agent_class = resolve_agent_class("ai.agent:Agent", project_root="/remote-home/lizilong/shenji_agent")
data_root = "/remote-home/lizilong/shenji_agent/land_wargame_sdk/Data/Data"
output_root = "{SHENJI}/raw/pilot_v1"

deps = CollectionDependencies(
    env_factory=lambda: env_class(),
    red_agent_factory=lambda: agent_class(),
    blue_agent_factory=lambda: agent_class(),
    asset_loader=lambda j: load_miaosuan_assets(j, data_root),
)

results = []
for i, job in enumerate(jobs):
    t0 = time.time()
    print(f"[{{i+1}}/{{len(jobs)}}] {{job.episode_id}} starting...", flush=True)
    result = run_collection_job(
        job=job, dependencies=deps, output_root=output_root, resume=True
    )
    elapsed = time.time() - t0
    result["_elapsed"] = round(elapsed, 1)
    result["_job_index"] = i
    status = result.get("state", "?")
    print(f"[{{i+1}}/{{len(jobs)}}] {{job.episode_id}} -> {{status}} ({{elapsed:.1f}}s)", flush=True)
    results.append(result)

summary = {SHENJI}/raw/pilot_v1/_pilot_summary.json"
with open(summary, "w") as f:
    json.dump({{"results": results, "completed_at": time.strftime("%%Y%%m%%d_%%H%%M%%S")}}, f, indent=2)
print(f"Summary saved to {{summary}}", flush=True)

ok = all(r.get("state") in ("completed", "skipped") for r in results)
print(f"ALL JOBS DONE. Overall: {{'SUCCESS' if ok else 'PARTIAL FAILURE'}}", flush=True)
sys.exit(0 if ok else 1)
"""

def ssh(cmd, timeout=300):
    full_cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=NUL",
        "-i", ID,
        "-o", f'ProxyCommand={PROXY}',
        REMOTE,
        cmd,
    ]
    return subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)

# Write script to remote
print("Writing script to remote...")
r = ssh(f"cat > /tmp/run_pilot_full.py << 'ENDOFSCRIPT'\n{SCRIPT}\nENDOFSCRIPT")
print("stdout:", r.stdout[-500:] if r.stdout else "")
print("stderr:", r.stderr[-500:] if r.stderr else "")
if r.returncode != 0:
    print(f"Write failed (rc={r.returncode})")
    sys.exit(1)

# Execute
print("Starting pilot collection (in background on remote)...")
r = ssh(f"nohup {PYTHON} /tmp/run_pilot_full.py > {SHENJI}/raw/pilot_v1/_pilot_run.log 2>&1 &")
print("stdout:", r.stdout)
print("stderr:", r.stderr)
print(f"Return code: {r.returncode}")
print("Pilot collection dispatched in background. Check log at:")
print(f"  {SHENJI}/raw/pilot_v1/_pilot_run.log")
