#!/usr/bin/env python3
"""Promote a selected engine run to the demo worktree (no engine execution).

The demo artifact is copied byte-for-byte from ``trajectory.json``; the release
manifest records the run id, engine commit, scenario hash and the sha256 of the
copy.  Only a run that is ``best`` (selected in the test console) and passed
validation may be promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
for p in (SITUATION3D, os.path.join(SITUATION3D, "server")):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_store import RunError, RunStore  # noqa: E402


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _situations_dir(demo_root):
    root = os.path.abspath(demo_root)
    for rel in (("situation3d", "web", "data", "situations"), ("web", "data", "situations")):
        candidate = os.path.join(root, *rel)
        if os.path.isdir(candidate):
            return candidate
    raise SystemExit("demo situations dir not found under %s" % root)


def promote(case_id, run_id, runs_dir, demo_root, dry_run=False):
    store = RunStore(runs_dir)
    run = store.get_run(case_id, run_id)
    if run.get("status") != "completed":
        raise RunError("run is not completed: %s" % run.get("status"))
    if "best" not in (run.get("roles") or []):
        raise RunError("run is not selected as best; select it first")
    if not (run.get("validation") or {}).get("passed", False):
        raise RunError("run failed validation")
    check = store.verify(case_id, run_id)
    if not check["ok"]:
        raise RunError("checksum mismatch: %s" % ", ".join(check["mismatches"]))
    trajectory = store.read_artifact(case_id, run_id, "trajectory.json")
    digest = _sha256(trajectory)
    target_dir = _situations_dir(demo_root)
    target = os.path.join(target_dir, case_id + ".json")
    # Keep the manifest *outside* the scanned situations directory: a JSON
    # file in there with the same case_id would shadow the trajectory.
    release_path = os.path.join(os.path.dirname(target_dir), "demo_release.json")
    if dry_run:
        return {"ok": True, "dry_run": True, "target": target,
                "sha256": digest, "run_id": run_id}
    os.makedirs(target_dir, exist_ok=True)
    if os.path.isfile(target):
        backup = target + ".bak_prev_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        shutil.copy2(target, backup)
    tmp = target + ".tmp.%d" % os.getpid()
    shutil.copyfile(trajectory, tmp)
    os.replace(tmp, target)
    if _sha256(target) != digest:
        raise RunError("copy checksum mismatch after promotion")
    release = {
        "schema": "situation3d.demo_release.v1",
        "case_id": case_id,
        "run_id": run_id,
        "trajectory_sha256": digest,
        "engine_commit": run.get("engine_commit"),
        "scenario_hash": run.get("scenario_hash"),
        "envpack_hash": run.get("envpack_hash"),
        "seed": run.get("seed"),
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "src_run": os.path.join(runs_dir, case_id, run_id),
    }
    with open(release_path + ".tmp.%d" % os.getpid(), "w", encoding="utf-8") as fh:
        json.dump(release, fh, ensure_ascii=False, indent=1)
    os.replace(release_path + ".tmp.%d" % os.getpid(), release_path)
    store.audit_note(case_id, run_id, "promote", target)
    return {"ok": True, "target": target, "sha256": digest,
            "release": release_path, "run_id": run_id}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="taiwan_dense_transfer_v1")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-dir", default=os.path.join(SITUATION3D, "var", "sim_runs"))
    ap.add_argument("--demo-root", required=True,
                    help="demo worktree root, e.g. /remote-home/lizilong/shenji_agent_demo")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args(argv)
    if not args.confirm and not args.dry_run:
        print("refusing to promote without --confirm", file=sys.stderr)
        return 2
    try:
        result = promote(args.case, args.run_id, args.runs_dir, args.demo_root,
                         dry_run=args.dry_run)
    except RunError as exc:
        print("promote failed: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
