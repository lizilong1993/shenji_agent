import json, os

RAW_ROOT = "/remote-home/lizilong/shenji_agent/Shenji/raw/pilot_v1"
CANON_ROOT = "/remote-home/lizilong/shenji_agent/Shenji/raw/canonical"
PYTHON = "/remote-home/lizilong/venvs/shenji/bin/python3.10"
SHENJI_SRC = "/remote-home/lizilong/shenji_agent/Shenji/src"

script = """
import sys, json, os
sys.path.insert(0, %(src)r)
from shenji_dataset.labeling import derive_grade_a_intent_plan
from shenji_dataset.normalizer import normalize_episode

CANONICAL_TABLES = [
    "episodes", "entities", "states", "observations",
    "actions", "events", "relations", "intent_segments",
]

for ep_dir in sorted(os.listdir(%(raw)r)):
    ep_path = os.path.join(%(raw)r, ep_dir)
    if not os.path.isdir(ep_path) or not ep_dir.startswith("shenji_"):
        continue
    raw_path = os.path.join(ep_path, "raw_episode.json")
    if not os.path.isfile(raw_path):
        continue

    print(f"=== {ep_dir} ===", flush=True)
    raw = json.load(open(raw_path))

    # Re-label
    plan = derive_grade_a_intent_plan(raw)
    intent_path = os.path.join(ep_path, "intent_plan.json")
    with open(intent_path, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"  intent_plan: {len(plan['segments'])} segments", flush=True)

    # Re-normalize
    tables = normalize_episode(raw, plan)

    # Write canonical tables (no write_release to avoid dep issues)
    can_dir = os.path.join(%(can)s, ep_dir)
    os.makedirs(can_dir, exist_ok=True)
    for name in CANONICAL_TABLES:
        path = os.path.join(can_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for row in tables.get(name, []):
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\\n")

    # Write minimal manifest
    import hashlib
    artifacts = []
    for name in CANONICAL_TABLES:
        path = os.path.join(can_dir, f"{name}.jsonl")
        rows = 0
        with open(path) as f:
            for _ in f: rows += 1
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            sha.update(f.read())
        artifacts.append({
            "table": name, "format": "jsonl",
            "path": f"{name}.jsonl",
            "rows": rows, "bytes": os.path.getsize(path),
            "sha256": sha.hexdigest(),
        })
    meta = raw.get("metadata", {})
    manifest = {
        "dataset": "Shenji",
        "dataset_version": "1.0.0",
        "schema_version": "shenji_schema_v1",
        "ontology_version": "1.0.0",
        "table_order": CANONICAL_TABLES,
        "artifacts": artifacts,
    }
    with open(os.path.join(can_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  canonical: {len(artifacts)} tables", flush=True)

print("ALL DONE")
"""
print(script % {"raw": RAW_ROOT, "can": CANON_ROOT, "src": SHENJI_SRC})
