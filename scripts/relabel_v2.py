import sys, json, os
sys.path.insert(0, "/remote-home/lizilong/shenji_agent/Shenji/src")

from shenji_dataset.labeling import derive_grade_a_intent_plan
from shenji_dataset.normalizer import normalize_episode

CANONICAL_TABLES = [
    "episodes", "entities", "states", "observations",
    "actions", "events", "relations", "intent_segments",
]

RAW_ROOT = "/remote-home/lizilong/shenji_agent/Shenji/raw/pilot_v1"
CANON_ROOT = "/remote-home/lizilong/shenji_agent/Shenji/raw/canonical"

for ep_dir in sorted(os.listdir(RAW_ROOT)):
    ep_path = os.path.join(RAW_ROOT, ep_dir)
    if not os.path.isdir(ep_path) or not ep_dir.startswith("shenji_"):
        continue
    raw_path = os.path.join(ep_path, "raw_episode.json")
    if not os.path.isfile(raw_path):
        continue

    print("=== {} ===".format(ep_dir), flush=True)
    raw = json.load(open(raw_path))

    plan = derive_grade_a_intent_plan(raw)
    intent_path = os.path.join(ep_path, "intent_plan.json")
    with open(intent_path, "w") as f:
        json.dump(plan, f, indent=2)
    print("  intent_plan: {} segments".format(len(plan["segments"])), flush=True)

    tables = normalize_episode(raw, plan)

    can_dir = os.path.join(CANON_ROOT, ep_dir)
    os.makedirs(can_dir, exist_ok=True)
    for name in CANONICAL_TABLES:
        path = os.path.join(can_dir, "{}.jsonl".format(name))
        with open(path, "w") as f:
            for row in tables.get(name, []):
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    import hashlib
    artifacts = []
    for name in CANONICAL_TABLES:
        path = os.path.join(can_dir, "{}.jsonl".format(name))
        rows = 0
        with open(path) as f:
            for _ in f:
                rows += 1
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            sha.update(f.read())
        artifacts.append({
            "table": name, "format": "jsonl",
            "path": "{}.jsonl".format(name),
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
    print("  canonical: {} tables written".format(len(artifacts)), flush=True)

print("ALL DONE")
