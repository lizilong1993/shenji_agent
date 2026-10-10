import sys, json, os, hashlib
sys.path.insert(0, "/remote-home/lizilong/shenji_agent/Shenji/src")

from shenji_dataset.labeling import derive_grade_a_intent_plan, derive_grade_b_intent_plan, derive_grade_c_intent_plan
from shenji_dataset.normalizer import normalize_episode

CANONICAL_TABLES = [
    "episodes", "entities", "states", "observations",
    "actions", "events", "relations", "intent_segments",
]

RAW_ROOT = "/remote-home/lizilong/shenji_agent/Shenji/raw/pilot_v3"
CANON_ROOT = "/remote-home/lizilong/shenji_agent/Shenji/raw/canonical_v3"

all_intents = {}
for ep_dir in sorted(os.listdir(RAW_ROOT)):
    ep_path = os.path.join(RAW_ROOT, ep_dir)
    if not os.path.isdir(ep_path) or not ep_dir.startswith("shenji_"):
        continue
    raw_path = os.path.join(ep_path, "raw_episode.json")
    if not os.path.isfile(raw_path):
        continue
    print("=== {} ===".format(ep_dir), flush=True)
    raw = json.load(open(raw_path))

    # Run all three labelers and merge
    plan_a = derive_grade_a_intent_plan(raw)
    plan_b = derive_grade_b_intent_plan(raw)
    plan_c = derive_grade_c_intent_plan(raw)
    seen_ids = set()
    merged = []
    for plan in (plan_a, plan_b, plan_c):
        for seg in plan.get("segments", []):
            sid = seg.get("segment_id", "")
            if sid not in seen_ids:
                seen_ids.add(sid)
                merged.append(seg)
    merged.sort(key=lambda s: (s.get("start_step", 0), s.get("subject_id", ""), s.get("primary_intent", "")))
    plan = {"episode_id": plan_a["episode_id"], "generator": "shenji_all_grade_labeler_v1", "segments": merged}

    intent_path = os.path.join(ep_path, "intent_plan.json")
    with open(intent_path, "w") as f:
        json.dump(plan, f, indent=2)
    print("  segments: {}".format(len(plan["segments"])), flush=True)

    # Count by intent
    for s in merged:
        i = s.get("primary_intent", "?")
        all_intents[i] = all_intents.get(i, 0) + 1

    # Normalize
    tables = normalize_episode(raw, plan)

    # Write canonical tables
    can_dir = os.path.join(CANON_ROOT, ep_dir)
    os.makedirs(can_dir, exist_ok=True)
    for name in CANONICAL_TABLES:
        path = os.path.join(can_dir, "{}.jsonl".format(name))
        with open(path, "w") as f:
            for row in tables.get(name, []):
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    # Write manifest
    artifacts = []
    for name in CANONICAL_TABLES:
        path = os.path.join(can_dir, "{}.jsonl".format(name))
        rows = 0
        with open(path) as f:
            for _ in f: rows += 1
        sha = hashlib.sha256()
        with open(path, "rb") as f: sha.update(f.read())
        artifacts.append({"table": name, "format": "jsonl", "path": "{}.jsonl".format(name),
                          "rows": rows, "bytes": os.path.getsize(path), "sha256": sha.hexdigest()})
    manifest = {"dataset": "Shenji", "dataset_version": "1.0.0", "schema_version": "shenji_schema_v1",
                "table_order": CANONICAL_TABLES, "artifacts": artifacts}
    with open(os.path.join(can_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

print("\n=== INTENT COVERAGE (pilot_v3, all grades) ===", flush=True)
total = 0
for k, v in sorted(all_intents.items()):
    print("  {}: {}".format(k, v), flush=True)
    total += v
print("  Total: {}".format(total), flush=True)
print("ALL DONE", flush=True)
