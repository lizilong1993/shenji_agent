import json, os

CANO = "/remote-home/lizilong/shenji_agent/Shenji/raw/canonical"
EPID = "shenji_1231_20260711_challenger_v1_vs_baseline_rule_v1_f346b1c11c"
DIR = os.path.join(CANO, EPID)

print("=== ACTIONS ===")
with open(os.path.join(DIR, "actions.jsonl")) as f:
    total = 0; nonnull = 0
    for i, line in enumerate(f):
        d = json.loads(line)
        total += 1
        if d.get("action_type") is not None:
            nonnull += 1
            if nonnull <= 3:
                print("  row{}: agent={} eid={} type={} frame={}".format(
                    i, d.get("agent_id"), d.get("entity_id"),
                    d["action_type"], d.get("frame_idx")))
    print("  Total rows: {}, non-null: {}".format(total, nonnull))

print()
print("=== RED TRAJECTORY ===")
red_ids = ["14911","14913","14914"]
with open(os.path.join(DIR, "states.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        if d["entity_id"] in red_ids and d["step"] in [0, 100, 200, 400, 600, 800, 999]:
            print("  eid={} step={} hex={} hp={} spd={} ms={}".format(
                d["entity_id"], d["step"], d.get("hex_id"),
                d.get("health"), d.get("speed"), d.get("move_state")))

print()
print("=== OBSERVATIONS SIDES ===")
obs_sides = {}
with open(os.path.join(DIR, "observations.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        for k, v in d.items():
            if k in ("step","dataset","schema_version","episode_id"):
                continue
            if k not in obs_sides:
                obs_sides[k] = {"steps": 0, "ops": 0}
            obs_sides[k]["steps"] += 1
            if isinstance(v, dict):
                obs_sides[k]["ops"] += len(v.get("operators", []))
for k, v in sorted(obs_sides.items()):
    avg = v["ops"] / v["steps"] if v["steps"] else 0
    print("  {}: {} obs, avg {:.1f} ops/obs".format(k, v["steps"], avg))

print()
print("=== EVENTS ===")
ev_types = {}
with open(os.path.join(DIR, "events.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        ev_types[d["event_type"]] = ev_types.get(d["event_type"], 0) + 1
for t, c in sorted(ev_types.items(), key=lambda x: -x[1]):
    print("  {}: {}".format(t, c))

print()
print("=== INTENT PLAN ===")
ip = json.load(open("/remote-home/lizilong/shenji_agent/Shenji/raw/pilot_v1/" + EPID + "/intent_plan.json"))
print(json.dumps(ip, indent=2))

print()
print("=== RED ENTITY SURVIVAL ===")
with open(os.path.join(DIR, "entities.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        if d["side"] == "red":
            print("  eid={} first={} last={} type={}.{}".format(
                d["entity_id"], d["first_step"], d["last_step"],
                d["unit_type_id"], d["subtype_id"]))
            if d["last_step"] < 999:
                print("    WARNING: Lost before episode end!")

print()
print("=== BLUE ENTITY SURVIVAL ===")
with open(os.path.join(DIR, "entities.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        if d["side"] == "blue":
            print("  eid={} first={} last={} type={}.{}".format(
                d["entity_id"], d["first_step"], d["last_step"],
                d["unit_type_id"], d["subtype_id"]))
