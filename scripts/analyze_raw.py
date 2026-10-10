import json, os

ROOT = "/remote-home/lizilong/shenji_agent/Shenji/raw/pilot_v1"
EPID = "shenji_1231_20260711_challenger_v1_vs_baseline_rule_v1_f346b1c11c"
path = os.path.join(ROOT, EPID, "raw_episode.json")
d = json.load(open(path))

print("=== RAW EPISODE METADATA ===")
for k, v in d.get("metadata", {}).items():
    print("  {}: {}".format(k, v))

print()
print("=== OBSERVATION SIDES ===")
obs = d.get("observations", {})
for k, v in obs.items():
    if v:
        print("  {}: {} records, first keys={}".format(k, len(v), list(v[0].keys())[:6]))
    else:
        print("  {}: empty".format(k))

print()
print("=== ACTIONS ===")
acts = d.get("actions", {})
for k, v in acts.items():
    if v:
        if isinstance(v[0], dict):
            print("  {}: {} records, keys={}".format(k, len(v), list(v[0].keys())[:6]))
        else:
            print("  {}: {} records, type={}".format(k, len(v), type(v[0]).__name__))
    else:
        print("  {}: empty".format(k))
