import json, os

CANO = "/remote-home/lizilong/shenji_agent/Shenji/raw/canonical"
coverage = json.load(open("/remote-home/lizilong/shenji_agent/Shenji/validation/pilot_coverage.json"))

print("=== COVERAGE SUMMARY ===")
print("  Jobs: {}/{} completed".format(coverage["jobs"]["completed"], coverage["jobs"]["total"]))
print("  Policies: {}".format(json.dumps(coverage["policies"])))
print("  Scenarios: {}".format(json.dumps(coverage["scenarios"])))
print("  Intent segments: {} total, {} episodes without labels".format(
    coverage["segments"]["total"], coverage["segments"]["completed_episodes_without_labels"]))

targets = coverage["target_progress"]
met = sum(1 for v in targets.values() if v["actual"] >= v["pilot_target"])
total_targets = len(targets)
print("  Intent targets met: {}/{}".format(met, total_targets))

print()
print("=== CROSS-TABLE CONSISTENCY ===")
for ep in sorted(os.listdir(CANO)):
    ep_dir = os.path.join(CANO, ep)
    if not os.path.isdir(ep_dir):
        continue
    mf = os.path.join(ep_dir, "manifest.json")
    if not os.path.isfile(mf):
        print("  {}: NO MANIFEST".format(ep))
        continue
    m = json.load(open(mf))
    art = {a["table"]: a for a in m["artifacts"]}
    ok = True
    for table in ["episodes","entities","states","observations","actions","events","relations","intent_segments"]:
        if table not in art:
            print("  {}: MISSING {}".format(ep, table))
            ok = False
    for table, info in art.items():
        fpath = os.path.join(ep_dir, info["path"])
        if info["rows"] > 0:
            actual = 0
            with open(fpath) as f:
                for _ in f:
                    actual += 1
            if actual != info["rows"]:
                print("  {} {}: manifest={} file={}".format(ep, table, info["rows"], actual))
                ok = False
    print("  {}: {}".format(ep, "OK" if ok else "ISSUES"))
