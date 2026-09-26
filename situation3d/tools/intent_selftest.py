#!/usr/bin/env python3
"""Self-check for the intent layer (engine/intent.py).

Run on the target container, where there is no test runner:

    python3 tools/intent_selftest.py

Checks the things that can silently go wrong and would be invisible on screen:

1. **Ontology completeness** -- 22 intents, 6 parents, 4 status layers, every
   intent's parent exists, and exactly FIRE_SUPPORT / TARGET_GUIDANCE are
   secondary-only. A typo in the vendored file would otherwise just make one
   intent render with its raw English code and nobody would notice.
2. **Rule reachability** -- each derivable intent has a rule entry, and each
   undeliverable one states why.
3. **Hard invariants on real data** -- FEINT is never produced (the ontology
   forbids deriving it from behaviour), secondary-only intents are never the
   primary answer, and an intent produced is always in the ontology.
4. **Determinism** -- the same frames must produce byte-identical output; a
   non-deterministic rule would make the panel flicker frame to frame.

Exits non-zero on the first failure group with a readable message.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import cases, intent, scenario  # noqa: E402

FAILURES = []


def check(cond, label, detail=""):
    if cond:
        print("  ok   %s" % label)
    else:
        print("  FAIL %s %s" % (label, detail))
        FAILURES.append(label)


def main() -> int:
    print("== 1. ontology ==")
    try:
        doc = intent.load_ontology()
    except intent.IntentError as e:
        print("  FAIL ontology could not be loaded: %s" % e)
        return 1
    check(len(intent.intents()) == 22, "22 intents",
          "(got %d)" % len(intent.intents()))
    check(len(intent.parents()) == 6, "6 parent categories",
          "(got %d)" % len(intent.parents()))
    check([s["code"] for s in doc["status_layers"]] == list(intent.STATUS_LAYERS),
          "4 status layers in declared order")

    missing_parent = [i["code"] for i in intent.intents()
                      if i["parent_code"] not in intent.parents()]
    check(not missing_parent, "every intent's parent_code exists",
          "orphans: %s" % missing_parent)

    kids = sum(len(p.get("children") or []) for p in doc["parent_categories"])
    check(kids == 22, "parent children total 22", "(got %d)" % kids)

    sec = sorted(intent.secondary_only_codes())
    check(sec == ["FIRE_SUPPORT", "TARGET_GUIDANCE"],
          "secondary-only is exactly FIRE_SUPPORT/TARGET_GUIDANCE", "(got %s)" % sec)

    print("== 2. rule table ==")
    codes = set(intent.intent_codes())
    check(set(intent.SUPPORTED_RULES) == codes,
          "every intent has a SUPPORTED_RULES entry",
          "diff: %s" % sorted(set(intent.SUPPORTED_RULES) ^ codes))
    und = intent.underivable_report()
    check(all(u["reason"] for u in und), "each underivable intent states a reason")
    check(all(intent.SUPPORTED_RULES[c].get("derivable")
              for c in intent.derivable_codes()),
          "derivable_codes only lists derivable rules")
    check(not intent.SUPPORTED_RULES["FEINT"]["derivable"],
          "FEINT is declared underivable")
    print("       derivable=%d underivable=%d" % (len(intent.derivable_codes()), len(und)))
    for u in und:
        print("         %-16s %s" % (u["code"], u["reason"][:70]))

    print("== 3. real data invariants ==")
    case = cases.get_case(cases.DEFAULT_CASE_ID)
    targets = scenario.targets_for(case)
    frames = scenario.build_frames(targets)
    derived = intent.derive(frames, targets)
    check(len(derived["targets"]) == len(frames), "one result per frame")

    known = set(intent.intent_codes())
    sec_only = set(intent.secondary_only_codes())
    produced, bad_code, feint, sec_primary, unnamed = set(), [], 0, [], 0
    for fr in derived["targets"]:
        for tid, e in fr.items():
            if e.get("primary_intent") in sec_only:
                sec_primary.append((tid, e["primary_intent"]))
            for x in e.get("intent_set") or []:
                produced.add(x["code"])
                if x["code"] not in known:
                    bad_code.append(x["code"])
                if x["code"] == "FEINT":
                    feint += 1
                if not intent.intent_by_code(x["code"]).get("name_zh"):
                    unnamed += 1

    check(not bad_code, "no intent outside the ontology", str(set(bad_code)))
    check(feint == 0, "FEINT never derived (ontology forbids it)", "(got %d)" % feint)
    check(not sec_primary, "secondary-only never primary",
          str(sec_primary[:3]))
    check(unnamed == 0, "every produced intent has a Chinese name")
    print("       produced on demo data: %s" % sorted(produced))
    print("       never produced      : %s" % sorted(known - produced))

    print("== 4. determinism ==")
    h = []
    for _ in range(2):
        r = intent.derive(frames, targets)
        h.append(hashlib.sha256(
            json.dumps(r["targets"], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest())
    check(h[0] == h[1], "same input -> identical output", "%s vs %s" % (h[0][:12], h[1][:12]))

    print()
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), "; ".join(FAILURES)))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())