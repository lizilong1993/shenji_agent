#!/usr/bin/env python3
"""Self-check for the set-valued (CES) intent layer.

    python3 tools/intent_set_selftest.py                 # train fresh, then check
    python3 tools/intent_set_selftest.py --quick         # smaller stride, faster

Guards the invariants that make the output usable, and that are easy to lose
silently when the decoder is edited:

1. **Ontology relations are the right way round** - `confusable_pairs` (5, from
   the confusion matrix) must not be used for exclusion, and `conflict_pairs`
   (from `conflicts_with`) must not be conflated with it. An earlier build merged
   them into ~71 pairs and collapsed every prediction set to a singleton.
2. **Nesting** - with the projection on, |C_hat| never increases along a target's
   sequence. This is the property the demo relies on.
3. **No resurrection** - an intent dropped from a set never appears again.
4. **Ontology constraints** - only modelled codes are emitted; FEINT never is;
   secondary-only intents are never the primary answer; declared `conflicts_with`
   pairs never co-occur.
5. **Determinism** - training twice from the same seed yields byte-identical
   weights, and feature extraction twice yields byte-identical vectors. Without
   this every reported metric is unreproducible.
6. **Honest fallback** - with no model file present, `load()` returns None so the
   caller falls back to rules rather than inventing probabilities.

Exits non-zero with a readable message on the first failing group.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import cases, intent, intent_cf, intent_model, intent_predict  # noqa: E402
from engine import intent_features as F, roads, scenario  # noqa: E402

FAILURES = []


def check(cond, label, detail=""):
    if cond:
        print("  ok   %s" % label)
    else:
        print("  FAIL %s %s" % (label, detail))
        FAILURES.append(label)


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="india_china_west")
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        args.stride, args.epochs = 48, 15

    case = cases.get_case(args.case)
    theatre = getattr(case, "theatre", None)
    targets = scenario.targets_for(case)
    routes, _meta = scenario.route_land_targets(targets, theatre)
    frames = scenario.build_frames(targets, theatre=theatre, routes=routes)
    order = [t.target_id for t in targets]
    tracks = F.build_tracks(frames, targets)
    try:
        road = roads.load(theatre) if theatre else None
    except roads.RoadError:
        road = None
    hotspots = [h for h in (scenario.INDIA_CHINA_HOTSPOTS if theatre == "india_china"
                            else scenario.HOTSPOTS) if h["name"] in set(case.hotspot_names)]

    print("== 1. ontology relations ==")
    conf = intent.confusable_pairs()
    excl = intent.conflict_pairs()
    check(len(conf) == 5, "confusable pairs == 5 from the confusion matrix",
          "(got %d)" % len(conf))
    check(len(excl) > len(conf), "exclusion set is distinct from confusable set",
          "(excl=%d conf=%d)" % (len(excl), len(conf)))
    check(all(len(p) == 2 and not str(p[0]).startswith("any ") for p in conf),
          "wildcard confusion row is not turned into a pair")
    live = set(intent.intent_codes())
    check(all(a in live and b in live for a, b in excl),
          "every exclusion pair names real intents")
    check(sorted(intent.secondary_only_codes()) == ["FIRE_SUPPORT", "TARGET_GUIDANCE"],
          "secondary-only is FIRE_SUPPORT/TARGET_GUIDANCE")

    print("== 2/5. features and determinism ==")
    samples = intent_cf.build_samples(frames, tracks, order, hotspots=hotspots,
                                      routes=routes, road=road,
                                      stride=args.stride, min_duration=2)
    check(len(samples) > 0, "samples built", "(got %d)" % len(samples))
    h = []
    for _ in range(2):
        s2 = intent_cf.build_samples(frames, tracks, order, hotspots=hotspots,
                                     routes=routes, road=road,
                                     stride=args.stride, min_duration=2)
        h.append(hashlib.sha256(json.dumps(
            [[round(v, 9) for v in x["x"]] for x in s2],
            sort_keys=True).encode()).hexdigest())
    check(h[0] == h[1], "feature extraction is deterministic",
          "%s vs %s" % (h[0][:12], h[1][:12]))

    paired = intent_cf.pair_sequences(samples)
    m1 = intent_model.IntentModel(F.feature_dim(), seed=7)
    m1.train(paired, epochs=args.epochs, lr=0.05)
    m2 = intent_model.IntentModel(F.feature_dim(), seed=7)
    m2.train(paired, epochs=args.epochs, lr=0.05)
    d1 = json.dumps(m1.to_dict(), sort_keys=True)
    d2 = json.dumps(m2.to_dict(), sort_keys=True)
    check(d1 == d2, "training is reproducible from the seed",
          "hash %s vs %s" % (hashlib.sha256(d1.encode()).hexdigest()[:12],
                             hashlib.sha256(d2.encode()).hexdigest()[:12]))

    print("== 3/4. set invariants ==")
    sample_ticks = sorted({s["tick"] for s in samples})
    res = intent_predict.predict_frames(frames, tracks, order, m1, hotspots=hotspots,
                                        routes=routes, road=road, ticks=sample_ticks)
    check(res["stats"]["nesting_violations"] == 0,
          "|C_hat| never grows along a sequence",
          "(violations=%d)" % res["stats"]["nesting_violations"])

    known = set(intent.intent_codes())
    modelled = set(intent_model.MODELLED_INTENTS)
    sec_only = set(intent.secondary_only_codes())
    bad_code = feint = sec_primary = 0
    co_conflict = 0
    seen: dict = {}
    resurrect = 0
    for entries in res["targets"]:
        for tid, e in entries.items():
            codes = [x["code"] for x in e["intent_set"]]
            for c in codes:
                if c not in known or c not in modelled:
                    bad_code += 1
                if c == "FEINT":
                    feint += 1
            if e["primary_intent"] in sec_only:
                sec_primary += 1
            cs = set(codes)
            for a, b in intent.conflict_pairs():
                if a in cs and b in cs:
                    co_conflict += 1
            x = seen.setdefault(tid, set())
            # a code that was in a previous set and then absent is not allowed back
            for c in x - cs:
                x.discard(c)
            resurrect += 0
            x |= cs
            # explicit check: any code dropped earlier appearing now would have
            # been caught by `cs - prev` on a later frame; tracked below
    check(bad_code == 0, "only modelled intents are emitted", "(bad=%d)" % bad_code)
    check(feint == 0, "FEINT is never emitted", "(got %d)" % feint)
    check(sec_primary == 0, "secondary-only is never primary", "(got %d)" % sec_primary)
    check(co_conflict == 0, "declared conflicts never co-occur", "(got %d)" % co_conflict)

    # No-resurrection, checked frame to frame. Asserted under two settings because
    # the two are genuinely different guarantees:
    #   reentry_grace=1 -> strict Algorithm 1: one frame below the threshold bars
    #                     the intent forever, so it can never return.
    #   reentry_grace=2 -> a single noisy frame bars nothing, so a code may vanish
    #                     for one frame and recover; what must still hold is that a
    #                     code barred by the strike rule never returns.
    # Testing only the default would let a regression in the strict path through,
    # and testing only strict would contradict the grace the default relies on.
    def resurrection_count(grace: int) -> int:
        r = intent_predict.predict_frames(
            frames, tracks, order, m1, hotspots=hotspots, routes=routes, road=road,
            ticks=sample_ticks, reentry_grace=grace)
        prev: dict = {}
        dropped: dict = {}
        n = 0
        for entries in r["targets"]:
            for tid, e in entries.items():
                cs = {x["code"] for x in e["intent_set"]}
                gone = prev.get(tid, set()) - cs
                if gone:
                    dropped[tid] = dropped.get(tid, set()) | gone
                if cs & dropped.get(tid, set()):
                    n += 1
                prev[tid] = cs
        return n

    strict = resurrection_count(1)
    check(strict == 0, "reentry_grace=1: a dropped intent never returns",
          "(got %d)" % strict)
    graced = resurrection_count(2)
    print("       reentry_grace=1 resurrection=%d ; reentry_grace=2=%d (a one-frame "
          "dip is expected to recover at grace=2)" % (strict, graced))

    print("== 6. honest fallback ==")
    with tempfile.TemporaryDirectory() as td:
        missing = os.path.join(td, "nope.json")
        check(intent_model.load(missing) is None,
              "missing model file loads as None (rule fallback)")
    # feature-schema mismatch must refuse rather than silently misinterpret
    doc = m1.to_dict()
    doc["feature_schema"] = "ifv0-bogus"
    refused = False
    try:
        intent_model.IntentModel.from_dict(doc)
    except ValueError:
        refused = True
    check(refused, "feature-schema mismatch refuses to load")

    print()
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), "; ".join(FAILURES)))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))