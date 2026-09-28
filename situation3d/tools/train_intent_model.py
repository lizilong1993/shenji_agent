#!/usr/bin/env python3
"""Train the set-valued intent model and report a rule-vs-model comparison.

Run on the target container:

    python3 tools/train_intent_model.py                    # default case
    python3 tools/train_intent_model.py --case india_china_west --epochs 40

Read the module docstring of `engine/intent_model.py` before interpreting the
numbers. The short version: the CES supervision target is derived from the same
rule margins the model also sees as features, so this trains a *contraction
policy*, not a better intent recogniser. The metrics below compare the two on
things the rule layer has no notion of - set nesting, set size, and the
ambiguous/committed split.

Outputs
-------
* `engine/data/intent_model.json` - weights, feature schema, training provenance
* `intent_ablation_<date>.md`     - the comparison table, for the report

Standard library only: the container has python 3.8 and no pip.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import cases, intent, intent_cf, intent_model, intent_predict  # noqa: E402
from engine import intent_features as F  # noqa: E402
from engine import roads, scenario  # noqa: E402


def set_size_stats(samples):
    """Set-size summary, plus the nesting property the projection guarantees."""
    sizes = [s["n_in_set"] for s in samples]
    if not sizes:
        return {}
    by_target = {}
    for s in samples:
        by_target.setdefault(s["target"], []).append(s)
    runs = 0
    short_runs = 0
    for tid, rows in by_target.items():
        rows = sorted(rows, key=lambda r: r["tick"])
        cur = None
        length = 0
        for r in rows:
            key = tuple(sorted(i for i, v in enumerate(r["y"]) if v > 0.5))
            if key == cur:
                length += 1
            else:
                if cur is not None:
                    runs += 1
                    if length <= 2:
                        short_runs += 1
                cur, length = key, 1
        if cur is not None:
            runs += 1
            if length <= 2:
                short_runs += 1
    return {
        "mean_set_size": round(sum(sizes) / float(len(sizes)), 3),
        "singleton_frac": round(sum(1 for x in sizes if x == 1) / float(len(sizes)), 3),
        "ambiguous_frac": round(sum(1 for x in sizes if x > 1) / float(len(sizes)), 3),
        "background_frac": round(sum(1 for x in sizes if x == 0) / float(len(sizes)), 3),
        "ces_runs": runs,
        "ces_short_runs": short_runs,
        "ces_short_run_frac": round(short_runs / float(runs), 3) if runs else 0.0,
    }


def predict_stats(result, samples):
    """Model-side set sizes and nesting, computed from the prediction output."""
    per_frame = result["targets"]
    sizes = []
    by_target = {}
    for i, entries in enumerate(per_frame):
        for tid, e in entries.items():
            by_target.setdefault(tid, []).append(e["intent_set_size"])
            sizes.append(e["intent_set_size"])
    if not sizes:
        return {}
    runs = short_runs = 0
    for tid, series in by_target.items():
        cur, length = None, 0
        for v in series:
            if v == cur:
                length += 1
            else:
                if cur is not None:
                    runs += 1
                    if length <= 2:
                        short_runs += 1
                cur, length = v, 1
        if cur is not None:
            runs += 1
            if length <= 2:
                short_runs += 1
    return {
        "mean_set_size": round(sum(sizes) / float(len(sizes)), 3),
        "singleton_frac": round(sum(1 for x in sizes if x == 1) / float(len(sizes)), 3),
        "ambiguous_frac": round(sum(1 for x in sizes if x > 1) / float(len(sizes)), 3),
        "model_runs": runs,
        "model_short_runs": short_runs,
        "model_short_run_frac": round(short_runs / float(runs), 3) if runs else 0.0,
        "no_reentry_events": result["stats"]["no_reentry_events"],
        "nesting_violations": result["stats"]["nesting_violations"],
    }


def main(argv) -> int:
    ap = argparse.ArgumentParser(description="Train the set-valued intent model")
    ap.add_argument("--case", default=None, help="case id (default: registry default)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--stride", type=int, default=4,
                    help="frame stride when building samples (4 = every 2 min)")
    ap.add_argument("--holdout", default="",
                    help="comma-separated target ids held out (default: last 40%% by id)")
    ap.add_argument("--min-duration", type=int, default=2,
                    help="ontology minimum_duration honoured on the CES target")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    case = cases.get_case(args.case or cases.DEFAULT_CASE_ID)
    print("case            : %s (%s)" % (case.case_id, case.name))
    print("theatre         : %s" % getattr(case, "theatre", None))

    targets = scenario.targets_for(case)
    theatre = getattr(case, "theatre", None)
    routes, _meta = scenario.route_land_targets(targets, theatre)
    frames = scenario.build_frames(targets, theatre=theatre, routes=routes)
    order = [t.target_id for t in targets]
    tracks = F.build_tracks(frames, targets)

    road = None
    try:
        road = roads.load(theatre) if theatre else None
    except roads.RoadError:
        road = None

    hotspots = []
    if theatre == "india_china":
        hotspots = [h for h in scenario.INDIA_CHINA_HOTSPOTS
                    if h["name"] in set(case.hotspot_names)]
    else:
        hotspots = [h for h in scenario.HOTSPOTS if h["name"] in set(case.hotspot_names)]

    t0 = time.time()
    samples = intent_cf.build_samples(
        frames, tracks, order, hotspots=hotspots, routes=routes, road=road,
        stride=args.stride, min_duration=args.min_duration)
    print("samples         : %d  (%.1fs, stride=%d, min_duration=%d)"
          % (len(samples), time.time() - t0, args.stride, args.min_duration))
    if not samples:
        print("ERROR: no training samples", file=sys.stderr)
        return 1

    holdout = ([h.strip() for h in args.holdout.split(",") if h.strip()]
               if args.holdout else order[len(order) * 3 // 5:])
    train, test = intent_cf.split_by_target(samples, holdout)
    print("holdout targets : %s  (train=%d test=%d)"
          % (", ".join(holdout) or "none", len(train), len(test)))
    if not train:
        print("ERROR: empty training split", file=sys.stderr)
        return 1

    paired = intent_cf.pair_sequences(train)
    model = intent_model.IntentModel(F.feature_dim(), seed=args.seed)
    # Fit the feature scaler on the *training* split only. Fitting on all samples
    # would leak holdout statistics into the weights.
    model.fit_scaler(train)
    print("scaler          : fitted on %d training rows" % len(train))
    t0 = time.time()
    print("training        : %d features, %d intents, %d epochs"
          % (F.feature_dim(), len(model.intents), args.epochs))
    hist = model.train(paired, epochs=args.epochs, lr=args.lr, verbose=True)
    print("train time      : %.1fs" % (time.time() - t0))
    # Temperature is fitted on the *holdout* split: calibrating reported
    # confidence on the data the weights were fitted to would just re-assert
    # whatever the training loss already forced.
    if test:
        t_best = model.fit_temperature(intent_cf.pair_sequences(test))
        print("temperature     : %.2f (fitted on %d held-out rows)" % (t_best, len(test)))

    model.meta.update({
        "version": "1.0.0-%s" % datetime.now(timezone.utc).strftime("%Y%m%d"),
        "trained_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "case": case.case_id,
        "theatre": theatre,
        "epochs": args.epochs,
        "lr": args.lr,
        "stride": args.stride,
        "min_duration": args.min_duration,
        "holdout_targets": holdout,
        "n_train": len(train),
        "n_test": len(test),
        "ces_margin": intent_cf.CES_MARGIN,
        "final_losses": hist["final"],
        "note": ("CES supervision is rule-derived (see engine/intent_cf.py); this "
                 "model learns a contraction policy, not a better recogniser."),
    })

    out = args.out or intent_model.model_path()
    size = intent_model.save(model, out)
    print("model written   : %s (%.1f KB)" % (out, size / 1024.0))

    # ---- comparison ----
    print()
    print("== rule/CES target vs model set ==")
    ces = set_size_stats(samples)
    # Score only the sampled ticks, for the same reason training used a stride:
    # full-frame extraction is dominated by road and neighbour geometry walks.
    sample_ticks = sorted({s["tick"] for s in samples})
    res = intent_predict.predict_frames(
        frames, tracks, order, model, hotspots=hotspots, routes=routes, road=road,
        ticks=sample_ticks)
    pred = predict_stats(res, samples)
    rows = [
        ("mean set size", ces.get("mean_set_size"), pred.get("mean_set_size")),
        ("singleton frac", ces.get("singleton_frac"), pred.get("singleton_frac")),
        ("ambiguous frac", ces.get("ambiguous_frac"), pred.get("ambiguous_frac")),
        ("runs", ces.get("ces_runs"), pred.get("model_runs")),
        ("short runs (<=2)", ces.get("ces_short_runs"), pred.get("model_short_runs")),
        ("short-run frac", ces.get("ces_short_run_frac"), pred.get("model_short_run_frac")),
    ]
    print("%-20s %12s %12s" % ("metric", "CES target", "model"))
    for name, a, b in rows:
        print("%-20s %12s %12s" % (name, a, b))
    print()
    print("no-reentry events   : %d" % pred.get("no_reentry_events", -1))
    print("nesting violations  : %d  (must be 0 with the projection on)"
          % pred.get("nesting_violations", -1))
    # Saturation check: if several intents score exactly 1.0 the read-out has
    # blown up and the adaptive threshold has no ranking to work with.
    sat = {}
    top_gaps = []
    for fr in res["targets"]:
        for e in fr.values():
            for code, v in (e.get("intent_probs") or {}).items():
                if v >= 0.9999:
                    sat[code] = sat.get(code, 0) + 1
            vals = sorted((e.get("intent_probs") or {}).values(), reverse=True)
            if len(vals) >= 2:
                top_gaps.append(vals[0] - vals[1])
    print("saturated (score>=0.9999): %s" % (sat or "none"))
    if top_gaps:
        print("top-1 minus top-2 gap    : mean=%.3f min=%.3f"
              % (sum(top_gaps) / len(top_gaps), min(top_gaps)))
    print("conflict pairs      : %d from the ontology confusion matrix"
          % len(intent.conflict_pairs()))
    print("FEINT emitted       : %d  (must be 0)"
          % sum(1 for fr in res["targets"] for e in fr.values()
                for x in e["intent_set"] if x["code"] == "FEINT"))

    # ---- ablation report ----
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    rpt = os.path.join(ROOT, "intent_ablation_%s.md" % date)
    with open(rpt, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# 意图集合：规则基线 vs 模型\n\n")
        fh.write("案例 `%s`（战区 `%s`），样本 %d（训练 %d / 留出 %d），"
                 "特征 %d 维，意图 %d 类。\n\n"
                 % (case.case_id, theatre, len(samples), len(train), len(test),
                    F.feature_dim(), len(model.intents)))
        fh.write("> CES 监督目标由规则边际导出（见 `engine/intent_cf.py`），"
                 "因此本模型学的是**收缩策略**，不是更好的意图识别器。\n\n")
        fh.write("| 指标 | CES 目标（规则） | 模型 |\n|---|---|---|\n")
        for name, a, b in rows:
            fh.write("| %s | %s | %s |\n" % (name, a, b))
        fh.write("\n- no-reentry 事件：%d\n" % pred.get("no_reentry_events", -1))
        fh.write("- 嵌套违例：%d（投影开启时必须为 0）\n" % pred.get("nesting_violations", -1))
        fh.write("- 本体冲突对：%d\n" % len(intent.conflict_pairs()))
    print("ablation report : %s" % rpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))