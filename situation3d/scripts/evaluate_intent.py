#!/usr/bin/env python3
"""Compare rule-based intent inference with JEV zero-shot intent recognition.

The synthetic windows are generated from scripted trajectories (assault,
defend, recon, retreat), so ground truth is exact.  The rule baseline uses
kinematics/events only; JEV uses the same observation history.  This is a
controlled comparison, not a claim about real-world accuracy.

Usage:
    python3 scripts/evaluate_intent.py --out /tmp/intent_eval.json
    PYTHONPATH=/tmp/jevlibs:$PYTHONPATH python3 scripts/evaluate_intent.py \
        --model /tmp/jev_model/qwen3.5-0.8b-nli-v2s-long --jev-limit 8 \
        --out /tmp/intent_eval_jev.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from policy.intent import IntentEstimator  # noqa: E402
from policy.intent_rules import RuleIntentClassifier  # noqa: E402
from policy.jev import HfSpaceJevClient, LocalOpenJevClient, MockJevClient  # noqa: E402

LABELS = ("assault", "defend", "recon", "retreat")
BEHAVIORS = ("advance", "hold", "withdraw", "attack", "reposition")
BEHAVIOR_TRUTH = {"assault": "attack", "defend": "hold",
                  "recon": "reposition", "retreat": "withdraw"}
BEHAVIOR_OPTIONS = {
    "advance": "The enemy will advance.",
    "hold": "The enemy will hold position.",
    "withdraw": "The enemy will withdraw.",
    "attack": "The enemy will attack.",
    "reposition": "The enemy will reposition.",
}
RED_LON, RED_LAT = 119.90, 23.10


def _entity(uid, affiliation, lon, lat, speed, kind="vehicle"):
    return {"id": uid, "affiliation": affiliation, "unit_kind": kind,
            "name": "unit-%s" % uid, "domain": "land", "blood": 4,
            "pos": {"lon": lon, "lat": lat}, "speed_ms": speed}


def make_window(label: str, sample: int) -> Tuple[List[dict], List[dict]]:
    frames = []
    events = []
    lon, lat = 119.95 + sample * 0.0005, 23.10
    step = 0.0008
    objective = None
    for i in range(6):
        t = i * 10.0
        speed = 0.0
        if label == "assault":
            lon -= step
            speed = 8.0
        elif label == "retreat":
            lon += step
            speed = 8.0
        elif label == "recon":
            lat += step
            speed = 6.0
        elif label == "reinforce":
            lat += step
            speed = 7.0
        if label == "defend":
            objective = {"pos": {"lon": lon, "lat": lat}, "owner": 1,
                         "name": "OBJ"}
        elif label == "reinforce":
            objective = {"pos": {"lon": 119.95, "lat": 23.15}, "owner": 1,
                         "name": "OBJ"}
        obs = {"time_s": t,
               "entities": [_entity(1, 0, RED_LON, RED_LAT, 0.0),
                            _entity(2, 1, lon, lat, speed)],
               "objectives": [objective] if objective else []}
        frames.append(obs)
        if label == "assault" and i >= 4:
            events.append({"type": "direct_fire", "actors": [2]})
    return frames, events


def macro_f1(truth: List[str], pred: List[str], labels) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(truth, pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(truth, pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(truth, pred) if t == label and p != label)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall)
                      if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_rule(windows) -> dict:
    clf = RuleIntentClassifier(LABELS)
    truth, pred = [], []
    for label, frames, events in windows:
        truth.append(label)
        pred.append(clf.classify(frames, events, target_uid=2, reference_uid=1))
    acc = sum(1 for t, p in zip(truth, pred) if t == p) / max(1, len(truth))
    return {"samples": len(truth), "accuracy": round(acc, 3),
            "macro_f1": round(macro_f1(truth, pred, LABELS), 3),
            "predictions": pred}


OPTION_TEXT = {
    "assault": "The enemy is conducting an assault.",
    "defend": "The enemy is defending.",
    "recon": "The enemy is conducting reconnaissance.",
    "retreat": "The enemy is retreating.",
}


def semantic_state(features: dict) -> str:
    return ("Over the last 50 s, the enemy unit has a closing speed of %.1f m/s, "
            "lateral speed of %.1f m/s, current speed %.1f m/s, fired %d times, "
            "distance to friendly %.0f m, and is near its own objective: %s."
            % (features["closing"], features.get("lateral", 0.0),
               features["speed"], int(features["fired"]),
               features["objective_dist"], bool(features["objective_owned"])))


def rule_behavior(features: dict) -> str:
    if features["fired"]:
        return "attack"
    if features["speed"] < 0.6:
        return "hold"
    if features["closing"] > 0.5:
        return "advance"
    if features["closing"] < -0.5 and features.get("lateral", 0.0) < 0.8:
        return "withdraw"
    if features.get("lateral", 0.0) >= 0.5:
        return "reposition"
    return "hold"


def evaluate_behavior_rule(windows) -> dict:
    clf = RuleIntentClassifier(LABELS)
    truth, pred = [], []
    for label, frames, events in windows:
        f = clf.features(frames, events, 2, 1)
        truth.append(BEHAVIOR_TRUTH[label])
        pred.append(rule_behavior(f))
    acc = sum(1 for t, p in zip(truth, pred) if t == p) / max(1, len(truth))
    return {"samples": len(truth), "accuracy": round(acc, 3),
            "macro_f1": round(macro_f1(truth, pred, BEHAVIORS), 3),
            "predictions": pred}


def evaluate_behavior_jev(windows, client) -> dict:
    from policy.jev import DecisionQuestion, DecisionRequest
    clf = RuleIntentClassifier(LABELS)
    truth, pred, latencies = [], [], []
    inverse = {v: k for k, v in BEHAVIOR_OPTIONS.items()}
    for label, frames, events in windows:
        f = clf.features(frames, events, 2, 1)
        options = [BEHAVIOR_OPTIONS[x] for x in BEHAVIORS]
        request = DecisionRequest(
            state=semantic_state(f),
            questions=[DecisionQuestion(
                type="choice",
                instructions="Predict the enemy unit's next coarse behavior.",
                options=options, criteria={str(i): o for i, o in enumerate(options)})])
        answer = client.decide(request)
        probs = {k: float(v) for k, v in (answer.probabilities or {}).items()}
        best = max(probs.items(), key=lambda kv: kv[1])[0] if probs else None
        truth.append(BEHAVIOR_TRUTH[label])
        pred.append(inverse.get(best, "hold"))
        latency = getattr(client, "last_latency_s", None)
        if latency:
            latencies.append(latency)
    acc = sum(1 for t, p in zip(truth, pred) if t == p) / max(1, len(truth))
    return {"samples": len(truth), "accuracy": round(acc, 3),
            "macro_f1": round(macro_f1(truth, pred, BEHAVIORS), 3),
            "predictions": pred,
            "latency_ms_avg": round(1000.0 * sum(latencies) / len(latencies), 1)
            if latencies else None}


def evaluate_jev_semantic(windows, client) -> dict:
    from policy.jev import DecisionQuestion, DecisionRequest
    clf = RuleIntentClassifier(LABELS)
    truth, pred, latencies = [], [], []
    for label, frames, events in windows:
        f = clf.features(frames, events, target_uid=2, reference_uid=1)
        options = [OPTION_TEXT[x] for x in LABELS]
        request = DecisionRequest(
            state=semantic_state(f),
            questions=[DecisionQuestion(
                type="choice",
                instructions="Classify the enemy force's operational intent.",
                options=options, criteria={str(i): o for i, o in enumerate(options)})])
        answer = client.decide(request)
        probs = {k: float(v) for k, v in (answer.probabilities or {}).items()}
        best = max(probs.items(), key=lambda kv: kv[1])[0] if probs else None
        inverse = {v: k for k, v in OPTION_TEXT.items()}
        truth.append(label)
        pred.append(inverse.get(best, "hold"))
        latency = getattr(client, "last_latency_s", None)
        if latency:
            latencies.append(latency)
    acc = sum(1 for t, p in zip(truth, pred) if t == p) / max(1, len(truth))
    return {"samples": len(truth), "accuracy": round(acc, 3),
            "macro_f1": round(macro_f1(truth, pred, LABELS), 3),
            "predictions": pred,
            "latency_ms_avg": round(1000.0 * sum(latencies) / len(latencies), 1)
            if latencies else None,
            "latency_ms_max": round(1000.0 * max(latencies), 1) if latencies else None}


def evaluate_jev(windows, client) -> dict:
    truth, pred, latencies = [], [], []
    estimator = IntentEstimator(client, window=6, labels=LABELS)
    for label, frames, events in windows:
        estimator.history.clear()
        for obs in frames:
            estimator.update(obs, events)
        results = estimator.estimate(frames[-1], seat=0)
        truth.append(label)
        result = results.get(2)
        pred.append(result.label if result else "hold")
        latency = getattr(client, "last_latency_s", None)
        if latency:
            latencies.append(latency)
    acc = sum(1 for t, p in zip(truth, pred) if t == p) / max(1, len(truth))
    return {"samples": len(truth), "accuracy": round(acc, 3),
            "macro_f1": round(macro_f1(truth, pred, LABELS), 3),
            "predictions": pred,
            "latency_ms_avg": round(1000.0 * sum(latencies) / len(latencies), 1)
            if latencies else None,
            "latency_ms_max": round(1000.0 * max(latencies), 1) if latencies else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--space", action="store_true",
                    help="use the public openjev HF Space instead of a local model")
    ap.add_argument("--samples-per-class", type=int, default=6)
    ap.add_argument("--task", choices=("both", "intent", "behavior"), default="both")
    ap.add_argument("--jev-limit", type=int, default=8,
                    help="JEV window count (CPU is slow); rule is also scored on this subset")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    all_windows = []
    for label in LABELS:
        for sample in range(args.samples_per_class):
            frames, events = make_window(label, sample)
            all_windows.append((label, frames, events))

    rule_all = evaluate_rule(all_windows)
    subset = []
    per_class = max(1, args.jev_limit // len(LABELS))
    for label in LABELS:
        subset.extend([w for w in all_windows if w[0] == label][:per_class])
    rule_subset = evaluate_rule(subset)

    if args.space:
        client = HfSpaceJevClient()
    elif args.model:
        client = LocalOpenJevClient(args.model, device=args.device)
    else:
        client = MockJevClient(preference=("assault", "recon", "retreat", "defend"))
    jev = evaluate_jev_semantic(subset, client) if args.task in ("both", "intent") else None
    payload = {
        "labels": list(LABELS),
        "behaviors": list(BEHAVIORS),
        "model": "hf-space" if args.space else (args.model or "mock"),
        "rule_all": rule_all,
        "rule_subset": rule_subset,
        "jev_subset": jev,
        "behavior_rule_subset": evaluate_behavior_rule(subset),
        "behavior_jev_subset": (evaluate_behavior_jev(subset, client)
                                if args.task in ("both", "behavior") else None),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
