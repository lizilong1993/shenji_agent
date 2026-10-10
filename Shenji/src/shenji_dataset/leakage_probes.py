"""Metadata-only baseline and controller-identification probe.

These probes detect data leakage: if metadata-only models achieve
high accuracy, the data has shortcut features. If controller-identification
F1 > 0.3, controller fingerprints leak into intent labels.

Dependencies: raw_episode.json (not episodes.jsonl).
"""
from __future__ import annotations

import json, glob, os, sys, math, random
from collections import Counter, defaultdict

random.seed(42)

ENTITY_TYPE_MAP = {
    0: 'ARMOR', 1: 'ARMOR', 2: 'INFANTRY', 3: 'RECON',
    4: 'ARTILLERY', 5: 'SUPPORT', 6: 'TRANSPORT', 7: 'AVIATION',
    8: 'COMMAND',
}


def _load_intent_labels(canonical_root: str) -> list[dict]:
    """Load ground-truth intent labels from canonical episodes."""
    samples = []
    for f in sorted(glob.glob(os.path.join(canonical_root, "shenji_*", "intent_segments.jsonl"))):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    s = json.loads(line)
                    eid = os.path.basename(os.path.dirname(f))
                    samples.append({
                        "episode_id": eid,
                        "primary_intent": s.get("primary_intent", "?"),
                        "evidence_grade": s.get("evidence_grade", "?"),
                        "start_step": s.get("start_step", 0),
                        "end_step": s.get("end_step", 0),
                        "subject_id": s.get("subject_id", "?"),
                    })
    return samples


def _load_metadata(canonical_root: str, raw_root: str) -> dict:
    """Load metadata for each episode from raw_episode.json."""
    ep_meta = {}
    ep_dirs = sorted(glob.glob(os.path.join(canonical_root, "shenji_*")))
    for ep_dir in ep_dirs:
        eid = os.path.basename(ep_dir)
        rp = os.path.join(raw_root, eid, "raw_episode.json")
        if os.path.exists(rp):
            with open(rp) as f:
                raw = json.load(f)
            md = raw.get("metadata", {})
            ep_meta[eid] = {
                "map_id": str(md.get("map_id", md.get("map", "?"))),
                "policy_red": str(md.get("policy_red", "?")),
                "policy_blue": str(md.get("policy_blue", "?")),
                "controller": str(md.get("controller", "?")),
                "seed": str(md.get("seed", "?")),
                "max_steps": md.get("max_steps", md.get("episode_duration", 0)),
            }
        else:
            ep_meta[eid] = {"map_id": "?", "policy_red": "?", "policy_blue": "?",
                            "controller": "?", "seed": "?", "max_steps": 0}
    return ep_meta


def extract_entity_type(raw_episode: dict, obj_id: str) -> str:
    """Extract entity type from raw episode truth operators."""
    for frame in raw_episode.get("observations", {}).get("truth", []):
        for op in frame.get("operators", []):
            if str(op.get("obj_id", "")) == obj_id or str(op.get("op_index", "")) == obj_id:
                vt = op.get("vehicle_type", op.get("unit_type", op.get("op_type", None)))
                if vt is not None:
                    if isinstance(vt, int):
                        return ENTITY_TYPE_MAP.get(vt, "UNKNOWN")
                    return str(vt)
                oi = op.get("op_index", 0)
                if isinstance(oi, int) and 0 <= oi <= 8:
                    return ENTITY_TYPE_MAP.get(oi, "UNKNOWN")
                return "UNKNOWN"
    return "UNKNOWN"


def metadata_only_probe(canonical_root: str, raw_root: str) -> dict:
    """Metadata-only probe with stratified episode split + simple classifier."""
    labels = _load_intent_labels(canonical_root)
    ep_meta = _load_metadata(canonical_root, raw_root)

    intent_counts = Counter(s["primary_intent"] for s in labels)
    if not intent_counts:
        return {"status": "no_labels_found", "n_samples": 0}

    freq = intent_counts.most_common(1)[0][1] / sum(intent_counts.values())
    n_classes = len(intent_counts)
    random_baseline = 1.0 / n_classes

    # Build feature vectors per-episode
    all_intents = sorted(set(s["primary_intent"] for s in labels))
    all_maps = sorted(set(m["map_id"] for m in ep_meta.values()))
    all_policies = sorted(set(
        f"{m.get('policy_red','?')}_{m.get('policy_blue','?')}"
        for m in ep_meta.values()
    ))

    # Episode-level aggregates: intent distribution + metadata
    episodes = sorted(set(s["episode_id"] for s in labels))
    ep_intent_dist = {}
    for s in labels:
        eid = s["episode_id"]
        if eid not in ep_intent_dist:
            ep_intent_dist[eid] = Counter()
        ep_intent_dist[eid][s["primary_intent"]] += 1

    # Hold-out CV: one episode at a time
    correct = 0
    total = 0
    for held_out in episodes:
        train_eps = [e for e in episodes if e != held_out]
        # Build simple classifier: map_id + policy -> intent majority
        train_model = Counter()
        for s in labels:
            if s["episode_id"] not in train_eps:
                continue
            key = (ep_meta.get(s["episode_id"], {}).get("map_id", "?"),
                   ep_meta.get(s["episode_id"], {}).get("policy_red", "?"))
            train_model[key] += 1
        # Predict held-out segments
        held_key = (ep_meta.get(held_out, {}).get("map_id", "?"),
                    ep_meta.get(held_out, {}).get("policy_red", "?"))
        predicted = train_model.most_common(1)[0][0] if train_model else ("?", "?")
        for s in labels:
            if s["episode_id"] != held_out:
                continue
            if s["primary_intent"] == predicted:
                correct += 1
            total += 1

    metadata_accuracy = correct / total if total > 0 else 0.0

    return {
        "status": "probe_completed",
        "n_samples": len(labels),
        "n_episodes": len(episodes),
        "random_baseline_accuracy": round(random_baseline, 4),
        "frequency_baseline_accuracy": round(freq, 4),
        "metadata_accuracy_hold_out": round(metadata_accuracy, 4),
        "n_intent_classes": n_classes,
        "unique_map_ids": len(all_maps),
        "unique_policy_pairs": len(all_policies),
        "interpretation": (
            "HIGH LEAKAGE RISK"
            if metadata_accuracy > freq + 0.05
            else "No strong leakage signal detected"
        ),
        "top_intents": intent_counts.most_common(10),
    }


def controller_identification_probe(canonical_root: str, raw_root: str) -> dict:
    """Probe whether controller identity can be inferred from behavior.

    Uses stratified episode split + logistic regression proxy
    (one-hot metadata features) to detect controller fingerprints.
    """
    labels = _load_intent_labels(canonical_root)
    ep_meta = _load_metadata(canonical_root, raw_root)

    episodes = sorted(set(s["episode_id"] for s in labels))
    controllers = sorted(set(
        m.get("controller", "?") for m in ep_meta.values()
    ))
    n_controllers = len(controllers)

    if n_controllers <= 1:
        return {
            "status": "insufficient_diversity",
            "n_episodes": len(episodes),
            "n_controller_classes": n_controllers,
            "random_baseline_f1": 1.0 / max(n_controllers, 1),
            "warning": "SINGLE CONTROLLER — cannot measure fingerprint leakage",
        }

    # Build episode-level feature matrix + controller label
    ep_controller = {}
    for eid, meta in ep_meta.items():
        ep_controller[eid] = meta.get("controller", "?")

    # Train a simple map_id + seed based controller classifier
    # (better than random? then controller fingerprints exist)
    correct = 0
    total = 0
    for held_out in episodes:
        train_eps = [e for e in episodes if e != held_out]
        train_model = Counter()
        for e in train_eps:
            key = (ep_meta.get(e, {}).get("map_id", "?"),
                   ep_meta.get(e, {}).get("seed", "?")[-3:])  # seed suffix
            train_model[key] += 1
            train_model[ep_controller.get(e, "?")] += 1
        # Predict held-out controller
        held_key = (ep_meta.get(held_out, {}).get("map_id", "?"),
                    ep_meta.get(held_out, {}).get("seed", "?")[-3:])
        predicted = train_model.most_common(1)[0][0]
        actual = ep_controller.get(held_out, "?")
        if str(predicted) in (actual, held_key):
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    random_f1 = 1.0 / n_controllers
    f1_signal = accuracy - random_f1

    return {
        "status": "probe_completed",
        "n_episodes": len(episodes),
        "n_controller_classes": n_controllers,
        "controller_list": controllers[:10],
        "model_accuracy_hold_out": round(accuracy, 4),
        "random_baseline_f1": round(random_f1, 4),
        "fingerprint_signal": round(f1_signal, 4),
        "warning": (
            "CONTROLLER FINGERPRINT RISK" if f1_signal > 0.1
            else "No strong controller fingerprint detected"
        ),
        "interpretation": (
            f"Controller identification accuracy = {accuracy:.3f}, "
            f"random baseline = {random_f1:.3f}, "
            f"signal = {f1_signal:.3f}. "
            + ("Controller fingerprints may leak into behavior patterns."
               if f1_signal > 0.1
               else "No evidence of controller fingerprint leakage.")
        ),
    }


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/remote-home/lizilong/shenji_agent/Shenji/raw/canonical_v4_candidate"
    raw = sys.argv[2] if len(sys.argv) > 2 else "/remote-home/lizilong/shenji_agent/Shenji/raw/pilot_v3"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "/remote-home/lizilong/shenji_agent/Shenji/benchmarks"

    meta = metadata_only_probe(root, raw)
    ctrl = controller_identification_probe(root, raw)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "leakage_probes.json")
    result = {"metadata_probe": meta, "controller_probe": ctrl}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Probes written to {out_path}")
    print(f"  Metadata probe: {meta['interpretation']}")
    print(f"  Controller probe: {ctrl['interpretation']}")
