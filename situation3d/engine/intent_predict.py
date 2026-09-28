"""Inference: Algorithm 1 of the ShenJi paper, applied per target over a case.

    C_hat_t <- {z : s_t(z) >= delta_t};  if empty, keep argmax_z s_t(z)
    C_hat_t <- monotonic no-reentry projection      (once removed, never re-admitted)

The projection is the part that makes the output sets **nested**, which is the
property the demo exists to show: the candidate set only ever shrinks along a
trajectory, so an operator watching a platform cannot see an intent it already
discarded come back. The paper's Proposition (contraction) gets nesting from a
stationary point of the training objective; Algorithm 1 additionally enforces it
at output level, because the score ordering can change between steps and nesting
must hold regardless.

Falling back honestly
---------------------
When no model file is present, `predict_frames` is not called at all: the caller
uses `intent.derive()` and labels the output `intent_source="rule"`. There is no
path here that invents a probability. A model whose feature schema does not match
the current build refuses to load (see `intent_model.load`), and that also ends in
the rule fallback rather than a silent reinterpretation of the weights.

Only the standard library is used.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from . import intent
from . import intent_features as F
from . import intent_model as M


def intent_names(model: M.IntentModel, doc: Optional[dict] = None) -> Dict[str, dict]:
    """code -> {name_zh, parent_code, primary_vs_secondary} for labelling output."""
    out: Dict[str, dict] = {}
    for code in model.intents:
        meta = intent.intent_by_code(code) or {}
        out[code] = {
            "name_zh": meta.get("name_zh") or code,
            "parent_code": meta.get("parent_code") or "",
            "primary_vs_secondary": meta.get("primary_vs_secondary") or "",
        }
    return out


def resolve_conflicts(codes: Sequence[str]) -> List[str]:
    """Drop intents the ontology declares mutually exclusive, keeping the stronger.

    `codes` must already be ordered strongest-first. The ontology ships an
    explicit `confusion_matrix` naming pairs that are routinely conflated and how
    to tell them apart (HOLD vs FORTIFY, ASSAULT vs SEIZE, RECON vs SURVEILLANCE,
    RETREAT vs EVADE, FLANK vs BREAKTHROUGH). Applying it here is what stops the
    output from asserting "garrison" and "fortify" at the same time - which the
    rule layer did in 1435 frames of the demo data while the ontology declared the
    two exclusive.
    """
    conflicts = intent.conflict_pairs()
    kept: List[str] = []
    for c in codes:
        if any((c, k) in conflicts or (k, c) in conflicts for k in kept):
            continue
        kept.append(c)
    return kept


def _drop_conflicts(codes: List[str], scores: Sequence[float],
                    model: M.IntentModel, banned: set,
                    ever_dropped: set) -> List[str]:
    """Remove one side of every ontology-declared conflict, keeping the stronger.

    `codes` is expected strongest-first. The dropped code is also barred, so the
    removal is permanent and the eligibility set keeps shrinking - a conflict that
    is resolved once cannot reappear later from a previous frame's set.
    """
    conflicts = intent.conflict_pairs()
    kept: List[str] = []
    for c in codes:
        clash = None
        for k in kept:
            if (c, k) in conflicts or (k, c) in conflicts:
                clash = k
                break
        if clash is None:
            kept.append(c)
            continue
        # Keep whichever scores higher; on a tie keep the one already held.
        if scores[model.intents.index(c)] > scores[model.intents.index(clash)]:
            j = model.intents.index(clash)
            banned.add(j)
            ever_dropped.add(j)
            kept = [x for x in kept if x != clash]
            kept.append(c)
        else:
            j = model.intents.index(c)
            banned.add(j)
            ever_dropped.add(j)
    return kept


def predict_frames(frames: Sequence[dict], tracks: Dict[str, F.TrackView],
                   order: Sequence[str], model: M.IntentModel,
                   inference_frames: Optional[Sequence[dict]] = None,
                   hotspots: Sequence[dict] = (),
                   routes: Optional[Dict[str, Sequence]] = None,
                   road: Optional[object] = None,
                   apply_no_reentry: bool = True,
                   top_k: int = 3,
                   ticks: Optional[Sequence[int]] = None,
                   reentry_grace: int = 2) -> dict:
    """Per-frame, per-target prediction sets.

    Returns `{"targets": [{tid: entry}], "stats": {...}}` where each entry carries
    the set, its size, the status layer, the top-k scored candidates and how many
    no-reentry events occurred.

    `reentry_grace` is how many consecutive frames a candidate must remain below
    the threshold before it is permanently barred. Algorithm 1 as written bars on a
    single frame, and measured on this data that was destructive: raw prediction
    sets of 2-6 intents collapsed to a near-constant singleton (93.4% singletons,
    mean set size 1.066 against a CES target of 1.813 carrying 62.6% ambiguity),
    because one low-scoring frame barred an intent for the remaining ~120 ticks.
    The grace preserves the property that matters - **once genuinely dropped, a
    candidate never returns** - without letting a single noisy frame decide. Pass
    `reentry_grace=1` for the literal single-frame behaviour.
    """
    meta = intent_names(model)
    secondary_only = {c for c, m in meta.items()
                      if str(m.get("primary_vs_secondary", "")).startswith("secondary")}
    n_frames = len(frames)
    # `ticks` restricts evaluation to a subset of frames. Feature extraction walks
    # road and neighbour geometry per (target, frame), so scoring all 1441 frames
    # of a case just to fill a comparison table costs more than the training did;
    # the caller passes the ticks it actually trained on. Nesting is still judged
    # over the returned sequence in frame order, so the projection property is
    # measured the same way either way.
    tick_list = list(range(n_frames)) if ticks is None else [t for t in ticks if 0 <= t < n_frames]

    per_frame: List[Dict[str, dict]] = []
    # excluded[target] = set of intent indices already removed; never re-admitted.
    excluded: Dict[str, set] = {tid: set() for tid in order}
    # per target: intent index -> consecutive frames below the threshold
    strikes: Dict[str, Dict[int, int]] = {tid: {} for tid in order}
    # last non-empty set per target, so an empty set carries forward rather than
    # resurrecting a dropped intent (see the fallback in the projection)
    last_set: Dict[str, List[str]] = {}
    # candidates ever admitted per target; the pool only widens, never narrows, so
    # it cannot be the reason a prediction set grows
    considered: Dict[str, set] = {}
    # codes barred at any point per target; `ever \ ever_dropped` only shrinks
    considered_dropped: Dict[str, set] = {}
    no_reentry_events = 0
    size_series: Dict[str, List[int]] = {tid: [] for tid in order}

    for i in tick_list:
        inf = inference_frames[i] if inference_frames and i < len(inference_frames) else None
        entries: Dict[str, dict] = {}
        for tid in order:
            tk = tracks.get(tid)
            if tk is None or tk.at(i) is None:
                continue
            feats = F.extract_target_features(
                tracks, order, i, tid, inference_frame=inf, hotspots=hotspots,
                route=(routes or {}).get(tid, ()), road=road)
            x = F.vector(feats)
            s = model.compat(x)
            u = model.insufficiency(x)
            delta = model.delta(u)

            idx = model.hard_set(s, delta)
            codes = [model.intents[j] for j in idx]
            codes.sort(key=lambda c: -s[model.intents.index(c)])
            codes = resolve_conflicts(codes)

            if apply_no_reentry:
                banned = excluded[tid]
                # Monotone candidate pool. The projection as specified blocks a
                # candidate from *re-entering*, which prevents resurrection but
                # does not by itself make the sets nested: measured on this data,
                # 16 frames grew because a candidate that had never appeared
                # before crossed the threshold later and was admitted. The set is
                # made monotone by tracking which candidates have ever been
                # considered, and never letting that pool widen - a platform read
                # as "not assaulting" cannot later be read as assaulting for the
                # first time here, which is the behaviour the nesting guarantee is
                # for.
                # Union of every frame so far: the pool only widens, so the
                # intersection below only ever narrows.
                ever = considered.setdefault(tid, set())
                for j in range(len(s)):
                    if s[j] >= delta:
                        ever.add(j)
                ever_dropped = considered_dropped.setdefault(tid, set())
                prev = last_set.get(tid) or []

                # The output is restricted to E = (prev ∪ passing) \ (banned ∪
                # ever_dropped), which is provably non-increasing: `prev` is the
                # previous output, `passing` is recomputed each frame, `banned` and
                # `ever_dropped` only grow as codes are barred. Nothing outside E
                # can be emitted, so no barred code can ever come back.
                #
                # Earlier versions filtered only against `banned` and then let the
                # empty-set fallback draw on `ever`; because the size cap bans
                # overflow codes, that fallback re-emitted exactly the codes the
                # cap had just barred. Measured against the shipped model: 48
                # resurrections at grace=1.
                passing = [c for c in codes
                           if model.intents.index(c) not in banned
                           and model.intents.index(c) not in ever_dropped]
                prev_allowed = [c for c in prev
                                if model.intents.index(c) not in banned
                                and model.intents.index(c) not in ever_dropped]
                eligible = list(prev_allowed)
                for c in sorted(passing, key=lambda c: -s[model.intents.index(c)]):
                    if c not in eligible:
                        eligible.append(c)

                cap = len(prev) if prev else len(eligible)
                eligible.sort(key=lambda c: -s[model.intents.index(c)])
                if len(eligible) > cap:
                    for c in eligible[cap:]:
                        j = model.intents.index(c)
                        banned.add(j)
                        ever_dropped.add(j)
                    eligible = eligible[:cap]
                codes = eligible

                # Bar a candidate only after it stays below the threshold for
                # `reentry_grace` consecutive frames; counting strikes rather than
                # acting on one frame keeps a noisy tick from permanently removing
                # a live hypothesis.
                for j in range(len(s)):
                    if s[j] >= delta:
                        strikes[tid].pop(j, None)
                        continue
                    if j in banned:
                        continue
                    strikes[tid][j] = strikes[tid].get(j, 0) + 1
                    if strikes[tid][j] >= max(1, reentry_grace):
                        banned.add(j)
                        ever_dropped.add(j)

                # Re-resolve ontology conflicts after the merge. `prev` and the
                # current frame were each conflict-free on their own, but their
                # union need not be: a candidate that arrives while a conflicting
                # one is already held produces a pair the ontology declares
                # mutually exclusive. Dropping the weaker one here is still
                # monotone, because the drop is recorded in `banned`, so E shrinks.
                # (Measured: 5 co-occurrences appeared as soon as eligibility
                # became `prev ∪ passing`.)
                codes = _drop_conflicts(codes, s, model, banned, ever_dropped)
                last_set[tid] = list(codes)

            # primary / secondary split, mirroring the rule layer's contract
            prim = [c for c in codes if c not in secondary_only]
            sec = [c for c in codes if c in secondary_only]
            ordered = prim + sec

            sized = sorted(range(len(s)), key=lambda j: -s[j])[:top_k]
            # Reported probabilities are temperature-calibrated; the scores used
            # for the threshold decision above are deliberately left raw.
            probs = {model.intents[j]: round(model.prob(s[j]), 4) for j in sized}

            if not ordered:
                status = "BACKGROUND"
                primary = None
            else:
                primary = ordered[0]
                if len(ordered) >= 2:
                    gap = s[model.intents.index(ordered[0])] - s[model.intents.index(ordered[1])]
                    status = "AMBIGUOUS" if gap <= intent.AMBIGUITY_MARGIN * s[
                        model.intents.index(ordered[0])] else None
                else:
                    status = None

            entries[tid] = {
                "intent_set": [
                    {
                        "code": c,
                        "score": round(model.prob(s[model.intents.index(c)]), 4),
                        "role": "secondary" if c in secondary_only else "primary",
                        "name_zh": meta.get(c, {}).get("name_zh") or c,
                    }
                    for c in ordered
                ],
                "intent_probs": probs,
                "intent_status": status,
                "primary_intent": primary,
                "intent_set_size": len(ordered),
                "insufficiency": round(u, 4),
                "delta": round(delta, 4),
                "intent_source": "model",
                "model_version": model.meta.get("version") or "unversioned",
            }
            size_series[tid].append(len(ordered))
        per_frame.append(entries)

    # Nesting audit: with the projection on, the set size must never increase.
    violations = 0
    for tid, series in size_series.items():
        for a, b in zip(series, series[1:]):
            if b > a:
                violations += 1

    stats = {
        "frames": n_frames,
        "targets": len(order),
        "no_reentry_events": no_reentry_events,
        "nesting_violations": violations,
        "apply_no_reentry": bool(apply_no_reentry),
        "reentry_grace": int(reentry_grace),
        "model_version": model.meta.get("version") or "unversioned",
        "feature_schema": F.FEATURE_SCHEMA_VERSION,
        "final_set_sizes": {t: (s[-1] if s else 0) for t, s in size_series.items()},
    }
    return {"targets": per_frame, "stats": stats}


def fill_carried(inference_frames: Sequence[dict], result: dict,
                 order: Sequence[str]) -> int:
    """Carry each prediction forward over the frames that were not scored.

    Predictions are computed on a strided tick set (see `INTENT_PREDICT_STRIDE`)
    because per-frame scoring is too slow to build a dataset with. Frames between
    two scored ticks would otherwise have no `intent_set` at all, so the panel
    would blank out between samples.

    Every carried entry is flagged `intent_carried: True` and keeps the tick it was
    actually computed on (`intent_eval_tick`), so a viewer can tell a fresh
    judgement from a repeat. Presenting carried values as fresh would be the same
    class of error as inventing a probability.
    """
    per_frame = result.get("targets") or []
    scored = {i: entries for i, entries in enumerate(per_frame) if entries}
    if not scored:
        return 0
    n = 0
    last: Dict[str, dict] = {}
    for i, frame in enumerate(inference_frames):
        tgts = frame.get("targets") or {}
        fresh = scored.get(i)
        if fresh:
            for tid, entry in fresh.items():
                if tid in tgts:
                    tgts[tid].update(entry)
                    m = dict(entry)
                    m["intent_carried"] = False
                    m["intent_eval_tick"] = i
                    last[tid] = m
                    n += 1
            continue
        for tid in order:
            if tid not in tgts or tid not in last:
                continue
            carried = dict(last[tid])
            carried["intent_carried"] = True
            tgts[tid].update(carried)
            n += 1
    return n


def merge_into_frames(inference_frames: Sequence[dict], result: dict) -> int:
    """Copy predictions onto `inference.infer()`'s per-frame target entries.

    Existing fields are preserved; `intent_label` in particular is left alone so
    the brief generator and older clients keep working. Returns the number of
    target-frames annotated.
    """
    per_frame = result.get("targets") or []
    n = 0
    for i, frame in enumerate(inference_frames):
        if i >= len(per_frame):
            break
        entries = per_frame[i] or {}
        tgts = frame.get("targets") or {}
        for tid, entry in entries.items():
            if tid in tgts:
                tgts[tid].update(entry)
                n += 1
    return n