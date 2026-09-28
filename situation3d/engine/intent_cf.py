"""Counterfactual equivalence set (CES) targets, and the temporal pairing that
gives the contraction objective something to act on.

What CES is
-----------
Per the paper (main.v1.5.tex, "Intent Identifiability under Partial Observation"):

    C*_t = { z : D_obs(O^z_{1:t}, O^{z*}_{1:t}) <= eps_o }

the set of intents whose counterfactual observation prefix is still
indistinguishable from what was observed, and the supervision target for the
compatibility score. It also fixes the insufficiency target, Definition u:

    u*_t = 1 - 1/|C*_t|        (0 once the set is a singleton)

How the target is built here, stated plainly
-------------------------------------------
The paper generates counterfactual trajectories by intervening on the latent
intent inside the MiaoSuan wargame and holding scenario conditions fixed. **This
system has no wargame engine**, so the target is built from the per-intent rule
margins that `intent_features` already computes:

    C*_t = { z : margin_z >= CES_MARGIN }

This is weaker than the paper's construction and the difference must not be
papered over:

* It makes the supervision **rule-derived**, so the model's compatibility score
  largely learns to agree with the rule layer rather than revealing intents the
  rules miss. A truth-based evaluation needs MiaoSuan `label_kind=commanded`
  annotations, which are not available here.
* What the model adds is therefore *not* new evidence. It is the **temporal
  contraction policy**: which candidate to drop, when, and how to keep the set
  nested over time, trained against an objective the rules do not have. That is
  the property the demo is meant to show, and it is a real property - it is just
  narrower than "better intent recognition".

Why `prev_*` fields exist
-------------------------
`L_ctr` (Eq. lctr) is a sum over consecutive steps: max(0, |C_hat_{t+1}| - |C_hat_t|).
With one sample per (target, frame), the "previous" set has to be carried in the
sample. `pair_sequences()` supplies it, in frame order, so the regulariser
actually sees a trajectory instead of independent rows. Without that pairing
L_ctr is identically zero and the contraction guarantee has nothing behind it.

Only the standard library is used.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from . import intent
from . import intent_features as F
from . import intent_model as M

# A rule margin at or above this counts as evidence for that intent in the CES
# target. Kept as a named constant rather than a literal so the calibration
# family the paper describes (a criterion-indexed set of CES variants) can be
# explored by moving it.
CES_MARGIN = 0.35

# Margin band treated as a tie: two intents this close must both stay in the set,
# because excluding one would be a commitment the evidence does not support.
TIE_BAND = 0.15


def ces_from_margins(margins: Dict[str, dict],
                     modelled: Sequence[str] = M.MODELLED_INTENTS
                     ) -> Tuple[List[float], Optional[str]]:
    """C*_t as a 0/1 vector over `modelled`, plus the status layer.

    Returns `(y, status)` where status is one of `None`, `AMBIGUOUS`,
    `BACKGROUND` or `UNKNOWN`, mirroring the ontology's own status layers.
    """
    scored = []
    for code in modelled:
        info = margins.get(code)
        if info is None:
            scored.append((code, 0.0))
            continue
        # Only primary-capable intents can be a standalone answer; the ontology
        # marks FIRE_SUPPORT/TARGET_GUIDANCE secondary-only.
        scored.append((code, float(info.get("margin") or 0.0)))

    live = [(c, m) for c, m in scored if m >= CES_MARGIN]
    y = [1.0 if dict(live).get(c, 0.0) >= CES_MARGIN else 0.0 for c in modelled]

    if not live:
        # Nothing fired. Whether that means "no intent is in play" or "the
        # evidence needed is missing" is a different statement, and the ontology
        # separates them, so the status carries it.
        return y, "BACKGROUND"

    live.sort(key=lambda cm: -cm[1])
    if len(live) >= 2 and (live[0][1] - live[1][1]) <= TIE_BAND:
        return y, "AMBIGUOUS"
    return y, None


def u_star(y: Sequence[float]) -> float:
    """Definition u: u*_t = 1 - 1/|C*_t|, the ground-truth insufficiency."""
    n = sum(1 for v in y if v > 0.5)
    if n <= 0:
        return 1.0
    return 1.0 - 1.0 / float(n)


def build_samples(frames: Sequence[dict], tracks: Dict[str, F.TrackView],
                  order: Sequence[str],
                  inference_frames: Optional[Sequence[dict]] = None,
                  hotspots: Sequence[dict] = (),
                  routes: Optional[Dict[str, Sequence]] = None,
                  road: Optional[object] = None,
                  stride: int = 1,
                  min_duration: int = 2) -> List[dict]:
    """One sample per (target, frame): features, CES target, insufficiency target.

    `min_duration` drops intents that fire for fewer than that many consecutive
    frames. The ontology declares a `minimum_duration` for all 22 intents (1-5
    frames) and this is where it is honoured: without it the CES target inherits
    the rule layer's chattering, and the model would be trained to reproduce the
    churn the objective exists to remove.
    """
    samples: List[dict] = []
    n_frames = len(frames)
    for i in range(0, n_frames, stride):
        inf = inference_frames[i] if inference_frames and i < len(inference_frames) else None
        for tid in order:
            tk = tracks.get(tid)
            if tk is None or tk.at(i) is None:
                continue
            feats = F.extract_target_features(
                tracks, order, i, tid, inference_frame=inf, hotspots=hotspots,
                route=(routes or {}).get(tid, ()), road=road)
            margins = F.rule_margins(tracks, order, i, tid)
            y, status = ces_from_margins(margins)
            samples.append({
                "tick": i,
                "target": tid,
                "x": F.vector(feats),
                "y": y,
                "u_star": u_star(y),
                "status": status,
                "n_in_set": int(sum(y)),
            })

    if min_duration > 1:
        samples = _hold_runs(samples, min_duration)
    return samples


def _hold_runs(samples: List[dict], min_duration: int) -> List[dict]:
    """Drop CES members that never persist `min_duration` consecutive frames.

    Applied per target, on the 0/1 vector, in tick order. This is the ontology's
    `minimum_duration` field doing real work rather than sitting in a JSON file.
    """
    by_target: Dict[str, List[dict]] = {}
    for s in samples:
        by_target.setdefault(s["target"], []).append(s)

    out: List[dict] = []
    for tid, rows in by_target.items():
        rows.sort(key=lambda r: r["tick"])
        n = len(rows)
        nz = len(rows[0]["y"])
        keep = [[False] * nz for _ in range(n)]
        for zi in range(nz):
            run_start = 0
            for k in range(n + 1):
                live = k < n and rows[k]["y"][zi] > 0.5
                if not live:
                    if k - run_start >= min_duration:
                        for j in range(run_start, k):
                            keep[j][zi] = True
                    run_start = k + 1
        for k, row in enumerate(rows):
            row["y"] = [1.0 if keep[k][zi] else 0.0 for zi in range(nz)]
            row["n_in_set"] = int(sum(row["y"]))
            row["u_star"] = u_star(row["y"])
            if row["n_in_set"] == 0:
                row["status"] = "BACKGROUND"
            elif row["n_in_set"] == 1:
                row["status"] = None
            else:
                row["status"] = "AMBIGUOUS"
            out.append(row)
    out.sort(key=lambda r: (r["tick"], r["target"]))
    return out


def pair_sequences(samples: Sequence[dict]) -> List[dict]:
    """Attach `prev_excluded` so L_ctr can see consecutive steps.

    Returns new dicts (the inputs are not mutated). The first row of each target's
    sequence gets `prev_excluded=None`, which the trainer skips: there is no t-1
    to compare against.
    """
    by_target: Dict[str, List[dict]] = {}
    for s in samples:
        by_target.setdefault(s["target"], []).append(s)

    out: List[dict] = []
    for tid, rows in by_target.items():
        rows = sorted(rows, key=lambda r: r["tick"])
        prev_excluded = None
        prev_status = None
        for row in rows:
            r = dict(row)
            r["prev_excluded"] = prev_excluded
            r["prev_status"] = prev_status
            out.append(r)
            # The soft count of the *predicted* set at t feeds the t+1 penalty.
            # The trainer recomputes it from the current parameters each epoch, so
            # this is only the initial bookkeeping value.
            prev_excluded = float(len(M.MODELLED_INTENTS) - row["n_in_set"])
            prev_status = row["status"]
    out.sort(key=lambda r: (r["tick"], r["target"]))
    return out


def split_by_target(samples: Sequence[dict], holdout: Sequence[str]
                    ) -> Tuple[List[dict], List[dict]]:
    """Train/holdout split **by target id**, never by frame.

    Splitting frames at random would put consecutive samples of the same target on
    both sides, and since consecutive frames are near-identical the holdout score
    would mostly measure memorisation. Target-disjoint splits at least ask whether
    the policy transfers to a platform it never saw.
    """
    hold = set(holdout)
    train = [s for s in samples if s["target"] not in hold]
    test = [s for s in samples if s["target"] in hold]
    return train, test