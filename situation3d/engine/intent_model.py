"""ShenJi-style set-valued intent reasoning, standard library only.

This implements the *decision* half of the paper's ShenJi reasoner, taken from
`papers/paper/manuscript/iclr2027/main.v1.5.tex` section "ShenJi: Evidence-Aware
Set-Valued Reasoning" (Eq. delta, Eq. lctr, Eq. total) and Algorithm 1 in
`appendix.tex`.

What is faithful, and what is not
---------------------------------
The container has **python 3.8, no pip, no numpy and no torch**, so the paper's
encoders (entity MLP, causal Transformer, GRU evidence memory, CIGR message
passing) cannot be reproduced here and are not attempted. What *is* implemented
exactly as written:

* evidence-insufficiency head      u_t, a learned linear read-out + logistic
* compatibility head               s_t(z), a learned linear read-out + logistic
* adaptive threshold               delta_t = sigma(a (1 - u_t) + b)     [Eq. delta]
* contraction regulariser          L_ctr over soft cardinality, tau = 0.05 [Eq. lctr]
* full objective                   L_CES + l1 L_rank + l2 L_ctr + lu L_u [Eq. total]
* inference                        C_hat = {z : s_t(z) >= delta_t}, argmax if
                                   empty, then the monotonic no-reentry
                                   projection                          [Alg. 1]

So the **decision layer and the contraction mechanism are faithful; the encoder is
a simplified linear read-out over the fixed feature window** produced by
`intent_features.py`. The trained model is a logistic read-out, not a neural
encoder, and nothing here should be read as reproducing the paper's network.

One deliberate deviation, flagged rather than hidden
---------------------------------------------------
Eq. lctr defines the soft cardinality as `sum_z sigma((s_t(z) - delta_t) / tau)`.
Read literally that term *rises* as compatibility improves, so a
"cardinality" penalty built on it would punish supported hypotheses and reverse
the intended contraction. The quantity it means to soften is the number of
intents *excluded*, which is `sum_z sigma((delta_t - s_t(z)) / tau)`: an intent
far below the threshold contributes ~1. We implement that, and keep the literal
form available as `soft_cardinality_literal()` so the discrepancy is auditable
instead of silently patched.

Only the standard library is used.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

from . import intent_features as F

MODEL_SCHEMA = "situation3d.intent_model.v1"

# Loss weights (Eq. total). The paper does not publish its own values for these
# in the main text; they are stated here so a reviewer can see what was used and
# re-tune without touching code.
LAMBDA_RANK = 1.0
LAMBDA_CTR = 1.0
# Weight on the evidence-insufficiency term. `u` trains toward
# `u* = 1 - 1/|C*|`, which for the observed 3-4 candidate sets is only ~0.69-0.75,
# so with a weight of 1.0 the head sits mid-range and cannot raise delta enough to
# widen the prediction sets. Measured, the model produced 76.9% singletons against
# the CES target's 7.4%, i.e. it commits far earlier than the evidence supports.
# The term is up-weighted so the contraction schedule is driven by `u` rather than
# overridden by the compatibility term's tendency to saturate individual scores.
LAMBDA_U = 3.0
RANK_MARGIN = 0.1
TAU = 0.05          # soft-cardinality temperature, as specified
# L2 on the compatibility weights. Raised from 1e-4 after measuring the trained
# read-out: scores saturated at exactly 1.0 for several intents at once
# (ASSEMBLE 1.0, DISPERSE 0.9998, PURSUIT 0.736), which destroys the ranking the
# adaptive threshold depends on and made every target collapse to one intent.
# Logistic read-outs on unnormalised features saturate easily; the penalty keeps
# the weights small enough that the scores stay discriminative.
L2 = 1e-2
# Global gradient-norm clip for the full-batch update. See the note in `train`.
GRAD_CLIP = 5.0

# Intents the model may ever emit. FEINT is excluded outright: the ontology marks
# it auto_label_grade C with requires_plan_tag, i.e. behaviour alone can never
# establish it. TARGET_GUIDANCE is excluded for the same reason the rule layer
# excludes it (needs an action-type-9 event this system does not have).
# `intent.derive` owns the authoritative capability table; these are mirrored so
# the model cannot emit something the rules refused to judge.
MODELLED_INTENTS = [
    "ASSAULT", "FLANK", "ENCIRCLEMENT", "PURSUIT", "HOLD", "RECON",
    "SURVEILLANCE", "REINFORCE", "ESCORT", "FIRE_SUPPORT", "REDEPLOY",
    "ASSEMBLE", "DISPERSE", "FORTIFY", "RETREAT", "EVADE",
]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, p))
    return math.log(p / (1.0 - p))


class IntentModel(object):
    """Linear compatibility + insufficiency read-outs with the adaptive threshold."""

    def __init__(self, n_features: int, intents: Optional[Sequence[str]] = None,
                 seed: int = 20260926):
        self.n_features = int(n_features)
        self.intents = list(intents or MODELLED_INTENTS)
        self.seed = int(seed)
        rng = random.Random(self.seed)
        # Small random init; deterministic given the seed, which the self-test
        # relies on for byte-identical retraining.
        s = 0.01
        self.W = [[rng.uniform(-s, s) for _ in range(self.n_features)]
                  for _ in self.intents]
        self.b = [0.0 for _ in self.intents]
        self.wu = [0.0 for _ in range(self.n_features)]
        self.bu = 0.0
        # Eq. delta scalars. a starts positive so a higher insufficiency widens
        # the set from the first step rather than having to discover the sign.
        self.a = 1.0
        self.b_delta = 0.0
        # Temperature for *reported* probabilities only.
        #
        # It must not enter `compat()`, because the adaptive threshold delta_t was
        # trained against the untempered scores: tempering before the comparison
        # would move every score relative to a threshold that no longer matches.
        # So decisions keep using the raw score and the UI reads a separated
        # distribution. Fitted by `fit_temperature`.
        self.temperature = 1.0
        self.meta: Dict[str, object] = {}
        # Feature standardisation (mean/std per feature), computed from the training
        # set by `fit_scaler`. Kept as part of the model because inference must
        # apply exactly the same transform the weights were fitted under; deriving
        # it from whatever data happens to be at hand would silently shift every
        # score.
        self.mean: Optional[List[float]] = None
        self.std: Optional[List[float]] = None

    def fit_scaler(self, samples: Sequence[dict]) -> None:
        """Per-feature mean/std over the training rows.

        The features are raw physical quantities on wildly different scales (km,
        knots, 0/1 flags, road class codes). Without standardisation the logistic
        read-out saturates: several intents scored exactly 1.0 at once, which
        leaves the adaptive threshold nothing to rank.
        """
        if not samples:
            raise ValueError("no samples to fit the scaler on")
        n = len(samples)
        k = self.n_features
        mean = [0.0] * k
        for s in samples:
            x = s["x"]
            for i in range(k):
                mean[i] += x[i]
        mean = [v / n for v in mean]
        var = [0.0] * k
        for s in samples:
            x = s["x"]
            for i in range(k):
                d = x[i] - mean[i]
                var[i] += d * d
        # A constant feature has zero variance; dividing by it would emit inf/NaN.
        std = [math.sqrt(v / n) if v > 1e-12 else 1.0 for v in var]
        self.mean, self.std = mean, std

    def transform(self, x: Sequence[float]) -> List[float]:
        """Standardise, then clamp to +-6 sigma.

        The clamp matters. Several features are near-constant for long stretches
        (a platform holding a steady course), so their standard deviation is tiny
        and a small physical change yields a large z-score. Unclamped, those
        outliers drove the logistic read-out into saturation: measured across the
        generated datasets, 39915 of 17292 Taiwan target-frames carried a top-k
        probability of exactly 1.0 and India had more saturated entries than
        targets (7228 vs 7205), with a mean top1-top2 gap of 0.038. A saturated
        score carries no ranking information, and ranking is exactly what the
        adaptive threshold consumes.
        """
        if self.mean is None or self.std is None:
            return list(x)
        out = []
        for v, m, s in zip(x, self.mean, self.std):
            z = (v - m) / s
            out.append(6.0 if z > 6.0 else (-6.0 if z < -6.0 else z))
        return out

    # -- forward ---------------------------------------------------------

    def prob(self, s: float) -> float:
        """Reported probability for a raw score, after temperature scaling.

        Maps the score back through its logit, divides by T, and re-applies the
        logistic. T > 1 flattens an over-confident distribution; the ranking is
        unchanged, so the decision that consumed the raw score still stands.
        """
        t = self.temperature if self.temperature and self.temperature > 1e-6 else 1.0
        if t == 1.0:
            return s
        return _sigmoid(_logit(s) / t)

    def fit_temperature(self, samples: Sequence[dict],
                        grid: Optional[Sequence[float]] = None) -> float:
        """Pick the temperature minimising per-intent log loss on `samples`.

        Reported confidence should be checked against held-out data rather than
        asserted. Measured before this: 39915 of 17292 Taiwan target-frames had a
        top-k probability of exactly 1.0, so "100%" was shown for several
        competing intents at once and meant nothing.

        The search is restricted to T >= 1, i.e. flattening only. Unconstrained,
        the held-out log-loss optimum was **T = 0.50** - it *sharpened* the
        distribution - while the model already under-reports ambiguity badly
        (76.9% singleton predictions against 7.4% in the CES target). Minimising
        average log loss is not the same objective as not overstating confidence
        to an operator; where they disagree, this display errs toward showing the
        ambiguity, so sharpening is not available.
        """
        if not samples:
            return self.temperature
        grid = grid or [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        best_t, best_ll = 1.0, None
        for t in grid:
            if t < 1.0:
                continue
            self.temperature = t
            ll = 0.0
            n = 0
            for smp in samples:
                s = self.compat(smp["x"])
                for zi, y in enumerate(smp["y"]):
                    p = self.prob(s[zi])
                    p = min(1.0 - 1e-9, max(1e-9, p))
                    ll += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                    n += 1
            ll = ll / max(1, n)
            if best_ll is None or ll < best_ll:
                best_t, best_ll = t, ll
        self.temperature = best_t
        return best_t

    def compat(self, x: Sequence[float]) -> List[float]:
        """s_t(z) for every modelled intent."""
        v = self.transform(x)
        return [_sigmoid(sum(w * q for w, q in zip(row, v)) + bi)
                for row, bi in zip(self.W, self.b)]

    def insufficiency(self, x: Sequence[float]) -> float:
        """u_t in [0, 1]: how little the evidence separates the candidates."""
        v = self.transform(x)
        return _sigmoid(sum(w * q for w, q in zip(self.wu, v)) + self.bu)

    def delta(self, u: float) -> float:
        """Eq. delta: delta_t = sigma(a (1 - u_t) + b)."""
        return _sigmoid(self.a * (1.0 - u) + self.b_delta)

    @staticmethod
    def soft_cardinality_literal(s: Sequence[float], delta: float,
                                 tau: float = TAU) -> float:
        """Literal Eq. lctr form, kept for auditability. See the module docstring."""
        return sum(_sigmoid((si - delta) / tau) for si in s)

    @staticmethod
    def soft_cardinality(s: Sequence[float], delta: float,
                         tau: float = TAU) -> float:
        """Softened *excluded* count; the decreasing form Eq. lctr intends."""
        return sum(_sigmoid((delta - si) / tau) for si in s)

    def hard_set(self, s: Sequence[float], delta: float) -> List[int]:
        """Alg. 1: {z : s_t(z) >= delta_t}, falling back to argmax when empty."""
        idx = [i for i, si in enumerate(s) if si >= delta]
        if idx:
            return idx
        best = max(range(len(s)), key=lambda i: s[i])
        return [best]

    # -- training --------------------------------------------------------

    def train(self, samples: Sequence[dict], epochs: int = 60, lr: float = 0.05,
              verbose: bool = False) -> dict:
        """Fit on (x, y, u_star) samples with the Eq. total objective.

        `samples` are dicts with keys `x` (feature vector), `y` (list of 0/1 per
        modelled intent) and `u_star` (target insufficiency 1 - 1/|C*|).

        Full-batch gradient descent. The data set is a few thousand rows with
        ~64 features and 16 intents, so this converges in seconds and, more
        importantly, is deterministic: no shuffling, no threads, so training twice
        with the same inputs produces byte-identical weights (asserted by the
        self-test, because a non-reproducible model makes every reported metric
        unreproducible too).
        """
        if not samples:
            raise ValueError("no training samples")
        n = len(samples)
        nz = len(self.intents)
        history = []
        for ep in range(epochs):
            gW = [[0.0] * self.n_features for _ in range(nz)]
            gb = [0.0] * nz
            gwu = [0.0] * self.n_features
            gbu = 0.0
            ga = 0.0
            gbd = 0.0
            tot_ces = tot_rank = tot_ctr = tot_u = 0.0

            for smp in samples:
                # Standardise once per row: the gradients below assume the same
                # transformed input the forward pass used.
                x = self.transform(smp["x"])
                y = smp["y"]
                u_star = smp["u_star"]
                s = self.compat(x)
                u = self.insufficiency(x)
                delta = self.delta(u)

                # dL/du and dL/ddelta accumulate across the three terms that
                # touch delta (via u) and the set boundary.
                du = 0.0
                ddelta = 0.0

                # ---- L_CES: per-intent binary cross-entropy ----
                for zi in range(nz):
                    si = s[zi]
                    yi = y[zi]
                    tot_ces += -(yi * math.log(max(1e-12, si))
                                 + (1 - yi) * math.log(max(1e-12, 1 - si)))
                    ds = (si - yi)                     # d/dlogit of BCE
                    for k in range(self.n_features):
                        gW[zi][k] += ds * x[k]
                    gb[zi] += ds

                # ---- L_u: BCE of u against u_star ----
                # `LAMBDA_U` must be applied here. It was previously declared and
                # used only in the reported loss total, so the gradient carried
                # weight 1.0 regardless of the constant: the u head was nominally
                # tunable but actually fixed. Eq. total writes the term as
                # `lambda_u L_u`, so an unwired weight silently changes the
                # objective the model is trained on.
                tot_u += -(u_star * math.log(max(1e-12, u))
                           + (1 - u_star) * math.log(max(1e-12, 1 - u)))
                du += LAMBDA_U * (u - u_star)

                # ---- L_rank: max over wrong intents, only where CES is a singleton
                pos = [i for i in range(nz) if y[i] > 0.5]
                if len(pos) == 1:
                    zi = pos[0]
                    wrong = [i for i in range(nz) if i != zi]
                    j = max(wrong, key=lambda i: s[i])
                    margin = RANK_MARGIN - s[zi] + s[j]
                    if margin > 0:
                        tot_rank += margin
                        for k in range(self.n_features):
                            gW[zi][k] += -LAMBDA_RANK * x[k]
                            gW[j][k] += LAMBDA_RANK * x[k]

                # ---- L_ctr: penalise growth of the excluded count over time ----
                # Within a single sample the count itself is what we shrink: the
                # paper's summations are over consecutive steps t, and with one
                # sample per (target, frame) the temporal pairing is applied by
                # `train_sequence`, which passes deltas in order.
                if smp.get("prev_excluded") is not None:
                    cur = self.soft_cardinality(s, delta)
                    growth = cur - smp["prev_excluded"]
                    if growth > 0:
                        tot_ctr += growth
                        # d(soft count)/d(delta) then chain to u
                        dsoft_ddelta = sum(
                            _sigmoid((delta - si) / TAU) * (1 - _sigmoid((delta - si) / TAU))
                            for si in s) / TAU
                        ddelta += LAMBDA_CTR * dsoft_ddelta
                        for zi in range(nz):
                            dsd = -(_sigmoid((delta - s[zi]) / TAU)
                                    * (1 - _sigmoid((delta - s[zi]) / TAU))) / TAU
                            for k in range(self.n_features):
                                gW[zi][k] += LAMBDA_CTR * dsd * x[k]

                # ---- chain delta -> u for every term that touched delta ----
                ddelta_du = delta * (1.0 - delta) * (-self.a)
                du += ddelta * ddelta_du
                # and the direct delta parameters
                ddelta_da = delta * (1.0 - delta) * (1.0 - u)
                ddelta_db = delta * (1.0 - delta)
                ga += ddelta * ddelta_da
                gbd += ddelta * ddelta_db

                # ---- u -> wu ----
                du_dlogit = u * (1.0 - u)
                for k in range(self.n_features):
                    gwu[k] += du * du_dlogit * x[k]
                gbu += du * du_dlogit

            # average, add L2, step
            inv = 1.0 / n
            # Global gradient-norm clip. Without it the CES term (16 intents x 108
            # features) diverges immediately: an observed run at lr=0.05 went
            # ces 14.5 -> 55.5 and never recovered, with the u term pinned at 8.64.
            # Clipping bounds the per-epoch step, which matters more here than
            # hand-tuning lr because training has to be reproducible from the
            # recorded hyper-parameters alone.
            gn = 0.0
            for zi in range(nz):
                for k in range(self.n_features):
                    gn += (gW[zi][k] * inv) ** 2
            for k in range(self.n_features):
                gn += (gwu[k] * inv) ** 2
            gn = math.sqrt(gn)
            scale = 1.0 if gn <= GRAD_CLIP else GRAD_CLIP / gn

            for zi in range(nz):
                for k in range(self.n_features):
                    g = gW[zi][k] * inv + L2 * self.W[zi][k]
                    self.W[zi][k] -= lr * scale * g
                self.b[zi] -= lr * scale * gb[zi] * inv
            for k in range(self.n_features):
                self.wu[k] -= lr * scale * (gwu[k] * inv + L2 * self.wu[k])
            self.bu -= lr * scale * gbu * inv
            self.a -= lr * scale * ga * inv
            self.b_delta -= lr * scale * gbd * inv
            # Keep `a` positive: Eq. delta's monotonicity ("delta_t decreasing in
            # u_t") holds only for a > 0, and the proposition's contraction
            # guarantee depends on it.
            if self.a < 0.05:
                self.a = 0.05
            history.append({
                "epoch": ep,
                "ces": round(tot_ces * inv, 5),
                "rank": round(tot_rank * inv, 5),
                "ctr": round(tot_ctr * inv, 5),
                "u": round(tot_u * inv, 5),
            })
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                h = history[-1]
                print("    ep %3d  ces=%.4f rank=%.4f ctr=%.4f u=%.4f"
                      % (ep, h["ces"], h["rank"], h["ctr"], h["u"]))
        self.meta["history"] = history
        return {"history": history, "final": history[-1] if history else None}

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema": MODEL_SCHEMA,
            "feature_schema": F.FEATURE_SCHEMA_VERSION,
            "feature_names": F.feature_names(),
            "intents": self.intents,
            "n_features": self.n_features,
            "seed": self.seed,
            "temperature": self.temperature,
            "mean": self.mean,
            "std": self.std,
            "a": self.a,
            "b_delta": self.b_delta,
            "bu": self.bu,
            "wu": self.wu,
            "b": self.b,
            "W": self.W,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(doc: dict) -> "IntentModel":
        if doc.get("schema") != MODEL_SCHEMA:
            raise ValueError("unexpected model schema: %r" % doc.get("schema"))
        # A feature-schema mismatch means the weights are positional against a
        # different layout; loading anyway would silently reinterpret every
        # weight, so refuse.
        if doc.get("feature_schema") != F.FEATURE_SCHEMA_VERSION:
            raise ValueError(
                "model was trained on feature schema %r but this build produces %r; "
                "retrain with tools/train_intent_model.py"
                % (doc.get("feature_schema"), F.FEATURE_SCHEMA_VERSION))
        if doc.get("feature_names") != F.feature_names():
            raise ValueError("model feature_names do not match the current layout")
        m = IntentModel(int(doc["n_features"]), doc.get("intents"), int(doc.get("seed", 0)))
        m.temperature = float(doc.get("temperature", 1.0))
        m.mean = [float(v) for v in doc["mean"]] if doc.get("mean") else None
        m.std = [float(v) for v in doc["std"]] if doc.get("std") else None
        m.a = float(doc["a"])
        m.b_delta = float(doc["b_delta"])
        m.bu = float(doc["bu"])
        m.wu = [float(v) for v in doc["wu"]]
        m.b = [float(v) for v in doc["b"]]
        m.W = [[float(v) for v in row] for row in doc["W"]]
        m.meta = doc.get("meta") or {}
        return m


def save(model: IntentModel, path: str) -> int:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(model.to_dict(), fh, ensure_ascii=False,
                  separators=(",", ":"), sort_keys=True)
    os.replace(tmp, path)
    return os.path.getsize(path)


def load(path: str) -> Optional[IntentModel]:
    """Load a model, or None when absent.

    Returning None rather than raising is deliberate: the caller falls back to the
    rule-based `intent.derive` and marks the output `intent_source="rule"`, so a
    missing model degrades to the previous behaviour instead of breaking the
    dataset build.
    """
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return IntentModel.from_dict(json.load(fh))


def model_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "intent_model.json")