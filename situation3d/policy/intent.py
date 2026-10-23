"""Intent recognition and behavior prediction with JEV decision models.

Both are advisory layers: they read the same seat-filtered observation the
engine already produces, keep a short kinematic/event history, and ask a JEV
client to score a *closed set* of intent or next-behavior labels.  The output
is written to ``World.intel`` / the observation JSON for commanders and the
UI; rules and adjudication never consume it as ground truth.

Why JEV fits:
* intent recognition = CHOICE over intent labels (assault/defend/recon/...);
* behavior prediction = CHOICE over coarse next actions or sector candidates,
  not coordinate regression (the planner remains responsible for geometry).

The models are not trained on this domain; treat probabilities as advisory
until evaluated against labelled replays.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional

from policy.jev import (
    DECISION_CHOICE,
    DecisionQuestion,
    DecisionRequest,
    JevClient,
    _fmt_pos,
    _sanitize,
)

INTENT_LABELS = (
    "assault", "defend", "recon", "feint", "retreat",
    "reinforce", "fire_support", "resupply", "hold",
)

BEHAVIOR_LABELS = (
    "advance", "hold", "retreat", "attack", "reposition", "withdraw",
)


def _entity_summary(entity: dict, events: Optional[Iterable[dict]] = None) -> str:
    parts = ["t=%.1f" % float(entity.get("time_s", 0.0)),
             "%s %s" % (entity.get("unit_kind", "unit"), entity.get("id")),
             "pos=%s" % _fmt_pos(entity.get("pos") or {}),
             "spd=%.1f" % float(entity.get("speed_ms", 0.0)),
             "hp=%s" % entity.get("blood", "?"),
             "dom=%s" % _sanitize(entity.get("domain", "land"))]
    if events:
        tags = []
        for ev in events:
            kind = str(ev.get("type") or "")
            actors = ev.get("actors") or []
            if entity.get("id") in actors:
                tags.append(kind)
        if tags:
            parts.append("events=" + ",".join(tags[-4:]))
    return " ".join(parts)


@dataclass
class IntentResult:
    uid: int
    label: str
    probabilities: Dict[str, float]
    confidence: float


@dataclass
class BehaviorResult:
    uid: int
    label: str
    probabilities: Dict[str, float]
    confidence: float


class IntentEstimator(object):
    def __init__(self, client: JevClient, window: int = 6,
                 labels: Optional[Iterable[str]] = None):
        self.client = client
        self.window = int(window)
        self.labels = tuple(labels or INTENT_LABELS)
        self.history: Dict[int, Deque[str]] = {}
        self.last_latency_s = None

    def update(self, observation: dict, events: Optional[List[dict]] = None) -> None:
        """Append one observation window per visible entity."""
        for entity in observation.get("entities", []):
            uid = int(entity.get("id"))
            entity = dict(entity)
            entity["time_s"] = observation.get("time_s", 0.0)
            buf = self.history.setdefault(uid, deque(maxlen=self.window))
            buf.append(_entity_summary(entity, events))

    def _state(self, observation: dict, uid: int, current: dict) -> str:
        lines = ["Entity history (oldest to newest):"]
        lines.extend(self.history.get(uid, ()))
        entity = dict(current)
        entity["time_s"] = observation.get("time_s", 0.0)
        lines.append("Current: " + _entity_summary(entity))
        objectives = observation.get("objectives") or []
        if objectives:
            lines.append("Objectives: " + " ".join(
                "owner=%s" % o.get("owner", "?") for o in objectives[:3]))
        return "\n".join(lines)

    def estimate(self, observation: dict, seat: int) -> Dict[int, IntentResult]:
        results: Dict[int, IntentResult] = {}
        for entity in observation.get("entities", []):
            if entity.get("affiliation") == seat:
                continue
            uid = int(entity.get("id"))
            request = DecisionRequest(
                state=self._state(observation, uid, entity),
                questions=[DecisionQuestion(
                    type=DECISION_CHOICE,
                    instructions="Classify the enemy force's operational intent.",
                    options=list(self.labels),
                    criteria={str(i): label
                              for i, label in enumerate(self.labels)})])
            answer = self.client.decide(request)
            probs = {k: float(v) for k, v in (answer.probabilities or {}).items()
                     if k in self.labels}
            if not probs:
                continue
            total = sum(probs.values()) or 1.0
            probs = {k: v / total for k, v in probs.items()}
            label = max(probs.items(), key=lambda kv: kv[1])[0]
            results[uid] = IntentResult(uid, label, probs,
                                        confidence=float(probs[label]))
        self.last_latency_s = getattr(self.client, "last_latency_s", None)
        return results

    def format(self, results: Dict[int, IntentResult]) -> dict:
        return {str(uid): {"label": r.label,
                           "confidence": round(r.confidence, 3),
                           "probabilities": {k: round(v, 3)
                                             for k, v in r.probabilities.items()}}
                for uid, r in results.items()}


class BehaviorPredictor(object):
    def __init__(self, client: JevClient,
                 labels: Optional[Iterable[str]] = None):
        self.client = client
        self.labels = tuple(labels or BEHAVIOR_LABELS)

    def predict(self, observation: dict, seat: int) -> Dict[int, BehaviorResult]:
        results: Dict[int, BehaviorResult] = {}
        for entity in observation.get("entities", []):
            if entity.get("affiliation") == seat:
                continue
            uid = int(entity.get("id"))
            request = DecisionRequest(
                state="Current: " + _entity_summary(entity),
                questions=[DecisionQuestion(
                    type=DECISION_CHOICE,
                    instructions="Predict the enemy unit's next coarse behavior.",
                    options=list(self.labels),
                    criteria={str(i): label
                              for i, label in enumerate(self.labels)})])
            answer = self.client.decide(request)
            probs = {k: float(v) for k, v in (answer.probabilities or {}).items()
                     if k in self.labels}
            if not probs:
                continue
            total = sum(probs.values()) or 1.0
            probs = {k: v / total for k, v in probs.items()}
            label = max(probs.items(), key=lambda kv: kv[1])[0]
            results[uid] = BehaviorResult(uid, label, probs,
                                          confidence=float(probs[label]))
        return results

    def format(self, results: Dict[int, BehaviorResult]) -> dict:
        return {str(uid): {"label": r.label,
                           "confidence": round(r.confidence, 3),
                           "probabilities": {k: round(v, 3)
                                             for k, v in r.probabilities.items()}}
                for uid, r in results.items()}
