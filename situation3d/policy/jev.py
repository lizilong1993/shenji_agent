"""JEV-style decision policy adapter for the geo engine (research prototype).

JEV models are *decision classifiers*: given a textual state and a closed set
of options, they return a probability per option in one forward pass instead
of generating text.  This module treats them as a tactical candidate scorer:

    observation JSON -> candidate actions (fire / move / hold)
                     -> JEV CHOICE decision -> probability per candidate
                     -> chosen action dict for ``geoio.actions.apply_action``

The engine's physics/rules remain authoritative; JEV only picks among
already-valid candidates (the same idea as ``valid_actions``).  No model
weights are bundled here; ``HttpJevClient`` talks to an openjev/decision
server, and ``MockJevClient`` keeps tests and offline demos deterministic.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

DECISION_CHOICE = "choice"
DECISION_NOUL = "noul"
DECISION_SCORE = "score"


@dataclass
class DecisionQuestion:
    type: str
    instructions: str
    options: List[str] = field(default_factory=list)
    criteria: Dict[str, str] = field(default_factory=dict)


@dataclass
class DecisionRequest:
    state: str
    questions: List[DecisionQuestion]


@dataclass
class DecisionAnswer:
    probabilities: Dict[str, float]
    confidence: float = 0.0
    raw: Optional[dict] = None

    def best(self) -> Optional[str]:
        if not self.probabilities:
            return None
        return max(self.probabilities.items(), key=lambda kv: kv[1])[0]


class JevClient(Protocol):
    def decide(self, request: DecisionRequest) -> DecisionAnswer: ...


class HttpJevClient(object):
    """Minimal client for an openjev/typed-decision HTTP server.

    The server is expected to accept ``{"state": ..., "questions": {...}}``
    and return ``{"answers": {"decision": {"probabilities": {...}}}}`` (the
    openjev Decisions gateway shape).  No retries: a stalled model must not
    block the simulation loop.
    """

    def __init__(self, url: str, question_key: str = "decision", timeout_s: float = 5.0):
        self.url = url.rstrip("/")
        self.question_key = question_key
        self.timeout_s = float(timeout_s)

    def decide(self, request: DecisionRequest) -> DecisionAnswer:
        q = request.questions[0]
        payload = {
            "state": request.state,
            "questions": {self.question_key: {
                "type": q.type,
                "instructions": q.instructions,
                "options": list(q.options),
                "criteria": q.criteria or {str(i): o for i, o in enumerate(q.options)},
            }},
        }
        req = urllib.request.Request(
            self.url + "/v1/systemone",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        answer = (data.get("answers") or {}).get(self.question_key) or {}
        probs = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
        return DecisionAnswer(probabilities=probs,
                              confidence=float(answer.get("confidence") or 0.0),
                              raw=data)


class HfSpaceJevClient(object):
    """Calls the public openjev HF Space ``run_nli`` endpoint.

    This is for evaluation/demo only: it depends on a third-party Space and
    network access.  Each option is one NLI request; the Space returns HTML
    with an entailment percentage.
    """

    def __init__(self, base_url: str = "https://alexwortega-openjev.hf.space",
                 timeout_s: float = 30.0, poll_s: float = 0.6):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self.last_latency_s = None

    def _one(self, premise: str, hypothesis: str) -> float:
        import re
        import time
        payload = json.dumps({"data": [premise, hypothesis]}).encode("utf-8")
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    self.base_url + "/gradio_api/call/run_nli", data=payload,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    event_id = json.loads(resp.read().decode("utf-8"))["event_id"]
                deadline = time.time() + self.timeout_s
                while time.time() < deadline:
                    time.sleep(self.poll_s)
                    with urllib.request.urlopen(
                            self.base_url + "/gradio_api/call/run_nli/" + event_id,
                            timeout=self.timeout_s) as resp:
                        text = resp.read().decode("utf-8")
                    if "event: complete" not in text:
                        continue
                    match = re.search(r"entailment</span><span class='opt-btn-pct'>(\d+)%",
                                      text)
                    if match:
                        return float(match.group(1)) / 100.0
                    return 0.0
            except Exception:
                time.sleep(2.0)
        return 0.0

    def decide(self, request: DecisionRequest) -> DecisionAnswer:
        import time
        q = request.questions[0]
        premise = request.state + "\nQuestion: " + q.instructions
        t0 = time.perf_counter()
        probs = []
        for option in list(q.options)[:8]:
            probs.append(self._one(premise, "The chosen action is: " + option))
        self.last_latency_s = time.perf_counter() - t0
        total = sum(probs) or 1.0
        return DecisionAnswer(probabilities={opt: float(p / total)
                                             for opt, p in zip(q.options, probs)},
                              confidence=float(max(probs) / total))


class LocalOpenJevClient(object):
    """Runs an openjev/Qwen sequence-classification checkpoint on CPU/GPU.

    Each option is scored as an NLI hypothesis against the state; the
    entailment probabilities are normalised over options.  The model is
    loaded lazily so importing this module never requires torch.
    """

    def __init__(self, model_path: str, device: str = "cpu",
                 entailment_label: str = "entailment", max_options: int = 8):
        self.model_path = model_path
        self.device = device
        self.entailment_label = entailment_label
        self.max_options = max_options
        self.last_latency_s = None
        self._model = None
        self._tok = None
        self._ent_idx = None
        self._template = "Premise: {premise}\nHypothesis: {hypothesis}"

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self._model.to(self.device).eval()
        id2label = getattr(self._model.config, "id2label", {}) or {}
        inv = {str(v).lower(): int(k) for k, v in id2label.items()}
        self._ent_idx = inv.get(self.entailment_label, 1)
        self._template = getattr(self._model.config, "nli_template", self._template)

    def decide(self, request: DecisionRequest) -> DecisionAnswer:
        import time
        import torch
        self._load()
        q = request.questions[0]
        options = list(q.options)[:self.max_options]
        if not options:
            return DecisionAnswer(probabilities={})
        premise = request.state + "\nQuestion: " + q.instructions
        texts = [self._template.format(premise=premise,
                                       hypothesis="The chosen action is: " + opt)
                 for opt in options]
        t0 = time.perf_counter()
        probs = []
        with torch.no_grad():
            for text in texts:
                # One option per forward: CPU batching on this architecture is
                # slower than sequential short sequences.
                batch = self._tok(text, return_tensors="pt", truncation=True,
                                  max_length=256)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self._model(**batch).logits
                probs.append(float(torch.softmax(logits, dim=-1)[0, self._ent_idx]))
        self.last_latency_s = time.perf_counter() - t0
        total = sum(probs) or 1.0
        return DecisionAnswer(probabilities={opt: float(p / total)
                                             for opt, p in zip(options, probs)},
                              confidence=float(max(probs) / total))


class MockJevClient(object):
    """Deterministic stand-in: prefers options containing given keywords."""

    def __init__(self, preference: Sequence[str] = ("fire", "move", "hold")):
        self.preference = list(preference)

    def decide(self, request: DecisionRequest) -> DecisionAnswer:
        options = request.questions[0].options
        scores = {}
        for option in options:
            score = 0.01
            for i, key in enumerate(self.preference):
                if key in option.lower():
                    score += 1.0 / (i + 1)
            scores[option] = score
        total = sum(scores.values()) or 1.0
        return DecisionAnswer(probabilities={k: v / total for k, v in scores.items()},
                              confidence=0.5)


@dataclass
class Candidate:
    label: str
    action: dict


def _fmt_pos(pos: dict) -> str:
    return "%.5f,%.5f" % (float(pos.get("lon", 0.0)), float(pos.get("lat", 0.0)))


def _sanitize(text: str, limit: int = 40) -> str:
    """Strip control characters/newlines and cap length before it enters a prompt."""
    cleaned = "".join(ch if 32 <= ord(ch) < 127 or ord(ch) >= 160 else " " for ch in str(text))
    return " ".join(cleaned.split())[:limit]


def summarize_observation(observation: dict, seat: int) -> str:
    """Compact textual state for the decision model (detected entities only)."""
    lines = ["Time: %.1f s" % float(observation.get("time_s", 0.0))]
    for entity in observation.get("entities", []):
        side = "own" if entity.get("affiliation") == seat else "enemy"
        lines.append("%s %s %s at %s hp=%s speed=%.1f" % (
            side, _sanitize(entity.get("unit_kind", "unit")),
            _sanitize(entity.get("name", "?")),
            _fmt_pos(entity.get("pos") or {}), entity.get("blood", "?"),
            float(entity.get("speed_ms", 0.0))))
    for obj in observation.get("objectives", []):
        lines.append("objective %s owner=%s value=%s" % (
            _sanitize(obj.get("name", "?")), obj.get("owner", "?"),
            obj.get("value", "?")))
    return "\n".join(lines)


def summarize_entity(observation: dict, seat: int, uid: int) -> str:
    """Compact per-entity state (own unit + up to 3 nearest enemies)."""
    own = None
    enemies = []
    for entity in observation.get("entities", []):
        if entity.get("id") == uid:
            own = entity
        elif entity.get("affiliation") != seat:
            enemies.append(entity)
    lines = ["Time: %.1f s" % float(observation.get("time_s", 0.0))]
    if own is not None:
        lines.append("own unit %s %s at %s hp=%s speed=%.1f" % (
            own.get("id"), _sanitize(own.get("unit_kind", "unit")),
            _fmt_pos(own.get("pos") or {}), own.get("blood", "?"),
            float(own.get("speed_ms", 0.0))))
    # Keep the prompt short: only 3 contacts, nearest first.
    enemies.sort(key=lambda e: _fmt_pos(e.get("pos") or {}))
    for entity in enemies[:2]:
        lines.append("enemy unit %s %s at %s hp=%s" % (
            entity.get("id"), _sanitize(entity.get("unit_kind", "unit")),
            _fmt_pos(entity.get("pos") or {}), entity.get("blood", "?")))
    for obj in observation.get("objectives", [])[:2]:
        lines.append("objective owner=%s" % (obj.get("owner", "?"),))
    return "\n".join(lines)


def build_candidates(observation: dict, seat: int,
                     max_per_entity: int = 6,
                     max_move_candidates: int = 2) -> List[Candidate]:
    """Flatten the observation's per-entity valid actions into candidates."""
    candidates: List[Candidate] = []
    for entity in observation.get("entities", []):
        if entity.get("affiliation") != seat:
            continue
        uid = entity.get("id")
        actions = entity.get("actions") or []
        fires = [a for a in actions if a.get("type") == "fire"]
        moves = [a for a in actions if a.get("type") == "move_to"][:max_move_candidates]
        stops = [a for a in actions if a.get("type") == "stop"]
        picked = (fires + moves + stops)[:max_per_entity]
        for action in picked:
            if action["type"] == "fire":
                weapon = action.get("weapon")
                label = "unit %s fire target %s with %s" % (
                    uid, action.get("target"), weapon)
                act = {"type": "fire", "entity": uid, "target": action["target"]}
                if weapon is not None:
                    act["weapon"] = weapon
            elif action["type"] == "move_to":
                label = "unit %s move to %s" % (uid, _fmt_pos(action.get("target") or {}))
                act = {"type": "move_to", "entity": uid,
                       "target": action.get("target")}
            else:
                label = "unit %s hold position" % uid
                act = {"type": "stop", "entity": uid}
            candidates.append(Candidate(label=label, action=act))
    # Guarantee unique labels (the decision API returns label-keyed probabilities).
    seen = {}
    unique = []
    for c in candidates:
        count = seen.get(c.label, 0)
        seen[c.label] = count + 1
        unique.append(Candidate(label=c.label if count == 0 else "%s #%d" % (c.label, count + 1),
                                action=c.action))
    return unique


class JevPolicy(object):
    def __init__(self, client: Optional[JevClient] = None,
                 instructions: str = "Choose the best tactical action for this side."):
        self.client = client
        self.instructions = instructions

    def _fallback(self, candidates: List[Candidate]) -> dict:
        for c in candidates:
            if c.action.get("type") == "fire":
                return c.action
        return candidates[0].action

    def choose(self, observation: dict, seat: int,
               candidates: Optional[List[Candidate]] = None,
               state: Optional[str] = None) -> Optional[dict]:
        candidates = candidates if candidates is not None else build_candidates(observation, seat)
        if not candidates:
            return None
        if self.client is None:
            return self._fallback(candidates)
        state = state or summarize_observation(observation, seat)
        labels = [c.label for c in candidates]
        request = DecisionRequest(
            state=state,
            questions=[DecisionQuestion(
                type=DECISION_CHOICE, instructions=self.instructions,
                options=labels,
                criteria={str(i): label for i, label in enumerate(labels)})])
        try:
            answer = self.client.decide(request)
            probs = dict(answer.probabilities or {})
            # Accept index-keyed responses ("0", "1", ...) from simple servers.
            if probs and not any(k in labels for k in probs):
                mapped = {}
                for key, value in probs.items():
                    try:
                        mapped[labels[int(key)]] = float(value)
                    except (ValueError, IndexError, TypeError):
                        continue
                probs = mapped
            valid = {k: v for k, v in probs.items() if k in labels and isinstance(v, (int, float))}
            if not valid:
                return self._fallback(candidates)
            best = max(valid.items(), key=lambda kv: kv[1])[0]
        except Exception:
            # A stalled or failing model must never block the simulation.
            return self._fallback(candidates)
        for c in candidates:
            if c.label == best:
                return c.action
        return self._fallback(candidates)

    def step(self, world, seat: int) -> Optional[dict]:
        """Convenience: build an observation, choose, and apply the action."""
        from geoio.actions import apply_action
        from geoio.observation import observation
        action = self.choose(observation(world, seat=seat), seat)
        if action is not None:
            apply_action(world, action)
        return action
