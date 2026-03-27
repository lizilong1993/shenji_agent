from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .features import action_feature_vector, find_operator, operator_to_feature

try:
    import torch

    from .models import MultiHorizonIntentionModel, TacticalScorer

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    MultiHorizonIntentionModel = None  # type: ignore[assignment]
    TacticalScorer = None  # type: ignore[assignment]


_DIRECTION_STEPS = [
    (0, 0),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
]


def _hex_to_axial(pos: int) -> Tuple[int, int]:
    row, col = divmod(pos, 100)
    q = col - (row - (row & 1)) // 2
    r = row
    return q, r


def _axial_to_hex(q: int, r: int) -> int:
    col = q + (r - (r & 1)) // 2
    return r * 100 + col


class LearnedPolicy:
    def __init__(self) -> None:
        self.history_len = 8
        self.faction: Optional[int] = None
        self.policy_profile = "challenger"
        self.intent_model = None
        self.tactical_model = None
        self.enemy_histories: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.history_len))
        self.loaded_from: Optional[Path] = None

    def configure(self, faction: int, artifact_dir: Optional[str] = None, policy_profile: str = "challenger") -> None:
        self.faction = faction
        self.policy_profile = policy_profile or "challenger"
        self.enemy_histories = defaultdict(lambda: deque(maxlen=self.history_len))
        if not TORCH_AVAILABLE:
            return
        candidate_dirs = []
        if artifact_dir:
            candidate_dirs.append(Path(artifact_dir))
        candidate_dirs.append(Path(__file__).resolve().parent / "weights")
        for candidate_dir in candidate_dirs:
            manifest_path = candidate_dir / "model_manifest.json"
            if not manifest_path.exists():
                continue
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.history_len = int(payload.get("history_len", self.history_len))
            self.enemy_histories = defaultdict(lambda: deque(maxlen=self.history_len))
            intent_path = candidate_dir / payload.get("intent_model", "")
            tactical_path = candidate_dir / payload.get("tactical_model", "")
            if intent_path.exists():
                model_state = torch.load(intent_path, map_location="cpu")
                self.intent_model = MultiHorizonIntentionModel(
                    input_dim=int(model_state.get("feature_dim", 108)),
                    hidden_dim=128,
                    horizons=int(model_state.get("horizons", 5)),
                    num_classes=int(model_state.get("num_classes", 7)),
                )
                self.intent_model.load_state_dict(model_state["model_state"])
                self.intent_model.eval()
            if tactical_path.exists():
                tactical_state = torch.load(tactical_path, map_location="cpu")
                self.tactical_model = TacticalScorer(input_dim=int(tactical_state.get("input_dim", 32)))
                self.tactical_model.load_state_dict(tactical_state["model_state"])
                self.tactical_model.eval()
            self.loaded_from = candidate_dir
            return

    def reset(self) -> None:
        self.enemy_histories = defaultdict(lambda: deque(maxlen=self.history_len))

    def observe(self, observation: Dict[str, Any]) -> None:
        if self.faction is None:
            return
        for operator in observation.get("operators", []):
            if operator.get("color") == self.faction:
                continue
            feature = operator_to_feature(operator, frame=observation, fallback_pos=operator.get("cur_hex"))
            self.enemy_histories[int(operator.get("obj_id"))].append(feature)

    def _intent_distribution(self, enemy_id: int) -> Optional[np.ndarray]:
        if self.intent_model is None or enemy_id not in self.enemy_histories:
            return None
        history = list(self.enemy_histories[enemy_id])
        if not history:
            return None
        while len(history) < self.history_len:
            history.insert(0, np.zeros_like(history[0]))
        tensor = torch.tensor(np.stack(history), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.intent_model(tensor)[0, 0]
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def predict_intent(self, enemy_id: int) -> Dict[str, Any]:
        distribution = self._intent_distribution(enemy_id)
        if distribution is None:
            return {"top_direction": 0, "distribution": [1.0] + [0.0] * 6}
        top_direction = int(np.argmax(distribution))
        return {
            "top_direction": top_direction,
            "distribution": distribution.tolist(),
        }

    def predict_location(self, enemy_id: int, current_hex: int, tactical_map: Any) -> int:
        intent = self.predict_intent(enemy_id)
        direction = int(intent["top_direction"])
        if direction == 0 or tactical_map is None:
            return current_hex
        q, r = _hex_to_axial(current_hex)
        dq, dr = _DIRECTION_STEPS[direction]
        target = _axial_to_hex(q + dq, r + dr)
        return target if tactical_map.is_valid(target) else current_hex

    def _score_tactical_vector(self, vector: np.ndarray) -> float:
        if self.tactical_model is None:
            return 0.0
        with torch.no_grad():
            tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
            return float(self.tactical_model(tensor).view(-1).item())

    def choose_shoot_candidate(
        self,
        operator: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        observation: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        enemy_lookup = {item.get("obj_id"): item for item in observation.get("operators", [])}
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for candidate in candidates:
            target = enemy_lookup.get(candidate.get("target_obj_id"))
            vector = action_feature_vector(operator, candidate, target_operator=target)
            model_score = self._score_tactical_vector(vector)
            target_hp = candidate.get("target_blood", target.get("blood", 100) if target else 100)
            damage = candidate.get("damage", 10)
            hit_prob = candidate.get("hit_prob", 0.5)
            kill_bonus = 100.0 if target_hp <= damage else 0.0
            profile_bonus = 10.0 if self.policy_profile == "aggressive_exploiter" else 0.0
            score = model_score + kill_bonus + hit_prob * 50.0 - float(target_hp) * 0.1 + profile_bonus
            scored.append((score, candidate))
        return max(scored, key=lambda item: item[0])[1]

    def choose_city(
        self,
        operator: Dict[str, Any],
        cities: Iterable[Dict[str, Any]],
        observation: Dict[str, Any],
        tactical_map: Any,
    ) -> Optional[Dict[str, Any]]:
        best: Optional[Tuple[float, Dict[str, Any]]] = None
        visible_enemies = [enemy for enemy in observation.get("operators", []) if enemy.get("color") != self.faction]
        for city in cities:
            vector = action_feature_vector(operator, {"move_path": []}, city=city)
            model_score = self._score_tactical_vector(vector)
            city_hex = int(city.get("coord", 0))
            distance_penalty = 0.0
            if tactical_map is not None and operator.get("cur_hex"):
                distance_penalty = tactical_map.get_distance(operator["cur_hex"], city_hex)
            threat_penalty = 0.0
            for enemy in visible_enemies:
                predicted = self.predict_location(int(enemy["obj_id"]), int(enemy.get("cur_hex", 0)), tactical_map)
                if tactical_map is not None:
                    threat_penalty += max(0.0, 5.0 - tactical_map.get_distance(predicted, city_hex))
            occupy_bias = 8.0 if self.policy_profile == "occupy_exploiter" else 0.0
            score = model_score + float(city.get("value", 0)) * 0.2 + occupy_bias - distance_penalty * 0.3 - threat_penalty
            if best is None or score > best[0]:
                best = (score, city)
        return best[1] if best else None
