from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .features import FEATURE_DIM, TACTICAL_FEATURE_DIM, action_feature_vector, build_feature_similarity_graph, operator_to_feature, operator_to_graph_feature

try:
    import torch

    from .models import GraphSituationModel, MultiHorizonIntentionModel, TacticalScorer

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    GraphSituationModel = None  # type: ignore[assignment]
    MultiHorizonIntentionModel = None  # type: ignore[assignment]
    TacticalScorer = None  # type: ignore[assignment]


def _torch_inference_mode():
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is unavailable")
    return getattr(torch, "inference_mode", torch.no_grad)()


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


def _coverage(total: int, learned: int) -> float:
    if total <= 0:
        return 0.0
    return float(learned / total)


class LearnedPolicy:
    def __init__(self) -> None:
        self.history_len = 8
        self.faction: Optional[int] = None
        self.policy_profile = "challenger"
        self.device = "cpu"
        self.requested_device = "auto"
        self.intent_model = None
        self.situation_model = None
        self.tactical_model = None
        self._feature_normalizer: Dict[str, Any] = {}
        self.enemy_histories: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.history_len))
        self.loaded_from: Optional[Path] = None
        self.manifest: Dict[str, Any] = {}
        self._observation_cache: Optional[Dict[str, Any]] = None
        self._reset_diagnostics()

    def _reset_diagnostics(self) -> None:
        self.action_counts = {
            "move": {"total": 0, "learned": 0},
            "shoot": {"total": 0, "learned": 0},
        }
        self.score_totals = {
            "move": {"samples": 0, "intent_bias": 0.0, "situation_score": 0.0, "tactical_score": 0.0, "combined_score": 0.0},
            "shoot": {"samples": 0, "intent_bias": 0.0, "situation_score": 0.0, "tactical_score": 0.0, "combined_score": 0.0},
        }
        self.trace_samples: List[Dict[str, Any]] = []
        self._reset_move_route_diagnostics()

    def _load_intent_model(self, payload: Dict[str, Any]) -> Any:
        hidden_dim = int(payload.get("hidden_dim") or payload["model_state"]["lstm.weight_ih_l0"].shape[0] // 4)
        model = MultiHorizonIntentionModel(
            input_dim=int(payload.get("feature_dim", FEATURE_DIM)),
            hidden_dim=hidden_dim,
            horizons=int(payload.get("horizons", 5)),
            num_classes=int(payload.get("num_classes", 7)),
        )
        model.load_state_dict(payload["model_state"])
        model.eval()
        return model

    def _load_situation_model(self, payload: Dict[str, Any]) -> Any:
        hidden_dim = int(payload.get("hidden_dim") or payload["model_state"]["backbone.sage1.linear.weight"].shape[0])
        model = GraphSituationModel(
            num_features=int(payload.get("input_dim", FEATURE_DIM)),
            hidden_dim=hidden_dim,
        )
        model.load_state_dict(payload["model_state"])
        model.eval()
        return model

    def _load_tactical_model(self, payload: Dict[str, Any]) -> Any:
        hidden_dim = int(payload["model_state"]["model.0.weight"].shape[0])
        model = TacticalScorer(
            input_dim=int(payload.get("input_dim", 32)),
            hidden_dim=hidden_dim,
        )
        model.load_state_dict(payload["model_state"])
        model.eval()
        return model

    def _load_normalizer(self, artifact_dir: Path) -> Dict[str, Any]:
        for name in ("situation_feature_normalizer.json", "feature_normalizer.json"):
            path = artifact_dir / name
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def has_models(self) -> bool:
        return self.intent_model is not None and self.situation_model is not None

    def has_tactical(self) -> bool:
        return self.tactical_model is not None

    def has_normalizer(self) -> bool:
        return bool(self._feature_normalizer)

    def _resolve_device(self) -> str:
        if not TORCH_AVAILABLE:
            return "cpu"
        requested = (os.environ.get("SHENJI_POLICY_DEVICE") or "auto").strip()
        self.requested_device = requested or "auto"
        normalized = self.requested_device.lower()
        cuda = getattr(torch, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        try:
            cuda_available = callable(is_available) and bool(is_available())
        except Exception:
            cuda_available = False
        if normalized == "auto":
            return "cuda" if cuda_available else "cpu"
        if normalized.startswith("cuda") and not cuda_available:
            return "cpu"
        if normalized == "cpu" or normalized.startswith("cuda"):
            return normalized
        return self.requested_device

    def _move_to_device(self, value: Any) -> Any:
        to_method = getattr(value, "to", None)
        if callable(to_method):
            return to_method(self.device)
        return value

    def _load_models(self, intent_path: Path, situation_path: Path) -> Tuple[Any, Any]:
        intent_state = torch.load(intent_path, map_location="cpu")
        situation_state = torch.load(situation_path, map_location="cpu")
        return self._move_to_device(self._load_intent_model(intent_state)), self._move_to_device(
            self._load_situation_model(situation_state)
        )

    def configure(self, faction: int, artifact_dir: Optional[str] = None, policy_profile: str = "challenger") -> None:
        self.faction = faction
        self.policy_profile = policy_profile or "challenger"
        self.history_len = 8
        self.device = "cpu"
        self.requested_device = "auto"
        self.intent_model = None
        self.situation_model = None
        self.tactical_model = None
        self._feature_normalizer = {}
        self.loaded_from = None
        self.manifest = {}
        self._observation_cache = None
        self.enemy_histories = defaultdict(lambda: deque(maxlen=self.history_len))
        self._reset_diagnostics()
        if not TORCH_AVAILABLE or self.policy_profile == "baseline":
            return
        self.device = self._resolve_device()

        candidate_dirs: List[Path] = []
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

            intent_name = payload.get("intent_lstm") or payload.get("intent_model")
            situation_name = payload.get("situation_gcn") or payload.get("situation_model")
            if not intent_name or not situation_name:
                continue

            intent_path = candidate_dir / intent_name
            situation_path = candidate_dir / situation_name
            if not intent_path.exists() or not situation_path.exists():
                continue

            try:
                self.intent_model, self.situation_model = self._load_models(intent_path, situation_path)
            except Exception:
                if self.device == "cpu":
                    raise
                self.device = "cpu"
                self.intent_model, self.situation_model = self._load_models(intent_path, situation_path)
            self.loaded_from = candidate_dir
            self.manifest = payload

            # Load normalizer
            self._feature_normalizer = self._load_normalizer(candidate_dir)

            # Load tactical model if available
            tactical_name = payload.get("tactical_model") or "tactical_model.pt"
            tactical_path = candidate_dir / tactical_name
            if tactical_path.exists():
                try:
                    tactical_state = torch.load(tactical_path, map_location="cpu")
                    self.tactical_model = self._move_to_device(self._load_tactical_model(tactical_state))
                except Exception:
                    pass
            return

    def reset(self) -> None:
        self.enemy_histories = defaultdict(lambda: deque(maxlen=self.history_len))
        self._observation_cache = None
        self._reset_diagnostics()

    def note_action(self, action_type: str, used_learned: bool, trace: Optional[Dict[str, Any]] = None) -> None:
        key = action_type.lower()
        if key not in self.action_counts:
            return
        self.action_counts[key]["total"] += 1
        if used_learned:
            self.action_counts[key]["learned"] += 1
        if trace:
            totals = self.score_totals[key]
            numeric_fields = ("intent_bias", "situation_score", "tactical_score", "combined_score")
            if any(field in trace for field in numeric_fields):
                totals["samples"] += 1
                for field in numeric_fields:
                    if field in trace:
                        totals[field] = totals.get(field, 0.0) + float(trace.get(field, 0.0))
            if len(self.trace_samples) < 12:
                sample = {"action": key, "used_learned": used_learned}
                for field in ("obj_id", "candidate_id", "intent_bias", "situation_score", "tactical_score", "combined_score", "reason"):
                    if field in trace:
                        sample[field] = trace[field]
                self.trace_samples.append(sample)

    def _reset_move_route_diagnostics(self) -> None:
        self.move_route_aggregate = {
            "total_attempts": 0,
            "empty_route_count": 0,
            "reachable_city_total": 0,
            "by_move_type": {},
            "route_failure_reasons": {},
        }
        self.move_route_samples: List[Dict[str, Any]] = []

    def record_move_route(self, diagnostic: Dict[str, Any]) -> None:
        agg = self.move_route_aggregate
        agg["total_attempts"] += 1
        if diagnostic.get("route_empty"):
            agg["empty_route_count"] += 1
        agg["reachable_city_total"] += int(diagnostic.get("reachable_city_count", 0))
        move_type_name = str(diagnostic.get("move_type_name", "unknown"))
        mt = agg["by_move_type"].setdefault(move_type_name, {"total": 0, "empty": 0})
        mt["total"] += 1
        if diagnostic.get("route_empty"):
            mt["empty"] += 1
        reason = str(diagnostic.get("route_failure_reason", "none"))
        if reason != "none":
            agg["route_failure_reasons"][reason] = agg["route_failure_reasons"].get(reason, 0) + 1
        if len(self.move_route_samples) < 50:
            self.move_route_samples.append(diagnostic)

    def get_diagnostics(self) -> Dict[str, Any]:
        move_total = int(self.action_counts["move"]["total"])
        move_learned = int(self.action_counts["move"]["learned"])
        shoot_total = int(self.action_counts["shoot"]["total"])
        shoot_learned = int(self.action_counts["shoot"]["learned"])
        total_actions = move_total + shoot_total
        learned_actions = move_learned + shoot_learned
        score_means: Dict[str, Dict[str, float]] = {}
        for key, totals in self.score_totals.items():
            samples = int(totals["samples"])
            score_means[key] = {
                "intent_bias": float(totals["intent_bias"] / samples) if samples else 0.0,
                "situation_score": float(totals["situation_score"] / samples) if samples else 0.0,
                "tactical_score": float(totals["tactical_score"] / samples) if samples else 0.0,
                "combined_score": float(totals["combined_score"] / samples) if samples else 0.0,
                "samples": samples,
            }
        return {
            "policy_profile": self.policy_profile,
            "device": self.device,
            "requested_device": self.requested_device,
            "loaded_from": str(self.loaded_from) if self.loaded_from else None,
            "intent_model_loaded": self.intent_model is not None,
            "situation_model_loaded": self.situation_model is not None,
            "tactical_model_loaded": self.tactical_model is not None,
            "normalizer_loaded": self.has_normalizer(),
            "counts": {
                "move": dict(self.action_counts["move"]),
                "shoot": dict(self.action_counts["shoot"]),
            },
            "learned_path_coverage": _coverage(total_actions, learned_actions),
            "move_path_coverage": _coverage(move_total, move_learned),
            "shoot_path_coverage": _coverage(shoot_total, shoot_learned),
            "score_means": score_means,
            "trace_samples": list(self.trace_samples),
            "move_route_aggregate": dict(self.move_route_aggregate),
            "move_route_samples": list(self.move_route_samples),
        }

    def observe(self, observation: Dict[str, Any]) -> None:
        if self.faction is None:
            return
        for operator in observation.get("operators", []):
            if operator.get("color") == self.faction:
                continue
            feature = operator_to_feature(operator, frame=observation, fallback_pos=operator.get("cur_hex"))
            self.enemy_histories[int(operator.get("obj_id"))].append(feature)
        self._observation_cache = self._build_observation_cache(observation)

    def _build_observation_cache(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        operators = list(observation.get("operators", []))
        if not operators:
            return {
                "operators": [],
                "features": np.zeros((1, FEATURE_DIM), dtype=np.float32),
                "index_lookup": {},
            }
        features = np.stack(
            [
                operator_to_feature(operator, frame=observation, fallback_pos=operator.get("cur_hex"))
                for operator in operators
            ]
        ).astype(np.float32)
        index_lookup = {
            int(operator.get("obj_id")): idx
            for idx, operator in enumerate(operators)
            if operator.get("obj_id") is not None
        }
        return {
            "operators": operators,
            "features": features,
            "index_lookup": index_lookup,
        }

    def _ensure_observation_cache(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self._observation_cache is None:
            self._observation_cache = self._build_observation_cache(observation)
        return self._observation_cache

    def _intent_distribution(self, enemy_id: int) -> Optional[List[float]]:
        if self.intent_model is None or enemy_id not in self.enemy_histories:
            return None
        history = list(self.enemy_histories[enemy_id])
        if not history:
            return None
        while len(history) < self.history_len:
            history.insert(0, np.zeros_like(history[0]))
        tensor = self._move_to_device(torch.tensor(np.stack(history), dtype=torch.float32).unsqueeze(0))
        with _torch_inference_mode():
            logits = self.intent_model(tensor)[0, 0]
        return torch.softmax(logits, dim=-1).detach().cpu().tolist()

    def predict_intent(self, enemy_id: int) -> Dict[str, Any]:
        distribution = self._intent_distribution(enemy_id)
        if distribution is None:
            return {"top_direction": 0, "confidence": 0.0, "distribution": [1.0] + [0.0] * 6}
        top_direction = max(range(len(distribution)), key=distribution.__getitem__)
        confidence = float(distribution[top_direction])
        return {
            "top_direction": top_direction,
            "confidence": confidence,
            "distribution": [float(score) for score in distribution],
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

    def _score_counterfactual(self, observation: Dict[str, Any], overrides: Dict[int, Dict[str, Any]]) -> float:
        if self.situation_model is None:
            return 0.0
        operators = list(observation.get("operators", []))
        if not operators:
            return 0.0

        # Build features using training-compatible format (operator_to_graph_feature)
        feature_list = []
        operator_ids = []
        for op in operators:
            op_id = op.get("obj_id")
            if op_id is None:
                continue
            feature_list.append(operator_to_graph_feature(op, frame=observation, fallback_pos=op.get("cur_hex")))
            operator_ids.append(int(op_id))

        if not feature_list:
            return 0.0

        features = np.stack(feature_list).astype(np.float32)
        index_lookup = {op_id: idx for idx, op_id in enumerate(operator_ids)}

        override_row = None
        for obj_id, operator in overrides.items():
            row_idx = index_lookup.get(int(obj_id))
            if row_idx is None:
                continue
            features[row_idx] = operator_to_graph_feature(
                operator,
                frame=observation,
                fallback_pos=operator.get("cur_hex"),
            )
            override_row = row_idx

        adjacency = build_feature_similarity_graph(features)

        # Apply training normalizer if available
        if self._feature_normalizer:
            feat_mean = np.array(self._feature_normalizer.get("feature_mean", []), dtype=np.float32)
            feat_std = np.array(self._feature_normalizer.get("feature_std", []), dtype=np.float32)
            if len(feat_mean) == features.shape[1] and len(feat_std) == features.shape[1]:
                feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
                features = (features - feat_mean) / feat_std

        with _torch_inference_mode():
            x = self._move_to_device(torch.tensor(features, dtype=torch.float32))
            adj = self._move_to_device(torch.tensor(adjacency, dtype=torch.float32))
            node_scores = self.situation_model.backbone(x, adj)
            if override_row is not None and 0 <= override_row < node_scores.shape[0]:
                return float(node_scores[override_row].item())
            return float(node_scores.mean().item())

    def _tactical_score(
        self,
        operator: Dict[str, Any],
        candidate: Dict[str, Any],
        *,
        target_operator: Optional[Dict[str, Any]] = None,
        city: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Evaluate action quality using the tactical model (BCE logit, higher = better)."""
        if self.tactical_model is None:
            return 0.0
        features = action_feature_vector(operator, candidate, target_operator=target_operator, city=city)
        tensor = self._move_to_device(torch.tensor(features, dtype=torch.float32).unsqueeze(0))
        with _torch_inference_mode():
            logit = self.tactical_model(tensor).item()
        return float(logit)

    def choose_shoot_candidate(
        self,
        operator: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        observation: Dict[str, Any],
        tactical_map: Any,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if not candidates or not self.has_models():
            return None, None
        enemy_lookup = {
            int(item["obj_id"]): item
            for item in observation.get("operators", [])
            if item.get("obj_id") is not None
        }
        best: Optional[Tuple[float, Dict[str, Any], Dict[str, Any]]] = None
        operator_hex = int(operator.get("cur_hex", 0))
        for candidate in candidates:
            target_id = candidate.get("target_obj_id")
            if target_id is None:
                continue
            target = enemy_lookup.get(int(target_id))
            if target is None:
                continue
            intent = self.predict_intent(int(target_id))
            predicted_hex = self.predict_location(int(target_id), int(target.get("cur_hex", 0)), tactical_map)
            intent_bias = 0.0
            if tactical_map is not None:
                distance = tactical_map.get_distance(operator_hex, predicted_hex)
                if distance >= 0:
                    intent_bias = max(0.0, 8.0 - float(distance)) * float(intent["confidence"]) * 2.0

            adjusted_target = dict(target)
            expected_damage = float(candidate.get("damage", 0.0)) * float(candidate.get("hit_prob", 0.0))
            adjusted_target["blood"] = max(0.0, float(target.get("blood", 0.0)) - expected_damage)
            situation_score = self._score_counterfactual(observation, {int(target_id): adjusted_target})

            target_hp = float(target.get("blood", 100.0))
            attack_level = float(candidate.get("attack_level", 0.0))
            hit_prob = float(candidate.get("hit_prob", 0.0))
            kill_bonus = 75.0 if adjusted_target["blood"] <= 0 else 0.0
            profile_bonus = 12.0 if self.policy_profile == "aggressive_exploiter" else 0.0
            target_value = float(target.get("value", 0.0))
            situation_contribution = 0.0
            if situation_score > 1e-6:
                situation_contribution = situation_score * 100.0
            tactical_score = self._tactical_score(operator, candidate, target_operator=target)
            tactical_contribution = float(torch.sigmoid(torch.tensor(tactical_score)).item()) * 20.0
            combined_score = (
                situation_contribution
                + attack_level * 8.0
                + hit_prob * 30.0
                + intent_bias
                + kill_bonus
                + profile_bonus
                + target_value * 0.3
                - target_hp * 0.05
                + tactical_contribution
            )
            trace = {
                "obj_id": int(operator.get("obj_id", -1)),
                "candidate_id": int(target_id),
                "intent_bias": float(intent_bias),
                "situation_score": float(situation_score),
                "tactical_score": float(tactical_score),
                "combined_score": float(combined_score),
            }
            if best is None or combined_score > best[0]:
                best = (combined_score, candidate, trace)
        if best is None:
            return None, None
        return best[1], best[2]

    def choose_city(
        self,
        operator: Dict[str, Any],
        cities: Iterable[Dict[str, Any]],
        observation: Dict[str, Any],
        tactical_map: Any,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if not self.has_models():
            return None, None
        operator_id = operator.get("obj_id")
        if operator_id is None:
            return None, None
        visible_enemies = [
            enemy
            for enemy in observation.get("operators", [])
            if enemy.get("color") != self.faction and enemy.get("obj_id") is not None
        ]
        best: Optional[Tuple[float, Dict[str, Any], Dict[str, Any]]] = None
        for city in cities:
            city_hex = int(city.get("coord", 0))
            adjusted_operator = dict(operator)
            adjusted_operator["cur_hex"] = city_hex
            adjusted_operator["cur_pos"] = 0
            situation_score = self._score_counterfactual(observation, {int(operator_id): adjusted_operator})

            distance_penalty = 0.0
            raw_distance = -1
            if tactical_map is not None and operator.get("cur_hex"):
                distance = tactical_map.get_distance(int(operator["cur_hex"]), city_hex)
                if distance >= 0:
                    raw_distance = float(distance)
                    distance_penalty = raw_distance

            threat_penalty = 0.0
            for enemy in visible_enemies:
                predicted_hex = self.predict_location(int(enemy["obj_id"]), int(enemy.get("cur_hex", 0)), tactical_map)
                if tactical_map is None:
                    continue
                distance = tactical_map.get_distance(predicted_hex, city_hex)
                if distance >= 0:
                    threat_penalty += max(0.0, 6.0 - float(distance))

            intent_bias = -threat_penalty
            occupy_bias = 8.0 if self.policy_profile == "occupy_exploiter" else 0.0
            city_value = float(city.get("value", 0.0))
            city_owner = int(city.get("flag", -1))
            owner_bonus = 0.0
            if self.faction is not None and city_owner >= 0:
                if city_owner != self.faction:
                    owner_bonus = 25.0
                else:
                    owner_bonus = -15.0

            occupy_proximity_bonus = 0.0
            if raw_distance >= 0 and raw_distance <= 1:
                occupy_proximity_bonus = 20.0

            situation_contribution = 0.0
            if situation_score > 1e-6:
                situation_contribution = situation_score * 100.0

            # Tactical model evaluates the move-to-city action
            move_candidate = {
                "move_path": [int(operator.get("cur_hex", 0)), city_hex],
            }
            tactical_score = self._tactical_score(operator, move_candidate, city=city)
            tactical_contribution = float(torch.sigmoid(torch.tensor(tactical_score)).item()) * 20.0

            combined_score = (
                situation_contribution
                + city_value * 0.5
                + owner_bonus
                + occupy_proximity_bonus
                + occupy_bias
                + intent_bias
                - distance_penalty * 0.08
                + tactical_contribution
            )
            trace = {
                "obj_id": int(operator_id),
                "candidate_id": city_hex,
                "intent_bias": float(intent_bias),
                "situation_score": float(situation_score),
                "tactical_score": float(tactical_score),
                "combined_score": float(combined_score),
            }
            if best is None or combined_score > best[0]:
                best = (combined_score, city, trace)
        if best is None:
            return None, None
        return best[1], best[2]
