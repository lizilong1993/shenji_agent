"""Enhanced rule-based policy for stronger baseline comparison (B7).

This module provides an enhanced rule-based agent that makes smarter decisions
than the basic baseline, creating a more meaningful comparison point for the
supervised hybrid approach.

Enhancements over baseline:
- Move: Select nearest unoccupied/enemy city (not random)
- Shoot: Score by expected_damage * target_value (not just attack_level)
- Occupy: Prioritize high-value cities near current position
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from .policy import LearnedPolicy

log = logging.getLogger(__name__)


class EnhancedPolicy(LearnedPolicy):
    """Rule-based policy with improved heuristics for stronger baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.policy_profile = "enhanced"

    def configure(
        self,
        faction: int,
        artifact_dir: Optional[str] = None,
        policy_profile: str = "enhanced",
        *,
        ablation_mode: str = "full",
    ) -> None:
        self.faction = faction
        self.policy_profile = policy_profile or "enhanced"
        # Enhanced baseline does NOT load neural models
        # It uses improved rule-based logic only
        self.intent_model = None
        self.situation_model = None
        self.tactical_model = None
        self.objective_model = None
        self.objective_weight = 1.0
        self._feature_normalizer = {}
        self.loaded_from = None
        self.manifest = {}
        self._observation_cache = None
        self.ablation_mode = "rule_only"
        from collections import defaultdict, deque
        self.history_len = 8
        self.enemy_histories = defaultdict(lambda: deque(maxlen=self.history_len))
        self._reset_diagnostics()
        log.info("EnhancedPolicy configured (no neural models loaded)")

    def enhanced_move_target(
        self,
        operator: Dict[str, Any],
        cities: List[Dict[str, Any]],
        observation: Dict[str, Any],
        tactical_map: Any,
    ) -> Optional[Dict[str, Any]]:
        """Select nearest unoccupied or enemy city."""
        if not cities or tactical_map is None:
            return None
        cur_hex = int(operator.get("cur_hex", 0))
        my_faction = self.faction

        scored_cities = []
        for city in cities:
            city_hex = city.get("coord")
            if city_hex is None:
                continue
            city_hex = int(city_hex)
            city_flag = int(city.get("flag", -1))
            city_value = float(city.get("value", 0.0))
            distance = tactical_map.get_distance(cur_hex, city_hex)
            if distance < 0:
                continue
            # Score: prefer unoccupied/enemy cities that are close and valuable
            score = city_value * 0.5
            if city_flag != my_faction:
                score += 50.0  # Bonus for unowned/enemy cities
            if city_flag == -1:
                score += 30.0  # Bonus for unoccupied
            score += max(0.0, 10.0 - float(distance)) * 5.0  # Distance bonus
            scored_cities.append((score, city))

        if not scored_cities:
            return None
        scored_cities.sort(key=lambda x: x[0], reverse=True)
        return scored_cities[0][1]

    def enhanced_shoot_target(
        self,
        operator: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        observation: Dict[str, Any],
        tactical_map: Any,
    ) -> Optional[Dict[str, Any]]:
        """Select target by expected damage * value, not just attack_level."""
        if not candidates:
            return None

        enemy_lookup = {
            int(op["obj_id"]): op
            for op in observation.get("operators", [])
            if op.get("obj_id") is not None
        }

        best_score = -float("inf")
        best_candidate = None

        for candidate in candidates:
            target_id = candidate.get("target_obj_id")
            if target_id is None:
                continue
            target = enemy_lookup.get(int(target_id))
            if target is None:
                continue

            expected_damage = float(candidate.get("damage", 0.0)) * float(candidate.get("hit_prob", 0.0))
            target_hp = max(0.0, float(target.get("blood", 0.0)))
            target_value = float(target.get("value", 0.0))

            # Score: expected damage weighted by target value
            score = expected_damage * (1.0 + target_value / 100.0)
            # Finish-off bonus
            if expected_damage >= target_hp and target_hp > 0:
                score += 200.0
            # Low-HP target bonus
            if target_hp < 30:
                score += 50.0

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def enhanced_occupy_priority(
        self,
        operator: Dict[str, Any],
        observation: Dict[str, Any],
        tactical_map: Any,
    ) -> bool:
        """Decide whether to occupy based on proximity and city value."""
        if tactical_map is None:
            return random.random() < 0.5  # fallback

        cur_hex = int(operator.get("cur_hex", 0))
        cities = observation.get("cities", [])
        my_faction = self.faction

        for city in cities:
            city_hex = city.get("coord")
            if city_hex is None:
                continue
            city_hex = int(city_hex)
            city_flag = int(city.get("flag", -1))
            if city_flag == my_faction:
                continue  # Already ours
            distance = tactical_map.get_distance(cur_hex, city_hex)
            if distance is not None and 0 <= distance <= 2:
                return True  # Close to an unowned/enemy city

        return False
