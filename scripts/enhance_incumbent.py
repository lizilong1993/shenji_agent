#!/usr/bin/env python3
"""
B7: Enhanced Incumbent -- Create an improved rule-based baseline agent.

This script generates:
  1. ai/weights_incumbent_enhanced/ directory with a manifest.json
  2. ai/enhanced_policy.py -- a Python module implementing smarter heuristics

The enhanced incumbent uses the same model weights as the baseline but with
improved rule-based fallback logic for stronger baseline comparison.

Usage:
    python scripts/enhance_incumbent.py [--output-dir OUTPUT_DIR]
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path


def create_manifest(output_dir: Path) -> None:
    """Create the manifest.json for the enhanced incumbent weights directory."""
    weights_dir = output_dir / "ai" / "weights_incumbent_enhanced"
    weights_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "agent_name": "shenji_agent_enhanced_incumbent",
        "policy_profile": "enhanced",
        "version": "1.1.0",
        "description": (
            "Enhanced rule-based incumbent with improved heuristics for "
            "move, shoot, and occupy actions. Uses same model weights as "
            "baseline but with smarter fallback logic."
        ),
        "weight_files": [
            {
                "path": "intent_encoder.pt",
                "type": "intent_encoder",
                "format": "pytorch",
                "description": "Intent sequence encoder weights (shared with baseline)"
            },
            {
                "path": "situation_gnn.pt",
                "type": "situation_gnn",
                "format": "pytorch",
                "description": "Situation graph neural network weights (shared with baseline)"
            },
            {
                "path": "tactical_mlp.pt",
                "type": "tactical_mlp",
                "format": "pytorch",
                "description": "Tactical decision MLP weights (shared with baseline)"
            },
            {
                "path": "action_head.pt",
                "type": "action_head",
                "format": "pytorch",
                "description": "Action prediction head weights (shared with baseline)"
            }
        ],
        "policy_module": "ai.enhanced_policy",
        "policy_class": "EnhancedPolicy",
        "enhancements": {
            "move": "Nearest unoccupied/enemy city selection instead of random",
            "shoot": "Weight targets by expected_damage * target_value",
            "occupy": "Prioritize high-value cities near current position"
        },
        "fallback_strategy": "enhanced_rule_based",
        "confidence_threshold": 0.6,
        "created_by": "scripts/enhance_incumbent.py"
    }

    manifest_path = weights_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[OK] Manifest written to {manifest_path}")


def create_enhanced_policy_module(output_dir: Path) -> None:
    """Create the enhanced_policy.py module with improved heuristics."""
    ai_dir = output_dir / "ai"
    ai_dir.mkdir(parents=True, exist_ok=True)

    policy_code = textwrap.dedent('''\
        """Enhanced rule-based policy for stronger baseline comparison.

        This module extends LearnedPolicy with improved heuristics for:
          - Move: select nearest unoccupied/enemy city instead of random city
          - Shoot: weight targets by expected_damage * target_value, not just attack_level
          - Occupy: prioritize high-value cities near current position

        The enhanced policy uses the same neural network weights as the baseline
        but applies smarter rule-based fallback when model confidence is low.
        """

        import math
        import random
        from typing import Any, Dict, List, Optional, Tuple


        class LearnedPolicy:
            """Base learned policy that loads model weights and provides predictions.

            This is a stub representing the existing learned policy. The enhanced
            policy extends this with improved fallback heuristics.
            """

            def __init__(self, weights_dir: str, confidence_threshold: float = 0.6):
                self.weights_dir = weights_dir
                self.confidence_threshold = confidence_threshold
                self._models_loaded = False

            def load_models(self) -> None:
                """Load neural network weights from the weights directory."""
                # In production, this loads intent_encoder.pt, situation_gnn.pt,
                # tactical_mlp.pt, and action_head.pt
                self._models_loaded = True

            def predict_action(
                self, observation: Dict[str, Any]
            ) -> Tuple[Dict[str, Any], float]:
                """Predict action using neural network.

                Returns:
                    Tuple of (action_dict, confidence_score).
                """
                # Stub: returns empty action with low confidence
                return {}, 0.0

            def get_valid_actions(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
                """Get list of valid actions for the current observation."""
                return observation.get("valid_actions", [])


        def hex_distance(pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> float:
            """Compute hex grid distance between two positions.

            Uses cube coordinate distance for hex grids. Positions are given
            as (row, col) offset coordinates which we convert to cube coords.

            Args:
                pos_a: (row, col) of first position.
                pos_b: (row, col) of second position.

            Returns:
                Hex distance as a float.
            """
            r1, c1 = pos_a
            r2, c2 = pos_b
            # Offset to cube coordinate conversion (odd-r layout)
            x1 = c1 - (r1 - (r1 & 1)) // 2
            z1 = r1
            y1 = -x1 - z1
            x2 = c2 - (r2 - (r2 & 1)) // 2
            z2 = r2
            y2 = -x2 - z2
            return max(abs(x1 - x2), abs(y1 - y2), abs(z1 - z2))


        def euclidean_distance(pos_a: Tuple[float, float], pos_b: Tuple[float, float]) -> float:
            """Compute Euclidean distance between two positions."""
            return math.sqrt((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2)


        class EnhancedPolicy(LearnedPolicy):
            """Enhanced rule-based policy with improved heuristics.

            Extends LearnedPolicy by overriding choose_city and choose_shoot_candidate
            with smarter selection strategies that serve as a stronger baseline.

            Enhancements:
                - Move: Select the nearest unoccupied or enemy-held city rather than
                  choosing randomly from all cities.
                - Shoot: Score each target as expected_damage * target_value instead
                  of ranking solely by attack_level.
                - Occupy: Prioritize high-value cities that are close to the current
                  operator position, using a value/distance ratio.
            """

            def __init__(
                self,
                weights_dir: str,
                confidence_threshold: float = 0.6,
                move_distance_weight: float = 1.0,
                shoot_damage_weight: float = 1.0,
                occupy_value_weight: float = 2.0,
                occupy_distance_penalty: float = 0.5,
            ):
                super().__init__(weights_dir, confidence_threshold)
                self.move_distance_weight = move_distance_weight
                self.shoot_damage_weight = shoot_damage_weight
                self.occupy_value_weight = occupy_value_weight
                self.occupy_distance_penalty = occupy_distance_penalty

            def choose_city(
                self,
                operator_pos: Tuple[int, int],
                cities: List[Dict[str, Any]],
                action_type: str,
                faction_id: int,
            ) -> Optional[Dict[str, Any]]:
                """Choose a target city using enhanced heuristics.

                For MOVE actions: select the nearest city that is either unoccupied
                or held by the enemy. Avoids cities already held by own faction.

                For OCCUPY actions: rank cities by (value / (distance + epsilon))
                to prefer high-value cities that are nearby.

                Args:
                    operator_pos: Current (row, col) of the operator.
                    cities: List of city dicts with keys:
                        'id', 'row', 'col', 'value', 'owner_faction'.
                    action_type: One of 'move', 'occupy'.
                    faction_id: The faction ID of the current agent.

                Returns:
                    The chosen city dict, or None if no valid target exists.
                """
                if not cities:
                    return None

                if action_type == "move":
                    return self._choose_move_target(operator_pos, cities, faction_id)
                elif action_type == "occupy":
                    return self._choose_occupy_target(operator_pos, cities, faction_id)
                else:
                    # For other action types, fall back to random selection
                    return random.choice(cities) if cities else None

            def _choose_move_target(
                self,
                operator_pos: Tuple[int, int],
                cities: List[Dict[str, Any]],
                faction_id: int,
            ) -> Optional[Dict[str, Any]]:
                """Select the nearest unoccupied or enemy-held city.

                Filters out cities owned by the agent's own faction, then sorts
                remaining cities by hex distance to the operator position.
                Returns the closest valid city.

                If all cities are owned by own faction, returns the nearest city
                overall (to allow repositioning).
                """
                # Filter to unoccupied or enemy cities
                candidates = [
                    c for c in cities
                    if c.get("owner_faction", -1) != faction_id
                ]

                # If no unoccupied/enemy cities, consider all cities
                if not candidates:
                    candidates = list(cities)

                if not candidates:
                    return None

                # Sort by distance to operator, prefer closer cities
                candidates.sort(
                    key=lambda c: hex_distance(
                        operator_pos,
                        (c.get("row", 0), c.get("col", 0))
                    )
                )

                return candidates[0]

            def _choose_occupy_target(
                self,
                operator_pos: Tuple[int, int],
                cities: List[Dict[str, Any]],
                faction_id: int,
            ) -> Optional[Dict[str, Any]]:
                """Select the best city to occupy using value/distance scoring.

                Scores each city as:
                    score = occupy_value_weight * city_value
                            - occupy_distance_penalty * distance

                Prefers high-value cities that are close to the current position.
                Cities already owned by own faction receive a large penalty.
                """
                if not cities:
                    return None

                epsilon = 1e-6  # Avoid division by zero
                scored = []
                for city in cities:
                    city_pos = (city.get("row", 0), city.get("col", 0))
                    dist = hex_distance(operator_pos, city_pos)
                    value = city.get("value", 1.0)

                    # Value-to-distance ratio with configurable weights
                    score = (
                        self.occupy_value_weight * value
                        / (dist + epsilon)
                    )

                    # Penalize cities already owned by own faction
                    if city.get("owner_faction", -1) == faction_id:
                        score *= 0.1

                    # Bonus for unoccupied cities
                    if city.get("owner_faction", -1) == -1:
                        score *= 1.5

                    scored.append((score, city))

                # Return the city with highest score
                scored.sort(key=lambda x: x[0], reverse=True)
                return scored[0][1] if scored else None

            def choose_shoot_candidate(
                self,
                operator: Dict[str, Any],
                targets: List[Dict[str, Any]],
                cities: Optional[List[Dict[str, Any]]] = None,
            ) -> Optional[Dict[str, Any]]:
                """Choose a shoot target using expected_damage * target_value scoring.

                Instead of simply selecting the target with highest attack_level,
                this method computes a composite score:

                    score = expected_damage(attacker, target) * target_value(target)

                Where:
                    expected_damage = attack_power * hit_probability
                    hit_probability depends on distance and terrain
                    target_value considers remaining HP and strategic importance

                Args:
                    operator: Dict with operator info including 'attack_level',
                        'position', 'attack_range'.
                    targets: List of target dicts with 'id', 'position',
                        'defense_level', 'hp', 'max_hp', 'unit_value'.
                    cities: Optional list of cities for strategic context.

                Returns:
                    The best target dict, or None if no valid targets.
                """
                if not targets:
                    return None

                op_pos = operator.get("position", (0, 0))
                op_attack = operator.get("attack_level", 1.0)
                op_range = operator.get("attack_range", 3)

                scored_targets = []
                for target in targets:
                    tgt_pos = target.get("position", (0, 0))
                    tgt_defense = target.get("defense_level", 0.0)
                    tgt_hp = target.get("hp", 1.0)
                    tgt_max_hp = target.get("max_hp", 1.0)
                    tgt_value = target.get("unit_value", 1.0)

                    dist = euclidean_distance(op_pos, tgt_pos)

                    # Skip targets out of range
                    if dist > op_range:
                        continue

                    # Compute expected damage
                    raw_damage = max(0.0, op_attack - tgt_defense * 0.5)
                    # Hit probability decreases with distance
                    hit_prob = max(0.1, 1.0 - (dist / (op_range + 1.0)) * 0.5)
                    expected_damage = raw_damage * hit_prob

                    # Target value: prioritize damaged high-value targets
                    hp_ratio = tgt_hp / max(tgt_max_hp, 1.0)
                    # Finishing off low-HP targets gets a bonus
                    finish_bonus = 1.0 + (1.0 - hp_ratio) * 0.5
                    target_value = tgt_value * finish_bonus

                    # Bonus for targets near valuable cities
                    city_bonus = 1.0
                    if cities:
                        for city in cities:
                            city_pos = (city.get("row", 0), city.get("col", 0))
                            if euclidean_distance(tgt_pos, city_pos) <= 2.0:
                                city_bonus += city.get("value", 0) * 0.1
                                break  # Only count nearest city

                    composite_score = (
                        self.shoot_damage_weight
                        * expected_damage
                        * target_value
                        * city_bonus
                    )
                    scored_targets.append((composite_score, target))

                if not scored_targets:
                    # All targets out of range; pick the closest one
                    targets_sorted = sorted(
                        targets,
                        key=lambda t: euclidean_distance(
                            op_pos, t.get("position", (0, 0))
                        ),
                    )
                    return targets_sorted[0] if targets_sorted else None

                scored_targets.sort(key=lambda x: x[0], reverse=True)
                return scored_targets[0][1]

            def select_action(
                self, observation: Dict[str, Any]
            ) -> Dict[str, Any]:
                """Select action using model prediction with enhanced fallback.

                First attempts neural network prediction. If confidence is below
                threshold, falls back to enhanced rule-based heuristics.

                Args:
                    observation: Full game observation dict containing:
                        'operator': current operator state,
                        'cities': list of city states,
                        'enemies': list of enemy states,
                        'faction_id': agent's faction,
                        'valid_actions': list of valid action dicts,
                        'step': current step number.

                Returns:
                    Selected action dict.
                """
                # Try neural network prediction first
                nn_action, confidence = self.predict_action(observation)

                if confidence >= self.confidence_threshold:
                    return nn_action

                # Fallback to enhanced rule-based logic
                return self._rule_based_fallback(observation)

            def _rule_based_fallback(
                self, observation: Dict[str, Any]
            ) -> Dict[str, Any]:
                """Apply enhanced heuristics when model confidence is low.

                Priority order:
                  1. Shoot if enemies in range (using enhanced target selection)
                  2. Occupy if valuable unoccupied cities nearby
                  3. Move toward nearest strategic city

                Args:
                    observation: Full game observation dict.

                Returns:
                    Action dict following the enhanced heuristic priority.
                """
                operator = observation.get("operator", {})
                cities = observation.get("cities", [])
                enemies = observation.get("enemies", [])
                faction_id = observation.get("faction_id", 0)
                op_pos = operator.get("position", (0, 0))

                # Priority 1: Shoot if enemies in range
                if enemies:
                    target = self.choose_shoot_candidate(operator, enemies, cities)
                    if target is not None:
                        return {
                            "action_type": "shoot",
                            "target_id": target.get("id"),
                            "operator_id": operator.get("id"),
                        }

                # Priority 2: Occupy valuable nearby cities
                unoccupied = [
                    c for c in cities
                    if c.get("owner_faction", -1) != faction_id
                ]
                if unoccupied:
                    # Check if any unoccupied city is within move range
                    move_range = operator.get("move_range", 3)
                    nearby = [
                        c for c in unoccupied
                        if hex_distance(
                            op_pos, (c.get("row", 0), c.get("col", 0))
                        ) <= move_range
                    ]
                    if nearby:
                        target_city = self.choose_city(
                            op_pos, nearby, "occupy", faction_id
                        )
                        if target_city is not None:
                            return {
                                "action_type": "occupy",
                                "city_id": target_city.get("id"),
                                "operator_id": operator.get("id"),
                            }

                # Priority 3: Move toward nearest strategic city
                if cities:
                    move_target = self.choose_city(
                        op_pos, cities, "move", faction_id
                    )
                    if move_target is not None:
                        return {
                            "action_type": "move",
                            "city_id": move_target.get("id"),
                            "operator_id": operator.get("id"),
                        }

                # Last resort: return a no-op / random valid action
                valid_actions = observation.get("valid_actions", [])
                if valid_actions:
                    return random.choice(valid_actions)
                return {"action_type": "noop"}
    ''')

    policy_path = ai_dir / "enhanced_policy.py"
    with open(policy_path, "w", encoding="utf-8") as f:
        f.write(policy_code)

    print(f"[OK] Enhanced policy module written to {policy_path}")


def main():
    parser = argparse.ArgumentParser(
        description="B7: Create enhanced rule-based incumbent baseline."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Root output directory (default: current directory)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("B7: Enhanced Incumbent Baseline Generator")
    print("=" * 60)
    print(f"Output directory: {output_dir.resolve()}")
    print()

    # Step 1: Create the weights directory with manifest
    print("[1/2] Creating weights_incumbent_enhanced manifest...")
    create_manifest(output_dir)

    # Step 2: Create the enhanced policy module
    print("[2/2] Creating enhanced_policy.py module...")
    create_enhanced_policy_module(output_dir)

    print()
    print("=" * 60)
    print("Enhanced incumbent creation complete.")
    print(f"  Manifest: {output_dir / 'ai' / 'weights_incumbent_enhanced' / 'manifest.json'}")
    print(f"  Policy:   {output_dir / 'ai' / 'enhanced_policy.py'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
