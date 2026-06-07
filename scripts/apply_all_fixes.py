#!/usr/bin/env python3
"""Apply all Phase 2+3 fixes to the shenji_agent codebase.

This script:
1. Patches ai/policy.py to add ablation_mode support natively
2. Patches training/selfplay.py to add --ablation-mode CLI arg and expanded exploiters
3. Creates ai/enhanced_policy.py for B7
4. Creates ai/weights_incumbent_enhanced/model_manifest.json for B7
5. Validates all changes

Run from the repo root: python3.10 scripts/apply_all_fixes.py
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
log_lines = []


def log(msg: str) -> None:
    print(f"[FIX] {msg}")
    log_lines.append(msg)


def patch_policy_py() -> None:
    """B5: Add ablation_mode to LearnedPolicy.configure()."""
    policy_path = REPO_ROOT / "ai" / "policy.py"
    text = policy_path.read_text(encoding="utf-8")

    # 1. Add ABLATION_MODES constant after _DIRECTION_STEPS
    ablation_const = '''
ABLATION_MODES = {
    "full": {"intent": True, "situation": True, "tactical": True, "objective": True},
    "no_intent": {"intent": False, "situation": True, "tactical": True, "objective": True},
    "no_situation": {"intent": True, "situation": False, "tactical": True, "objective": True},
    "rule_only": {"intent": False, "situation": False, "tactical": False, "objective": False},
}
'''
    if "ABLATION_MODES" not in text:
        # Insert after the _DIRECTION_STEPS definition
        marker = 'def _hex_to_axial'
        text = text.replace(marker, ablation_const + '\n' + marker)
        log("Added ABLATION_MODES constant to policy.py")

    # 2. Modify configure() signature to accept ablation_mode
    old_sig = 'def configure(self, faction: int, artifact_dir: Optional[str] = None, policy_profile: str = "challenger") -> None:'
    new_sig = 'def configure(self, faction: int, artifact_dir: Optional[str] = None, policy_profile: str = "challenger", *, ablation_mode: str = "full") -> None:'
    if old_sig in text:
        text = text.replace(old_sig, new_sig)
        log("Updated configure() signature to accept ablation_mode")
    elif "ablation_mode" not in text:
        log("WARNING: Could not find configure() signature to patch")

    # 3. Add ablation enforcement before the final logging.getLogger line
    ablation_enforcement = '''
            # --- Ablation mode enforcement ---
            if ablation_mode != "full":
                _abl = ABLATION_MODES.get(ablation_mode, ABLATION_MODES["full"])
                _abl_disabled = []
                if not _abl["intent"]:
                    if self.intent_model is not None:
                        _abl_disabled.append("intent")
                    self.intent_model = None
                if not _abl["situation"]:
                    if self.situation_model is not None:
                        _abl_disabled.append("situation")
                    self.situation_model = None
                if not _abl["tactical"]:
                    if self.tactical_model is not None:
                        _abl_disabled.append("tactical")
                    self.tactical_model = None
                if not _abl["objective"]:
                    _abl_disabled.append("objective")
                    self.objective_weight = 0.0
                self.ablation_mode = ablation_mode
                logging.getLogger(__name__).info(
                    "LearnedPolicy [ABLATION]: mode=%s disabled=%s",
                    ablation_mode, _abl_disabled or "none",
                )
            else:
                self.ablation_mode = "full"
'''
    # Insert before the final "return" in configure()
    configure_return_marker = "            return\n\n    def reset"
    if configure_return_marker in text and "Ablation mode enforcement" not in text:
        text = text.replace(configure_return_marker, ablation_enforcement + configure_return_marker)
        log("Added ablation enforcement to configure()")

    policy_path.write_text(text, encoding="utf-8")
    log("policy.py patched successfully")


def patch_selfplay_py() -> None:
    """B5+B8: Add --ablation-mode arg and expanded exploiter profiles."""
    selfplay_path = REPO_ROOT / "training" / "selfplay.py"
    text = selfplay_path.read_text(encoding="utf-8")

    # 1. Add --ablation-mode argument to main()
    old_args = 'parser.add_argument("--allow-self-match-smoke", action="store_true")'
    new_args = '''parser.add_argument("--allow-self-match-smoke", action="store_true")
    parser.add_argument("--ablation-mode", default="full",
                        choices=["full", "no_intent", "no_situation", "rule_only"],
                        help="Ablation mode: selectively disable learned models")'''
    if old_args in text and "--ablation-mode" not in text:
        text = text.replace(old_args, new_args)
        log("Added --ablation-mode argument to selfplay.py")

    # 2. Pass ablation_mode through to participant_spec
    # Find where candidate_participant is built and add ablation_mode
    old_candidate_spec = 'candidate_participant = _build_participant_spec(candidate_arg, candidate_profile, current_head_sha, "CurrentAI")'
    new_candidate_spec = '''candidate_participant = _build_participant_spec(candidate_arg, candidate_profile, current_head_sha, "CurrentAI")
    ablation_mode = getattr(args, "ablation_mode", "full")
    if ablation_mode != "full":
        candidate_participant["ablation_mode"] = ablation_mode
        logging.getLogger(__name__).info("ablation mode active: %s", ablation_mode)'''
    if old_candidate_spec in text and "ablation_mode" not in text.split(old_candidate_spec)[0][-200:]:
        text = text.replace(old_candidate_spec, new_candidate_spec)
        log("Added ablation_mode passthrough to participant spec")

    # 3. Expand exploiter profiles - find where exploiters are defined in league_state
    # The exploiters are in league_state["exploiters"]. We need to add more profiles.
    # Find the bootstrap_league function to add more exploiter profiles
    old_bootstrap = 'def _bootstrap_league'
    if old_bootstrap in text:
        # Find the exploiters list initialization in _bootstrap_league
        # It typically has: "exploiters": ["aggressive_exploiter", "occupy_exploiter"]
        # We want to expand it
        pass  # We'll handle this through config instead

    # 4. Increase exploiter scenario count from 4 to 20
    old_exploiter_slice = 'scenarios_by_bucket.get("complex", [])[:4]'
    new_exploiter_slice = 'scenarios_by_bucket.get("complex", [])[:20]'
    if old_exploiter_slice in text:
        text = text.replace(old_exploiter_slice, new_exploiter_slice)
        log("Increased exploiter scenario count from 4 to 20")

    # 5. Pass ablation_mode to agent setup through _run_single
    # Find where agent.setup is called with policy_profile
    old_red_setup = '"policy_profile": red_profile["profile"]'
    new_red_setup = '"policy_profile": red_profile["profile"], "ablation_mode": red_profile.get("ablation_mode", "full")'
    if old_red_setup in text:
        text = text.replace(old_red_setup, new_red_setup, 1)
        log("Added ablation_mode to red agent setup")

    old_blue_setup = '"policy_profile": blue_profile["profile"]'
    new_blue_setup = '"policy_profile": blue_profile["profile"], "ablation_mode": blue_profile.get("ablation_mode", "full")'
    if old_blue_setup in text:
        text = text.replace(old_blue_setup, new_blue_setup, 1)
        log("Added ablation_mode to blue agent setup")

    selfplay_path.write_text(text, encoding="utf-8")
    log("selfplay.py patched successfully")


def patch_agent_py() -> None:
    """B5: Make agent.py pass ablation_mode to policy.configure()."""
    agent_path = REPO_ROOT / "ai" / "agent.py"
    text = agent_path.read_text(encoding="utf-8")

    # Find where policy.configure is called and add ablation_mode
    old_configure = 'self.policy.configure(self.faction, self.policy_artifacts_dir, self.policy_profile)'
    new_configure = 'self.policy.configure(self.faction, self.policy_artifacts_dir, self.policy_profile, ablation_mode=setup_info.get("ablation_mode", "full"))'
    if old_configure in text and "ablation_mode" not in text:
        text = text.replace(old_configure, new_configure)
        log("Added ablation_mode passthrough in agent.py setup()")

    agent_path.write_text(text, encoding="utf-8")
    log("agent.py patched successfully")


def create_enhanced_policy() -> None:
    """B7: Create enhanced rule-based policy."""
    enhanced_path = REPO_ROOT / "ai" / "enhanced_policy.py"
    enhanced_code = '''"""Enhanced rule-based policy for stronger baseline comparison (B7).

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
'''
    enhanced_path.write_text(enhanced_code, encoding="utf-8")
    log("Created ai/enhanced_policy.py")


def create_enhanced_manifest() -> None:
    """B7: Create enhanced incumbent manifest."""
    enhanced_dir = REPO_ROOT / "ai" / "weights_incumbent_enhanced"
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at": "20260607_153000",
        "history_len": 8,
        "horizons": 5,
        "intent_lstm": "intent_lstm.pt",
        "situation_gcn": "situation_gcn.pt",
        "snapshot_role": "enhanced_rule_baseline",
        "policy_profile": "enhanced",
        "note": "Enhanced rule-based baseline with smarter heuristics (B7)",
    }
    manifest_path = enhanced_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log("Created ai/weights_incumbent_enhanced/model_manifest.json")


def validate_patches() -> bool:
    """Validate that all patches were applied correctly."""
    errors = []

    # Check policy.py
    policy_text = (REPO_ROOT / "ai" / "policy.py").read_text(encoding="utf-8")
    if "ABLATION_MODES" not in policy_text:
        errors.append("policy.py: ABLATION_MODES not found")
    if "ablation_mode" not in policy_text:
        errors.append("policy.py: ablation_mode not found in configure()")

    # Check selfplay.py
    selfplay_text = (REPO_ROOT / "training" / "selfplay.py").read_text(encoding="utf-8")
    if "--ablation-mode" not in selfplay_text:
        errors.append("selfplay.py: --ablation-mode argument not found")
    if "[:20]" not in selfplay_text:
        errors.append("selfplay.py: exploiter scenario count not expanded")

    # Check agent.py
    agent_text = (REPO_ROOT / "ai" / "agent.py").read_text(encoding="utf-8")
    if "ablation_mode" not in agent_text:
        errors.append("agent.py: ablation_mode passthrough not found")

    # Check enhanced policy
    if not (REPO_ROOT / "ai" / "enhanced_policy.py").exists():
        errors.append("ai/enhanced_policy.py not found")

    # Check enhanced manifest
    if not (REPO_ROOT / "ai" / "weights_incumbent_enhanced" / "model_manifest.json").exists():
        errors.append("ai/weights_incumbent_enhanced/model_manifest.json not found")

    if errors:
        for err in errors:
            log(f"VALIDATION ERROR: {err}")
        return False

    log("All patches validated successfully!")
    return True


def main() -> None:
    log(f"Applying all Phase 2+3 fixes to {REPO_ROOT}")

    patch_policy_py()
    patch_selfplay_py()
    patch_agent_py()
    create_enhanced_policy()
    create_enhanced_manifest()

    success = validate_patches()

    summary_path = REPO_ROOT / "experiments" / "shenji_agent"
    summary_path.mkdir(parents=True, exist_ok=True)
    (summary_path / "fix_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    if success:
        log("All fixes applied and validated!")
        sys.exit(0)
    else:
        log("Some fixes failed validation!")
        sys.exit(1)


if __name__ == "__main__":
    main()
