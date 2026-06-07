"""RL-inspired baseline agent for comparison experiments.

This implements a lightweight heuristic agent that serves as a baseline
for comparing against the supervised hybrid approach. Rather than full
PPO (which requires environment-specific training infrastructure), this
agent uses a learned scoring function over hand-crafted features.

The agent:
  - Inherits the same ``Agent`` interface (``setup``, ``step``, ``reset``).
  - Uses a simple state representation (operator positions, health, city control).
  - Discrete action space: for Move, pick the closest unowned city;
    for Shoot, pick the highest-damage target.
  - Includes a simple learned scoring function (2-layer MLP) that can be
    trained offline on collected self-play experience.

Architecture
------------
The scoring network is a 2-layer MLP::

    state_features -> Linear(state_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)

State features (per operator-action pair):
  - Operator position (hex row/col, normalised)
  - Operator health fraction
  - Operator type (one-hot: infantry, vehicle, aircraft)
  - Nearest city distance
  - Nearest unowned city distance
  - Number of visible enemies
  - Nearest enemy distance
  - City control fraction (owned / total)

This gives a compact 16-dimensional feature vector per operator-action pair.

Usage
-----
::

    from ai.rl_baseline_agent import LearnedHeuristicAgent

    agent = LearnedHeuristicAgent()
    agent.setup(setup_info)
    actions = agent.step(observation)

To train offline on collected experience::

    agent.train_from_experience(experience_records, epochs=50)

To save/load learned weights::

    agent.save_policy("ai/weights_rl_baseline/policy.pt")
    agent.load_policy("ai/weights_rl_baseline/policy.pt")
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_FEATURE_DIM = 16
DEFAULT_HIDDEN_DIM = 64
MAX_MAP_SIZE = 30  # for normalising hex coordinates


class ActionType:
    """Mirrors ai.agent.ActionType for standalone use."""
    (
        Move,
        Shoot,
        GetOn,
        GetOff,
        Occupy,
        ChangeState,
        RemoveKeep,
        JMPlan,
        GuideShoot,
        StopMove,
        WeaponLock,
        WeaponUnFold,
        CancelJMPlan,
        Fork,
        Union,
        ChangeAltitude,
        ActivateRadar,
        EnterFort,
        ExitFort,
        LayMine,
    ) = range(1, 21)


class BopType:
    Infantry, Vehicle, Aircraft = range(1, 4)


# ---------------------------------------------------------------------------
# Scoring network
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE and nn is not None:

    class ActionScoringMLP(nn.Module):
        """2-layer MLP that scores an operator-action pair.

        Input: STATE_FEATURE_DIM-dimensional feature vector.
        Output: scalar score (logit, higher = better action).
        """

        def __init__(
            self,
            input_dim: int = STATE_FEATURE_DIM,
            hidden_dim: int = DEFAULT_HIDDEN_DIM,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

else:

    class ActionScoringMLP:  # type: ignore[no-redef]
        """Stub when torch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, x: Any) -> float:
            return 0.0


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _hex_to_rc(hex_pos: int) -> Tuple[int, int]:
    """Convert hex position integer to (row, col)."""
    return divmod(hex_pos, 100)


def _normalise_position(hex_pos: int) -> Tuple[float, float]:
    """Normalise hex position to [0, 1] range."""
    row, col = _hex_to_rc(hex_pos)
    return row / MAX_MAP_SIZE, col / MAX_MAP_SIZE


def _hex_distance(pos1: int, pos2: int) -> int:
    """Axial hex distance between two positions."""
    r1, c1 = _hex_to_rc(pos1)
    q1 = c1 - (r1 - (r1 & 1)) // 2
    s1 = -q1 - r1
    r2, c2 = _hex_to_rc(pos2)
    q2 = c2 - (r2 - (r2 & 1)) // 2
    s2 = -q2 - r2
    return (abs(q1 - q2) + abs(r1 - r2) + abs(s1 - s2)) // 2


def _one_hot_type(bop_type: int) -> List[float]:
    """One-hot encode operator type (infantry, vehicle, aircraft)."""
    vec = [0.0, 0.0, 0.0]
    if bop_type == BopType.Infantry:
        vec[0] = 1.0
    elif bop_type == BopType.Vehicle:
        vec[1] = 1.0
    elif bop_type == BopType.Aircraft:
        vec[2] = 1.0
    return vec


def extract_state_features(
    operator: Dict[str, Any],
    observation: Dict[str, Any],
    faction: int,
    action_type: str = "move",
    candidate: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Extract a fixed-size feature vector for an operator-action pair.

    Parameters
    ----------
    operator:
        The operator dict from the observation.
    observation:
        Full observation frame.
    faction:
        The agent's faction id (0 or 1).
    action_type:
        The action type being evaluated (``"move"`` or ``"shoot"``).
    candidate:
        The action candidate dict (e.g. move target or shoot target).

    Returns
    -------
    np.ndarray
        Feature vector of shape ``(STATE_FEATURE_DIM,)``.
    """
    features = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)
    cur_hex = int(operator.get("cur_hex", 0))

    # 0-1: Normalised position
    norm_r, norm_c = _normalise_position(cur_hex)
    features[0] = norm_r
    features[1] = norm_c

    # 2: Health fraction
    max_blood = float(operator.get("max_blood", 100.0) or 100.0)
    blood = float(operator.get("blood", max_blood) or max_blood)
    features[2] = min(1.0, max(0.0, blood / max(1.0, max_blood)))

    # 3-5: Type one-hot
    type_vec = _one_hot_type(int(operator.get("type", 0)))
    features[3] = type_vec[0]
    features[4] = type_vec[1]
    features[5] = type_vec[2]

    # Compute city-related features
    cities = observation.get("cities", [])
    total_cities = max(1, len(cities))
    owned_cities = sum(1 for c in cities if int(c.get("flag", -1)) == faction)

    # 6: City control fraction
    features[6] = owned_cities / total_cities

    # 7: Nearest city distance (normalised)
    min_city_dist = 999.0
    min_unowned_dist = 999.0
    for city in cities:
        city_hex = int(city.get("coord", 0))
        dist = _hex_distance(cur_hex, city_hex)
        if dist < min_city_dist:
            min_city_dist = dist
        if int(city.get("flag", -1)) != faction and dist < min_unowned_dist:
            min_unowned_dist = dist

    features[7] = min(min_city_dist / 20.0, 1.0)
    # 8: Nearest unowned city distance (normalised)
    features[8] = min(min_unowned_dist / 20.0, 1.0)

    # Enemy-related features
    operators = observation.get("operators", [])
    enemies = [
        op for op in operators
        if op.get("color") != faction and op.get("obj_id") is not None
    ]
    friendlies = [
        op for op in operators
        if op.get("color") == faction and op.get("obj_id") is not None
    ]

    # 9: Number of visible enemies (normalised)
    features[9] = min(len(enemies) / 10.0, 1.0)

    # 10: Nearest enemy distance (normalised)
    min_enemy_dist = 999.0
    for enemy in enemies:
        enemy_hex = int(enemy.get("cur_hex", 0))
        dist = _hex_distance(cur_hex, enemy_hex)
        if dist < min_enemy_dist:
            min_enemy_dist = dist
    features[10] = min(min_enemy_dist / 20.0, 1.0)

    # 11: Number of friendlies (normalised)
    features[11] = min(len(friendlies) / 10.0, 1.0)

    # 12: Action type indicator (0=move, 1=shoot)
    features[12] = 1.0 if action_type == "shoot" else 0.0

    # 13: Candidate distance (if applicable, normalised)
    if candidate is not None:
        if "target_obj_id" in candidate:
            # Shoot candidate: distance to target
            target_id = int(candidate["target_obj_id"])
            target_op = next(
                (op for op in operators if op.get("obj_id") == target_id), None
            )
            if target_op is not None:
                target_hex = int(target_op.get("cur_hex", 0))
                features[13] = min(_hex_distance(cur_hex, target_hex) / 20.0, 1.0)
        elif "coord" in candidate:
            # Move candidate (city)
            city_hex = int(candidate["coord"])
            features[13] = min(_hex_distance(cur_hex, city_hex) / 20.0, 1.0)

    # 14: Candidate expected damage (normalised, for shoot actions)
    if candidate is not None and action_type == "shoot":
        damage = float(candidate.get("damage", 0.0) or 0.0)
        hit_prob = float(candidate.get("hit_prob", 0.0) or 0.0)
        features[14] = min((damage * hit_prob) / 100.0, 1.0)

    # 15: Candidate city value (normalised, for move actions)
    if candidate is not None and "value" in candidate:
        features[15] = min(float(candidate.get("value", 0.0) or 0.0) / 100.0, 1.0)

    return features


# ---------------------------------------------------------------------------
# Tactical heuristic scorer (non-learned fallback)
# ---------------------------------------------------------------------------

def heuristic_move_score(
    operator: Dict[str, Any],
    city: Dict[str, Any],
    observation: Dict[str, Any],
    faction: int,
) -> float:
    """Rule-based score for a move-to-city action.

    Higher is better.  Considers:
    - City ownership (prefer unowned/enemy cities)
    - City value
    - Distance (prefer closer cities)
    - Threat avoidance
    """
    cur_hex = int(operator.get("cur_hex", 0))
    city_hex = int(city.get("coord", 0))
    city_flag = int(city.get("flag", -1))
    city_value = float(city.get("value", 0.0) or 0.0)

    score = city_value * 0.5

    # Ownership bonus
    if city_flag == faction:
        score -= 30.0  # already owned, less interesting
    elif city_flag >= 0:
        score += 60.0  # enemy city, high value target
    else:
        score += 25.0  # unowned, moderate value

    # Distance penalty
    dist = _hex_distance(cur_hex, city_hex)
    score -= dist * 2.0

    # Proximity bonus for adjacent cities
    if dist <= 1 and city_flag != faction:
        score += 50.0

    # Threat: count nearby enemies around the target city
    enemies = [
        op for op in observation.get("operators", [])
        if op.get("color") != faction
    ]
    for enemy in enemies:
        enemy_hex = int(enemy.get("cur_hex", 0))
        enemy_dist = _hex_distance(enemy_hex, city_hex)
        if enemy_dist <= 3:
            score -= (4 - enemy_dist) * 8.0

    return float(score)


def heuristic_shoot_score(
    operator: Dict[str, Any],
    candidate: Dict[str, Any],
    observation: Dict[str, Any],
    faction: int,
) -> float:
    """Rule-based score for a shoot action.

    Higher is better.  Considers:
    - Expected damage
    - Kill potential
    - Target value
    """
    target_id = int(candidate.get("target_obj_id", -1))
    target = next(
        (op for op in observation.get("operators", []) if op.get("obj_id") == target_id),
        None,
    )
    if target is None:
        return 0.0

    damage = float(candidate.get("damage", 0.0) or 0.0)
    hit_prob = float(candidate.get("hit_prob", 0.0) or 0.0)
    expected_damage = damage * hit_prob
    target_hp = float(target.get("blood", 100.0) or 100.0)
    target_value = float(target.get("value", 0.0) or 0.0)

    score = expected_damage * 8.0
    score += target_value * 0.3

    # Kill bonus
    if expected_damage >= target_hp and target_hp > 0:
        score += 80.0

    # Low-HP target bonus (finish them off)
    if target_hp < 30.0 and target_hp > 0:
        score += 20.0

    return float(score)


# ---------------------------------------------------------------------------
# LearnedHeuristicAgent
# ---------------------------------------------------------------------------

class LearnedHeuristicAgent:
    """RL-inspired baseline agent using a learned scoring function.

    This agent implements the same interface as ``ai.agent.Agent``:
    - ``setup(setup_info)``
    - ``step(observation) -> List[Dict]``
    - ``reset()``

    Action selection uses either:
    1. The learned MLP scoring function (if weights are loaded/trained).
    2. A hand-crafted heuristic fallback (if no learned weights are available).

    The agent can be trained offline on collected self-play experience via
    ``train_from_experience()``.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        log_level: int = logging.INFO,
    ) -> None:
        self.seed = seed if seed is not None else 42
        self._rng = random.Random(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s %(levelname)s %(name)s %(message)s",
            )
        self.logger.setLevel(log_level)

        self.hidden_dim = hidden_dim
        self.device = "cpu"
        self.scoring_model: Optional[Any] = None
        self._model_trained = False

        # Agent state (mirrors ai.agent.Agent)
        self.scenario: Optional[Any] = None
        self.color: Optional[int] = None
        self.faction: Optional[int] = None
        self.seat: Optional[int] = None
        self.role: Optional[int] = None
        self.observation: Optional[Dict[str, Any]] = None
        self.map: Optional[Any] = None
        self.map_data: Optional[Any] = None
        self.controllable_ops: Optional[List[int]] = None
        self.my_direction: Optional[Dict[str, Any]] = None
        self.my_mission: Optional[Dict[str, Any]] = None
        self.pending_actions: List[Dict[str, Any]] = []

        # Diagnostics
        self.action_counts: Dict[str, int] = {}
        self.learned_action_count = 0
        self.heuristic_action_count = 0

        # Initialise model if torch is available
        if TORCH_AVAILABLE:
            self.scoring_model = ActionScoringMLP(
                input_dim=STATE_FEATURE_DIM,
                hidden_dim=self.hidden_dim,
            )
            self.scoring_model.eval()
            # Resolve device
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                self.device = "cuda"
                self.scoring_model.to(self.device)

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def setup(self, setup_info: Dict[str, Any]) -> None:
        """Initialise the agent from setup info (same interface as Agent.setup)."""
        try:
            self.scenario = setup_info.get("scenario")
            self.color = setup_info.get("faction")
            self.faction = setup_info.get("faction")
            self.seat = setup_info.get("seat")
            self.role = setup_info.get("role")

            # Build a lightweight map object if basic_data is available
            basic_data = setup_info.get("basic_data")
            if basic_data and "map_data" in basic_data:
                self.map_data = basic_data["map_data"]
            else:
                self.map_data = None

            self.observation = None
            self.controllable_ops = None
            self.my_direction = None
            self.my_mission = None
            self.pending_actions = []
            self.action_counts = {}
            self.learned_action_count = 0
            self.heuristic_action_count = 0

            # Try to load pre-trained weights
            weights_dir = setup_info.get("policy_artifacts_dir")
            if weights_dir:
                policy_path = Path(weights_dir) / "rl_baseline_policy.pt"
                if policy_path.exists():
                    self.load_policy(str(policy_path))

            self.logger.info(
                "LearnedHeuristicAgent.setup: seat=%s faction=%s model_trained=%s",
                self.seat, self.faction, self._model_trained,
            )
        except Exception:
            self.logger.critical("setup failed", exc_info=True)
            raise

    def reset(self) -> None:
        """Reset per-episode state."""
        self.observation = None
        self.controllable_ops = None
        self.my_direction = None
        self.my_mission = None
        self.pending_actions = []
        self.action_counts = {}
        self.learned_action_count = 0
        self.heuristic_action_count = 0
        self.logger.info("LearnedHeuristicAgent.reset complete")

    def step(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actions for the current observation.

        Parameters
        ----------
        observation:
            The observation dict from the environment.

        Returns
        -------
        list
            List of action dicts in the engine format.
        """
        if observation is None:
            return []

        self.observation = observation
        self.pending_actions = []

        # Handle deployment phase (stage == 1)
        if observation.get("time", {}).get("stage") == 1:
            return self._handle_deployment_phase(observation)

        # Determine controllable operators
        team_info = observation.get("role_and_grouping_info", {})
        if self.seat in team_info:
            self.controllable_ops = team_info[self.seat].get("operators", [])
        else:
            self.controllable_ops = [
                op["obj_id"] for op in observation.get("operators", [])
            ]

        # Parse mission/direction commands
        for command in observation.get("communication", []):
            cmd_type = command.get("type")
            cmd_info = command.get("info", {})
            if cmd_type in (200, 201) and cmd_info.get("company_id") == self.seat:
                if cmd_type == 200:
                    self.my_mission = command
                elif cmd_type == 201:
                    self.my_direction = command

        # Process valid actions
        valid_actions = observation.get("valid_actions", {})
        total_actions: List[Dict[str, Any]] = []

        priority = [
            ActionType.Shoot,
            ActionType.Move,
            ActionType.Occupy,
        ]

        for obj_id, action_map in valid_actions.items():
            if obj_id not in (self.controllable_ops or []):
                continue
            if not isinstance(action_map, dict):
                continue

            for action_type in priority:
                if action_type not in action_map:
                    continue

                candidates = action_map[action_type]
                action = self._select_action(obj_id, action_type, candidates)
                if action is not None:
                    total_actions.append(action)
                    action_name = {
                        ActionType.Shoot: "Shoot",
                        ActionType.Move: "Move",
                        ActionType.Occupy: "Occupy",
                    }.get(action_type, str(action_type))
                    self.action_counts[action_name] = self.action_counts.get(action_name, 0) + 1
                    break

        self.logger.info(
            "step: %d actions generated (learned=%d, heuristic=%d)",
            len(total_actions),
            self.learned_action_count,
            self.heuristic_action_count,
        )
        return total_actions

    # ------------------------------------------------------------------
    # Internal action selection
    # ------------------------------------------------------------------

    def _handle_deployment_phase(
        self, observation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Handle stage 1 (deployment) with simple default actions."""
        actions: List[Dict[str, Any]] = []
        for item in observation.get("operators", []):
            if item.get("obj_id") in (self.controllable_ops or []):
                if item.get("sub_type") in [2, 4]:
                    actions.append({
                        "actor": self.seat,
                        "obj_id": item["obj_id"],
                        "type": 303,
                        "target_obj_id": item.get("launcher"),
                    })
        actions.append({"actor": self.seat, "type": 333})
        return actions

    def _select_action(
        self,
        obj_id: int,
        action_type: int,
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Select the best action for a given operator and action type."""
        if action_type == ActionType.Occupy:
            return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.Occupy}

        if not candidates:
            return None

        operator = self._get_operator(obj_id)
        if operator is None:
            return None

        if action_type == ActionType.Shoot:
            return self._select_shoot(operator, candidates)
        elif action_type == ActionType.Move:
            return self._select_move(operator, candidates)

        return None

    def _get_operator(self, obj_id: int) -> Optional[Dict[str, Any]]:
        """Look up an operator by obj_id in the current observation."""
        if self.observation is None:
            return None
        for op in self.observation.get("operators", []):
            if op.get("obj_id") == obj_id:
                return op
        return None

    def _select_shoot(
        self,
        operator: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Select the best shoot target using learned or heuristic scoring."""
        if not candidates:
            return None

        best_candidate: Optional[Dict[str, Any]] = None
        best_score: float = float("-inf")
        used_learned = False

        for candidate in candidates:
            if self._model_trained and self.scoring_model is not None:
                features = extract_state_features(
                    operator, self.observation or {}, self.faction or 0,
                    action_type="shoot", candidate=candidate,
                )
                score = self._score_with_model(features)
                used_learned = True
            else:
                score = heuristic_shoot_score(
                    operator, candidate, self.observation or {}, self.faction or 0,
                )

            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            return None

        if used_learned:
            self.learned_action_count += 1
        else:
            self.heuristic_action_count += 1

        return {
            "actor": self.seat,
            "obj_id": operator.get("obj_id"),
            "type": ActionType.Shoot,
            "target_obj_id": best_candidate.get("target_obj_id"),
            "weapon_id": best_candidate.get("weapon_id"),
        }

    def _select_move(
        self,
        operator: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Select the best move target using learned or heuristic scoring.

        If move candidates (paths) are empty, falls back to evaluating cities
        from the observation.
        """
        if not candidates and not (self.observation or {}).get("cities"):
            return None

        cities = (self.observation or {}).get("cities", [])
        cur_hex = int(operator.get("cur_hex", 0))

        # Check if operator is on an unowned city (should occupy instead)
        is_on_unowned_city = any(
            int(c.get("coord", -1)) == cur_hex and int(c.get("flag", -1)) != (self.faction or -1)
            for c in cities
        )
        if is_on_unowned_city:
            return None

        # Determine move destinations
        move_targets: List[Dict[str, Any]] = []
        if candidates:
            # Use provided candidates (may contain move_path entries)
            for cand in candidates:
                move_targets.append(cand)
        else:
            # Evaluate all cities as potential move destinations
            move_targets = list(cities)

        if not move_targets:
            return None

        best_target: Optional[Dict[str, Any]] = None
        best_score: float = float("-inf")
        used_learned = False

        for target in move_targets:
            if self._model_trained and self.scoring_model is not None:
                features = extract_state_features(
                    operator, self.observation or {}, self.faction or 0,
                    action_type="move", candidate=target,
                )
                score = self._score_with_model(features)
                used_learned = True
            else:
                score = heuristic_move_score(
                    operator, target, self.observation or {}, self.faction or 0,
                )

            if score > best_score:
                best_score = score
                best_target = target

        if best_target is None:
            return None

        if used_learned:
            self.learned_action_count += 1
        else:
            self.heuristic_action_count += 1

        # Build move route
        destination = int(best_target.get("coord", 0))
        if destination == 0 or destination == cur_hex:
            return None

        # Simple route: just the destination hex (engine handles pathfinding)
        # In a full implementation, we'd use the Map object for A* routing
        move_path = self._build_simple_route(cur_hex, destination)
        if not move_path:
            return None

        return {
            "actor": self.seat,
            "obj_id": operator.get("obj_id"),
            "type": ActionType.Move,
            "move_path": move_path,
        }

    def _build_simple_route(self, start: int, end: int) -> List[int]:
        """Build a simple move route (single-step toward destination).

        In production, this would use the Map's A* pathfinder.
        Here we return just the destination as a single-step move.
        """
        if start == end:
            return []
        # Return destination as a single-step route
        # The engine will validate and expand if needed
        return [end]

    def _score_with_model(self, features: np.ndarray) -> float:
        """Run the scoring MLP on a feature vector."""
        if not TORCH_AVAILABLE or self.scoring_model is None:
            return 0.0
        self.scoring_model.eval()
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        if self.device != "cpu":
            tensor = tensor.to(self.device)
        with torch.inference_mode():
            score = self.scoring_model(tensor).item()
        return float(score)

    # ------------------------------------------------------------------
    # Offline training
    # ------------------------------------------------------------------

    def train_from_experience(
        self,
        experience_records: List[Dict[str, Any]],
        *,
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """Train the scoring MLP offline on collected experience.

        Each experience record should contain:
        - ``"features"``: np.ndarray of shape ``(STATE_FEATURE_DIM,)``
        - ``"reward"``: float (e.g. +1 for good outcome, -1 for bad)

        Parameters
        ----------
        experience_records:
            List of dicts with ``"features"`` and ``"reward"`` keys.
        epochs:
            Number of training epochs.
        batch_size:
            Mini-batch size.
        learning_rate:
            Adam learning rate.
        validation_split:
            Fraction of data held out for validation.

        Returns
        -------
        dict
            Training metrics (loss history, validation loss).
        """
        if not TORCH_AVAILABLE or self.scoring_model is None:
            self.logger.warning("Cannot train: torch unavailable")
            return {"error": "torch_unavailable"}

        if not experience_records:
            self.logger.warning("Cannot train: no experience records")
            return {"error": "no_data"}

        # Build tensors
        features = np.stack([r["features"] for r in experience_records]).astype(np.float32)
        rewards = np.array([r["reward"] for r in experience_records], dtype=np.float32)

        # Normalise features
        feat_mean = features.mean(axis=0)
        feat_std = features.std(axis=0)
        feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
        features = (features - feat_mean) / feat_std

        # Train/validation split
        n = len(features)
        n_val = max(1, int(n * validation_split))
        indices = np.random.permutation(n)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        train_features = torch.tensor(features[train_idx], dtype=torch.float32)
        train_rewards = torch.tensor(rewards[train_idx], dtype=torch.float32).unsqueeze(1)
        val_features = torch.tensor(features[val_idx], dtype=torch.float32)
        val_rewards = torch.tensor(rewards[val_idx], dtype=torch.float32).unsqueeze(1)

        if self.device != "cpu":
            train_features = train_features.to(self.device)
            train_rewards = train_rewards.to(self.device)
            val_features = val_features.to(self.device)
            val_rewards = val_rewards.to(self.device)

        # Reset model for training
        self.scoring_model = ActionScoringMLP(
            input_dim=STATE_FEATURE_DIM,
            hidden_dim=self.hidden_dim,
        )
        if self.device != "cpu":
            self.scoring_model.to(self.device)

        optimizer = optim.Adam(self.scoring_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        train_losses: List[float] = []
        val_losses: List[float] = []

        self.scoring_model.train()
        for epoch in range(epochs):
            # Mini-batch training
            perm = torch.randperm(len(train_features))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_features), batch_size):
                batch_idx = perm[i:i + batch_size]
                batch_features = train_features[batch_idx]
                batch_rewards = train_rewards[batch_idx]

                optimizer.zero_grad()
                predictions = self.scoring_model(batch_features)
                loss = criterion(predictions, batch_rewards)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(1, n_batches)
            train_losses.append(avg_train_loss)

            # Validation
            self.scoring_model.eval()
            with torch.inference_mode():
                val_pred = self.scoring_model(val_features)
                val_loss = criterion(val_pred, val_rewards).item()
            val_losses.append(val_loss)
            self.scoring_model.train()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.logger.info(
                    "Epoch %d/%d: train_loss=%.4f val_loss=%.4f",
                    epoch + 1, epochs, avg_train_loss, val_loss,
                )

        self.scoring_model.eval()
        self._model_trained = True

        # Store normalisation stats for inference
        self._feature_mean = feat_mean
        self._feature_std = feat_std

        training_result = {
            "train_loss_history": train_losses,
            "val_loss_history": val_losses,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "n_samples": n,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "epochs": epochs,
            "feature_mean": feat_mean.tolist(),
            "feature_std": feat_std.tolist(),
        }

        self.logger.info(
            "Training complete: final_train_loss=%.4f final_val_loss=%.4f n_samples=%d",
            train_losses[-1] if train_losses else 0,
            val_losses[-1] if val_losses else 0,
            n,
        )
        return training_result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_policy(self, path: str) -> None:
        """Save the learned scoring model to disk.

        Parameters
        ----------
        path:
            File path for the saved model (e.g. ``"ai/weights_rl_baseline/policy.pt"``).
        """
        if not TORCH_AVAILABLE or self.scoring_model is None:
            self.logger.warning("Cannot save: torch unavailable or no model")
            return

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state": self.scoring_model.state_dict(),
            "hidden_dim": self.hidden_dim,
            "input_dim": STATE_FEATURE_DIM,
            "model_trained": self._model_trained,
            "feature_mean": getattr(self, "_feature_mean", None),
            "feature_std": getattr(self, "_feature_std", None),
        }
        torch.save(payload, str(out_path))
        self.logger.info("Policy saved to %s (trained=%s)", path, self._model_trained)

    def load_policy(self, path: str) -> bool:
        """Load a previously saved scoring model.

        Parameters
        ----------
        path:
            File path to the saved model.

        Returns
        -------
        bool
            True if loaded successfully.
        """
        if not TORCH_AVAILABLE:
            self.logger.warning("Cannot load: torch unavailable")
            return False

        policy_path = Path(path)
        if not policy_path.exists():
            self.logger.warning("Policy file not found: %s", path)
            return False

        try:
            payload = torch.load(str(policy_path), map_location="cpu")
            hidden_dim = int(payload.get("hidden_dim", DEFAULT_HIDDEN_DIM))
            self.hidden_dim = hidden_dim
            self.scoring_model = ActionScoringMLP(
                input_dim=STATE_FEATURE_DIM,
                hidden_dim=hidden_dim,
            )
            self.scoring_model.load_state_dict(payload["model_state"])
            self.scoring_model.eval()
            if self.device != "cpu":
                self.scoring_model.to(self.device)

            self._model_trained = bool(payload.get("model_trained", True))
            if payload.get("feature_mean") is not None:
                self._feature_mean = np.array(payload["feature_mean"], dtype=np.float32)
            if payload.get("feature_std") is not None:
                self._feature_std = np.array(payload["feature_std"], dtype=np.float32)

            self.logger.info(
                "Policy loaded from %s (trained=%s, hidden_dim=%d)",
                path, self._model_trained, hidden_dim,
            )
            return True
        except Exception:
            self.logger.critical("Failed to load policy from %s", path, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics about the agent's behaviour."""
        total = self.learned_action_count + self.heuristic_action_count
        return {
            "agent_type": "LearnedHeuristicAgent",
            "model_trained": self._model_trained,
            "device": self.device,
            "hidden_dim": self.hidden_dim,
            "state_feature_dim": STATE_FEATURE_DIM,
            "total_actions": total,
            "learned_actions": self.learned_action_count,
            "heuristic_actions": self.heuristic_action_count,
            "learned_fraction": self.learned_action_count / max(1, total),
            "action_type_counts": dict(self.action_counts),
        }


# ---------------------------------------------------------------------------
# Entry point for standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    agent = LearnedHeuristicAgent(seed=42)
    log.info("Agent created: %s", agent.get_diagnostics())

    # Test with synthetic data
    if TORCH_AVAILABLE:
        log.info("Testing offline training with synthetic data...")
        synthetic_experience = []
        for _ in range(200):
            features = np.random.randn(STATE_FEATURE_DIM).astype(np.float32)
            # Reward correlates with first feature (simulates a learnable signal)
            reward = float(features[0] > 0) * 2.0 - 1.0
            synthetic_experience.append({"features": features, "reward": reward})

        result = agent.train_from_experience(
            synthetic_experience,
            epochs=20,
            batch_size=32,
        )
        log.info("Training result: %s", {
            k: v for k, v in result.items() if k != "feature_mean" and k != "feature_std"
        })

        # Test inference
        test_features = np.random.randn(STATE_FEATURE_DIM).astype(np.float32)
        score = agent._score_with_model(test_features)
        log.info("Test inference score: %.4f", score)

        # Test save/load
        test_path = "experiments/test_rl_baseline/policy.pt"
        agent.save_policy(test_path)
        agent2 = LearnedHeuristicAgent(seed=99)
        loaded = agent2.load_policy(test_path)
        log.info("Load successful: %s", loaded)

    log.info("Diagnostics: %s", agent.get_diagnostics())
