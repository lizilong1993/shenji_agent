"""EscortAgent: formation-follow controller (ESCORT intent) for Shenji dataset.

Behavior (coverage_gap_analysis.md):
- Leader unit moves on its own mission (attack/hold).
- Escort units follow the leader, keeping within ~3 hexes (bounded offset).
- When an enemy approaches the leader, escort units prioritize SHOOT
  (defensive engagement).

Implementation: minimal-invasion — each step the escort sets `my_direction`
to the leader's current hex so the base Agent's move logic routes toward the
leader; then delegates to super().step() which produces normal valid actions
(Move/Shoot/Occupy). The realized ESCORT label is derived by the labeling
detector from the cross-entity pattern; this controller only needs to
produce the behavior.

Usage: manifest policy agent_class = "ai.escort_agent:EscortAgent",
pre_mission {mission_type: 0, target_pos: ...}. The collector calls
arm_escort(lead_obj_id) with the leader's obj_id.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .agent import Agent


class EscortAgent(Agent):
    def __init__(self, seed: int | None = None, log_level: int = 30) -> None:
        super().__init__(seed=seed, log_level=log_level)
        self.escort_lead_obj: Any = None
        self._escort_armed = False

    def arm_escort(self, lead_obj: Any) -> None:
        self.escort_lead_obj = lead_obj
        self._escort_armed = True
        self.logger.info("EscortAgent armed lead=%s", lead_obj)

    def _leader_hex(self, observation: Dict[str, Any]) -> Any:
        # explicit lead, else first enemy (color != self.faction) unit
        if self.escort_lead_obj is not None:
            for op in observation.get("operators", []):
                if op.get("obj_id") == self.escort_lead_obj:
                    return op.get("cur_hex")
            return None
        for op in observation.get("operators", []):
            if op.get("color") != self.faction:
                return op.get("cur_hex")
        return None

    def step(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        if observation is None:
            return super().step(observation)
        if self._escort_armed:
            lead_hex = self._leader_hex(observation)
            if lead_hex is not None:
                # route escort move toward the leader's current hex
                self.my_direction = {"type": 201, "info": {"target_pos": lead_hex}}
        return super().step(observation)
