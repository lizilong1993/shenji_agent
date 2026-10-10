"""FeintAgent: executes a pre-registered deception (FEINT) for the Shenji dataset.

Design (collection_feint_plan.md):
- Phase 1 (steps < trigger_step): behave like a real attack (mission_type=0,
  Shoot enabled) so the approach trajectory is indistinguishable from assault.
- Phase 2 (steps >= trigger_step): clear the attack mission, inject a retreat
  direction command (type-201 semantics via my_direction), and deprioritize
  Shoot so the force withdraws without engaging.

The deception is pre-registered: the collector's raw actions carry the
`info.deception_plan = "feint_v1"` marker (see jobs.py / collection.py), and
the labeling pipeline consumes that plan tag to emit a Grade C FEINT segment.
Behavior alone is never sufficient for FEINT (ontology: requires_plan_tag).

Usage: manifest policy entries point agent_class at "ai.feint_agent:FeintAgent"
with pre_mission {mission_type: 0, target_pos: ...} and optional
feint_trigger_step / feint_direction in pre_mission info.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .agent import Agent, ActionType


class FeintAgent(Agent):
    def __init__(self, seed: int | None = None, log_level: int = 30) -> None:
        super().__init__(seed=seed, log_level=log_level)
        self.feint_trigger_step: int | None = None
        self.feint_direction: Any = None
        self._feint_armed = False

    def arm_feint(self, trigger_step: int | None, direction: Any) -> None:
        """Called by the collector after set_mission to arm the deception."""
        self.feint_trigger_step = trigger_step
        self.feint_direction = direction
        self._feint_armed = True
        self.logger.info(
            "FeintAgent armed trigger=%s direction=%s",
            self.feint_trigger_step, self.feint_direction,
        )

    def reset(self) -> None:
        super().reset()
        # re-arm after reset; trigger is re-evaluated from observation time

    def step(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        if observation is None:
            return super().step(observation)
        cur_step = observation.get("time", {}).get("cur_step", 0)
        triggered = (
            self._feint_armed
            and self.feint_trigger_step is not None
            and cur_step >= self.feint_trigger_step
            and self.my_mission is not None
        )
        if triggered:
            # switch from attack to withdrawal: clear mission, inject direction
            self.my_mission = None
            if self.feint_direction is not None:
                self.my_direction = {
                    "type": 201,
                    "info": {"target_pos": self.feint_direction},
                }
            self._feint_armed = False
            self.logger.info(
                "FeintAgent triggered at step %s: mission cleared, direction set",
                cur_step,
            )
        actions = super().step(observation)
        # suppress Shoot while withdrawing (after trigger)
        if not self._feint_armed and self.feint_trigger_step is not None and cur_step >= self.feint_trigger_step:
            filtered = []
            for a in actions:
                if a.get("type") in (ActionType.Shoot, ActionType.GuideShoot):
                    continue
                filtered.append(a)
            return filtered
        return actions
