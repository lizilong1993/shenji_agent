import random
import logging

class MockTrainEnv:
    def __init__(self):
        self.logger = logging.getLogger("MockTrainEnv")
        self.step_count = 0
        self.max_steps = 100
        self.red_score = 0
        self.blue_score = 0
        
    def setup(self, env_step_info):
        self.logger.info("MockTrainEnv: Setup")
        self.step_count = 0
        self.red_score = 0
        self.blue_score = 0
        return self._get_state()
        
    def step(self, actions):
        self.step_count += 1
        
        # Simple rule: Shoot actions increase score
        for action in actions:
            actor = action.get('actor')
            type_ = action.get('type')
            # ActionType.Shoot = 2 (approx, need to check enum)
            # From agent.py: Shoot=2, GuideShoot=9...
            # Actually ActionType class: Shoot is 2? 
            # range(1, 21) -> Move=1, Shoot=2...
            
            if type_ == 2: # Shoot
                if actor == 0: # Red
                    self.red_score += 10
                elif actor == 1: # Blue
                    self.blue_score += 10
            elif type_ == 1: # Move
                if actor == 0:
                    self.red_score += 1
                elif actor == 1:
                    self.blue_score += 1

        done = self.step_count >= self.max_steps
        
        # Random outcome at end if close
        if done:
            if self.red_score == self.blue_score:
                if random.random() > 0.5:
                    self.red_score += 1
                else:
                    self.blue_score += 1
                    
        state = self._get_state()
        
        # Inject reward into state
        state[0]['reward'] = self.red_score
        state[1]['reward'] = self.blue_score
        
        return state, done
        
    def reset(self):
        self.logger.info("MockTrainEnv: Reset")
        self.step_count = 0
        
    def _get_state(self):
        # Return a minimal valid state structure for Agent.step
        # Agent expects: operators, cities, communication, time, valid_actions
        
        # Mock Observation
        obs = {
            "time": {"stage": 2, "cur_step": self.step_count},
            "operators": [
                {"obj_id": 101, "cur_hex": 101, "type": 1, "sub_type": 2, "seat": 1, "launcher": 0, "blood": 100}, # Red Unit
                {"obj_id": 201, "cur_hex": 201, "type": 1, "sub_type": 2, "seat": 11, "launcher": 0, "blood": 100} # Blue Unit
            ],
            "cities": [{"coord": 303, "value": 50}],
            "communication": [],
            "role_and_grouping_info": {
                1: {"operators": [101]},
                11: {"operators": [201]}
            },
            "valid_actions": {
                101: { # Red actions
                    1: [{"move_path": [102]}], # Move
                    2: [{"target_obj_id": 201, "weapon_id": 1, "damage": 20, "hit_prob": 0.8, "target_blood": 100}] # Shoot
                },
                201: { # Blue actions
                    1: [{"move_path": [202]}],
                    2: [{"target_obj_id": 101, "weapon_id": 1, "damage": 20, "hit_prob": 0.8, "target_blood": 100}]
                }
            }
        }
        
        return {
            0: obs, # Red
            1: obs, # Blue
            -1: obs # Green/Global
        }
