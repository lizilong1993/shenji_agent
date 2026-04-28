import copy
import logging
import random

class MockTrainEnv:
    def __init__(self):
        self.logger = logging.getLogger("MockTrainEnv")
        self.step_count = 0
        self.max_steps = 100
        self.red_score = 0
        self.blue_score = 0
        self._shoot_rewards = {
            101: 5,
            102: 15,
            201: 5,
            202: 15,
        }
        
    def setup(self, env_step_info):
        self.logger.info("MockTrainEnv: Setup")
        self.step_count = 0
        self.red_score = 0
        self.blue_score = 0
        return self._get_state()
        
    def step(self, actions):
        self.step_count += 1
        
        # Learned policies should be able to separate "good" and "bad" targets in smoke tests.
        for action in actions:
            actor = action.get('actor')
            type_ = action.get('type')
            seat = int(actor) if actor is not None else None
            if type_ == 2:  # Shoot
                target_obj_id = int(action.get("target_obj_id", -1))
                reward = self._shoot_rewards.get(target_obj_id, 0)
                if seat == 1:
                    self.red_score += reward
                elif seat == 11:
                    self.blue_score += reward
            elif type_ == 1:  # Move
                if seat == 1:
                    self.red_score += 1
                elif seat == 11:
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
                {"obj_id": 101, "cur_hex": 101, "type": 1, "sub_type": 2, "seat": 1, "launcher": 0, "blood": 100},
                {"obj_id": 102, "cur_hex": 102, "type": 1, "sub_type": 2, "seat": 1, "launcher": 0, "blood": 20},
                {"obj_id": 201, "cur_hex": 201, "type": 1, "sub_type": 2, "seat": 11, "launcher": 0, "blood": 100},
                {"obj_id": 202, "cur_hex": 202, "type": 1, "sub_type": 2, "seat": 11, "launcher": 0, "blood": 20},
            ],
            "cities": [{"coord": 303, "value": 50}],
            "communication": [],
            "role_and_grouping_info": {
                1: {"operators": [101]},
                11: {"operators": [201]}
            },
            "valid_actions": {
                101: {
                    1: [{"move_path": [103]}],
                    2: [
                        {"target_obj_id": 201, "weapon_id": 1, "attack_level": 1, "damage": 10, "hit_prob": 0.4, "target_blood": 100},
                        {"target_obj_id": 202, "weapon_id": 1, "attack_level": 1, "damage": 20, "hit_prob": 0.9, "target_blood": 20},
                    ],
                },
                201: {
                    1: [{"move_path": [202]}],
                    2: [
                        {"target_obj_id": 101, "weapon_id": 1, "attack_level": 1, "damage": 10, "hit_prob": 0.4, "target_blood": 100},
                        {"target_obj_id": 102, "weapon_id": 1, "attack_level": 1, "damage": 20, "hit_prob": 0.9, "target_blood": 20},
                    ],
                }
            }
        }
        
        return {
            0: copy.deepcopy(obs),
            1: copy.deepcopy(obs),
            -1: copy.deepcopy(obs),
        }
