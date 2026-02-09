"""A simple agent implementation.

DemoAgent class inherits from BaseAgent and implement all three
abstract methods: setup(), step() and reset().
"""
import json
import os
import random

from .base_agent import BaseAgent
from .map import Map


class Agent(BaseAgent):
    def __init__(self):
        self.scenario = None
        self.color = None
        self.priority = None
        self.observation = None
        self.map = None
        self.scenario_info = None
        self.map_data = None
        self.seat = None
        self.faction = None
        self.role = None
        self.controllable_ops = None
        self.team_info = None
        self.my_direction = None
        self.my_mission = None
        self.user_name = None
        self.user_id = None
        self.history = None

    def setup(self, setup_info):
        self.scenario = setup_info["scenario"]
        # self.get_scenario_info(setup_info["scenario"])
        self.color = setup_info["faction"]
        self.faction = setup_info["faction"]
        self.seat = setup_info["seat"]
        self.role = setup_info["role"]
        self.user_name = setup_info["user_name"]
        self.user_id = setup_info["user_id"]
        self.priority = {
            ActionType.Occupy: self.gen_occupy,
            ActionType.Shoot: self.gen_shoot,
            ActionType.GuideShoot: self.gen_guide_shoot,
            ActionType.JMPlan: self.gen_jm_plan,
            ActionType.LayMine: self.gen_lay_mine,
            ActionType.ActivateRadar: self.gen_activate_radar,
            ActionType.ChangeAltitude: self.gen_change_altitude,
            ActionType.GetOn: self.gen_get_on,
            ActionType.GetOff: self.gen_get_off,
            ActionType.Fork: self.gen_fork,
            ActionType.Union: self.gen_union,
            ActionType.EnterFort: self.gen_enter_fort,
            ActionType.ExitFort: self.gen_exit_fort,
            ActionType.Move: self.gen_move,
            ActionType.RemoveKeep: self.gen_remove_keep,
            ActionType.ChangeState: self.gen_change_state,
            ActionType.StopMove: self.gen_stop_move,
            ActionType.WeaponLock: self.gen_WeaponLock,
            ActionType.WeaponUnFold: self.gen_WeaponUnFold,
            ActionType.CancelJMPlan: self.gen_cancel_JM_plan
        }  # choose action by priority
        self.observation = None
        self.map = Map(
            setup_info["basic_data"],
            setup_info["cost_data"],
            setup_info["see_data"]
        )  # use 'Map' class as a tool
        self.map_data = self.map.get_map_data()

    # def command(self, observation):
    #     self.team_info = observation["role_and_grouping_info"]
    #     return (
    #         self.gen_grouping_info(observation)
    #         + self.gen_battle_direction_info(observation)
    #         + self.gen_battle_mission_info(observation)
    #     )

    # def deploy(self, observation):
    #     self.team_info = observation["role_and_grouping_info"]
    #     self.controllable_ops = observation["role_and_grouping_info"][self.seat][
    #         "operators"
    #     ]
    #     communications = observation["communication"]
    #     for command in communications:
    #         if command["info"]["company_id"] == self.seat:
    #             if command["type"] == 200:
    #                 self.my_mission = command
    #             elif command["type"] == 201:
    #                 self.my_direction = command
    #     actions = []
    #     for item in observation["operators"]:
    #         if item["obj_id"] in self.controllable_ops:
    #             operator = item
    #             if operator["sub_type"] == 2 or operator["sub_type"] == 4:
    #                 actions.append(
    #                     {
    #                         "actor": self.seat,
    #                         "obj_id": operator["obj_id"],
    #                         "type": 303,
    #                         "target_obj_id": operator["launcher"],
    #                     }
    #                 )
    #     return actions

    def reset(self):
        self.scenario = None
        self.color = None
        self.priority = None
        self.observation = None
        self.map = None
        self.scenario_info = None
        self.map_data = None

    def step(self, observation: dict):
        self.observation = observation  # save observation for later use
        self.team_info = observation["role_and_grouping_info"]
        self.controllable_ops = observation["role_and_grouping_info"][self.seat][
            "operators"
        ]
        communications = observation["communication"]
        for command in communications:
            if command["type"] in [200, 201] and command["info"]["company_id"] == self.seat:
                if command["type"] == 200:
                    self.my_mission = command
                elif command["type"] == 201:
                    self.my_direction = command
        total_actions = []

        if observation["time"]["stage"] == 1:
            actions = []
            for item in observation["operators"]:
                if item["obj_id"] in self.controllable_ops:
                    operator = item
                    if operator["sub_type"] == 2 or operator["sub_type"] == 4:
                        actions.append(
                            {
                                "actor": self.seat,
                                "obj_id": operator["obj_id"],
                                "type": 303,
                                "target_obj_id": operator["launcher"],
                            }
                        )
            actions.append({
                "actor": self.seat,
                "type": 333
            })
            return actions

        # loop all bops and their valid actions
        for obj_id, valid_actions in observation["valid_actions"].items():
            if obj_id not in self.controllable_ops:
                continue
            for (
                action_type
            ) in self.priority:  # 'dict' is order-preserving since Python 3.6
                if action_type not in valid_actions:
                    continue
                # find the action generation method based on type
                gen_action = self.priority[action_type]
                action = gen_action(obj_id, valid_actions[action_type])
                if action:
                    total_actions.append(action)
                    break  # one action per bop at a time
        # if total_actions:
        #     print(
        #         f'{self.color} actions at step: {observation["time"]["cur_step"]}', end='\n\t')
        #     print(total_actions)
        return total_actions

    def get_scenario_info(self, scenario: int):
        SCENARIO_INFO_PATH = os.path.join(
            os.path.dirname(__file__), f"scenario_{scenario}.json"
        )
        with open(SCENARIO_INFO_PATH, encoding="utf8") as f:
            self.scenario_info = json.load(f)

    def get_bop(self, obj_id):
        """Get bop in my observation based on its id."""
        for bop in self.observation["operators"]:
            if obj_id == bop["obj_id"]:
                return bop

    def gen_occupy(self, obj_id, candidate):
        """Generate occupy action."""
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.Occupy,
        }

    def gen_shoot(self, obj_id, candidate):
        """Generate shoot action with the highest attack level."""
        best = max(candidate, key=lambda x: x["attack_level"])
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.Shoot,
            "target_obj_id": best["target_obj_id"],
            "weapon_id": best["weapon_id"],
        }

    def gen_guide_shoot(self, obj_id, candidate):
        """Generate guide shoot action with the highest attack level."""
        best = max(candidate, key=lambda x: x["attack_level"])
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.GuideShoot,
            "target_obj_id": best["target_obj_id"],
            "weapon_id": best["weapon_id"],
            "guided_obj_id": best["guided_obj_id"],
        }

    def gen_jm_plan(self, obj_id, candidate):
        """Generate jm plan action aimed at a random city."""
        weapon_id = random.choice(candidate)["weapon_id"]
        jm_pos = random.choice([city["coord"] for city in self.observation["cities"]])
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.JMPlan,
            "jm_pos": jm_pos,
            "weapon_id": weapon_id,
        }

    def gen_get_on(self, obj_id, candidate):
        """Generate get on action with some probability."""
        get_on_prob = 0.5
        if random.random() < get_on_prob:
            target_obj_id = random.choice(candidate)["target_obj_id"]
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.GetOn,
                "target_obj_id": target_obj_id,
            }

    def gen_get_off(self, obj_id, candidate):
        """Generate get off action only if the bop is within some distance of a random city."""
        bop = self.get_bop(obj_id)
        destination = random.choice(
            [city["coord"] for city in self.observation["cities"]]
        )
        if bop and self.map.get_distance(bop["cur_hex"], destination) <= 10:
            target_obj_id = random.choice(candidate)["target_obj_id"]
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.GetOff,
                "target_obj_id": target_obj_id,
            }

    def gen_change_state(self, obj_id, candidate):
        """Generate change state action with some probability."""
        change_state_prob = 0.001
        if random.random() < change_state_prob:
            target_state = random.choice(candidate)["target_state"]
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.ChangeState,
                "target_state": target_state,
            }

    def gen_remove_keep(self, obj_id, candidate):
        """Generate remove keep action with some probability."""
        remove_keep_prob = 0.2
        if random.random() < remove_keep_prob:
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.RemoveKeep,
            }

    def gen_move(self, obj_id, candidate):
        """Generate move action to a random city."""
        bop = self.get_bop(obj_id)
        if bop["sub_type"] == 3:
            return
        destination = random.choice(
            [city["coord"] for city in self.observation["cities"]]
        )
        if self.my_direction:
            destination = self.my_direction["info"]["target_pos"]
        if bop and bop["cur_hex"] != destination:
            move_type = self.get_move_type(bop)
            route = self.map.gen_move_route(bop["cur_hex"], destination, move_type)
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.Move,
                "move_path": route,
            }

    def get_move_type(self, bop):
        """Get appropriate move type for a bop."""
        bop_type = bop["type"]
        if bop_type == BopType.Vehicle:
            if bop["move_state"] == MoveType.March:
                move_type = MoveType.March
            else:
                move_type = MoveType.Maneuver
        elif bop_type == BopType.Infantry:
            move_type = MoveType.Walk
        else:
            move_type = MoveType.Fly
        return move_type

    def gen_stop_move(self, obj_id, candidate):
        """Generate stop move action only if the bop is within some distance of a random city.

        High probability for the bop with passengers and low for others.
        """
        bop = self.get_bop(obj_id)
        destination = random.choice(
            [city["coord"] for city in self.observation["cities"]]
        )
        if self.map.get_distance(bop["cur_hex"], destination) <= 10:
            stop_move_prob = 0.9 if bop["passenger_ids"] else 0.01
            if random.random() < stop_move_prob:
                return {
                    "actor": self.seat,
                    "obj_id": obj_id,
                    "type": ActionType.StopMove,
                }

    def gen_WeaponLock(self, obj_id, candidate):
        bop = self.get_bop(obj_id)
        prob_weaponlock = 0.001
        if (
            max(self.map_data[bop["cur_hex"] // 100][bop["cur_hex"] % 100]["roads"]) > 0
            or random.random() < prob_weaponlock
        ):
            return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.WeaponLock}

    def gen_WeaponUnFold(self, obj_id, candidate):
        bop = self.get_bop(obj_id)
        destination = random.choice(
            [city["coord"] for city in self.observation["cities"]]
        )
        if self.map.get_distance(bop["cur_hex"], destination) <= 10:
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.WeaponUnFold,
            }

    def gen_cancel_JM_plan(self, obj_id, candidate):
        cancel_prob = 0.0001
        if random.random() < cancel_prob:
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.CancelJMPlan,
            }

    def gen_grouping_info(self, observation):
        def partition(lst, n):
            return [lst[i::n] for i in range(n)]

        operator_ids = []
        for operator in observation["operators"] + observation["passengers"]:
            if operator["color"] == self.color:
                operator_ids.append(operator["obj_id"])
        lists_of_ops = partition(operator_ids, len(self.team_info.keys()))
        grouping_info = {"actor": self.seat, "type": 100}
        info = {}
        for teammate_id in self.team_info.keys():
            info[teammate_id] = {"operators": lists_of_ops.pop()}
        grouping_info["info"] = info
        return [grouping_info]

    def gen_battle_direction_info(self, observation):
        direction_info = []
        for teammate_id in self.team_info.keys():
            direction = {
                "actor": self.seat,
                "type": 201,
                "info": {
                    "company_id": teammate_id,
                    "target_pos": random.choice(observation["cities"])["coord"],
                    "start_time": 0,
                    "end_time": 1800,
                },
            }
            direction_info.append(direction)
        return direction_info

    def gen_battle_mission_info(self, observation):
        mission_info = []
        for teammate_id in self.team_info.keys():
            mission = {
                "actor": self.seat,
                "type": 200,
                "info": {
                    "company_id": teammate_id,
                    "mission_type": random.randint(0, 2),
                    "target_pos": random.choice(observation["cities"])["coord"],
                    "route": [
                        random.randint(0, 9000),
                        random.randint(0, 9000),
                        random.randint(0, 9000),
                    ],
                    "start_time": 0,
                    "end_time": 1800,
                },
            }
            mission_info.append(mission)
        return mission_info

    def gen_fork(self, obj_id, candidate):
        prob = 0.01
        if random.random() < prob:
            return None
        return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.Fork}

    def gen_union(self, obj_id, candidate):
        prob = 0.1
        if random.random() < prob:
            return None
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "target_obj_id": random.choice(candidate)["target_obj_id"],
            "type": ActionType.Union,
        }

    def gen_change_altitude(self, obj_id, candidate):
        prob = 0.05
        if random.random() < prob:
            return None
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "target_obj_id": random.choice(candidate)["target_altitude"],
            "type": ActionType.ChangeAltitude,
        }

    def gen_activate_radar(self, obj_id, candidate):
        prob = 1
        if random.random() < prob:
            return None
        return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.ActivateRadar}

    def gen_enter_fort(self, obj_id, candidate):
        prob = 0.5
        if random.random() < prob:
            return None
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.EnterFort,
            "target_obj_id": random.choice(candidate)["target_obj_id"],
        }

    def gen_exit_fort(self, obj_id, candidate):
        prob = 0.1
        if random.random() < prob:
            return None
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": ActionType.ExitFort,
            "target_obj_id": random.choice(candidate)["target_obj_id"],
        }

    def gen_lay_mine(self, obj_id, candidate):
        prob = 1
        if random.random() < prob:
            return None
        return {
            "actor": self.seat,
            "obj_id": obj_id,
            "type": 20,
            "target_pos": random.randint(0, 9177),
        }


class BopType:
    Infantry, Vehicle, Aircraft = range(1, 4)


class ActionType:
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


class MoveType:
    Maneuver, March, Walk, Fly = range(4)
