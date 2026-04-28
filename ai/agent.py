import heapq
import json
import logging
import os
import random
from copy import deepcopy
from functools import reduce
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .policy import LearnedPolicy



class BopType:
    Infantry, Vehicle, Aircraft = range(1, 4)


class BopSubTypes:
    PM = 0
    Infantry = 2


class UnitSubType:
    UGV = 3


class MissionType:
    Defense = 0
    Attack = 1
    Reconnaissance = 2


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


ACTION_TYPE_NAMES = {
    value: name
    for name, value in ActionType.__dict__.items()
    if not name.startswith("_") and isinstance(value, int)
}


class MoveType:
    Maneuver, March, Walk, Fly = range(4)


class Map:
    def __init__(self, basic_data: Dict[str, Any], cost_data: Sequence[Any], see_data: Optional[Any]):
        self.basic = basic_data["map_data"]
        self.max_row = len(self.basic)
        self.max_col = len(self.basic[0]) if self.max_row > 0 else 0
        self.cost = cost_data
        self.see = see_data

    def is_valid(self, pos: int) -> bool:
        row, col = divmod(pos, 100)
        return 0 <= row < self.max_row and 0 <= col < self.max_col

    def get_map_data(self) -> Any:
        return self.basic

    def get_neighbors(self, pos: int) -> List[int]:
        if not self.is_valid(pos):
            return []
        row, col = divmod(pos, 100)
        return self.basic[row][col]["neighbors"]

    def can_see(self, pos1: int, pos2: int, mode: int = 0) -> bool:
        if (
            not self.is_valid(pos1)
            or not self.is_valid(pos2)
            or self.see is None
            or not 0 <= mode < len(self.see)
        ):
            return False
        row1, col1 = divmod(pos1, 100)
        row2, col2 = divmod(pos2, 100)
        return bool(self.see[mode][row1, col1, row2, col2])

    def get_distance(self, pos1: int, pos2: int) -> int:
        if not self.is_valid(pos1) or not self.is_valid(pos2):
            return -1
        row1, col1 = divmod(pos1, 100)
        q1 = col1 - (row1 - (row1 & 1)) // 2
        r1 = row1
        s1 = -q1 - r1
        row2, col2 = divmod(pos2, 100)
        q2 = col2 - (row2 - (row2 & 1)) // 2
        r2 = row2
        s2 = -q2 - r2
        return (abs(q1 - q2) + abs(r1 - r2) + abs(s1 - s2)) // 2

    def gen_move_route(self, begin: int, end: int, mode: int) -> List[int]:
        if (
            not self.is_valid(begin)
            or not self.is_valid(end)
            or not 0 <= mode < len(self.cost)
            or begin == end
        ):
            return []
        frontier: List[Tuple[float, float, int]] = [(0.0, random.random(), begin)]
        cost_so_far: Dict[int, float] = {begin: 0}
        came_from: Dict[int, Optional[int]] = {begin: None}

        while frontier:
            _, _, cur = heapq.heappop(frontier)
            if cur == end:
                break
            row, col = divmod(cur, 100)
            for neigh, edge_cost in self.cost[mode][row][col].items():
                neigh_cost = cost_so_far[cur] + edge_cost
                if neigh not in cost_so_far or neigh_cost < cost_so_far[neigh]:
                    cost_so_far[neigh] = neigh_cost
                    came_from[neigh] = cur
                    heuristic = self.get_distance(neigh, end)
                    heapq.heappush(frontier, (neigh_cost + heuristic, random.random(), neigh))

        path: List[int] = []
        if end in came_from:
            cur = end
            while cur != begin:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
        return path

    def get_grid_distance(self, center: int, distance_start: int, distance_end: int) -> Set[int]:
        gridset: Set[int] = set()
        if not self.is_valid(center) or not 0 <= distance_start <= distance_end:
            return gridset
        if distance_end == 0:
            gridset.add(center)
            return gridset
        row = center // 100
        col = center % 100
        row1 = min(row + distance_end + 1, self.max_row)
        row0 = max(row - distance_end, 0)
        col1 = min(col + distance_end + 1, self.max_col)
        col0 = max(col - distance_end, 0)
        for row_x in range(row0, row1):
            for col_x in range(col0, col1):
                pos = row_x * 100 + col_x
                if self.is_valid(pos):
                    dis = self.get_distance(center, pos)
                    if distance_start <= dis <= distance_end:
                        gridset.add(pos)
        return gridset

    def get_road_type(self, pos: int) -> int:
        if not self.is_valid(pos):
            return 0
        for i in self.basic[pos // 100][pos % 100]["roads"]:
            if i != 0:
                return i
        return 0

    def get_grid_type(self, pos: int) -> int:
        if not self.is_valid(pos):
            return 0
        return self.basic[pos // 100][pos % 100]["cond"]

    def is_in_cover(self, pos: int) -> bool:
        return self.get_grid_type(pos) in [1, 2]

    def is_in_soft(self, coord: int) -> bool:
        return self.get_grid_type(coord) == 3

    def get_height(self, pos: int) -> int:
        if not self.is_valid(pos):
            return 0
        return self.basic[pos // 100][pos % 100]["elev"]

    def get_height_change_level(self, pos1: int, pos2: int) -> int:
        return (self.get_height(pos2) - self.get_height(pos1)) // 20

    def check_2bop_see(self, bop_a: Dict[str, Any], bop_b: Dict[str, Any], mod: int = 0) -> bool:
        if mod:
            bop_a, bop_b = bop_b, bop_a
        if not bop_a or not bop_b:
            return False
        pos_a = bop_a["cur_hex"]
        pos_b = bop_b["cur_hex"]
        hex_distance = self.get_distance(pos_a, pos_b)
        if isinstance(bop_a.get("observe_distance"), dict):
            max_distance = bop_a["observe_distance"].get(bop_b["type"], 0)
        elif isinstance(bop_a.get("observe_distance"), list):
            max_distance = bop_a["observe_distance"][bop_b["sub_type"]]
        else:
            return False
        if hex_distance > max_distance or max_distance == 0:
            return False
        fly_a = 1 if bop_a["type"] == 3 else 0
        fly_b = 1 if bop_b["type"] == 3 else 0
        mod = 1 if fly_a == 1 and fly_b == 1 else 2 if fly_a == 1 and fly_b == 0 else 3 if fly_a == 0 and fly_b == 1 else 0
        if not self.can_see(pos_a, pos_b, mod):
            return False
        distance_threshold = max_distance
        flag_in_hide = False
        if bop_b["type"] in [1, 2]:
            if bop_b.get("move_state") == 4:
                flag_in_hide = True
            if self.is_in_cover(pos_b):
                distance_threshold = distance_threshold // 2
            if flag_in_hide and bop_b["type"] == 2:
                if bop_a["type"] == 3:
                    flag_in_hide = False
                else:
                    if self.get_height_change_level(pos_a, pos_b) <= -1:
                        flag_in_hide = False
            if flag_in_hide:
                distance_threshold = distance_threshold // 2
        return hex_distance <= distance_threshold

    def get_PM_scope(self, grid: int) -> Set[int]:
        return self.get_grid_distance(grid, 0, 2) - set(self.get_grid_in_cover(self.get_grid_distance(grid, 2, 2)))

    def get_grid_in_cover(self, gridset: Set[int]) -> Set[int]:
        grids_in_cover = [pos for pos in gridset if self.is_valid(pos) and self.is_in_cover(pos)]
        return set(grids_in_cover)

    def get_hex_types(self, pos: int) -> Any:
        if not self.is_valid(pos):
            return 0
        basic_data = self.basic[pos // 100][pos % 100]
        return {
            "neighbors": basic_data["neighbors"],
            "roads": basic_data["roads"],
            "rivers": basic_data["rivers"],
            "cond": basic_data["cond"],
            "elev": basic_data["elev"],
        }


def get_direction(current: int, target: int) -> Optional[str]:
    dx = target % 100 - current % 100
    dy = target // 100 - current // 100
    if dx == 0 and dy == 0:
        return "原点"
    if dx == 0 and dy > 0:
        return "向下"
    if dx == 0 and dy < 0:
        return "向上"
    if dx < 0 and dy == 0:
        return "向左"
    if dx > 0 and dy == 0:
        return "向右"
    if dx < 0 and dy < 0:
        return "左上"
    if dx > 0 and dy > 0:
        return "右下"
    if dx < 0 < dy:
        return "左下"
    if dx > 0 > dy:
        return "右上"
    return None


def get_target_pos(agent: "Agent", cur_hex: int, direction: str) -> List[int]:
    neighbors = agent.map.get_neighbors(cur_hex)
    if not neighbors:
        return []
    if direction in ["向上", "左上"]:
        targets = neighbors[2:4]
    elif direction in ["向右", "右上"]:
        targets = neighbors[0:2]
    elif direction in ["向下", "右下"]:
        targets = [neighbors[0]] + ([neighbors[-1]] if len(neighbors) > 1 else [])
    else:
        targets = neighbors[3:5]
    if not targets:
        targets = neighbors[:1]
    if len(targets) == 1:
        return [targets[0]]
    cond_last = agent.map.get_hex_types(targets[-1])["cond"]
    cond_first = agent.map.get_hex_types(targets[0])["cond"]
    if cond_last < cond_first:
        return [targets[-1]]
    if cond_last == cond_first:
        return [random.choice(targets)]
    return [targets[0]]


def stand_line(
    current: int,
    target: int,
    UGVs: List[Dict[str, Any]],
    IFV: Dict[str, Any],
    IFVs: List[Dict[str, Any]],
    Tanks: List[Dict[str, Any]],
    UAV: List[Dict[str, Any]],
    Infantry: List[Dict[str, Any]],
    agent: "Agent",
) -> None:
    if not current:
        return
    direction = get_direction(current, target)
    if not direction:
        return
    base_hex = IFV["cur_hex"]
    if IFV in IFVs:
        IFVs.remove(IFV)
    if len(IFVs) > 0:
        for unit in IFVs:
            target_pos = get_target_pos(agent=agent, cur_hex=base_hex, direction=direction)
            base_hex = target_pos[-1]
            agent.act_gen.move(unit["obj_id"], target_pos)
            agent.flag_act[unit["obj_id"]] = True
    if len(UGVs) > 0:
        for unit in UGVs:
            target_pos = get_target_pos(agent=agent, cur_hex=base_hex, direction=direction)
            base_hex = target_pos[-1]
            agent.act_gen.move(unit["obj_id"], target_pos)
            agent.flag_act[unit["obj_id"]] = True
    if len(Tanks) > 0:
        for unit in Tanks:
            target_pos = get_target_pos(agent=agent, cur_hex=base_hex, direction=direction)
            base_hex = target_pos[-1]
            agent.act_gen.move(unit["obj_id"], target_pos)
            agent.flag_act[unit["obj_id"]] = True


def diffuse_grids(gridList: Iterable[int], cur_map: Map, r1: int, r0: int = 0) -> Set[int]:
    gridList_d = map(lambda x: cur_map.get_grid_distance(x, r0, r1), gridList)
    return reduce(lambda x, y: x | y, gridList_d, set())


def position_evaluate(
    see_enemy: List[Dict[str, Any]], bop: Dict[str, Any], cur_map: Map, mission_type: int
) -> bool:
    if mission_type < 0:
        return False
    if not see_enemy:
        return mission_type in [MissionType.Defense, MissionType.Reconnaissance]
    distances = [cur_map.get_distance(bop["cur_hex"], enemy["cur_hex"]) for enemy in see_enemy]
    min_distance = min(distances) if distances else 999
    if mission_type == MissionType.Attack:
        return min_distance <= 6
    if mission_type == MissionType.Defense:
        return 3 <= min_distance <= 8
    if mission_type == MissionType.Reconnaissance:
        return min_distance >= 4
    return False


def target_pos_select(
    bop: Dict[str, Any],
    see_enemy_bops: List[Dict[str, Any]],
    cur_map: Map,
    mission_type: int = -1,
) -> Optional[List[int]]:
    start = 1
    target_pos: List[int] = []
    tmp_bop = deepcopy(bop)
    see_enemy = [enemy for enemy in see_enemy_bops if enemy["type"] != BopType.Aircraft]

    def transfer_position(see_enemy_inner: List[Dict[str, Any]], bop_inner: Dict[str, Any], ranges: int) -> None:
        if mission_type < 0:
            return
        if ranges >= 20:
            return
        search_area = cur_map.get_grid_distance(bop_inner["cur_hex"], ranges, ranges)
        search_tmp_bop = deepcopy(bop_inner)
        for pos in search_area:
            if cur_map.is_in_soft(pos):
                continue
            search_tmp_bop["cur_hex"] = pos
            evaluate_result = position_evaluate(see_enemy_inner, search_tmp_bop, cur_map, mission_type)
            if evaluate_result and pos != bop_inner["cur_hex"]:
                target_pos.append(pos)
        if not target_pos:
            transfer_position(see_enemy_inner, bop_inner, ranges + 1)

    if 0 <= mission_type <= 2:
        transfer_position(see_enemy, tmp_bop, start)
    if target_pos:
        return target_pos
    return None


def aircraft_recon_cities(
    bop: Dict[str, Any],
    observation: Dict[str, Any],
    cur_map: Map,
    see_enemy_bops: List[Dict[str, Any]],
    mission_type: int = -1,
) -> Optional[List[int]]:
    if bop["type"] != BopType.Aircraft or mission_type < 0:
        return None
    if mission_type == 2:
        ct_scopeList = list(
            map(
                lambda ct: cur_map.get_grid_distance(ct["coord"], 0, 3)
                if ct["value"] == 50
                else cur_map.get_grid_distance(ct["coord"], 0, 3),
                observation["cities"],
            )
        )
        if ct_scopeList:
            need_recon_poss = reduce(lambda x, y: x | y, ct_scopeList)
            need_recon_poss = cur_map.get_grid_in_cover(need_recon_poss)
            if len(need_recon_poss):
                need_recon_poss = {
                    pos for pos in need_recon_poss
                    if sum(map(lambda op: cur_map.can_see(op["cur_hex"], pos), observation["operators"])) == 0
                }
        else:
            need_recon_poss = None
        return list(need_recon_poss) if need_recon_poss else None
    if mission_type in [0, 1]:
        see_enemy = [enemy for enemy in see_enemy_bops if enemy["type"] != BopType.Aircraft]
        if bop["sub_type"] == BopSubTypes.PM and mission_type == 0:
            target = None
            target_pos = None
            if not target and len(see_enemy):
                min_blood = 100
                for enemy in see_enemy:
                    if enemy["blood"] < min_blood:
                        target = enemy
                        min_blood = enemy["blood"]
            if not target:
                return None
            if target["cur_hex"] not in cur_map.get_PM_scope(bop["cur_hex"]):
                target_pos = target["cur_hex"]
            return list(target_pos) if target_pos else None
        need_recon_enemies = {enemy["cur_hex"] for enemy in see_enemy if enemy["sub_type"] in [0, 1, 2]}
        if observation["time"]["cur_step"] < 700:
            need_recon_enemies = need_recon_enemies - {
                infan["cur_hex"] for infan in see_enemy if infan["sub_type"] == BopSubTypes.Infantry
            }
        if not need_recon_enemies:
            return None
        return list(need_recon_enemies)
    return None


class ActionGenerator:
    def __init__(self, agent: "Agent"):
        self.agent = agent

    def move(self, obj_id: int, target_path: List[int]) -> Dict[str, Any]:
        action = {
            "actor": self.agent.seat,
            "obj_id": obj_id,
            "type": ActionType.Move,
            "move_path": target_path,
        }
        self.agent.pending_actions.append(action)
        return action


class Agent:
    def __init__(self, seed: Optional[int] = None, log_level: int = logging.INFO) -> None:
        self.seed = seed if seed is not None else 0
        self._rng = random.Random(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        self.logger.setLevel(log_level)
        self.scenario = None
        self.color = None
        self.priority = None
        self.observation = None
        self.map: Optional[Map] = None
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
        self.pending_actions: List[Dict[str, Any]] = []
        self.flag_act: Dict[int, bool] = {}
        self.act_gen = ActionGenerator(self)
        self.use_ai_optimization = False
        self.policy_profile = "baseline"
        self.policy_artifacts_dir = None
        self.policy = LearnedPolicy()
        self.valid_action_type_counts: Dict[str, int] = {}
        self.generated_action_type_counts: Dict[str, int] = {}

    def _action_name(self, action_type: int) -> str:
        return ACTION_TYPE_NAMES.get(int(action_type), str(action_type))

    def _reset_action_diagnostics(self) -> None:
        self.valid_action_type_counts = {}
        self.generated_action_type_counts = {}

    def setup(self, setup_info: Dict[str, Any]) -> None:
        try:
            requested_policy = setup_info.get("policy_profile")
            requested_optimization = bool(setup_info.get("use_ai_optimization", False))
            if setup_info.get("user_name") == "CurrentAI" and requested_policy is None:
                requested_policy = "challenger"
            self.policy_profile = str(requested_policy or ("challenger" if requested_optimization else "baseline"))
            self.use_ai_optimization = requested_optimization or self.policy_profile != "baseline"
            self.policy_artifacts_dir = setup_info.get("policy_artifacts_dir")

            self.scenario = setup_info.get("scenario")
            self.color = setup_info.get("faction")
            self.faction = setup_info.get("faction")
            self.seat = setup_info.get("seat")
            self.role = setup_info.get("role")
            self.user_name = setup_info.get("user_name")
            self.user_id = setup_info.get("user_id")
            self.priority = [
                ActionType.Shoot,
                ActionType.Move,
                ActionType.Occupy,
                ActionType.GuideShoot,
                ActionType.JMPlan,
                ActionType.LayMine,
                ActionType.ActivateRadar,
                ActionType.ChangeAltitude,
                ActionType.GetOn,
                ActionType.GetOff,
                ActionType.Fork,
                ActionType.Union,
                ActionType.EnterFort,
                ActionType.ExitFort,
                ActionType.RemoveKeep,
                ActionType.ChangeState,
                ActionType.StopMove,
                ActionType.WeaponLock,
                ActionType.WeaponUnFold,
                ActionType.CancelJMPlan,
            ]
            self.observation = None
            self._reset_action_diagnostics()
            self.map = Map(setup_info["basic_data"], setup_info["cost_data"], setup_info.get("see_data"))
            self.map_data = self.map.get_map_data()
            self.policy.configure(self.faction, self.policy_artifacts_dir, self.policy_profile)
            if self.use_ai_optimization and not self.policy.has_models():
                self.logger.warning(
                    "learned policy requested but canonical intent+gcn artifacts unavailable, falling back to rules"
                )
                self.use_ai_optimization = False
            self.logger.info("setup complete seat=%s faction=%s role=%s", self.seat, self.faction, self.role)
        except Exception:
            self.logger.critical("setup failed", exc_info=True)
            raise

    def reset(self) -> None:
        try:
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
            self.pending_actions = []
            self.flag_act = {}
            self._reset_action_diagnostics()
            self.policy.reset()
            self.logger.info("reset complete")
        except Exception:
            self.logger.critical("reset failed", exc_info=True)
            raise

    def step(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            if observation is None:
                self.logger.info("step received empty observation")
                return []
            self.observation = observation
            self.policy.observe(observation)
            obs_keys = list(observation.keys())
            reward = observation.get("reward")
            done = observation.get("done") or observation.get("terminated") or observation.get("truncated")
            self.logger.info("step observation keys=%s reward=%s done=%s", obs_keys, reward, done)
            self.team_info = observation.get("role_and_grouping_info", {})
            if self.seat in self.team_info:
                self.controllable_ops = self.team_info[self.seat].get("operators", [])
            else:
                self.controllable_ops = [op["obj_id"] for op in observation.get("operators", [])]
            communications = observation.get("communication", [])
            for command in communications:
                if command.get("type") in [200, 201] and command.get("info", {}).get("company_id") == self.seat:
                    if command.get("type") == 200:
                        self.my_mission = command
                    elif command.get("type") == 201:
                        self.my_direction = command
            if observation.get("time", {}).get("stage") == 1:
                actions: List[Dict[str, Any]] = []
                for item in observation.get("operators", []):
                    if item.get("obj_id") in self.controllable_ops:
                        operator = item
                        if operator.get("sub_type") in [2, 4]:
                            actions.append(
                                {
                                    "actor": self.seat,
                                    "obj_id": operator["obj_id"],
                                    "type": 303,
                                    "target_obj_id": operator.get("launcher"),
                                }
                            )
                actions.append({"actor": self.seat, "type": 333})
                self.logger.info("step actions_count=%s", len(actions))
                return actions
            total_actions: List[Dict[str, Any]] = []
            valid_actions = observation.get("valid_actions", {})
            for obj_id, action_map in valid_actions.items():
                if obj_id not in self.controllable_ops:
                    continue
                for available_action_type in action_map:
                    name = self._action_name(int(available_action_type))
                    self.valid_action_type_counts[name] = self.valid_action_type_counts.get(name, 0) + 1
                for action_type in self.priority or []:
                    if action_type not in action_map:
                        continue
                    action = self._generate_action(obj_id, action_type, action_map[action_type])
                    if action:
                        generated_name = self._action_name(int(action.get("type", action_type)))
                        self.generated_action_type_counts[generated_name] = self.generated_action_type_counts.get(generated_name, 0) + 1
                        total_actions.append(action)
                        break
            self.logger.info("step actions_count=%s", len(total_actions))
            return total_actions
        except Exception:
            self.logger.critical("step failed", exc_info=True)
            return []

    def save(self, path: str) -> None:
        try:
            payload = {
                "seed": self.seed,
                "seat": self.seat,
                "faction": self.faction,
                "role": self.role,
                "user_name": self.user_name,
                "user_id": self.user_id,
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            self.logger.info("saved to %s", path)
        except Exception:
            self.logger.critical("save failed", exc_info=True)
            raise

    def load(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.seed = payload.get("seed", self.seed)
            self._rng = random.Random(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            self.seat = payload.get("seat")
            self.faction = payload.get("faction")
            self.role = payload.get("role")
            self.user_name = payload.get("user_name")
            self.user_id = payload.get("user_id")
            self.logger.info("loaded from %s", path)
        except Exception:
            self.logger.critical("load failed", exc_info=True)
            raise

    def get_policy_stats(self) -> Dict[str, Any]:
        stats = self.policy.get_diagnostics()
        stats["use_ai_optimization"] = self.use_ai_optimization
        stats["valid_action_type_counts"] = dict(sorted(self.valid_action_type_counts.items()))
        stats["generated_action_type_counts"] = dict(sorted(self.generated_action_type_counts.items()))
        return stats

    def get_bop(self, obj_id: int) -> Optional[Dict[str, Any]]:
        for bop in self.observation.get("operators", []):
            if obj_id == bop.get("obj_id"):
                return bop
        return None

    def get_move_type(self, bop: Dict[str, Any]) -> int:
        bop_type = bop.get("type")
        if bop_type == BopType.Vehicle:
            if bop.get("move_state") == MoveType.March:
                move_type = MoveType.March
            else:
                move_type = MoveType.Maneuver
        elif bop_type == BopType.Infantry:
            move_type = MoveType.Walk
        else:
            move_type = MoveType.Fly
        return move_type

    def _generate_action(self, obj_id: int, action_type: int, candidate: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if self.use_ai_optimization:
            if action_type == ActionType.Shoot:
                bop = self.get_bop(obj_id)
                if not candidate or not bop:
                    return None
                best, trace = self.policy.choose_shoot_candidate(bop, candidate, self.observation, self.map)
                if best is not None:
                    self.policy.note_action("shoot", True, trace)
                    return {
                        "actor": self.seat,
                        "obj_id": obj_id,
                        "type": ActionType.Shoot,
                        "target_obj_id": best.get("target_obj_id"),
                        "weapon_id": best.get("weapon_id"),
                    }

            if action_type == ActionType.Move:
                bop = self.get_bop(obj_id)
                if not bop or bop.get("sub_type") == 3:
                    return None
                missing_move_candidate = not candidate
                cities = self.observation.get("cities", [])
                if not cities or not self.map:
                    return None
                cur_hex = bop.get("cur_hex")
                is_on_city = any(
                    city.get("coord") is not None and int(city["coord"]) == cur_hex
                    for city in cities
                )
                if is_on_city:
                    city_on = next(
                        (city for city in cities if city.get("coord") is not None and int(city["coord"]) == cur_hex),
                        None,
                    )
                    city_flag = city_on.get("flag", -1) if city_on else -1
                    if city_flag != self.faction:
                        return None
                move_type = self.get_move_type(bop)
                all_move_types = [move_type] + [m for m in [0, 1, 2, 3] if m != move_type]
                best_route_by_city: Dict[int, Tuple[Dict[str, Any], List[int], int]] = {}
                used_move_type = move_type
                for try_move_type in all_move_types:
                    for city in cities:
                        city_hex = city.get("coord")
                        if city_hex is None:
                            continue
                        city_id = int(city_hex)
                        route = self.map.gen_move_route(bop["cur_hex"], city_id, try_move_type)
                        if route:
                            route_len = len(route)
                            existing = best_route_by_city.get(city_id)
                            if existing is None or route_len < existing[2]:
                                best_route_by_city[city_id] = (city, route, route_len, try_move_type)
                reachable_city_routes = [
                    (rec[0], rec[1]) for rec in best_route_by_city.values()
                ]
                if reachable_city_routes:
                    used_move_type = best_route_by_city[next(iter(best_route_by_city))][3]

                if not reachable_city_routes:
                    nearest_city = min(
                        (city for city in cities if city.get("coord") is not None),
                        key=lambda c: self.map.get_distance(bop["cur_hex"], int(c["coord"])),
                        default=None,
                    )
                    fallback_hex = None
                    if nearest_city is not None:
                        target_city_hex = int(nearest_city["coord"])
                        neighbors = self.map.get_neighbors(bop["cur_hex"])
                        best_dist = self.map.get_distance(bop["cur_hex"], target_city_hex)
                        for n_hex in neighbors:
                            if n_hex < 0:
                                continue
                            dist = self.map.get_distance(n_hex, target_city_hex)
                            if dist >= 0 and dist < best_dist:
                                best_dist = dist
                                fallback_hex = n_hex
                    if fallback_hex is not None:
                        simple_route = [fallback_hex]
                        self.policy.record_move_route({
                            "operator_id": obj_id,
                            "operator_type": bop.get("type"),
                            "current_hex": bop.get("cur_hex"),
                            "move_type": move_type,
                            "move_type_name": {0: "Maneuver", 1: "March", 2: "Walk", 3: "Fly"}.get(move_type, str(move_type)),
                            "target_kind": "nearest_city_neighbor",
                            "target_source": "fallback_neighbor",
                            "target_city_id": int(nearest_city["coord"]),
                            "target_hex": fallback_hex,
                            "reachable_city_count": 0,
                            "total_cities": len(cities),
                            "route_len": 1,
                            "route_empty": False,
                            "route_failure_reason": "none",
                            "selected_by_learned": False,
                        })
                        self.policy.note_action(
                            "move",
                            False,
                            {"obj_id": obj_id, "target_hex": fallback_hex, "reason": "fallback_neighbor"},
                        )
                        return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.Move, "move_path": simple_route}
                    self.policy.record_move_route({
                        "operator_id": obj_id,
                        "operator_type": bop.get("type"),
                        "current_hex": bop.get("cur_hex"),
                        "move_type": move_type,
                        "move_type_name": {0: "Maneuver", 1: "March", 2: "Walk", 3: "Fly"}.get(move_type, str(move_type)),
                        "target_kind": "city",
                        "target_source": "no_valid_move_candidate" if missing_move_candidate else "empty_route",
                        "target_city_id": None,
                        "target_hex": None,
                        "reachable_city_count": 0,
                        "total_cities": len(cities),
                        "route_len": 0,
                        "route_empty": True,
                        "route_failure_reason": "no_valid_move_candidate" if missing_move_candidate else "empty_route",
                        "selected_by_learned": False,
                    })
                    self.policy.note_action(
                        "move",
                        False,
                        {
                            "obj_id": obj_id,
                            "reason": "no_valid_move_candidate" if missing_move_candidate else "empty_route",
                        },
                    )
                    return None
                selected_city, trace = self.policy.choose_city(
                    bop,
                    [city for city, _ in reachable_city_routes],
                    self.observation,
                    self.map,
                )
                if selected_city is not None:
                    best_city = int(selected_city["coord"])
                    route_by_city = {
                        int(city["coord"]): route for city, route in reachable_city_routes if city.get("coord") is not None
                    }
                    route = route_by_city.get(best_city, [])
                    if route:
                        target_source = "learned" if used_move_type == move_type else f"learned_fallback_move_type_{used_move_type}"
                        self.policy.record_move_route({
                            "operator_id": obj_id,
                            "operator_type": bop.get("type"),
                            "current_hex": bop.get("cur_hex"),
                            "move_type": used_move_type,
                            "move_type_name": {0: "Maneuver", 1: "March", 2: "Walk", 3: "Fly"}.get(used_move_type, str(used_move_type)),
                            "target_kind": "city",
                            "target_source": target_source,
                            "target_city_id": best_city,
                            "target_hex": best_city,
                            "reachable_city_count": len(reachable_city_routes),
                            "total_cities": len(cities),
                            "route_len": len(route),
                            "route_empty": False,
                            "route_failure_reason": "none",
                            "selected_by_learned": True,
                        })
                        self.policy.note_action("move", True, trace)
                        return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.Move, "move_path": route}
                    self.policy.record_move_route({
                        "operator_id": obj_id,
                        "operator_type": bop.get("type"),
                        "current_hex": bop.get("cur_hex"),
                        "move_type": move_type,
                        "move_type_name": {0: "Maneuver", 1: "March", 2: "Walk", 3: "Fly"}.get(move_type, str(move_type)),
                        "target_kind": "city",
                        "target_source": "learned",
                        "target_city_id": best_city,
                        "target_hex": best_city,
                        "reachable_city_count": len(reachable_city_routes),
                        "total_cities": len(cities),
                        "route_len": 0,
                        "route_empty": True,
                        "route_failure_reason": "empty_route",
                        "selected_by_learned": False,
                    })
                    self.policy.note_action(
                        "move",
                        False,
                        {"obj_id": obj_id, "candidate_id": best_city, "reason": "empty_route"},
                    )
                return None

        # Original Rule-based Logic (only when AI optimization disabled
        # or action type not handled by learned path)
        if self.use_ai_optimization and action_type in (ActionType.Shoot, ActionType.Move):
            return None

        if action_type == ActionType.Occupy:
            return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.Occupy}
        if action_type == ActionType.Shoot:
            if not candidate:
                return None
            best = max(candidate, key=lambda x: x.get("attack_level", 0))
            self.policy.note_action(
                "shoot",
                False,
                {"obj_id": obj_id, "candidate_id": best.get("target_obj_id"), "reason": "rule_fallback"},
            )
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.Shoot,
                "target_obj_id": best.get("target_obj_id"),
                "weapon_id": best.get("weapon_id"),
            }
        if action_type == ActionType.GuideShoot:
            if not candidate:
                return None
            best = max(candidate, key=lambda x: x.get("attack_level", 0))
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.GuideShoot,
                "target_obj_id": best.get("target_obj_id"),
                "weapon_id": best.get("weapon_id"),
                "guided_obj_id": best.get("guided_obj_id"),
            }
        if action_type == ActionType.JMPlan:
            if not candidate:
                return None
            weapon_id = self._rng.choice(candidate).get("weapon_id")
            jm_pos = self._rng.choice([city["coord"] for city in self.observation.get("cities", [])])
            return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.JMPlan, "jm_pos": jm_pos, "weapon_id": weapon_id}
        if action_type == ActionType.GetOn:
            if not candidate:
                return None
            if self._rng.random() < 0.5:
                target_obj_id = self._rng.choice(candidate).get("target_obj_id")
                return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.GetOn, "target_obj_id": target_obj_id}
            return None
        if action_type == ActionType.GetOff:
            if not candidate:
                return None
            bop = self.get_bop(obj_id)
            if not bop:
                return None
            destination = self._rng.choice([city["coord"] for city in self.observation.get("cities", [])])
            if self.map and self.map.get_distance(bop["cur_hex"], destination) <= 10:
                target_obj_id = self._rng.choice(candidate).get("target_obj_id")
                return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.GetOff, "target_obj_id": target_obj_id}
            return None
        if action_type == ActionType.ChangeState:
            if not candidate:
                return None
            if self._rng.random() < 0.001:
                target_state = self._rng.choice(candidate).get("target_state")
                return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.ChangeState, "target_state": target_state}
            return None
        if action_type == ActionType.RemoveKeep:
            if self._rng.random() < 0.2:
                return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.RemoveKeep}
            return None
        if action_type == ActionType.Move:
            bop = self.get_bop(obj_id)
            if not bop:
                return None
            if bop.get("sub_type") == 3:
                return None
            cities_list = self.observation.get("cities", [])
            cur_hex = bop.get("cur_hex")
            is_on_city = any(
                city.get("coord") is not None and int(city["coord"]) == cur_hex
                for city in cities_list
            )
            if is_on_city:
                city_on = next(
                    (city for city in cities_list if city.get("coord") is not None and int(city["coord"]) == cur_hex),
                    None,
                )
                city_flag = city_on.get("flag", -1) if city_on else -1
                if city_flag != self.faction:
                    return None
            destination = self._rng.choice([city["coord"] for city in cities_list])
            if self.my_direction:
                destination = self.my_direction.get("info", {}).get("target_pos", destination)
            if self.map and bop.get("cur_hex") != destination:
                move_type = self.get_move_type(bop)
                route = self.map.gen_move_route(bop["cur_hex"], destination, move_type)
                cities = self.observation.get("cities", [])
                self.policy.record_move_route({
                    "operator_id": obj_id,
                    "operator_type": bop.get("type"),
                    "current_hex": bop.get("cur_hex"),
                    "move_type": move_type,
                    "move_type_name": {0: "Maneuver", 1: "March", 2: "Walk", 3: "Fly"}.get(move_type, str(move_type)),
                    "target_kind": "city",
                    "target_source": "rule_fallback",
                    "target_city_id": destination,
                    "target_hex": destination,
                    "reachable_city_count": 0,
                    "total_cities": len(cities),
                    "route_len": len(route),
                    "route_empty": not route,
                    "route_failure_reason": "empty_route" if not route else "none",
                    "selected_by_learned": False,
                })
                self.policy.note_action(
                    "move",
                    False,
                    {"obj_id": obj_id, "candidate_id": destination, "reason": "rule_fallback"},
                )
                return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.Move, "move_path": route}
            return None
        if action_type == ActionType.StopMove:
            bop = self.get_bop(obj_id)
            if not bop or not self.map:
                return None
            destination = self._rng.choice([city["coord"] for city in self.observation.get("cities", [])])
            if self.map.get_distance(bop["cur_hex"], destination) <= 10:
                stop_move_prob = 0.9 if bop.get("passenger_ids") else 0.01
                if self._rng.random() < stop_move_prob:
                    return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.StopMove}
            return None
        if action_type == ActionType.WeaponLock:
            bop = self.get_bop(obj_id)
            if not bop or not self.map or self.map_data is None:
                return None
            try:
                cur_hex = int(bop.get("cur_hex", -1))
            except (TypeError, ValueError):
                return None
            if not self.map.is_valid(cur_hex):
                return None
            prob_weaponlock = 0.001
            roads = self.map_data[cur_hex // 100][cur_hex % 100].get("roads") or []
            if (roads and max(roads) > 0) or self._rng.random() < prob_weaponlock:
                return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.WeaponLock}
            return None
        if action_type == ActionType.WeaponUnFold:
            bop = self.get_bop(obj_id)
            if not bop or not self.map:
                return None
            destination = self._rng.choice([city["coord"] for city in self.observation.get("cities", [])])
            if self.map.get_distance(bop["cur_hex"], destination) <= 10:
                return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.WeaponUnFold}
            return None
        if action_type == ActionType.CancelJMPlan:
            if self._rng.random() < 0.0001:
                return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.CancelJMPlan}
            return None
        if action_type == ActionType.Fork:
            if self._rng.random() < 0.01:
                return None
            return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.Fork}
        if action_type == ActionType.Union:
            if not candidate:
                return None
            if self._rng.random() < 0.1:
                return None
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "target_obj_id": self._rng.choice(candidate).get("target_obj_id"),
                "type": ActionType.Union,
            }
        if action_type == ActionType.ChangeAltitude:
            if not candidate:
                return None
            if self._rng.random() < 0.05:
                return None
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "target_obj_id": self._rng.choice(candidate).get("target_altitude"),
                "type": ActionType.ChangeAltitude,
            }
        if action_type == ActionType.ActivateRadar:
            if self._rng.random() < 1:
                return None
            return {"actor": self.seat, "obj_id": obj_id, "type": ActionType.ActivateRadar}
        if action_type == ActionType.EnterFort:
            if not candidate:
                return None
            if self._rng.random() < 0.5:
                return None
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.EnterFort,
                "target_obj_id": self._rng.choice(candidate).get("target_obj_id"),
            }
        if action_type == ActionType.ExitFort:
            if not candidate:
                return None
            if self._rng.random() < 0.1:
                return None
            return {
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.ExitFort,
                "target_obj_id": self._rng.choice(candidate).get("target_obj_id"),
            }
        if action_type == ActionType.LayMine:
            if self._rng.random() < 1:
                return None
            return {"actor": self.seat, "obj_id": obj_id, "type": 20, "target_pos": self._rng.randint(0, 9177)}
        return None
