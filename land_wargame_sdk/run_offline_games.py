"""Run some games in some scenarios locally.

This is the correct entry to start the program.
To test your agent, the major modifications you need to make are:
1. import your own agent class (you definitely know where it is)
2. instantiate your agent and its opponent (replace the Agent below)

The code looks like:

from ai.agent import Agent

red = Agent()
blue = DemoAgent()
"""
import json
import pickle
import time
import zipfile
import os
import numpy

from offline_replay import write_offline_replay
from ai import Agent
from train_env import TrainEnv

RED, BLUE, GREEN = 0, 1, -1

# I15：本次运行的随机种子（由模式函数入口设置，写盘时记录**同一个值**）
_RUN_SEED = None
 

def main():
    run_in_single_agent_mode()
    # run_in_multi_agents_mode()


def run_in_single_agent_mode():
    """
    run demo in single agent mode
    """
    # I15：在**对局开始前**真正固定随机流，使写入复盘的 seed 名副其实。
    # 边界：只固定本进程可见的随机性（python/numpy/torch）；
    # 引擎内部是否确定本环境无法验证（train_env 闭源、鉴权不通过）。
    try:
        from training.common import set_seed as _set_seed
        global _RUN_SEED
        import time as _time
        _RUN_SEED = int(_time.time())
        _set_seed(_RUN_SEED)
        print(f"[replay] run seed fixed: {_RUN_SEED}")
    except Exception as _exc:
        _RUN_SEED = None
        print(f"[replay] could not fix a run seed: {_exc}")

    print("running in single agent mode...")
    # instantiate agents and env
    red1 = Agent()
    blue1 = Agent()
    env1 = TrainEnv()
    begin = time.time()

    # get data ready, data can from files, web, or any other sources
    with open("data/Data/scenarios/1231.json", encoding='utf8') as f:
        scenario_data = json.load(f)
    with open("data/Data/maps/map_123/basic.json", encoding='utf8') as f:
        basic_data = json.load(f)
    with open('data/Data/maps/map_123/cost.pickle', 'rb') as file:
        cost_data = pickle.load(file)
    see_data = numpy.load("data/Data/maps/map_123/123see.npz")['data']

    # varialbe to build replay
    all_states = []

    # player setup info
    player_info = [{
        "seat": 1,
        "faction": 0,
        "role": 1,
        "user_name": "demo",
        "user_id": 0
    },
    {     
        "seat": 11,
        "faction": 1,
        "role": 1,
        "user_name": "demo",
        "user_id": 0
    }]

    # env setup info
    env_step_info = {
        "scenario_data": scenario_data,
        "basic_data": basic_data,
        "cost_data": cost_data,
        "see_data": see_data,
        "player_info": player_info
    }

    # setup env
    state = env1.setup(env_step_info)
    all_states.append(state[GREEN])
    print("Environment is ready.")

    # setup AIs
    red1.setup(
        {
            "scenario": scenario_data,
            "basic_data": basic_data,
            "cost_data": cost_data,
            "see_data": see_data,
            "seat": 1,
            "faction": 0,
            "role": 0,
            "user_name": "demo",
            "user_id": 0,
            "state": state,
        }
    )
    blue1.setup(
        {
            "scenario": scenario_data,
            "basic_data": basic_data,
            "cost_data": cost_data,
            "see_data": see_data,
            "seat": 11,
            "faction": 1,
            "role": 0,
            "user_name": "demo",
            "user_id": 0,
            "state": state,
        }
    )
    print("agents are ready.")

    # loop until the end of game
    print("steping")
    done = False
    while not done:
        actions = []
        actions += red1.step(state[RED])
        actions += blue1.step(state[BLUE])
        state, done = env1.step(actions)
        all_states.append(state[GREEN])

    env1.reset()
    red1.reset()
    blue1.reset()

    print(f"Total time: {time.time() - begin:.3f}s")

    # save replay（I15：默认写 v1.1.0 双核心；可用 SHENJI_REPLAY_VERSION=1.0.0 切回旧格式）
    #
    # 原文是直接 `state[GREEN]` 快照 → json.dumps → writestr，即 **v1.0.0 单核心**，
    # 双核心（人类操作 + 智能体数据）缺失。现交给 offline_replay.write_offline_replay：
    #   - 默认升级为 1.1.0 并**过 schema 校验**（不合格不发布）；
    #   - 升级助手 import 不到、或校验失败时，**退回与原文逐字节等价的 v1 写法**，
    #     保证离线对局的产物不会整个丢失。
    zip_name = f"logs/replays/replay_{begin}.zip"
    # ⚠ 种子必须**真的被用**，否则记录它只是假的可复现证据。
    #
    # 独立评审指出原实现**完全不传 seed** → 1.1.0 校验必然失败 →
    # **每次都静默退回 1.0.0** → 双核心写入端从未在产线路径上真正生效。
    #
    # 但我随后核实：本脚本**既不设置也不使用任何随机种子**
    # （`set_seed`/`random.seed` 零命中）。故若只把某个时间戳写进复盘，
    # 那个"seed"**对复现毫无作用** —— 比不记更糟，因为它会让第三方
    # 误以为可按此种子复跑。
    #
    # 故此处**先在运行开始时真正固定随机流**，再把**同一个值**写入复盘：
    # 二者是同一个数，记录才有意义。
    #
    # **边界（必须如实声明）**：这只能固定**本进程可见的**随机性
    # （python `random` / numpy / torch）。**引擎内部**是否确定
    # 本环境**无法验证**（`train_env` 闭源且鉴权不通过），
    # 故不得据此宣称"整局逐字节可复现"。
    # 用**入口处真正设置的那个**种子，而不是另取一个时间戳
    _seed_for_record = globals().get("_RUN_SEED")
    replay_report = write_offline_replay(
        all_states, zip_name=zip_name, run_id=str(begin), scenario=str(begin),
        seed=_seed_for_record,
    )
    print(
        "[replay] %s version=%s frames=%s validate_ok=%s"
        % (replay_report.get("path"), replay_report.get("frame_version"),
           replay_report.get("frames"), replay_report.get("validate_ok"))
    )


def to_json_string(o, indent=None):
    return json.dumps(o, ensure_ascii=False, indent=indent, separators=(',', ':'))


def run_in_multi_agents_mode():
    """
    run demo in multi agent mode
    """
    # I15：在**对局开始前**真正固定随机流，使写入复盘的 seed 名副其实。
    # 边界：只固定本进程可见的随机性（python/numpy/torch）；
    # 引擎内部是否确定本环境无法验证（train_env 闭源、鉴权不通过）。
    try:
        from training.common import set_seed as _set_seed
        global _RUN_SEED
        import time as _time
        _RUN_SEED = int(_time.time())
        _set_seed(_RUN_SEED)
        print(f"[replay] run seed fixed: {_RUN_SEED}")
    except Exception as _exc:
        _RUN_SEED = None
        print(f"[replay] could not fix a run seed: {_exc}")

    print("running in multi agent mode...")
    # instantiate agents and env
    red1 = Agent()
    red2 = Agent()
    red3 = Agent()
    blue1 = Agent()
    blue2 = Agent()
    blue3 = Agent()
    env1 = TrainEnv()
    begin = time.time()

    # get data ready, data can from files, web, or any other sources
    with open("data/Data/scenarios/1231.json", encoding='utf8') as f:
        scenario_data = json.load(f)
    with open("data/Data/maps/map_123/basic.json", encoding='utf8') as f:
        basic_data = json.load(f)
    with open('data/Data/maps/map_123/cost.pickle', 'rb') as file:
        cost_data = pickle.load(file)
    see_data = numpy.load("data/Data/maps/map_123/123see.npz")['data']

    # varialbe to build replay
    all_states = []

    # player setup info
    player_info = [{
        "seat": 1,
        "faction": 0,
        "role": 1,
        "user_name": "red1",
        "user_id": 1
    },
    {
        "seat": 2,
        "faction": 0,
        "role": 0,
        "user_name": "red2",
        "user_id": 2
    },
    {
        "seat": 3,
        "faction": 0,
        "role": 0,
        "user_name": "red3",
        "user_id": 3
    },
    {
        "seat": 11,
        "faction": 1,
        "role": 1,
        "user_name": "blue1",
        "user_id": 11
    },
    {
        "seat": 12,
        "faction": 1,
        "role": 0,
        "user_name": "blue2",
        "user_id": 12
    },
    {
        "seat": 13,
        "faction": 1,
        "role": 0,
        "user_name": "blue3",
        "user_id": 13
    }]

    # env setup info
    env_step_info = {
        "scenario_data": scenario_data,
        "basic_data": basic_data,
        "cost_data": cost_data,
        "see_data": see_data,
        "player_info": player_info
    }

    # setup env
    state = env1.setup(env_step_info)
    all_states.append(state[GREEN])
    print("Environment is ready.")

    # setup AIs
    red1.setup(
        {
            "scenario": scenario_data,
            "basic_data": basic_data,
            "cost_data": cost_data,
            "see_data": see_data,
            "seat": 1,
            "faction": 0,
            "role": 1,
            "user_name": "red1",
            "user_id": 1,
            "state": state,
        }
    )
    red2.setup(
        {
            "scenario": scenario_data,
            "basic_data": basic_data,
            "cost_data": cost_data,
            "see_data": see_data,
            "seat": 2,
            "faction": 0,
            "role": 0,
            "user_name": "red2",
            "user_id": 2,
            "state": state,
        }
    )
    red3.setup(
        {
            "scenario": scenario_data,
            "basic_data": basic_data,
            "cost_data": cost_data,
            "see_data": see_data,
            "seat": 3,
            "faction": 0,
            "role": 0,
            "user_name": "red3",
            "user_id": 3,
            "state": state,
        }
    )
    blue1.setup(
        {
            "scenario": scenario_data,
            "basic_data": basic_data,
            "cost_data": cost_data,
            "see_data": see_data,
            "seat": 11,
            "faction": 1,
            "role": 1,
            "user_name": "blue1",
            "user_id": 11,
            "state": state,
        }
    )
    blue2.setup(
        {
            "scenario": scenario_data,
            "basic_data": basic_data,
            "cost_data": cost_data,
            "see_data": see_data,
            "seat": 12,
            "faction": 1,
            "role": 0,
            "user_name": "blue2",
            "user_id": 12,
            "state": state,
        }
    )
    blue3.setup(
        {
            "scenario": scenario_data,
            "basic_data": basic_data,
            "cost_data": cost_data,
            "see_data": see_data,
            "seat": 13,
            "faction": 1,
            "role": 0,
            "user_name": "blue3",
            "user_id": 13,
            "state": state,
        }
    )
    print("agents are ready.")

    # loop until the end of game
    print("steping")
    done = False
    while not done:
        actions = []
        actions += red1.step(state[RED])
        actions += red2.step(state[RED])
        actions += red3.step(state[RED])
        actions += blue1.step(state[BLUE])
        actions += blue2.step(state[BLUE])
        actions += blue3.step(state[BLUE])
        state, done = env1.step(actions)
        all_states.append(state[GREEN])

    env1.reset()
    red1.reset()
    red2.reset()
    red3.reset()
    blue1.reset()
    blue2.reset()
    blue3.reset()

    print(f"Total time: {time.time() - begin:.3f}s")

    # save replay（I15：默认写 v1.1.0 双核心；可用 SHENJI_REPLAY_VERSION=1.0.0 切回旧格式）
    #
    # 原文是直接 `state[GREEN]` 快照 → json.dumps → writestr，即 **v1.0.0 单核心**，
    # 双核心（人类操作 + 智能体数据）缺失。现交给 offline_replay.write_offline_replay：
    #   - 默认升级为 1.1.0 并**过 schema 校验**（不合格不发布）；
    #   - 升级助手 import 不到、或校验失败时，**退回与原文逐字节等价的 v1 写法**，
    #     保证离线对局的产物不会整个丢失。
    zip_name = f"logs/replays/replay_{begin}.zip"
    # ⚠ 种子必须**真的被用**，否则记录它只是假的可复现证据。
    #
    # 独立评审指出原实现**完全不传 seed** → 1.1.0 校验必然失败 →
    # **每次都静默退回 1.0.0** → 双核心写入端从未在产线路径上真正生效。
    #
    # 但我随后核实：本脚本**既不设置也不使用任何随机种子**
    # （`set_seed`/`random.seed` 零命中）。故若只把某个时间戳写进复盘，
    # 那个"seed"**对复现毫无作用** —— 比不记更糟，因为它会让第三方
    # 误以为可按此种子复跑。
    #
    # 故此处**先在运行开始时真正固定随机流**，再把**同一个值**写入复盘：
    # 二者是同一个数，记录才有意义。
    #
    # **边界（必须如实声明）**：这只能固定**本进程可见的**随机性
    # （python `random` / numpy / torch）。**引擎内部**是否确定
    # 本环境**无法验证**（`train_env` 闭源且鉴权不通过），
    # 故不得据此宣称"整局逐字节可复现"。
    # 用**入口处真正设置的那个**种子，而不是另取一个时间戳
    _seed_for_record = globals().get("_RUN_SEED")
    replay_report = write_offline_replay(
        all_states, zip_name=zip_name, run_id=str(begin), scenario=str(begin),
        seed=_seed_for_record,
    )
    print(
        "[replay] %s version=%s frames=%s validate_ok=%s"
        % (replay_report.get("path"), replay_report.get("frame_version"),
           replay_report.get("frames"), replay_report.get("validate_ok"))
    )

if __name__ == "__main__":
    main()
