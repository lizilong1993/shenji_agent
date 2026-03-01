# 庙算平台陆战兵棋AI开发项目

## 项目简介

本项目是基于**庙算平台**（中科院自动化研究所开发的陆战兵棋推演平台）的AI开发SDK。庙算平台提供了一个完整的陆战兵棋模拟环境，支持红蓝双方对抗推演，开发者可以在此基础上开发智能兵棋AI。

项目官网：[http://wargame.ia.ac.cn](http://wargame.ia.ac.cn)

---

## 项目结构

```
miaosuan20260119/
├── docs/                          # 文档目录
│   ├── docs.md                   # 平台详细API文档
│   ├── 裁决表格.xlsx              # 裁决规则表
│   └── image*.png                # 文档配图
├── ai/                            # AI开发模块
│   └── agent.py                  # AI实现
├── land_wargame_sdk/             # SDK主目录
│   ├── Data/                     # 地图和想定数据
│   │   ├── maps/                # 地图数据
│   │   └── scenarios/           # 想定数据
│   ├── run_offline_games.py     # 推演启动入口程序
│   └── land_wargame_train_env-*.whl  # 环境安装包
├── logs/                         # 运行日志与复盘输出目录（自动生成）
│   └── replays/                  # 复盘文件输出
├── Dockerfile                    # Docker镜像构建文件
├── docker-compose.yml            # Docker容器编排配置
├── .dockerignore                 # Docker构建排除文件
├── docker-start.bat              # 启动开发容器脚本
├── docker-run.bat                # 运行推演脚本
├── docker-shell.bat              # 进入容器脚本
├── .git/                         # Git仓库
└── README.md                     # 本文件
```

---

## 项目逻辑与架构

### 1. 分层与模块职责

- **应用入口层**：`land_wargame_sdk/run_offline_games.py` 负责推演流程组织、数据加载、AI实例化与复盘输出。
- **AI算法层**：`ai/agent.py` 是唯一的 AI 实现入口，容器内挂载到 `/workspace/ai`。
- **环境仿真层**：`train_env` 来自 SDK 安装包 `land_wargame_train_env-*.whl`，封装了推演引擎与裁决规则。
- **地图与工具层**：`ai/map.py` 提供地图读取、距离计算、通视与寻路等基础能力。
- **数据资源层**：`land_wargame_sdk/Data/` 中存放地图与想定数据（场景配置）。

### 2. 入口程序与主流程

`run_offline_games.py` 是本仓库的主入口，单 Agent 与多 Agent 模式都在这里编排：

```
加载想定与地图数据
      │
      ▼
TrainEnv.setup(env_step_info)
      │
      ▼
Agent.setup(setup_info)
      │
      ▼
循环直到结束:
  Agent.step(observation)  →  生成 actions 列表
  TrainEnv.step(actions)   →  生成下一帧 state
      │
      ▼
TrainEnv.reset / Agent.reset
      │
      ▼
输出 replay_{timestamp}.zip
```

单 Agent 模式下红蓝各一个 Agent；多 Agent 模式下每方可创建多个 Agent 并共享同一环境。

### 3. AI开发框架

开发者需要继承 `BaseAgent` 类并实现三个抽象方法：

| 方法 | 功能 | 调用时机 |
|------|------|----------|
| `setup(setup_info)` | 初始化AI，接收推演配置 | 每场推演开始时 |
| `step(observation)` | 接收态势，返回动作列表 | 每个步长调用一次 |
| `reset()` | 重置AI状态，释放资源 | 每场推演结束时 |

在 `run_offline_games.py` 中，会向 `setup()` 注入包含想定、地图、席位等信息的 `setup_info`，而 `step()` 则以观测态势 `observation` 为输入并返回动作列表。

### 4. 环境与数据结构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   环境初始化  │ --> │  AI初始化   │ --> │  推演循环   │
│  TrainEnv   │     │   setup()   │     │   step()    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌─────────────┐     ┌──────▼──────┐
                    │   生成复盘   │ <-- │  双方AI决策  │
                    │  replay.zip  │     │ 返回actions │
                    └─────────────┘     └─────────────┘
```

`TrainEnv` 维护的是完整状态 `state`，而 AI 看到的是 `observation`（通常是 `state` 的子集）。在 `run_offline_games.py` 中，示例代码直接将红蓝双方态势 `state[RED]`、`state[BLUE]` 传给对应 Agent。

### 5. 态势数据结构

`observation` 包含以下关键信息：
- `operators`: 算子信息（位置、血量、弹药等）
- `valid_actions`: 当前可做动作
- `cities`: 夺控点信息
- `time`: 时间信息（当前步长、阶段等）
- `judge_info`: 裁决信息（射击结果等）
- `communication`: 通信信息（任务指令等）

### 6. 动作类型

支持的动作包括：
- 机动：`Move`, `StopMove`
- 射击：`Shoot`, `GuideShoot`, `JMPlan`
- 状态：`ChangeState`, `RemoveKeep`, `WeaponLock/Unlock`
- 载具：`GetOn`, `GetOff`
- 工事：`EnterFort`, `ExitFort`
- 其他：`Occupy`, `LayMine`, `Fork`, `Union` 等

### 7. 对抗模式

**单Agent模式**：一方只有一个AI实例，控制所有算子
**多Agent模式**：一方有多个AI实例（队长+队员），各自控制部分算子

### 8. 数据与文件关系

地图与想定数据由 `run_offline_games.py` 加载并传入 `TrainEnv.setup`，常用文件类型如下：

- `*_basic.json`：地图基础网格与静态要素
- `*_cost.pickle`：通行成本
- `*_see.npz`：通视矩阵
- `scenarios/*.json`：想定配置（双方编成、初始部署等）

本仓库数据位于 `land_wargame_sdk/Data/Data/` 目录；推演代码通过相对路径 `data/` 读取，Docker 环境会将 `land_wargame_sdk/Data` 挂载到容器内 `/workspace/data`，从而匹配读取路径。

### 9. 复盘输出

推演结束后会在 `logs/replays/` 生成 `replay_{timestamp}.zip`，其中每个条目是一帧态势的 JSON，用于回放与分析。

---

## 快速开始

### 方式一：Docker 环境（推荐）

使用 Docker Compose 命令：

```bash
# 构建并启动容器
docker-compose up -d miaosuan-ai

# 进入容器
docker exec -it miaosuan-ai-dev bash

# 运行推演
python run_offline_games.py

# 快速运行一次推演
docker-compose --profile run run --rm miaosuan-run
```

## 开发自己的AI

1. 修改 `ai/agent.py` 中的 `Agent` 类
2. 实现 `setup()`、`step()`、`reset()` 三个方法
3. 通过 Docker 运行推演

### 示例：简单的step实现

```python
def step(self, observation: dict):
    actions = []
    # 获取可控算子
    controllable_ops = observation["role_and_grouping_info"][self.seat]["operators"]
    
    # 遍历每个算子的合法动作
    for obj_id, valid_actions in observation["valid_actions"].items():
        if obj_id not in controllable_ops:
            continue
        # 优先执行射击
        if "Shoot" in valid_actions:
            target = valid_actions["Shoot"][0]
            actions.append({
                "actor": self.seat,
                "obj_id": obj_id,
                "type": ActionType.Shoot,
                "target_obj_id": target["target_obj_id"],
                "weapon_id": target["weapon_id"],
            })
    return actions
```

---

## 地图工具类 Map

`ai/agent.py` 内置了地图工具 `Map`，提供以下实用功能：

- `gen_move_route(begin, end, mode)`: A*寻路
- `get_distance(pos1, pos2)`: 计算六边形距离
- `can_see(pos1, pos2, mode)`: 判断通视
- `get_neighbors(pos)`: 获取相邻格子

---

## Docker 环境说明

项目已配置完整的 Docker 开发环境，基于 **Ubuntu 20.04 + Python 3.10**。

### 容器配置

| 服务 | 说明 | 命令 |
|------|------|------|
| `miaosuan-ai` | 开发容器（保持运行） | `docker-compose up -d miaosuan-ai` |
| `miaosuan-run` | 运行容器（执行一次） | `docker-compose --profile run run --rm miaosuan-run` |

### 数据挂载

| 宿主机路径 | 容器路径 | 用途 |
|------------|----------|------|
| `./ai` | `/workspace/ai` | AI代码（只读挂载，实时同步） |
| `./land_wargame_sdk/Data` | `/workspace/data` | 地图/想定数据 |
| `./logs` | `/workspace/logs` | 日志输出 |
| `./replays` | `/workspace/logs/replays` | 复盘文件 |

### 资源限制

- CPU: 最多 4 核 / 预留 2 核
- 内存: 最多 8GB / 预留 4GB

---

## Git 工作流说明

本项目采用**分支开发模式**，每次功能更新后都会更新 README.md 保持文档最新：

1. **每次开发新功能前**，我会新建一个分支：
   ```bash
   git checkout -b feature/xxx
   ```

2. **在分支上进行开发修改**

3. **更新 README.md 文档**，记录新功能

4. **完成后提交到分支**：
   ```bash
   git add .
   git commit -m "描述信息"
   ```

5. **由你决定是否合并到主分支**：
   ```bash
   git checkout master
   git merge feature/xxx
   ```

---

## 打包校验

```bash
unzip -l ai.zip | grep -E 'ai/$|agent.py'
```

---

## 参考资料

- `docs/docs.md`: 详细的API文档
- `ai/agent.py`: DemoAI实现参考
- 官网文档: https://wargame.ia.ac.cn/docs/

---

## 开发者

- Git用户名: lizilong1993
- 邮箱: lizilong.1993@qq.com
