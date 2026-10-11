"""I15：自博弈写入 v1.1.0 双核心复盘的记录器。

## 为什么单独成模块

`training/selfplay.py` 的 `_run_single()` 是对局主循环，**Bash 级改动面越小越好**。
故本模块只提供一个**记录器类**：主循环每步 `record(...)` 一次，结束时 `write(...)` 一次。
`selfplay.py` 只需三处小改（创建、每步记录、结束写盘）。

## 数据来源（**全部取自主循环已有变量，不新造**）

| 六类记录 | 本模块从哪里取 |
| --- | --- |
| 动作 | `actions`（主循环每步的 `red.step(...)+blue.step(...)`） |
| 裁决 | `after_state` 里的 `judge_info`（v1 帧字段；实测确有数据） |
| 时间线 | `after_state` 的 `time` 段 |
| 智能体内部决策摘要 | `agent_stats`（`red.get_policy_stats()`）+ 该步动作 |
| 随机种子 | 调用方传入（**不传则不写**，不编造） |
| 人类操作 | 自博弈**没有人类操作** → `actor_kind` 全为 `agent`，**如实标注** |

> **人类操作在自博弈里本就不存在。** 本模块不伪造人类命令，
> 而是把该步全部命令标为 `actor_kind: "agent"` —— 这正是 `dual_core.human_ops`
> 要求区分的那个字段。**如实标"无人类"，而不是假装有。**
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.replay import (  # noqa: E402
    FRAME_VERSION_V1_1,
    build_frame,
    write_replay,
)

RED, BLUE = 0, 1
GREEN = -1                      # `env.step()` 返回的 state 末位为绿方/全局面


def _observation_from_state(state: Any) -> Optional[Dict[str, Any]]:
    """从 `env.step()` 的返回里取出"整帧"观测。

    `state` 形如 `[red_obs, blue_obs, ..., green_obs]`（`state[-1]` 为绿方/全局）。
    取不到时返回 `None` —— **不伪造空帧**。
    """
    if isinstance(state, dict):
        for key in (GREEN, -1, "green", "GREEN"):
            v = state.get(key)
            if isinstance(v, dict):
                return v
        return state if state else None
    if isinstance(state, (list, tuple)) and state:
        last = state[-1]
        return last if isinstance(last, dict) else None
    return None


def _annotate_actions(actions: Any) -> List[Dict[str, Any]]:
    """把该步命令标为 `actor_kind`（自博弈**全部为 agent**，如实标注，不伪造人类）。

    同时保留原始 `message`/`error`，便于与 v1 对照。
    """
    out: List[Dict[str, Any]] = []
    for a in actions or []:
        if not isinstance(a, dict):
            continue
        actor = a.get("actor")
        if actor is None:
            msg = a.get("message")
            actor = msg.get("actor") if isinstance(msg, dict) else None
        kind = "agent"            # 自博弈里没有人类操作 —— 如实标 agent
        out.append({
            "cur_step": a.get("cur_step"),
            "actor": actor,
            "actor_kind": kind,
            "message": a.get("message"),
            "error": a.get("error"),
        })
    return out


def build_selfplay_frame(
    *, index: int, observation: Dict[str, Any], actions: Any,
    agent_summary: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None, scenario: Optional[str] = None,
) -> Dict[str, Any]:
    """把主循环的一步组装成 v1.1.0 帧（动作 + 裁决 + 时间线 + 决策摘要 + 种子 + 人类标注）。"""
    base: Dict[str, Any] = dict(observation)
    base["actions"] = _annotate_actions(actions)

    agent: Dict[str, Any] = {}
    t = observation.get("time")
    if isinstance(t, dict):
        agent["timeline"] = {k: t[k] for k in ("cur_step", "tick", "max_step", "stage") if k in t}
    sc = observation.get("scores")
    if isinstance(sc, dict):
        agent["scores"] = dict(sc)
    va = observation.get("valid_actions")
    if isinstance(va, dict):
        agent["valid_action_kinds"] = sorted(str(k) for k in va)
    if isinstance(agent_summary, dict) and agent_summary:
        agent["decisions"] = [agent_summary]      # 该步的决策摘要（调用方给，不编造）

    frame = build_frame(base, seed=seed, agent=agent or None,
                        scenario=scenario, frame_version=FRAME_VERSION_V1_1)

    # 人类操作主线：自博弈里**没有**人类命令，故 action_count 恒为 0 ——
    # **如实记录"无人类操作"**，而不是省略该段（省略会让"有没有人类"无法区分）。
    frame["human"] = {
        "actor_kind": "human",
        "source": "selfplay_has_no_human_operator",
        "action_count": 0,
        "actions": [],
        "agent_action_count": len(frame["actions"]),
        # schema 规范字段：自博弈**没有人类操作** → **空数组**，如实为空
        "operations": [],
    }
    return frame


class MatchReplayRecorder:
    """按步累积帧，结束时写出 v1.1.0 复盘。**默认关闭**，由调用方决定是否记录。"""

    def __init__(self, *, match_id: str, seed: Optional[int] = None,
                 scenario: Optional[str] = None) -> None:
        self.match_id = match_id
        self.seed = seed
        self.scenario = scenario
        self.frames: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.frames)

    def record(self, *, observation: Any, actions: Any,
               agent_summary: Optional[Dict[str, Any]] = None) -> bool:
        """记录一步；观测取不到时**跳过并返回 False**（不伪造空帧）。"""
        obs = _observation_from_state(observation)
        if obs is None:
            return False
        self.frames.append(build_selfplay_frame(
            index=len(self.frames), observation=obs, actions=actions,
            agent_summary=agent_summary, seed=self.seed, scenario=self.scenario))
        return True

    def write(self, path: Path, *, size_limit_bytes: Optional[int] = None) -> Dict[str, Any]:
        """写盘并返回 `validate` 报告。**帧为空时直接报错**，不写空文件。"""
        if not self.frames:
            raise ValueError("no frames recorded; refusing to write an empty replay")
        return write_replay(Path(path), self.frames, run_id=self.match_id,
                            size_limit_bytes=size_limit_bytes, validate_first=True)

    def summary(self) -> Dict[str, Any]:
        """六类记录的实际覆盖（**只报事实**）。"""
        n = len(self.frames)
        return {
            "match_id": self.match_id,
            "frame_count": n,
            "actions": sum(1 for f in self.frames if f.get("actions")),
            "adjudication": sum(1 for f in self.frames if f.get("judge_info")),
            "timeline": sum(1 for f in self.frames if isinstance(f.get("time"), dict)),
            "agent_decisions": sum(1 for f in self.frames
                                   if (f.get("agent") or {}).get("decisions")),
            "seed": sum(1 for f in self.frames
                        if isinstance(f.get("meta"), dict) and f["meta"].get("seed") is not None),
            "human_action_count": sum((f.get("human") or {}).get("action_count", 0)
                                      for f in self.frames),
            "agent_action_count": sum((f.get("human") or {}).get("agent_action_count", 0)
                                      for f in self.frames),
        }
