"""`run_offline_games.py` 的复盘写盘助手（I15 步骤 2 的引擎侧收尾）。

## 背景

`land_wargame_sdk/run_offline_games.py` 有**两处**写盘逻辑
（单智能体模式 L134-141、多智能体模式 L344-351），原样都写：
`state[GREEN]` 整帧快照 → `json.dumps` → `z.writestr(f"{begin}/{i}", data)`。

**即：产出的是 v1.0.0 单核心复盘**，`I15` 要的双核心（人类操作 + 智能体数据）**没有**。

## 本模块

把该逻辑抽成一个函数，**保留 v1 原行为**，并可选升级为 v1.1.0：

| `frame_version` | 行为 |
| --- | --- |
| `"1.0.0"` | **与改动前逐字节等价**（同样的 `json.dumps` + 条目命名） |
| `"1.1.0"` | 经 `build_frame` + `annotate_human_actions` 升级后写出，并**校验** |

**默认取 `1.1.0`**（`I15` 的目标），但环境变量 `SHENJI_REPLAY_VERSION=1.0.0`
可切回旧格式 —— 这是 `I15`【风险与回滚】要求的"新格式写入可用开关切换"。

## 为什么用环境变量而不是新参数

`run_offline_games.py` 是引擎侧脚本，改它的调用签名会影响别的调用方；
环境变量开关**不改签名**，回滚时只需设一个变量。

## 依赖缺失时的行为（**重要**）

该脚本在 `land_wargame_sdk/` 下运行，可能 **import 不到 repo 根的 `training` 包**
（例如从别的目录启动）。此时**退化为 v1 原行为**并打印一行说明 ——
**不抛异常、不产半成品**。宁可产出旧格式可读的复盘，也不要让离线对局整个失败。
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

FRAME_VERSION_V1 = "1.0.0"
FRAME_VERSION_V1_1 = "1.1.0"
DEFAULT_VERSION = FRAME_VERSION_V1_1

_ENV_KEY = "SHENJI_REPLAY_VERSION"


def _load_upgrade_helpers():
    """把 repo 根的 `training.replay` 引入。**失败时返回 None**（调用方退化）。"""
    try:
        from training.replay import annotate_human_actions, build_frame  # noqa: PLC0415
        return build_frame, annotate_human_actions
    except ImportError:
        pass
    # 尝试从本文件位置向上找 repo 根
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "training" / "replay.py").is_file():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            try:
                from training.replay import (  # noqa: PLC0415
                    annotate_human_actions, build_frame,
                )
                return build_frame, annotate_human_actions
            except ImportError:
                return None
    return None


def _agent_summary(observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从 v1 帧里提取智能体侧摘要（只用已有字段，不编造）。"""
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
    return agent or None


def resolve_frame_version(explicit: Optional[str] = None) -> str:
    """决定用哪个帧版本：显式参数 > 环境变量 > 默认（1.1.0）。"""
    if explicit:
        return str(explicit)
    env = os.environ.get(_ENV_KEY)
    if env in (FRAME_VERSION_V1, FRAME_VERSION_V1_1):
        return env
    return DEFAULT_VERSION


def _upgrade_frames(all_states: List[Dict[str, Any]], *,
                    seed: Optional[int], scenario: Optional[str],
                    human_actor: int) -> Optional[List[Dict[str, Any]]]:
    """把 v1 帧列表升级为 v1.1.0；助手不可用时返回 **None**（调用方退化）。"""
    helpers = _load_upgrade_helpers()
    if helpers is None:
        return None
    build_frame, annotate_human_actions = helpers
    out: List[Dict[str, Any]] = []
    for obs in all_states:
        if not isinstance(obs, dict):
            continue
        frame = build_frame(obs, seed=seed, agent=_agent_summary(obs),
                            scenario=scenario, frame_version=FRAME_VERSION_V1_1)
        if frame.get("actions"):
            frame = annotate_human_actions(frame, human_actor=human_actor)
        out.append(frame)
    return out or None


def write_offline_replay(all_states: List[Dict[str, Any]], *,
                         zip_name: str, run_id: Optional[str] = None,
                         seed: Optional[int] = None,
                         scenario: Optional[str] = None,
                         human_actor: int = 0,
                         frame_version: Optional[str] = None,
                         allow_missing_seed: bool = False,
                         ) -> Dict[str, Any]:
    """写出复盘 zip。**返回结构化结论**，便于调用方取证。

    `frame_version` 为 `1.0.0`，或升级助手不可用时：走**与改动前逐字节等价**的旧路径。
    """
    version = resolve_frame_version(frame_version)
    rid = run_id or Path(zip_name).stem
    parent = os.path.dirname(zip_name)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    # ⚠ **不静默降级**：独立评审实测，产线调用曾完全不传 seed，
    # 于是 1.1.0 校验必然失败、**每次都退回 1.0.0** ——
    # 双核心写入端**从未在产线路径上真正执行**，而日志只有一行"falling back"。
    # 故 1.1.0 且未给 seed 时**直接报错**，强制调用方显式提供种子。
    if version == FRAME_VERSION_V1_1 and seed is None and not allow_missing_seed:
        raise ValueError(
            "frame_version=1.1.0 requires an explicit seed (I15 dual-core records the "
            "random seed as one of its six record types). Callers that genuinely have no "
            "seed must pass frame_version='1.0.0' or allow_missing_seed=True, and must not "
            "claim the dual-core format.")

    frames = None
    if version == FRAME_VERSION_V1_1:
        frames = _upgrade_frames(all_states, seed=seed, scenario=scenario,
                                 human_actor=human_actor)
        if frames is None:
            print("[replay] upgrade helper unavailable -> falling back to v1.0.0 format")
            version = FRAME_VERSION_V1

    if version == FRAME_VERSION_V1_1 and frames is not None:
        # 走 schema 校验的写入路径（不合格就不发布）
        try:
            from training.replay import write_replay  # noqa: PLC0415
            report = write_replay(Path(zip_name), frames, run_id=rid, validate_first=True)
            return {
                "path": zip_name, "run_id": rid, "frame_version": FRAME_VERSION_V1_1,
                "frames": len(frames), "validate_ok": bool(report.get("ok")),
                "declared_version": report.get("declared_version"),
                "size_bytes": report.get("size_bytes"),
            }
        except Exception as exc:  # noqa: BLE001
            # 校验失败**不静默**：打印原因并退回旧格式，保证对局产物不会整个丢
            print("[replay] v1.1.0 write/validate failed (%s); falling back to v1.0.0" % exc)
            version = FRAME_VERSION_V1

    # ── v1 旧路径：与改动前**逐字节等价** ──────────────────────────────
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for i, ob in enumerate(all_states):
            z.writestr("%s/%d" % (rid, i),
                       json.dumps(ob, ensure_ascii=False, separators=(",", ":")))
    return {
        "path": zip_name, "run_id": rid, "frame_version": FRAME_VERSION_V1,
        "frames": len(all_states), "validate_ok": None, "declared_version": None,
        "size_bytes": os.path.getsize(zip_name) if os.path.exists(zip_name) else None,
    }
