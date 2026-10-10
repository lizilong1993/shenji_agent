# -*- coding: utf-8 -*-
"""I15 步骤 2/3：把 v1 复盘**升级**为 v1.1.0 双核心格式（写侧）。

## 为什么这样做（而不是直接改 `run_offline_games.py`）

`I15`【落点】写的是 `land_wargame_sdk/run_offline_games.py`，但实测：
该脚本**仍写旧的 `state[GREEN]` 整帧快照**（第 83/125/229/331 行），
且它是引擎侧脚本（`from train_env import TrainEnv`）。

**导入真检查结果**：`training/selfplay.py` 可正常导入，但 `run_offline_games.py`
是引擎侧入口；而**本地推演可跑**（`collect_cli run` 已验证）。

故本模块提供一个**纯函数式**的写侧实现：读 v1 复盘 → 逐帧升级为 v1.1.0 → 写新文件 →
**用真实 schema 校验** → **回读核对**。这样：
  - 可在**不依赖引擎**的前提下验证写侧正确；
  - `run_offline_games.py` 将来只需把"构造帧"换成调用本模块，改动面最小。

## 三条主线（对应 `replay.schema.json` 的 `dual_core`）

| 主线 | v1 现状 | 升级后 |
| --- | --- | --- |
| **人类操作** | 只记原始命令 `{cur_step, message, error}`，**未标发出者** | `annotate_human_actions()` 逐条补 `actor_kind` |
| **智能体数据** | **完全没有** | `agent` 段（`valid_actions` 条数 + 时间线 + 局面摘要） |
| **随机种子** | 无 | `meta.seed`（由调用方给出，**不编造**） |
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.replay import (  # noqa: E402
    FRAME_VERSION_V1_1,
    ReplayFormatError,
    annotate_human_actions,
    build_frame,
    frame_entries,
    validate,
    write_replay,
)


def _agent_from_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    """从 v1 帧提取**智能体侧**摘要（v1 完全没有这一主线）。

    只用帧里**已有**的字段，不编造：
      - `valid_actions`：该步合法的动作集合（v1 有，但从未进入复盘主线）；
      - `time`：时间线（`cur_step`/`tick`/`max_step`/`stage`）；
      - `scores`：局面摘要（红蓝双方得分）。

    ⚠ **缺失的字段不填假值** —— 例如 v1 帧里没有"智能体内部决策摘要"
    （模型输出/置信度/候选排序），故 `decision` 段**只在真有数据时才写**。
    """
    agent: Dict[str, Any] = {}
    va = frame.get("valid_actions")
    if isinstance(va, dict):
        agent["valid_action_kinds"] = sorted(str(k) for k in va)
        agent["valid_action_count"] = int(sum(
            len(v) if isinstance(v, (list, tuple)) else 0 for v in va.values()))
    elif isinstance(va, list):
        agent["valid_action_count"] = len(va)
    t = frame.get("time")
    if isinstance(t, dict):
        agent["timeline"] = {
            k: t[k] for k in ("cur_step", "tick", "max_step", "stage") if k in t
        }
    sc = frame.get("scores")
    if isinstance(sc, dict):
        agent["scores"] = dict(sc)
    return agent


def upgrade_frame(v1_frame: Dict[str, Any], *,
                  seed: Optional[int] = None,
                  scenario: Optional[str] = None,
                  human_actor: int = 0) -> Dict[str, Any]:
    """把单个 v1 帧升级为 v1.1.0 帧（人类操作 + 智能体数据 + 种子）。"""
    agent = _agent_from_frame(v1_frame)
    frame = build_frame(v1_frame, seed=seed, agent=agent or None,
                        scenario=scenario, frame_version=FRAME_VERSION_V1_1)
    # 人类操作：补"发出者"标注（v1 只有原始命令，未标是否人类）
    if frame.get("actions"):
        frame = annotate_human_actions(frame, human_actor=human_actor)
    return frame


def upgrade_replay(src: Path, dst: Path, *,
                   seed: Optional[int] = None,
                   scenario: Optional[str] = None,
                   run_id: Optional[str] = None,
                   human_actor: int = 0,
                   size_limit_bytes: Optional[int] = None,
                   ) -> Dict[str, Any]:
    """读 v1 复盘 → 写 v1.1.0 复盘 → 校验 → 回读核对。

    Returns
    -------
    dict
        `{"frames": n, "report": <validate 结果>, "upgraded": n_agent, "annotated": n_human}`。
        **`validate` 的结果原样返回**，不粉饰。
    """
    src = Path(src)
    if not src.is_file():
        raise ReplayFormatError("source replay not found: %s" % src)

    v1_frames: List[Tuple[int, Dict[str, Any]]] = []
    with zipfile.ZipFile(src) as z:
        for idx, name in frame_entries(z):
            v1_frames.append((idx, json.loads(z.read(name).decode("utf-8"))))
    if not v1_frames:
        raise ReplayFormatError("source replay has no frames: %s" % src)

    rid = run_id or src.stem
    upgraded: List[Dict[str, Any]] = []
    n_agent = n_human = 0
    for idx, f in v1_frames:
        nf = upgrade_frame(f, seed=seed, scenario=scenario, human_actor=human_actor)
        if nf.get("agent"):
            n_agent += 1
        if nf.get("human"):
            n_human += 1
        upgraded.append(nf)

    report = write_replay(dst, upgraded, run_id=rid,
                          size_limit_bytes=size_limit_bytes,
                          validate_first=True)

    # 回读核对：帧数与逐帧条数必须一致（否则"写成功"是假的）
    with zipfile.ZipFile(dst) as z:
        back = [json.loads(z.read(n).decode("utf-8")) for _, n in frame_entries(z)]
    if len(back) != len(upgraded):
        raise ReplayFormatError(
            "round-trip frame count mismatch: wrote %d, read %d"
            % (len(upgraded), len(back)))

    return {
        "frames": len(upgraded),
        "readback_frames": len(back),
        "report": report,
        "upgraded_with_agent": n_agent,
        "upgraded_with_human": n_human,
        "version": (back[0].get("meta") or {}).get("schema_version"),
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="upgrade_replay_v11",
        description="Upgrade a v1 replay to the v1.1.0 dual-core format (I15 step 2/3).")
    ap.add_argument("source")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="随机种子；不给则 meta.seed 不写（不编造）")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--human-actor", type=int, default=0)
    ap.add_argument("--size-limit-bytes", type=int, default=None)
    args = ap.parse_args(argv)

    res = upgrade_replay(Path(args.source), Path(args.dest),
                         seed=args.seed, scenario=args.scenario,
                         run_id=args.run_id, human_actor=args.human_actor,
                         size_limit_bytes=args.size_limit_bytes)
    print("replay upgrade v1 -> v1.1.0")
    print("  frames: %d (readback %d)" % (res["frames"], res["readback_frames"]))
    print("  version in file: %s" % res["version"])
    print("  frames with agent section: %d" % res["upgraded_with_agent"])
    print("  frames with human section: %d" % res["upgraded_with_human"])
    rep = res["report"]
    print("  validate ok: %s  version: %s" % (rep.get("ok"), rep.get("declared_version")))
    if rep.get("missing_keys"):
        print("  missing_keys: %s" % rep["missing_keys"])
    if rep.get("bad_entries"):
        print("  bad_entries: %s" % rep["bad_entries"])
    print("  written: %s" % args.dest)
    return 0 if rep.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
