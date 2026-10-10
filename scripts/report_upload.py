#!/usr/bin/env python3
"""对抗成绩与复盘文件回传（I18）。

契约来源
--------
31006 5.1.2 数据回传系列接口：
  - 对抗成绩回传：对抗双方、对抗 ID、成绩、时间；
  - 对抗复盘文件回传：复盘文件 <= 100MB（索引注「建议」）。

字段不是照抄标准，而是**对齐仓库里真实产物**（实读
`experiments/shenji_agent_smoke/league_20260427_140817/`）：
  - `match_results.json` 的 `matches[]` 有 21 个字段，含
    `scenario` / `bucket` / `map_id` / `paired_score` / `first_leg` / `second_leg` /
    `candidate_*` / `opponent_*` 双方身份与来源提交号；
  - `participants.json` 给出双方的 artifacts / profile / user_name / source_commit_sha；
  - `rating.json`、`promotion.json` 给出评级与晋升结论。
**但不含任何 replay 引用**（实测 `replay` 字段出现 0 次）——所以「复盘文件回传」
必须先建立「哪场对局对应哪个复盘文件」的绑定，本脚本用**确定性 match_id** 承担该绑定。

对抗 ID（match_id）为什么必须确定性
------------------------------------
31006 5.1.2 要求回传可重复而不产生重复记录（幂等）。若 ID 含随机数或当前时间，
同一场对局重跑脚本就会得到新 ID，幂等失效。故：

    match_id = sha256(f"{scenario}|{map_id}|{order_index}|{candidate_sha}|{opponent_sha}")[:16]

同一场对局在任何机器、任何时刻算出的 ID 都相同；换对手或换场景则必然不同。

回传目标
--------
平台侧接口未开放（I18 风险项已提示），故本脚本**以本地目录回传先行**，
服务回传作为可选后端；`--dry-run` 只报告不写盘。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MIB = 1024 * 1024
REPLAY_LIMIT_BYTES = 100 * MIB          # 31006 5.1.2，索引注「建议」
CONTRACT_VERSION = "1.0.0"


class UploadError(Exception):
    """回传失败（可重试或无望，见 retryable）。"""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


# ---------------------------------------------------------------- match_id

def compute_match_id(match: Dict[str, Any]) -> str:
    """确定性对抗 ID。见模块 docstring 的说明。

    注意：`candidate_source_commit_sha` / `opponent_source_commit_sha` 在真实产物里
    位于 `matches[]` 元素的**顶层**（不在任何子字典里）——实读 21 个字段确认。
    """
    parts = [
        str(match.get("scenario", "")),
        str(match.get("map_id", "")),
        str(match.get("order_index", "")),
        str(match.get("candidate_source_commit_sha", "")),
        str(match.get("opponent_source_commit_sha", "")),
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- 记录构造

def build_result_record(run_dir: Path, match: Dict[str, Any], *,
                        replay: Optional[Path] = None,
                        uploaded_utc: Optional[str] = None) -> Dict[str, Any]:
    """构造一条「对抗成绩」回传记录（31006 5.1.2 四要素 + 可追溯身份）。"""
    mid = compute_match_id(match)
    ts = uploaded_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "match_id": mid,
        "uploaded_utc": ts,
        # 对抗双方
        "sides": {
            "candidate": {
                "user_name": match.get("candidate_user_name"),
                "profile": match.get("candidate_profile"),
                "artifacts": match.get("candidate_artifacts"),
                "source_commit_sha": match.get("candidate_source_commit_sha"),
                "code_origin": match.get("candidate_code_origin"),
            },
            "opponent": {
                "user_name": match.get("opponent_user_name"),
                "profile": match.get("opponent_profile"),
                "artifacts": match.get("opponent_artifacts"),
                "source_commit_sha": match.get("opponent_source_commit_sha"),
                "code_origin": match.get("opponent_code_origin"),
            },
        },
        # 成绩
        "score": {
            "paired_score": match.get("paired_score"),
            "first_leg": match.get("first_leg"),
            "second_leg": match.get("second_leg"),
        },
        # 场景上下文（便于按场景/地图聚合）
        "scenario": match.get("scenario"),
        "bucket": match.get("bucket"),
        "map_id": match.get("map_id"),
        "order_index": match.get("order_index"),
        "run_dir": str(run_dir),
    }
    if replay is not None:
        rec["replay"] = {
            "path": replay.name,
            "size_bytes": replay.stat().st_size if replay.exists() else None,
            "sha256": sha256_file(replay) if replay.exists() else None,
        }
    return rec


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_run_summary(run_dir: Path) -> Dict[str, Any]:
    """汇总本目录的评级与晋升结论（缺失则为 None，不编造）。"""
    out: Dict[str, Any] = {"run_dir": str(run_dir)}
    for name, key in (("rating.json", "rating"), ("promotion.json", "promotion")):
        p = run_dir / name
        if p.exists():
            try:
                out[key] = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                out[key] = {"unreadable": str(e)}
        else:
            out[key] = None
    return out


# ---------------------------------------------------------------- 大小校验

def check_replay_size(path: Path, limit: int = REPLAY_LIMIT_BYTES) -> Tuple[bool, str]:
    """返回 (是否可回传, 说明)。31006 5.1.2 的 100MB 为**建议**值。

    处置口径：**超限即拒绝**（不静默放行），并给出分片建议。
    这一点与 I15 的 schema 校验有意不同——校验是「提示」，回传是「拒绝」：
    回传会把文件推给对端，超限直接损害对端，故必须挡在这里。
    """
    if not path.exists():
        return False, "replay not found: %s" % path
    size = path.stat().st_size
    if size > limit:
        return False, (
            "replay %.2f MB exceeds limit %.2f MB (31006 5.1.2, advisory); "
            "shard the replay (e.g. split by tick range into <=%.0f MB parts) before uploading"
            % (size / MIB, limit / MIB, limit / MIB))
    return True, "ok (%.2f MB)" % (size / MIB)


# ---------------------------------------------------------------- 幂等落盘

def upload_to_dir(records: List[Dict[str, Any]], target: Path, *,
                  dry_run: bool = False, retries: int = 3) -> Dict[str, Any]:
    """把成绩记录写到 target，按 match_id 幂等。

    幂等语义：同一 match_id 重复回传**不产生第二条记录**。
    - 已存在且内容相同 → 记 `skipped_identical`；
    - 已存在但内容不同 → 记 `conflict`，**不覆盖**（避免静默改写历史）。
    """
    target = Path(target)
    results = {"written": [], "skipped_identical": [], "conflicts": [], "failed": [], "dry_run": dry_run}

    for rec in records:
        mid = rec["match_id"]
        dst = target / ("match_%s.json" % mid)
        try:
            if dst.exists():
                old = json.loads(dst.read_text(encoding="utf-8"))
                # 比较时忽略 uploaded_utc：重复回传的时间不同，不应算冲突
                a = {k: v for k, v in old.items() if k != "uploaded_utc"}
                b = {k: v for k, v in rec.items() if k != "uploaded_utc"}
                if a == b:
                    results["skipped_identical"].append(mid)
                else:
                    results["conflicts"].append(mid)
                continue
            if dry_run:
                results["written"].append(mid)
                continue
            _atomic_write_json(dst, rec, retries=retries)
            results["written"].append(mid)
        except Exception as e:  # noqa: BLE001
            results["failed"].append({"match_id": mid, "error": str(e)})
    return results


def _atomic_write_json(dst: Path, payload: Dict[str, Any], *, retries: int = 3) -> None:
    """原子写：先写临时文件再 rename，避免半截文件。

    重试用于「目录瞬时不可写」这类可恢复故障；不掩盖持续性错误——
    重试耗尽后抛 `UploadError`，交由上层记 `failed`。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    last: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        try:
            fd, tmp = tempfile.mkstemp(dir=str(dst.parent), prefix=".upload-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
                    fh.write("\n")
                # `mkstemp` 建的是 0600；回传产物是给别人读的共享数据，
                # 留 0600 会让换机/容器场景下**别人读不到**（正是本任务要消除的那类阻塞）。
                os.chmod(tmp, 0o644)
                os.replace(tmp, dst)        # 原子
            except Exception:
                if os.path.exists(tmp):
                    os.unlink(tmp)
                raise
            return
        except Exception as e:  # noqa: BLE001
            last = e
    raise UploadError("failed to write %s after %d attempt(s): %s" % (dst, retries, last),
                      retryable=True)


# ---------------------------------------------------------------- 编排

def collect(run_dir: Path, replay: Optional[Path] = None,
            uploaded_utc: Optional[str] = None) -> List[Dict[str, Any]]:
    mr = run_dir / "match_results.json"
    if not mr.exists():
        raise UploadError("match_results.json not found in %s" % run_dir)
    payload = json.loads(mr.read_text(encoding="utf-8"))
    matches = payload.get("matches") or []
    return [build_result_record(run_dir, m, replay=replay, uploaded_utc=uploaded_utc) for m in matches]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Upload Shenji match scores and replay files (I18).")
    ap.add_argument("--run-dir", required=True, help="experiment dir containing match_results.json")
    ap.add_argument("--target", default="reports/uploaded", help="local target dir (service backend not open)")
    ap.add_argument("--replay", default=None, help="replay zip to bind to these matches")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--now", default=None, help="override uploaded_utc (for reproducible tests)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    replay = Path(args.replay) if args.replay else None

    if replay is not None:
        ok, why = check_replay_size(replay)
        if not ok:
            msg = {"ok": False, "stage": "size_check", "error": why}
            print(json.dumps(msg, ensure_ascii=False, indent=2) if args.json else "REJECTED: %s" % why)
            return 2
        print("replay size check: %s" % why)

    records = collect(run_dir, replay=replay, uploaded_utc=args.now)
    summary = build_run_summary(run_dir)
    if not args.dry_run:
        Path(args.target).mkdir(parents=True, exist_ok=True)
        _atomic_write_json(Path(args.target) / "run_summary.json", summary, retries=3)

    res = upload_to_dir(records, Path(args.target), dry_run=args.dry_run)
    out = {"ok": not res["failed"] and not res["conflicts"],
           "records": len(records), "target": args.target, **res}
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("records=%d  target=%s  dry_run=%s" % (len(records), args.target, args.dry_run))
        print("  written=%d skipped_identical=%d conflicts=%d failed=%d"
              % (len(res["written"]), len(res["skipped_identical"]), len(res["conflicts"]), len(res["failed"])))
        for mid in res["written"]:
            print("    + match_%s.json" % mid)
        for mid in res["skipped_identical"]:
            print("    = match_%s.json (identical, skipped)" % mid)
        for mid in res["conflicts"]:
            print("    ! match_%s.json CONFLICT (not overwritten)" % mid)
        for f in res["failed"]:
            print("    x %s: %s" % (f["match_id"], f["error"]))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
