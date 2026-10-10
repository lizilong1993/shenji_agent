"""复盘文件读取、schema 校验与按索引回放（I15 步骤 4/5/6 的读取侧）。

设计前提（I15 前置条件）
------------------------
「先定义 schema 再改写入端，避免历史复盘文件不可读」。
所以本模块**只做读取与校验，不改任何写入端**；并且**必须能读历史 v1 文件**。

v1 实测格式（依据 `replays/replay_1770632023.2594852.zip`，1802 帧 / 3.64 MB）
--------------------------------------------------------------------------
- 容器：zip；条目名 `<run_id>/<frame_index>`，索引从 0 连续递增；
- 每帧：`json.dumps(state[GREEN], ensure_ascii=False, separators=(",",":"))`，
  共 14 个顶层键；**单一 GREEN 视角整帧快照**；
- `actions[].message` 是原始下达命令，元素实测只有 `{cur_step, message, error}`；
- **没有**随机种子、**没有**智能体内部决策摘要、`judge_info` 实测为空数组
  → 这三项正是 31006 4.4.2「双核心」要求的缺口。

v1.1 扩展（本 schema 定义，写入端尚未实现）
------------------------------------------
在帧上新增三个**可选**对象：`meta`（含 `seed`）、`agent`（决策摘要）、`human`（人类操作）。
对旧文件它们缺失，故校验按 `schema_version` 决定是否需要。

为什么校验**不把未知字段当错误**
--------------------------------
`operators` 元素实测有 87 个字段，属 SDK 侧模型；若要求逐字段精确匹配，
SDK 一升级校验就会全线失败。故只**要求**定位与状态必需的少数几个字段，
其余声明为「允许出现、不强制」。
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# schema 与代码同仓，默认取仓库根下的 replays/schema.json
SCHEMA_RELATIVE = Path("replays") / "schema.json"

ENTRY_RE = re.compile(r"^[^/]+/[0-9]+$")

MIB = 1024 * 1024


class ReplayFormatError(Exception):
    """复盘文件不满足 schema 约定。"""


def repo_root() -> Path:
    """本文件位于 <repo>/training/，故上两级即仓库根。"""
    return Path(__file__).resolve().parent.parent


def load_schema(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else repo_root() / SCHEMA_RELATIVE
    if not p.exists():
        raise ReplayFormatError("schema not found: %s" % p)
    return json.loads(p.read_text(encoding="utf-8"))


def _frame_required(schema: Dict[str, Any], version: str) -> List[str]:
    """按 schema_version 决定必需字段：v1.1 才要求 meta/agent/human。"""
    req = list(schema.get("frame_required_v1", []))
    if version and _ver_tuple(version) >= (1, 1, 0):
        req += list(schema.get("frame_optional_v1_1", []))
    return req


def _ver_tuple(v: str) -> Tuple[int, int, int]:
    parts = re.findall(r"\d+", str(v))[:3]
    while len(parts) < 3:
        parts.append("0")
    return tuple(int(x) for x in parts)  # type: ignore[return-value]


def frame_entries(zf: zipfile.ZipFile) -> List[Tuple[int, str]]:
    """返回 [(frame_index, entry_name)]，按索引升序；忽略非帧条目。"""
    out: List[Tuple[int, str]] = []
    for name in zf.namelist():
        if not ENTRY_RE.match(name):
            continue
        out.append((int(name.rsplit("/", 1)[1]), name))
    out.sort(key=lambda x: x[0])
    return out


def validate(size_limit_bytes: Optional[int] = None,
             path: Optional[Path] = None,
             schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """校验单个复盘 zip。返回报告 dict（不抛异常，便于批量扫描）。

    报告字段：
      ok            是否通过（size 超限只记 warning，符合 31006 5.1.2「建议」口径）
      frame_count   帧数
      indices       索引是否从 0 连续
      missing_keys  各帧缺失的必需键（取并集）
      schema_version
      warnings      例如超限与分片提示
    """
    schema = schema or load_schema()
    p = Path(path) if path else Path("replays/replay_1770632023.2594852.zip")
    limit = int(size_limit_bytes if size_limit_bytes is not None
                else schema.get("size_limit", {}).get("limit_bytes", 100 * MIB))

    rep: Dict[str, Any] = {
        "path": str(p), "ok": False, "frame_count": 0,
        "contiguous": False, "missing_keys": [], "bad_entries": [],
        "warnings": [], "schema_version": schema.get("schema_version"),
        "size_bytes": None, "limit_bytes": limit,
    }
    if not p.exists():
        rep["error"] = "file not found"
        return rep

    size = p.stat().st_size
    rep["size_bytes"] = size
    if size > limit:
        # 31006 5.1.2 注为「建议」，故提示分片而非判失败
        rep["warnings"].append(
            "size %.2f MB exceeds suggested limit %.2f MB; consider sharding"
            % (size / MIB, limit / MIB))

    with zipfile.ZipFile(p) as zf:
        entries = frame_entries(zf)
        rep["frame_count"] = len(entries)
        if not entries:
            rep["error"] = "no <run_id>/<index> entries"
            return rep
        idx = [i for i, _ in entries]
        rep["contiguous"] = idx == list(range(len(idx)))
        if not rep["contiguous"]:
            rep["bad_entries"].append(
                "frame indices not contiguous from 0: min=%d max=%d n=%d" % (min(idx), max(idx), len(idx)))

        # 用**首帧**的 schema_version 决定必需字段（同一文件内应一致）
        first = json.loads(zf.read(entries[0][1]).decode("utf-8"))
        # 版本判定**只能来自复盘文件自身**。
        # ⚠ 不能回退到 schema 的版本：历史 v1 文件没有 meta，若回退就会把 v1 文件
        # 当成 v1.1 来判，于是「缺 meta/agent/human」被报成失败 —— 而那恰恰是
        # I15 要记录的**缺口**，不是文件的错。缺 meta 即判定为 1.0.0。
        _meta = first.get("meta") if isinstance(first.get("meta"), dict) else {}
        version = str(_meta.get("schema_version") or "1.0.0")
        req = _frame_required(schema, version)
        declared_v1 = _ver_tuple(version) < (1, 1, 0)
        rep["declared_version"] = version
        # 历史 v1 文件不应因缺 v1.1 新字段而失败
        if declared_v1:
            rep["warnings"].append(
                "v1 replay: meta/agent/human absent by design (dual-core gap recorded in schema.dual_core)")

        missing = set()
        for _, name in entries:
            frame = json.loads(zf.read(name).decode("utf-8"))
            if not isinstance(frame, dict):
                rep["bad_entries"].append("frame %s is %s, expected object" % (name, type(frame).__name__))
                continue
            missing |= {k for k in req if k not in frame}
        rep["missing_keys"] = sorted(missing)

    rep["ok"] = not rep["missing_keys"] and not rep["bad_entries"]
    return rep


class Replay:
    """按索引 / 按 tick 定位并读取帧（I15 步骤 4：按索引回放）。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise ReplayFormatError("replay not found: %s" % self.path)
        self._zf = zipfile.ZipFile(self.path)
        self._entries = frame_entries(self._zf)
        if not self._entries:
            raise ReplayFormatError("no frame entries in %s" % self.path)
        self._by_index = {i: n for i, n in self._entries}
        self._tick_index: Optional[Dict[int, int]] = None

    def __enter__(self) -> "Replay":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._zf.close()

    @property
    def run_id(self) -> str:
        return self._entries[0][1].rsplit("/", 1)[0]

    @property
    def frame_count(self) -> int:
        return len(self._entries)

    def index_range(self) -> Tuple[int, int]:
        idx = [i for i, _ in self._entries]
        return min(idx), max(idx)

    def frame(self, index: int) -> Dict[str, Any]:
        """按**帧序号**读取（O(1)）。"""
        name = self._by_index.get(index)
        if name is None:
            lo, hi = self.index_range()
            raise ReplayFormatError("frame index %d out of range [%d, %d]" % (index, lo, hi))
        return json.loads(self._zf.read(name).decode("utf-8"))

    def _build_tick_index(self) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for i, name in self._entries:
            step = json.loads(self._zf.read(name).decode("utf-8")).get("time", {}).get("cur_step")
            if isinstance(step, int) and step not in out:  # 同 step 多帧时取第一帧
                out[step] = i
        return out

    def frame_at_tick(self, tick: int) -> Dict[str, Any]:
        """按**仿真步**读取（首次调用建索引，之后 O(1)）。"""
        if self._tick_index is None:
            self._tick_index = self._build_tick_index()
        idx = self._tick_index.get(tick)
        if idx is None:
            known = sorted(self._tick_index)
            raise ReplayFormatError(
                "tick %d not present (available steps: %d..%d, %d distinct)"
                % (tick, known[0], known[-1], len(known)))
        return self.frame(idx)

    def iter_frames(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        for i, name in self._entries:
            yield i, json.loads(self._zf.read(name).decode("utf-8"))


def run_summary(path: Path) -> Dict[str, Any]:
    """一次遍历给出可读摘要：步范围、阵营、单位数、分数轨迹端点。"""
    with Replay(path) as r:
        first = r.frame(r.index_range()[0])
        last = r.frame(r.index_range()[1])
        steps = []
        for _, f in r.iter_frames():
            s = f.get("time", {}).get("cur_step")
            if isinstance(s, int):
                steps.append(s)
        return {
            "run_id": r.run_id, "frames": r.frame_count,
            "step_min": min(steps) if steps else None,
            "step_max": max(steps) if steps else None,
            "scenario_id": first.get("scenario_id"),
            "terrain_id": first.get("terrain_id"),
            "operators_first": len(first.get("operators") or []),
            "operators_last": len(last.get("operators") or []),
            "scores_first": first.get("scores"),
            "scores_last": last.get("scores"),
        }
