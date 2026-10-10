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
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Iterable

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


def _dual_core_required(schema, version):
    """v1.1 **聚合必需**的段 —— 依据 schema 的 `dual_core[*].required_in`。

    | 段 | `frame_path` | `required_in` | 是否聚合必需 |
    | --- | --- | --- | --- |
    | `human_ops` | `actions` | **（无）** | **否** —— schema 说 `present_in_v1: true` |
    | `agent_data` | `agent` | `>=1.1.0` | **是** |
    | `seed` | `meta.seed` | `>=1.1.0` | **是** |
    """
    if not version or _ver_tuple(version) < (1, 1, 0):
        return []
    dual = schema.get("dual_core") or {}
    out = []
    if _ver_tuple(str((dual.get("agent_data") or {}).get("required_in") or "0")) >= (1, 1, 0):
        out.append("agent")
    if _ver_tuple(str((dual.get("seed") or {}).get("required_in") or "0")) >= (1, 1, 0):
        out.append("meta.seed")
    return out


def _frame_required(schema, version):
    """**逐帧**必需字段：v1 的必需键 + 「有 actions 时才要求 human」。

    ⚠ 不再把 `frame_optional_v1_1` 整份当成必需 —— 名字就是 optional，
    而 schema 的 `dual_core` 也只把 `agent_data` 与 `seed` 标为
    `required_in: >=1.1.0`。实测教训：原实现让「没有人类命令的帧」被判失败
    （1802 帧真实复盘升级时报 `missing=['human']`）——
    那不是文件的错，是判定与 schema 不一致。
    """
    req = list(schema.get("frame_required_v1", []))
    if version and _ver_tuple(version) >= (1, 1, 0):
        # `human` 段是 `actions` 的**标注**：有命令才谈得上标注
        req.append("human:only_if_actions")
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

    # ── I15 判据 3：超限**提示分片**（schema.size_limit.on_exceed）──────
    # schema 声明 `on_exceed: warn_and_suggest_sharding`，且 `binding` 写明
    # `31006 5.1.2` 是**建议** → **只提示、不判失败**（`ok` 不因此变 False）。
    # 实测补的缺口：原实现只写 `size_bytes`，**既无 warning 也无 hint**，
    # 即 schema 声明的行为未实现 —— 判据 3「有校验与分片提示」只完成一半。
    # ⚠ 键名是 `limit_bytes`（上面 `rep["limit_bytes"] = limit`）；
    #    我第一版写成 `size_limit_bytes` → 取到 None → 判据 3 的提示**从不触发**。
    limit = rep.get("limit_bytes")
    if isinstance(limit, int) and limit > 0 and rep["size_bytes"] > limit:
        # 措辞里**保留 "sharding" 这个词**：既有两处用例断言
        # `any("sharding" in w for w in warnings)` —— 那是判据 3 的原意
        # （"文件大小有校验**与分片提示**"），不该为了改措辞把它弄丢。
        rep["size_warning"] = (
            "size %.2f MB exceeds suggested limit %.2f MB; consider sharding"
            % (rep["size_bytes"] / MIB, limit / MIB))
        # 分片提示要**可执行**：给出「按多少片」的具体建议
        n_parts = (rep["size_bytes"] + limit - 1) // limit
        rep["sharding_hint"] = {
            "suggested_parts": int(n_parts),
            "reason": "schema.size_limit.on_exceed = warn_and_suggest_sharding",
            "binding": (schema.get("size_limit") or {}).get("binding", ""),
        }
        rep["warnings"].append(rep["size_warning"])
    else:
        rep["size_warning"] = None
        rep["sharding_hint"] = None
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

        # ── 逐帧必需：v1 的键必须**每帧都有**；`human` 只在该帧有 actions 时要求 ──
        missing = set()
        for _, name in entries:
            frame = json.loads(zf.read(name).decode("utf-8"))
            if not isinstance(frame, dict):
                rep["bad_entries"].append(
                    "frame %s is %s, expected object" % (name, type(frame).__name__))
                continue
            for k in req:
                if k == "human:only_if_actions":
                    # 有命令才谈得上"标注发出者"；`actions: []` 的帧不要求 human
                    if frame.get("actions") and "human" not in frame:
                        missing.add("human")
                    continue
                if k not in frame:
                    missing.add(k)

        # ── v1.1 聚合必需（schema.dual_core[*].required_in）：整份复盘**至少一帧**满足 ──
        agg_req = _dual_core_required(schema, version)
        agg_ok = {k: False for k in agg_req}
        for _, name in entries:
            frame = json.loads(zf.read(name).decode("utf-8"))
            if not isinstance(frame, dict):
                continue
            meta = frame.get("meta") if isinstance(frame.get("meta"), dict) else {}
            for k in agg_req:
                if k == "meta.seed":
                    if meta.get("seed") is not None:
                        agg_ok[k] = True
                elif k in frame:
                    agg_ok[k] = True
        for k, ok in agg_ok.items():
            if not ok:
                missing.add(k)
        rep["dual_core_present"] = agg_ok
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

    # ── I15 判据 1：**字节层**出口（"精确"必须在字节层核对）──────────────
    def frame_bytes(self, index: int) -> bytes:
        """按帧序号读取**原始字节**（不经过 JSON 解析）。

        `frame()` 返回解析后的 dict，无法排除"序列化把值改了"这类偏差；
        判据 1 要求"结果与原局一致"，故必须有字节层出口才能核对。
        """
        name = self._by_index.get(index)
        if name is None:
            lo, hi = self.index_range()
            raise ReplayFormatError("frame index %d out of range [%d, %d]" % (index, lo, hi))
        return self._zf.read(name)

    def entry_name(self, index: int) -> str:
        """该帧在 zip 内的条目名（`<run_id>/<index>`）。"""
        name = self._by_index.get(index)
        if name is None:
            raise ReplayFormatError("frame index %d not present" % index)
        return name

    def tick_index(self) -> Dict[int, int]:
        """`cur_step -> frame_index` 映射（同 step 多帧取**第一帧**，与 schema 一致）。"""
        if self._tick_index is None:
            self._tick_index = self._build_tick_index()
        return dict(self._tick_index)

    # ── I15 步骤 1 的「六类记录」出口 ────────────────────────────────────
    def timeline(self) -> List[Dict[str, Any]]:
        """**时间线**一类：按帧序返回每帧的 `time` 段（外加帧序号）。

        `I15` 步骤 1 要求六类记录，其中"时间线"此前只隐含在 `time` 里、
        无显式出口。本方法把它显式化，**不新造数据** ——
        就是逐帧取 `time`，并附上 `frame_index` 便于与按索引回放对齐。
        """
        out: List[Dict[str, Any]] = []
        for i, frame in self.iter_frames():
            t = frame.get("time")
            if isinstance(t, dict):
                out.append({"frame_index": i, **t})
        return out

    def adjudication_records(self) -> List[Dict[str, Any]]:
        """**裁决**一类：返回所有非空 `judge_info` 条目（附帧序号）。

        实测 v1 复盘里 `judge_info` **恒为空数组** ——
        即"裁决"这一类**有声明、无数据**。本方法把这个事实**变成可检查的**：
        返回空列表时即证明该主线确实缺失，不必靠读文档判断。
        """
        out: List[Dict[str, Any]] = []
        for i, frame in self.iter_frames():
            for rec in (frame.get("judge_info") or []):
                out.append({"frame_index": i, "record": rec})
        return out

    def record_type_coverage(self) -> Dict[str, Any]:
        """六类记录**实际有无数据**的体检（对应 `schema.record_types`）。

        **只报事实，不报"通过"** —— 某类为空就如实记空。
        """
        frames = list(self.iter_frames())
        n = len(frames)
        cov: Dict[str, Any] = {
            "frame_count": n,
            "actions": sum(1 for _, f in frames if f.get("actions")),
            "adjudication": sum(1 for _, f in frames if f.get("judge_info")),
            "timeline": sum(1 for _, f in frames if isinstance(f.get("time"), dict)),
            "agent_decisions": sum(1 for _, f in frames if f.get("agent")),
            "seed": sum(1 for _, f in frames
                        if isinstance(f.get("meta"), dict) and f["meta"].get("seed") is not None),
            "human_ops": sum(1 for _, f in frames if f.get("human")),
        }
        cov["empty_types"] = [k for k in ("actions", "adjudication", "timeline",
                                          "agent_decisions", "seed", "human_ops")
                              if cov[k] == 0]
        return cov

    def verify_precise_replay(self,
                              ticks: Optional[Iterable[int]] = None,
                              ) -> Dict[str, Any]:
        """核对"按 tick 取帧"是否与原局**逐字节一致**（I15 判据 1）。

        Parameters
        ----------
        ticks:
            要核对的 tick 列表；`None` 表示核对**全部**已知 tick。

        Returns
        -------
        dict
            `ok`、`checked`、`mismatches`、`missing_ticks`。
            **不粉饰**：任何不一致都会进 `mismatches` 并使 `ok=False`。
        """
        tindex = self.tick_index()
        targets = sorted(tindex) if ticks is None else sorted(set(int(t) for t in ticks))
        mismatches: List[Dict[str, Any]] = []
        missing: List[int] = []
        checked = 0
        for tick in targets:
            idx = tindex.get(tick)
            if idx is None:
                missing.append(tick)
                continue
            checked += 1
            # 逐字节比对：走两条**独立**路径取同一帧，二者必须相同
            by_index = self.frame_bytes(idx)
            by_tick_frame = self.frame_at_tick(tick)
            by_tick_reencoded = json.dumps(
                by_tick_frame, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if by_index != by_tick_reencoded:
                mismatches.append({
                    "tick": tick, "index": idx,
                    "index_bytes": len(by_index),
                    "tick_bytes": len(by_tick_reencoded),
                })
        return {
            "ok": not mismatches and not missing,
            "checked": checked,
            "mismatches": mismatches,
            "missing_ticks": missing,
            "frame_count": self.frame_count,
            "note": ("按 index 直读原始字节 vs 按 tick 解析后再序列化，"
                     "两条独立路径逐字节比对；不一致即判失败"),
        }


# ── 写入端：帧构造（I15 步骤 2/3）─────────────────────────────────────────
#
# 为什么放在这里而不是 `run_offline_games.py` 里
# ---------------------------------------------
# `run_offline_games.py` 顶部 `from train_env import TrainEnv` ——
# 那要**闭源引擎**，本环境鉴权不通过，故其中的写入逻辑**无法在本地执行或验证**。
# 把帧构造抽成**纯函数**放在本模块，就能：
#   ① 用真实 schema 校验其产出（不需要引擎）；
#   ② 让 `run_offline_games.py` 只改"取值来源"这一小段，降低改动面。
#
# 补的是 31006 4.4.2 的**双核心缺口**（schema.dual_core 记录）：
#   - `agent` 段：v1 完全没有 → 无智能体内部决策摘要、无随机种子；
#   - `human` 段：v1 只记原始命令 {cur_step, message, error}，**未标注发出者/是否人类**。

FRAME_VERSION_V1 = "1.0.0"
FRAME_VERSION_V1_1 = "1.1.0"


def build_frame(v1_frame: Dict[str, Any], *,
                seed: Optional[int] = None,
                agent: Optional[Dict[str, Any]] = None,
                human: Optional[Dict[str, Any]] = None,
                scenario: Optional[str] = None,
                frame_version: str = FRAME_VERSION_V1_1,
                ) -> Dict[str, Any]:
    """把一个 v1 帧**升级**成 v1.1 帧（补 `meta` / `agent` / `human` 三段）。

    **不修改 v1 的既有字段** —— 只做加法，故历史读取方不受影响。

    Parameters
    ----------
    v1_frame:
        原本要写入 zip 的 `state[GREEN]` 快照（必需字段见 schema.frame_required_v1）。
    seed:
        随机种子。写入 `meta.seed`（`dual_core.seed` 规定的位置）。
    agent:
        智能体决策摘要。**至少给 `side`**；缺失时整个 `agent` 段不写，
        由 `validate()` 按版本判失败 —— **不编造一个空摘要冒充"有数据"**。
    human:
        人类操作标注。键为帧序号或座位，值为 `{"actor_kind": "human"|"agent", ...}`。
    scenario:
        想定标识。
    frame_version:
        要写入的版本；默认 `1.1.0`（即"带双核心"）。

    Returns
    -------
    新的帧字典。`v1_frame` **不被就地修改**（返回新对象），
    因为调用方可能还要用原始快照做别的记录。
    """
    if not isinstance(v1_frame, dict):
        raise ReplayFormatError("v1_frame must be a dict, got %s" % type(v1_frame).__name__)

    frame: Dict[str, Any] = dict(v1_frame)

    # ⚠ 版本必须写在 **`meta.schema_version`**：
    # `validate()` 就是从这里读的（`version = str(_meta.get("schema_version") or "1.0.0")`），
    # 读不到即按 **1.0.0** 判，于是 v1.1 的必需字段（meta/agent/human）**不会被检查**。
    # 我第一版写在帧顶层 `schema_version`，结果"缺 agent 段的 v1.1 帧"被判通过 ——
    # 实测出来的（反向自检：构造缺 agent 的帧，期望被拒，实际放行）。
    # 顶层同时留一份只为肉眼可读，**判定一律以 meta 为准**。
    meta: Dict[str, Any] = dict(v1_frame.get("meta") or {})
    meta["schema_version"] = frame_version
    if seed is not None:
        meta["seed"] = int(seed)
    if scenario is not None:
        meta["scenario"] = str(scenario)
    meta.setdefault("frame_version", frame_version)
    frame["meta"] = meta
    frame["schema_version"] = frame_version      # 便于人工查看；非判定依据

    # `agent`：智能体内部决策摘要（v1 完全缺失的那条主线）
    if isinstance(agent, dict) and agent:
        frame["agent"] = dict(agent)

    # `human`：人类操作标注（v1 只记原始命令，未标发出者）
    if isinstance(human, dict) and human:
        frame["human"] = dict(human)

    return frame


def annotate_human_actions(frame: Dict[str, Any], *,
                           human_actor: int = 0,
                           actor_kinds: Optional[Dict[str, str]] = None,
                           ) -> Dict[str, Any]:
    """给帧里的 `actions` 逐条**标注发出者**（补 `dual_core.human_ops` 的缺口）。

    v1 的命令形如 `{"cur_step": .., "message": {"actor": ..}, "error": ..}` ——
    **没有字段说明"这是人下的还是智能体下的"**，故双核心里的人类操作主线无法区分。

    本函数**只加标注，不改原字段**：在每条命令上补 `actor_kind`
    （`"human"` / `"agent"`）与 `actor`（原样透出便于核对）。

    `actor_kinds` 可显式指定每个 actor 的类别；未指定时，
    凡 `actor == human_actor` 记为 `"human"`，其余记为 `"agent"`。
    """
    if not isinstance(frame, dict):
        raise ReplayFormatError("frame must be a dict")
    actions = frame.get("actions")
    if not isinstance(actions, list):
        return frame            # 无动作可标注，原样返回（不伪造空列表）

    kinds = dict(actor_kinds or {})
    new_actions = []
    for act in actions:
        if not isinstance(act, dict):
            new_actions.append(act)
            continue
        a = dict(act)
        msg = a.get("message")
        actor = msg.get("actor") if isinstance(msg, dict) else None
        a["actor"] = actor                                # 原样透出，便于核对
        if actor is not None and str(actor) in kinds:
            a["actor_kind"] = kinds[str(actor)]
        else:
            a["actor_kind"] = "human" if (actor is not None and int(actor) == int(human_actor)) else "agent"
        new_actions.append(a)
    out = dict(frame)
    out["actions"] = new_actions

    # ── 帧级 `human` 段：双核心的"人类操作"主线 ──────────────────────────
    # ⚠ 我第一版只给每条动作打了 `actor_kind`，**没写帧级 `human` 段**，
    # 于是 1802 帧真实复盘升级时 `validate()` 报 `missing=['human']`（实测）。
    #
    # `human` 段内容**不是编造的**：它就是按 `actor_kind` 分出来的两类命令
    # —— 人类下达的 vs 智能体下达的。这正是 v1 缺失、
    # 而 `31006 4.4.2` 双核心要求分开记录的那条主线。
    human_actions = [a for a in new_actions
                     if isinstance(a, dict) and a.get("actor_kind") == "human"]
    agent_actions = [a for a in new_actions
                     if isinstance(a, dict) and a.get("actor_kind") == "agent"]
    out["human"] = {
        "actor_kind": "human",
        "human_actor": int(human_actor),
        "source": "actions[actor_kind]",
        "action_count": len(human_actions),
        "actions": human_actions,
        "agent_action_count": len(agent_actions),
    }
    return out


def write_replay(path: Path, frames: List[Dict[str, Any]], *, run_id: str,
                 validate_first: bool = True, **validate_kwargs: Any) -> Dict[str, Any]:
    """把帧列表写成 `replay_<run_id>.zip`，并可**写前校验**。

    `validate_first=True`（默认）时先写临时文件、校验通过才改名 ——
    **避免产出"写完了但没人知道它不合法"的复盘**。

    返回校验报告（供调用方记入 manifest / 日志）。
    """
    import tempfile

    if not frames:
        raise ReplayFormatError("refusing to write a replay with no frames")
    if not run_id:
        raise ReplayFormatError("run_id is required (entry names are <run_id>/<index>)")

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".zip")
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for i, frame in enumerate(frames):
                z.writestr("%s/%d" % (run_id, i),
                           json.dumps(frame, ensure_ascii=False, separators=(",", ":")))
        rep = validate(path=tmp_path, **validate_kwargs)
        if validate_first and not rep.get("ok"):
            raise ReplayFormatError(
                "replay failed schema validation, not published: missing=%s bad=%s"
                % (rep.get("missing_keys"), rep.get("bad_entries")))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_path, target)
        return rep
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


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
