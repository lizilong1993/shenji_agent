"""指挥链中断（可选）：把"命令下不来"和"同级看不见彼此"变成可配置的窗口。

建模口径（本模块据此实现，见 TODOLIST §5.4 的三问）：

* **断的是哪一层**——拆两层，各自只锁一个动作：
  - ``upper``：上级命令送达。窗口内不给受影响单位下发**新的** commander 指令；
    已下发的指令按原有到期规则自然失效，自动还击不受影响。
  - ``peer``：同级协同。窗口内 ``radio_share`` 不再把这两家的探测结果互相广播。
* **恢复到什么程度算恢复**——两种判据都支持，显式优先：
  ``until_s``（按仿真时间收口）或 ``restore_when.all_alive``（每步重算，
  相关单位全部存活即恢复，模仿 gp-rts"源站存活即立即恢复"）。
* **多跳 ``comms_range_m`` 怎么算**——**不做多跳**。共享仍是单跳
  （``sim/spotting.py`` 的既有语义与测试 ``test_radio_share_does_not_multihop``
  守住这一点），本模块只决定某一跳在窗口内是否可达。

案例里没有 ``command_links`` 块时，``World.command_links`` 为 None，
所有路径与今天完全一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

KINDS = ("upper", "peer")


def _num(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number


def _uids(value: Any) -> Optional[frozenset]:
    """``"all"`` 或缺省表示整侧；否则是 uid 集合。"""
    if value is None or (isinstance(value, str) and value.strip().lower() == "all"):
        return None
    if not isinstance(value, (list, tuple)):
        return frozenset()
    out = set()
    for item in value:
        try:
            out.add(int(item))
        except (TypeError, ValueError):
            continue
    return frozenset(out)


@dataclass(frozen=True)
class Cut:
    kind: str                       # "upper" | "peer"
    side: Optional[int] = None      # upper 用：受影响阵营（None = 不限）
    units: Optional[frozenset] = None   # upper 用：受影响 uid；None = 整侧
    pair: Optional[Tuple[int, int]] = None  # peer 用：一对 uid
    from_s: float = 0.0
    until_s: Optional[float] = None
    all_alive: bool = False         # restore_when.all_alive
    note: str = ""

    def active(self, now: float, world) -> bool:
        """窗口何时算开着：``from_s`` 之后，直到任一恢复判据成立。

        恢复判据两个，满足**任一**即恢复（与"恢复到什么程度算恢复"一致：
        写哪种判据就有哪种语义，两种都写就取先到者）：
          * ``until_s``     —— 时间到；
          * ``all_alive``   —— 被监视单位全部存活。
        两个都缺 = 窗口永不结束（解析时会留一条警告）。
        """
        if now < self.from_s:
            return False
        if self.until_s is not None and now > self.until_s:
            return False
        if self.all_alive:
            uids = self._watch_uids()
            if uids and all(self._alive(world, uid) for uid in uids):
                return False
        return True

    def _watch_uids(self) -> List[int]:
        if self.pair:
            return [self.pair[0], self.pair[1]]
        return sorted(self.units) if self.units else []

    @staticmethod
    def _alive(world, uid: int) -> bool:
        entity = world.entities.get(int(uid))
        return bool(entity is not None and entity.alive)


class CommandLinks(object):
    """一组已声明的链路中断窗口；``active_*`` 是查询入口。"""

    def __init__(self, cuts: Sequence[Cut]):
        self.cuts: List[Cut] = list(cuts)

    # -- parsing ------------------------------------------------------------
    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]],
                  warnings: Optional[List[str]] = None,
                  side_of=None) -> Optional["CommandLinks"]:
        """解析案例块；空块返回 None（等于没配）。

        ``side_of(uid) -> side`` 用来校验 peer 对是否同阵营——跨阵营的
        "同级协同中断"没有意义，应当报出来而不是悄悄放行。
        """
        if not payload:
            return None
        if isinstance(payload, str) and payload.strip().lower() in ("none", "off"):
            return None
        if not isinstance(payload, dict):
            raise ValueError("command_links must be an object")
        warn = warnings if warnings is not None else []
        cuts: List[Cut] = []
        for kind in KINDS:
            for raw in payload.get(kind + "_cuts") or []:
                cut = _parse_cut(kind, raw, warn, side_of)
                if cut is not None:
                    cuts.append(cut)
        return cls(cuts) if cuts else None

    # -- queries ------------------------------------------------------------
    def upper_cut(self, uid: int, side: int, now: float, world) -> Optional[Cut]:
        """该单位此刻是否收不到上级的新命令。"""
        for cut in self.cuts:
            if cut.kind != "upper" or not cut.active(now, world):
                continue
            if cut.side is not None and int(cut.side) != int(side):
                continue
            if cut.units is None or int(uid) in cut.units:
                return cut
        return None

    def peer_blocked(self, a: int, b: int, now: float, world) -> bool:
        """这两家此刻是否不能互相共享接触。"""
        pair = (min(int(a), int(b)), max(int(a), int(b)))
        for cut in self.cuts:
            if cut.kind != "peer" or not cut.pair:
                continue
            if tuple(cut.pair) != pair:
                continue
            if cut.active(now, world):
                return True
        return False

    def describe(self) -> List[dict]:
        out = []
        for cut in self.cuts:
            item = {"kind": cut.kind, "from_s": cut.from_s,
                    "until_s": cut.until_s, "all_alive": cut.all_alive}
            if cut.side is not None:
                item["side"] = cut.side
            if cut.units is not None:
                item["units"] = sorted(cut.units)
            if cut.pair is not None:
                item["pair"] = list(cut.pair)
            if cut.note:
                item["note"] = cut.note
            out.append(item)
        return out


def _parse_cut(kind: str, raw: Any, warn: List[str], side_of) -> Optional[Cut]:
    if not isinstance(raw, dict):
        warn.append("command_links: %s_cuts 的条目不是对象，已跳过" % kind)
        return None
    from_s = _num(raw.get("from_s", raw.get("cut_at_s")), 0.0)
    until = raw.get("until_s", raw.get("restore_at_s"))
    until_s = None if until is None else _num(until, 0.0)
    restore = raw.get("restore_when") or {}
    all_alive = bool(isinstance(restore, dict) and restore.get("all_alive"))
    note = str(raw.get("note") or "")
    if kind == "upper":
        side_raw = raw.get("side")
        side = None
        if side_raw is not None:
            try:
                side = int(side_raw)
            except (TypeError, ValueError):
                warn.append("command_links: upper_cuts 的 side 不是整数，已跳过")
                return None
        return Cut(kind="upper", side=side, units=_uids(raw.get("units")),
                   from_s=from_s, until_s=until_s, all_alive=all_alive,
                   note=note)
    pair_raw = raw.get("units") or raw.get("pair")
    pair = _uids(pair_raw)
    if pair is None or len(pair) != 2:
        warn.append("command_links: peer_cuts 需要恰好两个 uid，已跳过")
        return None
    a, b = sorted(pair)
    if side_of is not None:
        try:
            if side_of(a) != side_of(b):
                warn.append("command_links: peer_cuts 的两个单位不同阵营，已跳过")
                return None
        except Exception:  # noqa: BLE001 - uid 可能不在册
            warn.append("command_links: peer_cuts 引用了不存在的单位，已跳过")
            return None
    if until_s is None and not all_alive:
        warn.append("command_links: peer_cuts 没有 until_s 或 restore_when，"
                    "窗口永不结束——多半是漏写")
    return Cut(kind="peer", pair=(a, b), from_s=from_s, until_s=until_s,
               all_alive=all_alive, note=note)
