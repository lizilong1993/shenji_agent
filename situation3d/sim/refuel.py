"""空中加油（可选，仅地理引擎）：一对加油机/受油机的有限输油窗口。

建模口径（TODOLIST §0.5「空中加油」待定三问的最小答案）：

* **配对**——案例显式写出加油机与受油机各一个 uid，不做"就近自动分配"。
  两机必须同阵营、同在空域（``domain == "air"``）、都不在搭载状态，
  且水平距离与高度差同时落在案例给的阈值内才算会合。
* **窗口**——首次会合成立时开窗，``window_s`` 秒后自动关闭；窗口内任一方
  不再满足会合条件，或受油机满油、加油机余额耗尽，也立即关闭。
  关闭后不再自动重开（同一次会合只加一次油），需要案例另写一次配置。
* **输油**——每步转移 ``min(rate_per_s * dt, 受油缺口, 加油机余额)``，
  余额同步扣减。受油机不因缺油改变机动：地理引擎本来就没有飞行油耗，
  本模块不引入通用油耗，只给显式配置的案例加上"油从哪来、加多少"。

案例里没有 ``refuel`` 块时 ``World.refuel`` 为 None，所有路径与今天一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from geo.geodesy import haversine_m


def _num(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number


def _uid(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class Refuel:
    tanker: int
    receiver: int
    rate_per_s: float = 0.02
    range_m: float = 300.0
    alt_tolerance_m: float = 300.0
    window_s: Optional[float] = 600.0
    supply: float = 0.5
    started_s: Optional[float] = None
    done: bool = False              # 关过一次就不再重开（同一次会合只加一次油）

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], warnings=None,
                  side_of=None) -> Optional["Refuel"]:
        """解析案例块；写错返回 None 并留下一条警告。"""
        warn = warnings if warnings is not None else []
        if not isinstance(payload, dict):
            warn.append("refuel: 不是对象，已忽略")
            return None
        tanker = _uid(payload.get("tanker", payload.get("tanker_uid")))
        receiver = _uid(payload.get("receiver", payload.get("receiver_uid")))
        if tanker is None or receiver is None:
            warn.append("refuel: 需要 tanker 与 receiver 两个 uid，已忽略")
            return None
        if tanker == receiver:
            warn.append("refuel: tanker 与 receiver 不能是同一单位，已忽略")
            return None
        if side_of is not None:
            try:
                if side_of(tanker) != side_of(receiver):
                    warn.append("refuel: 两机不同阵营，已忽略")
                    return None
            except Exception:  # noqa: BLE001 - uid 不在册
                warn.append("refuel: 引用了不存在的单位，已忽略")
                return None
        rate = _num(payload.get("rate_per_s"), 0.02)
        supply = _num(payload.get("supply"), 0.5)
        if rate <= 0.0:
            warn.append("refuel: rate_per_s 必须为正，已忽略")
            return None
        if supply <= 0.0:
            warn.append("refuel: supply 必须为正，已忽略")
            return None
        window = payload.get("window_s")
        window_s = None if window is None else _num(window, 0.0)
        if window_s is not None and window_s <= 0.0:
            warn.append("refuel: window_s 必须为正，已忽略")
            return None
        return cls(tanker=tanker, receiver=receiver, rate_per_s=rate,
                   range_m=_num(payload.get("range_m"), 300.0),
                   alt_tolerance_m=_num(payload.get("alt_tolerance_m"), 300.0),
                   window_s=window_s, supply=min(1.0, supply))

    # -- queries ------------------------------------------------------------
    def active(self) -> bool:
        return self.started_s is not None

    def _pair(self, world) -> Tuple[Optional[object], Optional[object]]:
        return (world.entities.get(self.tanker),
                world.entities.get(self.receiver))

    def rendezvous_ok(self, world) -> bool:
        tanker, receiver = self._pair(world)
        if tanker is None or receiver is None:
            return False
        for entity in (tanker, receiver):
            if not entity.alive or getattr(entity, "on_board", False):
                return False
            if getattr(entity, "domain", "") != "air":
                return False
            if entity.affiliation != tanker.affiliation:
                return False
        distance = haversine_m((tanker.pose.lon, tanker.pose.lat),
                               (receiver.pose.lon, receiver.pose.lat))
        return (distance <= self.range_m
                and abs(tanker.pose.alt_m - receiver.pose.alt_m)
                <= self.alt_tolerance_m)

    # -- per step -----------------------------------------------------------
    def apply(self, world, dt: float) -> None:
        """一步的输油记账；dt <= 0 时只做开窗/关窗判定。"""
        now = world.time_s
        if self.done:
            return
        if not self.active():
            if self.rendezvous_ok(world) and self.supply > 0.0:
                receiver = world.entities[self.receiver]
                if receiver.fuel < 1.0:
                    self.started_s = now
                    world.emit("refuel_start", [self.tanker, self.receiver],
                               "%s 为 %s 开始空中加油"
                               % (world.entities[self.tanker].name,
                                  receiver.name))
            return
        if not self._still_flowing(world):
            self._close(world, "加油窗口结束")
            return
        if dt <= 0.0:
            return
        if self.window_s is not None:
            # 快进路径一次能给到很大的 dt，按窗口剩余时长截断，避免超窗输油。
            remaining = float(self.window_s) - (world.time_s -
                                                float(self.started_s))
            dt = min(float(dt), max(0.0, remaining))
            if dt <= 0.0:
                return
        receiver = world.entities[self.receiver]
        amount = min(self.rate_per_s * dt, max(0.0, 1.0 - receiver.fuel),
                     self.supply)
        if amount <= 0.0:
            self._close(world, "受油机已满油或加油机余额耗尽")
            return
        receiver.fuel += amount
        self.supply -= amount

    def _still_flowing(self, world) -> bool:
        if self.supply <= 0.0:
            return False
        if (self.window_s is not None
                and world.time_s - float(self.started_s) >= self.window_s):
            return False
        return self.rendezvous_ok(world)

    def _close(self, world, reason: str) -> None:
        receiver = world.entities.get(self.receiver)
        tanker = world.entities.get(self.tanker)
        if tanker is not None and receiver is not None:
            world.emit("refuel_end", [self.tanker, self.receiver],
                       "%s <- %s 空中加油结束：%s（受油 %.0f%%）"
                       % (tanker.name, receiver.name, reason,
                          receiver.fuel * 100.0))
        self.started_s = None
        self.done = True

    def describe(self) -> dict:
        return {"tanker": self.tanker, "receiver": self.receiver,
                "rate_per_s": self.rate_per_s, "range_m": self.range_m,
                "alt_tolerance_m": self.alt_tolerance_m,
                "window_s": self.window_s,
                "supply": round(self.supply, 3),
                "active": self.active()}


def install(world, scenario: dict) -> None:
    payload = scenario.get("refuel")
    if not payload:
        return
    warnings: list = []

    def _side_of(uid: int):
        entity = world.entities[int(uid)]
        return entity.affiliation

    refuel = Refuel.from_dict(payload, warnings, _side_of)
    if warnings:
        world.refuel_warning = "；".join(warnings)
    world.refuel = refuel
