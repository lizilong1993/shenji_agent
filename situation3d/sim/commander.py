"""Pluggable commander layer: model answers become expiring orders.

Architecture borrowed from Claude-of-Tanks ``src/game/jevCommander.ts`` /
``jevProtocol.ts`` (MIT, Copyright © 2026 Kevin B. Liu): a model (or a rule
fallback) receives a compact, side-filtered state and returns a small set of
posture/target/objective orders.  The orders expire on their own; if the model
is slow or unavailable, the rule commander or the existing controller keeps
running.  No external service is bundled: ``RemoteCommander`` is an optional
HTTP adapter, ``RuleCommander`` is deterministic and offline.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

POSTURES = ("hold", "push", "flank_left", "flank_right", "retreat", "capture", "support")


@dataclass(frozen=True)
class CommanderOrder:
    uid: int
    posture: str = "hold"
    target_uid: Optional[int] = None
    lon: Optional[float] = None
    lat: Optional[float] = None
    expires_s: float = 10.0
    source: str = "rule"


class CommanderModel(Protocol):
    def decide(self, state: dict) -> List[CommanderOrder]: ...


def _hp_fraction(unit: dict) -> float:
    blood = float(unit.get("blood", 1) or 0)
    max_blood = float(unit.get("max_blood", 1) or 1)
    return blood / max(1.0, max_blood)


def _nearest_objective(unit: dict, objectives: List[dict],
                       prefer_owned: Optional[bool] = None) -> Optional[dict]:
    if not objectives:
        return None
    ox, oy = unit.get("lon"), unit.get("lat")
    if ox is None or oy is None:
        return objectives[0]
    ranked = []
    for obj in objectives:
        if prefer_owned is True and obj.get("owner") != unit.get("affiliation"):
            continue
        if prefer_owned is False and obj.get("owner") == unit.get("affiliation"):
            continue
        dx = float(obj.get("lon", 0.0)) - float(ox)
        dy = float(obj.get("lat", 0.0)) - float(oy)
        ranked.append((dx * dx + dy * dy, obj))
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1] if ranked else None


class RuleCommander(object):
    """Deterministic offline commander used as the default/fallback brain."""

    def decide(self, state: dict) -> List[CommanderOrder]:
        units = state.get("units") or []
        enemies = state.get("enemies") or []
        objectives = state.get("objectives") or []
        orders = []
        for unit in units:
            uid = int(unit["uid"])
            hp = _hp_fraction(unit)
            domain = str(unit.get("domain", "land"))
            ground = domain in ("land", "amphibious")
            usable_enemies = [e for e in enemies
                              if (str(e.get("domain", "land")) in ("land", "amphibious")) == ground]
            if hp <= 0.35:
                obj = _nearest_objective(unit, objectives, prefer_owned=True)
                if obj is None:
                    orders.append(CommanderOrder(uid=uid, posture="hold",
                                                 expires_s=5.0))
                    continue
                orders.append(CommanderOrder(
                    uid=uid, posture="retreat",
                    lon=obj.get("lon"), lat=obj.get("lat"),
                    expires_s=8.0))
                continue
            if not ground and not usable_enemies:
                orders.append(CommanderOrder(uid=uid, posture="hold",
                                             expires_s=8.0))
                continue
            if usable_enemies and unit.get("in_cover"):
                orders.append(CommanderOrder(uid=uid, posture="hold",
                                             target_uid=int(usable_enemies[0]["uid"]),
                                             expires_s=6.0))
                continue
            if usable_enemies:
                orders.append(CommanderOrder(uid=uid, posture="push",
                                             target_uid=int(usable_enemies[0]["uid"]),
                                             expires_s=6.0))
                continue
            obj = _nearest_objective(unit, objectives, prefer_owned=False)
            if obj is None:
                obj = _nearest_objective(unit, objectives)
            orders.append(CommanderOrder(
                uid=uid, posture="capture",
                lon=obj.get("lon") if obj else unit.get("lon"),
                lat=obj.get("lat") if obj else unit.get("lat"),
                expires_s=12.0))
        return orders


class RemoteCommander(object):
    """Optional HTTP commander adapter (JSON in / JSON out).

    The endpoint is expected to accept ``{"state": ...}`` and return
    ``{"orders": [{uid, posture, target_uid?, lon?, lat?, expires_s?}, ...]}``.
    Any network error, timeout or malformed response returns an empty list so
    the caller can fall back to the rule commander / existing controller.
    """

    def __init__(self, url: str, timeout_s: float = 2.0,
                 token: Optional[str] = None):
        self.url = str(url)
        self.timeout_s = float(timeout_s)
        self.token = token
        self.last_error: Optional[str] = None

    def decide(self, state: dict) -> List[CommanderOrder]:
        payload = json.dumps({"state": state}, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer %s" % self.token
        request = urllib.request.Request(self.url, data=payload, headers=headers,
                                         method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
            raw_orders = body.get("orders") if isinstance(body, dict) else None
            if not isinstance(raw_orders, list):
                self.last_error = "malformed response"
                return []
            orders = []
            for raw in raw_orders[:128]:
                if not isinstance(raw, dict) or "uid" not in raw:
                    continue
                posture = str(raw.get("posture", "hold"))
                if posture not in POSTURES:
                    posture = "hold"
                orders.append(CommanderOrder(
                    uid=int(raw["uid"]), posture=posture,
                    target_uid=(int(raw["target_uid"]) if raw.get("target_uid") is not None else None),
                    lon=(float(raw["lon"]) if raw.get("lon") is not None else None),
                    lat=(float(raw["lat"]) if raw.get("lat") is not None else None),
                    expires_s=float(raw.get("expires_s", 10.0) or 10.0),
                    source="remote"))
            self.last_error = None
            return orders
        except (urllib.error.URLError, TimeoutError, ValueError, TypeError,
                KeyError, OverflowError, OSError) as exc:
            self.last_error = str(exc)
            return []


class CommanderRegistry(object):
    """Holds the active model and its last successful orders."""

    def __init__(self, model: Optional[CommanderModel] = None,
                 interval_s: float = 2.0):
        self.model = model or RuleCommander()
        self.interval_s = float(interval_s)
        self.orders: Dict[int, tuple] = {}
        self.last_error: Optional[str] = None
        self.last_success_s: Optional[float] = None

    def decide(self, state: dict, now_s: float) -> List[CommanderOrder]:
        orders = []
        try:
            orders = self.model.decide(state) or []
            model_error = getattr(self.model, "last_error", None)
            if model_error and not orders:
                raise RuntimeError(str(model_error))
            self.last_success_s = now_s
            self.last_error = None
        except Exception as exc:  # fallback must never stop the battle
            self.last_error = str(exc)
            if isinstance(self.model, RuleCommander):
                return []
            orders = RuleCommander().decide(state)
        result = []
        for order in orders:
            try:
                uid = int(order.uid)
                expires_s = max(1.0, float(order.expires_s))
            except (TypeError, ValueError):
                continue
            self.orders[uid] = (order, now_s + expires_s)
            result.append(order)
        return result

    def active(self, uid: int, now_s: float) -> Optional[CommanderOrder]:
        entry = self.orders.get(int(uid))
        if entry is None:
            return None
        order, expires = entry
        if now_s > expires:
            self.orders.pop(int(uid), None)
            return None
        return order
