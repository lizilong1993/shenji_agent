"""Unified automatic + manual intelligence overview and conflict resolution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .association import (canonical_type, extract_key_items, link_items)
from .convergence import convergence_events, summarize as convergence_summary
from .manual import ManualIntelStore
from .correlations import CorrelationStore
from .lifecycle import LifecycleStore, STAGES as LIFECYCLE_STAGES, \
    STAGE_LABELS as LIFECYCLE_LABELS, TRANSITIONS as LIFECYCLE_TRANSITIONS
from .quality import quality_of
from .sources import SourceStore
from .case import load_case_profile

CONFLICT_RADIUS_KM = 25.0
CONFLICT_WINDOW_H = 6.0
CONVERGENCE_WINDOW_H = 24.0
CONVERGENCE_MIN_TYPES = 3
_CASE_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_CASE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


class InvalidCaseId(ValueError):
    pass


def _case_key(value: Any) -> str:
    text = str(value or "default")
    if not _CASE_RE.match(text) or ".." in text:
        raise InvalidCaseId("invalid case_id")
    return text


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _time(value: Any) -> Optional[datetime]:
    try:
        text = str(value or "").strip()
        if not text:
            return None
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    lon1r, lat1r, lon2r, lat2r = map(
        math.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    h = (math.sin(dlat / 2.0) ** 2
         + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2.0) ** 2)
    return 2.0 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))


def _lat_lon(record: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    lat, lon = record.get("lat"), record.get("lon")
    geometry = record.get("geometry") or {}
    coords = geometry.get("coordinates") or []
    if lat is None and len(coords) > 1:
        lat = coords[1]
    if lon is None and len(coords) > 0:
        lon = coords[0]
    try:
        return round(float(lat), 6), round(float(lon), 6)
    except (TypeError, ValueError):
        return None, None


def _uncertainty(record: Dict[str, Any]) -> float:
    """位置误差半径（米）。缺值按 0 处理，不用 None：前端算 ±X km 时
    None 会变成 NaN，而"没写"和"精确定位"在展示上是两回事。"""
    try:
        value = float(record.get("geo_uncertainty_m") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if value >= 0.0 and value == value else 0.0


def _auto_item(record: Dict[str, Any], source_names: Dict[str, str]) -> Optional[dict]:
    lat, lon = _lat_lon(record)
    if lat is None or lon is None:
        return None
    source_id = str(record.get("source_id") or "unknown")
    geometry = record.get("geometry") or {}
    return {
        "id": str(record.get("observation_id") or ""),
        "priority": "auto",
        "source_id": source_id,
        "source_name": source_names.get(source_id, source_id),
        "domain": str(record.get("domain") or "other"),
        "type": str(record.get("type") or "other"),
        "event_time": str(record.get("event_time") or ""),
        "ingest_time": str(record.get("ingest_time") or ""),
        "lat": lat,
        "lon": lon,
        # 定位不确定性与"质心"标记必须一起透出：前端只有拿到半径才能把点
        # 画成置信区，否则一个 admin0 质心和一个实测点看起来一模一样。
        "geo_uncertainty_m": _uncertainty(record),
        "is_centroid": bool(record.get("is_centroid")),
        "place": str((record.get("properties") or {}).get("place") or ""),
        "title": str(record.get("title") or record.get("type") or "自动情报"),
        # 自动情报的 content 以前直接复制 title，卡片上就会出现"标题说一遍、
        # 正文再说一遍"。真正有内容的是源记录里的 description/正文。
        "content": str((record.get("properties") or {}).get("description")
                       or record.get("description") or ""),
        "verification": "automatic",
        "confidence": record.get("confidence"),
        "status": str(record.get("status") or "active"),
        "effective": True,
        "superseded_by": None,
        "conflicts": [],
        "license": record.get("license") or "",
        "attribution": record.get("attribution") or "",
        # 把源特有字段整理成 [标签, 值] 对。前端不用知道每个源有什么字段，
        # 也能把"气温 22°C / 风速 14 km/h"这类关键读数显示出来——否则卡片上
        # 只剩标题和坐标，读者没法判断这条情报到底说了什么。
        "facts": _facts(record),
        "quality": quality_of(record),
        "geometry": geometry,
        "record": record,
    }


# 各源的关键字段 → 中文标签与格式化。只列**读者要用来判断**的字段，
# 不把 properties 整个倒出来——那是原始记录，不是情报。
_FACT_SPECS = {
    "open-meteo": [
        ("temperature_c", "气温", "%s°C"),
        ("wind_speed_kmh", "风速", "%s km/h"),
        ("precipitation_mm", "降水", "%s mm"),
        ("relative_humidity_pct", "湿度", "%s%"),
        ("visibility_m", "能见度", "%s m"),
        ("weather_code", "天气", "%s"),
    ],
    "adsb": [
        ("aircraft_type", "机型", "%s"),
        ("altitude_baro_ft", "气压高度", "%s ft"),
        ("ground_speed_kt", "地速", "%s kt"),
        ("military", "军用", "%s"),
    ],
    "usgs": [
        ("magnitude", "震级", "M%s"),
        ("depth_km", "深度", "%s km"),
        ("place", "位置", "%s"),
    ],
    "firms": [
        ("brightness", "亮度", "%s K"),
        ("frp", "辐射功率", "%s MW"),
        ("confidence", "可信度", "%s"),
    ],
    "gdacs": [
        ("alert_level", "警报级别", "%s"),
        ("event_type", "事件类型", "%s"),
        ("country", "国家", "%s"),
    ],
    "gdelt": [
        ("source_language", "语种", "%s"),
        ("mentioned_count", "提及数", "%s"),
    ],
}

# weather_code 与 military 要转成人能读的说法，不能直接显示数字/布尔。
_VALUE_FORMS = {
    "open-meteo": {"weather_code": lambda v: _weather(v)},
    "adsb": {"military": lambda v: "是" if v else "否"},
}


def _weather(code):
    from .weather_text import describe
    return describe(code) or str(code)


def _facts(record: Dict[str, Any]) -> List[dict]:
    source_id = str(record.get("source_id") or "")
    props = record.get("properties") or {}
    forms = _VALUE_FORMS.get(source_id, {})
    out = []
    for key, label, pattern in _FACT_SPECS.get(source_id, ()):
        value = props.get(key)
        if value is None or value == "":
            continue
        if key in forms:
            try:
                out.append({"label": label, "value": forms[key](value)})
            except Exception:      # noqa: BLE001 - 格式化失败就跳过这一项
                pass
            continue
        try:
            text = pattern % value
        except (TypeError, ValueError):
            text = str(value)
        out.append({"label": label, "value": text})
    return out


def _apply_verdicts(links, verdicts):
    """把人工结论盖在算法链接上。

    confirm → 保留链接，但把 kind 标成 ``manual``、分数顶到 0.99，并附上
              分析员的原话，这样"这条是谁确认的"在界面上可追溯；
    reject  → 整条链接移除。否决的意义就是"别把它当关联看"，留下来只会
              让人反复看到一条已知错误的提示。
    """
    if not verdicts:
        return links
    out = []
    for link in links:
        key = "|".join(sorted([str(link.get("from_id")),
                               str(link.get("to_id"))]))
        verdict = verdicts.get(key)
        if verdict is None:
            out.append(link)
            continue
        if verdict.get("verdict") == "reject":
            continue
        merged = dict(link)
        merged["kind"] = "manual"
        merged["score"] = 0.99
        reasons = list(link.get("reasons") or [])
        note = str(verdict.get("note") or "").strip()
        reasons.append("人工确认" + ("：" + note if note else ""))
        merged["reasons"] = reasons
        merged["correlation_id"] = verdict.get("correlation_id")
        merged["reviewed_at"] = verdict.get("updated_at")
        out.append(merged)
    return out


def _latest_per_snapshot_site(items: List[dict]) -> List[dict]:
    """同一地点的快照型情报（气象、ADS-B 位置）只保留最新一条。

    这些源每次采集都会生成一条新记录（id 带时间戳），所以同一个观测点上
    十分钟内的三次采集会在列表里并列出现，看起来像"三条一样的天气"。
    历史价值在缓存里仍然保留（可用于趋势），列表层只展示当前状态。

    只对**快照型**源生效。事件型源（地震、火点、新闻）同一地点多次发生是
    真实的多起事件，合并掉就是丢情报。
    """
    SNAPSHOT = ("open-meteo",)
    best = {}
    out = []
    for item in items:
        source_id = str(item.get("source_id") or "")
        if source_id not in SNAPSHOT:
            out.append(item)
            continue
        key = (source_id, round(float(item.get("lat") or 0.0), 4),
               round(float(item.get("lon") or 0.0), 4), str(item.get("type") or ""))
        current = best.get(key)
        if current is None:
            best[key] = item
            continue
        # 时间戳格式统一为 ISO，字符串比较即时间比较
        if str(item.get("event_time") or "") > str(current.get("event_time") or ""):
            best[key] = item
    # 保持原有排序语义：把选中的最新条目放回原位置
    chosen = {id(row) for row in best.values()}
    out = []
    emitted = set()
    for item in items:
        source_id = str(item.get("source_id") or "")
        if source_id not in SNAPSHOT:
            out.append(item)
            continue
        key = (source_id, round(float(item.get("lat") or 0.0), 4),
               round(float(item.get("lon") or 0.0), 4), str(item.get("type") or ""))
        winner = best.get(key)
        if winner is None or key in emitted:
            continue
        if id(winner) == id(item):
            emitted.add(key)
            out.append(item)
    return out


def _manual_item(record: Dict[str, Any]) -> dict:
    return {
        "id": str(record.get("intel_id") or ""),
        "priority": "manual",
        "source_id": str(record.get("source_id") or "manual"),
        "source_name": str(record.get("source_name") or "人工录入"),
        "domain": str(record.get("domain") or "other"),
        "type": str(record.get("intel_type") or record.get("type") or "other"),
        "event_time": str(record.get("event_time") or ""),
        # 人工录入没有"入库时间"这个概念以外的来源信息，但字段必须齐：
        # 前端按固定字段渲染，缺一个就是"这条永远显示无时效"。
        "ingest_time": str(record.get("created_at") or ""),
        "lat": record.get("lat"),
        "lon": record.get("lon"),
        "geo_uncertainty_m": 0.0,
        "is_centroid": False,
        "place": str(record.get("place") or ""),
        "title": str(record.get("title") or "人工情报"),
        "content": str(record.get("content") or ""),
        "verification": str(record.get("verification") or "unverified"),
        "confidence": None,
        # NSL：命中即不得提名打击。字段透传到前端，让每个渲染自定义
        # 条目的地方都能看到"这条为什么不能打"，而不只是悄悄过滤掉。
        "no_strike": bool(record.get("no_strike")),
        "no_strike_reason": str(record.get("no_strike_reason") or ""),
        "status": str(record.get("status") or "active"),
        "effective": str(record.get("status") or "active") == "active",
        "superseded_by": None,
        "conflicts": [],
        "license": "",
        "attribution": "人工录入",
        "geometry": {"type": "Point",
                     "coordinates": [record.get("lon"), record.get("lat")]},
        "revision": record.get("revision"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "record": record,
    }


class IntelManager:
    def __init__(self, root: str, profile_root: Optional[str] = None):
        self.root = root
        self.profile_root = profile_root
        self.manual = ManualIntelStore(os.path.join(root, "manual.json"))
        self.sources = SourceStore(os.path.join(root, "sources.json"))
        self.correlations = CorrelationStore(
            os.path.join(root, "correlations.json"))
        self.lifecycle = LifecycleStore(os.path.join(root, "lifecycle.json"))
        # 基线是进程内状态（不落盘）：它描述的是"运行期间见过的分布"，
        # 重启后从零开始累积，避免把上一轮任务的历史混进来。
        self._baseline_tracker = None
        self._cache = {}

    def _bundle(self, case_id: str) -> Optional[dict]:
        safe = _CASE_SAFE.sub("_", case_id or "default") or "default"
        path = os.path.join(self.root, safe, "bundle.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def _pipeline(self, case_id: str) -> Optional[dict]:
        safe = _CASE_SAFE.sub("_", case_id or "default") or "default"
        path = os.path.join(self.root, safe, "pipeline.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError):
            return None

    def _ai(self, case_id: str):
        from .ai import AISuggestionStore
        safe = _CASE_SAFE.sub("_", case_id or "default") or "default"
        return AISuggestionStore(os.path.join(self.root, safe, "ai.json"))

    def _profile(self, case_id: str):
        try:
            return load_case_profile(case_id, root=self.profile_root)
        except (OSError, ValueError):
            return None

    def _baseline_block(self, items, profile):
        """按网格统计情报条数，和历史基线比，列出偏离的网格。

        为什么按网格而不按"总量"：总量受案例大小与采集节奏影响，涨落几乎是
        采集行为的镜像；网格级才反映"某片区域忽然热起来了"。
        """
        from .baseline import BaselineTracker
        from .convergence import cell_of, cell_size_for
        size = cell_size_for(list(profile.expanded_bbox) if profile
                             else (119.0, 22.0, 123.0, 26.0))
        counts = {}
        for item in items:
            try:
                lat = float(item.get("lat"))
                lon = float(item.get("lon"))
            except (TypeError, ValueError):
                continue
            key = "%d,%d" % cell_of(lat, lon, size)
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return {"cells": 0, "anomalies": [], "note": "没有可用坐标"}
        tracker = self._baseline_tracker
        if tracker is None:
            tracker = self._baseline_tracker = BaselineTracker()
        found = []
        # 顺序固定：同一次观察必须可复现（dict 顺序在 Py3.7+ 是插入序，
        # 但插入序取决于 items 顺序，这里再显式排序一次）。
        for key in sorted(counts):
            anomaly = tracker.observe(key, counts[key])
            if anomaly is not None:
                found.append(anomaly.as_dict())
        return {
            "cells": len(counts),
            "sample_size": tracker.min_samples,
            "threshold": tracker.threshold,
            "anomalies": found,
            "baseline": tracker.snapshot(),
        }

    def overview(self, case_id: str) -> dict:
        case_id = _case_key(case_id)
        profile = self._profile(case_id)
        sources = self.sources.list()
        cache_stats = None
        try:
            from .cache import IntelCache, cache_db_path
            db_path = cache_db_path(self.root)
            if os.path.isfile(db_path):
                cache = IntelCache(db_path, readonly=True)
                try:
                    cache_stats = cache.stats()
                finally:
                    cache.close()
        except Exception:  # advisory only; overview must never fail on cache IO
            cache_stats = None
        source_names = {entry.get("source_id"): entry.get("name")
                        for entry in sources}
        bundle = self._bundle(case_id)
        pipeline = self._pipeline(case_id)
        try:
            ai_store = self._ai(case_id)
            ai_counts = ai_store.counts()
            ai_items = ai_store.list()[:20]
        except (OSError, ValueError):
            ai_counts = {"total": 0, "suggested": 0, "accepted": 0,
                         "rejected": 0, "needs_review": 0}
            ai_items = []
        warnings = []
        if bundle is None:
            warnings.append("案例 %s 暂无自动情报 bundle（var/intel/%s/bundle.json）"
                            % (case_id, _CASE_SAFE.sub("_", case_id)))
            auto_records = []
        else:
            bundle_case = str(bundle.get("scenario_id") or "")
            if bundle_case and bundle_case != case_id:
                warnings.append("bundle scenario_id=%s 与案例 %s 不一致，已忽略"
                                % (bundle_case, case_id))
                auto_records = []
            else:
                auto_records = [record for record in bundle.get("observations") or []
                                if isinstance(record, dict)
                                and record.get("status", "active") == "active"]
        auto_items = [item for item in (_auto_item(record, source_names)
                                        for record in auto_records)
                      if item is not None]
        manual_records = self.manual.list(case_id, include_revoked=True)
        manual_items = [_manual_item(record) for record in manual_records]
        active_manual = [item for item in manual_items
                         if item.get("status") == "active"]
        auto_by_id = {item["id"]: item for item in auto_items}
        manual_by_id = {item["id"]: item for item in manual_items}
        superseded = {}
        for manual in sorted(active_manual,
                             key=lambda item: str(item.get("updated_at") or "")):
            manual_time = _time(manual.get("event_time"))
            for auto in auto_items:
                if manual.get("domain") != auto.get("domain"):
                    continue
                manual_type = canonical_type(manual.get("type"))
                auto_type = canonical_type(auto.get("type"))
                if manual_type != auto_type:
                    continue
                auto_time = _time(auto.get("event_time"))
                if manual_time is None or auto_time is None:
                    continue
                if abs((manual_time - auto_time).total_seconds()) / 3600.0 > CONFLICT_WINDOW_H:
                    continue
                try:
                    distance = _haversine_km(
                        float(manual["lat"]), float(manual["lon"]),
                        float(auto["lat"]), float(auto["lon"]))
                except (TypeError, ValueError):
                    continue
                if distance > CONFLICT_RADIUS_KM:
                    continue
                manual["conflicts"].append(auto["id"])
                current = superseded.get(auto["id"])
                if current is None or str(manual.get("updated_at") or "") >= str(
                        manual_by_id.get(current, {}).get("updated_at") or ""):
                    superseded[auto["id"]] = manual["id"]
        for auto_id, manual_id in superseded.items():
            auto = auto_by_id.get(auto_id)
            if auto is not None:
                auto["effective"] = False
                auto["superseded_by"] = manual_id
        for manual in manual_items:
            manual["effective"] = manual.get("status") == "active"
        items = manual_items + auto_items
        items.sort(key=lambda item: (
            0 if item.get("priority") == "manual"
            and item.get("status") == "active" else
            1 if item.get("priority") == "auto" and item.get("effective") else
            2 if item.get("priority") == "manual" else 3,
            -(_time(item.get("event_time")) or datetime(1970, 1, 1,
                                                         tzinfo=timezone.utc)).timestamp(),
            str(item.get("id")),
        ))
        # 排序完成后再合并快照型重复项，保证"最新一条"按时间戳选出。
        # 放在排序之后而不是之前，是因为合并逻辑依赖原顺序把赢家放回原位。
        items = _latest_per_snapshot_site(items)
        revision = hashlib.sha256("|".join([
            self.manual.revision(),
            self.sources.revision(),
            self.correlations.revision(),
            str((bundle or {}).get("hash") or ""),
            str((profile.to_dict() if profile else "")),
        ]).encode("utf-8")).hexdigest()[:16]
        cache_key = (case_id, revision,
                     datetime.now(timezone.utc).strftime("%Y%m%d%H"))
        cached = self._cache.get(cache_key)
        if cached is None:
            links = link_items(items, profile,
                               radius_km=max(CONFLICT_RADIUS_KM, 50.0),
                               window_h=max(CONFLICT_WINDOW_H, 24.0))
            links = _apply_verdicts(links, self.correlations.verdicts(case_id))
            key_items = extract_key_items(items, links, profile)
            events = convergence_events(
                items,
                bbox=list(profile.expanded_bbox) if profile else None,
                window_h=CONVERGENCE_WINDOW_H,
                min_types=CONVERGENCE_MIN_TYPES,
            )
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[cache_key] = (links, key_items, events)
        else:
            links, key_items, events = cached
        counts = {
            "manual_active": len(active_manual),
            "manual_revoked": len(manual_items) - len(active_manual),
            "no_strike": sum(1 for item in items
                             if item.get("no_strike")
                             and item.get("status") == "active"),
            "auto": len(auto_items),
            "effective": sum(1 for item in items if item.get("effective")),
            "superseded": len(superseded),
            "links": len(links),
            "key_items": len(key_items),
            "convergence": len(events),
        }
        return {
            "case_id": case_id,
            "generated_at": _now(),
            "revision": revision,
            "profile": profile.to_dict() if profile else None,
            "pipeline": pipeline,
            "conflict_rule": {
                "radius_km": CONFLICT_RADIUS_KM,
                "window_h": CONFLICT_WINDOW_H,
                "rule": "同领域、同规范类型、时空邻近时人工情报覆盖自动情报",
            },
            "counts": counts,
            "items": items,
            "links": links,
            # 人工关联结论单列，界面可以只显示"谁确认/否决了哪一对"，
            # 不必从 links 里反推哪些是算法给的、哪些是人改过的。
            "correlations": {
                "rule": "人工确认的关联提升为 manual 并按 0.99 计分；否决的直接移除",
                "items": self.correlations.list(case_id),
            },
            "key_items": key_items,
            "convergence": {
                "rule": {
                    "window_h": CONVERGENCE_WINDOW_H,
                    "min_types": CONVERGENCE_MIN_TYPES,
                    "cell": "bbox/16 自适应，最小 0.05°",
                    "rule": "同一网格内 24h 出现 3 类以上不同规范类型即生成汇聚事件",
                },
                "events": events,
                "summary": convergence_summary(events),
            },
            # 基线异常：只输出"哪个网格的情报量偏离了平时"，不判定是否告警。
            # 判定交给调用方——同一组数字在不同任务下的含义完全不同。
            "baseline": self._baseline_block(items, profile),
            "sources": sources,
            # 目标生命周期（五段式）。只读输出：写操作走 /api/intel/lifecycle。
            "lifecycle": {
                "stages": [{"key": stage, "label": LIFECYCLE_LABELS[stage]}
                           for stage in LIFECYCLE_STAGES],
                "transitions": {stage: list(LIFECYCLE_TRANSITIONS[stage])
                                for stage in LIFECYCLE_STAGES},
                "board": self.lifecycle.board(),
            },
            "cache": cache_stats,
            "ai": {"counts": ai_counts, "items": ai_items},
            "warnings": warnings,
        }
