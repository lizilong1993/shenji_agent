"""数据质量评估：把"这条情报有多可信"变成可计算的枚举。

三条与项目现有约定一致的原则：

1. **不给单一分数**。质量由三件正交的事决定——来源可靠性、定位精度、
   时效——合成一个数会让人无法判断"到底哪里不行"。
2. **过期要能被看见**。半衰期复用 ``trust.py`` 的那张表，避免两处各写一套。
3. **缺省不等于差**。没有提供位置误差的来源（例如精确台站坐标的 USGS）
   按精确处理；只有明确声明了"质心/admin"的才降级。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .trust import GEO_FACTOR, HALF_LIFE_H, SOURCE_RELIABILITY

# measured   —— 传感器直接测得
# interpolated —— 由观测插值（例如相邻时刻线性内插的轨迹点）
# predicted  —— 模型外推
# stale      —— 数据本身没错，但已经超过该领域半衰期
QUALITY_LEVELS = ("measured", "interpolated", "predicted", "stale")


def _parse(value: Any) -> Optional[datetime]:
    try:
        text = str(value or "").strip()
        if not text:
            return None
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def age_hours(event_time: Any, now: Optional[datetime] = None) -> Optional[float]:
    stamp = _parse(event_time)
    if stamp is None:
        return None
    current = now or datetime.now(timezone.utc)
    return max(0.0, (current - stamp.astimezone(timezone.utc)).total_seconds()
               / 3600.0)


def quality_of(record: Dict[str, Any],
               now: Optional[datetime] = None) -> Dict[str, Any]:
    """返回 ``{level, reasons, provenance, age_h, half_life_h}``。

    ``level`` 是四选一的枚举；判据按"最差的那一项"决定，因为一条情报
    只要有一环不达标，整体就不能当"实测"用。
    """
    source_id = str(record.get("source_id") or "unknown")
    domain = str(record.get("domain") or "other")
    geo_precision = str(record.get("geo_precision") or "exact")
    is_centroid = bool(record.get("is_centroid"))
    half_life = HALF_LIFE_H.get(domain, HALF_LIFE_H["other"])
    age = age_hours(record.get("event_time"), now)

    reasons = []
    # 1) 时效：超半衰期直接判 stale，其余判据不再有意义
    if age is not None and age > half_life:
        reasons.append("数据已 %d 小时，超过该领域半衰期 %d 小时"
                       % (round(age), round(half_life)))
        level = "stale"
    else:
        # 2) 生成方式：记录里显式声明的优先（仿真/外推产物）
        declared = str(record.get("quality") or "").strip().lower()
        if declared in ("interpolated", "predicted"):
            level = declared
            reasons.append("记录声明为 " + declared)
        elif is_centroid or geo_precision in ("admin0", "admin1", "unknown"):
            # 定位只到行政区一级：这类记录本来就是"大致位置"，不能当实测
            level = "interpolated"
            reasons.append("定位精度 %s%s"
                           % (geo_precision, "（质心）" if is_centroid else ""))
        else:
            level = "measured"

    if source_id in SOURCE_RELIABILITY:
        reasons.append("来源可靠性 %.2f" % SOURCE_RELIABILITY[source_id])
    else:
        reasons.append("来源未登记（按 %.2f 计）" % SOURCE_RELIABILITY.get(
            source_id, 0.7))
    if geo_precision in GEO_FACTOR:
        reasons.append("定位系数 %.2f" % GEO_FACTOR[geo_precision])

    return {
        "level": level,
        "reasons": reasons,
        "age_h": None if age is None else round(age, 1),
        "half_life_h": half_life,
        "provenance": {
            "source_id": source_id,
            "geo_precision": geo_precision,
            "is_centroid": is_centroid,
            "ingest_time": str(record.get("ingest_time") or ""),
        },
    }
