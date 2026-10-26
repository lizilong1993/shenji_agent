"""Shared domain/type vocabulary for automatic and manual intelligence."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

DOMAINS = (
    "hazard", "conflict", "maritime", "aviation", "infrastructure",
    "weather", "news", "geography", "society", "social", "selfmedia",
    "cyber", "health", "other",
)

DOMAIN_LABELS = {
    "hazard": "灾害",
    "conflict": "冲突",
    "maritime": "海上",
    "aviation": "空中",
    "infrastructure": "基础设施",
    "weather": "气象",
    "news": "新闻",
    "geography": "地理",
    "society": "人文社会",
    "social": "社交媒体",
    "selfmedia": "自媒体",
    "cyber": "网络",
    "health": "卫生",
    "other": "其他",
}

INTEL_TYPES = (
    "target", "aircraft", "vessel", "ground", "earthquake", "fire",
    "cyclone", "flood", "weather", "news", "geography", "society",
    "social", "selfmedia", "infrastructure", "threat", "other",
)

SOURCE_KINDS = ("auto", "manual", "rss", "weather", "web", "social", "other")

VERIFICATION_STATES = ("unverified", "verified", "disputed")


def normalize_domain(value: Optional[str]) -> str:
    text = str(value or "").strip().lower()
    return text if text in DOMAINS else "other"


def normalize_domains(values: Optional[Iterable[str]]) -> Tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = [values]
    out = []
    for value in values:
        domain = normalize_domain(value)
        if domain not in out:
            out.append(domain)
    return tuple(out)


def normalize_type(value: Optional[str]) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    return text or "other"


def normalize_source_kind(value: Optional[str]) -> str:
    text = str(value or "").strip().lower()
    return text if text in SOURCE_KINDS else "other"


def normalize_verification(value: Optional[str]) -> str:
    text = str(value or "").strip().lower()
    return text if text in VERIFICATION_STATES else "unverified"
