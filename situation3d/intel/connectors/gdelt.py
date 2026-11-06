"""GDELT DOC 2.0 news connector.

GDELT DOC returns articles without coordinates.  Observations are placed at
the case centroid with ``geo_precision="unknown"``; they are text leads, not
located events.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence
from urllib.parse import urlencode

from ..model import parse_time
from ..normalize import normalize_gdelt

API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT DOC 支持的 sourcelang 取值（按 3 字母代码）。扩展的目标是让非英语
# 地区的案例也能拿到本地报道：台海案例只看英文源会漏掉大量中文报道。
LANGUAGES = ("eng", "chi", "zho", "jpn", "kor", "rus", "spa", "ara",
             "fra", "deu", "hin", "por", "tur", "vie", "tha", "ind")


def _terms(keywords: Sequence[str]) -> list:
    return [str(term).strip() for term in keywords or [] if str(term).strip()]


def build_url(keywords: Sequence[str], start: Any, end: Any,
              maxrecords: int = 75,
              languages: Optional[Sequence[str]] = None,
              countries: Optional[Sequence[str]] = None,
              extra_terms: Optional[Sequence[str]] = None) -> str:
    """构造 GDELT DOC 查询。

    ``languages`` / ``countries`` 是可选的**收窄**条件（GDELT 的
    sourcelang: 与 sourcecountry: 运算符）。不传就完全等同于旧查询，
    保证默认行为不变。

    ``extra_terms`` 是额外的**收窄**关键词（同一主题下的限定词，例如
    "台海" 之外再加 "军事演习"）。它们用 AND 连接，与主关键词的 OR 分开：
    主关键词决定"哪些案例相关"，额外词决定"哪些更值得看"。
    """
    terms = _terms(keywords)
    if not terms:
        raise ValueError("GDELT needs at least one case keyword")
    query = " OR ".join('"%s"' % term.replace('"', "") for term in terms[:20])
    extra = _terms(extra_terms)[:10]
    if extra:
        query = "(" + query + ") " + " ".join(
            '("%s")' % term.replace('"', "") for term in extra)
    langs = [str(value).strip().lower() for value in (languages or ())
             if str(value).strip()]
    valid = [code for code in langs if code in LANGUAGES]
    if valid:
        query += " " + " OR ".join("sourcelang:%s" % code for code in valid[:5])
    places = [str(value).strip().upper() for value in (countries or ())
              if str(value).strip()]
    if places:
        query += " " + " OR ".join(
            "sourcecountry:%s" % code for code in places[:5])
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": max(1, min(int(maxrecords), 250)),
        "format": "json",
        "sort": "datedesc",
        "startdatetime": parse_time(start).strftime("%Y%m%d%H%M%S"),
        "enddatetime": parse_time(end).strftime("%Y%m%d%H%M%S"),
    }
    return API_URL + "?" + urlencode(params)


def parse_payload(payload: dict, fetched_at: Any, centroid,
                  start_seq: int = 0):
    return normalize_gdelt(payload, fetched_at, centroid,
                           start_seq=start_seq)
