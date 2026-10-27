"""Case-driven three-stage intelligence pipeline.

Stage 1 collects broadly inside the case AOI/time window without keyword
pruning.  Stage 2 removes invalid/duplicate/out-of-scope records and scores
relevance without treating missing keywords as irrelevance.  Stage 3 links
cross-domain items and extracts key items.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .association import extract_key_items, link_items, term_hit
from .case import CaseIntelProfile
from .gateway import IntelBundle
from .model import Observation, iso, parse_time
from .sources import SourceStore
from .connectors.fetch import build_bundle

_CASE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


@dataclass(frozen=True)
class PipelineResult:
    candidates: IntelBundle
    bundle: IntelBundle
    report: dict
    statuses: tuple
    errors: tuple
    warnings: tuple


def _case_key(value: str) -> str:
    text = str(value or "")
    if not _CASE.match(text) or ".." in text:
        raise ValueError("invalid case_id")
    return text


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1,
                  sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _text(obs: Observation) -> str:
    return " ".join([
        str(obs.title or ""), str(obs.source_url or ""),
        " ".join(str(v) for v in (obs.properties or {}).values()),
    ]).lower()


def _relevance(obs: Observation, profile: CaseIntelProfile) -> Tuple[str, float, List[str]]:
    text = _text(obs)
    hits = [term for term in profile.keywords if term_hit(term, text)]
    try:
        confidence = float(obs.confidence or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    domain_focus = 1.0 if (not profile.domains
                           or obs.domain in profile.domains) else 0.4
    score = (0.35 * min(1.0, len(hits) / 2.0)
             + 0.30 * confidence + 0.20 * domain_focus
             + 0.15 * (1.0 if obs.geo_precision == "exact" else 0.6))
    reasons = []
    if hits:
        reasons.append("命中案例关键词：" + "、".join(hits[:3]))
    if domain_focus >= 1.0:
        reasons.append("领域与案例匹配：" + obs.domain)
    if confidence:
        reasons.append("来源置信度 %.2f" % confidence)
    if not hits:
        reasons.append("未命中关键词，保留为潜在线索")
    level = ("high" if hits or score >= 0.62 else
             "medium" if score >= 0.40 else "low")
    return level, round(score, 3), reasons


def _dedupe(observations: Sequence[Observation]) -> Tuple[List[Observation], int]:
    seen_ids = set()
    seen_titles = set()
    out = []
    dropped = 0
    for obs in observations:
        if obs.observation_id in seen_ids:
            dropped += 1
            continue
        title = re.sub(r"\s+", " ", str(obs.title or "").strip().lower())
        title_key = (title, str(obs.event_time or "")[:13])
        if title and title_key in seen_titles:
            dropped += 1
            continue
        seen_ids.add(obs.observation_id)
        if title:
            seen_titles.add(title_key)
        out.append(obs)
    return out, dropped


def clean_candidates(candidates: IntelBundle, profile: CaseIntelProfile,
                     start: Any, end: Any,
                     sources: Sequence[str]) -> Tuple[IntelBundle, dict]:
    start_dt = parse_time(start)
    end_dt = parse_time(end)
    filter_profile = profile.with_overrides(sources=tuple(sources))
    valid = []
    outside = invalid = 0
    for obs in candidates.observations:
        try:
            event = obs.event_dt()
        except (TypeError, ValueError):
            invalid += 1
            continue
        if not (start_dt <= event <= end_dt):
            outside += 1
            continue
        if not filter_profile.matches(obs, use_text=False):
            outside += 1
            continue
        valid.append(obs)
    deduped, duplicates = _dedupe(valid)
    cleaned = []
    levels = {"high": 0, "medium": 0, "low": 0}
    no_keyword = 0
    for obs in deduped:
        level, score, reasons = _relevance(obs, filter_profile)
        levels[level] += 1
        if not any(term_hit(term, _text(obs)) for term in filter_profile.keywords):
            no_keyword += 1
        properties = dict(obs.properties or {})
        properties.update({
            "relevance": level,
            "relevance_score": score,
            "relevance_reasons": reasons,
            "candidate_stage": "cleaned",
        })
        cleaned.append(dataclasses.replace(obs, properties=properties))
    cleaned.sort(key=lambda obs: (
        {"high": 0, "medium": 1, "low": 2}.get(
            (obs.properties or {}).get("relevance"), 3),
        -obs.event_ms(), obs.observation_id))
    bundle = IntelBundle(
        bundle_id="clean-" + (candidates.bundle_id or "bundle"),
        scenario_id=candidates.scenario_id,
        observations=cleaned,
        scenario_start=candidates.scenario_start,
        synthetic=candidates.synthetic,
        domain=candidates.domain,
        bbox=candidates.bbox,
        time_window=candidates.time_window,
        source_allowlist=candidates.source_allowlist,
    )
    report = {
        "candidates": len(candidates.observations),
        "kept": len(cleaned),
        "dropped_out_of_scope": outside,
        "dropped_duplicates": duplicates,
        "dropped_invalid": invalid,
        "kept_without_keyword": no_keyword,
        "relevance": levels,
        "stage": "cleaned",
        "cleaned_at": _now(),
    }
    return bundle, report


def _item(obs: Observation, source_name: str) -> dict:
    return {
        "id": obs.observation_id,
        "priority": "auto",
        "source_id": obs.source_id,
        "source_name": source_name,
        "domain": obs.domain,
        "type": obs.type,
        "event_time": obs.event_time,
        "lat": obs.lat,
        "lon": obs.lon,
        "title": obs.title,
        "content": str((obs.properties or {}).get("content") or obs.title or ""),
        "effective": True,
        "status": obs.status,
        "confidence": obs.confidence,
        "verification": "automatic",
        "record": obs.as_dict(),
    }


def analyze(bundle: IntelBundle, profile: CaseIntelProfile,
            source_names: Optional[Dict[str, str]] = None) -> dict:
    names = source_names or {}
    items = [_item(obs, names.get(obs.source_id, obs.source_id))
             for obs in bundle.observations]
    links = link_items(items, profile, radius_km=50.0, window_h=24.0)
    key_items = extract_key_items(items, links, profile)
    return {
        "links": links,
        "key_items": key_items,
        "counts": {"items": len(items), "links": len(links),
                   "key_items": len(key_items)},
        "analyzed_at": _now(),
    }


def run_case(case_id: str, profile: CaseIntelProfile, start: Any, end: Any,
             sources: Sequence[str], root: str,
             custom_sources: Optional[Sequence[dict]] = None,
             archive_root: Optional[str] = None,
             fetcher=None) -> PipelineResult:
    """Collect broadly, clean, analyze, and persist all three artifacts."""
    case_id = _case_key(case_id)
    result = build_bundle(
        source_ids=tuple(sources),
        bbox=profile.bbox,
        start=start,
        end=end,
        scenario_id=case_id,
        case_profile=profile,
        custom_sources=custom_sources,
        archive_root=archive_root,
        fetcher=fetcher,
        domain="multi",
        filter_observations=False,
    )
    candidates = result.bundle
    cleaned, clean_report = clean_candidates(candidates, profile, start, end,
                                             sources)
    source_names = {}
    try:
        for entry in SourceStore(os.path.join(root, "sources.json")).list():
            source_names[entry.get("source_id")] = entry.get("name")
    except Exception:  # presentation metadata must not break analysis
        source_names = {}
    source_names.update({
        entry.get("source_id"): entry.get("name")
        for entry in (custom_sources or []) if isinstance(entry, dict)})
    analysis = analyze(cleaned, profile, source_names)
    report = {
        "case_id": case_id,
        "generated_at": _now(),
        "stage1": {
            "sources": list(sources),
            "observations": len(candidates.observations),
            "statuses": [status.as_dict() for status in result.statuses],
            "errors": list(result.errors),
            "warnings": list(result.warnings),
            "bundle_hash": candidates.hash,
        },
        "stage2": clean_report,
        "stage3": analysis,
        "candidates_bundle": candidates.bundle_id,
        "cleaned_bundle": cleaned.bundle_id,
        "truncated": result.truncated,
    }
    directory = os.path.join(root, case_id)
    _write_json(os.path.join(directory, "candidates.json"),
                candidates.as_dict())
    existing = None
    try:
        with open(os.path.join(directory, "bundle.json"), "r",
                  encoding="utf-8") as handle:
            value = json.load(handle)
        existing = value if isinstance(value, dict) else None
    except (OSError, ValueError):
        existing = None
    if (not cleaned.observations and existing
            and existing.get("observations")):
        # A transient upstream outage must not erase the last useful bundle.
        report["preserved_bundle"] = True
        _write_json(os.path.join(directory, "last_attempt.json"), report)
        return PipelineResult(
            candidates=candidates,
            bundle=cleaned,
            report=report,
            statuses=result.statuses,
            errors=result.errors,
            warnings=result.warnings,
        )
    _write_json(os.path.join(directory, "bundle.json"), cleaned.as_dict())
    _write_json(os.path.join(directory, "pipeline.json"), report)
    return PipelineResult(
        candidates=candidates,
        bundle=cleaned,
        report=report,
        statuses=result.statuses,
        errors=result.errors,
        warnings=result.warnings,
    )


def custom_rss_sources(root: str) -> List[dict]:
    """Enabled user RSS/social sources from var/intel/sources.json."""
    path = os.path.join(root, "sources.json")
    if not os.path.isfile(path):
        return []
    try:
        store = SourceStore(path)
        return [entry for entry in store.list(enabled=True)
                if entry.get("kind") in ("rss", "social", "selfmedia")
                and entry.get("url")]
    except (OSError, ValueError):
        return []
