"""One-click situation brief (TS简报) built from the inference output.

Why this exists
---------------
`inference.py` answers "what is the situation" as per-frame numbers: L1 per
target, L2 per grouping, L3 per region. Those numbers are correct but they are
*frames*, not a *narrative*. An operator handed the raw frames still has to
answer the questions that actually matter when briefing someone:

  * What was the worst moment, and when?
  * Which platform drove it?
  * Did the situation escalate, and where did it turn?
  * Which region is the problem, and who holds it?

Reading that off a few hundred frames by hand is exactly the work this module
does instead. It aggregates the existing rule output -- it does **not** invent a
new scoring model, so every sentence traces back to a number that `inference.py`
already produced and can be defended in review.

Design notes
------------
* **Standard library only.** The container has python3.8 with no pip, matching
  the rest of the engine.
* **Severity is derived from peak, not mean.** A run that is calm for 90% of the
  timeline and briefly spikes to 高危 is not a "关注" situation. Averaging would
  hide precisely the moment the brief exists to surface.
* **Timeline runs always include a start and an end**, so the reader can see the
  trend rather than a single snapshot. Isolated single-frame flips are collapsed
  (see MIN_RUN_FRAMES) -- otherwise sensor-boundary jitter produces dozens of
  meaningless one-frame runs and buries the real transitions.
* **`headline` is a single sentence** because that is what gets read aloud first.

Conventions match inference.py: distances km, threat 0-100, levels 1-4.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

# A level run shorter than this many frames is treated as jitter and merged into
# its neighbours. One frame at 30 s of sim time is well inside the noise band of
# a sensor-coverage boundary crossing.
MIN_RUN_FRAMES = 2

# How many targets to list per section. A brief is meant to be read, not scrolled.
TOP_N = 5


def _level_label(model: dict, level: int) -> str:
    codes = (model or {}).get("level_codes") or {}
    # JSON turns integer keys into strings.
    return codes.get(str(level)) or codes.get(level) or ("%d 级" % level)


def _fmt_clock(start_utc: Optional[str], t_seconds: Optional[float]) -> Optional[str]:
    """Absolute UTC clock for a frame, if the dataset carries a start time.

    A brief that says "第 120 帧" is much less useful than one that says
    "08:32:00Z" -- the reader can line it up against a real log.
    """
    if not start_utc or t_seconds is None:
        return None
    try:
        from datetime import datetime, timedelta, timezone
        base = datetime.strptime(start_utc.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
        base = base.replace(tzinfo=timezone.utc)
        return (base + timedelta(seconds=float(t_seconds))).strftime("%H:%M:%SZ")
    except (ValueError, TypeError):
        return None


def _collapse_runs(levels: Sequence[int]) -> List[dict]:
    """Turn a per-frame level series into runs, merging jitter.

    Two passes: build raw runs, then absorb runs shorter than MIN_RUN_FRAMES
    into the preceding run (or the following one when at the head).
    """
    if not levels:
        return []

    raw: List[dict] = []
    for i, lv in enumerate(levels):
        if raw and raw[-1]["level"] == lv:
            raw[-1]["end"] = i
        else:
            raw.append({"level": lv, "start": i, "end": i})

    merged: List[dict] = []
    for run in raw:
        if merged and (run["end"] - run["start"] + 1) < MIN_RUN_FRAMES:
            merged[-1]["end"] = run["end"]
        else:
            merged.append(dict(run))
    # A short head run has nothing before it to merge into; fold it forward.
    if len(merged) > 1 and (merged[0]["end"] - merged[0]["start"] + 1) < MIN_RUN_FRAMES:
        merged[1]["start"] = merged[0]["start"]
        merged.pop(0)
    return merged


def _describe_levels(model: dict, levels: Sequence[int], time_at) -> List[str]:
    """Human-readable summary of the level timeline.

    A 29-run alternation rendered as "A → B → A → B ..." is technically accurate
    and useless to read. So when the series is long, report what it *did* --
    where it started, where it peaked and when, how it ended, and how much time
    it spent at each level -- instead of the raw chain.

    Note this does **not** smooth the oscillation away: measured on joint_full the
    runs are 7-34 frames (3.5-17 min of sim time) and recur 29 times, i.e. real
    dynamics. Flattening that would be misreporting. It only changes how the same
    facts are worded.
    """
    if not levels:
        return ["无等级数据。"]

    runs = _collapse_runs(levels)
    total = len(levels)
    counts: Dict[int, int] = {}
    for lv in levels:
        counts[lv] = counts.get(lv, 0) + 1
    share = "、".join(
        "%s %.0f%%" % (_level_label(model, lv), 100.0 * n / total)
        for lv, n in sorted(counts.items())
    )
    peak = max(levels)
    peak_idx = levels.index(peak)
    clock = time_at(peak_idx)
    lines = [
        "起始 %s，结束 %s，峰值 %s（第 %d 帧%s）。" % (
            _level_label(model, levels[0]),
            _level_label(model, levels[-1]),
            _level_label(model, peak),
            peak_idx,
            ("，" + clock) if clock else "",
        ),
        "各等级占比：%s。" % share,
    ]
    if len(runs) <= 6:
        lines.append("等级序列：%s。" % " → ".join(
            _level_label(model, r["level"]) for r in runs))
    else:
        longest = max(runs, key=lambda r: r["end"] - r["start"])
        lines.append(
            "等级在 %d 段之间往复 %d 次，最长的一段为%s，持续 %d 帧（约 %.1f 小时）。" % (
                len(counts), len(runs) - 1,
                _level_label(model, longest["level"]),
                longest["end"] - longest["start"] + 1,
                (longest["end"] - longest["start"] + 1) * 30.0 / 3600.0,
            ))
    return lines


def _target_names(targets) -> Dict[str, dict]:
    out = {}
    for t in targets or ():
        out[getattr(t, "target_id", "")] = {
            "name": getattr(t, "name", ""),
            "affiliation": getattr(t, "affiliation", ""),
            "domain": getattr(t, "domain", ""),
            "platform_class": getattr(t, "platform_class", ""),
            "model": getattr(t, "model", ""),
            "role": getattr(t, "role", ""),
        }
    return out


def _analyse_targets(frames: Sequence[dict], names: Dict[str, dict],
                     start_utc: Optional[str], timestamps: Sequence[float]) -> List[dict]:
    """Per-target summary: peak threat, the frame it happened, and state changes."""
    stats: Dict[str, dict] = {}
    for fi, frame in enumerate(frames):
        for tid, l1 in (frame.get("targets") or {}).items():
            st = stats.get(tid)
            if st is None:
                st = stats[tid] = {
                    "id": tid,
                    "peak_threat": -1.0,
                    "peak_frame": 0,
                    "peak_clock": None,
                    "states": [],
                    "intents": [],
                    "min_nearest_km": None,
                    "max_covered": 0,
                    "alert_counts": {},
                    "first_threat": None,
                    "last_threat": None,
                }
            threat = float(l1.get("threat") or 0.0)
            if st["first_threat"] is None:
                st["first_threat"] = threat
            st["last_threat"] = threat
            if threat > st["peak_threat"]:
                st["peak_threat"] = threat
                st["peak_frame"] = fi
                st["peak_clock"] = _fmt_clock(start_utc, timestamps[fi])
            state = l1.get("state_label")
            if state and (not st["states"] or st["states"][-1] != state):
                st["states"].append(state)
            intent = l1.get("intent_label")
            if intent and (not st["intents"] or st["intents"][-1] != intent):
                st["intents"].append(intent)
            nk = l1.get("nearest_km")
            if nk is not None and (st["min_nearest_km"] is None or nk < st["min_nearest_km"]):
                st["min_nearest_km"] = nk
            covered = l1.get("covered_by") or 0
            if isinstance(covered, int) and covered > st["max_covered"]:
                st["max_covered"] = covered
            label = l1.get("alert_label") or "?"
            st["alert_counts"][label] = st["alert_counts"].get(label, 0) + 1

    out = []
    for tid, st in stats.items():
        meta = names.get(tid, {})
        trend = (st["last_threat"] or 0) - (st["first_threat"] or 0)
        out.append({
            "id": tid,
            "name": meta.get("name") or tid,
            "affiliation": meta.get("affiliation"),
            "domain": meta.get("domain"),
            "platform_class": meta.get("platform_class"),
            "model": meta.get("model"),
            "peak_threat": round(st["peak_threat"], 1),
            "peak_frame": st["peak_frame"],
            "peak_clock": st["peak_clock"],
            "min_nearest_km": (round(st["min_nearest_km"], 1)
                               if st["min_nearest_km"] is not None else None),
            "max_covered": st["max_covered"],
            "state_changes": st["states"],
            # A target reporting one state for the whole run is the common case;
            # only transitions are interesting in a brief.
            "state_transition": (st["states"][0] + " → " + st["states"][-1]
                                 if len(st["states"]) > 1 else None),
            "intent_changes": st["intents"],
            "alert_counts": st["alert_counts"],
            "threat_trend": round(trend, 1),
            "trend_label": ("上升" if trend > 2 else ("下降" if trend < -2 else "平稳")),
        })
    out.sort(key=lambda r: -r["peak_threat"])
    return out


def _analyse_regions(frames: Sequence[dict], hotspot_names: Sequence[str],
                     start_utc: Optional[str], timestamps: Sequence[float]) -> List[dict]:
    n_regions = 0
    for frame in frames:
        regs = frame.get("regions") or []
        if regs:
            n_regions = max(n_regions, len(regs))
    names = list(hotspot_names or [])
    out = []
    for ri in range(n_regions):
        peak_level = 0
        peak_frame = 0
        peak_red = peak_blue = 0
        dominance_counts: Dict[str, int] = {}
        series = []
        for fi, frame in enumerate(frames):
            regs = frame.get("regions") or []
            if ri >= len(regs):
                continue
            level, n_red, n_blue, dominance = (list(regs[ri]) + [0, 0, 0, 0])[:4]
            series.append(level)
            if level > peak_level:
                peak_level, peak_frame = level, fi
                peak_red, peak_blue = n_red, n_blue
            code = (frame.get("model") or {}).get("dominance_codes") or {}
            dom_label = code.get(str(dominance)) or code.get(dominance) or "—"
            dominance_counts[dom_label] = dominance_counts.get(dom_label, 0) + 1
        runs = _collapse_runs(series)
        out.append({
            "index": ri,
            "name": names[ri] if ri < len(names) else ("区域 %d" % (ri + 1)),
            "peak_level": peak_level,
            "peak_frame": peak_frame,
            "peak_clock": _fmt_clock(start_utc, timestamps[peak_frame] if timestamps else None)
                          if peak_frame < len(timestamps) else None,
            "peak_red": peak_red,
            "peak_blue": peak_blue,
            "level_runs": len(runs),
            "level_series_head": series[:12],
            "dominance_counts": dominance_counts,
        })
    out.sort(key=lambda r: -r["peak_level"])
    return out


def build_brief(frames: Sequence[dict], targets=None, hotspot_names: Sequence[str] = (),
                model: Optional[dict] = None, start_utc: Optional[str] = None,
                case_id: Optional[str] = None, case_name: Optional[str] = None,
                tick_seconds: Optional[float] = None) -> dict:
    """Build the structured brief.

    `frames` is the per-frame situation. `inference.infer()`'s frames carry only
    `tick`, not `t_seconds`, so `tick_seconds` is what lets the brief convert a
    frame index into a wall clock -- without it every timestamp comes out as the
    start time. `targets` are the scenario Target objects. `model` is
    `inference.infer()`'s model block, used for the code tables so the wording
    never diverges from the levels the front end shows.
    """
    frames = list(frames or [])
    if not frames:
        return {
            "case_id": case_id, "case_name": case_name, "empty": True,
            "headline": "该TS没有可用的帧数据。",
            "sections": [], "key_targets": [], "regions": [],
        }

    # The inference output is aligned frame-by-frame with the dataset; callers
    # may pass either, so pick whichever carries L1.
    inferred = [f for f in frames if "targets" in f and isinstance(f.get("targets"), dict)]
    levels = []
    for f in inferred:
        ov = f.get("overall") or {}
        lv = ov.get("level")
        if lv is None:
            # Derive from L3 when `overall` is absent (e.g. hand-built frames).
            regs = f.get("regions") or []
            lv = max([r[0] for r in regs], default=1)
        levels.append(int(lv))

    # 每帧的仿真秒数。**inference_frames 里没有 t_seconds**（只有 tick），
    # 之前直接取 `f.get("t_seconds") or 0.0` 于是全为 0，所有钟点都算成
    # 00:00:00Z —— 简报里每一处时间戳都是同一个值，看着像功能正常，其实是错的。
    # 优先用帧自带的 t_seconds（原始数据集有），退化为 tick * tick_seconds。
    secs_per_frame = float(tick_seconds or 30.0)
    timestamps = []
    for i, f in enumerate(inferred):
        ts = f.get("t_seconds")
        if ts is None:
            ts = float(f.get("tick", i)) * secs_per_frame
        timestamps.append(float(ts))

    def time_at(idx: int) -> Optional[str]:
        if 0 <= idx < len(timestamps):
            return _fmt_clock(start_utc, timestamps[idx])
        return None

    names = _target_names(targets)
    model = model or {}
    level_codes = model.get("level_codes") or {}

    # ---- headline ------------------------------------------------------
    peak_level = max(levels) if levels else 1
    peak_idx = levels.index(peak_level) if levels else 0
    final_level = levels[-1] if levels else 1
    start_level = levels[0] if levels else 1
    peak_clock = time_at(peak_idx)
    peak_label = _level_label(model, peak_level)
    final_label = _level_label(model, final_level)

    trend_word = ("升级" if peak_level > start_level
                  else ("缓和" if final_level < peak_level else "维持"))
    headline = "%s：全程峰值%s%s，TS%s。" % (
        case_name or case_id or "本TS",
        (" " + peak_label) if peak_label else "",
        ("（第 %d 帧%s）" % (inferred[peak_idx].get("tick", peak_idx),
                            ("，" + peak_clock) if peak_clock else "")) if inferred else "",
        trend_word,
    )

    # ---- sections ------------------------------------------------------
    sections: List[dict] = []

    sections.append({
        "title": "总体判断",
        "kind": "text",
        "lines": ["共 %d 帧。" % len(frames)] + _describe_levels(model, levels, time_at),
    })

    target_rows = _analyse_targets(inferred, names, start_utc, timestamps)
    hot = [r for r in target_rows if r["peak_threat"] >= 60]
    if hot:
        sections.append({
            "title": "高威胁目标",
            "kind": "targets",
            "lines": [
                "%s（%s）峰值威胁 %s（%s），最近接敌 %s km%s%s。" % (
                    r["name"], r["affiliation"] or "—", r["peak_threat"],
                    r["alert_counts"] and max(r["alert_counts"], key=r["alert_counts"].get) or "—",
                    r["min_nearest_km"] if r["min_nearest_km"] is not None else "—",
                    ("，" + r["peak_clock"]) if r["peak_clock"] else "",
                    ("，" + r["state_transition"]) if r["state_transition"] else "",
                )
                for r in hot[:TOP_N]
            ],
            "data": hot[:TOP_N],
        })

    changed = [r for r in target_rows if r["state_transition"] or r["trend_label"] == "上升"]
    if changed:
        sections.append({
            "title": "状态与意图变化",
            "kind": "targets",
            "lines": [
                "%s：%s%s。" % (
                    r["name"],
                    r["state_transition"] or "状态未变",
                    ("，威胁" + r["trend_label"] + " %+.1f" % r["threat_trend"])
                    if r["trend_label"] == "上升" else "",
                )
                for r in changed[:TOP_N]
            ],
            "data": changed[:TOP_N],
        })

    region_rows = _analyse_regions(inferred, hotspot_names, start_utc, timestamps)
    if region_rows:
        prob = [r for r in region_rows if r["peak_level"] >= 2]
        sections.append({
            "title": "区域TS",
            "kind": "regions",
            "lines": [
                "%s：峰值%s（第 %d 帧%s），最高时红方 %d / 蓝方 %d。" % (
                    r["name"], _level_label(model, r["peak_level"]), r["peak_frame"],
                    ("，" + r["peak_clock"]) if r["peak_clock"] else "",
                    r["peak_red"], r["peak_blue"],
                )
                for r in (prob or region_rows)[:TOP_N]
            ],
            "data": (prob or region_rows)[:TOP_N],
        })

    return {
        "case_id": case_id,
        "case_name": case_name,
        "empty": False,
        "frame_count": len(frames),
        "start_utc": start_utc,
        "peak_level": peak_level,
        "peak_level_label": peak_label,
        "peak_frame": peak_idx,
        "peak_clock": peak_clock,
        "final_level": final_level,
        "final_level_label": final_label,
        "trend": trend_word,
        "headline": headline,
        "sections": sections,
        "key_targets": target_rows[:TOP_N],
        "regions": region_rows,
        "level_runs": _collapse_runs(levels),
    }


def render_text(brief: dict) -> str:
    """Plain-text rendering, for the CLI and for copy-paste into a report."""
    if brief.get("empty"):
        return brief.get("headline") or "（无数据）"
    out = []
    out.append("TS简报 —— %s" % (brief.get("case_name") or brief.get("case_id") or "未命名"))
    out.append("=" * 46)
    out.append(brief.get("headline") or "")
    out.append("")
    for sec in brief.get("sections") or []:
        out.append("【%s】" % sec.get("title"))
        for line in sec.get("lines") or []:
            out.append("  · " + line)
        out.append("")
    return "\n".join(out).rstrip() + "\n"