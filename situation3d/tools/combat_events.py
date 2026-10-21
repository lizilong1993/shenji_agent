# -*- coding: utf-8 -*-
"""为案例合成开火/命中/击毁事件（combat.synth.v1）。

现有案例没有交战事件字段；本脚本从 frames（位置）+ inference_frames（alert/threat）
确定性合成事件，写回案例 JSON 的 `events`（保留原有事件，先删除旧的本脚本产物）。

用法：
    python3 tools/combat_events.py web/data/situations/taiwan_landing.json
    python3 tools/combat_events.py --dry-run web/data/situations/*.json

约定：
- 只在交战案例上运行（case_id 含 dense_transfer 的转运演示跳过，除非 --include-noncombat）。
- 事件 schema 与 campaign events 对齐：id/tick/stage_id/type/source/producer/actors/
  location/effects/cause_event_ids/evidence_ids/visibility/description；
  战斗事件额外带 weapon/result/damage/shooter_id/target_id/target_location。
- 击毁时点与 web/src/targetStatus.js 的推导完全一致（同一套暴露度/比例/阈值），
  避免图标显示“已损毁”但模型还在开火的矛盾。
- 事件是**演示合成**，不是实测战损；`combat.model = "combat.synth.v1"` 记录来源。
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time

WEIGHTS = {2: 1, 3: 3}
BANDS = {"destroyed": 0.06, "severe": 0.10, "damaged": 0.22}
SETTLE_AT = 0.8
MIN_DESTROY_EXPOSURE = 800
SRC = "combat.synth.v1"


def hav_km(a, b):
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * math.asin(min(1.0, math.sqrt(h)))


def stable01(*parts):
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(0xFFFFFFFFFFFF)


def weapon_for(defn):
    d = str(defn.get("domain") or "").lower()
    p = str(defn.get("platform_class") or "") + str(defn.get("model") or "")
    if d == "air":
        if "直升" in p or "helicopter" in p.lower():
            return ("空地导弹", 8, 2, 18)
        if "无人机" in p or "uav" in p.lower() or "mq" in p.lower():
            return ("空地导弹", 9, 2, 14)
        if "预警" in p or "aew" in p.lower():
            return None
        return ("空空导弹", 8, 1, 18)
    if d == "land":
        if "岸舰" in p or "anti" in p.lower():
            return ("岸舰导弹", 12, 3, 30)
        if "防空" in p or "sam" in p.lower() or "红旗" in p or "patriot" in p.lower():
            return ("防空导弹", 10, 2, 26)
        if "火箭" in p or "phl" in p.lower() or "mlrs" in p.lower():
            return ("远程火箭弹", 10, 2, 16)
        if "榴弹" in p or "m777" in p.lower() or "m109" in p.lower() or "plz" in p.lower() or "炮" in p:
            return ("155mm 炮弹", 6, 1, 12)
        if "坦克" in p or "tank" in p.lower() or "装甲" in p:
            return ("125mm 穿甲弹", 5, 1, 10)
        return None
    if d in ("sea", "naval", "maritime"):
        if "补给" in p or "登陆" in p or "两栖" in p:
            return ("近防炮", 6, 1, 6)
        if "巡逻" in p or "轻护" in p or "056" in p or "沱江" in p:
            return ("反舰导弹", 12, 2, 24)
        return ("反舰导弹", 12, 3, 34)
    return None


def can_engage(shooter, target):
    sd = str(shooter.get("domain") or "").lower()
    td = str(target.get("domain") or "").lower()
    sp = str(shooter.get("platform_class") or "") + str(shooter.get("model") or "")
    if sd == "sea":
        return td in ("sea", "air") and td != "land"
    if sd == "land":
        if "岸舰" in sp or "anti" in sp.lower():
            return td == "sea"
        if "防空" in sp or "sam" in sp.lower() or "红旗" in sp or "patriot" in sp.lower():
            return td == "air"
        return td == "land"
    if sd == "air":
        return td in ("air", "sea", "land")
    return False


def status_series(case):
    """复刻 targetStatus.js：返回 (destroyedAt, hp_at)。"""
    targets = case.get("targets") or []
    inf = case.get("inference_frames") or []
    ids = [t["id"] for t in targets]
    total = {i: 0 for i in ids}
    for t in range(1, len(inf)):
        fr = (inf[t] or {}).get("targets") or {}
        for i in ids:
            e = fr.get(i)
            if e:
                total[i] += WEIGHTS.get(int(e.get("alert") or 0), 0)
    active = [i for i in ids if total[i] > 0]
    active.sort(key=lambda i: (-total[i], i))
    roster = len(ids)
    if roster >= 5:
        dN = max(1, int(roster * BANDS["destroyed"]))
        sN = max(1, int(roster * BANDS["severe"]))
        mN = max(1, int(roster * BANDS["damaged"]))
    else:
        dN = 0
        sN = 1 if len(active) > 1 else 0
        mN = 1 if len(active) > 2 else 0
    n = len(active)
    if n > 0 and dN + sN + mN >= n:
        cap = max(1, n - 1)
        tot = dN + sN + mN
        dN = int(dN * cap / tot)
        sN = max(0, int(sN * cap / tot))
        mN = max(0, cap - dN - sN)
    if dN > 0 and active and total[active[0]] < MIN_DESTROY_EXPOSURE:
        dN = 0
    final = {i: 100.0 for i in ids}
    for idx, i in enumerate(active):
        if idx < dN:
            final[i] = 0.0
        elif idx < dN + sN:
            k = idx - dN
            final[i] = 30.0 if sN <= 1 else 30.0 - 25.0 * k / (sN - 1)
        elif idx < dN + sN + mN:
            j = idx - dN - sN
            final[i] = 68.0 if mN <= 1 else 68.0 - 28.0 * j / (mN - 1)
        else:
            final[i] = 100.0
    expo = {i: 0 for i in ids}
    hp_at = {}
    destroyed_at = {}
    for t in range(0, len(inf)):
        fr = (inf[t] or {}).get("targets") or {}
        if t > 0:
            for i in ids:
                e = fr.get(i)
                if e:
                    expo[i] += WEIGHTS.get(int(e.get("alert") or 0), 0)
        row = {}
        for i in ids:
            tot = total[i]
            hp = 100.0
            if tot > 0 and final[i] < 100:
                hp = 100.0 - (100.0 - final[i]) * min(1.0, expo[i] / (SETTLE_AT * tot))
            if final[i] == 0 and tot > 0 and expo[i] >= SETTLE_AT * tot:
                hp = 0.0
                destroyed_at.setdefault(i, t)
            row[i] = round(hp, 1)
        hp_at[t] = row
    return destroyed_at, hp_at


def stage_of(case, tick):
    for ph in case.get("phases") or []:
        if ph.get("tick") is not None and ph.get("tick") == tick:
            return ph.get("id") or ""
    stages = case.get("stages") or []
    for st in stages:
        if st.get("start_tick", 0) <= tick < st.get("end_tick", 10 ** 9):
            return st.get("id") or ""
    return ""


def synth(case, include_noncombat=False):
    cid = str(case.get("case_id") or "")
    if "dense_transfer" in cid and not include_noncombat:
        return None
    targets = case.get("targets") or []
    frames = case.get("frames") or []
    inf = case.get("inference_frames") or []
    if not targets or not frames or not inf:
        return None
    defs = {t["id"]: t for t in targets}
    destroyed_at, hp_at = status_series(case)
    # 位置索引
    pos = {}
    for fr in frames:
        tick = int(fr.get("tick") or 0)
        p = {}
        for row in fr.get("targets") or []:
            p[row["id"]] = (float(row["lat"]), float(row["lon"]), float(row.get("alt") or 0))
        pos[tick] = p
    # 告警/威胁
    alert = {}
    for fr in inf:
        tick = int(fr.get("tick") or 0)
        a = {}
        for k, v in (fr.get("targets") or {}).items():
            a[k] = (int((v or {}).get("alert") or 0), float((v or {}).get("threat") or 0))
        alert[tick] = a

    events = []
    seq = 0
    cooldown = {}
    recent_hits = {}
    for tick in range(0, len(frames), 1):
        p = pos.get(tick)
        a = alert.get(tick)
        if not p or not a:
            continue
        alive = [i for i in defs if hp_at.get(tick, {}).get(i, 100) > 0]
        shooters = [i for i in alive if a.get(i, (0, 0))[0] >= 2]
        if not shooters:
            continue
        # 每个射手选最近敌对目标
        pairs = []
        for s in shooters:
            sd = defs[s]
            best = None
            for tg in alive:
                td = defs[tg]
                if td.get("affiliation") == sd.get("affiliation"):
                    continue
                if not can_engage(sd, td):
                    continue
                if s not in p or tg not in p:
                    continue
                d = hav_km(p[s][:2], p[tg][:2])
                rng = float(sd.get("engage_km") or 0)
                if rng <= 0:
                    rng = 30.0
                if d <= rng and (best is None or d < best[2]):
                    best = (s, tg, d)
            if best:
                pairs.append(best)
        pairs.sort(key=lambda x: x[2])
        pairs = pairs[:4]
        for s, tg, dist in pairs:
            w = weapon_for(defs[s])
            if not w:
                continue
            wname, cd, flight, dmg = w
            key = (s, tg)
            if tick - cooldown.get(key, -10 ** 9) < cd:
                continue
            cooldown[key] = tick
            seq += 1
            fid = "CF%06d" % seq
            sloc = p[s]
            tloc = p[tg]
            events.append({
                "id": fid, "tick": tick, "stage_id": stage_of(case, tick),
                "type": "fire", "source": "combat", "producer": SRC,
                "actors": [s, tg], "weapon": wname,
                "shooter_id": s, "target_id": tg,
                "location": [round(sloc[1], 6), round(sloc[0], 6), round(sloc[2], 1)],
                "target_location": [round(tloc[1], 6), round(tloc[0], 6), round(tloc[2], 1)],
                "effects": {"before": {}, "after": {}},
                "cause_event_ids": [], "evidence_ids": [],
                "visibility": "global",
                "description": "%s 向 %s 发射%s" % (s, tg, wname),
            })
            # 命中判定
            p_hit = 0.55 + 0.10 * min(3, a.get(s, (0, 0))[0]) + min(0.15, a.get(s, (0, 0))[1] / 600.0)
            hit = stable01(cid, tick, s, tg, "hit") < min(0.9, p_hit)
            itick = min(len(frames) - 1, tick + flight)
            ip = pos.get(itick, {}).get(tg, tloc)
            seq += 1
            iid = "CF%06d" % seq
            if hit and hp_at.get(itick, {}).get(tg, 100) <= 0:
                # 已在此前的击毁事件中处理，不重复命中
                hit = False
            events.append({
                "id": iid, "tick": itick, "stage_id": stage_of(case, itick),
                "type": "impact", "source": "combat", "producer": SRC,
                "actors": [s, tg], "weapon": wname,
                "shooter_id": s, "target_id": tg,
                "result": "hit" if hit else "miss",
                "damage": dmg if hit else 0,
                "location": [round(ip[1], 6), round(ip[0], 6), round(ip[2], 1)],
                "target_location": [round(ip[1], 6), round(ip[0], 6), round(ip[2], 1)],
                "effects": {"before": {}, "after": {}},
                "cause_event_ids": [fid], "evidence_ids": [],
                "visibility": "global",
                "description": "%s 的%s%s" % (tg, wname, "命中" if hit else "未命中"),
            })
            if hit:
                recent_hits.setdefault(tg, []).append(iid)
    # 击毁事件（时点与 targetStatus 一致）
    for tid, dt in sorted(destroyed_at.items(), key=lambda kv: kv[1]):
        p = pos.get(dt, {}).get(tid)
        if not p:
            continue
        seq += 1
        events.append({
            "id": "CF%06d" % seq, "tick": dt, "stage_id": stage_of(case, dt),
            "type": "destroy", "source": "combat", "producer": SRC,
            "actors": [tid], "target_id": tid,
            "location": [round(p[1], 6), round(p[0], 6), round(p[2], 1)],
            "effects": {"before": {"hp": None}, "after": {"hp": 0}},
            "cause_event_ids": recent_hits.get(tid, [])[-2:],
            "evidence_ids": [], "visibility": "global",
            "description": "%s 战损判定为击毁" % tid,
        })
    events.sort(key=lambda e: (e["tick"], e["id"]))
    existing = [e for e in (case.get("events") or []) if e.get("producer") != SRC]
    case["events"] = existing + events
    case["combat"] = {
        "model": SRC,
        "event_count": len(events),
        "fire": sum(1 for e in events if e["type"] == "fire"),
        "impact": sum(1 for e in events if e["type"] == "impact"),
        "destroy": sum(1 for e in events if e["type"] == "destroy"),
        "rules": "alert>=2 + distance<=engage_km sum; deterministic sha1",
        "disclaimer": "演示合成交战事件，非实测战损",
    }
    return case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-noncombat", action="store_true")
    args = ap.parse_args()
    for path in args.files:
        case = json.load(open(path, encoding="utf-8"))
        out = synth(case, args.include_noncombat)
        if out is None:
            print("SKIP", path)
            continue
        n = (out.get("combat") or {}).get("event_count", 0)
        print("SYNTH", os.path.basename(path), n, "events",
              out["combat"]["fire"], out["combat"]["impact"], out["combat"]["destroy"])
        if args.dry_run:
            continue
        bak = path + ".bak-combat-" + time.strftime("%Y%m%d_%H%M%S")
        shutil.copyfile(path, bak)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        print("WROTE", path, "backup", os.path.basename(bak))


if __name__ == "__main__":
    main()
