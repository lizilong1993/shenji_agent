"""GP-01 / GP-02 / GP-04 的浏览器验收。

在真实页面里做三件事，而不是只跑单测：

  1. GP-01：打开 TW3，推进到有损毁的 tick，确认
       (a) 战损呈现层真的在跑（烟柱实体存在）；
       (b) 血量档位与数据集里那一帧的 blood 一致（此前 62 % 是错的）。
  2. GP-02：在目标最密集的一帧采样，确认没有两个血条的屏幕矩形重叠。
  3. GP-04：派发 webglcontextlost，确认提示条出现、时间轴暂停、状态点变红；
     再派发 restored，确认恢复。

用法：python tools/verify_gp_browser.py [--port 8848] [--headed]
"""

from __future__ import annotations

import argparse
import json
import sys

from playwright.sync_api import sync_playwright

CASE_TW3 = "taiwan_campaign"


def _probe(page, script, arg=None):
    return page.evaluate(script, arg)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8848)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args(argv)

    base = "http://127.0.0.1:%d/" % args.port
    results = []

    def rec(name, ok, detail=""):
        results.append((name, ok, detail))
        print(("  ok   " if ok else "  FAIL ") + name + (("  —  " + detail) if detail else ""))

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=not args.headed,
            args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader",
                  "--disable-gpu-sandbox"])
        page = browser.new_page(viewport={"width": 1600, "height": 900})
        console = []
        page.on("console", lambda m: console.append((m.type, m.text)))
        page.on("pageerror", lambda e: console.append(("pageerror", str(e))))

        page.goto(base, wait_until="domcontentloaded", timeout=60000)

        # 等到取证出口就绪（viewer/state 是 let，只在 SituationHost 上暴露）。
        # 注意**不等 state.data**：默认案例是 TW1（41 MB），首次加载要十几秒，
        # 我们要的其实是 TW3，先等案例索引到齐就能点菜单了。
        page.wait_for_function(
            "() => window.SituationHost && window.SituationHost.viewer() "
            "&& window.SituationHost.state.caseIndex "
            "&& (window.SituationHost.state.caseIndex.cases || []).length > 0",
            timeout=90000)

        # 切到 TW3：loadCase 不导出，走真实用户路径——点案例菜单项。
        page.evaluate("""() => {
            const btn = document.getElementById('menuCase') ||
                        document.querySelector('[data-menu="case"]') ||
                        document.querySelector('.menu-btn');
            if (btn) btn.click();
        }""")
        page.wait_for_timeout(400)
        clicked = page.evaluate("""(cid) => {
            const items = document.querySelectorAll('[data-case-id]');
            for (const it of items) {
              if (it.dataset.caseId === cid) { it.click(); return true; }
            }
            return Array.from(items).map(i => i.dataset.caseId);
        }""", CASE_TW3)
        if clicked is not True:
            raise SystemExit("案例菜单里没找到 %s，实际有 %s" % (CASE_TW3, clicked))

        page.wait_for_function(
            "(cid) => window.SituationHost.state.data "
            "&& window.SituationHost.state.data.case_id === cid",
            arg=CASE_TW3, timeout=90000)
        page.wait_for_timeout(3000)

        print("1. GP-01 战损呈现由 HP 驱动")
        # 找一个已击毁的目标出现在数据集里的 tick
        probe = _probe(page, """() => {
            const H = window.SituationHost, d = H.state.data, mega = {};
            (d.targets || []).forEach(t => mega[t.id] = t.max_blood);
            let tick = -1, id = null;
            for (let t = 0; t < d.frames.length; t++) {
              const rows = (d.frames[t] || {}).targets || [];
              for (let i = 0; i < rows.length; i++) {
                const r = rows[i];
                if (mega[r.id] && r.status === 'destroyed' && r.blood === 0) {
                  tick = t; id = r.id; break;
                }
              }
              if (tick >= 0) break;
            }
            return {tick, id, frameCount: d.frames.length};
        }""")
        rec("找到一条击毁记录", probe["tick"] >= 0, json.dumps(probe))

        if probe["tick"] >= 0:
            tick = min(probe["tick"] + 5, probe["frameCount"] - 1)
            _probe(page, """(t) => {
                if (window.TransportBridge) TransportBridge.setTick(t);
                return window.SituationHost.state.tick;
            }""", tick)
            page.wait_for_timeout(2200)

            # 战损层是否在跑：看 smokeState / fireStage 这些内部状态。
            # 它们不在 window 上，改为看实体数量与 HP 一致性。
            hpchk = _probe(page, """(id) => {
                const H = window.SituationHost, d = H.state.data;
                const maxBlood = {};
                (d.targets || []).forEach(t => maxBlood[t.id] = t.max_blood);
                const rows = ((d.frames[H.state.tick] || {}).targets) || [];
                let bad = 0, total = 0;
                rows.forEach(r => {
                  if (!maxBlood[r.id]) return;
                  total++;
                  const at = window.TargetStatus.at(d, r.id, H.state.tick);
                  const want = Math.round(r.blood / maxBlood[r.id] * 1000) / 10;
                  if (Math.abs(at.hp - want) > 0.05) bad++;
                  if ((r.status === 'destroyed') !== (at.status === 3)) bad++;
                });
                return {bad, total, tick: H.state.tick};
            }""", probe["id"])
            rec("当帧 HP/存活判定与数据一致", hpchk["bad"] == 0,
                "%d/%d 不一致 @tick %s" % (hpchk["bad"], hpchk["total"], hpchk["tick"]))

            smoke = _probe(page, """() => {
                // 冒烟档（hp<98）的目标数 vs 页面里可见的烟柱 billboard 数。
                const H = window.SituationHost, d = H.state.data, mega = {};
                (d.targets || []).forEach(t => mega[t.id] = t.max_blood);
                const rows = ((d.frames[H.state.tick] || {}).targets) || [];
                let smoking = 0;
                rows.forEach(r => {
                  if (!mega[r.id]) return;
                  const at = window.TargetStatus.at(d, r.id, H.state.tick);
                  if (at.hp < 98) smoking++;
                });
                const imgs = new Set();
                const v2 = H.viewer(); const ents = v2.entities.values;
                let smokeEnts = 0;
                for (let i = 0; i < ents.length; i++) {
                  const bb = ents[i].billboard;
                  if (!bb || !bb.image) continue;
                  let src = '';
                  try { src = String(bb.image.getValue ? bb.image.getValue(v2.clock.currentTime) : bb.image); } catch (e) {}
                  if (/smoke\\.png/.test(src)) smokeEnts++;
                }
                return {smoking, smokeEnts};
            }""")
            rec("冒烟档目标存在", smoke["smoking"] > 0, "smoking=%d" % smoke["smoking"])
            rec("页面确实画了烟柱（战损层在跑）", smoke["smokeEnts"] > 0,
                "smokeEntities=%d" % smoke["smokeEnts"])

        print("2. GP-02 血条不重叠")
        # 推进到目标最多的帧。注意 **不能是 tick 0**：本项目的 CZML 采样从
        # 00:00:30 开始，tick 0 落在第一个采样之前，投影拿不到位置，
        # 那是既有行为（与血条避让无关），会把这一节的结论带偏。
        dense = _probe(page, """() => {
            const H = window.SituationHost;
            let best = 1;
            (H.state.data.frames || []).forEach((f, t) => {
              if (t === 0) return;
              if (((f || {}).targets || []).length > ((H.state.data.frames[best] || {}).targets || []).length) best = t;
            });
            return best;
        }""")
        _probe(page, """(t) => { TransportBridge.setTick(t);
              return window.SituationHost.state.tick; }""", dense)
        # 机位必须落到目标上：默认机位是战役起点的俯视，跳到别的 tick 后
        # 多数目标在视口外（实测 y=954 > 画布高 810），那会让"血条避让"
        # 这一节测不到东西。飞到当前帧目标群的质心，保持在能看见几十个目标的
        # 高度，才谈得上密集。
        _probe(page, """() => {
            const H = window.SituationHost, st = H.state, C = window.Cesium;
            const fr = (st.data.frames[st.tick] || {}).targets || [];
            if (!fr.length) return null;
            let lon = 0, lat = 0;
            fr.forEach(r => { lon += r.lon; lat += r.lat; });
            lon /= fr.length; lat /= fr.length;
            H.viewer().camera.setView({
              destination: C.Cartesian3.fromDegrees(lon, lat, 120000),
              orientation: { heading: 0, pitch: C.Math.toRadians(-70), roll: 0 },
            });
            return [lon, lat, 120000];
        }""")
        page.wait_for_timeout(3500)

        layout = _probe(page, """() => {
            const C = window.Cesium, H = window.SituationHost, v = H.viewer();
            const time = v.clock.currentTime;
            const rects = [];
            Object.keys(H.state.targets).forEach(id => {
              const rec = H.state.targets[id], bar = rec.hpBar;
              if (!bar) return;
              const shown = (bar.show && bar.show.getValue) ? bar.show.getValue(time) : bar.show;
              if (!shown) return;
              const p = rec.entity.position && rec.entity.position.getValue
                ? rec.entity.position.getValue(time) : null;
              if (!p) return;
              const w = C.SceneTransforms.worldToWindowCoordinates(v.scene, p);
              if (!w) return;
              const off = (bar.pixelOffset && bar.pixelOffset.getValue)
                ? bar.pixelOffset.getValue(time) : bar.pixelOffset;
              rects.push({id, x: w.x + off.x - 17, y: w.y + off.y, w: 34, h: 4});
            });
            let overlap = 0, pairs = [];
            for (let i = 0; i < rects.length; i++)
              for (let j = i + 1; j < rects.length; j++) {
                const a = rects[i], b = rects[j];
                if (!(a.x + a.w + 2 < b.x || b.x + b.w + 2 < a.x ||
                      a.y + a.h + 2 < b.y || b.y + b.h + 2 < a.y)) {
                  overlap++; pairs.push([a.id, b.id]);
                }
              }
            return {count: rects.length, overlap, pairs: pairs.slice(0, 5),
                    offsets: Object.keys(H.state.targets).map(id => {
                      const b2 = H.state.targets[id] && H.state.targets[id].hpBar;
                      if (!b2) return null;
                      const o = (b2.pixelOffset && b2.pixelOffset.getValue)
                        ? b2.pixelOffset.getValue(time) : b2.pixelOffset;
                      return o ? o.x + '|' + o.y : null;
                    })};
        }""")
        rec("血条有显示", layout["count"] >= 1, "count=%d" % layout["count"])
        rec("血条两两不重叠", layout["overlap"] == 0,
            "overlap=%d %s" % (layout["overlap"], layout["pairs"]))
        # 画面上同时展开的血条数量受编队聚合限制（120 km 高度下 23/33 个目标
        # 被聚成簇，成员血条按设计隐藏）。密集情形下的避让由
        # tools/health_layout_dense_test.js 用合成数据覆盖，这里只保证
        # "在真实页面上确实排出了位置且不重叠"。
        rec("血条位置已按避让排布过",
            any(o not in ("0|17", "0|0", None) for o in layout.get("offsets", [])),
            "offsets=%s" % (layout.get("offsets"),))

        print("3. GP-04 WebGL 上下文丢失")
        ctx = _probe(page, """() => {
            const c = window.SituationHost.viewer().canvas;
            if (!c) return {err: 'no canvas'};
            const ev = new Event('webglcontextlost', {bubbles: true, cancelable: true});
            c.dispatchEvent(ev);
            return {defaultPrevented: ev.defaultPrevented};
        }""")
        rec("丢失事件被 preventDefault", ctx.get("defaultPrevented") is True, json.dumps(ctx))
        page.wait_for_timeout(600)
        ui = _probe(page, """() => {
            const H = window.SituationHost;
            const el = document.getElementById('renderRecovery');
            const dot = document.querySelector('#statusDot') || document.querySelector('.dot');
            return {
              panelVisible: !!(el && getComputedStyle(el).display !== 'none'),
              playing: !!H.state.playing,
              dotClass: dot ? dot.className : '',
              statusText: (document.getElementById('statusText') || {}).textContent || '',
            };
        }""")
        rec("恢复提示条出现", ui["panelVisible"] is True, json.dumps(ui, ensure_ascii=False))
        rec("时间轴已暂停", ui["playing"] is False)
        rec("状态点变红", "err" in (ui["dotClass"] or ""), ui["dotClass"])

        _probe(page, """() => {
            const c = window.SituationHost.viewer().canvas;
            c.dispatchEvent(new Event('webglcontextrestored', {bubbles: true}));
            return true;
        }""")
        page.wait_for_timeout(600)
        ui2 = _probe(page, """() => {
            const el = document.getElementById('renderRecovery');
            const dot = document.querySelector('#statusDot') || document.querySelector('.dot');
            return {
              panelVisible: !!(el && getComputedStyle(el).display !== 'none'),
              dotClass: dot ? dot.className : '',
            };
        }""")
        rec("恢复后提示条收起", ui2["panelVisible"] is False)
        rec("恢复后状态点回绿", "ok" in (ui2["dotClass"] or ""), ui2["dotClass"])

        # 只认**未捕获异常**与真正的脚本错误。已知且预期内的 404 不算：
        #   · /api/sim/cases —— demo 模式根本不开放模拟 API；
        #   · 离线瓦片缺片 —— 演示环境没补齐的图幅，与本次改动无关。
        errors = [c for c in console
                  if c[0] == "pageerror"
                  and "api/sim/cases" not in c[1]
                  and "Failed to load resource" not in c[1]]
        rec("页面无未捕获异常", len(errors) == 0,
            "; ".join(t for _, t in errors[:3]))

        if args.headed:
            page.screenshot(path="tools/_gp_verify.png", full_page=False)
        browser.close()

    print("")
    failed = [r for r in results if not r[1]]
    if failed:
        print("FAILED: %d 项" % len(failed))
        return 1
    print("PASSED: %d 项" % len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
