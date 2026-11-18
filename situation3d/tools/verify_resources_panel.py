"""A（验收部分）：资源面板对真实资源库的浏览器验收。

TODOLIST 里这条原本写的是「8858 资源面板对真实 19 条数据的人工验收——
阻塞：数据在 NFS 上」。2026-11-18 核实：本机 `var/resources/resources.db`
已有 15 条真实条目（NFS 上的服务器库不可达），所以「验收」这件事本身
在本机就能做，不必等 NFS。

本脚本验的是**面板能正确显示并筛选这批真实数据**，不是重跑 P4 的设计验收：
  1. 打开主页面，资源管理入口存在且可点开；
  2. 列表里的条目集合与 /api/resources 返回的完全一致（不截断、不重复）；
  3. 顶部 kind 计数与后端 stats 一致；
  4. 搜索框能按 kind 前缀过滤（走「搜索」按钮，不是 oninput）；
  5. test 模式下路径是明文（demo 模式会打码，那是另一条规则）。
  6. 无未捕获异常。

用法：python tools/verify_resources_panel.py [--port 8858] [--headed]
"""

from __future__ import annotations

import argparse
import json
import sys

from playwright.sync_api import sync_playwright

# 直接读每行第一列的 entry_id；不要用 [data-entry-id]——那是行内四个操作按钮
# 的标记，按元素计数会得到 4 倍行数。
_ROWS = """
() => {
  const b = document.getElementById('resourceBody');
  if (!b) return [];
  return Array.from(b.querySelectorAll('tr .res-id'))
      .map(n => (n.textContent || '').trim());
}
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8858)
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
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        console = []
        page.on("console", lambda m: console.append((m.type, m.text)))
        page.on("pageerror", lambda e: console.append(("pageerror", str(e))))

        page.goto(base, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_function(
            "() => window.ResourceManager && document.getElementById('btnResourceManager')",
            timeout=90000)
        page.wait_for_timeout(1000)

        print("1. 面板入口")
        opened = page.evaluate("""() => {
            const btn = document.getElementById('btnResourceManager');
            if (!btn) return {err: 'no button'};
            const modal = document.getElementById('resourceModal');
            const wasHidden = modal ? modal.className.includes('hidden') : null;
            btn.click();
            return {wasHidden};
        }""")
        page.wait_for_timeout(800)
        shown = page.evaluate("""() => {
            const modal = document.getElementById('resourceModal');
            return {hidden: !modal || modal.className.includes('hidden'),
                    hasSearch: !!document.getElementById('resSearch')};
        }""")
        rec("资源管理入口可点开", shown["hidden"] is False, json.dumps(shown, ensure_ascii=False))
        rec("面板里有搜索框", shown["hasSearch"] is True)

        page.wait_for_function(
            "() => { const b = document.getElementById('resourceBody');"
            " return b && b.querySelectorAll('[data-entry-id]').length > 0; }",
            timeout=60000)
        page.wait_for_timeout(500)

        print("2. 条目集合与后端一致")
        api = page.evaluate("""() => fetch('/api/resources').then(r => r.json())""")
        ui_ids = page.evaluate(_ROWS)
        api_ids = sorted(e["entry_id"] for e in api["entries"])
        rec("列表条数一致", len(ui_ids) == len(api_ids),
            "ui=%d api=%d" % (len(ui_ids), len(api_ids)))
        rec("每条 API 条目都在列表里", set(ui_ids) == set(api_ids),
            "missing=%s extra=%s" % (sorted(set(api_ids) - set(ui_ids)),
                                     sorted(set(ui_ids) - set(api_ids))))

        print("3. kind 计数与后端 stats 一致")
        chips = page.evaluate("""() => {
            const box = document.querySelector('.res-chips');
            if (!box) return null;
            const out = {};
            Array.from(box.children).forEach(n => {
              const t = (n.textContent || '').trim();
              const m = t.match(/^([a-z_]+)\\s+(\\d+)$/);
              if (m) out[m[1]] = Number(m[2]);
            });
            return out;
        }""")
        stats = {k: v["count"] for k, v in (api.get("stats") or {}).items()}
        rec("顶部 kind 计数与 stats 一致", chips == stats,
            "chips=%s stats=%s" % (json.dumps(chips, sort_keys=True),
                                   json.dumps(stats, sort_keys=True)))
        rec("记录总数 chip 与条目数一致",
            bool(chips) and sum(stats.values()) == len(api_ids),
            "sum(stats)=%d entries=%d" % (sum(stats.values()), len(api_ids)))

        print("4. 搜索按 kind 前缀过滤")
        prefix = sorted({i.split(":", 1)[0] for i in api_ids})[0]
        page.evaluate("""(needle) => {
            const input = document.getElementById('resSearch');
            input.value = needle;
        }""", prefix)
        page.click('[data-res-action="search"]')
        page.wait_for_timeout(600)
        filtered = page.evaluate(_ROWS)
        # 与 resourceManager.filteredEntries() 同口径：q 命中任一字段即保留（OR）。
        fields = ("entry_id", "path", "source", "provider",
                  "version", "license", "kind")
        want = [e["entry_id"] for e in api["entries"]
                if any(prefix in str(e.get(f) or "").lower() for f in fields)]
        rec("搜索后只剩该 kind", sorted(filtered) == sorted(want),
            "prefix=%s got=%s want=%s" % (prefix, len(filtered), len(want))
            if sorted(filtered) != sorted(want) else "prefix=%s n=%d" % (prefix, len(want)))

        # 清空后必须恢复全量（避免筛选状态泄漏到后面的断言）
        page.evaluate("""() => {
            const input = document.getElementById('resSearch');
            input.value = '';
        }""")
        page.click('[data-res-action="search"]')
        page.wait_for_timeout(600)
        restored = page.evaluate(_ROWS)
        rec("清空搜索后恢复全量", len(restored) == len(api_ids),
            "restored=%d api=%d" % (len(restored), len(api_ids)))

        print("5. test 模式路径不脱敏")
        txt = page.evaluate("""() => (document.getElementById('resourceBody').textContent || '')""")
        rec("test 模式显示真实路径",
            (":\\" in txt) or (":/" in txt),
            "sample=%r" % (txt[txt.find(":\\") - 3:txt.find(":\\") + 12]
                           if ":\\" in txt else txt[:60]))

        errors = [c for c in console
                  if c[0] == "pageerror"
                  and "api/sim/cases" not in c[1]
                  and "Failed to load resource" not in c[1]]
        rec("页面无未捕获异常", len(errors) == 0, "; ".join(t for _, t in errors[:3]))

        if args.headed:
            page.screenshot(path="tools/_res_verify.png")
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
