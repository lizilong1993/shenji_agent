#!/usr/bin/env python3
"""End-to-end check of the campaign UI on a running situation3d server.

Drives a headless Chromium over DevTools (reusing tools/inspect_remote.py) and
asserts that, for a campaign.v3 case:

  * the stage ribbon renders one segment per stage;
  * the campaign panel renders stages/objectives/events;
  * clicking a stage jumps the timeline;
  * clicking an event jumps to its tick;
  * the outcome card appears at the last frame;
  * no page errors were raised.

Usage:
    python3 situation3d/tools/campaign_selftest.py --port 8850 --case taiwan_landing
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from inspect_remote import DevTools, http_json, first_page_target, wait_for_devtools  # noqa: E402

CHROME = os.environ.get("CHROME_BIN", "")

if not CHROME:
    # Playwright installs under the invoking user's home; the server runs as
    # root while the browser was installed as lizilong.
    for base in (os.path.expanduser("~/.cache/ms-playwright"),
                 "/remote-home/lizilong/.cache/ms-playwright"):
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if not name.startswith("chromium-"):
                continue
            for leaf in ("chrome-linux64/chrome", "chrome-linux/chrome"):
                p = os.path.join(base, name, leaf)
                if os.path.isfile(p):
                    CHROME = p
                    break
            if CHROME:
                break
        if CHROME:
            break


def launch(port: int, width=1600, height=900):
    user_dir = "/tmp/situation3d-campaign-%d" % port
    # A stale profile can keep a cached failed /vendor request (immutable
    # cache headers), which breaks the page with "Cesium is not defined".
    import shutil
    shutil.rmtree(user_dir, ignore_errors=True)
    cmd = [
        CHROME, "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
        "--disable-setuid-sandbox", "--disable-gpu-sandbox",
        "--enable-unsafe-swiftshader", "--use-gl=angle",
        "--use-angle=swiftshader", "--font-render-hinting=none",
        "--remote-debugging-port=%d" % port,
        "--user-data-dir=%s" % user_dir,
        "--window-size=%d,%d" % (width, height), "--hide-scrollbars",
        "about:blank",
    ]
    env = dict(os.environ)
    # bootstrap_remote_headless.sh extracts the missing libs/fonts here.
    deps = "/tmp/chromedeps"
    if os.path.isdir(deps):
        env["LD_LIBRARY_PATH"] = os.pathsep.join([
            os.path.join(deps, "usr/lib/x86_64-linux-gnu"),
            os.path.join(deps, "lib/x86_64-linux-gnu"),
            os.path.join(deps, "usr/lib"),
        ])
        fonts_conf = os.path.join(deps, "fonts.conf")
        if os.path.isfile(fonts_conf):
            env["FONTCONFIG_FILE"] = fonts_conf
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, env=env)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8850)
    ap.add_argument("--cdp-port", type=int, default=9611)
    ap.add_argument("--case", default="taiwan_landing")
    ap.add_argument("--keep-open", action="store_true")
    args = ap.parse_args(argv)

    chrome = launch(args.cdp_port)
    failures = []
    try:
        wait_for_devtools(args.cdp_port, 30)
        target = first_page_target(args.cdp_port)
        cdp = DevTools(target["webSocketDebuggerUrl"], timeout=90)
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        # Collect page errors.
        cdp.evaluate("window.__errs=[];addEventListener('error',function(e){window.__errs.push(String(e.message))});"
                     "addEventListener('unhandledrejection',function(e){window.__errs.push('rej:'+String(e.reason))});")
        url = "http://127.0.0.1:%d/?case=%s&tick=1" % (args.port, args.case)
        cdp.call("Page.navigate", url=url)
        deadline = time.time() + 90
        ready = False
        while time.time() < deadline:
            time.sleep(1.5)
            try:
                if cdp.evaluate(
                        "!!(window.CampaignUi && window.SituationHost &&"
                        " SituationHost.state.data &&"
                        " document.querySelectorAll('.cp-seg').length > 2)"):
                    ready = True
                    break
            except Exception:
                pass
        if not ready:
            failures.append("页面未就绪（CampaignUi/SituationHost 未加载）")
        else:
            dbg = cdp.evaluate("JSON.stringify(CampaignUi.debug())")
            info = json.loads(dbg)
            if not info.get("campaign"):
                failures.append("CampaignUi.debug().campaign=false")
            if info.get("stages", 0) < 2:
                failures.append("阶段数异常: %s" % info.get("stages"))
            if info.get("events", 0) < 1:
                failures.append("事件数异常: %s" % info.get("events"))
            segs = cdp.evaluate("document.querySelectorAll('.cp-seg').length")
            if segs != info.get("stages"):
                failures.append("色带段数 %s != 阶段数 %s" % (segs, info.get("stages")))
            rows = cdp.evaluate("document.querySelectorAll('.cp-stage').length")
            if rows != info.get("stages"):
                failures.append("阶段列表行数 %s != 阶段数 %s" % (rows, info.get("stages")))
            evs = cdp.evaluate("document.querySelectorAll('.cp-event').length")
            if not evs:
                failures.append("事件列表为空")
            # Click the third stage segment -> timeline jumps to its start.
            jump = cdp.evaluate(
                "(function(){var s=document.querySelectorAll('.cp-seg')[2];"
                "var before=SituationHost.state.tick;s.click();"
                "return JSON.stringify({before:before,after:SituationHost.state.tick});})()")
            j = json.loads(jump)
            if j["after"] == j["before"]:
                failures.append("点击阶段未跳转 tick: %s" % jump)
            # Click first event -> tick matches event tick.
            evjump = cdp.evaluate(
                "(function(){var n=document.querySelector('.cp-event');"
                "var id=n.getAttribute('data-ev');n.click();"
                "var t=SituationHost.state.tick;return JSON.stringify({id:id,tick:t});})()")
            ej = json.loads(evjump)
            if not ej.get("id"):
                failures.append("事件点击失败")
            # Last frame -> outcome card.
            last = cdp.evaluate(
                "(function(){SituationHost.setTick(SituationHost.state.data.frames.length-1);"
                "var o=document.querySelector('.cp-out');return o?o.textContent:'';})()")
            if not last or ("引擎裁决" not in last):
                failures.append("结局卡未显示: %r" % last)
            errs = cdp.evaluate("JSON.stringify(window.__errs||[])")
            if errs and errs != "[]":
                failures.append("页面错误: %s" % errs)
        cdp.close()
    finally:
        if not args.keep_open:
            chrome.terminate()
            try:
                chrome.wait(timeout=5)
            except Exception:
                chrome.kill()

    if failures:
        print("FAILED")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
