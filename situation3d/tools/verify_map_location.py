"""Offline Chromium smoke test for the map locate action on civilian place cards.

Run from situation3d: python tools/verify_map_location.py
Use --executable-path for a cached Chromium and --cesium-dir for a local
Cesium distribution containing Cesium.js, Widgets, Assets and Workers.
"""

import argparse
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

from playwright.sync_api import Error, expect, sync_playwright


WEB = Path(__file__).resolve().parents[1] / "web"
ORIGIN = "http://map-location.test"
CESIUM_PREFIX = "/vendor/cesium/"
SCRIPTS = {"vendor/cesium/Cesium.js", "src/theme.js", "src/intelUi.js", "src/app.js"}

# Input fixture and assertions below are deliberately independent.
OVERVIEW = {"items": [
    {"id": "library", "title": "测试社区图书馆", "type": "geography", "domain": "geography",
     "priority": "auto", "effective": True, "lon": -73.414, "lat": 41.327,
     "geo_uncertainty_m": 0, "is_centroid": False},
    {"id": "park", "title": "测试市民公园", "type": "geography", "domain": "geography",
     "priority": "auto", "effective": True, "lon": 12.684, "lat": 51.231,
     "geo_uncertainty_m": 3200, "is_centroid": True},
]}

BOOTSTRAP = """
viewer = new Cesium.Viewer('cesiumContainer', {
  baseLayer: false, baseLayerPicker: false, geocoder: false,
  skyBox: false, skyAtmosphere: false, animation: false, timeline: false,
  useDefaultRenderLoop: false
});
viewer.scene.rethrowRenderErrors = true;
state.caseId = 'default';
bindMenu('menuData', 'menuDataList');
IntelPanel.bind();
document.getElementById('btnIntel').addEventListener('click', () => closeAllMenus());
"""

SNAPSHOT = """
([lon, lat]) => {
  const map = SituationHost.viewer();
  map.resize();
  map.render();
  const point = Cesium.Cartesian3.fromDegrees(lon, lat, 0);
  const px = Cesium.SceneTransforms.worldToWindowCoordinates(map.scene, point);
  const time = map.clock.currentTime;
  return {
    size: [map.canvas.clientWidth, map.canvas.clientHeight],
    projected: px ? [px.x, px.y] : null,
    circles: map.entities.values.filter(e => e.ellipse).map(e => {
      const carto = Cesium.Cartographic.fromCartesian(e.position.getValue(time));
      return {
        position: [Cesium.Math.toDegrees(carto.longitude),
                   Cesium.Math.toDegrees(carto.latitude), carto.height],
        major: e.ellipse.semiMajorAxis?.getValue(time),
        minor: e.ellipse.semiMinorAxis?.getValue(time),
        shown: e.isShowing && e.ellipse.show?.getValue(time) !== false,
        label: e.label?.text?.getValue(time)
      };
    })
  };
}
"""


def require(condition, detail):
    if not condition:
        raise AssertionError(detail)


def install_routes(context, cesium, errors):
    require((cesium / "Cesium.js").is_file(), "Missing local Cesium.js in {}".format(cesium))
    html = (WEB / "index.html").read_text(encoding="utf-8")
    html = re.sub(r'<script src="\./([^"\n]+)"></script>',
                  lambda tag: tag[0] if tag[1] in SCRIPTS else "", html)
    source = (WEB / "src/app.js").read_text(encoding="utf-8")
    # shortcut: tied to the current main() entry point, update this seam if startup moves.
    source, count = re.subn(r"(?m)^main\(\);[ \t]*$", BOOTSTRAP, source)
    require(count == 1, "Expected one standalone main(); in src/app.js")

    def fulfill(route):
        url = route.request.url
        path = urlsplit(url).path
        if route.request.method == "GET" and url.startswith(ORIGIN + "/"):
            if path == "/":
                return route.fulfill(body=html, content_type="text/html; charset=utf-8")
            if path == "/api/intel/overview":
                return route.fulfill(json=OVERVIEW)
            if path == "/src/app.js":
                return route.fulfill(body=source, content_type="text/javascript; charset=utf-8")
            if path == "/favicon.ico":
                return route.fulfill(status=204)
            if path.startswith(CESIUM_PREFIX):
                root, relative = cesium, path[len(CESIUM_PREFIX):]
            elif path[1:] in SCRIPTS or (path.startswith("/src/") and path.endswith(".css")):
                root, relative = WEB, path[1:]
            else:
                root, relative = None, None
            if root is not None:
                file = (root / relative).resolve()
                if root in file.parents and file.is_file():
                    return route.fulfill(path=file)
        errors.append("Unserved request: {} {}".format(route.request.method, url))
        route.abort()

    context.route("**/*", fulfill)


def locate(page, card, lon, lat):
    card.locator(".intel-locate").click()
    result = page.evaluate(SNAPSHOT, [lon, lat])
    width, height = result["size"]
    px = result["projected"]
    require(width > 0 and height > 0 and px is not None and
            math.hypot(px[0] - width / 2, px[1] - height / 2) <= 2,
            "({}, {}) did not project within 2 CSS px of canvas center: {} on {}".format(
                lon, lat, px, result["size"]))
    return result["circles"]


def verify(page):
    page.goto(ORIGIN, wait_until="load")
    page.locator("#cesiumContainer canvas").wait_for()
    page.locator("#menuData").click()
    page.locator("#btnIntel").click()
    cards = page.locator("#intelModal .intel-card")
    exact = cards.filter(has=page.get_by_text("测试社区图书馆", exact=True))
    centroid = cards.filter(has=page.get_by_text("测试市民公园", exact=True))
    require(locate(page, exact, -73.414, 41.327) == [], "Zero radius created a circle")

    expect(centroid.locator(".intel-geo")).to_have_text("定位 ±3.2 km（质心）")
    circles = locate(page, centroid, 12.684, 51.231)
    require(len(circles) == 1, "Expected one centroid circle, got {!r}".format(circles))
    circle = circles[0]
    lon, lat, altitude = circle["position"]
    require(abs(lon - 12.684) <= 1e-6 and abs(lat - 51.231) <= 1e-6 and abs(altitude) <= 0.01,
            "Wrong circle position: {!r}".format(circle["position"]))
    require(circle.get("major") == 3200 and circle.get("minor") == 3200 and circle["shown"],
            "Wrong circle axes or visibility: {!r}".format(circle))
    require(circle.get("label") == "测试市民公园", "Wrong map label: {!r}".format(circle.get("label")))
    require(locate(page, exact, -73.414, 41.327) == [], "Zero radius left the old circle on the map")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--executable-path", type=Path, help="Cached Chromium executable")
    parser.add_argument("--cesium-dir", type=Path, default=WEB / "vendor/cesium")
    args = parser.parse_args()
    errors = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=args.executable_path,
                args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                      "--host-resolver-rules=MAP * ~NOTFOUND"],
            )
            try:
                context = browser.new_context(viewport={"width": 1280, "height": 800},
                                              device_scale_factor=1, service_workers="block")
                install_routes(context, args.cesium_dir.resolve(), errors)
                page = context.new_page()
                page.set_default_timeout(8000)
                page.on("pageerror", lambda error: errors.append("Page error: " + str(error)))
                verify(page)
            finally:
                browser.close()
        require(not errors, "Browser reported errors")
    except (AssertionError, Error, OSError) as error:
        print("FAIL: {}".format(error), file=sys.stderr)
        for detail in errors:
            print(detail, file=sys.stderr)
        return 1
    print("PASS: map locate center, zero-radius clearing, centroid circle and labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
