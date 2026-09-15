#!/bin/sh
# Render one screenshot with headless Chromium. Runs ON the remote container.
#
# Prerequisite: bootstrap_remote_headless.sh has been run (provides /tmp/chromedeps).
#
# Usage: sh remote_shot.sh <url> <out.png> [virtual-time-budget-ms] [width] [height]
#
# Note on --virtual-time-budget: it is the only way to make plain `--screenshot`
# wait, but it is NOT a reliable readiness signal -- the same budget can yield a
# fully rendered globe on one run and an unrendered one on the next, because tile
# decode and SwiftShader rasterisation consume real time while the budget counts
# virtual time. Callers must validate the PNG and retry; see capture.py.

DEPS=/tmp/chromedeps
CH=/remote-home/lizilong/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome
URL="$1"
OUT="$2"
VTB="${3:-30000}"
W="${4:-1600}"
H="${5:-900}"

if [ -z "$URL" ] || [ -z "$OUT" ]; then
  echo "usage: sh remote_shot.sh <url> <out.png> [vtb_ms] [width] [height]" >&2
  exit 2
fi

if [ ! -d "$DEPS/usr/lib/x86_64-linux-gnu" ]; then
  echo "ERROR: $DEPS missing -- run bootstrap_remote_headless.sh first" >&2
  exit 3
fi

LD_LIBRARY_PATH="$DEPS/usr/lib/x86_64-linux-gnu:$DEPS/lib/x86_64-linux-gnu:$DEPS/usr/lib"
FONTCONFIG_FILE="$DEPS/fonts.conf"
FONTCONFIG_PATH="$DEPS"
HOME=/remote-home/lizilong
XDG_CONFIG_HOME=/tmp/chrome_cfg
XDG_CACHE_HOME=/tmp/chrome_cache
export LD_LIBRARY_PATH FONTCONFIG_FILE FONTCONFIG_PATH HOME XDG_CONFIG_HOME XDG_CACHE_HOME
mkdir -p "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" /tmp/fontcache

rm -f "$OUT"
timeout 300 "$CH" \
  --headless=new --no-sandbox --disable-dev-shm-usage \
  --disable-setuid-sandbox --disable-gpu-sandbox \
  --enable-unsafe-swiftshader --use-gl=angle --use-angle=swiftshader \
  --font-render-hinting=none \
  --window-size="$W,$H" --hide-scrollbars \
  --virtual-time-budget="$VTB" \
  --screenshot="$OUT" "$URL" >/dev/null 2>&1

if [ -f "$OUT" ]; then
  echo "shot OK $(stat -c %s "$OUT") bytes"
  exit 0
fi
echo "shot FAILED (no PNG produced)" >&2
exit 1
