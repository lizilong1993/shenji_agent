#!/bin/sh
# One-time bootstrap: make headless Chromium usable on the remote container.
#
# Background: the SSH landing container has no root, no system Chromium, and a
# bare base image (no shared libs, no fonts). Playwright's Chromium binaries are
# already cached under ~/.cache/ms-playwright, so we only need to supply the
# missing shared libraries and fonts, extracted into user space.
#
# Layout produced:
#   /remote-home/lizilong/.local/chromedeps/deb   <- downloaded .deb files (persistent)
#   /tmp/chromedeps                               <- extracted libs + fonts (ephemeral)
#
# /tmp is container-local disk: extraction there is fast, but it is wiped when
# the container restarts. Re-run this script to rebuild it (deb files are kept
# on NFS, so the second run only re-extracts and takes well under a minute).
#
# Usage:  sh bootstrap_remote_headless.sh

set -e

DEB_DIR=/remote-home/lizilong/.local/chromedeps/deb
DST=/tmp/chromedeps

# --- shared libraries ------------------------------------------------------
# Direct deps of chrome plus their transitive deps. Names are the Debian
# packages, which do not always match the soname (libpixman-1-0 provides
# libpixman-1.so.0, libharfbuzz0b provides libharfbuzz.so.0, ...), so they are
# listed explicitly rather than derived.
LIBS="libnspr4 libnss3 libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0
libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0
libxkbcommon-x11-0 libx11-6 libx11-xcb1 libxcb1 libxcb-render0 libxcb-shm0
libxext6 libxi6 libxtst6 libxrender1 libxau6 libxdmcp6 libxshmfence1
libcups2 libdbus-1-3 libexpat1 libdrm2 libegl1 libgl1 libglx0 libglvnd0
libgles2 libopengl0 libpango-1.0-0 libcairo2 libpixman-1-0 libpng16-16
libfontconfig1 libfreetype6 libfribidi0 libgraphite2-3 libharfbuzz0b
libthai0 libdatrie1 libbrotli1 libwebp6 libavahi-client3 libavahi-common3
libwayland-server0 libwayland-client0 libpcre2-8-0 libuuid1 libbsd0 libmd0
libbs2b0 libsndfile1 libflac8 libvorbis0a libvorbisenc2 libogg0 libmpg123-0
libasyncns0 libsystemd0 liblz4-1 libgcrypt20 libgpg-error0 libcap2 libudev1"

# --- fonts -----------------------------------------------------------------
# Latin + CJK. Without any font, Skia aborts the renderer
# (SkFontMgr_FontConfigInterface.cpp: "Not implemented"), and without a CJK
# font every Chinese label renders as tofu boxes.
FONTS="fonts-dejavu-core fonts-liberation2 fonts-wqy-zenhei"

mkdir -p "$DEB_DIR" "$DST"
cd "$DEB_DIR"

echo "=== fetching packages ==="
for p in $LIBS $FONTS; do
  if ls ${p}_*.deb >/dev/null 2>&1; then continue; fi
  apt-get download "$p" >/dev/null 2>&1 || echo "  skip $p"
done

echo "=== extracting to $DST ==="
rm -rf "$DST"
mkdir -p "$DST" /tmp/fontcache
for f in *.deb; do
  dpkg-deb -x "$f" "$DST" 2>/dev/null || true
done

# fontconfig config pointing only at the extracted font directories
cat > "$DST/fonts.conf" <<'XML'
<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <dir>/tmp/chromedeps/usr/share/fonts</dir>
  <dir>/tmp/chromedeps/usr/local/share/fonts</dir>
  <cachedir>/tmp/fontcache</cachedir>
  <match target="pattern">
    <test qual="any" name="family"><string>sans-serif</string></test>
    <edit name="family" mode="assign" binding="strong"><string>DejaVu Sans</string></edit>
  </match>
</fontconfig>
XML

echo "=== verify ==="
CH=/remote-home/lizilong/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome
if [ ! -x "$CH" ]; then
  echo "ERROR: chromium not found at $CH"
  echo "Playwright browser cache is missing; install it first."
  exit 1
fi
LD_LIBRARY_PATH="$DST/usr/lib/x86_64-linux-gnu:$DST/lib/x86_64-linux-gnu:$DST/usr/lib" \
  ldd "$CH" 2>&1 | grep "not found" && echo "ERROR: unresolved shared libraries above" && exit 1

echo "libs     : $(find "$DST" -name '*.so*' | wc -l) files"
echo "fonts    : $(find "$DST" -name '*.tt[cf]' -o -name '*.otf' | wc -l) files"
LD_LIBRARY_PATH="$DST/usr/lib/x86_64-linux-gnu:$DST/lib/x86_64-linux-gnu:$DST/usr/lib" \
  "$CH" --version
echo "bootstrap OK"
