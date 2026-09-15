#!/bin/sh
# Fetch the pinned CesiumJS build into web/vendor/cesium.
#
# CesiumJS is 45 MB unpacked, so it is not committed -- this script pins the
# exact version and source instead. Run it once after cloning, and again only
# when CESIUM_VERSION changes.
#
# Source: the npm registry tarball, which is reachable from the remote container
# (GitHub release assets are not).
#
# Usage:  sh fetch_cesium.sh [target-dir]     (default: ../web/vendor/cesium)

set -e

CESIUM_VERSION=1.145.0
HERE=$(cd "$(dirname "$0")" && pwd)
DEST="${1:-$HERE/../web/vendor/cesium}"

if [ -f "$DEST/Cesium.js" ]; then
  echo "already present: $DEST/Cesium.js"
  exit 0
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "downloading cesium@$CESIUM_VERSION ..."
curl -fsSL "https://registry.npmjs.org/cesium/-/cesium-${CESIUM_VERSION}.tgz" -o "$WORK/cesium.tgz"

echo "unpacking ..."
tar -xzf "$WORK/cesium.tgz" -C "$WORK"

# The npm package ships the prebuilt browser bundle under Build/Cesium.
mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
cp -r "$WORK/package/Build/Cesium" "$DEST"

echo "installed $DEST"
echo "  files : $(find "$DEST" -type f | wc -l)"
echo "  size  : $(du -sh "$DEST" | cut -f1)"
[ -f "$DEST/Cesium.js" ] || { echo "ERROR: Cesium.js missing"; exit 1; }
[ -d "$DEST/Assets/Textures/NaturalEarthII" ] || { echo "ERROR: offline basemap missing"; exit 1; }
echo "fetch_cesium OK"
