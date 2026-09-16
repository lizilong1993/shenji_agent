#!/usr/bin/env bash
# Sync the local situation3d working copy to the remote container, then verify.
#
# The remote repository lives on NFS and is slow for many small operations, so
# this pushes a single tar stream rather than walking the tree twice. Generated
# and third-party content (vendor/, data/, tiles/, shots/) is deliberately not
# synced: those are rebuilt on the remote by the fetch and generate scripts.
#
# Usage:
#   tools/sync_remote.sh                    # default set (code only)
#   tools/sync_remote.sh engine web/src     # explicit subset
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
REMOTE="${SITUATION3D_REMOTE:-lizilong@100.113.178.69}"
DEST="${SITUATION3D_DEST:-/remote-home/lizilong/shenji_agent/situation3d}"

SSH_OPTS=(-o ConnectTimeout=20 -o BatchMode=yes -o StrictHostKeyChecking=no
          -o UserKnownHostsFile=/dev/null)

paths=("$@")
if [ ${#paths[@]} -eq 0 ]; then
  paths=(engine scripts server tools web/src web/index.html)
fi

# `sync web` and `sync web/src` should both bring index.html along.
for p in "${paths[@]}"; do
  case "$p" in
    web|web/src)
      case " ${paths[*]} " in
        *" web/index.html "*) ;;
        *) paths+=(web/index.html) ;;
      esac
      ;;
  esac
done

for p in "${paths[@]}"; do
  if [ ! -e "$SRC/$p" ]; then
    echo "[sync] ERROR: no such path in source: $p" >&2
    exit 2
  fi
done

cd "$SRC"
echo "[sync] $SRC -> $REMOTE:$DEST"
printf '       %s\n' "${paths[@]}"

tar czf - --exclude='__pycache__' --exclude='*.pyc' --exclude='tools/shots' "${paths[@]}" \
  | ssh "${SSH_OPTS[@]}" "$REMOTE" \
      "mkdir -p '$DEST' && tar xzf - -C '$DEST' && echo '[sync] pushed'"

# Checksums read every file on both sides. On the NFS-backed remote that is the
# slow part of the round trip, so it is opt-in: SITUATION3D_VERIFY=1.
if [ "${SITUATION3D_VERIFY:-0}" != "1" ]; then
  exit 0
fi

FIND_EXPR=(-type f \( -name '*.py' -o -name '*.js' -o -name '*.html' -o -name '*.sh' -o -name '*.css' \))

echo "[sync] local  checksums:"
find "${paths[@]}" "${FIND_EXPR[@]}" -print0 | sort -z | xargs -0 md5sum | sed 's/^/  /'

echo "[sync] remote checksums:"
REMOTE_CMD="cd '$DEST' && find $(printf '%q ' "${paths[@]}") -type f \\( -name '*.py' -o -name '*.js' -o -name '*.html' -o -name '*.sh' -o -name '*.css' \\) -print0 | sort -z | xargs -0 md5sum | sed 's/^/  /'"
ssh "${SSH_OPTS[@]}" "$REMOTE" "$REMOTE_CMD"
