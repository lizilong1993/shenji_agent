#!/usr/bin/env python3
# Fetch Sketchfab downloadable GLB archives with a personal API token.
# Token is read from /tmp/sketchfab_token (chmod 600) or SKETCHFAB_TOKEN.
# Outputs to /tmp/sketchfab_src/<name>.glb (or .zip if the archive is zip).

import json
import os
import sys
import time
import urllib.request
import zipfile

MODELS = {
    "m777": "2a2c381888ef4e4790ed883cddda0aca",
    "m60a3": "3d298af87c2c4a0691e61ab8ee9ad803",
    "ah64d": "c3b58008c46b45048fdd7dd283a3c8c8",
    "himars": "95d255cd685e4cc5afc35ba5f58679d4",
    "type055": "f0a96b2475da407eb6ea2b430ba9ea85",
    "j20": "735da22f59d04542b5b33e5b8fd699b1",
    "p3c": "be11ab6c970741d78e62c9d8ff2e33e8",
    "pj26gun": "0c88bdaddd104020bcd1c978e0dbb3e5",
}

TOKEN = os.environ.get("SKETCHFAB_TOKEN")
if not TOKEN and os.path.exists("/tmp/sketchfab_token"):
    TOKEN = open("/tmp/sketchfab_token").read().strip()
if not TOKEN:
    sys.exit("missing token")

OUT = "/tmp/sketchfab_src"
os.makedirs(OUT, exist_ok=True)


def get(url, auth=True):
    headers = {"Authorization": "Token " + TOKEN} if auth else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    only = set(sys.argv[1:])
    for name, uid in MODELS.items():
        if only and name not in only:
            continue
        try:
            meta = json.loads(get("https://api.sketchfab.com/v3/models/" + uid))
            print("%-8s %-58s license=%s faces=%s" % (
                name, meta.get("name", "")[:58], (meta.get("license") or {}).get("label"), meta.get("faceCount")))
            dl = json.loads(get("https://api.sketchfab.com/v3/models/%s/download" % uid))
            url = (dl.get("glb") or {}).get("url")
            if not url:
                print("  no glb url", list(dl.keys()))
                continue
            data = get(url, auth=False)
            if data[:2] == b"PK":
                zpath = os.path.join(OUT, name + ".zip")
                open(zpath, "wb").write(data)
                with zipfile.ZipFile(zpath) as z:
                    members = [m for m in z.namelist() if m.lower().endswith(".glb")]
                    if not members:
                        print("  zip has no glb:", z.namelist()[:5])
                        continue
                    z.extract(members[0], OUT)
                    src = os.path.join(OUT, members[0])
                    dst = os.path.join(OUT, name + ".glb")
                    os.replace(src, dst)
                print("  saved", dst, os.path.getsize(dst))
            else:
                dst = os.path.join(OUT, name + ".glb")
                open(dst, "wb").write(data)
                print("  saved", dst, len(data), "magic", data[:4])
            time.sleep(1)
        except Exception as e:
            print("FAIL", name, type(e).__name__, getattr(e, "code", ""), str(e)[:200])


if __name__ == "__main__":
    main()
