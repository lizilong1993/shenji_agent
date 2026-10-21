"""Disk cache shared by the online and offline data paths.

Online sources write here; offline builds read the same keys.  ``mode``
controls behaviour when a key is missing: ``online`` fetches, ``offline``
raises ``OfflineCacheMiss``, ``auto`` fetches and falls back to the caller.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class OfflineCacheMiss(FileNotFoundError):
    pass


class CacheStore(object):
    def __init__(self, root: str, mode: str = "auto"):
        if mode not in ("online", "offline", "auto"):
            raise ValueError("mode must be online/offline/auto")
        self.root = Path(root)
        self.mode = mode
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, kind: str, name: str) -> Path:
        safe = hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
        return self.root / kind / safe

    def has(self, kind: str, name: str) -> bool:
        return self.key(kind, name).is_file()

    def read_bytes(self, kind: str, name: str):
        p = self.key(kind, name)
        if not p.is_file():
            raise OfflineCacheMiss("cache miss: %s/%s" % (kind, name))
        return p.read_bytes()

    def write_bytes(self, kind: str, name: str, data: bytes) -> None:
        p = self.key(kind, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    def read_json(self, kind: str, name: str):
        return json.loads(self.read_bytes(kind, name).decode("utf-8"))

    def write_json(self, kind: str, name: str, obj) -> None:
        self.write_bytes(kind, name,
                         json.dumps(obj, ensure_ascii=False).encode("utf-8"))
