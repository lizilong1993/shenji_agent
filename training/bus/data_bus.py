"""数据总线：历史态势序列的持久化、压缩与分片（I19 步骤 2），并复用 I15 的复盘格式。

31006 4.4.1 对数据总线要求：持久化传输、历史态势序列、多模态复盘数据、压缩与分片。
本实现把「历史态势序列」的**最小可用**形态定死为：

    每个实例一个目录；序列按 **分片** 落盘；每片是 **zstd/gzip 压缩的 JSONL**；
    并写一份 `manifest.json` 记录每片的起止 tick、条目数与校验和。

为什么分片而不是单文件
----------------------
I15 已把「复盘文件 <=100MB（建议）」写进 schema，而 I18 对超限是**拒绝**。
一份长对局的态势序列很容易超过该阈值，所以数据总线**从第一天就分片**，
避免将来为了过闸而重构存储。

为什么压缩用 gzip 而不是 zstd
------------------------------
远程环境的解释器版本与可用模块不确定；`gzip` 属标准库、**零依赖**，
符合 I19「最小可用、旁路、可关闭」的定位。压缩算法属可替换细节，
`manifest.json` 里记了 `compression` 字段，将来换 zstd 不会改变契约。
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

MANIFEST_NAME = "manifest.json"
SCHEMA_ID = "shenji.bus.data/1"
DEFAULT_SHARD_BYTES = 8 * 1024 * 1024      # 每片原始字节上限，远低于 I15 的 100MB 建议值


class DataBus:
    """按实例分目录的历史态势序列存储。"""

    def __init__(self, root: Path, instance_id: str, *,
                 enabled: bool = True, shard_bytes: int = DEFAULT_SHARD_BYTES,
                 compression: str = "gzip"):
        self.root = Path(root) / instance_id
        self.instance_id = instance_id
        self.enabled = enabled
        self.shard_bytes = int(shard_bytes)
        self.compression = compression
        self._seq = 0
        self._shard_idx = 0
        self._shard_raw_bytes = 0
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            self._load_state()

    # ------------------------------------------------------------------ 内部
    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def _load_state(self) -> None:
        m = self._read_manifest()
        if m:
            self._seq = int(m.get("total_entries", 0))
            self._shard_idx = len(m.get("shards", []))

    def _read_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def _shard_path(self, idx: int) -> Path:
        return self.root / ("shard-%05d.jsonl.gz" % idx)

    def _write_manifest(self, shards: List[Dict[str, Any]]) -> None:
        payload = {
            "schema": SCHEMA_ID,
            "instance_id": self.instance_id,
            "compression": self.compression,
            "shard_bytes_limit": self.shard_bytes,
            "total_entries": sum(int(s["entries"]) for s in shards),
            "shards": shards,
        }
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    # ------------------------------------------------------------------ 写入
    def append(self, tick: int, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """追加一条历史态势。达到分片阈值则自动开新片。

        返回本条所在分片的信息（`enabled=False` 时返回 None —— 这就是「可关闭」）。
        """
        if not self.enabled:
            return None
        rec = {"schema": SCHEMA_ID, "instance_id": self.instance_id,
               "seq": self._seq, "tick": tick, "state": state}
        line = (json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")

        if self._shard_raw_bytes and self._shard_raw_bytes + len(line) > self.shard_bytes:
            self._shard_idx += 1
            self._shard_raw_bytes = 0

        path = self._shard_path(self._shard_idx)
        with gzip.open(path, "ab") as fh:
            fh.write(line)
        self._shard_raw_bytes += len(line)
        self._seq += 1

        self._refresh_manifest()
        return {"shard": path.name, "seq": self._seq - 1}

    def _refresh_manifest(self) -> None:
        shards: List[Dict[str, Any]] = []
        for p in sorted(self.root.glob("shard-*.jsonl.gz")):
            entries = 0
            first_tick: Optional[int] = None
            last_tick: Optional[int] = None
            raw = 0
            with gzip.open(p, "rb") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    entries += 1
                    raw += len(line)
                    try:
                        t = json.loads(line.decode("utf-8")).get("tick")
                    except Exception:  # noqa: BLE001
                        t = None
                    if isinstance(t, int):
                        if first_tick is None:
                            first_tick = t
                        last_tick = t
            shards.append({
                "name": p.name, "entries": entries,
                "tick_first": first_tick, "tick_last": last_tick,
                "raw_bytes": raw, "file_bytes": p.stat().st_size,
                "sha256": _sha256(p),
            })
        self._write_manifest(shards)

    # ------------------------------------------------------------------ 读取
    def read_shard(self, name: str) -> Iterator[Dict[str, Any]]:
        p = self.root / name
        with gzip.open(p, "rb") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line.decode("utf-8"))

    def iter_states(self) -> Iterator[Dict[str, Any]]:
        for s in self._read_manifest().get("shards", []):
            yield from self.read_shard(s["name"])

    def state_at_tick(self, tick: int) -> Optional[Dict[str, Any]]:
        """按 tick 取态：先用 manifest 的 tick 区间**跳过无关分片**，再片内线性查。"""
        for s in self._read_manifest().get("shards", []):
            f, l = s.get("tick_first"), s.get("tick_last")
            if isinstance(f, int) and isinstance(l, int) and not (f <= tick <= l):
                continue
            for rec in self.read_shard(s["name"]):
                if rec.get("tick") == tick:
                    return rec
        return None

    def manifest(self) -> Dict[str, Any]:
        return self._read_manifest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
