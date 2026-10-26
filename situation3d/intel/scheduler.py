"""Deterministic per-side intel scheduling and run logging.

The gateway decides what a side may know; this scheduler decides when it is
asked.  Each tick calls ``consume`` for every side with a fixed lookahead and
watermark lag, and records the batch hashes so a run manifest can replay the
same intel stream.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .gateway import CueBatch, IntelGateway


class IntelScheduler:
    def __init__(self, gateway: IntelGateway, sides: Iterable[int] = (0, 1),
                 lookahead_s: float = 300.0,
                 watermark_lag_s: float = 0.0):
        self.gateway = gateway
        self.sides = tuple(sorted(int(side) for side in sides))
        self.lookahead_s = max(0.0, float(lookahead_s))
        self.watermark_lag_s = max(0.0, float(watermark_lag_s))
        self.batches: List[dict] = []

    def advance(self, sim_time: float,
                lookahead_s: Optional[float] = None) -> Dict[int, CueBatch]:
        sim_time = float(sim_time)
        lookahead = (self.lookahead_s if lookahead_s is None
                     else max(0.0, float(lookahead_s)))
        watermark = max(0.0, sim_time - self.watermark_lag_s)
        out: Dict[int, CueBatch] = {}
        for side in self.sides:
            batch = self.gateway.consume(side, sim_time, watermark, lookahead)
            out[side] = batch
            self.batches.append({
                "sim_time": round(sim_time, 3),
                "side": side,
                "watermark": round(watermark, 3),
                "lookahead_s": lookahead,
                "cue_count": len(batch.cues),
                "cue_ids": [cue.cue_id for cue in batch.cues],
                "log_hash": batch.log_hash,
            })
        return out

    def manifest(self) -> dict:
        return {
            "bundle_hash": self.gateway.bundle.hash,
            "bundle_envelope_hash": self.gateway.bundle.envelope_hash,
            "bundle_id": self.gateway.bundle.bundle_id,
            "scenario_id": self.gateway.bundle.scenario_id,
            "scenario_start": self.gateway.bundle.scenario_start,
            "domain": self.gateway.bundle.domain,
            "bbox": list(self.gateway.bundle.bbox),
            "time_window": list(self.gateway.bundle.time_window),
            "source_allowlist": list(self.gateway.bundle.source_allowlist),
            "case_id": getattr(self.gateway.case_filter, "case_id", None),
            "case_filter_hash": self.gateway.filter_hash,
            "synthetic": self.gateway.bundle.synthetic,
            "lookahead_s": self.lookahead_s,
            "watermark_lag_s": self.watermark_lag_s,
            "sides": list(self.sides),
            "batches": [dict(item, cue_ids=list(item["cue_ids"]))
                        for item in self.batches],
            "late_ids": list(self.gateway.late),
            "cue_total": sum(item["cue_count"] for item in self.batches),
        }
