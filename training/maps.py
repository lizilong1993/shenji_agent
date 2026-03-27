from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .common import ensure_dir, save_json


@dataclass
class MapRecord:
    map_id: int
    name: str
    rows: int
    cols: int
    terrain_entropy: float
    elevation_std: float
    cover_density: float
    road_sparsity: float
    river_density: float
    connectivity_penalty: float
    complexity_score: float = 0.0
    bucket: str = "medium"


def _entropy(values: Iterable[int]) -> float:
    counts: Dict[int, int] = {}
    total = 0
    for value in values:
        counts[value] = counts.get(value, 0) + 1
        total += 1
    probs = [count / total for count in counts.values() if total]
    return float(-sum(p * math.log(p + 1e-12) for p in probs))


def _roads_present(cell: Dict[str, object]) -> bool:
    roads = cell.get("roads", [])
    return any(int(road) != 0 for road in roads)


def build_map_catalog(maps_root: Path, output_path: Path | None = None) -> List[MapRecord]:
    records: List[MapRecord] = []
    for map_dir in sorted(path for path in maps_root.iterdir() if path.is_dir()):
        basic_path = map_dir / "basic.json"
        if not basic_path.exists():
            continue
        data = json.loads(basic_path.read_text(encoding="utf-8"))
        grid = data["map_data"]
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        cells = [cell for row in grid for cell in row]
        conds = [int(cell.get("cond", 0)) for cell in cells]
        elevations = np.array([float(cell.get("elev", 0.0)) for cell in cells], dtype=np.float32)
        road_fraction = float(sum(1 for cell in cells if _roads_present(cell)) / max(len(cells), 1))
        river_fraction = float(sum(1 for cell in cells if any(int(r) != 0 for r in cell.get("rivers", []))) / max(len(cells), 1))
        mean_neighbors = float(np.mean([len(cell.get("neighbors", [])) for cell in cells])) if cells else 0.0
        record = MapRecord(
            map_id=int(data.get("map_id", map_dir.name.split("_")[-1])),
            name=map_dir.name,
            rows=rows,
            cols=cols,
            terrain_entropy=_entropy(conds),
            elevation_std=float(np.std(elevations)) if len(elevations) else 0.0,
            cover_density=float(sum(1 for cond in conds if cond in {1, 2}) / max(len(conds), 1)),
            road_sparsity=1.0 - road_fraction,
            river_density=river_fraction,
            connectivity_penalty=max(0.0, 1.0 - mean_neighbors / 6.0),
        )
        records.append(record)

    if not records:
        return records

    metric_names = [
        "terrain_entropy",
        "elevation_std",
        "cover_density",
        "road_sparsity",
        "river_density",
        "connectivity_penalty",
    ]
    metric_matrix = np.array([[getattr(record, name) for name in metric_names] for record in records], dtype=np.float32)
    means = metric_matrix.mean(axis=0)
    stds = metric_matrix.std(axis=0) + 1e-6
    z_scores = (metric_matrix - means) / stds
    complexity = z_scores.mean(axis=1)
    order = np.argsort(complexity)
    for idx, record in enumerate(records):
        record.complexity_score = float(complexity[idx])
    for rank, idx in enumerate(order.tolist()):
        fraction = rank / max(len(order) - 1, 1)
        if fraction < 0.34:
            records[idx].bucket = "simple"
        elif fraction < 0.67:
            records[idx].bucket = "medium"
        else:
            records[idx].bucket = "complex"

    if output_path is not None:
        ensure_dir(output_path.parent)
        save_json({"maps": [asdict(record) for record in records]}, output_path)
    return records


def infer_map_id_from_scenario_id(scenario_id: str | int, map_ids: Iterable[int]) -> int | None:
    value = str(scenario_id)
    choices = sorted((str(map_id) for map_id in map_ids), key=len, reverse=True)
    for candidate in choices:
        if value.startswith(candidate) or value.endswith(candidate):
            return int(candidate)
    for candidate in choices:
        if candidate in value:
            return int(candidate)
    return None


def map_bucket_lookup(records: Iterable[MapRecord]) -> Dict[int, str]:
    return {record.map_id: record.bucket for record in records}
