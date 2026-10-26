"""Resource kinds, their default criticality and freshness policy.

Criticality here is only the *default* used when a requirement is derived
without enough task context.  The resolver (P2) refines it per user task:
for example bathymetry is advisory for a road-march task but critical for an
amphibious landing.  Keeping the defaults in one table prevents the server,
the runner and the CLI from growing their own copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class KindSpec:
    kind: str
    description: str
    criticality: str
    ttl_days: Optional[float] = None
    schema: str = ""
    providers: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "description": self.description,
            "criticality": self.criticality,
            "ttl_days": self.ttl_days,
            "schema": self.schema,
            "providers": list(self.providers),
        }


KINDS: Dict[str, KindSpec] = {
    "envpack": KindSpec(
        "envpack", "环境包：地形/地类/水系/建筑/路网（运行期唯一空间权威）",
        "critical", 30.0, "msgeo.envpack",
        ("local-scan", "osm-overpass", "osm-pbf", "copernicus-glo30", "gebco")),
    "case_roads": KindSpec(
        "case_roads", "案例路网（初始编成与道路路由）",
        "critical", None, "situation3d.roads",
        ("local-scan", "osm-overpass", "osm-pbf")),
    "rolecards": KindSpec(
        "rolecards", "角色卡数据（类型/装备/子类）",
        "critical", None, "situation3d.rolecard",
        ("local-scan",)),
    "platforms": KindSpec(
        "platforms", "平台剖面数据（尺寸/运动/传感器/武器）",
        "critical", None, "msplatform",
        ("local-scan",)),
    "weapons": KindSpec(
        "weapons", "规则数据（武器/CRT/观察/弹道）",
        "critical", None, "msgeo.rules",
        ("local-scan",)),
    "intel_bundle": KindSpec(
        "intel_bundle", "情报包（VGI 观测/线索，只作提示不作真值）",
        "advisory", 1.0, "msgeo.intel.v1",
        ("local-scan", "usgs", "gdacs", "firms", "adsb", "eonet", "gdelt")),
    "bathymetry": KindSpec(
        "bathymetry", "水深栅格（GEBCO 或等价产品）",
        "advisory", 365.0, "msgeo.bathymetry",
        ("gebco", "local-scan")),
    "weather": KindSpec(
        "weather", "天气时间线（能见度/海况/风）",
        "advisory", 1.0, "msgeo.weather",
        ("local-scan", "open-meteo")),
    "commander_service": KindSpec(
        "commander_service", "远程指挥模型（可选，离线回退 RuleCommander）",
        "advisory", 0.0, "",
        ("http",)),
    "building_footprints": KindSpec(
        "building_footprints", "建筑轮廓/高度（GBA、Overture 等）",
        "advisory", 365.0, "s3d.vector.v1",
        ("gba", "overture")),
    "canopy_height": KindSpec(
        "canopy_height", "树冠/植被高度栅格（CHMv2 等）",
        "advisory", 365.0, "s3d.raster.v1",
        ("chmv2",)),
    "landcover": KindSpec(
        "landcover", "地类栅格（ESA WorldCover 等）",
        "advisory", 365.0, "s3d.raster.v1",
        ("worldcover",)),
    "tile_layer": KindSpec(
        "tile_layer", "XYZ 地图/影像瓦片集（在线获取或本地栅格切片）",
        "cosmetic", 365.0, "s3d.tileset.v1",
        ("xyz", "raster", "file")),
    "tiles3d": KindSpec(
        "tiles3d", "3D Tiles / quantized-mesh 显示数据集",
        "cosmetic", 365.0, "",
        ("local-scan", "ion")),
    "imagery": KindSpec(
        "imagery", "底图/影像瓦片",
        "cosmetic", 365.0, "",
        ("local-scan", "tianditu")),
}

_CRITICALITY_RANK = {"critical": 0, "advisory": 1, "cosmetic": 2}


def kinds():
    return sorted(KINDS)


def kind_spec(kind: str) -> KindSpec:
    try:
        return KINDS[str(kind)]
    except KeyError:
        raise KeyError("unknown resource kind %r (known: %s)"
                       % (kind, ", ".join(kinds())))


def validate_kind(kind: str) -> str:
    kind_spec(kind)
    return str(kind)


def default_criticality(kind: str) -> str:
    return kind_spec(kind).criticality


def criticality_rank(criticality: str) -> int:
    if criticality not in _CRITICALITY_RANK:
        raise ValueError("criticality must be critical/advisory/cosmetic")
    return _CRITICALITY_RANK[criticality]
