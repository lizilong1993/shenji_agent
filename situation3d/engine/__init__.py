"""situation3d engine package.

Self-developed 3D campaign engine. Replicates the land-warfare adjudication
semantics of the closed-source 庙算 SDK (see docs/wargame_engine_data_dictionary.md)
and extends them with naval and air domains, working on real WGS84 geography and
real equipment performance parameters instead of hex-grid indices.

Modules
-------
geo          WGS84 geodesic helpers
symbology    military symbol catalogue (single source of truth for icon keys)
scenario     order of battle, hotspot regions, per-tick trajectory generation
inference    three-level situation inference (target / grouping / region)
cases        demonstration case registry backing the front-end case menu
"""

from . import cases, geo, inference, scenario, symbology

__all__ = ["cases", "geo", "inference", "scenario", "symbology"]
