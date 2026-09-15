"""situation3d engine package.

Self-developed 3D campaign engine. Replicates the land-warfare adjudication
semantics of the closed-source 庙算 SDK (see docs/wargame_engine_data_dictionary.md)
and extends them with naval and air domains, working on real WGS84 geography and
real equipment performance parameters instead of hex-grid indices.
"""

from . import geo, scenario

__all__ = ["geo", "scenario"]
