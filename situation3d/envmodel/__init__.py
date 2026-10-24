"""V3 environment authority prototype.

This package is the P1 vertical-slice facade: it validates an
``msgeo.envpack.v3`` manifest, wraps the existing ``EnvPackEnvironment`` in a
time-bound ``SpatialQuery`` contract, and exposes the building/obstacle set
used by LOS and passability queries.
"""

from .gates import GateResult, evaluate_gates
from .manifest import ManifestError, V3Manifest, load_manifest
from .obstacles import ObstacleSet
from .query import Envelope, SpatialQueryV3


def open_pack(root: str, time_s: float = 0.0) -> SpatialQueryV3:
    """Load a v3 pack and return a time-bound query view."""
    from env.pack_env import load_envpack_environment

    manifest = load_manifest(root)
    env, _grid = load_envpack_environment(root)
    return SpatialQueryV3(env, manifest, time_s=time_s)


__all__ = [
    "Envelope",
    "GateResult",
    "ManifestError",
    "ObstacleSet",
    "SpatialQueryV3",
    "V3Manifest",
    "evaluate_gates",
    "load_manifest",
    "open_pack",
]
