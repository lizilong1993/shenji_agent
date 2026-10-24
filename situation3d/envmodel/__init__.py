"""V3 environment authority prototype.

This package is the P1 vertical-slice facade: it validates an
``msgeo.envpack.v3`` manifest, wraps the existing ``EnvPackEnvironment`` in a
time-bound ``SpatialQuery`` contract, and exposes the building/obstacle set
used by LOS and passability queries.
"""

from .authorization import AuthorizationDecision, authorize_engagement
from .gates import GateResult, evaluate_gates
from .manifest import ManifestError, V3Manifest, load_manifest
from .obstacles import BridgeSet, ObstacleSet
from .policy import PolicyError, gate_overrides, load_policy, validate_policy
from .query import Envelope, SpatialQueryV3
from .route import plan_valid_route, validate_route


def open_pack(root: str, time_s: float = 0.0,
              policy_path: str = None,
              policy: dict = None) -> SpatialQueryV3:
    """Load a v3 pack and return a time-bound query view.

    ``policy_path`` or ``policy`` can provide approved gate overrides; a path is
    loaded and validated with ``require_approved=True``.
    """
    from env.pack_env import load_envpack_environment

    manifest = load_manifest(root)
    env, _grid = load_envpack_environment(root)
    if policy_path:
        policy = load_policy(policy_path, require_approved=True)
    return SpatialQueryV3(env, manifest, time_s=time_s, policy=policy)


__all__ = [
    "AuthorizationDecision",
    "BridgeSet",
    "Envelope",
    "GateResult",
    "ManifestError",
    "ObstacleSet",
    "PolicyError",
    "SpatialQueryV3",
    "V3Manifest",
    "authorize_engagement",
    "evaluate_gates",
    "gate_overrides",
    "load_manifest",
    "load_policy",
    "open_pack",
    "plan_valid_route",
    "validate_policy",
    "validate_route",
]
