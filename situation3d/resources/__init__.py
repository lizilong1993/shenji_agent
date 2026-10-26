"""Unified resource registry for the situation3d engine (P0)."""

from .catalog import KINDS, default_criticality, kind_spec, kinds, validate_kind
from .model import (INDEX_SCHEMA, PLAN_SCHEMA, ResourceDecision, ResourceEntry,
                    ResourceGap, ResourcePlan, ResourceRequirement,
                    canonical_json, hash_path, slugify, utc_now)
from .store import (ResourceConflict, ResourceError, ResourceNotFound,
                    ResourceStore, default_root)

__all__ = [
    "INDEX_SCHEMA", "PLAN_SCHEMA", "KINDS", "ResourceDecision",
    "ResourceEntry", "ResourceGap", "ResourcePlan", "ResourceRequirement",
    "ResourceError", "ResourceNotFound", "ResourceConflict",
    "canonical_json", "default_criticality", "default_root", "hash_path",
    "kind_spec", "kinds", "slugify", "utc_now", "validate_kind",
]
