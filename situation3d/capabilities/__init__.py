"""Capability catalog and physical gates (P2 slice)."""

from .catalog import CapabilityCatalog, build_entity_type, resolve_type
from .gates import GATES, evaluate, evaluate_all
from .model import EntityType, GateResult, Quantity, SourceRef, unknown

__all__ = [
    "CapabilityCatalog",
    "EntityType",
    "GATES",
    "GateResult",
    "Quantity",
    "SourceRef",
    "build_entity_type",
    "evaluate",
    "evaluate_all",
    "resolve_type",
    "unknown",
]
