"""Role cards (大类/小类/unknown) for situation3d targets.

Public API:
    load_registry()          -- load and validate the data package
    RoleCardRegistry         -- registry (effective cards, lookups, validation)
    card_from_legacy(target) -- P0 adapter from the old Target model
    project_legacy(card)     -- project an effective card back to legacy fields
"""

from .registry import (
    RoleCardError,
    RoleCardRegistry,
    deep_merge,
    load_registry,
)
from .legacy import card_from_legacy, legacy_view, project_legacy

__all__ = [
    "RoleCardError",
    "RoleCardRegistry",
    "deep_merge",
    "load_registry",
    "card_from_legacy",
    "legacy_view",
    "project_legacy",
]
