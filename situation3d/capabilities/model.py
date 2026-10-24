"""Typed quantities and entity types for the capability catalog (P2 slice)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class SourceRef:
    refs: Tuple[str, ...] = ()
    note: str = ""
    confidence: str = "medium"

    def primary(self) -> str:
        return self.refs[0] if self.refs else ""


@dataclass(frozen=True)
class Quantity:
    """A physical value with unit and provenance.

    ``value is None`` means the card does not state the quantity; gates then
    fall back to the entity's ``unknown_policy`` instead of guessing.
    """

    value: Optional[float]
    unit: str
    source: SourceRef = field(default_factory=SourceRef)

    @property
    def known(self) -> bool:
        return self.value is not None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "known": self.known,
            "source": {
                "refs": list(self.source.refs),
                "confidence": self.source.confidence,
            },
        }


def unknown(unit: str, source: Optional[SourceRef] = None) -> Quantity:
    return Quantity(None, unit, source or SourceRef())


POLICIES = ("deny", "advisory", "allow_with_penalty", "allow")


def normalize_policy(value: Any, default: str = "deny") -> str:
    """Whitelist an unknown-policy string; anything else falls back."""
    if isinstance(value, str) and value in POLICIES:
        return value
    return default


@dataclass(frozen=True)
class EntityType:
    type_id: str
    class_id: str
    label: str
    domain: str
    modes: Tuple[str, ...]
    quantities: Mapping[str, Quantity]
    flags: Mapping[str, bool] = field(default_factory=dict)
    terrain_factors: Mapping[str, float] = field(default_factory=dict)
    unknown_policy: str = "deny"
    sdk: Mapping[str, Any] = field(default_factory=dict)

    def quantity(self, key: str) -> Quantity:
        try:
            return self.quantities[key]
        except KeyError:
            return unknown("unit")


@dataclass(frozen=True)
class GateResult:
    gate: str
    decision: str  # "pass" | "deny" | "unknown"
    reason: str
    limit: Optional[float] = None
    actual: Optional[float] = None
    unknown_policy: str = "deny"

    @property
    def blocking(self) -> bool:
        return self.decision == "deny" or (
            self.decision == "unknown"
            and normalize_policy(self.unknown_policy, "deny") == "deny")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate,
            "decision": self.decision,
            "blocking": self.blocking,
            "reason": self.reason,
            "limit": self.limit,
            "actual": self.actual,
            "unknown_policy": self.unknown_policy,
        }
