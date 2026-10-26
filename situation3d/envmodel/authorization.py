"""Minimal policy authorization for generic simulation actions.

This is the bridge between the P0 ``Policy`` object and a future behavior layer.
It deliberately handles only generic authorization (ROE + affiliation/domain/
weapon matrix); it does not encode tactics or target selection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    reason: str
    policy_id: Optional[str] = None


def _deny(reason: str, policy: Optional[Dict[str, Any]]) -> AuthorizationDecision:
    return AuthorizationDecision(
        False, reason, (policy or {}).get("policy_id")
    )


def _allow(reason: str, policy: Dict[str, Any]) -> AuthorizationDecision:
    return AuthorizationDecision(True, reason, policy.get("policy_id"))


def _matrix_entry(matrix: Dict[str, Any], side_key: str,
                   target_domain: str) -> Any:
    side = matrix.get(side_key)
    if isinstance(side, dict):
        return side.get(target_domain, side.get("*"))
    return side


def authorize_engagement(policy: Optional[Dict[str, Any]], side_key: str,
                         target_domain: str, weapon_category: str,
                         under_fire: bool = False) -> AuthorizationDecision:
    """Authorize a generic engagement.

    ``policy['authorization_matrix']`` is expected as::

        {"blue": {"land": ["cannon", "atgm"], "air": ["sam"]},
         "red":  {"*": ["cannon"]}}

    A side entry may be a dict keyed by target domain (or ``"*"``), or a
    sequence/string of allowed weapon categories. ``roe`` gates the action
    before the matrix: hold never fires, return_fire/defensive require
    ``under_fire``, weapons_free relies on the matrix.
    """
    if not policy:
        return _deny("no_policy", policy)
    if policy.get("approval_status") != "approved":
        return _deny("policy_not_approved", policy)
    roe = str(policy.get("roe") or "hold")
    if roe == "hold":
        return _deny("roe_hold", policy)
    if roe in ("return_fire", "defensive") and not under_fire:
        return _deny("roe_requires_threat", policy)
    if roe not in ("return_fire", "defensive", "weapons_free"):
        return _deny("roe_unknown", policy)

    matrix = policy.get("authorization_matrix") or {}
    if not isinstance(matrix, dict):
        return _deny("matrix_invalid", policy)
    entry = _matrix_entry(matrix, side_key, target_domain)
    if entry is None:
        return _deny("no_side_domain_entry", policy)
    if isinstance(entry, str):
        allowed = set(entry.split())
    elif isinstance(entry, Sequence):
        allowed = {str(item) for item in entry}
    else:
        return _deny("matrix_entry_invalid", policy)
    if "*" not in allowed and weapon_category not in allowed:
        return _deny("weapon_not_authorized", policy)
    return _allow("authorized", policy)
