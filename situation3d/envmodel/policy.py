"""Policy loading and gate overrides for the P1 environment slice.

The P0 policy schema is intentionally separate from the platform catalog: a
policy is an approved decision object, not a platform attribute.  This module
keeps the runtime loader minimal and makes the approved gate overrides explicit.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

REQUIRED = (
    "policy_id",
    "revision",
    "approval_status",
    "approved_by",
    "roe",
    "posture",
    "authorization_matrix",
)


class PolicyError(ValueError):
    """Raised when a policy does not satisfy the P1 loader contract."""


def load_policy(path: str, require_approved: bool = True) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return validate_policy(data, require_approved=require_approved)


def validate_policy(data: Any, require_approved: bool = True) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise PolicyError("policy must be a JSON object")
    if data.get("schema") != "situation3d.policy.v1":
        raise PolicyError("unsupported policy schema %r" % (data.get("schema"),))
    for key in REQUIRED:
        if key not in data or data.get(key) is None:
            raise PolicyError("policy missing %r" % key)
    if require_approved and data.get("approval_status") != "approved":
        raise PolicyError("policy is not approved")
    gates = data.get("gates")
    if gates is not None and not isinstance(gates, dict):
        raise PolicyError("policy gates must be an object")
    for name, value in (gates or {}).items():
        unknown = value.get("unknown_policy") if isinstance(value, dict) else value
        if unknown is not None and str(unknown) not in (
                "deny", "advisory", "allow_with_penalty"):
            raise PolicyError("invalid unknown_policy for gate %r" % name)
    return data


def gate_overrides(policy: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Return ``{gate_name: unknown_policy}`` from an approved policy."""
    if not policy or policy.get("approval_status") != "approved":
        return {}
    out: Dict[str, str] = {}
    for key, value in (policy.get("gates") or {}).items():
        if isinstance(value, dict):
            policy_value = value.get("unknown_policy")
        else:
            policy_value = value
        if policy_value:
            out[str(key)] = str(policy_value)
    return out
