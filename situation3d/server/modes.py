"""Server modes: demo (read-only replay) vs test (live simulation + runs).

The demo port must never be able to execute the simulator.  Keep the mode
logic in one tiny module so ``serve.py`` and the tests agree on the exact
capability surface.
"""

from __future__ import annotations

import sys

MODES = ("test", "demo")
DEFAULT_MODE = "demo"
SIMULATION_MODULES = ("sim", "rules")


def normalize_mode(value):
    mode = str(value or DEFAULT_MODE).strip().lower()
    if mode not in MODES:
        raise ValueError("unknown server mode: %r (expected test|demo)" % (value,))
    return mode


def capabilities(mode):
    mode = normalize_mode(mode)
    is_test = mode == "test"
    return {
        "simulate": is_test,
        "runs": is_test,
        "promote": is_test,
        "replay": True,
    }


def loaded_simulation_modules(modules=None):
    names = set(modules if modules is not None else sys.modules)
    found = []
    for name in sorted(names):
        for prefix in SIMULATION_MODULES:
            if name == prefix or name.startswith(prefix + "."):
                found.append(name)
                break
    return found


def assert_no_simulation(mode, modules=None):
    """Refuse to start a demo server that has the simulator loaded."""
    if normalize_mode(mode) != "demo":
        return
    found = loaded_simulation_modules(modules)
    if found:
        raise RuntimeError(
            "demo mode must not load simulation modules: %s" % ", ".join(found))
