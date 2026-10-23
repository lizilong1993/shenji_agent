"""Clean-room joint wargame core for the situation3d campaign demo.

The package deliberately has no dependency on ``train_env`` or any compiled
wheel.  It implements a small, testable subset of the public 庙算 land rules and
extends the same model to sea and air:

* 100 x 100 odd-r hex map (``basic.json`` compatible)
* terrain / elevation / road-aware land movement
* sea and air movement with altitude bands
* sensors (radar / IR / sonar / ESM), line of sight and radar horizon
* an explicit engagement matrix (who can engage whom with what)
* land direct fire with public attack-level tables
* sea/air weapons with synthetic, documented parameters
* fuel, resupply and logistics interdiction
* objective capture and score-based victory

The sea/air parameters are demonstration values (``approximated: true``), not
real performance data.
"""

from .hexmap import HexMap, distance, encode, decode, line
from .state import GameState, Objective, Unit
from .engine import Order, WargameEngine, auto_orders, run_demo
from .scenario import load_scenario, load_map, build_state

__all__ = [
    "HexMap", "distance", "encode", "decode", "line",
    "GameState", "Objective", "Unit",
    "Order", "WargameEngine", "auto_orders", "run_demo",
    "load_scenario", "load_map", "build_state",
]
