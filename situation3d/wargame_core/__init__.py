"""Clean-room land wargame core for the situation3d campaign demo.

The package deliberately has no dependency on ``train_env`` or any compiled
wheel.  It implements a small, testable subset of the public 庙算 rules:

* 100 x 100 odd-r hex map (``basic.json`` compatible)
* terrain / elevation / road-aware movement
* line of sight and observation distances
* direct fire with attack levels, damage, suppression and ammo
* objective capture and score-based victory

Naval and air modules are the next extension; they are not part of this MVP.
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
