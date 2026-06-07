#!/usr/bin/env python3
"""
B10: Scenario Pool Expansion via Variant Generation.

Expands the scenario pool by reading existing scenario JSON files and
generating 1-2 variants per scenario through:
  1. Randomly shuffling operator starting positions within valid hex range
  2. Adjusting city values by +/-20%
  3. Modifying initial deployment (faction assignment swap)

Usage:
    python scripts/generate_scenario_variants.py \\
        --input-dir land_wargame_sdk/Data/Data/scenarios \\
        --output-dir data/scenarios_expanded

Options:
    --input-dir   Directory containing original scenario JSON files
    --output-dir  Directory for expanded scenario pool (originals + variants)
    --variants    Number of variants per scenario (default: 2)
    --seed        Random seed for reproducibility (default: 42)
"""

import argparse
import copy
import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================================
# Hex grid utilities
# ============================================================================

def hex_distance(pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> float:
    """Compute hex grid distance between two (row, col) positions."""
    r1, c1 = pos_a
    r2, c2 = pos_b
    x1 = c1 - (r1 - (r1 & 1)) // 2
    z1 = r1
    y1 = -x1 - z1
    x2 = c2 - (r2 - (r2 & 1)) // 2
    z2 = r2
    y2 = -x2 - z2
    return float(max(abs(x1 - x2), abs(y1 - y2), abs(z1 - z2)))


def neighbors_hex(row: int, col: int, map_rows: int, map_cols: int) -> List[Tuple[int, int]]:
    """Get valid neighboring hex positions."""
    if row & 1:  # odd row
        deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
    else:  # even row
        deltas = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
    result = []
    for dr, dc in deltas:
        nr, nc = row + dr, col + dc
        if 0 <= nr < map_rows and 0 <= nc < map_cols:
            result.append((nr, nc))
    return result


def get_positions_in_range(
    center_row: int,
    center_col: int,
    radius: int,
    map_rows: int,
    map_cols: int,
    occupied: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Get all valid positions within hex range of center, excluding occupied."""
    valid = []
    for r in range(max(0, center_row - radius), min(map_rows, center_row + radius + 1)):
        for c in range(max(0, center_col - radius), min(map_cols, center_col + radius + 1)):
            if (r, c) not in occupied:
                dist = hex_distance((center_row, center_col), (r, c))
                if dist <= radius:
                    valid.append((r, c))
    return valid


# ============================================================================
# Scenario validation
# ============================================================================

def validate_scenario(scenario: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a scenario dict for structural correctness.

    Checks:
      1. Required top-level keys exist
      2. Operator coordinates are within map bounds
      3. City coordinates are within map bounds
      4. No duplicate positions for operators
      5. Faction IDs are valid (0 or 1)

    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors = []

    # Check required keys
    required_keys = ["map", "operators", "cities"]
    for key in required_keys:
        if key not in scenario:
            errors.append(f"Missing required key: '{key}'")

    if errors:
        return False, errors

    map_info = scenario["map"]
    map_rows = map_info.get("rows", 20)
    map_cols = map_info.get("cols", 20)

    # Validate operators
    operator_positions: Set[Tuple[int, int]] = set()
    for i, op in enumerate(scenario.get("operators", [])):
        row = op.get("row", -1)
        col = op.get("col", -1)

        if not (0 <= row < map_rows):
            errors.append(
                f"Operator {i} row {row} out of bounds [0, {map_rows})"
            )
        if not (0 <= col < map_cols):
            errors.append(
                f"Operator {i} col {col} out of bounds [0, {map_cols})"
            )

        pos = (row, col)
        if pos in operator_positions:
            errors.append(
                f"Operator {i} at {pos} has duplicate position"
            )
        operator_positions.add(pos)

        faction = op.get("faction_id", -1)
        if faction not in (0, 1):
            errors.append(
                f"Operator {i} has invalid faction_id: {faction}"
            )

    # Validate cities
    for i, city in enumerate(scenario.get("cities", [])):
        row = city.get("row", -1)
        col = city.get("col", -1)

        if not (0 <= row < map_rows):
            errors.append(
                f"City {i} row {row} out of bounds [0, {map_rows})"
            )
        if not (0 <= col < map_cols):
            errors.append(
                f"City {i} col {col} out of bounds [0, {map_cols})"
            )

    is_valid = len(errors) == 0
    return is_valid, errors


# ============================================================================
# Variant generation strategies
# ============================================================================

def variant_shuffle_positions(
    scenario: Dict[str, Any],
    shuffle_radius: int = 3,
    rng: random.Random = None,
) -> Dict[str, Any]:
    """Generate a variant by shuffling operator starting positions.

    Each operator is moved to a random valid position within shuffle_radius
    hexes of its original position. City positions are not changed.

    Args:
        scenario: Original scenario dict.
        shuffle_radius: Max hex distance for shuffling.
        rng: Random number generator.

    Returns:
        New scenario dict with shuffled positions.
    """
    if rng is None:
        rng = random.Random()

    variant = copy.deepcopy(scenario)
    map_info = variant.get("map", {})
    map_rows = map_info.get("rows", 20)
    map_cols = map_info.get("cols", 20)

    # Collect all fixed positions (cities)
    occupied: Set[Tuple[int, int]] = set()
    for city in variant.get("cities", []):
        occupied.add((city["row"], city["col"]))

    # Shuffle each operator
    for op in variant.get("operators", []):
        orig_row, orig_col = op["row"], op["col"]
        occupied.discard((orig_row, orig_col))

        valid_positions = get_positions_in_range(
            orig_row, orig_col, shuffle_radius, map_rows, map_cols, occupied
        )

        if valid_positions:
            new_pos = rng.choice(valid_positions)
            op["row"] = new_pos[0]
            op["col"] = new_pos[1]
            occupied.add(new_pos)
        else:
            # No valid position found; keep original
            occupied.add((orig_row, orig_col))

    # Update variant metadata
    variant["_variant_of"] = scenario.get("_name", "unknown")
    variant["_variant_type"] = "position_shuffle"
    variant["_variant_radius"] = shuffle_radius

    return variant


def variant_adjust_city_values(
    scenario: Dict[str, Any],
    adjustment_range: float = 0.20,
    rng: random.Random = None,
) -> Dict[str, Any]:
    """Generate a variant by adjusting city values by +/-20%.

    Each city's value is multiplied by a random factor in
    [1 - adjustment_range, 1 + adjustment_range].

    Args:
        scenario: Original scenario dict.
        adjustment_range: Maximum relative adjustment (default 0.20 = 20%).
        rng: Random number generator.

    Returns:
        New scenario dict with adjusted city values.
    """
    if rng is None:
        rng = random.Random()

    variant = copy.deepcopy(scenario)

    for city in variant.get("cities", []):
        original_value = city.get("value", 1.0)
        factor = rng.uniform(1.0 - adjustment_range, 1.0 + adjustment_range)
        city["value"] = round(original_value * factor, 2)
        city["_original_value"] = original_value

    variant["_variant_of"] = scenario.get("_name", "unknown")
    variant["_variant_type"] = "city_value_adjustment"
    variant["_adjustment_range"] = adjustment_range

    return variant


def variant_swap_factions(
    scenario: Dict[str, Any],
    rng: random.Random = None,
) -> Dict[str, Any]:
    """Generate a variant by swapping faction assignments.

    All operators' faction_ids are swapped (0 -> 1, 1 -> 0), and city
    ownership is similarly swapped. This creates a mirror-image scenario.

    Args:
        scenario: Original scenario dict.
        rng: Random number generator (unused, kept for interface consistency).

    Returns:
        New scenario dict with swapped factions.
    """
    variant = copy.deepcopy(scenario)

    for op in variant.get("operators", []):
        faction = op.get("faction_id", 0)
        op["faction_id"] = 1 - faction  # Swap 0 <-> 1

    for city in variant.get("cities", []):
        owner = city.get("owner_faction", -1)
        if owner in (0, 1):
            city["owner_faction"] = 1 - owner

    variant["_variant_of"] = scenario.get("_name", "unknown")
    variant["_variant_type"] = "faction_swap"

    return variant


# ============================================================================
# Main pipeline
# ============================================================================

def load_scenarios(input_dir: Path) -> List[Dict[str, Any]]:
    """Load all scenario JSON files from a directory."""
    scenarios = []

    if not input_dir.exists():
        print(f"  Input directory {input_dir} does not exist.")
        return scenarios

    json_files = sorted(input_dir.glob("**/*.json"))
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                scenario = json.load(f)
            scenario["_source_file"] = str(json_file)
            scenario["_name"] = json_file.stem
            scenarios.append(scenario)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  WARNING: Could not load {json_file}: {e}")

    return scenarios


def generate_variants(
    scenario: Dict[str, Any],
    num_variants: int = 2,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate variants for a single scenario.

    Creates up to num_variants variants using a rotating strategy:
      - Variant 1: shuffle operator positions
      - Variant 2: adjust city values
      - Variant 3+: alternate between swap and shuffle

    Args:
        scenario: Original scenario dict.
        num_variants: Number of variants to generate.
        seed: Random seed.

    Returns:
        List of variant scenario dicts.
    """
    rng = random.Random(seed)
    variants = []

    strategies = [
        ("pos_shuffle", variant_shuffle_positions),
        ("value_adjust", variant_adjust_city_values),
        ("faction_swap", variant_swap_factions),
    ]

    for v_idx in range(num_variants):
        strategy_name, strategy_fn = strategies[v_idx % len(strategies)]
        variant_seed = seed + v_idx * 31

        variant_rng = random.Random(variant_seed)
        variant = strategy_fn(scenario, rng=variant_rng)

        # Name the variant
        orig_name = scenario.get("_name", "unknown")
        variant["_name"] = f"{orig_name}_v{strategy_name}_{v_idx:02d}"
        variant["_source_file"] = "generated"

        variants.append(variant)

    return variants


def generate_catalog(
    originals: List[Dict[str, Any]],
    all_variants: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Generate a catalog JSON file listing all scenarios.

    Args:
        originals: List of original scenario dicts.
        all_variants: List of all variant scenario dicts.
        output_dir: Output directory for the catalog file.

    Returns:
        Catalog dict.
    """
    catalog = {
        "generated_at": datetime.now().isoformat(),
        "original_count": len(originals),
        "variant_count": len(all_variants),
        "total_count": len(originals) + len(all_variants),
        "scenarios": [],
    }

    for orig in originals:
        catalog["scenarios"].append({
            "name": orig.get("_name", "unknown"),
            "type": "original",
            "source_file": orig.get("_source_file", ""),
            "variant_of": None,
        })

    for var in all_variants:
        catalog["scenarios"].append({
            "name": var.get("_name", "unknown"),
            "type": "variant",
            "variant_type": var.get("_variant_type", "unknown"),
            "source_file": var.get("_source_file", "generated"),
            "variant_of": var.get("_variant_of", "unknown"),
        })

    catalog_path = output_dir / "scenario_catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    return catalog


def main():
    parser = argparse.ArgumentParser(
        description="B10: Scenario Pool Expansion via Variant Generation"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="land_wargame_sdk/Data/Data/scenarios",
        help="Directory containing original scenario JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/scenarios_expanded",
        help="Output directory for expanded scenario pool",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=2,
        help="Number of variants per scenario (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("B10: Scenario Pool Expansion")
    print("=" * 70)
    print(f"  Input directory:  {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Variants per scenario: {args.variants}")
    print(f"  Seed: {args.seed}")
    print()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original scenarios
    print("[1/4] Loading original scenarios...")
    originals = load_scenarios(input_dir)

    if not originals:
        print("  No scenarios found in input directory.")
        print("  Generating synthetic scenarios for demonstration...")
        _create_synthetic_scenarios(input_dir)
        originals = load_scenarios(input_dir)

    print(f"  Loaded {len(originals)} original scenarios")
    print()

    # Copy originals to output
    print("[2/4] Copying originals to output directory...")
    originals_output = output_dir / "originals"
    originals_output.mkdir(parents=True, exist_ok=True)
    for orig in originals:
        src = orig.get("_source_file")
        if src and Path(src).exists():
            dest = originals_output / Path(src).name
            shutil.copy2(src, dest)
        else:
            # Write from memory
            dest = originals_output / f"{orig.get('_name', 'unknown')}.json"
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(orig, f, indent=2, ensure_ascii=False)
    print(f"  Copied {len(originals)} originals")
    print()

    # Generate variants
    print("[3/4] Generating scenario variants...")
    variants_output = output_dir / "variants"
    variants_output.mkdir(parents=True, exist_ok=True)

    all_variants = []
    validation_errors = []

    for orig_idx, orig in enumerate(originals):
        variants = generate_variants(
            orig, num_variants=args.variants, seed=args.seed + orig_idx * 100
        )

        for var in variants:
            # Validate the variant
            is_valid, errors = validate_scenario(var)
            if not is_valid:
                validation_errors.append({
                    "variant": var.get("_name", "unknown"),
                    "errors": errors,
                })
                print(f"  WARNING: Variant {var.get('_name')} failed validation:")
                for err in errors:
                    print(f"    - {err}")
                continue

            # Save variant
            var_path = variants_output / f"{var['_name']}.json"
            # Remove internal metadata before saving
            save_var = {k: v for k, v in var.items() if not k.startswith("_")}
            # Preserve variant metadata
            save_var["_variant_of"] = var.get("_variant_of", "")
            save_var["_variant_type"] = var.get("_variant_type", "")
            save_var["_name"] = var.get("_name", "")

            with open(var_path, "w", encoding="utf-8") as f:
                json.dump(save_var, f, indent=2, ensure_ascii=False)

            all_variants.append(var)

        if (orig_idx + 1) % 10 == 0 or orig_idx + 1 == len(originals):
            print(f"  Processed {orig_idx + 1}/{len(originals)} scenarios, "
                  f"generated {len(all_variants)} variants so far")

    print(f"\n  Total variants generated: {len(all_variants)}")
    if validation_errors:
        print(f"  Validation errors: {len(validation_errors)} variants rejected")
    print()

    # Generate catalog
    print("[4/4] Generating scenario catalog...")
    catalog = generate_catalog(originals, all_variants, output_dir)
    print(f"  Catalog: {len(catalog['scenarios'])} total scenarios")
    print(f"    Originals: {catalog['original_count']}")
    print(f"    Variants:  {catalog['variant_count']}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Original scenarios:    {catalog['original_count']}")
    print(f"  Generated variants:    {catalog['variant_count']}")
    print(f"  Total scenario pool:   {catalog['total_count']}")
    expansion = catalog['total_count'] / max(catalog['original_count'], 1)
    print(f"  Expansion ratio:       {expansion:.2f}x")
    if validation_errors:
        print(f"  Rejected (invalid):    {len(validation_errors)}")
    print(f"\n  Output directory: {output_dir.resolve()}")
    print(f"  Catalog file:     {output_dir / 'scenario_catalog.json'}")
    print("=" * 70)

    print("\nB10: Scenario Pool Expansion complete.")


def _create_synthetic_scenarios(output_dir: Path) -> None:
    """Create synthetic scenario files for demonstration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(99)

    configs = [
        {"name": "scenario_alpha", "rows": 10, "cols": 10, "cities": 5, "ops": 3},
        {"name": "scenario_beta", "rows": 12, "cols": 12, "cities": 7, "ops": 4},
        {"name": "scenario_gamma", "rows": 8, "cols": 8, "cities": 4, "ops": 2},
        {"name": "scenario_delta", "rows": 15, "cols": 15, "cities": 10, "ops": 5},
        {"name": "scenario_epsilon", "rows": 10, "cols": 10, "cities": 6, "ops": 3},
    ]

    for cfg in configs:
        cities = []
        used = set()
        for i in range(cfg["cities"]):
            while True:
                r, c = rng.randint(0, cfg["rows"] - 1), rng.randint(0, cfg["cols"] - 1)
                if (r, c) not in used:
                    used.add((r, c))
                    break
            cities.append({
                "id": f"city_{i}",
                "row": r,
                "col": c,
                "value": round(rng.uniform(1.0, 5.0), 2),
                "owner_faction": -1,
            })

        operators = []
        for faction in [0, 1]:
            for j in range(cfg["ops"]):
                while True:
                    r, c = rng.randint(0, cfg["rows"] - 1), rng.randint(0, cfg["cols"] - 1)
                    if (r, c) not in used:
                        used.add((r, c))
                        break
                operators.append({
                    "id": f"op_f{faction}_{j}",
                    "faction_id": faction,
                    "row": r,
                    "col": c,
                    "hp": 100.0,
                    "max_hp": 100.0,
                    "attack_level": round(rng.uniform(8.0, 15.0), 2),
                    "defense_level": round(rng.uniform(3.0, 8.0), 2),
                    "attack_range": rng.randint(2, 4),
                    "move_range": rng.randint(2, 4),
                })

        scenario = {
            "name": cfg["name"],
            "map": {"rows": cfg["rows"], "cols": cfg["cols"]},
            "cities": cities,
            "operators": operators,
            "max_steps": 300,
        }

        path = output_dir / f"{cfg['name']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scenario, f, indent=2)


if __name__ == "__main__":
    main()
