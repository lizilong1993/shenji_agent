#!/usr/bin/env python3
"""
B9: Training Data Augmentation.

Augments training data for the three model types:
  1. Intent data (JSON sequences): temporal jittering, Gaussian feature noise,
     sequence truncation.
  2. Situation data (graph CSVs): edge dropout (10%), node feature noise.
  3. Tactical data (CSV): feature perturbation, SMOTE-like oversampling of
     minority class.

Usage:
    python scripts/augment_training_data.py \\
        --input-dir data/training \\
        --output-dir data/training_augmented

The script processes all three data types and generates a summary report
comparing original vs augmented data sizes.
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Utility functions
# ============================================================================

def gaussian_noise(value: float, sigma: float, rng: random.Random) -> float:
    """Add Gaussian noise to a numeric value."""
    return value + rng.gauss(0, sigma)


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Intent data augmentation (JSON sequences)
# ============================================================================

def augment_intent_data(
    input_dir: Path,
    output_dir: Path,
    noise_sigma: float = 0.01,
    jitter_range: int = 2,
    truncation_ratios: Tuple[float, ...] = (0.7, 0.8, 0.9),
    augment_factor: int = 3,
    seed: int = 42,
) -> Dict[str, int]:
    """Augment intent data stored as JSON sequences.

    Applies three augmentation strategies:
      1. Temporal jittering: randomly shift the order of adjacent steps by
         up to jitter_range positions.
      2. Feature noise: add Gaussian noise (sigma=0.01) to all numeric features.
      3. Sequence truncation: randomly truncate sequences to 70%, 80%, 90% length.

    Args:
        input_dir: Directory containing original intent JSON files.
        output_dir: Directory to write augmented data.
        noise_sigma: Standard deviation for Gaussian feature noise.
        jitter_range: Maximum positions to jitter in temporal ordering.
        truncation_ratios: Ratios at which to truncate sequences.
        augment_factor: Number of augmented copies per original.
        seed: Random seed.

    Returns:
        Dict with 'original' and 'augmented' counts.
    """
    rng = random.Random(seed)
    ensure_dir(output_dir)

    original_count = 0
    augmented_count = 0

    # Find all JSON files in input directory
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"  [Intent] No JSON files found in {input_dir}")
        return {"original": 0, "augmented": 0}

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        original_count += 1

        # Always copy original
        out_original = output_dir / json_file.name
        with open(out_original, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Generate augmented copies
        sequences = data if isinstance(data, list) else [data]

        for aug_idx in range(augment_factor):
            aug_sequences = []

            for seq in sequences:
                # Strategy selection: rotate through strategies
                strategy = aug_idx % 3

                if strategy == 0:
                    aug_seq = _jitter_temporal(seq, jitter_range, rng)
                elif strategy == 1:
                    aug_seq = _add_feature_noise(seq, noise_sigma, rng)
                else:
                    ratio = rng.choice(truncation_ratios)
                    aug_seq = _truncate_sequence(seq, ratio, rng)

                aug_sequences.append(aug_seq)

            # Save augmented data
            stem = json_file.stem
            suffix = json_file.suffix
            aug_filename = f"{stem}_aug{aug_idx:02d}{suffix}"
            aug_path = output_dir / aug_filename

            aug_data = aug_sequences if isinstance(data, list) else aug_sequences[0]
            with open(aug_path, "w", encoding="utf-8") as f:
                json.dump(aug_data, f, indent=2, ensure_ascii=False)

            augmented_count += 1

    return {"original": original_count, "augmented": augmented_count}


def _jitter_temporal(data: Any, jitter_range: int, rng: random.Random) -> Any:
    """Apply temporal jittering to a sequence of steps.

    Randomly swaps adjacent elements within jitter_range positions.
    """
    result = copy.deepcopy(data)

    if isinstance(result, dict) and "steps" in result:
        steps = result["steps"]
    elif isinstance(result, list):
        steps = result
    else:
        return result

    if len(steps) <= 1:
        return result

    # Perform random swaps within jitter range
    num_swaps = max(1, len(steps) // 4)
    for _ in range(num_swaps):
        i = rng.randint(0, len(steps) - 2)
        j = min(i + rng.randint(1, jitter_range), len(steps) - 1)
        steps[i], steps[j] = steps[j], steps[i]

    if isinstance(result, dict):
        result["steps"] = steps
        result["_augmented"] = "temporal_jitter"
    return result


def _add_feature_noise(data: Any, sigma: float, rng: random.Random) -> Any:
    """Add Gaussian noise to all numeric features in the data."""
    result = copy.deepcopy(data)

    def _noisify(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _noisify(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_noisify(item) for item in obj]
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            noisy = gaussian_noise(float(obj), sigma, rng)
            return round(noisy, 6)
        return obj

    result = _noisify(result)
    if isinstance(result, dict):
        result["_augmented"] = "feature_noise"
    return result


def _truncate_sequence(data: Any, ratio: float, rng: random.Random) -> Any:
    """Truncate a sequence to a given ratio of its original length."""
    result = copy.deepcopy(data)

    if isinstance(result, dict) and "steps" in result:
        steps = result["steps"]
        new_len = max(1, int(len(steps) * ratio))
        # Random start position for the truncation window
        max_start = max(0, len(steps) - new_len)
        start = rng.randint(0, max_start) if max_start > 0 else 0
        result["steps"] = steps[start:start + new_len]
        result["_augmented"] = f"truncation_{ratio}"
    elif isinstance(result, list):
        new_len = max(1, int(len(result) * ratio))
        max_start = max(0, len(result) - new_len)
        start = rng.randint(0, max_start) if max_start > 0 else 0
        result = result[start:start + new_len]

    return result


# ============================================================================
# Situation data augmentation (graph CSVs)
# ============================================================================

def augment_situation_data(
    input_dir: Path,
    output_dir: Path,
    edge_dropout_rate: float = 0.10,
    feature_noise_sigma: float = 0.01,
    augment_factor: int = 2,
    seed: int = 42,
) -> Dict[str, int]:
    """Augment situation data stored as graph CSV files.

    Applies two augmentation strategies:
      1. Edge dropout: randomly remove 10% of edges from the graph.
      2. Node feature noise: add Gaussian noise to node feature columns.

    Expected CSV formats:
      - nodes_*.csv: columns include 'node_id' plus numeric feature columns
      - edges_*.csv: columns include 'source', 'target', and optional 'weight'

    Args:
        input_dir: Directory containing original CSV files.
        output_dir: Directory to write augmented data.
        edge_dropout_rate: Fraction of edges to randomly drop.
        feature_noise_sigma: Standard deviation for node feature noise.
        augment_factor: Number of augmented copies per original.
        seed: Random seed.

    Returns:
        Dict with 'original' and 'augmented' counts.
    """
    rng = random.Random(seed)
    ensure_dir(output_dir)

    original_count = 0
    augmented_count = 0

    # Group files by base name (nodes/edges pairs)
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"  [Situation] No CSV files found in {input_dir}")
        return {"original": 0, "augmented": 0}

    # Separate node and edge files
    node_files = [f for f in csv_files if "node" in f.stem.lower()]
    edge_files = [f for f in csv_files if "edge" in f.stem.lower()]
    other_files = [f for f in csv_files if "node" not in f.stem.lower() and "edge" not in f.stem.lower()]

    # If no explicit node/edge separation, treat all as generic
    if not node_files and not edge_files:
        node_files = csv_files
        edge_files = []

    original_count = len(csv_files)

    # Copy originals
    for csv_file in csv_files:
        shutil.copy2(csv_file, output_dir / csv_file.name)

    # Generate augmented copies
    for aug_idx in range(augment_factor):
        # Augment node files (feature noise)
        for node_file in node_files:
            aug_rows = _augment_node_csv(node_file, feature_noise_sigma, rng)
            stem = node_file.stem
            aug_filename = f"{stem}_aug{aug_idx:02d}.csv"
            aug_path = output_dir / aug_filename
            _write_csv_rows(aug_rows, aug_path)
            augmented_count += 1

        # Augment edge files (edge dropout)
        for edge_file in edge_files:
            aug_rows = _augment_edge_csv(edge_file, edge_dropout_rate, rng)
            stem = edge_file.stem
            aug_filename = f"{stem}_aug{aug_idx:02d}.csv"
            aug_path = output_dir / aug_filename
            _write_csv_rows(aug_rows, aug_path)
            augmented_count += 1

        # Copy other files as-is
        for other_file in other_files:
            stem = other_file.stem
            aug_filename = f"{stem}_aug{aug_idx:02d}.csv"
            shutil.copy2(other_file, output_dir / aug_filename)

    return {"original": original_count, "augmented": augmented_count}


def _augment_node_csv(
    file_path: Path, sigma: float, rng: random.Random
) -> List[List[str]]:
    """Read a node CSV and add Gaussian noise to numeric columns."""
    rows = []
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows.append(header)

        # Identify numeric columns (try to parse first data row)
        first_row = next(reader, None)
        if first_row is None:
            return rows

        numeric_cols = set()
        for i, val in enumerate(first_row):
            try:
                float(val)
                if header[i].lower() != "node_id":
                    numeric_cols.add(i)
            except (ValueError, IndexError):
                pass

        # Process first row
        aug_row = list(first_row)
        for i in numeric_cols:
            try:
                aug_row[i] = str(round(gaussian_noise(float(first_row[i]), sigma, rng), 6))
            except (ValueError, IndexError):
                pass
        rows.append(aug_row)

        # Process remaining rows
        for row in reader:
            aug_row = list(row)
            for i in numeric_cols:
                try:
                    aug_row[i] = str(round(gaussian_noise(float(row[i]), sigma, rng), 6))
                except (ValueError, IndexError):
                    pass
            rows.append(aug_row)

    return rows


def _augment_edge_csv(
    file_path: Path, dropout_rate: float, rng: random.Random
) -> List[List[str]]:
    """Read an edge CSV and randomly drop edges."""
    rows = []
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows.append(header)

        data_rows = list(reader)

    # Keep edges with probability (1 - dropout_rate)
    for row in data_rows:
        if rng.random() > dropout_rate:
            rows.append(row)

    return rows


def _write_csv_rows(rows: List[List[str]], path: Path) -> None:
    """Write rows to a CSV file."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# ============================================================================
# Tactical data augmentation (CSV with labels)
# ============================================================================

def augment_tactical_data(
    input_dir: Path,
    output_dir: Path,
    perturbation_sigma: float = 0.01,
    smote_k: int = 5,
    augment_factor: int = 2,
    seed: int = 42,
) -> Dict[str, int]:
    """Augment tactical data stored as CSV files.

    Applies two augmentation strategies:
      1. Feature perturbation: add Gaussian noise to numeric features.
      2. SMOTE-like oversampling: generate synthetic samples for minority
         classes by interpolating between k-nearest neighbors.

    The label column is assumed to be the last column or a column named
    'label', 'class', or 'action_type'.

    Args:
        input_dir: Directory containing original tactical CSV files.
        output_dir: Directory to write augmented data.
        perturbation_sigma: Noise sigma for feature perturbation.
        smote_k: Number of nearest neighbors for SMOTE interpolation.
        augment_factor: Number of augmented copies per original.
        seed: Random seed.

    Returns:
        Dict with 'original' and 'augmented' counts.
    """
    rng = random.Random(seed)
    ensure_dir(output_dir)

    original_count = 0
    augmented_count = 0

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"  [Tactical] No CSV files found in {input_dir}")
        return {"original": 0, "augmented": 0}

    for csv_file in csv_files:
        original_count += 1

        # Read the CSV
        header, data_rows = _read_tactical_csv(csv_file)
        label_col_idx = _find_label_column(header)

        # Copy original
        shutil.copy2(csv_file, output_dir / csv_file.name)

        for aug_idx in range(augment_factor):
            if aug_idx % 2 == 0:
                # Feature perturbation
                aug_rows = _perturb_features(header, data_rows, perturbation_sigma, rng)
            else:
                # SMOTE-like oversampling
                aug_rows = _smote_oversample(
                    header, data_rows, label_col_idx, smote_k, rng
                )

            stem = csv_file.stem
            aug_filename = f"{stem}_aug{aug_idx:02d}.csv"
            aug_path = output_dir / aug_filename
            _write_csv_rows([header] + aug_rows, aug_path)
            augmented_count += 1

    return {"original": original_count, "augmented": augmented_count}


def _read_tactical_csv(path: Path) -> Tuple[List[str], List[List[str]]]:
    """Read a CSV file and return (header, data_rows)."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        data_rows = list(reader)
    return header, data_rows


def _find_label_column(header: List[str]) -> int:
    """Find the index of the label column."""
    label_names = {"label", "class", "action_type", "target", "y"}
    for i, col in enumerate(header):
        if col.strip().lower() in label_names:
            return i
    # Default: last column
    return len(header) - 1


def _perturb_features(
    header: List[str],
    data_rows: List[List[str]],
    sigma: float,
    rng: random.Random,
) -> List[List[str]]:
    """Add Gaussian noise to numeric feature columns."""
    # Identify numeric columns
    numeric_cols = set()
    if data_rows:
        for i, val in enumerate(data_rows[0]):
            try:
                float(val)
                numeric_cols.add(i)
            except ValueError:
                pass

    aug_rows = []
    for row in data_rows:
        aug_row = list(row)
        for i in numeric_cols:
            try:
                aug_row[i] = str(round(gaussian_noise(float(row[i]), sigma, rng), 6))
            except (ValueError, IndexError):
                pass
        aug_rows.append(aug_row)

    return aug_rows


def _smote_oversample(
    header: List[str],
    data_rows: List[List[str]],
    label_col_idx: int,
    k: int,
    rng: random.Random,
) -> List[List[str]]:
    """SMOTE-like oversampling of minority class.

    For each minority class sample, find k nearest neighbors in feature space
    and create synthetic samples by linear interpolation.
    """
    if not data_rows:
        return []

    # Identify numeric columns
    numeric_cols = []
    for i, val in enumerate(data_rows[0]):
        if i == label_col_idx:
            continue
        try:
            float(val)
            numeric_cols.append(i)
        except ValueError:
            pass

    if not numeric_cols:
        return list(data_rows)

    # Group rows by label
    label_groups: Dict[str, List[List[str]]] = defaultdict(list)
    for row in data_rows:
        label = row[label_col_idx] if label_col_idx < len(row) else "unknown"
        label_groups[label].append(row)

    # Find minority class(es)
    class_counts = {label: len(rows) for label, rows in label_groups.items()}
    if not class_counts:
        return list(data_rows)

    max_count = max(class_counts.values())
    min_count = min(class_counts.values())

    # Start with all original rows
    aug_rows = [list(row) for row in data_rows]

    # Oversample minority classes
    for label, rows in label_groups.items():
        if len(rows) >= max_count:
            continue

        # Number of synthetic samples needed
        n_synthetic = max_count - len(rows)

        if len(rows) < 2:
            # Cannot do SMOTE with < 2 samples, just duplicate
            for _ in range(n_synthetic):
                aug_rows.append(list(rng.choice(rows)))
            continue

        # Convert to numeric vectors for distance computation
        vectors = []
        for row in rows:
            vec = []
            for col_idx in numeric_cols:
                try:
                    vec.append(float(row[col_idx]))
                except (ValueError, IndexError):
                    vec.append(0.0)
            vectors.append(vec)

        # Generate synthetic samples
        actual_k = min(k, len(rows) - 1)
        for _ in range(n_synthetic):
            # Pick a random sample
            idx = rng.randint(0, len(rows) - 1)
            # Find k nearest neighbors (simple brute force)
            dists = []
            for j, vec_j in enumerate(vectors):
                if j == idx:
                    continue
                d = sum((a - b) ** 2 for a, b in zip(vectors[idx], vec_j))
                dists.append((d, j))
            dists.sort(key=lambda x: x[0])
            neighbors = [d[1] for d in dists[:actual_k]]

            if not neighbors:
                aug_rows.append(list(rows[idx]))
                continue

            # Pick a random neighbor
            nn_idx = rng.choice(neighbors)

            # Interpolate between sample and neighbor
            lam = rng.uniform(0.0, 1.0)
            synthetic_row = list(rows[idx])
            for col_pos, col_idx in enumerate(numeric_cols):
                try:
                    val_a = float(rows[idx][col_idx])
                    val_b = float(rows[nn_idx][col_idx])
                    synthetic_val = val_a + lam * (val_b - val_a)
                    synthetic_row[col_idx] = str(round(synthetic_val, 6))
                except (ValueError, IndexError):
                    pass

            aug_rows.append(synthetic_row)

    return aug_rows


# ============================================================================
# Report generation
# ============================================================================

def generate_summary_report(
    intent_stats: Dict[str, int],
    situation_stats: Dict[str, int],
    tactical_stats: Dict[str, int],
    input_dir: Path,
    output_dir: Path,
) -> str:
    """Generate a human-readable summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("B9: Training Data Augmentation -- Summary Report")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Input directory:  {input_dir.resolve()}")
    lines.append(f"Output directory: {output_dir.resolve()}")
    lines.append("")

    lines.append("-" * 70)
    lines.append(f"{'Data Type':<20} {'Original':>10} {'Augmented':>10} {'Total':>10} {'Ratio':>8}")
    lines.append("-" * 70)

    total_original = 0
    total_augmented = 0

    for name, stats in [("Intent (JSON)", intent_stats),
                         ("Situation (CSV)", situation_stats),
                         ("Tactical (CSV)", tactical_stats)]:
        orig = stats.get("original", 0)
        aug = stats.get("augmented", 0)
        total = orig + aug
        ratio = f"{total / max(orig, 1):.2f}x"
        lines.append(f"{name:<20} {orig:>10} {aug:>10} {total:>10} {ratio:>8}")
        total_original += orig
        total_augmented += aug

    lines.append("-" * 70)
    total_all = total_original + total_augmented
    overall_ratio = f"{total_all / max(total_original, 1):.2f}x"
    lines.append(
        f"{'TOTAL':<20} {total_original:>10} {total_augmented:>10} "
        f"{total_all:>10} {overall_ratio:>8}"
    )
    lines.append("-" * 70)
    lines.append("")

    lines.append("Augmentation strategies applied:")
    lines.append("  Intent data:")
    lines.append("    - Temporal jittering (swap adjacent steps within range 2)")
    lines.append("    - Gaussian feature noise (sigma=0.01)")
    lines.append("    - Sequence truncation (70%, 80%, 90% ratios)")
    lines.append("  Situation data:")
    lines.append("    - Edge dropout (10% random removal)")
    lines.append("    - Node feature noise (Gaussian sigma=0.01)")
    lines.append("  Tactical data:")
    lines.append("    - Feature perturbation (Gaussian sigma=0.01)")
    lines.append("    - SMOTE-like oversampling of minority class (k=5)")
    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    return report


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="B9: Training Data Augmentation for Shenji Agent"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Root input directory containing training data subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root output directory for augmented data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 70)
    print("B9: Training Data Augmentation")
    print("=" * 70)
    print(f"  Input directory:  {input_dir.resolve()}")
    print(f"  Output directory: {output_dir.resolve()}")
    print(f"  Seed:             {args.seed}")
    print()

    # Process intent data
    print("[1/3] Augmenting intent data (JSON sequences)...")
    intent_input = input_dir / "intent"
    intent_output = output_dir / "intent"
    if intent_input.exists():
        intent_stats = augment_intent_data(
            intent_input, intent_output, seed=args.seed
        )
    else:
        print(f"  Intent input dir {intent_input} not found, creating sample data...")
        intent_input.mkdir(parents=True, exist_ok=True)
        _create_sample_intent_data(intent_input, args.seed)
        intent_stats = augment_intent_data(
            intent_input, intent_output, seed=args.seed
        )
    print(f"  Intent: {intent_stats['original']} original -> "
          f"{intent_stats['augmented']} augmented files")
    print()

    # Process situation data
    print("[2/3] Augmenting situation data (graph CSVs)...")
    situation_input = input_dir / "situation"
    situation_output = output_dir / "situation"
    if situation_input.exists():
        situation_stats = augment_situation_data(
            situation_input, situation_output, seed=args.seed
        )
    else:
        print(f"  Situation input dir {situation_input} not found, creating sample data...")
        situation_input.mkdir(parents=True, exist_ok=True)
        _create_sample_situation_data(situation_input, args.seed)
        situation_stats = augment_situation_data(
            situation_input, situation_output, seed=args.seed
        )
    print(f"  Situation: {situation_stats['original']} original -> "
          f"{situation_stats['augmented']} augmented files")
    print()

    # Process tactical data
    print("[3/3] Augmenting tactical data (CSV)...")
    tactical_input = input_dir / "tactical"
    tactical_output = output_dir / "tactical"
    if tactical_input.exists():
        tactical_stats = augment_tactical_data(
            tactical_input, tactical_output, seed=args.seed
        )
    else:
        print(f"  Tactical input dir {tactical_input} not found, creating sample data...")
        tactical_input.mkdir(parents=True, exist_ok=True)
        _create_sample_tactical_data(tactical_input, args.seed)
        tactical_stats = augment_tactical_data(
            tactical_input, tactical_output, seed=args.seed
        )
    print(f"  Tactical: {tactical_stats['original']} original -> "
          f"{tactical_stats['augmented']} augmented files")
    print()

    # Generate report
    report = generate_summary_report(
        intent_stats, situation_stats, tactical_stats, input_dir, output_dir
    )
    print(report)

    # Save report
    report_path = output_dir / "augmentation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    print("\nB9: Training Data Augmentation complete.")


def _create_sample_intent_data(input_dir: Path, seed: int) -> None:
    """Create sample intent data for demonstration."""
    rng = random.Random(seed)
    for i in range(5):
        steps = []
        for s in range(rng.randint(10, 30)):
            steps.append({
                "step": s,
                "action": rng.choice(["move", "shoot", "occupy"]),
                "position": [rng.randint(0, 15), rng.randint(0, 15)],
                "confidence": round(rng.uniform(0.3, 1.0), 4),
                "features": [round(rng.gauss(0, 1), 4) for _ in range(8)],
            })
        data = {"episode_id": f"ep_{i:03d}", "steps": steps}
        with open(input_dir / f"intent_{i:03d}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def _create_sample_situation_data(input_dir: Path, seed: int) -> None:
    """Create sample situation graph data for demonstration."""
    rng = random.Random(seed)
    for i in range(3):
        # Node file
        node_rows = [["node_id", "x", "y", "value", "owner", "hp"]]
        n_nodes = rng.randint(8, 15)
        for n in range(n_nodes):
            node_rows.append([
                str(n),
                str(round(rng.uniform(0, 20), 2)),
                str(round(rng.uniform(0, 20), 2)),
                str(round(rng.uniform(1, 5), 2)),
                str(rng.randint(-1, 1)),
                str(round(rng.uniform(50, 100), 1)),
            ])
        with open(input_dir / f"nodes_{i:03d}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(node_rows)

        # Edge file
        edge_rows = [["source", "target", "weight"]]
        n_edges = rng.randint(10, 25)
        for _ in range(n_edges):
            edge_rows.append([
                str(rng.randint(0, n_nodes - 1)),
                str(rng.randint(0, n_nodes - 1)),
                str(round(rng.uniform(0.1, 1.0), 4)),
            ])
        with open(input_dir / f"edges_{i:03d}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(edge_rows)


def _create_sample_tactical_data(input_dir: Path, seed: int) -> None:
    """Create sample tactical decision data for demonstration."""
    rng = random.Random(seed)
    action_types = ["move", "shoot", "occupy", "retreat", "hold"]
    # Create imbalanced distribution (minority classes)
    weights = [0.35, 0.30, 0.15, 0.10, 0.10]

    for i in range(3):
        header = [
            "operator_hp", "operator_atk", "operator_def", "operator_range",
            "nearest_enemy_dist", "nearest_city_dist", "city_value",
            "num_allies_nearby", "num_enemies_nearby", "step_ratio",
            "action_type"
        ]
        rows = [header]
        n_samples = rng.randint(100, 200)
        for _ in range(n_samples):
            action = rng.choices(action_types, weights=weights, k=1)[0]
            row = [
                str(round(rng.uniform(20, 100), 2)),
                str(round(rng.uniform(5, 20), 2)),
                str(round(rng.uniform(2, 12), 2)),
                str(rng.randint(1, 5)),
                str(round(rng.uniform(0, 10), 2)),
                str(round(rng.uniform(0, 8), 2)),
                str(round(rng.uniform(1, 5), 2)),
                str(rng.randint(0, 4)),
                str(rng.randint(0, 3)),
                str(round(rng.uniform(0, 1), 4)),
                action,
            ]
            rows.append(row)

        with open(input_dir / f"tactical_{i:03d}.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)


if __name__ == "__main__":
    main()
