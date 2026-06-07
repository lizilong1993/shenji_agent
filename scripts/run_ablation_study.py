#!/usr/bin/env python3
"""Run a full ablation study across 4 modes and generate a comparison report.

Modes evaluated:
  - ``full``          : All learned models enabled (baseline for comparison).
  - ``no_intent``     : Intent prediction model disabled.
  - ``no_situation``  : Situation assessment (GCN) model disabled.
  - ``rule_only``     : All learned models disabled (pure rule-based agent).

For each mode the script:
  1. Invokes the self-play evaluation pipeline via subprocess.
  2. Parses the resulting ``promotion.json`` and ``rating.json`` artifacts.
  3. Collects key metrics into a comparison table.

Results are saved to ``experiments/shenji_agent/ablation_comparison/``.

Usage
-----
::

    python scripts/run_ablation_study.py
    python scripts/run_ablation_study.py --config configs/selfplay_ablation.yaml
    python scripts/run_ablation_study.py --candidate ai/weights --incumbent ai/weights_incumbent_rule
    python scripts/run_ablation_study.py --modes full rule_only   # subset
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABLATION_MODES = ("full", "no_intent", "no_situation", "rule_only")

DEFAULT_CONFIG = "configs/selfplay_ablation.yaml"
DEFAULT_CANDIDATE = "ai/weights"
DEFAULT_INCUMBENT = "ai/weights_incumbent_rule"

OUTPUT_ROOT = "experiments/shenji_agent/ablation_comparison"

PYTHON_BIN = "python3.10"

# Metrics we extract from each run
METRIC_KEYS = [
    "overall_point_estimate",
    "overall_ci95_lower",
    "complex_point_estimate",
    "complex_ci95_lower",
    "learned_path_coverage",
    "move_path_coverage",
    "shoot_path_coverage",
    "elo_delta_lower_bound",
    "promotion_accepted",
    "total_matches",
    "candidate_wins",
    "incumbent_wins",
    "draws",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    """Return the repository root (parent of ``scripts/``)."""
    return Path(__file__).resolve().parent.parent


def _find_latest_run_dir(experiments_root: Path, prefix: str = "league_") -> Optional[Path]:
    """Find the most recently created run directory under *experiments_root*."""
    if not experiments_root.exists():
        return None
    candidates = sorted(
        (d for d in experiments_root.iterdir() if d.is_dir() and d.name.startswith(prefix)),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def run_single_ablation(
    mode: str,
    *,
    config: str,
    candidate: str,
    incumbent: str,
    extra_args: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single ablation experiment and return parsed results.

    Parameters
    ----------
    mode:
        The ablation mode to test (e.g. ``"full"``, ``"no_intent"``).
    config:
        Path to the selfplay config YAML (relative to repo root).
    candidate:
        Path to candidate artifact directory.
    incumbent:
        Path to incumbent artifact directory.
    extra_args:
        Additional CLI arguments to forward to the selfplay command.
    dry_run:
        If True, print the command but do not execute it.

    Returns
    -------
    dict
        Parsed results including all ``METRIC_KEYS`` plus metadata.
    """
    root = repo_root()
    cmd = [
        PYTHON_BIN,
        "-m",
        "ablation_selfplay_patch",
        "--config", config,
        "--candidate", candidate,
        "--incumbent", incumbent,
        "--ablation-mode", mode,
        "--allow-self-match-smoke",
    ]
    if extra_args:
        cmd.extend(extra_args)

    result: Dict[str, Any] = {
        "ablation_mode": mode,
        "command": " ".join(cmd),
        "status": "pending",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": 0.0,
        "return_code": None,
    }

    log.info("=" * 72)
    log.info("ABLATION MODE: %s", mode)
    log.info("COMMAND: %s", result["command"])
    log.info("=" * 72)

    if dry_run:
        result["status"] = "dry_run"
        return result

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per mode
        )
        elapsed = time.monotonic() - t0
        result["duration_seconds"] = round(elapsed, 1)
        result["return_code"] = proc.returncode
        result["stdout_tail"] = proc.stdout[-3000:] if proc.stdout else ""
        result["stderr_tail"] = proc.stderr[-3000:] if proc.stderr else ""

        if proc.returncode != 0:
            result["status"] = "failed"
            log.error(
                "Mode %s FAILED (rc=%d):\nSTDOUT tail:\n%s\nSTDERR tail:\n%s",
                mode, proc.returncode, result["stdout_tail"], result["stderr_tail"],
            )
            return result

        # Locate the run directory (most recent league_* under output_root)
        experiments_root = root / "experiments" / "shenji_agent"
        run_dir = _find_latest_run_dir(experiments_root)
        if run_dir is None:
            result["status"] = "no_output_dir"
            log.error("Could not find output directory for mode %s", mode)
            return result

        result["run_dir"] = str(run_dir)

        # Parse rating.json
        rating_path = run_dir / "rating.json"
        if rating_path.exists():
            rating = json.loads(rating_path.read_text(encoding="utf-8"))
            result["overall_point_estimate"] = _safe_float(rating, "overall", "point_estimate")
            result["overall_ci95_lower"] = _safe_float(rating, "overall", "ci95", index=0)
            buckets = rating.get("buckets", {})
            result["complex_point_estimate"] = _safe_float(buckets.get("complex", {}), "point_estimate")
            result["complex_ci95_lower"] = _safe_float(buckets.get("complex", {}), "ci95", index=0)
            policy_stats = rating.get("policy", {})
            result["learned_path_coverage"] = policy_stats.get("learned_path_coverage", 0.0)
            result["move_path_coverage"] = policy_stats.get("move_path_coverage", 0.0)
            result["shoot_path_coverage"] = policy_stats.get("shoot_path_coverage", 0.0)

        # Parse promotion.json
        promotion_path = run_dir / "promotion.json"
        if promotion_path.exists():
            promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
            result["elo_delta_lower_bound"] = promotion.get("elo_delta_lower_bound", 0.0)
            result["promotion_accepted"] = promotion.get("accepted", False)

        # Parse match_results.json for win/loss/draw counts
        match_results_path = run_dir / "match_results.json"
        if match_results_path.exists():
            match_data = json.loads(match_results_path.read_text(encoding="utf-8"))
            matches_list = match_data.get("matches", [])
            result["total_matches"] = len(matches_list)
            candidate_wins = sum(1 for m in matches_list if m.get("winner") == "candidate")
            incumbent_wins = sum(1 for m in matches_list if m.get("winner") == "incumbent")
            draws = len(matches_list) - candidate_wins - incumbent_wins
            result["candidate_wins"] = candidate_wins
            result["incumbent_wins"] = incumbent_wins
            result["draws"] = draws

        result["status"] = "success"
        log.info("Mode %s completed in %.1fs", mode, elapsed)

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["duration_seconds"] = round(time.monotonic() - t0, 1)
        log.error("Mode %s TIMED OUT after %.1fs", mode, result["duration_seconds"])
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["duration_seconds"] = round(time.monotonic() - t0, 1)
        log.error("Mode %s raised exception: %s", mode, exc)

    return result


def _safe_float(data: Any, *keys: str, index: Optional[int] = None) -> float:
    """Safely extract a float from a nested dict structure."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return 0.0
    if index is not None and isinstance(current, (list, tuple)):
        if 0 <= index < len(current):
            current = current[index]
        else:
            return 0.0
    try:
        return float(current) if current is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_comparison_table(results: List[Dict[str, Any]]) -> str:
    """Generate a Markdown comparison table from ablation results.

    Parameters
    ----------
    results:
        List of result dicts, one per ablation mode.

    Returns
    -------
    str
        Markdown-formatted table.
    """
    lines: List[str] = []
    lines.append("# Shenji Agent Ablation Study Results")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

    # Summary table
    lines.append("## Summary Comparison")
    lines.append("")
    header = "| Mode | Win Rate | CI95 Lower | Complex WR | Learned Cov | Move Cov | Shoot Cov | Elo Delta | Promoted | Status |"
    separator = "|------|----------|------------|------------|-------------|----------|-----------|-----------|----------|--------|"
    lines.append(header)
    lines.append(separator)

    for r in results:
        mode = r.get("ablation_mode", "?")
        overall_pe = r.get("overall_point_estimate", 0.0)
        overall_ci = r.get("overall_ci95_lower", 0.0)
        complex_pe = r.get("complex_point_estimate", 0.0)
        learned_cov = r.get("learned_path_coverage", 0.0)
        move_cov = r.get("move_path_coverage", 0.0)
        shoot_cov = r.get("shoot_path_coverage", 0.0)
        elo_delta = r.get("elo_delta_lower_bound", 0.0)
        promoted = r.get("promotion_accepted", False)
        status = r.get("status", "unknown")

        lines.append(
            f"| {mode} "
            f"| {overall_pe:.4f} "
            f"| {overall_ci:.4f} "
            f"| {complex_pe:.4f} "
            f"| {learned_cov:.4f} "
            f"| {move_cov:.4f} "
            f"| {shoot_cov:.4f} "
            f"| {elo_delta:.2f} "
            f"| {'Yes' if promoted else 'No'} "
            f"| {status} |"
        )

    lines.append("")

    # Detailed per-mode sections
    lines.append("## Detailed Results")
    lines.append("")
    for r in results:
        mode = r.get("ablation_mode", "?")
        lines.append(f"### Mode: {mode}")
        lines.append("")
        lines.append(f"- **Status**: {r.get('status', 'unknown')}")
        lines.append(f"- **Duration**: {r.get('duration_seconds', 0):.1f}s")
        lines.append(f"- **Return code**: {r.get('return_code', 'N/A')}")
        if r.get("run_dir"):
            lines.append(f"- **Run directory**: `{r['run_dir']}`")
        lines.append(f"- **Total matches**: {r.get('total_matches', 'N/A')}")
        lines.append(f"- **Candidate wins**: {r.get('candidate_wins', 'N/A')}")
        lines.append(f"- **Incumbent wins**: {r.get('incumbent_wins', 'N/A')}")
        lines.append(f"- **Draws**: {r.get('draws', 'N/A')}")
        lines.append("")
        lines.append("Metrics:")
        lines.append("")
        for key in METRIC_KEYS:
            value = r.get(key, "N/A")
            if isinstance(value, float):
                lines.append(f"  - `{key}`: {value:.4f}")
            else:
                lines.append(f"  - `{key}`: {value}")
        lines.append("")

    # Analysis section
    lines.append("## Analysis")
    lines.append("")
    full_result = next((r for r in results if r.get("ablation_mode") == "full"), None)
    if full_result and full_result.get("status") == "success":
        full_wr = full_result.get("overall_point_estimate", 0.0)
        lines.append(f"Baseline (full model) win rate: **{full_wr:.4f}**")
        lines.append("")
        lines.append("| Mode | Delta vs Full | Interpretation |")
        lines.append("|------|---------------|----------------|")
        for r in results:
            mode = r.get("ablation_mode", "?")
            if mode == "full":
                continue
            mode_wr = r.get("overall_point_estimate", 0.0)
            delta = mode_wr - full_wr
            if abs(delta) < 0.02:
                interpretation = "Negligible impact"
            elif delta < -0.10:
                interpretation = "Critical component"
            elif delta < -0.05:
                interpretation = "Important component"
            elif delta < 0:
                interpretation = "Minor contribution"
            else:
                interpretation = "Removal improves performance (possible overfitting)"
            lines.append(f"| {mode} | {delta:+.4f} | {interpretation} |")
    else:
        lines.append("Full model baseline not available for comparison analysis.")
    lines.append("")

    return "\n".join(lines)


def generate_csv_results(results: List[Dict[str, Any]]) -> str:
    """Generate CSV output for programmatic consumption."""
    header = "ablation_mode," + ",".join(METRIC_KEYS) + ",status,duration_seconds"
    rows = [header]
    for r in results:
        values = [str(r.get("ablation_mode", ""))]
        for key in METRIC_KEYS:
            val = r.get(key, "")
            if isinstance(val, bool):
                values.append(str(val))
            elif isinstance(val, float):
                values.append(f"{val:.6f}")
            else:
                values.append(str(val))
        values.append(str(r.get("status", "")))
        values.append(f"{r.get('duration_seconds', 0):.1f}")
        rows.append(",".join(values))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Shenji Agent ablation study across multiple modes.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Config YAML path relative to repo root (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--candidate",
        default=DEFAULT_CANDIDATE,
        help=f"Candidate artifact directory (default: {DEFAULT_CANDIDATE}).",
    )
    parser.add_argument(
        "--incumbent",
        default=DEFAULT_INCUMBENT,
        help=f"Incumbent artifact directory (default: {DEFAULT_INCUMBENT}).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(ABLATION_MODES),
        choices=list(ABLATION_MODES),
        help="Subset of ablation modes to run (default: all 4).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: experiments/shenji_agent/ablation_comparison/<timestamp>).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=None,
        help="Extra arguments forwarded to the selfplay command (can be repeated).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    root = repo_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = root / OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Ablation study output directory: %s", output_dir)
    log.info("Modes to run: %s", args.modes)

    results: List[Dict[str, Any]] = []
    for mode in args.modes:
        result = run_single_ablation(
            mode,
            config=args.config,
            candidate=args.candidate,
            incumbent=args.incumbent,
            extra_args=args.extra_arg,
            dry_run=args.dry_run,
        )
        results.append(result)

        # Save intermediate result after each mode
        intermediate_path = output_dir / "results_intermediate.json"
        intermediate_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Save final results
    results_json_path = output_dir / "ablation_results.json"
    results_json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Generate Markdown report
    markdown_report = generate_comparison_table(results)
    markdown_path = output_dir / "ablation_comparison.md"
    markdown_path.write_text(markdown_report, encoding="utf-8")

    # Generate CSV
    csv_report = generate_csv_results(results)
    csv_path = output_dir / "ablation_comparison.csv"
    csv_path.write_text(csv_report, encoding="utf-8")

    log.info("=" * 72)
    log.info("RESULTS SAVED:")
    log.info("  JSON:     %s", results_json_path)
    log.info("  Markdown: %s", markdown_path)
    log.info("  CSV:      %s", csv_path)
    log.info("=" * 72)

    # Print the markdown table to stdout as well
    print("\n" + markdown_report)


if __name__ == "__main__":
    main()
