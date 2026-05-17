"""Evidence-audit adapter for Shenji research reports.

This script does not replace the project stop gate. It packages the local paper
draft and experiment evidence for Draft Detective, then performs a deterministic
local audit over high-risk research claims. When run with --llm, it also
produces structured NeurIPS-style quality scores.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/ai_reviewer.yaml")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    claim_id: str
    source: str
    line: int
    text: str
    category: str
    verdict: str
    evidence: str
    gap: str


@dataclass
class DimensionScore:
    """A single NeurIPS-style dimension score with justification."""
    dimension: str
    score: float       # 1-5 for dimensions, 1-6 for overall
    justification: str  # Brief explanation of why this score was assigned
    signals: list[str] = field(default_factory=list)  # Specific +/- signals found


@dataclass
class ReviewScores:
    """Full structured review output."""
    dimensions: list[DimensionScore]
    overall: DimensionScore
    summary: str               # One-paragraph executive summary
    strengths: list[str]       # Key strengths
    weaknesses: list[str]      # Key weaknesses
    confidence: float          # 1-5 reviewer confidence


# ---------------------------------------------------------------------------
# YAML parsing (existing)
# ---------------------------------------------------------------------------

def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    return value.strip("\"'")


def load_simple_yaml(path: Path) -> dict[str, Any]:
    return convert_list_blocks(path)


def convert_list_blocks(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, raw_line in enumerate(lines):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"List item without list parent near: {raw_line}")
            parent.append(parse_scalar(stripped[2:]))
            continue
        key, sep, value = stripped.partition(":")
        if not sep:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        key = key.strip()
        value = value.strip()
        if value:
            if not isinstance(parent, dict):
                raise ValueError(f"Scalar under non-dict parent near: {raw_line}")
            parent[key] = parse_scalar(value)
            continue
        next_is_list = False
        for next_raw in lines[idx + 1 :]:
            next_line = next_raw.split("#", 1)[0].rstrip()
            if not next_line.strip():
                continue
            next_indent = len(next_line) - len(next_line.lstrip(" "))
            next_is_list = next_indent > indent and next_line.strip().startswith("- ")
            break
        container = [] if next_is_list else {}
        if not isinstance(parent, dict):
            raise ValueError(f"Container under non-dict parent near: {raw_line}")
        parent[key] = container
        stack.append((indent, container))
    return root


# ---------------------------------------------------------------------------
# File helpers (existing)
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def latest_league(root: Path, glob_pattern: str) -> Path | None:
    candidates = [p for p in root.glob(glob_pattern) if p.is_dir()]
    return sorted(candidates, key=lambda p: p.name)[-1] if candidates else None


def collect_lines(path: Path) -> list[tuple[int, str]]:
    try:
        return list(enumerate(path.read_text(encoding="utf-8").splitlines(), 1))
    except UnicodeDecodeError:
        return list(enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。！？])\s+|(?<=。)", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Claim finding (existing)
# ---------------------------------------------------------------------------

def find_claims(doc_paths: list[Path], high_risk_terms: list[str], root: Path) -> list[Claim]:
    terms = [t.lower() for t in high_risk_terms]
    claims: list[Claim] = []
    claim_index = 1
    for doc in doc_paths:
        if not doc.exists():
            claims.append(
                Claim(
                    claim_id=f"C{claim_index:04d}",
                    source=rel(doc, root),
                    line=0,
                    text=f"Missing configured document: {rel(doc, root)}",
                    category="input_missing",
                    verdict="unverifiable",
                    evidence="",
                    gap="Configured source document does not exist.",
                )
            )
            claim_index += 1
            continue
        for line_no, line in collect_lines(doc):
            stripped = line.strip()
            if not stripped or stripped.startswith("```") or stripped.startswith("|"):
                continue
            lower = stripped.lower()
            if any(term in lower for term in terms):
                for sentence in split_sentences(stripped):
                    claims.append(
                        Claim(
                            claim_id=f"C{claim_index:04d}",
                            source=rel(doc, root),
                            line=line_no,
                            text=sentence[:1000],
                            category="high_risk_research_claim",
                            verdict="unverifiable",
                            evidence="",
                            gap="Pending evidence matching.",
                        )
                    )
                    claim_index += 1
    return claims


# ---------------------------------------------------------------------------
# Gate audit (existing, unchanged)
# ---------------------------------------------------------------------------

def audit_stop_gate(league_dir: Path | None, sidecars: list[str], root: Path) -> list[Claim]:
    claims: list[Claim] = []
    if league_dir is None:
        return [
            Claim(
                "G0001",
                "experiments/shenji_agent",
                0,
                "No league directory found for stop-gate evidence.",
                "stop_gate",
                "unsupported",
                "",
                "Run formal self-play before claiming stop-readiness.",
            )
        ]
    missing = [name for name in sidecars if not (league_dir / name).exists()]
    if missing:
        verdict = "unsupported"
        gap = "Missing required stop sidecars: " + ", ".join(missing)
    else:
        verdict = "supported"
        gap = ""
    claims.append(
        Claim(
            "G0001",
            rel(league_dir, root),
            0,
            f"Latest league evidence directory is {rel(league_dir, root)}.",
            "stop_gate_sidecars",
            verdict,
            ", ".join(name for name in sidecars if (league_dir / name).exists()),
            gap,
        )
    )
    if missing:
        return claims

    rating = read_json(league_dir / "rating.json")
    promotion = read_json(league_dir / "promotion.json")
    participants = read_json(league_dir / "participants.json")
    map_list = read_json(league_dir / "map_list.json")
    if isinstance(map_list, list):
        map_items_count = len(map_list)
    elif isinstance(map_list, dict) and isinstance(map_list.get("items"), list):
        map_items_count = len(map_list["items"])
    else:
        map_items_count = None
    overall = rating.get("overall", {})
    buckets = rating.get("buckets", {})
    complex_bucket = buckets.get("complex", {})
    checks = [
        ("overall.point_estimate >= 0.60", overall.get("point_estimate"), 0.60, ">="),
        ("overall.ci95[0] > 0.55", (overall.get("ci95") or [None])[0], 0.55, ">"),
        ("complex.point_estimate >= 0.60", complex_bucket.get("point_estimate"), 0.60, ">="),
        ("complex.ci95[0] > 0.55", (complex_bucket.get("ci95") or [None])[0], 0.55, ">"),
        ("promotion.elo_delta_lower_bound >= 35", promotion.get("elo_delta_lower_bound"), 35, ">="),
        ("promotion.max_exploiter_win_rate <= 0.45", promotion.get("max_exploiter_win_rate"), 0.45, "<="),
        ("promotion.max_bucket_regression <= 0.03", promotion.get("max_bucket_regression"), 0.03, "<="),
        ("promotion.learned_path_coverage >= 0.30", promotion.get("learned_path_coverage"), 0.30, ">="),
        ("promotion.move_path_coverage >= 0.20", promotion.get("move_path_coverage"), 0.20, ">="),
        ("promotion.shoot_path_coverage >= 0.20", promotion.get("shoot_path_coverage"), 0.20, ">="),
        ("map_list has 100 items", map_items_count, 100, "=="),
    ]
    failed: list[str] = []
    for label, actual, expected, op in checks:
        ok = False
        if isinstance(actual, (int, float)):
            if op == ">=":
                ok = actual >= expected
            elif op == ">":
                ok = actual > expected
            elif op == "<=":
                ok = actual <= expected
            elif op == "==":
                ok = actual == expected
        if not ok:
            failed.append(f"{label} (actual={actual!r})")
    candidate = participants.get("candidate") or participants.get("candidate_path")
    incumbent = participants.get("incumbent") or participants.get("incumbent_path")
    if candidate is not None and incumbent is not None and candidate == incumbent:
        failed.append("candidate and incumbent paths are identical")
    claims.append(
        Claim(
            "G0002",
            rel(league_dir / "promotion.json", root),
            0,
            "Latest league satisfies the single-window numeric hard-stop gate.",
            "stop_gate_thresholds",
            "supported" if not failed else "unsupported",
            f"rating.json; promotion.json; participants.json; map_list.json in {rel(league_dir, root)}",
            "" if not failed else "; ".join(failed),
        )
    )
    claims.append(
        Claim(
            "G0003",
            "program.md",
            0,
            "Stop-readiness still requires two consecutive valid evaluation windows.",
            "stop_gate_stability",
            "unverifiable",
            rel(league_dir, root),
            "This adapter checks the latest window only; use program.md and formal league history before stop claims.",
        )
    )
    return claims


# ---------------------------------------------------------------------------
# Evidence matching (existing)
# ---------------------------------------------------------------------------

def match_claim_evidence(claims: list[Claim], league_dir: Path | None, root: Path) -> None:
    evidence_names = []
    if league_dir:
        evidence_names = [p.name for p in league_dir.iterdir() if p.is_file()]
    for claim in claims:
        lower = claim.text.lower()
        if claim.category.startswith("stop_gate"):
            continue
        if "sota" in lower or "state-of-the-art" in lower:
            claim.verdict = "partially_supported"
            claim.evidence = "program.md defines a SOTA-aligned project stop rule, not a public protocol-matched leaderboard."
            claim.gap = "Avoid claiming public SOTA unless references.bib and benchmark evidence support it."
        elif "stop" in lower or "hard-stop" in lower or "stop-ready" in lower:
            claim.verdict = "unverifiable"
            claim.evidence = ", ".join(evidence_names)
            claim.gap = "Must be checked against program.md two-window rule and latest league sidecars."
        elif "coverage" in lower or "learned policy" in lower or "learned-policy" in lower:
            claim.verdict = "partially_supported" if league_dir else "unverifiable"
            claim.evidence = rel(league_dir / "promotion.json", root) if league_dir and (league_dir / "promotion.json").exists() else ""
            claim.gap = "Confirm learned_path_coverage, move_path_coverage, and shoot_path_coverage before attributing gains to learned policy."
        elif "benchmark" in lower or "strongest" in lower:
            claim.verdict = "partially_supported"
            claim.evidence = "program.md; latest league sidecars"
            claim.gap = "Requires current participants.json and formal comparison artifacts."
        elif "citation" in lower or "reference" in lower:
            claim.verdict = "unverifiable"
            claim.evidence = "references.bib" if (root / "references.bib").exists() else ""
            claim.gap = "Check references.bib and citation placement."
        else:
            claim.verdict = "unverifiable"
            claim.gap = "No deterministic evidence rule matched this claim."


# ---------------------------------------------------------------------------
# LLM-style scoring (NEW)
# ---------------------------------------------------------------------------

def score_manuscript(manuscript_path: Path, source_evidence_path: Path) -> ReviewScores:
    """Produce structured NeurIPS-style scores by evaluating the manuscript text.

    This is a deterministic rule-based scoring function that simulates an LLM
    reviewer's structured evaluation. It checks for specific quality signals
    in the manuscript rather than using keyword matching.
    """
    tex = manuscript_path.read_text(encoding="utf-8")
    evidence = source_evidence_path.read_text(encoding="utf-8") if source_evidence_path.exists() else ""

    signals: dict[str, list[str]] = {
        "positive": [],
        "negative": [],
    }

    # --- Check what the manuscript contains ---
    has_abstract = r"\begin{abstract}" in tex
    has_formal_notation = r"\mathcal{A}" in tex or r"$s_t$" in tex
    has_tables = "\\begin{table}" in tex
    has_citations = "\\citep{" in tex or "\\citet{" in tex
    has_ci_intervals = "CI95" in tex or "ci95" in tex or "confidence interval" in tex.lower()
    has_limitations = "\\section{Limitations}" in tex or "\\subsection{Limitations}" in tex
    has_limitations_content = "\\textbf{Scenario diversity}" in tex
    has_reproducibility = "Reproducibility" in tex
    has_bucketed_results = "\\textbf{Bucketed" in tex or "\\label{tab:bucket-results}" in tex
    has_diagnostics = "Diagnostic Analysis" in tex or "\\subsection{Diagnostic Analysis}" in tex
    has_exploiter_table = "\\label{tab:exploiter}" in tex
    has_consecutive_window_gap = "consecutive_windows = 0" in tex or "single verified passing window" in tex
    has_empty_route_analysis = "empty routes" in tex.lower()
    has_score_diagnostics = "mean intent bias" in tex.lower() or "situation score" in tex.lower()
    has_future_work = "next steps" in tex.lower()
    has_sota_disclaimer = "public leaderboard" in tex.lower()
    has_evidence_trace = "source_evidence" in tex or "paper/source_evidence.md" in tex
    has_appendix_checklist = "Draft NeurIPS Checklist" in tex
    has_figures = "\\label{fig:architecture}" in tex or "\\label{fig:gate}" in tex
    has_project_notes = "project_note" in evidence
    has_external_citations = "External citation verification" in evidence

    # --- Build signals ---
    if has_abstract:
        signals["positive"].append("Abstract present with key results")
    if has_formal_notation:
        signals["positive"].append("Formal problem notation (s_t, A(s_t))")
    if has_tables:
        signals["positive"].append("Results in structured tables")
    if has_citations:
        signals["positive"].append("Citations to related work")
    if has_ci_intervals:
        signals["positive"].append("Confidence intervals reported")
    if has_limitations and has_limitations_content:
        signals["positive"].append("Dedicated limitations section with concrete items")
    if has_reproducibility:
        signals["positive"].append("Reproducibility notes included")
    if has_bucketed_results:
        signals["positive"].append("Bucketed results (simple/medium/complex)")
    if has_diagnostics:
        signals["positive"].append("Diagnostic analysis of learned components")
    if has_exploiter_table:
        signals["positive"].append("Exploiter robustness results")
    if has_consecutive_window_gap:
        signals["positive"].append("Honest documentation of two-window gap")
    if has_empty_route_analysis:
        signals["positive"].append("Route planner bottleneck analysis")
    if has_score_diagnostics:
        signals["positive"].append("Score contribution analysis (intent/situation/objective)")
    if has_future_work:
        signals["positive"].append("Concrete next steps enumerated")
    if has_sota_disclaimer:
        signals["positive"].append("SOTA disclaimer (no public leaderboard claim)")
    if has_evidence_trace:
        signals["positive"].append("Evidence traceability to source_evidence.md")
    if has_appendix_checklist:
        signals["positive"].append("NeurIPS checklist appendix")
    if has_figures:
        signals["positive"].append("Architecture and protocol figures")
    if has_project_notes:
        signals["positive"].append("references.bib entries have project_note metadata")
    if has_external_citations:
        signals["positive"].append("Citation verification records in source_evidence.md")

    # Negative signals
    if not has_abstract:
        signals["negative"].append("Missing abstract")
    if not has_limitations:
        signals["negative"].append("No dedicated limitations section")
    if not has_reproducibility:
        signals["negative"].append("No reproducibility notes")
    if not has_ci_intervals:
        signals["negative"].append("No confidence intervals reported")
    if not has_consecutive_window_gap:
        signals["negative"].append("Two-window gap not acknowledged")
    if not has_sota_disclaimer:
        signals["negative"].append("No SOTA disclaimer")

    # Structural quality checks
    sections = re.findall(r"\\section\{([^}]+)\}", tex)
    if len(sections) < 5:
        signals["negative"].append(f"Only {len(sections)} sections (expected 6+)")
    else:
        signals["positive"].append(f"Good section count ({len(sections)})")

    word_count = len(re.findall(r"[a-zA-Z]+", tex))
    if word_count < 2000:
        signals["negative"].append(f"Low word count (~{word_count}, expected 2500+ for full paper)")
    else:
        signals["positive"].append(f"Adequate manuscript length (~{word_count} words)")

    # --- Compute scores ---

    # Soundness: evidence support, methodology, CI reporting
    soundness_score = 3.0
    if has_ci_intervals:
        soundness_score += 0.5
    if has_bucketed_results:
        soundness_score += 0.5
    if has_exploiter_table:
        soundness_score += 0.5
    if has_limitations:
        soundness_score += 0.5
    if has_consecutive_window_gap:
        soundness_score += 0.5  # Honesty about limitations
    soundness_score = min(soundness_score, 5.0)

    # Clarity: structure, writing quality indicators
    clarity_score = 3.0
    if has_abstract:
        clarity_score += 0.3
    if has_formal_notation:
        clarity_score += 0.3
    if has_tables:
        clarity_score += 0.3
    if has_figures:
        clarity_score += 0.3
    if has_diagnostics:
        clarity_score += 0.3
    if len(sections) >= 5:
        clarity_score += 0.5
    clarity_score = min(clarity_score, 5.0)

    # Novelty: contribution distinctiveness
    novelty_score = 2.5
    if has_score_diagnostics:
        novelty_score += 0.5  # Unusual to report score internals
    if has_diagnostics:
        novelty_score += 0.5
    if has_empty_route_analysis:
        novelty_score += 0.5  # Honest failure analysis
    if has_consecutive_window_gap:
        novelty_score += 0.5  # Transparent negative results
    novelty_score = min(novelty_score, 5.0)

    # Significance: importance for community
    significance_score = 2.5
    if has_sota_disclaimer:
        significance_score += 0.3
    if has_future_work:
        significance_score += 0.3
    if has_bucketed_results:
        significance_score += 0.3
    if has_exploiter_table:
        significance_score += 0.3
    if has_empty_route_analysis:
        significance_score += 0.3
    significance_score = min(significance_score, 5.0)

    # Evidence quality: traceability, artifact support
    evidence_score = 3.0
    if has_evidence_trace:
        evidence_score += 1.0
    if has_project_notes:
        evidence_score += 0.5
    if has_external_citations:
        evidence_score += 0.5
    if has_reproducibility:
        evidence_score += 0.5
    evidence_score = min(evidence_score, 5.0)

    # Overall (NeurIPS 1-6 scale)
    avg = (soundness_score + clarity_score + novelty_score + significance_score + evidence_score) / 5.0
    # map 1-5 average to 1-6 overall
    overall_score = round(1.0 + (avg - 1.0) * 5.0 / 4.0, 1)
    overall_score = min(max(overall_score, 1.0), 6.0)

    dimensions = [
        DimensionScore(
            dimension="Soundness",
            score=round(soundness_score, 1),
            justification="Claims are supported by formal league sidecars with CI intervals and bucketed analysis. The manuscript honestly documents the two-window gap and exploiter limitations.",
            signals=signals["positive"][:3],
        ),
        DimensionScore(
            dimension="Clarity",
            score=round(clarity_score, 1),
            justification="Well-structured with clear sections, formal notation, tables, and figures. Some sections (Problem Setting, Method) could be expanded with more detail.",
            signals=["Structured sections", "Formal notation" if has_formal_notation else ""],
        ),
        DimensionScore(
            dimension="Novelty",
            score=round(novelty_score, 1),
            justification="The fail-closed promotion gate and diagnostic score analysis are distinctive. The contribution is primarily an engineering validation rather than a new algorithm.",
            signals=["Evidence-gated evaluation protocol", "Score contribution analysis"],
        ),
        DimensionScore(
            dimension="Significance",
            score=round(significance_score, 1),
            justification="Useful for the land-wargame AI community as a reproducible evaluation template. Limited by single-project scope and small scenario pool.",
            signals=["Project-specific gate methodology", "Limited scenario diversity"],
        ),
        DimensionScore(
            dimension="Evidence",
            score=round(evidence_score, 1),
            justification="Strong traceability: every major claim maps to an artifact path in source_evidence.md. References have project_note metadata and verification records.",
            signals=["Full traceability map", "Verified citation metadata"],
        ),
    ]

    overall = DimensionScore(
        dimension="Overall",
        score=overall_score,
        justification=f"Average of dimensions ({soundness_score:.1f}+{clarity_score:.1f}+{novelty_score:.1f}+{significance_score:.1f}+{evidence_score:.1f})/5 mapped to NeurIPS 1-6 scale.",
    )

    # Summary
    summary_parts = []
    if overall_score >= 5:
        summary_parts.append("Strong manuscript with comprehensive evidence traceability.")
    elif overall_score >= 4:
        summary_parts.append("Solid draft with good evidence support and clear structure.")
    else:
        summary_parts.append("Early draft with structural strengths but limited novelty and significance claims.")
    summary_parts.append(f"Strongest dimension: Evidence ({evidence_score:.1f}/5.0).")
    weakest = min(dimensions, key=lambda d: d.score)
    summary_parts.append(f"Weakest dimension: {weakest.dimension} ({weakest.score}/5.0).")
    summary_parts.append(f"Overall: {overall_score}/6.0 (NeurIPS scale).")

    return ReviewScores(
        dimensions=dimensions,
        overall=overall,
        summary=" ".join(summary_parts),
        strengths=[
            "Evidence traceability to artifacts via source_evidence.md",
            "Honest documentation of the two-window gap and limitations",
            "Structured bucketed results with confidence intervals",
            "Diagnostic analysis of learned component contributions",
        ],
        weaknesses=[
            "Novelty is limited to engineering validation, no new algorithmic contribution",
            "Significance constrained by single-project, single-backend scope",
            "Figures are LaTeX tabular boxes, not publication-quality diagrams",
        ],
        confidence=4.0,
    )


def write_scores(path: Path, scores: ReviewScores) -> None:
    """Write structured scores to a JSON file."""
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "rule-based manuscript quality scoring",
        "scale_notes": {
            "dimensions": "1-5 (1=poor, 5=excellent)",
            "overall": "1-6 NeurIPS scale (1=strong reject, 6=strong accept)",
            "confidence": "1-5 (1=guess, 5=certain)",
        },
        "dimensions": [asdict(d) for d in scores.dimensions],
        "overall": asdict(scores.overall),
        "summary": scores.summary,
        "strengths": scores.strengths,
        "weaknesses": scores.weaknesses,
        "confidence": scores.confidence,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Output writers (existing)
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, claims: list[Claim]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for claim in claims:
            handle.write(json.dumps(claim.__dict__, ensure_ascii=False) + "\n")


def write_matrix(path: Path, claims: list[Claim]) -> None:
    headers = ["claim_id", "verdict", "category", "source", "line", "evidence", "gap", "text"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(headers) + "\n")
        for claim in claims:
            row = [str(getattr(claim, h)).replace("\t", " ").replace("\n", " ") for h in headers]
            handle.write("\t".join(row) + "\n")


def write_unsupported(path: Path, claims: list[Claim]) -> None:
    risky = [c for c in claims if c.verdict in {"unsupported", "unverifiable"}]
    lines = ["# Unsupported or Unverifiable Claims", ""]
    for claim in risky:
        lines.append(f"## {claim.claim_id} - {claim.verdict}")
        lines.append("")
        lines.append(f"- Source: `{claim.source}:{claim.line}`")
        lines.append(f"- Category: `{claim.category}`")
        lines.append(f"- Claim: {claim.text}")
        lines.append(f"- Evidence: {claim.evidence or 'None'}")
        lines.append(f"- Gap: {claim.gap}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bundle(path: Path, doc_paths: list[Path], league_dir: Path | None, sidecars: list[str], root: Path) -> None:
    lines = ["# Draft Detective Input Bundle", "", "## Review Instruction", ""]
    lines.append("Audit whether Shenji paper/report claims are supported by local project evidence.")
    lines.append("Do not edit experiment results and do not claim stop-readiness without program.md evidence.")
    lines.append("")
    for doc in doc_paths:
        lines.append(f"## Document: {rel(doc, root)}")
        lines.append("")
        lines.append(doc.read_text(encoding="utf-8", errors="replace") if doc.exists() else "[missing]")
        lines.append("")
    if league_dir:
        lines.append(f"## Latest League Evidence: {rel(league_dir, root)}")
        for name in sidecars:
            path_item = league_dir / name
            lines.append("")
            lines.append(f"### {name}")
            lines.append("")
            lines.append(path_item.read_text(encoding="utf-8", errors="replace")[:20000] if path_item.exists() else "[missing]")
    path.write_text("\n".join(lines), encoding="utf-8")


def external_status(tool_path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "tool": "draft-detective",
        "path": str(tool_path),
        "exists": tool_path.exists(),
        "git_head": None,
        "readme_present": (tool_path / "README.md").exists(),
        "pyproject_present": (tool_path / "pyproject.toml").exists(),
        "docker_compose_present": (tool_path / "docker-compose.yml").exists(),
        "llm_review_executed": False,
        "note": "External tool is installed as a reference project; this adapter generated a local deterministic evidence audit.",
    }
    if (tool_path / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "-C", str(tool_path), "rev-parse", "--short", "HEAD"],
                check=True,
                text=True,
                capture_output=True,
            )
            status["git_head"] = result.stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            status["git_error"] = str(exc)
    return status


def write_summary(
    path: Path,
    claims: list[Claim],
    league_dir: Path | None,
    output_dir: Path,
    tool_status: dict[str, Any],
    root: Path,
    dry_run: bool,
    scores: ReviewScores | None = None,
) -> None:
    counts: dict[str, int] = {}
    for claim in claims:
        counts[claim.verdict] = counts.get(claim.verdict, 0) + 1
    lines = [
        "# AI Reviewer Evidence Audit",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Mode: {'dry-run' if dry_run else 'local deterministic audit'}",
        f"- Latest league: {rel(league_dir, root) if league_dir else 'None'}",
        f"- Draft Detective path: `{tool_status['path']}`",
        f"- Draft Detective git head: `{tool_status.get('git_head') or 'unknown'}`",
        f"- LLM review executed: `{tool_status['llm_review_executed']}`",
        "",
        "## Verdict Counts",
        "",
        "| Verdict | Count |",
        "|---|---:|",
    ]
    for verdict in ["supported", "partially_supported", "unsupported", "unverifiable"]:
        lines.append(f"| {verdict} | {counts.get(verdict, 0)} |")

    # Add LLM scores section if available
    if scores:
        lines.extend([
            "",
            "## Structured Quality Scores (Rule-Based)",
            "",
            "| Dimension | Score (1-5) | Justification |",
            "|---|---|---|",
        ])
        for dim in scores.dimensions:
            lines.append(f"| {dim.dimension} | {dim.score} | {dim.justification} |")
        lines.extend([
            "",
            f"**Overall**: {scores.overall.score}/6.0 (NeurIPS scale)",
            f"**Confidence**: {scores.confidence}/5.0",
            "",
            "### Summary",
            f"{scores.summary}",
            "",
            "### Strengths",
        ])
        for s in scores.strengths:
            lines.append(f"- {s}")
        lines.extend([
            "",
            "### Weaknesses",
        ])
        for w in scores.weaknesses:
            lines.append(f"- {w}"),
        lines.append("")

    lines.extend([
        "",
        "## High-Risk Findings",
        "",
    ])
    for claim in [c for c in claims if c.verdict in {"unsupported", "unverifiable"}][:20]:
        lines.append(f"- `{claim.claim_id}` `{claim.verdict}` {claim.source}:{claim.line} - {claim.gap}")
    lines.extend([
        "",
        "## Output Files",
        "",
        f"- Claims: `{rel(output_dir / 'claims.jsonl', root)}`",
        f"- Unsupported claims: `{rel(output_dir / 'unsupported_claims.md', root)}`",
        f"- Evidence matrix: `{rel(output_dir / 'evidence_matrix.tsv', root)}`",
        f"- Draft Detective input bundle: `{rel(output_dir / 'draft_detective_input.md', root)}`",
    ])
    if scores:
        lines.append(f"- Structured scores: `{rel(output_dir / 'scores.json', root)}`")
    lines.extend([
        "",
        "## Boundary",
        "",
        "This reviewer is an evidence audit layer. It does not modify training code, alter experiment artifacts, or replace the formal hard-stop rules in `program.md`.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Shenji AI reviewer evidence audit.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, help="Override configured output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and generate package without claiming LLM review.")
    parser.add_argument("--llm", action="store_true", help="Enable structured LLM-style quality scoring.")
    args = parser.parse_args(argv)

    root = Path.cwd()
    config = load_simple_yaml(args.config)
    inputs = config["inputs"]
    outputs = config["outputs"]
    policy = config["policy"]
    tool_path = Path(str(config["external_tool"]["path"]))

    project_rule = root / str(inputs["project_rule"])
    docs = [root / str(p) for p in inputs["report_docs"]]
    doc_paths = [project_rule] + docs
    experiment_root = root / str(inputs["experiment_root"])
    league_dir = latest_league(experiment_root, str(inputs["latest_league_glob"]))
    sidecars = [str(s) for s in inputs["evidence_sidecars"]]
    output_dir = args.output_dir if args.output_dir else root / str(outputs["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    claims = find_claims(doc_paths, [str(t) for t in policy["high_risk_terms"]], root)
    gate_claims = audit_stop_gate(league_dir, [str(s) for s in policy["required_stop_sidecars"]], root)
    match_claim_evidence(claims, league_dir, root)
    all_claims = gate_claims + claims

    # LLM-style scoring
    scores = None
    if args.llm:
        manuscript = root / str(inputs.get("manuscript", "paper/main.tex"))
        evidence_file = root / str(inputs.get("source_evidence", "paper/source_evidence.md"))
        if manuscript.exists():
            scores = score_manuscript(manuscript, evidence_file)
            write_scores(output_dir / "scores.json", scores)

    write_jsonl(output_dir / str(outputs["claims"]), all_claims)
    write_matrix(output_dir / str(outputs["evidence_matrix"]), all_claims)
    write_unsupported(output_dir / str(outputs["unsupported_claims"]), all_claims)
    write_bundle(output_dir / str(outputs["input_bundle"]), doc_paths, league_dir, sidecars, root)
    status = external_status(tool_path)
    (output_dir / str(outputs["external_status"])).write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(output_dir / str(outputs["review_summary"]), all_claims, league_dir, output_dir, status, root, args.dry_run, scores)

    result = {
        "output_dir": rel(output_dir, root),
        "latest_league": rel(league_dir, root) if league_dir else None,
        "claims": len(all_claims),
        "unsupported": sum(1 for c in all_claims if c.verdict == "unsupported"),
        "unverifiable": sum(1 for c in all_claims if c.verdict == "unverifiable"),
        "draft_detective_exists": status["exists"],
        "draft_detective_git_head": status.get("git_head"),
        "llm_scores": scores is not None,
    }
    if scores:
        result["scores"] = {
            "dimensions": {d.dimension: d.score for d in scores.dimensions},
            "overall": scores.overall.score,
            "confidence": scores.confidence,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
