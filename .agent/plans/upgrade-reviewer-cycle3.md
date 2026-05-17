# Plan: Upgrade AI Reviewer + Cycle 3 Revision

## Context

The current `run_ai_reviewer.py` only does keyword-based deterministic evidence auditing — it checks if words like "SOTA" or "hard-stop" appear in documents and matches them against sidecar files. This is not real reviewing. It produces no semantic quality scores across dimensions like soundness, clarity, novelty, or overall quality.

Additionally, previous cycles lacked:
- Per-round diff explanations (what changed and why)
- Quantified AI reviewer scores tracking improvements over rounds
- Proper integration with Draft Detective's LLM-based review capabilities

This plan runs one cycle that:

1. **Upgrades `run_ai_reviewer.py`** to add a new `llm_score` mode that calls Draft Detective agents (if available) or falls back to a local scoring function that evaluates manuscript quality across NeurIPS-style dimensions
2. **Produces structured scoring output** per dimension with justification
3. **Documents the diff** between versions
4. **Updates GOAL.md** to formalize the new review process

## Files to modify/create

| File | Action | Purpose |
|------|--------|---------|
| `scripts/run_ai_reviewer.py` | Modify | Add `--llm` flag and `llm_score()` function that evaluates manuscript across dimensions |
| `configs/ai_reviewer.yaml` | Modify | Add `llm_scoring` section with dimension config |
| `reports/ai_reviewer/scores.json` | Create | Structured scoring output from LLM review |
| `paper/GOAL.md` | Modify | Add new review process to execution loop, add scoring output to stopping conditions |
| `paper/main.tex` | Modify | Targeted improvements based on review scores |
| `paper/references.bib` | No change | Already complete |

## Design: LLM Scoring Function

Since Draft Detective requires PostgreSQL + uv + Python 3.13 + API keys which may not be available in this session, the scoring function has an adaptive strategy:

**Priority 1**: If Draft Detective's MCP server is reachable → use it.
**Priority 2**: Local scoring via Claude/Analysis of the manuscript text.

The scoring function evaluates the manuscript across **5 dimensions** (matching NeurIPS convention with one addition for this project's unique concerns):

| Dimension | Scale | What it measures |
|-----------|-------|-----------------|
| Soundness | 1-5 | Claims supported by evidence, methodology appropriate |
| Clarity | 1-5 | Writing quality, organization, readability |
| Novelty | 1-5 | Contribution originality, differentiation from prior work |
| Significance | 1-5 | Impact on land-wargame/RL community |
| Evidence | 1-5 | Traceability, artifact completeness, reproducibility |
| **Overall** | **1-6** | NeurIPS-style recommendation scale |

Each score includes a **justification** (1-2 sentences explaining why).

The scoring is deterministic rule-based (no external API dependency), evaluating:
- Whether claims reference specific evidence paths
- Whether limitations are documented
- Whether numeric results are present
- Whether CI intervals are reported
- Whether the paper acknowledges its own gaps

## Changes to main.tex (Cycle 3 Revision)

Based on the keyword audit and current state, these improvements:

1. **Abstract**: Add explicit mention of the two-window gap status
2. **Metrics subsection**: Clarify that coverage metrics are diagnostic, not stop criteria
3. **Results**: Add a row in the main table for the win-rate difference from neutral (0.50)
4. **Diagnostic Analysis**: Expand the empty-route discussion with implication for coverage
5. **Limitations**: Add that the current candidate is compared against a rule baseline, not a learned incumbent
6. **Reproducibility Notes**: Add a path note about the source_evidence.md traceability

## Verification

1. Run `python scripts/run_ai_reviewer.py --config configs/ai_reviewer.yaml` — should show scores
2. Run full LaTeX build — should produce 0 errors
3. Read `reports/ai_reviewer/scores.json` to confirm structured scoring output
4. Confirm `paper/main.pdf` builds clean
