# Shenji paper optimization goal

## Durable objective

Use `/goal follow paper/GOAL.md` to keep improving the Shenji paper until the manuscript is evidence-backed, readable, and locally buildable.

The goal is to revise `paper/main.tex`, citations, and figures through repeated AI review and writing passes. Do not change training code, experiment artifacts, or formal stop criteria unless a separate task explicitly asks for that work.

## Code isolation

All paper-related code and configuration lives under `paper/`. The project root contains experiment, training, and evaluation code. Do not add paper-specific scripts, configs, or reports outside `paper/`.

- `paper/scripts/run_ai_reviewer.py` — paper AI reviewer
- `paper/configs/ai_reviewer.yaml` — reviewer configuration
- `paper/reports/ai_reviewer/` — reviewer outputs (separate from root `reports/`)
- `paper/figures/` — figure source files and generated images

The root `scripts/run_ai_reviewer.py` is a backward-compatible wrapper that delegates to `paper/scripts/run_ai_reviewer.py`.

## Paper versioning

Every revision of the paper must be saved as a versioned copy under `paper/archive/`. This ensures full traceability of the writing process.

**Major revision** — use `v1`, `v2`, `v3`, etc.:
- A full AI review + revision pass.
- Structural changes (new sections, major rewrites, figure changes).
- New experiment data integrated into the manuscript.

**Minor revision** — use `v1.1`, `v1.2`, etc.:
- Small edits (wording polish, citation fixes, minor corrections).
- humanizer-only passes.
- Formatting or LaTeX fixes that do not alter content claims.

**How to save a version:**

When a revision is complete and verified (builds clean, reviewer passes, humanizer applied):

1. Create or ensure `paper/archive/` exists.
2. Copy the full paper snapshot:
   ```powershell
   $v = "v1"  # update this each time
   New-Item -ItemType Directory -Force -Path paper\archive\$v
   Copy-Item paper\main.tex paper\archive\$v\
   Copy-Item paper\references.bib paper\archive\$v\
   Copy-Item paper\source_evidence.md paper\archive\$v\
   Copy-Item paper\main.pdf paper\archive\$v\
   robocopy paper\figures paper\archive\$v\figures /E > $null
   ```
3. Commit the version alongside the paper changes.
4. Update `paper/CHANGELOG.md` with a brief entry: version number, date, and what changed.

**Initial version**: the first saved version is `v1`. The raw manuscript before any revision cycle is considered `v0` (do not archive).

## Read first

Before planning or editing, read these files:

- `AGENTS.md`
- `.agent/context.md`
- `program.md`
- `paper/main.tex`
- `paper/source_evidence.md`
- `paper/references.bib`
- latest `experiments/shenji_agent/league_*/summary.md`
- latest `experiments/shenji_agent/league_*/rating.json`
- latest `experiments/shenji_agent/league_*/promotion.json`
- latest `experiments/shenji_agent/league_*/exploitability.json`
- latest `experiments/shenji_agent/league_*/participants.json`
- latest `experiments/shenji_agent/league_*/map_list.json`

Use the latest complete `league_*` directory with the required sidecars. If more than one candidate exists, document which run was chosen and why.

## Source of truth

- `program.md` and `configs/selfplay_sota_stop.yaml` define stop and SOTA-style boundaries.
- Formal evaluation claims must be supported by league sidecars, not README text, old plans, mock results, or single training losses.
- `paper/source_evidence.md` is the paper traceability map. Update it whenever the manuscript gains or changes a claim source.
- Do not claim a public SOTA result unless a protocol-matched public benchmark and verified comparison exist.
- Unknown or unverified results must be written as unknown or unverified.

## Execution loop

Run this loop until all stopping conditions are met or a hard blocker is documented.

1. Inspect the current paper state.

   ```powershell
   Get-Content -Encoding UTF8 paper\main.tex -Raw
   Get-Content -Encoding UTF8 paper\source_evidence.md -Raw
   Get-Content -Encoding UTF8 paper\references.bib -Raw
   ```

2. Run the local AI reviewer with structured scoring.

   ```powershell
   cd paper
   python scripts\run_ai_reviewer.py --config configs\ai_reviewer.yaml --llm
   ```

   Review these outputs:

   - `reports/ai_reviewer/review_summary.md` — claim audit results + structured quality scores
   - `reports/ai_reviewer/scores.json` — quantitative scores (JSON, machine-readable)
   - `reports/ai_reviewer/unsupported_claims.md` — claims without evidence support
   - `reports/ai_reviewer/evidence_matrix.tsv` — full claim-evidence matrix
   - `reports/ai_reviewer/claims.jsonl` — all extracted claims
   - `reports/ai_reviewer/draft_detective_input.md` — input bundle for Draft Detective
   - `reports/ai_reviewer/external_status.json` — Draft Detective availability

   **Scoring dimensions** (NeurIPS-inspired, rule-based):

   | Dimension | Scale | Meaning |
   |-----------|-------|---------|
   | Soundness | 1-5 | Claims supported by evidence, methodology sound |
   | Clarity | 1-5 | Writing quality, organization, readability |
   | Novelty | 1-5 | Contribution originality, differentiation |
   | Significance | 1-5 | Community impact, generalizability |
   | Evidence | 1-5 | Traceability, artifact completeness |
   | **Overall** | **1-6** | NeurIPS recommendation scale |

   Each score includes a justification and the signals that contributed to it.

3. Report per-round diff and score.

   Each revision round must document in `paper/CHANGELOG.md`:

   - **What changed**: precise list of modifications with file paths and line-level descriptions.
   - **Why**: rationale for each change (which score dimension it targets, which reviewer finding it addresses).
   - **Score delta**: AI reviewer scores before and after the revision, to show improvement.

   Format:
   ```
   | vX | YYYY-MM-DD | Score: Soundness X→Y, Clarity X→Y, Novelty X→Y, Significance X→Y, Evidence X→Y, Overall X→Y. Changes: ... |
   ```

4. Revise the manuscript.

   - Remove or qualify unsupported claims.
   - Replace vague claims with artifact-backed statements.
   - Keep claims, metrics, paths, citations, and limitations consistent with `paper/source_evidence.md`.
   - Keep LaTeX structure valid.
   - Do not edit generated `.aux`, `.bbl`, `.blg`, `.log`, `.out`, or `.pdf` files by hand.

5. Apply `humanizer` after each paper-generation or paper-revision pass.

   Use the local `humanizer` skill on prose sections of `paper/main.tex` after each generated draft or substantial revision. Remove AI-sounding patterns such as inflated significance, generic conclusions, repetitive three-part phrasing, vague attribution, excessive hedging, mechanical signposting, and promotional wording.

   Preserve:

   - technical meaning;
   - numerical results;
   - citations and BibTeX keys;
   - artifact paths;
   - LaTeX commands and environments;
   - formal claim boundaries;
   - cautious wording for unverified evidence.

   If `humanizer` is unavailable, use the closest local de-GPT-ify writing cleanup rules and document the fallback.

6. Search, verify, and record related papers.

   Every verified paper found during related-work search that is relevant to this manuscript must be recorded in `paper/references.bib`.

   Do not invent BibTeX metadata. Retrieve entries from a reliable source such as DOI/Crossref, arXiv, Semantic Scholar, an official publisher page, or another verifiable metadata source. If metadata cannot be verified, do not add the paper to `paper/references.bib`; list it in the progress notes as an unverified candidate.

   Each relevant entry must include this field:

   ```bibtex
   project_note = {Relevance: ...; Inspiration: ...; Relation: ...; Use in paper: ...; Status: ...}
   ```

   Field meanings:

   - `Relevance`: why the paper belongs in this Shenji bibliography.
   - `Inspiration`: what idea, framing, method, benchmark, or writing move it contributes.
   - `Relation`: how it relates to Shenji, such as baseline, adjacent benchmark, methods background, reviewer-system context, or contrast case.
   - `Use in paper`: where or how to cite it in `paper/main.tex`.
   - `Status`: verified metadata, background only, candidate comparison, not a stop-gate authority, or another precise status.

7. Use PaperOrchestra if available.

   First check whether a real PaperOrchestra skill exists:

   ```powershell
   Test-Path C:\Users\lizilong\.agents\skills\paper-orchestra\SKILL.md
   Test-Path C:\Users\lizilong\paper-orchestra\skills\paper-orchestra\SKILL.md
   ```

   If a real `SKILL.md` exists, use it for paper structure, reviewer-style critique, and revision planning.

   If PaperOrchestra is unavailable, use:

   ```text
   C:\Users\lizilong\.orchestra\skills\20-ml-paper-writing\SKILL.md
   ```

   Follow its citation-verification rules. Never fabricate references.

8. Improve or generate figures.

   Existing figure sources:

   - `paper/figures/system_architecture.tex`
   - `paper/figures/promotion_gate.tex`

   Try callable image-generation tools before falling back to manual prompts:

   - any local "paper banana" or paper-figure generation workflow;
   - any verified OpenAI image tooling available in the environment;
   - any verified `gpt image 2` or successor image model access.

   Do not assume image tooling is available. Probe it first and record the result. If no callable image tool works, produce manual prompts and require the generated files to be saved under `paper/figures/` as `.jpg`.

9. Rebuild and verify the paper.

   ```powershell
   cd paper
   pdflatex -interaction=nonstopmode main.tex
   bibtex main
   pdflatex -interaction=nonstopmode main.tex
   pdflatex -interaction=nonstopmode main.tex
   ```

   Check:

   - `paper/main.pdf` exists;
   - `paper/main.log` has no fatal LaTeX errors;
   - citations are resolved;
   - references are resolved;
   - the AI reviewer no longer reports unsupported claims, or remaining gaps are documented.

10. Archive versioned snapshot.

    After a successful build and before committing, save a versioned copy of the paper (see [Paper versioning](#paper-versioning) for naming rules):

    ```powershell
    # Determine version number — check paper/archive/ for the latest, then increment
    $v = "v3"
    New-Item -ItemType Directory -Force -Path paper\archive\$v
    Copy-Item paper\main.tex paper\archive\$v\
    Copy-Item paper\references.bib paper\archive\$v\
    Copy-Item paper\source_evidence.md paper\archive\$v\
    Copy-Item paper\main.pdf paper\archive\$v\
    robocopy paper\figures paper\archive\$v\figures /E > $null
    ```

11. Commit when safe.

    ```powershell
    git status --short --branch
    git add paper/GOAL.md paper/main.tex paper/source_evidence.md paper/references.bib paper/figures paper/archive/ paper/scripts/ paper/configs/
    git commit -m "Improve Shenji paper draft (v<N>)"
    ```

    Only stage files actually changed for the paper task. Do not stage unrelated user changes.

## Manual image prompts

If image generation cannot be called directly, generate the following files manually and place them in `paper/figures/`.

### `system_architecture.jpg`

Prompt:

```text
Create a clean academic figure for a machine-learning paper about an auditable hybrid controller for a land-wargame simulator. The figure should show a left-to-right architecture: simulator state and legal actions, feature normalization, LSTM intent predictor, GraphSAGE-style situation model, rule scaffold, objective action scoring, legal move and shoot ranking, selected action, and diagnostics. Use a restrained NeurIPS-style visual design with high contrast, thin lines, readable labels, and no decorative background. Emphasize that learned modules rank legal candidates rather than bypassing the simulator's valid-action contract. Use English labels only. Do not add extra claims, fake metrics, logos, watermarks, or unreadable text.
```

### `promotion_gate.jpg`

Prompt:

```text
Create a clean academic evaluation-flow figure for a land-wargame AI paper. Show a candidate agent compared against an incumbent on 100 ordered paired map items. The flow should branch into rating.json, promotion.json, exploitability.json, participants.json, map_list.json, and summary.md. Then show gates for confidence interval lower bound, complex-scenario performance, learned-path coverage, move and shoot coverage, exploiter cap, battle-phase evidence, selected-action evidence, and final accepted or rejected promotion verdict. Use a compact paper-figure style, colorblind-safe colors, crisp arrows, and readable English labels. Do not invent metric values unless they are explicitly supplied later.
```

### `ai_reviewer_loop.jpg`

Prompt:

```text
Create a diagram for an AI-assisted manuscript review loop. Inputs on the left: paper/main.tex, paper/source_evidence.md, paper/references.bib, program.md, reports, and latest league sidecars. Middle: scripts/run_ai_reviewer.py packages evidence for Draft Detective and runs deterministic claim auditing. Outputs on the right: review_summary.md, unsupported_claims.md, evidence_matrix.tsv, claims.jsonl, scores.json, draft_detective_input.md, and external_status.json. Then show a feedback arrow back to manuscript revision, citation verification, humanizer cleanup, and PDF rebuild. Style: clear academic workflow figure, white background, readable labels, no icons that imply unsupported automation.
```

### `evidence_map.jpg`

Prompt:

```text
Create an evidence-map figure for a research paper. Show manuscript claims grouped into method claims, evaluation claims, limitation claims, and related-work claims. Connect them to evidence sources: program.md, ai/policy.py, ai/models.py, latest league rating.json, promotion.json, exploitability.json, participants.json, map_list.json, summary.md, and paper/source_evidence.md. The visual message should be traceability: every claim must point to a source or be marked unverified. Use a compact grid or node-link layout suitable for a two-column ML paper. Use English labels only and avoid decorative or marketing style.
```

## Stopping conditions

The goal is complete when all of the following are true:

- `paper/main.tex` reflects the current verified Shenji evidence.
- `paper/source_evidence.md` maps major claims to sources.
- `paper/references.bib` contains all verified related papers used by the draft, and each entry has `project_note`.
- AI reviewer output has no unsupported claims, or every remaining item is listed with a reason and next action.
- **AI reviewer structured scores show overall >= 4.0/6.0 and no dimension below 3.0/5.0.**
- `humanizer` or a documented equivalent cleanup pass has been applied after the latest paper-generation pass.
- Figures either compile from existing LaTeX sources or generated `.jpg` files exist under `paper/figures/` and are referenced correctly.
- `paper/main.pdf` builds without fatal LaTeX errors.
- **Each revision round documents per-round scores and diff in CHANGELOG.md.**
- Git status is clean after a task commit, or the reason no commit was possible is documented.

## Hard blockers

Stop and document the blocker instead of inventing results if any of these occur:

- required league sidecars are missing;
- formal evidence contradicts a manuscript claim and cannot be reconciled without new experiments;
- citation metadata cannot be verified;
- image-generation tools are unavailable and manual prompts have already been written;
- LaTeX cannot build because a required local tool or package is missing;
- remote or local permissions prevent writing required outputs.

## Remote execution

Creating or revising the paper can usually run locally. Remote GPU execution is not required for manuscript editing.

If the goal changes small paper files that the remote worktree needs, sync only the changed small files to:

```text
/remote-home/lizilong/shenji_agent/paper/
```

Do not upload large checkpoints, datasets, or experiment outputs as part of this paper goal. If remote `reports/` is not writable, run AI reviewer with an override output directory under `.agent/`.
