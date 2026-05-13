
# V2 Top-Tier Evidence Upgrade Goal — Shenji_agent

## Objective

Upgrade Shenji_agent from the current Version 1 conservative evidence package into a credible top-tier submission candidate.

Target venues for quality calibration:

- NeurIPS / ICLR / ICML style AI-method review;
- CVPR style technical rigor if the task can be positioned as visual/spatial decision-making or embodied simulation;
- TPAMI-style journal rigor if the final contribution becomes a mature, deeply validated method.

This task must NOT simply polish the paper.

The task must fix the actual research blockers that prevent top-tier submission:

1. tactical contribution not proven;
2. route reliability not proven;
3. two-window hard-stop success not proven;
4. baseline/ablation evidence insufficient;
5. novelty currently too weak if framed as LSTM + GraphSAGE;
6. reproducibility still fragile because checkpoints/assets/validator expectations are inconsistent;
7. evaluation scope may be too narrow for top-tier claims.

SUCCESS means the project becomes a credible top-tier submission candidate based on executable evidence.

If that cannot be achieved in this repo, stop with SAFE_STOP and write exactly what remains missing.

---

## Version 1 baseline

Read first:

```text
paper/prewrite_package/PAPER_WRITING_READY.md
paper/prewrite_package/04_claim_to_evidence_matrix.md
paper/prewrite_package/05_results_for_tables.md
paper/prewrite_package/07_methods_package.md
paper/prewrite_package/08_novelty_and_positioning.md
paper/prewrite_package/09_reviewer_risk_register.md
paper/source_evidence.md
paper/CHANGELOG.md
paper/REVIEW_LOG.md
paper/ACCEPTANCE_CRITERIA.md
program.md
experiments/shenji_agent/
ai/
training/
scripts/
tests/
````

Current Version 1 final state was:

```text
PARTIAL_READY_WITH_BLOCKERS
```

Known forbidden V1 claims:

```text
no tactical contribution claim
no two-window success claim
no route reliability claim
no public SOTA claim
no top-tier-readiness claim
```

This V2 task may only lift these restrictions if new executable evidence supports doing so.

---

## Hard constraints

Do not:

* fabricate experiments;
* hand-edit formal sidecars;
* overwrite archived league artifacts;
* fake baselines;
* fake ablations;
* claim SOTA without protocol-matched evidence;
* claim tactical contribution unless diagnostics and ablations prove it;
* claim route reliability unless route diagnostics prove it;
* claim two-window hard-stop success unless formal executable evaluation proves it;
* claim top-tier readiness unless all required gates pass;
* delete assets destructively;
* use `git reset --hard`, `git clean -fdx`, `rsync --delete`, `robocopy /MIR`, or equivalent destructive sync operations.

Allowed:

* repair code;
* add tests;
* add validators;
* add training/evaluation scripts;
* run smoke and formal experiments;
* create new formal artifact directories;
* add baselines and ablations;
* improve method design;
* add statistical analysis;
* update documentation and evidence matrix;
* update the paper only after evidence changes.

---

## Core principle

Top-tier readiness cannot be achieved by wording.

Top-tier readiness requires:

```text
strong executable evidence
strong baselines
clear novelty
ablation support
statistical robustness
reproducibility
honest limitations
```

If the result is not strong enough, record that honestly.

---

# Phase 0 — Preserve Version 1

Before doing anything else:

```bash
git status --short
git log --oneline --decorate -n 20
git rev-parse HEAD
```

Ensure Version 1 is committed.

If no commit named `version 1 completed` exists, create one:

```bash
git add -A
git commit -m "version 1 completed"
```

If there is nothing to commit, create a tag instead:

```bash
git tag version-1-completed || true
```

Create backup branch:

```bash
git branch backup/before_v2_top_tier_upgrade_$(date +%Y%m%d_%H%M%S)
```

Create:

```text
reports/v2_top_tier_upgrade/version1_baseline.md
```

Record:

* branch;
* HEAD;
* commit/tag for Version 1;
* current blockers;
* current safe/forbidden claims.

---

# Phase 1 — Top-tier gap audit

Create:

```text
reports/v2_top_tier_upgrade/top_tier_gap_audit.md
reports/v2_top_tier_upgrade/top_tier_gap_audit.json
```

Audit the project against the following review dimensions:

## 1. Soundness

Check:

* evaluation protocol;
* promotion gate;
* sidecar validity;
* route reliability;
* tactical branch validity;
* leakage risks;
* deterministic seeds;
* failure cases;
* statistical confidence.

## 2. Novelty

Check whether the contribution is currently:

* algorithmic;
* systems/evaluation framework;
* diagnostic protocol;
* benchmark/protocol contribution;
* reproducibility/claim-boundary contribution.

Decide whether the project needs a new method contribution beyond LSTM/GraphSAGE.

## 3. Evidence

Check:

* training scripts;
* evaluation scripts;
* recovered assets;
* checkpoints;
* sidecars;
* baselines;
* ablations;
* multi-seed results;
* route/tactical diagnostics.

## 4. Significance

Check:

* task importance;
* SDK scope;
* generality;
* scenario/map diversity;
* reviewer interest;
* whether CVPR/NeurIPS/TPAMI fit is plausible.

## 5. Reproducibility

Check:

* commands;
* seeds;
* assets;
* configs;
* deterministic outputs;
* external dependencies;
* checkpoint provenance.

Output a table:

```text
Dimension | Current status | Required for top-tier | Gap | Planned fix | Can fix in this repo?
```

Do not proceed to major experiments until this audit is complete.

---

# Phase 2 — Reproducibility and asset contract repair

Create:

```text
reports/v2_top_tier_upgrade/reproducibility_contract.md
reports/v2_top_tier_upgrade/asset_contract.json
```

Fix all checkpoint naming and manifest inconsistencies.

Known issue:

```text
existing candidate checkpoint:
ai/weights/intention_lstm.pth
ai/weights/situation_gcn.pth

some validators may expect:
ai/weights/intent_lstm.pt
ai/weights/situation_gcn.pt
ai/weights/tactical_model.pt
```

Required actions:

1. inspect `ai/weights/model_manifest.json`;
2. inspect all loader code;
3. inspect all validators;
4. decide canonical checkpoint naming;
5. make loaders and validators manifest-driven;
6. avoid hard-coded stale checkpoint paths;
7. add tests.

Create tests for:

```text
candidate checkpoint manifest resolves correctly
candidate checkpoints exist
incumbent assets are either resolved or explicitly not required for candidate-only smoke
tactical checkpoint missing status is detected fail-closed
validators report naming mismatch instead of false missing assets
```

Run:

```bash
python -m unittest discover -v
python -m compileall paper/scripts scripts ai tests training
```

---

# Phase 3 — Tactical branch repair or removal

Create:

```text
reports/v2_top_tier_upgrade/tactical_branch_audit.md
reports/v2_top_tier_upgrade/tactical_branch_audit.json
```

Diagnose tactical branch end-to-end:

```text
checkpoint existence
manifest entry
model class
loader path
feature construction
tensor shapes
device placement
candidate eligibility
score aggregation
whether branch is reached
whether tactical score is overwritten
nonzero score count
effect on action choice
effect on win/loss metrics
```

Then choose one path:

## Path A — Tactical repair

Only if checkpoint or retraining path exists.

Required:

1. recover or train tactical checkpoint;
2. validate tactical score nonzero count;
3. run tactical-on vs tactical-off ablation;
4. measure action-level and outcome-level effect;
5. produce sidecars.

Minimum artifacts:

```text
reports/v2_top_tier_upgrade/tactical_diagnostics/
experiments/shenji_agent/v2_tactical_ablation/
```

Required metrics:

```text
tactical_loaded
tactical_nonzero_count
tactical_reached_count
tactical_changed_action_count
tactical_on_win_rate
tactical_off_win_rate
confidence interval or bootstrap interval
```

## Path B — Tactical removal / demotion

If tactical cannot be repaired, remove tactical from contribution claims entirely.

Allowed:

* keep code as inactive / future-work branch;
* update method to focus on intent/situation/fallback;
* update claim matrix.

Forbidden:

* implying tactical helps.

Top-tier impact:

If tactical remains unproven, the method must have novelty elsewhere. Record this.

---

# Phase 4 — Route planner / legal-move repair

Create:

```text
reports/v2_top_tier_upgrade/route_repair_audit.md
reports/v2_top_tier_upgrade/route_repair_audit.json
```

Known issue:

```text
empty routes: 15641 / 23580
empty route rate: 0.6633163698049194
main reason: no_valid_move_candidate
```

Diagnose:

```text
move candidate generation
legal action filtering
map/pathfinder constraints
Maneuver vs Fly route generation
fallback behavior
empty route legal no-op vs failure
distance/cost map lookup
unit type restrictions
terrain restrictions
SDK action schema mismatch
```

Repair target:

```text
empty_route_rate should be reduced substantially
```

Define measurable gates before repair:

Suggested minimum gates:

```text
empty_route_rate <= 0.20 for smoke
empty_route_rate <= 0.10 for formal evaluation
no_valid_move_candidate reduced by at least 70% relative to V1
no increase in illegal action rate
no degradation in win-rate beyond statistical noise
```

If these gates are unrealistic, write why and set evidence-based alternatives.

Required outputs:

```text
experiments/shenji_agent/v2_route_repair_smoke/
reports/v2_top_tier_upgrade/route_before_after.md
reports/v2_top_tier_upgrade/route_before_after.json
```

Must include before/after route metrics.

Do not claim route reliability if the gate fails.

---

# Phase 5 — Baselines and ablations

Create:

```text
reports/v2_top_tier_upgrade/baseline_ablation_plan.md
reports/v2_top_tier_upgrade/baseline_ablation_results.md
reports/v2_top_tier_upgrade/baseline_ablation_results.json
```

Required baselines, if feasible:

```text
rule incumbent
random legal action baseline
objective-only baseline
learned-only baseline
intent-only baseline
situation-only baseline
no-learned fallback baseline
no-route-repair baseline
route-repair baseline
tactical-off baseline
tactical-on baseline if tactical repaired
```

For each baseline:

```text
implementation path
command
output directory
number of games / scenarios / seeds
win rate
rating
promotion status if applicable
route empty-rate
tactical contribution metrics if applicable
confidence interval
failure modes
```

Minimum standard for top-tier candidate:

```text
at least 4 meaningful baselines/ablations
at least 3 random seeds for smoke robustness
clear improvement over rule incumbent or a clear diagnostic contribution
no evidence leakage
all outputs generated by executable scripts
```

If full evaluation is expensive, run smoke first, then formal subset.

Do not call smoke results formal results.

---

# Phase 6 — Multi-seed and statistical evaluation

Create:

```text
reports/v2_top_tier_upgrade/statistical_evaluation_plan.md
reports/v2_top_tier_upgrade/statistical_evaluation_results.md
reports/v2_top_tier_upgrade/statistical_evaluation_results.json
```

Required:

```text
multiple seeds
confidence intervals
bootstrap or binomial intervals where appropriate
effect sizes
scenario/map stratification if available
failure-rate analysis
sensitivity to route/tactical/fallback settings
```

Suggested minimum:

```text
3 seeds for smoke
5 seeds preferred for formal evidence
scenario/map stratified reporting
```

Promotion evidence requirement:

To lift the two-window prohibition, produce formal sidecars showing:

```text
two consecutive passing windows
no manual artifact editing
candidate/incumbent identities clear
promotion gate script executable
sidecar validator passes
```

If not achieved, keep the prohibition.

---

# Phase 7 — Novel method upgrade

If evidence audit shows novelty remains too weak, implement a real method-level contribution.

Possible contribution directions:

## Option 1 — Evidence-gated policy composition

A method that explicitly composes learned scores, objective scores, route feasibility, and fallback under a fail-closed gate.

Needs:

```text
formal scoring equation
configurable weights
route feasibility term
uncertainty/fallback gate
ablation support
```

## Option 2 — Route-aware learned action selection

A method that penalizes invalid or empty-route move candidates before action selection.

Needs:

```text
route feasibility features
candidate filtering
route-aware score calibration
before/after route evidence
```

## Option 3 — Diagnostic promotion protocol

A contribution focused on claim-bounded evaluation, sidecar validation, and SAFE_STOP discipline.

Needs:

```text
formal protocol definition
validators
case study
comparison with naive rule-score reviewer
evidence that protocol prevents overclaiming
```

## Option 4 — Counterfactual action attribution

A method to quantify whether learned components change selected actions and outcomes.

Needs:

```text
component-on/off replay
action-change rate
outcome-change analysis
confidence interval
```

Choose the strongest feasible option.

Create:

```text
reports/v2_top_tier_upgrade/method_upgrade_decision.md
```

Record:

```text
chosen contribution
why it is novel enough
what code changed
what evidence supports it
what ablations test it
remaining novelty risk
```

---

# Phase 8 — Formal validation suite

Create or update validators:

```text
scripts/validate_v2_top_tier_evidence.py
scripts/run_v2_baselines.py
scripts/run_v2_route_diagnostics.py
scripts/run_v2_tactical_diagnostics.py
scripts/run_v2_promotion_evaluation.py
```

Validators must check:

```text
required sidecars exist
candidate/incumbent identities are clear
no formal sidecar hand edits
checkpoint manifest resolves
SDK assets exist
route metrics are computed
tactical metrics are computed or explicitly absent
baseline artifacts exist
ablation artifacts exist
statistical summaries exist
two-window status not overstated
forbidden claims absent unless gates prove them
```

Add tests under:

```text
tests/test_v2_top_tier_*.py
```

Run:

```bash
python -m unittest discover -v
python -m compileall paper/scripts scripts ai tests training
```

---

# Phase 9 — Paper evidence upgrade package

Do not fully rewrite the paper yet unless evidence is ready.

Create:

```text
paper/v2_evidence_package/
```

Required files:

```text
paper/v2_evidence_package/V2_TOP_TIER_READINESS.md
paper/v2_evidence_package/claim_to_evidence_matrix_v2.md
paper/v2_evidence_package/results_tables_v2.csv
paper/v2_evidence_package/results_tables_v2.json
paper/v2_evidence_package/figure_plan_v2.md
paper/v2_evidence_package/method_delta_v2.md
paper/v2_evidence_package/limitation_delta_v2.md
paper/v2_evidence_package/reviewer_risk_v2.md
```

Update the claim matrix.

Claims can only be upgraded if supported by new evidence.

Explicitly classify:

```text
tactical contribution
route reliability
two-window success
baseline superiority
method novelty
reproducibility
public SOTA
top-tier readiness
```

---

# Phase 10 — Independent review simulation

Run a strict local review after evidence upgrade.

Create:

```text
reports/v2_top_tier_upgrade/strict_review_simulation.md
reports/v2_top_tier_upgrade/strict_review_simulation.json
```

Use dimensions:

```text
Soundness
Novelty
Significance
Evidence
Reproducibility
Clarity
```

Do not use this as proof of publishability.

Use it only to detect weaknesses.

Minimum target for top-tier candidate:

```text
Soundness >= 4.5 / 6
Novelty >= 4.0 / 6
Significance >= 4.0 / 6
Evidence >= 4.5 / 6
Reproducibility >= 4.5 / 6
Overall >= 4.5 / 6
```

If below target, either fix or SAFE_STOP.

---

# Phase 11 — Venue-fit decision

Create:

```text
reports/v2_top_tier_upgrade/venue_fit_decision.md
```

Classify venue target:

```text
NeurIPS/ICLR/ICML candidate
CVPR candidate
TPAMI candidate
Workshop/technical report only
Not publishable yet
```

Be honest.

Questions:

```text
Is the main contribution an AI method?
Is there enough novelty beyond engineering?
Is there enough empirical evidence?
Is the task relevant to the venue?
Is the dataset/protocol accessible enough?
Are baselines strong enough?
Are limitations acceptable?
```

If the best fit is workshop/technical report, say so.

---

# Phase 12 — Final readiness decision

Create:

```text
reports/v2_top_tier_upgrade/FINAL_DECISION.md
paper/v2_evidence_package/V2_TOP_TIER_READINESS.md
```

Use one final state:

## TOP_TIER_SUBMISSION_CANDIDATE

Only if all are true:

```text
tests pass
compileall passes
asset contract passes
checkpoint manifest passes
route gate passes or route limitation no longer central
tactical contribution either proven or removed from claims
at least 4 meaningful baselines/ablations exist
multi-seed evaluation exists
statistical confidence intervals exist
two-window success either proven or removed from claims
method novelty is clearly articulated and supported
claim matrix allows the main contribution
strict review simulation reaches target scores
venue-fit decision supports NeurIPS/CVPR/TPAMI-level submission candidate
no forbidden claims remain
```

## STRONG_TECHNICAL_REPORT_READY

Use if:

```text
implementation and evaluation are strong
but novelty/evidence is still below top-tier threshold
```

## PARTIAL_READY_WITH_BLOCKERS

Use if:

```text
some blockers are fixed
but major route/tactical/baseline/statistical issues remain
```

## NOT_READY_SAFE_STOP

Use if:

```text
core experiments cannot run
assets/checkpoints are missing
formal evaluation is blocked
or top-tier target is impossible in current repo
```

---

## SUCCESS criteria

SUCCESS only if final state is:

```text
TOP_TIER_SUBMISSION_CANDIDATE
```

and all required gates pass.

The task must commit:

```text
reports/v2_top_tier_upgrade/
paper/v2_evidence_package/
new or modified scripts/tests
updated source evidence / claim matrix
```

Suggested commit:

```bash
git add reports/v2_top_tier_upgrade paper/v2_evidence_package scripts tests ai training paper/source_evidence.md paper/CHANGELOG.md paper/REVIEW_LOG.md paper/ACCEPTANCE_CRITERIA.md 2>/dev/null || true
git status --short
git diff --cached --name-only | grep -E '\.(pt|pth|pickle|npz|zip|tar|tgz|whl)$' || true
git commit -m "upgrade evidence toward top-tier submission candidate"
```

Do not commit unexpected large binaries.

---

## SAFE_STOP criteria

SAFE_STOP if any of these remain unresolved after best effort:

```text
formal evaluation cannot run
tactical contribution cannot be proven or cleanly removed
route empty-rate remains too high for claims
baselines/ablations cannot be produced
multi-seed/statistical evidence cannot be produced
method novelty remains too weak
two-window claim remains unsupported and central to the story
venue-fit audit says top-tier target is unrealistic
```

In SAFE_STOP, create:

```text
reports/v2_top_tier_upgrade/SAFE_STOP.md
paper/v2_evidence_package/V2_TOP_TIER_READINESS.md
```

The SAFE_STOP must include:

```text
what was fixed
what remains blocked
which top-tier gates failed
whether the project should target workshop, technical report, or restart
exact next experiment needed
```

Commit the SAFE_STOP evidence.

---

## Important final instruction

Do not optimize for looking good.

Optimize for becoming genuinely defensible under a strict NeurIPS/CVPR/TPAMI-style review.

If the project cannot reach that level, the correct outcome is SAFE_STOP, not inflated SUCCESS.



