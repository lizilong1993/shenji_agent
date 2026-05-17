# V3B — Formal Tactical Outcome Ablation via SDK Replay

## Objective

Now that:

* tactical assets are restored and validated;
* tactical action-level contribution is supported;
* legacy intent checkpoint compatibility is restored;
* full-agent compatibility smoke passes;
* tactical-on vs tactical-off execution path is unblocked;

…the next milestone is to determine whether the tactical branch contributes measurable outcome-level improvements under formal SDK-backed replay.

This task must produce an honest, statistically grounded answer.

Allowed outcomes:

1. TACTICAL_OUTCOME_CONTRIBUTION_SUPPORTED
2. TACTICAL_OUTCOME_CONTRIBUTION_WEAK_OR_INCONCLUSIVE
3. TACTICAL_OUTCOME_CONTRIBUTION_NOT_SUPPORTED
4. V3B_SAFE_STOP

This task is NOT allowed to:

* claim public SOTA;
* claim top-tier readiness;
* claim tactical improves performance unless evidence supports it;
* fabricate statistical significance;
* hide failed seeds/scenarios;
* manually edit result sidecars.

---

# Scientific Goal

The central question:

> Does enabling the tactical branch improve outcome-level behavior under controlled replay conditions?

This must be evaluated through:

* paired tactical-on vs tactical-off replay;
* same seed;
* same scenario;
* same map;
* same starting conditions;
* statistically aggregated outcome metrics.

---

# Hard Constraints

## Forbidden operations

Do NOT:

* git reset --hard
* git clean -fdx
* rm -rf current project paths
* overwrite restored checkpoints
* rewrite historical evidence
* delete failed runs
* silently skip failed seeds

## Forbidden claims

Do NOT claim:

* route reliability solved;
* public SOTA;
* publication readiness;
* tactical causally improves win rate;
* strong contribution;
* generalization.

unless executable evidence supports it.

## Asset rules

Do not commit:

* *.pt
* *.pth
* *.npz
* *.pickle
* *.zip
* *.csv datasets
* SDK binaries
* replay dumps

Only commit:

* scripts;
* reports;
* tests;
* evidence summaries;
* small textual tables/json summaries.

---

# Output Directories

## Reports

reports/v3b_formal_tactical_ablation/

## Experiments

experiments/shenji_agent/v3b_formal_tactical_ablation/

## Paper evidence

paper/v3_evidence_package/

Required outputs:

* V3B_OUTCOME_ABLATION_STATUS.md
* V3B_RESULTS_SUMMARY.md
* v3b_results_tables.csv
* v3b_results_tables.json
* claim_to_evidence_matrix_v3.md

---

# Phase 0 — Start State

Create:

* reports/v3b_formal_tactical_ablation/00_start_state.md

Record:

* current branch;
* HEAD commit;
* tactical checkpoint availability;
* intent checkpoint compatibility state;
* SDK availability;
* replay environment status;
* current evaluation scripts;
* current route-empty metrics;
* current known blockers.

Create backup branch:

backup/before_v3b_formal_tactical_ablation_<timestamp>

If worktree is dirty beyond ignored assets:

SAFE_STOP.

---

# Phase 1 — Evaluation Protocol Definition

Create:

* reports/v3b_formal_tactical_ablation/01_protocol.md
* reports/v3b_formal_tactical_ablation/protocol.json

Define:

## Tactical modes

* tactical_mode="on"
* tactical_mode="off"

## Pairing protocol

Each replay pair MUST share:

* same seed;
* same scenario;
* same map;
* same initialization;
* same candidate/incumbent selection.

Only tactical mode changes.

## Minimum evaluation size

Target:

* > = 5 seeds;
* > = 3 scenarios/maps;
* > = 2 replay windows if feasible.

If SDK runtime cost is high:

minimum acceptable:

* > = 3 seeds;
* > = 2 scenarios.

Document exact limitations.

## Required metrics

Primary:

* win rate;
* score delta;
* rating delta if available.

Secondary:

* action divergence count;
* tactical invocation count;
* route empty-rate;
* move success rate;
* policy fallback rate;
* replay completion rate.

Reliability:

* crash count;
* replay abort rate;
* invalid action count.

---

# Phase 2 — Build Formal Replay Runner

Create/update:

* scripts/run_v3b_formal_tactical_ablation.py
* scripts/v3b_metrics.py
* scripts/v3b_pairing.py

Requirements:

The runner must:

* support tactical on/off;
* run paired evaluations;
* preserve identical seeds;
* save per-game metrics;
* fail closed;
* record exact replay failures;
* emit structured JSON;
* emit reproducible manifests.

Each replay must record:

* tactical_mode;
* seed;
* scenario;
* map;
* winner;
* score;
* route metrics;
* tactical activation stats;
* replay duration;
* errors.

Do not silently skip failed games.

---

# Phase 3 — Tactical Action Divergence Validation

Before outcome evaluation, validate:

> tactical on/off actually changes behavior under replay.

Create:

* reports/v3b_formal_tactical_ablation/03_action_divergence_validation.md
* action_divergence_validation.json

Run paired tactical on/off games.

Measure:

* action difference count;
* percentage of divergent turns;
* tactical branch activation frequency.

If tactical on/off produces effectively identical behavior:

mark:

TACTICAL_BRANCH_OUTCOME_TEST_NOT_MEANINGFUL

and explain why.

---

# Phase 4 — Smoke Evaluation

Run a minimal paired evaluation.

Create:

* reports/v3b_formal_tactical_ablation/04_smoke_eval.md
* smoke_eval_results.json

Requirements:

* at least one complete paired replay;
* tactical on/off both execute;
* metrics recorded;
* no silent replay failures.

Do NOT interpret smoke as proof.

---

# Phase 5 — Multi-seed Formal Evaluation

Run formal paired evaluation.

Create:

* reports/v3b_formal_tactical_ablation/05_multiseed_eval.md
* multiseed_eval_results.json
* bootstrap_statistics.json

Requirements:

## Statistical reporting

Compute:

* mean;
* std;
* bootstrap confidence intervals;
* paired deltas;
* effect size if possible.

At minimum:

* bootstrap CI for win-rate delta;
* bootstrap CI for score delta.

## Required transparency

Report:

* total games attempted;
* completed;
* failed;
* excluded;
* exact exclusion reasons.

No hidden filtering.

---

# Phase 6 — Route Reliability Audit

This phase is mandatory because previous route-empty rate was:

15641 / 23580 ≈ 0.6633

Create:

* reports/v3b_formal_tactical_ablation/06_route_reliability_audit.md
* route_reliability_audit.json

Measure:

* route-empty rate tactical on;
* route-empty rate tactical off;
* route failure reasons;
* fallback move frequency;
* learned move success rate.

Explicitly answer:

* does tactical reduce route failures?
* does tactical merely alter actions?
* are route failures still dominant?

Do not claim route reliability solved unless evidence strongly supports it.

---

# Phase 7 — Claim Boundary Decision

Create:

* reports/v3b_formal_tactical_ablation/07_claim_boundary.md
* FINAL_DECISION.md

Allowed conclusions:

## A. Supported

Use only if:

* paired evaluation completed;
* tactical on/off differs meaningfully;
* outcome metrics consistently favor tactical;
* confidence intervals are acceptable;
* failures are documented.

Allowed wording:

> Tactical-on replay demonstrated measurable outcome-level differences under paired SDK evaluation.

NOT:

> tactical solves gameplay.

## B. Weak/Inconclusive

Use if:

* signals exist;
* statistical confidence weak;
* replay count insufficient;
* high variance.

Allowed wording:

> Tactical outcome contribution remains inconclusive under current evaluation scale.

## C. Not supported

Use if:

* tactical does not improve metrics;
* behavior divergence exists but outcome does not;
* replay unstable.

Allowed wording:

> Tactical action-level influence is supported, but outcome-level contribution was not demonstrated.

---

# Phase 8 — Paper Evidence Update

Update:

* paper/v3_evidence_package/V3B_OUTCOME_ABLATION_STATUS.md
* paper/v3_evidence_package/V3_READINESS.md
* paper/v3_evidence_package/claim_to_evidence_matrix_v3.md
* paper/source_evidence.md

Add:

* exact evaluation protocol;
* exact replay counts;
* exact seeds;
* confidence intervals;
* failure accounting;
* claim boundaries.

The paper evidence package must clearly separate:

* supported claims;
* partially supported claims;
* unsupported claims.

---

# Phase 9 — Tests and Validation

Run:

python -m unittest discover -v
python -m pytest -q
python -m compileall paper/scripts scripts ai tests training

Create:

* reports/v3b_formal_tactical_ablation/09_validation_log.md

Record:

* commands;
* exit codes;
* failures;
* environment.

---

# Phase 10 — Commit

Before commit:

Remove pycache artifacts.

Do not stage:

* checkpoints;
* datasets;
* replay binaries;
* SDK assets.

Stage only:

* scripts;
* reports;
* tests;
* evidence summaries.

Commit message:

formalize tactical outcome ablation evaluation

---

# SUCCESS Criteria

SUCCESS only if:

* paired tactical on/off evaluation completed;
* outcome metrics collected;
* route audit completed;
* bootstrap/confidence reporting completed;
* claim boundaries documented honestly;
* tests pass;
* compileall passes;
* no large assets committed.

Allowed terminal states:

1. TACTICAL_OUTCOME_CONTRIBUTION_SUPPORTED
2. TACTICAL_OUTCOME_CONTRIBUTION_WEAK_OR_INCONCLUSIVE
3. TACTICAL_OUTCOME_CONTRIBUTION_NOT_SUPPORTED
4. V3B_SAFE_STOP

---

# Important Research Guidance

At this stage, scientific honesty matters more than optimistic metrics.

A negative or inconclusive V3B result is still scientifically valuable if:

* replay infrastructure is rigorous;
* tactical causality is isolated correctly;
* failure modes are characterized honestly;
* reproducibility is preserved.

The project is now transitioning from:

"Can the tactical branch load and affect actions?"

to:

"Does tactical influence materially improve outcome-level behavior under controlled replay evaluation?"

That distinction is critical for any future NeurIPS/CVPR/TPAMI-level positioning.

