
# V3H Baseline and Component-Ablation Expansion

## Objective

V3G completed full route coverage expansion.

Known V3G facts:
- Commit: `8ccd94e`
- 42/43 archived league scenarios evaluated with authenticated SDK replay
- Coverage: 97.7% of league scenarios
- All 11 maps covered
- Total move attempts: 5,425
- Scenario `1910631192` contradiction resolved
  - current authenticated replay: 0% empty
  - archived league 99% empty was caused by stale `.engine_config` MAC/auth artifact
- SDK diagnostic revised:
  - `no_valid_move_candidate` is a misleading label
  - route planner is called before the label fires
  - true failure class: A* cannot find any path to any city
- Route claim gate upgraded:
  - from `ROUTE_NO_DEGRADATION_OBSERVED_ONLY`
  - to `ROUTE_RELIABILITY_PARTIALLY_SUPPORTED_SCENARIO_DEPENDENT`
- Route evidence remains bounded:
  - 39/42 scenarios have 0% empty
  - 2 scenarios on map 94 have approximately 99% empty
  - 1 scenario on map 212 crashed
  - 1 scenario on map 53 was excluded due to timeout

V3H must expand baseline and component-ablation evidence so the project can decide whether it is ready for a conservative SDK evaluation manuscript.

This is an evaluation and evidence-expansion round.

Do not train.
Do not modify checkpoints.
Do not hand-edit archived sidecars.
Do not claim public SOTA.
Do not claim top-tier readiness.
Do not claim two-window promotion success unless formal sidecars prove it.
Do not claim general route reliability; preserve V3G’s scenario-dependent route limitation.
Do not strengthen tactical outcome claims; preserve V3D’s weak/inconclusive result.

## Parallel subagent plan

Use parallel subagents where available.

Recommended subagents:

1. `baseline_inventory_subagent`
   - Inventory all existing baseline, ablation, promotion, rating, and league evidence.
   - Identify which baselines are executable under current authenticated SDK.

2. `protocol_design_subagent`
   - Define a bounded paired evaluation protocol.
   - Select scenario subsets based on V3G route evidence:
     - route-safe scenarios;
     - map 94 route-failure scenarios;
     - crash/timeout exclusions with explicit justification.

3. `evaluator_implementation_subagent`
   - Reuse or extend existing V3B/V3D/V3G runners.
   - Add baseline/component-ablation modes without changing model weights.

4. `experiment_execution_subagent`
   - Run bounded authenticated SDK evaluations.
   - Record per-game, per-scenario, per-map, and per-component metrics.

5. `statistics_subagent`
   - Compute paired deltas, confidence intervals, effect sizes, win/loss/draw rates if available, and failure-mode summaries.

6. `evidence_claim_subagent`
   - Update `paper/v3_evidence_package/` and `paper/source_evidence.md` conservatively.
   - Enforce claim boundaries from V3D and V3G.

7. `validation_subagent`
   - Add tests.
   - Run unittest and compileall.
   - Check no sensitive SDK auth/license files or large checkpoints are staged.

Coordinator must merge all findings into final V3H reports and final decision.

## Source of truth

Read first:

- `reports/v3g_full_route_coverage/FINAL_DECISION.md`
- `reports/v3g_full_route_coverage/08_route_claim_gate.md`
- `reports/v3g_full_route_coverage/06_full_coverage_evaluation.md`
- `reports/v3g_full_route_coverage/07_archived_vs_current_comparison.md`
- `reports/v3g_full_route_coverage/02_scenario_1910631192_audit.md`
- `reports/v3g_full_route_coverage/manuscript_readiness_update.json`
- `reports/v3f_route_reliability/FINAL_DECISION.md`
- `reports/v3e_claim_freeze/03_baseline_ablation_audit.md`
- `reports/v3e_claim_freeze/baseline_ablation_audit.json`
- `reports/v3d_authenticated_tactical_ablation/statistics.json`
- `reports/v3d_authenticated_tactical_ablation/claim_gate.json`
- `paper/v3_evidence_package/V3_READINESS.md`
- `paper/v3_evidence_package/claim_to_evidence_matrix_v3.md`
- `paper/v3_evidence_package/results_tables_v3.csv`
- `paper/v3_evidence_package/results_tables_v3.json`
- `paper/source_evidence.md`
- `scripts/run_v3b_formal_tactical_ablation.py`
- `scripts/v3b_pairing.py`
- `scripts/v3b_metrics.py`
- `scripts/run_v3g_full_route_coverage_evaluation.py`
- `ai/policy.py`
- `ai/agent.py`
- `ai/models.py`
- `tests/`
- archived sidecars under `experiments/shenji_agent/`

## Hard constraints

Do not:
- train;
- alter `.pth`, `.pt`, checkpoint, or weight files;
- modify SDK internals;
- edit archived league sidecars by hand;
- delete unfavorable evidence;
- claim tactical outcome contribution;
- claim route reliability is generally solved;
- claim two-window promotion success without sidecar proof;
- claim public SOTA;
- claim top-tier readiness;
- run destructive git commands;
- commit SDK auth files, license files, secrets, tokens, or large binaries.

Allowed:
- run authenticated SDK evaluation;
- add bounded baseline/component-ablation runner scripts;
- add diagnostic wrappers;
- compare executable baselines;
- update tests;
- update reports;
- update V3 evidence package conservatively.

## Phase 0 — Start state and safety

Run:

```bash
pwd
git status --short
git log --oneline --decorate -n 50
git branch --show-current
````

Create:

```text
reports/v3h_baseline_expansion/00_start_state.md
```

Record:

* current branch;
* current HEAD;
* V3G commit `8ccd94e`;
* worktree status;
* V3G route claim gate;
* current tactical claim status;
* current baseline coverage status from V3E;
* whether V3G reports exist;
* whether existing V3B/V3D/V3G runners exist.

Create backup branch:

```bash
git branch backup/before_v3h_baseline_expansion_$(date +%Y%m%d_%H%M%S)
```

If there are unrelated worktree edits, document them and avoid overwriting them.

## Phase 1 — Baseline and ablation evidence inventory

Create:

```text
reports/v3h_baseline_expansion/01_baseline_inventory.md
reports/v3h_baseline_expansion/baseline_inventory.json
```

Search:

```bash
grep -RInE 'baseline|ablation|incumbent|candidate|rule|random|objective|intent|situation|tactical|route|promotion|rating|win_rate|score|league|participant' \
  reports experiments paper scripts ai training tests 2>/dev/null \
  > reports/v3h_baseline_expansion/baseline_inventory_grep.txt || true
```

Inventory all available baselines and component variants:

Required categories:

* learned candidate;
* rule incumbent;
* current incumbent policy;
* random legal baseline, if executable;
* objective-only baseline, if executable;
* rule-only baseline;
* learned without tactical;
* tactical on/off, from V3D;
* intent-disabled variant, if executable;
* situation-disabled variant, if executable;
* route-aware / route-disabled or route-fallback variants, if executable;
* no-learned fallback;
* archived league participant variants;
* promotion window evidence.

For each item record:

* name;
* implementation path;
* checkpoint dependency;
* executable now?
* requires training?
* requires model modification?
* authenticated SDK compatible?
* scenario coverage possible?
* metrics available?
* limitations;
* whether it can support a manuscript claim.

Classify each as:

```text
EXECUTABLE_FORMAL
EXECUTABLE_SMOKE_ONLY
EXISTS_BUT_NOT_CURRENTLY_EXECUTABLE
ARCHIVED_ONLY
REQUIRES_TRAINING
UNSAFE_OR_OUT_OF_SCOPE
MISSING
```

## Phase 2 — Evaluation protocol design

Create:

```text
reports/v3h_baseline_expansion/02_protocol.md
reports/v3h_baseline_expansion/protocol.json
```

Design a bounded authenticated paired protocol.

Scenario policy:

* Primary set: route-safe scenarios from V3G, preferably those with 0% empty.
* Stress set: include the two map 94 high-empty scenarios separately as limitation/stress cases.
* Exclusions:

  * map 212 crash scenario must be documented if excluded;
  * map 53 timeout scenario must be documented if excluded.

Minimum recommended protocol:

* 15 paired comparisons if runtime is limited;
* better: 30 paired comparisons;
* include at least 5 seeds if feasible;
* include route-safe and route-stress subsets separately.

Required protocol fields:

* scenario IDs;
* map IDs;
* seeds;
* pairing design;
* baseline list;
* component ablation list;
* outcome metrics;
* route metrics;
* illegal action metrics;
* candidate generation metrics;
* tactical invocation metrics;
* confidence interval method;
* stopping rules;
* claim gates.

Do not use map 94 failures to claim general route failure unless clearly labeled as stress cases.
Do not exclude map 94 silently.

## Phase 3 — Implement or extend baseline evaluator

Create or update:

```text
scripts/run_v3h_baseline_component_ablation.py
```

It may reuse:

* `scripts/run_v3b_formal_tactical_ablation.py`
* `scripts/v3b_pairing.py`
* `scripts/v3b_metrics.py`
* `scripts/run_v3g_full_route_coverage_evaluation.py`

Required CLI examples:

```bash
python scripts/run_v3h_baseline_component_ablation.py \
  --scenario-set route_safe \
  --seeds 0 1 2 3 4 \
  --pairs 3 \
  --baselines learned rule_incumbent objective_only random_legal \
  --out experiments/shenji_agent/v3h_baseline_expansion_route_safe
```

Stress mode:

```bash
python scripts/run_v3h_baseline_component_ablation.py \
  --scenario-set route_stress_map94 \
  --seeds 0 1 2 \
  --baselines learned rule_incumbent \
  --out experiments/shenji_agent/v3h_baseline_expansion_route_stress
```

Component mode:

```bash
python scripts/run_v3h_baseline_component_ablation.py \
  --scenario-set route_safe \
  --seeds 0 1 2 \
  --components full no_tactical no_intent no_situation rule_fallback \
  --out experiments/shenji_agent/v3h_component_ablation
```

Adapt CLI to existing project conventions if needed.

Evaluator must record:

* seed;
* scenario ID;
* map ID;
* policy variant;
* baseline/component mode;
* authenticated;
* deployment entered;
* battle entered;
* game completed;
* turns;
* score / objective metric;
* win/loss/draw if available;
* illegal action count;
* route empty count;
* route empty rate;
* raw SDK candidate empty count;
* route planner invocation count;
* fallback count;
* tactical invocation count;
* intent usage status;
* situation usage status;
* runtime errors.

Fail closed if:

* SDK auth fails;
* battle phase is not entered;
* metric denominator is missing;
* scenario/map cannot be loaded;
* selected baseline is not executable.

## Phase 4 — Preflight smoke

Run a minimal smoke:

```bash
python scripts/run_v3h_baseline_component_ablation.py \
  --scenario-set route_safe \
  --seeds 0 \
  --pairs 1 \
  --baselines learned rule_incumbent \
  --out experiments/shenji_agent/v3h_smoke
```

Create:

```text
reports/v3h_baseline_expansion/04_smoke.md
reports/v3h_baseline_expansion/smoke.json
```

Required checks:

* authenticated SDK replay works;
* battle phase is entered;
* learned variant runs;
* rule incumbent runs;
* paired metrics are emitted;
* route diagnostics are emitted;
* no sensitive files are produced.

If smoke fails, stop with:

```text
SAFE_STOP_BASELINE_EVALUATION_BLOCKED
```

## Phase 5 — Route-safe baseline evaluation

Run bounded route-safe evaluation.

Preferred command:

```bash
python scripts/run_v3h_baseline_component_ablation.py \
  --scenario-set route_safe \
  --seeds 0 1 2 3 4 \
  --pairs 6 \
  --baselines learned rule_incumbent objective_only random_legal \
  --out experiments/shenji_agent/v3h_baseline_expansion_route_safe
```

If some baselines are not executable, run executable ones and document missing variants.

Create:

```text
reports/v3h_baseline_expansion/05_route_safe_baseline_eval.md
reports/v3h_baseline_expansion/route_safe_baseline_eval.json
```

Required aggregate metrics:

* number of games;
* number of paired comparisons;
* scenario count;
* map count;
* completion rate;
* mean score by variant;
* mean paired score delta learned minus baseline;
* median paired delta;
* bootstrap confidence interval;
* Cohen’s d or equivalent effect size;
* win/loss/draw if available;
* illegal action rate;
* route empty rate;
* candidate-empty rate;
* runtime failure rate.

Claim rule:

* If confidence interval crosses zero, mark as weak/inconclusive.
* Do not claim broad superiority unless effect is stable across scenarios and baselines.

## Phase 6 — Component ablation evaluation

Run component ablations that are executable without training.

Preferred command:

```bash
python scripts/run_v3h_baseline_component_ablation.py \
  --scenario-set route_safe \
  --seeds 0 1 2 3 4 \
  --pairs 3 \
  --components full no_tactical no_intent no_situation rule_fallback \
  --out experiments/shenji_agent/v3h_component_ablation
```

If `no_intent` or `no_situation` cannot be implemented safely without changing checkpoints or core model semantics, mark as not executable and do not fake the ablation.

Create:

```text
reports/v3h_baseline_expansion/06_component_ablation.md
reports/v3h_baseline_expansion/component_ablation.json
```

For each component variant record:

* executable?
* implementation method;
* games;
* score / objective;
* delta vs full;
* route metrics;
* illegal actions;
* fallback rate;
* tactical invocation if applicable;
* limitation.

Required tactical handling:

* Preserve V3D:

  * tactical is diagnostic/action-level supported;
  * tactical outcome contribution remains weak/inconclusive.
* Do not rerun tactical to overturn V3D unless protocol is explicitly larger and paired.

## Phase 7 — Route-stress baseline check

Run a separate stress check on map 94 scenarios.

Preferred command:

```bash
python scripts/run_v3h_baseline_component_ablation.py \
  --scenario-set route_stress_map94 \
  --seeds 0 1 2 \
  --baselines learned rule_incumbent \
  --out experiments/shenji_agent/v3h_route_stress_map94
```

Create:

```text
reports/v3h_baseline_expansion/07_route_stress_map94.md
reports/v3h_baseline_expansion/route_stress_map94.json
```

Purpose:

* Do not use this as the main baseline result.
* Use it to document whether map 94 route failure affects learned and rule policies similarly.
* Record whether failures are policy-specific or SDK/map-specific.

Required classification:

```text
MAP94_FAILURE_POLICY_SPECIFIC
MAP94_FAILURE_SDK_OR_MAP_SPECIFIC
MAP94_FAILURE_MIXED
MAP94_STRESS_EVALUATION_BLOCKED
```

## Phase 8 — Statistical summary and claim gates

Create:

```text
reports/v3h_baseline_expansion/08_statistics.md
reports/v3h_baseline_expansion/statistics.json
reports/v3h_baseline_expansion/claim_gates.json
```

Required claim gates:

### Baseline status

Use exactly one:

```text
BASELINE_EVIDENCE_SUPPORTS_BOUNDED_IMPROVEMENT
BASELINE_EVIDENCE_WEAK_OR_INCONCLUSIVE
BASELINE_EVIDENCE_NOT_SUPPORTED
BASELINE_EVALUATION_BLOCKED
```

### Component ablation status

Use exactly one:

```text
COMPONENT_ABLATION_SUPPORTS_SELECTED_COMPONENTS
COMPONENT_ABLATION_WEAK_OR_INCONCLUSIVE
COMPONENT_ABLATION_NOT_EXECUTABLE
COMPONENT_ABLATION_BLOCKED
```

### Manuscript evidence status

Use exactly one:

```text
EVIDENCE_READY_FOR_CONSERVATIVE_SDK_EVALUATION_MANUSCRIPT
EVIDENCE_READY_FOR_TECHNICAL_REPORT_ONLY
EVIDENCE_NEEDS_MORE_BASELINE_COVERAGE
EVIDENCE_NEEDS_SDK_ROUTE_STRESS_INVESTIGATION
EVIDENCE_NOT_READY
```

Allowed wording if baseline evidence is positive but bounded:

```text
Under authenticated SDK replay on route-bounded scenarios, the learned policy shows bounded improvement over executable baselines in the tested protocol.
```

Allowed wording if weak:

```text
Authenticated baseline evaluation runs successfully, but learned-policy improvement over executable baselines remains weak or inconclusive under the tested protocol.
```

Forbidden unless proven:

* learned policy broadly outperforms strong baselines;
* public SOTA;
* robust generalization across all maps;
* route reliability solved;
* tactical improves outcome performance;
* two-window promotion success.

## Phase 9 — Update V3 evidence package

Update conservatively:

```text
paper/v3_evidence_package/V3_READINESS.md
paper/v3_evidence_package/claim_to_evidence_matrix_v3.md
paper/v3_evidence_package/results_tables_v3.csv
paper/v3_evidence_package/results_tables_v3.json
paper/source_evidence.md
```

Required updates:

* V3H baseline evaluation protocol;
* executable baseline list;
* missing/non-executable baselines;
* baseline result claim gate;
* component ablation claim gate;
* route-safe vs route-stress distinction;
* map 94 limitation;
* preservation of V3D tactical limitation;
* preservation of V3G route scenario-dependent limitation;
* manuscript readiness implication.

Do not strengthen unsupported claims.

## Phase 10 — Tests

Add or update:

```text
tests/test_v3h_baseline_component_ablation.py
```

Tests should cover:

* baseline evaluator emits required fields;
* learned and rule incumbent variants are distinguishable;
* missing baselines are marked non-executable rather than silently skipped;
* component ablation cannot claim support if not executed;
* tactical outcome status remains weak/inconclusive unless new claim gate explicitly supports otherwise;
* route claim remains scenario-dependent after V3G;
* map 94 stress results are not merged into route-safe aggregate without labeling;
* public SOTA and two-window promotion claims remain forbidden unless supported.

Run:

```bash
python -m unittest discover -v
python -m compileall paper/scripts scripts ai tests training
```

If pytest is available and useful:

```bash
python -m pytest tests/test_v3h_baseline_component_ablation.py -q 2>/dev/null || true
```

Create:

```text
reports/v3h_baseline_expansion/10_validation_log.md
```

Record commands, exit codes, output summary, warnings, and failures.

## Phase 11 — Manuscript readiness decision

Create:

```text
reports/v3h_baseline_expansion/11_manuscript_readiness_decision.md
reports/v3h_baseline_expansion/manuscript_readiness_decision.json
```

Use exactly one:

```text
READY_FOR_CONSERVATIVE_SDK_EVALUATION_MANUSCRIPT
```

Use only if:

* route is bounded as scenario-dependent;
* baseline evidence supports bounded improvement or at least a clear conservative contribution;
* component evidence is adequate or limitations are explicit;
* unsupported claims are removed;
* promotion/two-window/SOTA/top-tier claims remain forbidden.

```text
READY_FOR_CONSERVATIVE_TECHNICAL_REPORT_ONLY
```

Use if:

* evaluations run;
* claims are honest;
* but baseline/component evidence is not strong enough for paper-level contribution.

```text
NEEDS_MORE_BASELINE_EXPANSION
```

Use if:

* baseline evidence is still too narrow or weak.

```text
NEEDS_COMPONENT_ABLATION_DESIGN
```

Use if:

* component variants cannot be evaluated safely and this blocks the manuscript.

```text
NEEDS_ROUTE_STRESS_INVESTIGATION
```

Use if:

* map 94 or route-stress failures remain central enough to block manuscript.

```text
NOT_READY_FOR_MANUSCRIPT
```

Use if:

* evidence is internally inconsistent or main claims remain unsupported.

## Phase 12 — Final decision

Create:

```text
reports/v3h_baseline_expansion/FINAL_DECISION.md
reports/v3h_baseline_expansion/final_decision.json
```

Use exactly one final state:

```text
BASELINE_EXPANSION_SUPPORTS_CONSERVATIVE_MANUSCRIPT
BASELINE_EXPANSION_SUPPORTS_TECHNICAL_REPORT_ONLY
BASELINE_EXPANSION_WEAK_NEEDS_MORE_EVALUATION
BASELINE_EXPANSION_BLOCKED_BY_EXECUTABILITY
BASELINE_EXPANSION_BLOCKED_BY_ROUTE_STRESS
SAFE_STOP_BASELINE_EVALUATION_BLOCKED
```

Final decision must recommend the next goal:

* `V3I conservative manuscript rewrite`
* `V3I additional baseline expansion`
* `V3I component ablation redesign`
* `V3I route stress investigation`
* `V3I promotion evidence rebuild`
* or `SAFE_STOP evidence reconciliation`

## Phase 13 — Commit

Before staging:

```bash
git status --short
git diff --name-only
git diff --cached --name-only | grep -Ei 'engine_config|license|auth|key|secret|token|\.pt$|\.pth$|\.zip$|\.tar$|\.tgz$|\.whl$' || true
```

Stage only reports, scripts, tests, evidence text, experiment outputs, and goal:

```bash
git add reports/v3h_baseline_expansion
git add experiments/shenji_agent/v3h_smoke 2>/dev/null || true
git add experiments/shenji_agent/v3h_baseline_expansion_route_safe 2>/dev/null || true
git add experiments/shenji_agent/v3h_component_ablation 2>/dev/null || true
git add experiments/shenji_agent/v3h_route_stress_map94 2>/dev/null || true
git add scripts/run_v3h_baseline_component_ablation.py 2>/dev/null || true
git add tests/test_v3h_baseline_component_ablation.py 2>/dev/null || true
git add paper/v3_evidence_package paper/source_evidence.md 2>/dev/null || true
git add goals/codex_goal_v3h_baseline_component_ablation_expansion.md 2>/dev/null || true
```

Unstage sensitive or large files:

```bash
git restore --staged '.engine_config' 2>/dev/null || true
git restore --staged '*license*' 2>/dev/null || true
git restore --staged '*secret*' 2>/dev/null || true
git restore --staged '*token*' 2>/dev/null || true
git restore --staged '*.pt' '*.pth' '*.zip' '*.tar' '*.tgz' '*.whl' 2>/dev/null || true
```

Commit:

```bash
git commit -m "expand baseline and component ablation evidence"
```

## SUCCESS criteria

SUCCESS if:

* baseline inventory is complete;
* executable baselines are identified;
* non-executable baselines are explicitly marked;
* bounded authenticated baseline evaluation runs;
* component ablations are run or correctly marked non-executable;
* route-safe and route-stress results are separated;
* statistics and confidence intervals are computed where feasible;
* claim gates are written;
* V3 evidence package is updated conservatively;
* tests and compileall pass;
* no tactical, route, promotion, SOTA, or top-tier claim is overstated;
* no sensitive SDK auth/license files or large checkpoints are committed.

## SAFE_STOP criteria

SAFE_STOP if:

* SDK authentication regresses;
* baseline evaluator cannot enter battle phase;
* learned/rule policies cannot be executed under the same protocol;
* outcome metrics cannot be produced;
* route-safe scenario set cannot be loaded;
* component variants require training or checkpoint modification;
* results are internally inconsistent;
* validation fails in a way that invalidates the evaluation;
* sensitive SDK auth files would need to be committed.

If SAFE_STOP, create:

```text
reports/v3h_baseline_expansion/SAFE_STOP.md
```

It must state:

* what was attempted;
* exact blocker;
* whether V3G remains valid;
* current baseline claim boundary;
* which manuscript claims remain forbidden;
* human action needed next.
入 V3I manuscript rewrite。

