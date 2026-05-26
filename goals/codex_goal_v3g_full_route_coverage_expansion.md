
# V3G Full Route Coverage Expansion

## Objective

V3F completed route reliability diagnosis.

Known V3F facts:
- Final route claim gate: `ROUTE_NO_DEGRADATION_OBSERVED_ONLY`
- V3D route evidence: 590 genuine attempts, 0 route-empty events, 3 scenarios
- Archived league evidence: 23,580 attempts, 15,641 empty routes, 66.3% empty rate, approximately 100 scenarios
- Root cause class: SDK candidate generation constraints, especially `no_valid_move_candidate`
- Route planner is not the primary bug: `Map.gen_move_route` / A* works when called
- Instrumentation is correct: same diagnostic logic applies across current evaluations and archived league analysis
- Move-type empty rates were scenario-dependent:
  - Fly: 82.4%
  - Maneuver: 75.2%
  - Walk: 18.1%
- No code repair was applied in V3F because no route planner bug was found
- Critical unresolved contradiction:
  - Scenario `1910631192` shows 0% empty in V3D
  - The same scenario shows 99% empty in archived league evidence

V3G must expand authenticated route coverage across the full archived league scenario set, resolve the `1910631192` contradiction, and determine whether the archived high empty-route behavior still reproduces under the current authenticated SDK environment.

Do not train.
Do not change tactical claims.
Do not edit archived league sidecars by hand.
Do not modify model checkpoints.
Do not claim general route reliability unless authenticated full-coverage evaluation supports it with denominator evidence.
Do not claim tactical outcome contribution, two-window promotion success, public SOTA, or top-tier readiness.

## Parallel subagent plan

Use parallel subagents where available.

Recommended subagents:

1. `scenario_inventory_subagent`
   - Inventory archived league scenario IDs, map IDs, sidecars, seeds, move types, and route failure rates.
   - Identify all occurrences of scenario `1910631192`.

2. `sdk_candidate_diagnostic_subagent`
   - Inspect where SDK candidate lists are generated or consumed.
   - Distinguish SDK-empty candidate lists from policy filtering, fallback behavior, and route planner empty paths.

3. `route_evaluator_subagent`
   - Extend or run `scripts/run_v3f_route_reliability_evaluation.py` across the full league scenario set.
   - Ensure authenticated replay, battle-phase entry, and denominator logging.

4. `statistics_subagent`
   - Compute per-scenario, per-map, per-move-type, and aggregate route metrics.
   - Compare V3F/V3D/current V3G metrics against archived league evidence.

5. `evidence_and_claim_subagent`
   - Update V3 evidence package conservatively.
   - Ensure route claims do not exceed executable evidence.

6. `validation_subagent`
   - Run tests, compile checks, and claim-boundary consistency tests.
   - Ensure no sensitive SDK auth/license files or large checkpoints are staged.

Coordinator must merge subagent findings into final V3G reports and final decision.

## Source of truth

Read first:

- `reports/v3f_route_reliability/FINAL_DECISION.md`
- `reports/v3f_route_reliability/08_route_claim_gate.md`
- `reports/v3f_route_reliability/02_route_discrepancy_analysis.md`
- `reports/v3f_route_reliability/07_final_route_evaluation.md`
- `reports/v3f_route_reliability/route_discrepancy_analysis.json`
- `reports/v3f_route_reliability/final_route_evaluation.json`
- `scripts/run_v3f_route_reliability_evaluation.py`
- `tests/test_v3f_route_reliability.py`
- `reports/v3e_claim_freeze/02_route_reliability_audit.md`
- `reports/v3e_claim_freeze/route_reliability_audit.json`
- `reports/v3d_authenticated_tactical_ablation/multiseed_ablation.json`
- archived league sidecars under `experiments/shenji_agent/`
- `paper/v3_evidence_package/V3_READINESS.md`
- `paper/v3_evidence_package/claim_to_evidence_matrix_v3.md`
- `paper/source_evidence.md`
- `ai/policy.py`
- `ai/agent.py`
- route/map code under `training/`, `scripts/`, and `tests/`

## Hard constraints

Do not:
- train models;
- change model weights;
- hand-edit archived formal sidecars;
- rewrite historical route evidence;
- remove unfavorable route evidence;
- claim route reliability is solved without fresh authenticated full-coverage denominator evidence;
- claim tactical improves outcome performance;
- claim two-window promotion success;
- claim public SOTA;
- claim top-tier readiness;
- run destructive git commands;
- commit SDK auth files, license files, secrets, tokens, or large binaries.

Allowed:
- inspect archived sidecars;
- extract scenario and map IDs;
- run bounded authenticated evaluation;
- extend route evaluator instrumentation;
- add diagnostic scripts;
- add tests;
- update reports and evidence package conservatively;
- classify route evidence with explicit limitations.

## Phase 0 — Start state and safety

Run:

```bash
pwd
git status --short
git log --oneline --decorate -n 40
git branch --show-current
````

Create:

```text
reports/v3g_full_route_coverage/00_start_state.md
```

Record:

* current branch;
* current HEAD;
* V3F commit, using current HEAD if no explicit commit hash is available;
* worktree status;
* V3F final decision;
* V3F route claim gate;
* known `1910631192` contradiction;
* SDK auth status if available;
* whether V3F reports exist;
* whether archived league sidecars exist.

Create backup branch:

```bash
git branch backup/before_v3g_full_route_coverage_$(date +%Y%m%d_%H%M%S)
```

If the worktree contains unrelated edits, document them and avoid overwriting them.

## Phase 1 — Full archived league scenario inventory

Create:

```text
reports/v3g_full_route_coverage/01_league_scenario_inventory.md
reports/v3g_full_route_coverage/league_scenario_inventory.json
```

Inventory archived league evidence from `experiments/shenji_agent/`.

Run:

```bash
find experiments/shenji_agent -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name 'summary.md' -o -name 'league_state.json' -o -name 'rating.json' -o -name 'promotion.json' -o -name 'participants.json' \) \
  | sort > reports/v3g_full_route_coverage/league_sidecar_paths.txt

grep -RInE '1910631192|route_empty|empty_route|no_valid_move_candidate|move_attempt|scenario|map_id|Fly|Maneuver|Walk' \
  experiments reports scripts tests ai training 2>/dev/null \
  > reports/v3g_full_route_coverage/route_full_coverage_grep.txt || true
```

For each recoverable archived league scenario, record:

* scenario ID;
* map ID if available;
* seed if available;
* sidecar path;
* game or evaluation protocol;
* move attempts;
* route-eligible attempts;
* empty-route count;
* empty-route rate;
* `no_valid_move_candidate` count;
* move type distribution: Fly / Maneuver / Walk / others;
* whether battle phase was entered;
* whether game completed;
* whether the scenario appears in V3D or V3F;
* whether scenario `1910631192` appears;
* limitations.

Required output:

* unique scenario count;
* unique map count;
* number of scenarios with route denominator evidence;
* top 20 worst scenarios by empty-route rate;
* top 20 highest-volume scenarios by move attempts;
* all rows involving scenario `1910631192`.

## Phase 2 — Scenario 1910631192 focused contradiction audit

Create:

```text
reports/v3g_full_route_coverage/02_scenario_1910631192_audit.md
reports/v3g_full_route_coverage/scenario_1910631192_audit.json
```

This phase must directly answer why the same scenario produced:

```text
V3D: 0% empty
archived league: 99% empty
```

Audit:

* exact scenario file path;
* map path;
* map checksum if feasible;
* SDK scenario metadata;
* V3D run configuration;
* archived league run configuration;
* seed differences;
* side assignment differences;
* deployment state differences;
* policy mode differences;
* candidate/incumbent differences;
* tactical on/off differences;
* move type differences;
* candidate list size distribution;
* whether SDK candidate generation is called with the same state fields;
* whether the archived league result used old SDK/map data;
* whether current SDK/map data differs from archived league assumptions.

If possible, run a focused authenticated replay on scenario `1910631192`:

```bash
python scripts/run_v3f_route_reliability_evaluation.py \
  --seeds 0 1 2 3 4 \
  --scenario-ids 1910631192 \
  --out experiments/shenji_agent/v3g_scenario_1910631192_focus
```

If CLI differs, adapt the existing evaluator or add equivalent support.

Required output:

* current authenticated empty-route rate for scenario `1910631192`;
* current move attempts;
* current `no_valid_move_candidate` count;
* current candidate list size distribution;
* whether the 99% archived behavior is reproduced;
* exact reason if not reproduced.

Classify the contradiction as exactly one:

```text
SCENARIO_1910631192_CONTRADICTION_REPRODUCED
SCENARIO_1910631192_CONTRADICTION_OLD_SDK_OR_MAP_ARTIFACT
SCENARIO_1910631192_CONTRADICTION_PROTOCOL_DIFFERENCE
SCENARIO_1910631192_CONTRADICTION_SIDE_OR_SEED_DEPENDENT
SCENARIO_1910631192_CONTRADICTION_UNRESOLVED
SCENARIO_1910631192_EVALUATION_BLOCKED
```

Do not force a favorable classification.

## Phase 3 — SDK candidate generation diagnostic

Create:

```text
reports/v3g_full_route_coverage/03_sdk_candidate_generation_diagnostic.md
reports/v3g_full_route_coverage/sdk_candidate_generation_diagnostic.json
```

Inspect where SDK candidate lists are consumed by the agent/policy.

Required distinctions:

* SDK candidate list empty before policy;
* policy filters candidates to empty;
* route planner called and returns empty path;
* fallback selected due to no candidate;
* illegal/no-op selected;
* non-move actions incorrectly counted as route failures;
* deployment-phase actions contaminating route metrics.

Add instrumentation if needed, preferably in evaluator code rather than core policy code.

Metrics to record:

* raw SDK Move candidate count;
* route-eligible candidate count;
* reachable candidate count;
* selected move type;
* candidate move type distribution;
* empty candidate reason;
* route planner invocation count;
* route planner success count;
* route planner empty path count;
* fallback move count;
* no-op count;
* illegal action count.

If core code changes are needed, restrict to:

* `ai/policy.py`
* `ai/agent.py`
* route diagnostic helper code
* tests

Do not modify checkpoints or SDK internals.

## Phase 4 — Extend full-coverage route evaluator

Create or update:

```text
scripts/run_v3g_full_route_coverage_evaluation.py
```

It may wrap or extend:

```text
scripts/run_v3f_route_reliability_evaluation.py
```

Required CLI support:

```bash
python scripts/run_v3g_full_route_coverage_evaluation.py \
  --scenario-source archived_league \
  --max-scenarios 100 \
  --seeds 0 \
  --out experiments/shenji_agent/v3g_full_route_coverage
```

Also support a smaller smoke mode:

```bash
python scripts/run_v3g_full_route_coverage_evaluation.py \
  --scenario-source archived_league \
  --max-scenarios 5 \
  --seeds 0 \
  --out experiments/shenji_agent/v3g_full_route_coverage_smoke
```

Required output files:

* `experiments/shenji_agent/v3g_full_route_coverage/results.json`
* `experiments/shenji_agent/v3g_full_route_coverage/per_game.jsonl`
* `experiments/shenji_agent/v3g_full_route_coverage/per_scenario.csv`
* `experiments/shenji_agent/v3g_full_route_coverage/summary.md`
* `reports/v3g_full_route_coverage/04_evaluator.md`

Required per-game fields:

* seed;
* scenario_id;
* map_id;
* authenticated;
* deployment_entered;
* battle_entered;
* game_completed;
* turns;
* total_actions;
* move_actions;
* route_eligible_move_actions;
* raw_sdk_move_candidate_count_total;
* raw_sdk_move_candidate_count_zero_events;
* route_planner_invocation_count;
* route_planner_success_count;
* route_planner_empty_path_count;
* route_empty_count;
* route_empty_rate;
* no_valid_move_candidate_count;
* fallback_move_count;
* illegal_action_count;
* no_op_count;
* selected_move_type_counts;
* candidate_move_type_counts;
* failure_reasons;
* runtime_errors.

Evaluator must fail closed if:

* SDK auth fails;
* battle phase is not entered;
* route denominator is missing;
* scenario/map cannot be loaded;
* diagnostics are unavailable.

## Phase 5 — Smoke evaluation

Run a bounded smoke before full coverage:

```bash
python scripts/run_v3g_full_route_coverage_evaluation.py \
  --scenario-source archived_league \
  --max-scenarios 5 \
  --seeds 0 \
  --out experiments/shenji_agent/v3g_full_route_coverage_smoke
```

Create:

```text
reports/v3g_full_route_coverage/05_smoke_evaluation.md
reports/v3g_full_route_coverage/smoke_evaluation.json
```

Required checks:

* SDK authenticated;
* battle phase entered;
* scenario IDs loaded from archived league inventory;
* route denominators nonzero when move actions occur;
* candidate-generation diagnostics emitted;
* route planner invocation count separated from SDK-empty candidate count;
* no sensitive files produced.

If smoke fails, stop with `SAFE_STOP_FULL_ROUTE_COVERAGE_BLOCKED`.

## Phase 6 — Full archived league scenario coverage evaluation

Run authenticated full route coverage over the archived league scenario set.

Preferred command:

```bash
python scripts/run_v3g_full_route_coverage_evaluation.py \
  --scenario-source archived_league \
  --max-scenarios 100 \
  --seeds 0 \
  --out experiments/shenji_agent/v3g_full_route_coverage
```

If 100 scenarios is too expensive, use the maximum bounded scenario count that is feasible, but document the limitation clearly.

Create:

```text
reports/v3g_full_route_coverage/06_full_coverage_evaluation.md
reports/v3g_full_route_coverage/full_coverage_evaluation.json
```

Required aggregate metrics:

* scenarios attempted;
* scenarios completed;
* battle_entered_count;
* games completed;
* total move actions;
* route-eligible move actions;
* raw SDK Move candidate zero events;
* `no_valid_move_candidate` count;
* route planner invocation count;
* route planner empty path count;
* route empty count;
* route empty rate;
* empty rate by scenario;
* empty rate by map;
* empty rate by move type;
* empty rate by seed;
* max scenario empty rate;
* mean scenario empty rate;
* median scenario empty rate;
* top failure reasons;
* worst scenarios;
* whether scenario `1910631192` reproduces archived behavior.

## Phase 7 — Compare current full coverage against archived league

Create:

```text
reports/v3g_full_route_coverage/07_archived_vs_current_comparison.md
reports/v3g_full_route_coverage/archived_vs_current_comparison.json
```

Compare:

* archived league 66.3% empty rate;
* current V3G full coverage empty rate;
* scenario overlap;
* map overlap;
* scenario `1910631192`;
* move-type-specific empty rates;
* candidate-generation-specific empty rates;
* route planner invocation rate;
* route planner failure rate after invocation;
* protocol differences;
* SDK/map data differences if detectable.

Classify archived-vs-current relationship as exactly one:

```text
ARCHIVED_ROUTE_FAILURE_REPRODUCED_CURRENT
ARCHIVED_ROUTE_FAILURE_NOT_REPRODUCED_CURRENT
ARCHIVED_ROUTE_FAILURE_PARTIALLY_REPRODUCED_CURRENT
ARCHIVED_ROUTE_FAILURE_EXPLAINED_BY_PROTOCOL_DIFFERENCE
ARCHIVED_ROUTE_FAILURE_EXPLAINED_BY_SDK_OR_MAP_VERSION_DIFFERENCE
ARCHIVED_ROUTE_FAILURE_UNRESOLVED
```

If archived failure is not reproduced, do not claim it is irrelevant. State what changed or what remains unknown.

## Phase 8 — Route claim gate

Create:

```text
reports/v3g_full_route_coverage/08_route_claim_gate.md
reports/v3g_full_route_coverage/route_claim_gate.json
```

Use exactly one route status:

```text
ROUTE_RELIABILITY_SUPPORTED_AUTHENTICATED_FULL_COVERAGE
```

Use only if:

* authenticated replay covers the full or near-full archived league scenario set;
* route denominator is meaningful;
* route-empty rate is low;
* scenario `1910631192` contradiction is resolved;
* archived 66.3% failure is explained or not reproduced under matched current protocol;
* route planner and SDK candidate generation are both bounded under tested coverage.

```text
ROUTE_RELIABILITY_PARTIALLY_SUPPORTED_SCENARIO_DEPENDENT
```

Use if:

* some scenarios have low empty route;
* other scenarios or move types still show high `no_valid_move_candidate`;
* route reliability is scenario-dependent.

```text
ROUTE_NO_DEGRADATION_OBSERVED_ONLY
```

Use if:

* V3G expands coverage somewhat;
* no major degradation is observed;
* but coverage or contradiction resolution remains insufficient for stronger claim.

```text
ROUTE_RELIABILITY_NOT_SUPPORTED
```

Use if:

* archived high empty-route behavior reproduces;
* candidate generation remains frequently empty;
* route failure remains central.

```text
ROUTE_EVALUATION_BLOCKED
```

Use if:

* full coverage cannot run;
* SDK auth fails;
* battle phase cannot be entered;
* route denominators cannot be produced.

Allowed wording if full coverage supports it:

```text
Authenticated full-coverage route evaluation shows bounded route-empty behavior under the tested archived league scenario set.
```

Allowed wording if partially supported:

```text
Authenticated route evaluation indicates that route-empty behavior is strongly scenario-dependent, with no evidence of a route planner failure but persistent SDK candidate-generation limitations in selected scenarios or move types.
```

Forbidden unless strongly proven:

```text
Route reliability is solved generally.
The route planner is robust across the SDK.
The archived 66.3% failure is irrelevant.
All scenarios are route-safe.
```

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

* V3G full route coverage result;
* scenario count and denominator;
* route claim gate;
* scenario `1910631192` contradiction resolution or unresolved status;
* archived-vs-current comparison;
* remaining route limitations;
* manuscript readiness implications.

Do not modify tactical status except to preserve:

* tactical action-level / diagnostic supported;
* tactical outcome weak or inconclusive.

Do not claim:

* tactical outcome contribution;
* route reliability solved generally;
* two-window promotion success;
* public SOTA;
* top-tier readiness.

## Phase 10 — Tests

Add or update:

```text
tests/test_v3g_full_route_coverage.py
```

Tests should cover:

* evaluator records raw SDK empty Move candidate events separately from route planner empty path events;
* route planner empty count cannot increase if route planner was never invoked;
* `no_valid_move_candidate` is not mislabeled as A* route planner failure;
* route denominator is nonzero when move actions occur;
* scenario `1910631192` is included in focused audit if available;
* route claim gate cannot mark full support without scenario coverage and denominator evidence;
* archived 66.3% route failure cannot be dismissed without current comparison.

Run:

```bash
python -m unittest discover -v
python -m compileall paper/scripts scripts ai tests training
```

If pytest exists and is useful:

```bash
python -m pytest tests/test_v3g_full_route_coverage.py -q 2>/dev/null || true
```

Create:

```text
reports/v3g_full_route_coverage/10_validation_log.md
```

Record commands, exit codes, summary, warnings, and failures.

## Phase 11 — Manuscript readiness update

Create:

```text
reports/v3g_full_route_coverage/11_manuscript_readiness_update.md
reports/v3g_full_route_coverage/manuscript_readiness_update.json
```

Classify manuscript readiness as exactly one:

```text
READY_FOR_BASELINE_EXPANSION
```

Use if route evidence is now sufficiently bounded and the next blocker is baseline/component ablation coverage.

```text
READY_FOR_CONSERVATIVE_MANUSCRIPT_WITH_ROUTE_LIMITATIONS
```

Use only if route evidence is bounded enough, baseline evidence is acceptable for a conservative SDK evaluation paper, and limitations are explicit.

```text
NEEDS_ROUTE_COVERAGE_EXPANSION
```

Use if V3G coverage remains too narrow.

```text
NEEDS_SDK_CANDIDATE_GENERATION_INVESTIGATION
```

Use if no_valid_move_candidate remains high or scenario `1910631192` remains unresolved.

```text
NEEDS_BASELINE_EXPANSION_BEFORE_MANUSCRIPT
```

Use if route is no longer central and baseline evidence is the main blocker.

```text
NOT_READY_FOR_MANUSCRIPT
```

Use if route/baseline/promotion evidence remains internally inconsistent.

## Phase 12 — Final decision

Create:

```text
reports/v3g_full_route_coverage/FINAL_DECISION.md
reports/v3g_full_route_coverage/final_decision.json
```

Use exactly one final state:

```text
FULL_ROUTE_COVERAGE_SUPPORTS_BOUNDED_ROUTE_CLAIM
```

Use if full authenticated archived scenario coverage supports bounded route reliability.

```text
FULL_ROUTE_COVERAGE_PARTIAL_SCENARIO_DEPENDENT
```

Use if route behavior is scenario-dependent but better characterized.

```text
FULL_ROUTE_COVERAGE_REPRODUCES_ARCHIVED_FAILURE
```

Use if current authenticated evaluation reproduces high empty-route behavior.

```text
FULL_ROUTE_COVERAGE_RESOLVES_ARCHIVED_FAILURE_AS_OLD_ARTIFACT
```

Use only if evidence shows archived high empty-rate was due to old SDK/map/protocol artifact.

```text
FULL_ROUTE_COVERAGE_UNRESOLVED_NEEDS_SDK_INVESTIGATION
```

Use if scenario `1910631192` or candidate generation behavior remains unexplained.

```text
SAFE_STOP_FULL_ROUTE_COVERAGE_BLOCKED
```

Use if full coverage cannot run safely.

Final decision must recommend the next goal:

* `V3H baseline expansion`
* `V3H SDK candidate generation investigation`
* `V3H conservative manuscript rewrite`
* `V3H route scenario coverage expansion`
* `V3H promotion evidence rebuild`
* or `SAFE_STOP evidence reconciliation`

## Phase 13 — Commit

Before staging:

```bash
git status --short
git diff --name-only
git diff --cached --name-only | grep -Ei 'engine_config|license|auth|key|secret|token|\.pt$|\.pth$|\.zip$|\.tar$|\.tgz$|\.whl$' || true
```

Stage only reports, scripts, tests, evidence text, and goal:

```bash
git add reports/v3g_full_route_coverage
git add experiments/shenji_agent/v3g_full_route_coverage 2>/dev/null || true
git add experiments/shenji_agent/v3g_full_route_coverage_smoke 2>/dev/null || true
git add experiments/shenji_agent/v3g_scenario_1910631192_focus 2>/dev/null || true
git add scripts/run_v3g_full_route_coverage_evaluation.py 2>/dev/null || true
git add scripts/run_v3f_route_reliability_evaluation.py 2>/dev/null || true
git add tests/test_v3g_full_route_coverage.py 2>/dev/null || true
git add paper/v3_evidence_package paper/source_evidence.md 2>/dev/null || true
git add goals/codex_goal_v3g_full_route_coverage_expansion.md 2>/dev/null || true
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
git commit -m "expand authenticated route coverage across league scenarios"
```

## SUCCESS criteria

SUCCESS if:

* archived league scenario inventory is complete or limitations are documented;
* scenario `1910631192` is specifically audited;
* SDK empty Move candidate generation is distinguished from route planner failure;
* route evaluator supports full or bounded archived scenario coverage;
* smoke evaluation runs;
* full or maximum feasible coverage evaluation runs;
* archived-vs-current comparison is written;
* route claim gate is written conservatively;
* V3 evidence package is updated;
* tests and compileall pass;
* no unsupported route, tactical, promotion, SOTA, or top-tier claim is introduced;
* no sensitive SDK auth/license files or large checkpoints are committed.

## SAFE_STOP criteria

SAFE_STOP if:

* SDK authentication regresses;
* archived scenario IDs cannot be recovered;
* full coverage cannot enter battle phase;
* route diagnostics cannot produce denominator evidence;
* scenario `1910631192` cannot be located or evaluated and this blocks the goal;
* route metrics are internally inconsistent;
* validation fails in a way that invalidates the route audit;
* sensitive SDK auth files would need to be committed.

If SAFE_STOP, create:

```text
reports/v3g_full_route_coverage/SAFE_STOP.md
```

It must state:

* what was attempted;
* exact blocker;
* whether V3F remains valid;
* current route claim boundary;
* which route claims remain forbidden;
* what human action is needed next.


