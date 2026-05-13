# SAFE_STOP

Date: 2026-05-13

Final state: `NOT_READY_SAFE_STOP`

## Why This Stops

The V2 upgrade cannot honestly reach `TOP_TIER_SUBMISSION_CANDIDATE` in the current repo state.
The blockers are executable and material:

- Full candidate smoke remains fail-closed because `ai/weights/tactical_model.pt` is missing.
- No tactical checkpoint or tactical training CSV was found in the workspace.
- The current model-quality diagnostic resolves the restored intent/situation checkpoint paths,
  but the current-format diagnostic does not load them as usable policy checkpoints.
- The route empty-rate blocker has only V1 evidence: `15641 / 23580` empty routes
  (`0.6633163698049194`). No valid V2 before/after route experiment can be claimed without an
  executable candidate stack.
- Baselines, ablations, multi-seed statistics, and two-window hard-stop proof have not been
  generated.

Stopping here is the correct outcome under the goal constraints. Continuing to paper writing or
claim upgrades would overstate the evidence.

## What Was Fixed

- Preserved the Version 1 baseline under
  `reports/v2_top_tier_upgrade/version1_baseline.md`.
- Completed Phase 1 top-tier gap audit:
  `reports/v2_top_tier_upgrade/top_tier_gap_audit.md` and `.json`.
- Added a shared manifest asset resolver:
  `ai/artifacts.py`.
- Updated asset consumers to use manifest-driven resolution:
  `ai/policy.py`, `training/selfplay.py`,
  `scripts/run_round005_smoke_pipeline.py`, and
  `scripts/diagnose_model_quality.py`.
- Updated the candidate manifest to match restored intent/situation filenames:
  `intention_lstm.pth` and `situation_gcn.pth`.
- Added V2 asset-contract tests:
  `tests/test_v2_asset_contract.py`.
- Updated the stale diagnostic test expectation so a manifest-declared but absent tactical file
  is not treated as present.
- Recorded the Phase 2 reproducibility contract and asset contract.
- Recorded tactical demotion in `tactical_branch_audit.md` and `.json`.

## Top-Tier Gates Failed

| Gate | Status | Evidence |
|---|---|---|
| tests pass | pass | `python -m unittest discover -v`: 74 tests OK |
| compileall passes | pass | `python -m compileall paper/scripts scripts ai tests training`: exit 0 |
| asset contract passes | partial | intent/situation paths resolve; tactical missing fail-closed |
| checkpoint manifest passes | partial | manifest paths match files, but current model-quality diagnostic does not load intent/situation checkpoints |
| route gate passes | fail | V1 route empty-rate `0.6633163698049194` |
| tactical contribution proven or removed | removed from claims | tactical audit selects Path B |
| at least 4 meaningful baselines/ablations | fail | not generated |
| multi-seed evaluation | fail | not generated |
| statistical confidence intervals | fail | not generated |
| two-window success proven or removed | removed from claims | no V2 proof generated |
| method novelty supported | fail | no alternate novelty evidence generated |
| strict review simulation target | not run | downstream gates already fail |
| venue-fit supports top-tier candidate | fail | evidence insufficient |
| no forbidden claims remain | pass for new V2 docs | V2 readiness keeps claims forbidden |

## Exact Blockers

1. Missing tactical checkpoint:

   ```text
   ai/weights/tactical_model.pt
   ```

2. Missing tactical retraining data:

   ```text
   Wargame_Feature_Dataset.csv
   ```

   Search returned no file in the workspace.

3. Current-format learned candidate checkpoint usability is unresolved:

   ```text
   python scripts/diagnose_model_quality.py --artifacts ai/weights --output reports/v2_top_tier_upgrade/model_quality_v2.json
   ```

   The sidecar reports resolved intent/situation paths but `intent_checkpoint_loaded=false` and
   `situation_checkpoint_loaded=false`.

4. Route reliability remains unsupported:

   ```text
   empty_route_count = 15641
   total_move_attempts = 23580
   empty_route_rate = 0.6633163698049194
   route_failure_reason = no_valid_move_candidate
   ```

5. Baseline, ablation, multi-seed, statistical, and two-window evidence are absent for V2.

## Recommended Target

Current best target: technical report / workshop-style artifact after reproducibility repair.

Do not target NeurIPS/CVPR/TPAMI main-track review until the exact next experiment below
can execute and produce supporting evidence.

## Exact Next Experiment Needed

First repair the executable candidate stack:

1. Provide or retrain current-format intent, situation, and tactical checkpoints with matching
   manifest entries.
2. Verify:

   ```bash
   python scripts/diagnose_model_quality.py --artifacts ai/weights --output reports/v2_top_tier_upgrade/model_quality_v2.json
   python scripts/run_round005_smoke_pipeline.py --out reports/v2_top_tier_upgrade/smoke_preflight_v2
   ```

3. Only after both are usable, run a small V2 route before/after smoke with route metrics:

   ```bash
   python scripts/round005_route_diagnostic.py --out reports/v2_top_tier_upgrade/route_before_after/before.json
   python -m training.selfplay --config configs/selfplay_minimal.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule
   python scripts/round005_route_diagnostic.py --league-dir <new_league_dir> --out reports/v2_top_tier_upgrade/route_before_after/after.json
   ```

4. If route gates pass, proceed to baseline/ablation and multi-seed statistical evidence.

Until then, `TOP_TIER_SUBMISSION_CANDIDATE` is not supported.
