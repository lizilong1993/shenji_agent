# Shenji Agent Final Autonomous Blocked Report

## Final Status
B blocked-with-hard-evidence.

The Iteration 007 P0 training/inference normalization mismatch is fixed and verified. Normalized retraining materially improved situation node differentiation, but smoke and medium self-play still fail score/Elo gates. The remaining blocker is objective/label insufficiency: the current graph-level situation labels and tactical/action labels do not train a policy to produce reward-changing city capture or combat outcomes.

## Current State
- Date: 2026-04-29 Asia/Shanghai.
- Working directory: `/remote-home/lizilong/shenji_agent_fresh`.
- Original repo: `/remote-home/lizilong/shenji_agent`.
- Linux fresh clone HEAD before this report refresh: `2055da3ff8714b6149107e7c1762b1970fbc4554`.
- Reverts: none.
- Windows target: `100.127.108.19`, `C:\Users\lizilong\Desktop\WHUAI\Projects\SituationAwareness\code\miaosuan20260119`.
- Windows sync status: blocked. `tailscale ssh` still fails with `No ED25519 host key is known for desktop-t3ll42l.tailf6b2dc.ts.net... Host key verification failed.`
- OpenSSH fallback status: `ssh -o StrictHostKeyChecking=accept-new 100.127.108.19 ...` timed out on port 22.
- Final retry on 2026-04-29: `tailscale ssh` still failed with the same host-key verification error; OpenSSH fallback again timed out on port 22.

## Commits This Iteration
- `77849cd38e80fd3281e19e6f0ebfd074ad44fae5` — fixed situation normalizer consistency and tests.
- `30d9f7b922721d010dc05f6fe961eaceba4f762f` — retrained normalized models and recorded diagnostics.
- `ff4804fbda0cd598bcbd624398eddea5f5b19cc7` — added self-play action-type diagnostics.
- `f4e0dcf2dcbb9af0fc80f26b742099188b704fd5` — pointed manifest at diagnostics-capable agent code.
- `2055da3ff8714b6149107e7c1762b1970fbc4554` — documented Iteration 007 blocked evidence, medium self-play artifacts, and README/final report updates.

## Changed Files
- Source: `training/supervised.py`, `ai/policy.py`, `scripts/diagnose_model_quality.py`, `ai/agent.py`, `training/selfplay.py`.
- Tests: `tests/test_feature_normalizer_consistency.py`, `tests/test_selfplay_action_diagnostics.py`.
- Artifacts/reports: `experiments/shenji_agent/iteration_007.md`, `experiments/shenji_agent/normalized_retrain_report.md`, `experiments/shenji_agent/situation_model_architecture_blocked_analysis.md`, `experiments/shenji_agent/model_quality_before_iter007_normalized_retrain.json`, `experiments/shenji_agent/model_quality_after_iter007_normalized_retrain.json`, `experiments/shenji_agent/20260428_235555/*`.
- Published metadata: `ai/weights/model_manifest.json`, `ai/weights/situation_feature_normalizer.json`.
- Generated local weights used for evaluation but ignored by git: `ai/weights/intent_lstm.pt`, `ai/weights/situation_gcn.pt`, `ai/weights/tactical_model.pt`.

## Normalizer Evidence
- Before: normalizer loaded in inference, but not applied during training.
- After: `normalizer_applied_in_training=true`, `normalizer_loaded_in_inference=true`.
- Normalizer kind: `per_dimension_zscore`.
- Fitted on: `situation_train_split`.
- Feature dim: `108`.
- Zero-std count: `50`, safely transformed with std `1.0`.
- Manifest records: `situation_training_applied_normalizer=true`.

## Training Evidence
Command:
`PYTHONPATH=/remote-home/lizilong/.local/lib/python3.10/site-packages TMPDIR=$PWD/.tmp_claude TEMP=$PWD/.tmp_claude TMP=$PWD/.tmp_claude python3.10 train_models.py --config configs/supervised_score_quality.yaml`

Run dir: `experiments/shenji_agent/20260428_235555`.

Results:
- Intent immediate validation accuracy: `0.6221`.
- Tactical validation accuracy: `0.9622`.
- Situation validation accuracy: `0.5455`.
- Situation samples: `108`.
- Tactical samples: `24080`.

## Model Quality Before/After
Before: `experiments/shenji_agent/model_quality_before_iter007_normalized_retrain.json`.

After: `experiments/shenji_agent/model_quality_after_iter007_normalized_retrain.json`.

| Metric | Before | After |
|---|---:|---:|
| normalizer_applied_in_training | `false` | `true` |
| situation_pre_sigmoid_std | `0.005960` | `0.057212` |
| situation_output_mean | `0.508567` | `0.414247` |
| situation_per_node_variance_mean | `1.5056e-05` | `0.004074` |
| tactical_available | `true` | `true` |
| intent_move_bias_estimate | `-0.2365` | `-0.2010` |
| warning_count | `3` | `1` |

## Self-Play Evidence
Smoke artifact: `experiments/shenji_agent/league_20260429_001342`.

Smoke result:
- 2 paired matches, all legs `0.0-0.0`.
- Learned/move coverage: `1.0 / 1.0`.
- Shoot coverage: `0.0`.
- Valid action diagnostics show no valid `Shoot` or `Occupy` in the 50-step smoke window.
- Promotion accepted in smoke-only thresholds, but not A-complete evidence.

Medium artifact: `experiments/shenji_agent/league_20260429_001512`.

Medium result:
- 5 paired matches, all legs `0.0-0.0`.
- Overall point estimate: `0.5`.
- Complex point estimate: `0.5`.
- Learned/move/shoot coverage: `1.0 / 1.0 / 1.0`.
- Valid `Shoot`: `28`.
- Generated learned `Shoot`: `28`.
- Promotion accepted: `false`.
- Promotion failures: `overall_point_estimate`, `overall_ci_lower`, `complex_point_estimate`, `complex_ci_lower`, `elo_delta_lower`.

Route and action status:
- Empty route remains `0`.
- Learned policy engagement remains `1.0`.
- Tactical model loaded in all matches.
- The candidate can generate learned move and shoot actions, but reward remains zero.

## Architecture/Data Blocker
Analysis file: `experiments/shenji_agent/situation_model_architecture_blocked_analysis.md`.

Key facts:
- Situation dataset has `108` graph samples, each with `12` nodes.
- Labels are graph-level binary labels only.
- Label distribution: `72` negative / `36` positive.
- There are no node-level, city-level, target-level, occupy-success, damage, or score-delta labels.
- The original architecture symptom, near-zero node variance, improved about 270x after the normalizer fix.
- Therefore, GNN architecture is no longer the immediate proven blocker; the hard blocker is missing objective-level supervision.

## Verification
- `tests/test_feature_normalizer_consistency.py`: `8 passed`.
- `tests/test_model_quality_diagnostics.py`: `8 passed`.
- `tests/test_policy_score_tuning.py`: `12 passed`.
- `tests/test_mainline_contract.py`: `35 passed`.
- `tests/test_selfplay_action_diagnostics.py`: `3 passed`.
- `tests/test_mainline_contract.py tests/test_move_route_diagnostics.py`: `45 passed`.
- Final related-suite refresh: `tests/test_feature_normalizer_consistency.py tests/test_model_quality_diagnostics.py tests/test_policy_score_tuning.py tests/test_mainline_contract.py tests/test_selfplay_action_diagnostics.py`: `66 passed in 36.14s`.

## Remaining Risks
- Intent move bias remains negative.
- Medium self-play proves learned shooting happens, but scoring remains 0-0.
- Formal self-play was not run because smoke/medium did not show score improvement.
- Windows sync remains blocked by host-key verification despite safe Linux-side remediation attempts.
- Worktree still contains pre-existing generated/cache dirt such as `__pycache__`, `.tmp_claude/`, ignored weights, dataset symlinks, and incumbent artifacts.

## Final Git Status
- Linux status still contains generated/cache or pre-existing local assets: `__pycache__/*.pyc`, `ai/__pycache__/*.pyc`, `.tmp_claude/`, `ai/weights/weights/`, `ai/weights_incumbent_rule/`, `datasets`, `land_wargame_sdk/land_wargame_sdk`, `scripts/launch_detached_compare.py`, and `scripts/verify_learned_policy_engagement.py`.
- Windows status is unavailable because sync is blocked before remote command execution.

## Next Minimal Plan
1. Generate rollout-derived labels for score delta, occupy success, damage/kill delta, and per-operator/per-city counterfactual value.
2. Train a node/city-level situation head on those labels.
3. Re-run model diagnostics, 50-step smoke, and 400-step medium self-play.
4. Only run formal self-play after non-zero score behavior or candidate/baseline separation appears.
5. Fix Windows Tailscale SSH trust/known-host handling from a surface that can accept the host key, then sync Linux commits and generated weights.
