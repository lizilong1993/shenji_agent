# Iteration 007: Training/Inference Normalizer Consistency and Situation Node Differentiation

## Status
In progress.

## Goal
Continue from Iteration 006 blocked state. P0 is fixing the situation-model training/inference normalization mismatch. P0/P1 is improving or proving the limits of situation per-node differentiation. Final validation remains smoke/formal self-play, not supervised metrics alone.

## TODO
- [x] Confirm fresh clone status and HEAD.
- [x] Read Iteration 006 reports and latest smoke artifacts.
- [x] Fix or record Windows tailscale SSH host key verification.
- [x] Create `experiments/shenji_agent/iteration_007.md`.
- [x] Inspect normalizer training/inference paths.
- [x] Fix training path that computed but did not apply the situation normalizer.
- [x] Add normalizer training/inference consistency tests.
- [x] Retrain normalized situation/intent/tactical models.
- [x] Run model quality diagnostics before retrain.
- [x] Run model quality diagnostics after retrain.
- [x] Check situation per-node variance after normalized retrain.
- [ ] If variance remains low, design and implement GraphSituationModel architecture variant.
- [ ] Train architecture variant.
- [ ] Compare baseline vs normalized vs architecture variant.
- [x] Run related tests.
- [ ] Linux commit.
- [ ] Windows sync + Windows commit.
- [ ] Run smoke self-play.
- [ ] Analyze whether smoke is still 0-0.
- [ ] If smoke improves, run formal self-play.
- [ ] Update final report/docs.
- [ ] Close as A completed or B blocked-with-hard-evidence.

## Initial State
- Workdir: `/remote-home/lizilong/shenji_agent_fresh`
- HEAD: `850df7c308be54df636c85ba4ed39f2278c6c753`
- Current branch state includes pre-existing dirty generated files: `__pycache__/*.pyc`, `ai/__pycache__/*.pyc`, untracked `.tmp_claude/`, `ai/weights/weights/`, `ai/weights_incumbent_rule/`, `datasets`, `land_wargame_sdk/land_wargame_sdk`, `scripts/launch_detached_compare.py`, `scripts/verify_learned_policy_engagement.py`.
- Disk: `/remote-home` has 86G available, 98% used.
- This iteration continues from Iteration 006 B-blocked evidence: smoke/self-play still produced 0-0 match scores, situation per-node variance remained about `1.5e-05`, and training normalizer was computed but not applied.

## Iteration 006 Artifact Facts
- `experiments/shenji_agent/iteration_006.md` records tactical model enabled/trained with validation accuracy about 96.2%.
- `ai/weights/model_manifest.json` records `situation_feature_normalizer`, `tactical_model`, and source commit `5c3931fd2b747d2dd7d07fee27abe66a223bd7c9`.
- `ai/weights/situation_feature_normalizer.json` exists and has 108 `feature_mean` / `feature_std` entries.
- Inference loads the normalizer in `ai/policy.py` and applies it when feature dimensions match.
- Before this iteration, `training/supervised.py` computed and saved the normalizer but trained and validated situation on raw `sample["x"]`.
- Situation input dim: 108.
- Training feature dim: 108.
- Inference situation feature dim: 108 via `operator_to_graph_feature()`.
- `experiments/shenji_agent/model_quality_after_retrain.json` recorded situation pre-sigmoid mean `0.5077075`, std `0.0057299`, output mean `0.5079135`, output std `0.0046930`, per-node variance mean `1.4867e-05`.
- Tactical model loaded in diagnostics and self-play artifacts.
- Intent move bias remained negative: Iteration 006 after-retrain file recorded about `-0.155`; rerun on current code before Iteration 007 retrain recorded `-0.2365`.
- Latest inspected smoke artifact: `experiments/shenji_agent/league_20260428_230957`.
- Smoke 0-0 evidence: `match_results.json` first and second legs both candidate `0.0`, opponent `0.0`; rating overall point estimate `0.5`; shoot coverage `0.0`.
- Smoke route evidence: learned path coverage `1.0`, move path coverage `1.0`, empty route count `0`, tactical loaded count `2`.
- The smoke `promotion.json` says accepted, but it is not A-complete evidence because formal gates were not run and score/shoot behavior still shows draw/no-engagement.

## Windows Sync Status
- Initial command: `tailscale ssh 100.127.108.19 "powershell -NoProfile -Command \"Write-Output 'tailscale-windows-ok'\""`
- Result: `No ED25519 host key is known for desktop-t3ll42l.tailf6b2dc.ts.net. and you have requested strict checking. Host key verification failed.`
- Safe remediation attempted:
  - listed `~/.ssh/known_hosts`;
  - removed `100.127.108.19`, `[100.127.108.19]:22`, and `desktop-t3ll42l.tailf6b2dc.ts.net` entries with `ssh-keygen -R`;
  - attempted `ssh-keyscan -H` for IP and hostname;
  - retried `tailscale ssh`.
- `tailscale ssh` still failed with the same host-key verification error.
- Controlled OpenSSH fallback attempted: `ssh -o StrictHostKeyChecking=accept-new 100.127.108.19 ...`
- Fallback result: `ssh: connect to host 100.127.108.19 port 22: Connection timed out`.
- Current Windows sync state: blocked. Linux work continues; every Linux commit will retry sync.

## P0 Normalizer Fix
- Added `_fit_feature_normalizer()` in `training/supervised.py`.
- Normalizer now uses train split only and records `feature_std_raw`, safe `feature_std`, `zero_std_count`, `zero_std_features`, `fitted_on=situation_train_split`, and `normalizer_kind=per_dimension_zscore`.
- Added `_apply_feature_normalizer()` and applied it to both situation train and validation samples.
- Situation checkpoint now records `training_applied_normalizer: true`.
- Published manifest now records normalizer kind, fitted split, feature dim, zero-std count, and `situation_training_applied_normalizer`.
- `ai/policy.py` now records normalizer diagnostics: loaded/applied counts, feature dim before normalization, before/after mean/std, normalizer feature dim, and mismatch reason.
- `scripts/diagnose_model_quality.py` now reports `normalizer_applied_in_training` and `normalizer_loaded_in_inference`, and warns on `normalizer_not_applied_in_training`.

## Diagnostics Before Normalized Retrain
Command:
`PYTHONPATH=/remote-home/lizilong/.local/lib/python3.10/site-packages python3.10 scripts/diagnose_model_quality.py --artifacts ai/weights --datasets datasets --output experiments/shenji_agent/model_quality_before_iter007_normalized_retrain.json`

Result:
- situation loaded: true.
- intent loaded: true.
- tactical loaded: true.
- normalizer loaded: true.
- situation pre-sigmoid mean: `0.505834`.
- situation pre-sigmoid std: `0.005960`.
- situation output mean: `0.508567`.
- situation per-node variance mean: `1.5056050870043691e-05`.
- intent move bias estimate: `-0.2365`.
- warnings: `situation_pre_sigmoid_near_zero_variance`, `intent_negative_move_bias`, `normalizer_not_applied_in_training`.

## Tests
Environment:
`PYTHONPATH=/remote-home/lizilong/.local/lib/python3.10/site-packages python3.10`

Results:
- `tests/test_feature_normalizer_consistency.py -q`: `8 passed in 33.22s`
- `tests/test_model_quality_diagnostics.py -q`: `8 passed in 22.97s`
- `tests/test_policy_score_tuning.py -q`: `12 passed in 0.86s`
- `tests/test_mainline_contract.py -q`: `35 passed in 1.13s`

## Linux Commits
- `77849cd38e80fd3281e19e6f0ebfd074ad44fae5` — P0 normalizer consistency code, diagnostics, tests, before-retrain diagnostics, and initial Iteration 007 report.

## Windows Sync
- Sync attempt after `77849cd38e80fd3281e19e6f0ebfd074ad44fae5` used `.tmp_claude/sync_files.txt`.
- Sync file list:
  - `ai/policy.py`
  - `experiments/shenji_agent/iteration_007.md`
  - `experiments/shenji_agent/model_quality_before_iter007_normalized_retrain.json`
  - `scripts/diagnose_model_quality.py`
  - `tests/test_feature_normalizer_consistency.py`
  - `training/supervised.py`
- Result: failed with `No ED25519 host key is known for desktop-t3ll42l.tailf6b2dc.ts.net. and you have requested strict checking. Host key verification failed.`
- Windows commit sha: unavailable, sync blocked.

## Normalized Retrain
Command:
`PYTHONPATH=/remote-home/lizilong/.local/lib/python3.10/site-packages TMPDIR=$PWD/.tmp_claude TEMP=$PWD/.tmp_claude TMP=$PWD/.tmp_claude python3.10 train_models.py --config configs/supervised_score_quality.yaml`

Artifacts:
- Run dir: `experiments/shenji_agent/20260428_235555`
- Published dir: `ai/weights`
- Report: `experiments/shenji_agent/normalized_retrain_report.md`

Training results:
- Intent immediate validation accuracy: `0.6221`
- Tactical validation accuracy: `0.9622`
- Situation validation accuracy: `0.5455`
- Situation samples: `108`, all complex bucket.
- Situation normalizer fitted on `situation_train_split`, feature dim `108`, zero-std count `50`, `training_applied_normalizer=true`.
- Manifest source commit: `77849cd38e80fd3281e19e6f0ebfd074ad44fae5`.

After-retrain diagnostics:
- Output file: `experiments/shenji_agent/model_quality_after_iter007_normalized_retrain.json`
- normalizer_applied_in_training: `true`
- normalizer_loaded_in_inference: `true`
- situation pre-sigmoid mean/std/min/max: `0.411455` / `0.057212` / `0.330712` / `0.581594`
- situation output mean/std: `0.414247` / `0.008354`
- situation per-node variance mean: `0.0040744750294834375`
- tactical_available: `true`
- intent move bias estimate: `-0.2010`
- warnings: `intent_negative_move_bias`

Before/after conclusion:
- Explicit training/inference mismatch is fixed.
- Situation per-node variance improved from `1.5056e-05` to `0.004074`, roughly 270x.
- Since variance materially improved, the next branch is smoke self-play before implementing a GNN architecture variant.

## Next
1. Commit normalized retrain manifest/normalizer/reports.
2. Retry Windows sync after commit.
3. Run smoke self-play against `ai/weights_incumbent_rule`.
4. If smoke is still 0-0, analyze whether to proceed to architecture variant or blocked evidence.
5. If smoke improves, run formal self-play.
