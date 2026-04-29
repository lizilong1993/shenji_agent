# Shenji Agent Final Autonomous Blocked Report

## Final Status
B blocked-with-hard-evidence after Iteration 009.

Iteration 009 localized the runtime blocker to SDK authentication and added phase/action diagnostics that prevent setup-only self-play from being accepted as promotion evidence. The current repository can now prove when real self-play fails to reach battle phase, but it cannot repair the external SDK license/authentication mismatch from code without bypassing the SDK.

## Current State
- Working directory: `/remote-home/lizilong/shenji_agent_fresh`.
- Original repo: `/remote-home/lizilong/shenji_agent`; not synchronized.
- Linux HEAD after Iteration 009 manifest refresh: `e7cc4f31583690a45ffeca078cbcee55ef26adae`.
- Windows sync: intentionally not attempted.
- Windows target for later manual sync: `C:\Users\lizilong\Desktop\WHUAI\Projects\SituationAwareness\code\miaosuan20260119`.

## Authentication Failure Source
- The exact `did not pass authentication` string is not implemented in repository Python source.
- It is embedded in `train_env/env/authenticate/authenticator.cpython-310-x86_64-linux-gnu.so` inside `land_wargame_train_env-4.1.0`.
- `train_env/env/train_env.cpython-310-x86_64-linux-gnu.so` calls `authenticate.authenticator.Authenticator.run_authenticate_processes`.
- Installed `.engine_config` decodes to bound MAC `02:42:ac:11:00:45`.
- Current container MAC is `02:42:ac:11:00:02`.
- Iteration 009 sidecars classify the status as `failed_mac_mismatch`.
- No relevant auth/license/token environment variable was present.

## Battle Phase Evidence
Smoke artifact: `experiments/shenji_agent/league_20260429_094929`.

- Runtime printed `did not pass authentication`.
- `battle_phase_entered=false`.
- `first_battle_turn=None`.
- `total_turns=80`.
- `total_turn_records=640`.
- `turns_with_setup_deployment_only=640`.
- `turns_with_move=640` because deployment observations expose Move entries.
- `turns_with_shoot=0`.
- `turns_with_occupy=0`.
- Selected Move/Shoot/Occupy actions: `0`.
- `auth_failure_count=640`.
- no-battle reason distribution: `auth_failed_or_unverified_stage_stuck: 640`.

## Valid Action and Objective Evidence
- The scenario is not missing valid action data: stage-1 observations contain `Move` valid action entries.
- SDK remains in stage `1`, documented as deployment, and does not transition to stage `2`, documented as normal advancement.
- Repeated `333`, empty actions, and direct normal `Move` submissions did not reach battle phase in probes.
- Scores remain `0.0-0.0`.
- `action_outcome_diagnostics.json`: `5760` records, `0` score-delta actions, `0` damage actions, `0` occupy opportunities/successes.
- Objective labels were not rebuilt from this smoke because no promotion-valid battle outcomes exist.

## Code Changes
- `ai/agent.py`: records raw valid action types before stage-specific handling and counts generated deployment actions.
- `training/selfplay.py`: adds SDK auth status detection, phase/action diagnostics, valid action timeline, auth status sidecar, and battle-phase promotion gate.
- `configs/selfplay_smoke_battle_phase.yaml`: small smoke config for validating battle phase.
- `tests/test_phase_action_diagnostics.py`: validates setup-only/battle diagnostics and serialization.
- `tests/test_selfplay_battle_phase_contract.py`: validates promotion rejection when battle evidence is absent.
- `ai/weights/model_manifest.json`: points candidate source code to `3d3e69a5aa688cf4eeff948db60d507d90bee86c`, the commit containing battle-phase diagnostics code.

## Iteration 009 Commits
- `3d3e69a feat: add battle phase valid action diagnostics`
- `e7cc4f3 chore: point weights at battle phase diagnostics code`

## Reports Added
- `experiments/shenji_agent/authentication_failure_analysis.md`.
- `experiments/shenji_agent/valid_action_pipeline_analysis.md`.
- `experiments/shenji_agent/sdk_auth_blocked_analysis.md`.
- `experiments/shenji_agent/iteration_009.md`.

## Promotion Gate
- Smoke `promotion_accepted=false`.
- Failures: `battle_phase_not_entered`, `no_move_shoot_occupy_selected_actions`.
- Formal self-play was not run because smoke failed the battle-phase evidence contract.
- A-complete remains unsatisfied: no formal `promotion_accepted=true`, no non-zero score distribution, and no candidate superiority evidence.

## Verification
- `python3.10 -m pytest tests/test_phase_action_diagnostics.py tests/test_selfplay_battle_phase_contract.py tests/test_selfplay_action_diagnostics.py tests/test_objective_progress_diagnostics.py -q`: `15 passed in 1.00s`.
- `python3.10 -m pytest tests/test_phase_action_diagnostics.py tests/test_selfplay_battle_phase_contract.py tests/test_objective_label_builder.py tests/test_objective_policy_integration.py tests/test_mainline_contract.py -q`: `56 passed in 4.68s`.
- `python3.10 -m pytest tests -q`: `121 passed in 37.71s`.

## Remaining Blocker
The SDK/platform owner must provide valid authorization for the current Linux runtime, such as a regenerated `.engine_config`, a licensed SDK package for this container/host, or official authenticated offline-mode instructions. Repository code must not fake, bypass, or crack the SDK authentication check.

## Next Minimal Plan
1. Install/provide valid SDK auth for the current runtime.
2. Rerun `python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule`.
3. Confirm `battle_phase_entered=true` and selected Move/Shoot/Occupy actions appear.
4. Rebuild objective labels from the new smoke.
5. If score/city/damage progress appears, continue objective scorer/policy and formal self-play.

## Git Status
After Iteration 009 commits, remaining uncommitted entries are pre-existing/generated local entries: `__pycache__`, `.tmp_claude/`, `ai/weights/weights/`, `ai/weights_incumbent_rule/`, `datasets`, `land_wargame_sdk/land_wargame_sdk`, and two untracked helper scripts. The generated/cache/local asset entries are not part of the intended commit.

## Manual Sync Instructions
Suggested source directory:
`/remote-home/lizilong/shenji_agent_fresh`

Suggested target directory:
`C:\Users\lizilong\Desktop\WHUAI\Projects\SituationAwareness\code\miaosuan20260119`

Sync important files:
- `ai/**/*.py`
- `training/**/*.py`
- `scripts/**/*.py`
- `configs/**/*.yaml`
- `tests/**/*.py`
- `README.md`
- `program.md`
- `docs/**/*.md`
- `experiments/shenji_agent/**/*.md`
- `experiments/shenji_agent/**/summary.md`
- `experiments/shenji_agent/**/rating.json`
- `experiments/shenji_agent/**/promotion.json`
- `experiments/shenji_agent/**/participants.json`
- `experiments/shenji_agent/**/diagnostics.json`
- `experiments/shenji_agent/**/phase_action_diagnostics.json`
- `experiments/shenji_agent/**/valid_action_timeline.json`
- `experiments/shenji_agent/**/auth_status_diagnostics.json`
- `experiments/shenji_agent/**/objective*.json`
- `experiments/shenji_agent/**/objective*.jsonl`
- `ai/weights/model_manifest.json`

Do not sync:
- `.git`
- `.tmp_claude`
- `__pycache__`
- `.pytest_cache`
- `datasets`, unless explicitly needed
- `land_wargame_sdk`, unless explicitly modified
- large historical `experiments` directories unrelated to Iteration 009
- unrelated caches or local SDK/weight directories
