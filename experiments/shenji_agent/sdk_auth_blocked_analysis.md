# SDK Auth Blocked Analysis

## Final Blocker
Current self-play is blocked by SDK authentication/license binding. The installed SDK wheel is present and callable, but its local engine authentication fails with a MAC mismatch and the environment never reaches battle phase.

## Evidence
- Message source: `train_env/env/authenticate/authenticator.cpython-310-x86_64-linux-gnu.so` inside `land_wargame_train_env-4.1.0`.
- Outer call path: `training.selfplay._run_single -> TrainEnv() -> TrainEnv.setup(env_info) -> SDK authenticator`.
- Installed engine config decodes to bound MAC `02:42:ac:11:00:45`.
- Current container MAC is `02:42:ac:11:00:02`.
- Runtime output repeats `did not pass authentication`.
- `phase_action_diagnostics.json` records `auth_status=failed_mac_mismatch`.

## Does It Block Battle Phase?
Yes in this environment.

Observed probes:
- After `env.setup`, `time.stage == 1`.
- Repeated `333` deployment-end actions do not transition to `stage == 2`.
- Empty actions do not transition to `stage == 2`.
- Direct `Move` actions while stage is `1` are ignored.
- Smoke `league_20260429_094929` has `battle_phase_entered=false` across all `640` side-turn records.

## Offline Mode
- The project uses local `train_env.TrainEnv`; no remote server URL or token config was found in configs/scripts/docs.
- There is no repository-level offline-mode flag that bypasses SDK authentication.
- `mock_train_env.py` can be used for unit contracts only; `training.selfplay.main` intentionally rejects `mock_train_env.MockTrainEnv` for formal self-play.
- Mock evidence cannot be used for promotion.

## Why Current Evidence Cannot Promote
- Scores remain `0.0-0.0`.
- No battle phase is reached.
- No selected Move/Shoot/Occupy actions occur in battle phase.
- No Occupy success, score delta, damage, or kill records are collected.
- Objective labels from this run would again be zero/no-op and not promotion-quality.
- Iteration 009 now forces `promotion_accepted=false` when battle phase is absent.

## Minimal Reproduction Command
Use the current repo and environment:

```bash
cd /remote-home/lizilong/shenji_agent_fresh
mkdir -p .tmp_claude
TMPDIR=$PWD/.tmp_claude TEMP=$PWD/.tmp_claude TMP=$PWD/.tmp_claude \
PYTHONPATH=/remote-home/lizilong/.local/lib/python3.10/site-packages:. \
python3.10 -m training.selfplay \
  --config configs/selfplay_smoke_battle_phase.yaml \
  --candidate ai/weights \
  --incumbent ai/weights_incumbent_rule
```

Expected current result:
- `did not pass authentication`.
- `phase_action_diagnostics.json` shows `battle_phase_entered=false`.
- `auth_status_diagnostics.json` shows `failed_mac_mismatch`.
- `promotion.json` has `accepted=false`.

## Required External Condition
An SDK/platform owner must provide valid authorization for the current Linux runtime, such as:
- a regenerated `.engine_config` bound to the current runtime;
- a licensed SDK wheel/config package for this host/container;
- or official instructions for running the SDK in an authenticated offline mode.

Do not fake or bypass this check in repository code.

## Commands to Run After Auth Is Fixed
1. `python3.10 -m pytest tests -q`
2. `python3.10 -m training.selfplay --config configs/selfplay_smoke_battle_phase.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule`
3. Inspect `phase_action_diagnostics.json` for `battle_phase_entered=true` and selected Move/Shoot/Occupy actions.
4. If battle actions appear, run `python3.10 scripts/build_objective_labels.py --input <latest_smoke> --output-dir experiments/shenji_agent/objective_labels/latest_auth_fixed`.
5. If objective progress appears, continue objective scorer/policy loop and formal self-play.
