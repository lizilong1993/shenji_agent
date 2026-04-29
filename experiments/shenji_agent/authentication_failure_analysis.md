# Authentication Failure Analysis

## Status
Iteration 009 localized the `did not pass authentication` message to the SDK wheel and proved the current environment fails the visible MAC-bound SDK authentication check.

## Source
- Exact runtime message: `did not pass authentication`.
- Repository source search found no implementation of that string outside reports.
- Wheel string inspection found the string in `train_env/env/authenticate/authenticator.cpython-310-x86_64-linux-gnu.so`.
- The same binary exposes `Authenticator.pass_MAC_check`, `Authenticator.pass_time_check_since_first_use`, `Authenticator.run_authenticate_processes`, `did pass authentication`, and `did not pass authentication`.
- `train_env/env/train_env.cpython-310-x86_64-linux-gnu.so` references `authenticate.authenticator`, `Authenticator`, `run_authenticate_processes`, and `pass_authenticaiton`.
- Installed backend: `/usr/local/lib/python3.10/dist-packages/train_env/__init__.cpython-310-x86_64-linux-gnu.so`.
- SDK version printed at import/runtime: `4.1.0`.

## Engine Config Evidence
- Installed config path: `/usr/local/lib/python3.10/dist-packages/train_env/env/authenticate/.engine_config`.
- Decoded config value: `02:42:ac:11:00:45,1777298501090988694`.
- Current visible non-loopback MAC: `02:42:ac:11:00:02`.
- Iteration 009 self-play diagnostics classify this as `failed_mac_mismatch`.
- No relevant `AUTH`, `LICENSE`, `TOKEN`, `TRAIN`, `WARGAME`, `ENGINE`, or `SDK` environment variable was present.

## Minimal Reproduction
Command shape:
`python3.10 - <<'PY' ... from train_env import TrainEnv; env=TrainEnv(); env.setup(env_info) ... PY`

Observed output:
- `SDK version: 4.1.0`
- `did not pass authentication`

The setup call still returns observations, but the match remains in `time.stage == 1` deployment/setup flow.

## Impact on Runtime
- `TrainEnv.setup` returns state with `time.stage: 1`.
- Repeated `env.step` calls with deployment end actions `333` do not transition to stage `2`.
- Empty actions do not transition to stage `2`.
- Direct normal `Move` actions while stage is `1` are ignored and do not move units.
- Current self-play completion is therefore meaningless as promotion evidence: it runs fixed-length deployment/setup turns and returns `0.0-0.0`.

## Authentication Target
Visible wheel symbols and config indicate a local SDK engine authentication/license binding, specifically a MAC-bound engine config plus a timestamp/time check. There is no evidence in repository code that the message is produced by a remote replay service, telemetry service, user token, Python wrapper, runner config, or scenario file.

## Conclusion
The authentication failure is not a harmless log in this environment. It correlates with `TrainEnv` staying in deployment/setup stage and prevents normal battle-phase action execution. Because the SDK authentication code is compiled inside the wheel and the installed engine config is bound to a different MAC, this cannot be fixed in repository code without new valid SDK authorization/configuration from the SDK/platform owner.
