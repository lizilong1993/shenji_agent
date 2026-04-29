# Valid Action Pipeline Analysis

## Status
Iteration 009 traced valid actions from `TrainEnv` to `Agent.step` and added phase/action sidecars so self-play can distinguish setup/deployment-only runs from battle-phase evidence.

## Pipeline
1. `training/selfplay.py` resolves the backend with `_resolve_train_env_class`.
2. On Linux it imports `train_env.TrainEnv`; current backend is `train_env.TrainEnv`, not `mock_train_env.MockTrainEnv`.
3. `_run_single` loads scenario/map data, constructs `player_info`, calls `env.setup(env_info)`, and receives `state`.
4. Each loop calls `red.step(state[RED]) + blue.step(state[BLUE])`.
5. `Agent.step` receives one side's observation, reads `observation["time"]`, `role_and_grouping_info`, `operators`, and `valid_actions`.
6. If `time.stage == 1`, the inherited SDK demo branch returns deployment actions `303` and `333`.
7. If not in stage `1`, `Agent.step` iterates `valid_actions[obj_id][action_type]` in priority order and generates normal actions such as `Move`, `Shoot`, and `Occupy`.
8. `training/selfplay.py` submits the combined action list to `env.step(actions)`.

## Schema
- `observation["valid_actions"]` is a dict keyed by operator id.
- Each value is a dict keyed by integer action type.
- `ActionType.Move == 1`, `Shoot == 2`, `GetOn == 3`, `GetOff == 4`, `Occupy == 5`.
- Deployment actions include `303` deployment get-on, `304` deployment get-off, and `333` end deployment.
- Documentation says `time.stage`: `0` environment config, `1` deployment, `2` normal advancement.

## Iteration 009 Runtime Findings
Minimal SDK probe on `2230531221.json`:
- After setup, red stage is `1`, blue stage is `1`.
- Red valid action counts include `Move: 20`, `GetOn: 8`, `GetOff: 4`, `ChangeState: 22`, `JMPlan: 6`, `WeaponLock: 18`.
- Blue valid action counts include `Move: 28`, `GetOn: 8`, `ChangeState: 30`, `JMPlan: 6`, `WeaponLock: 22`.
- `333` only does not transition to stage `2`.
- Empty actions do not transition to stage `2`.
- Direct normal `Move` actions while stage is `1` do not move units or advance `cur_step`.
- The SDK repeatedly prints `did not pass authentication`.

## Agent Behavior
- Before Iteration 009, stage `1` observations bypassed normal valid-action accounting and generated only deployment actions, making match stats look like they had no valid normal action maps.
- Iteration 009 now records raw valid action types before the stage branch and records generated deployment actions.
- The runner also records phase/action diagnostics independently from agent policy stats, so stale agent snapshots cannot hide SDK phase state.

## Smoke Evidence
Battle smoke artifact: `experiments/shenji_agent/league_20260429_094929`.

- `battle_phase_entered=false`.
- `total_turns=80`.
- `total_turn_records=640`.
- `turns_with_setup_deployment_only=640`.
- Valid Move opportunities appeared in deployment observations: `turns_with_move=640`.
- No battle Shoot or Occupy opportunities appeared: `turns_with_shoot=0`, `turns_with_occupy=0`.
- Selected normal Move/Shoot/Occupy actions: `0`.
- `auth_failure_count=640`.
- `no_battle_action_reason_distribution={'auth_failed_or_unverified_stage_stuck': 640}`.
- Promotion was forced to `accepted=false` with failures `battle_phase_not_entered` and `no_move_shoot_occupy_selected_actions`.

## Answers to Required Questions
1. TrainEnv returns side observations with `operators`, `cities`, `scores`, `time`, `valid_actions`, and related fields.
2. `valid_actions` is read in `Agent.step`; Iteration 009 also reads it in runner diagnostics.
3. Schema is `operator_id -> action_type -> candidate payload/null`.
4. `Move/Shoot/Occupy` are action types `1/2/5`.
5. Deployment actions are `303/304/333` and related `31x` deployment forms.
6. Agent priority is not used while `time.stage == 1`; the deployment branch returns early.
7. Agent identifies phase only via `time.stage == 1` and otherwise assumes normal action selection.
8. Runner calls `env.step` repeatedly, but SDK does not transition from stage `1` to stage `2` under failed authentication.
9. Both sides submit deployment actions; battle step is never reached.
10. Normal action payload schema is correct per docs, but normal Move submitted in stage `1` is ignored.
11. Candidate/incumbent loop in setup/deployment action flow because stage remains `1`.
12. `max_steps` is not the root cause; 80-step smoke and 120-step previous smoke both stayed stage `1`.
13. Smoke scenario is not setup-only: its observations expose normal Move valid actions, but SDK phase remains deployment.
14. Authentication failure is the visible reason the real backend behaves as stage-stuck; mock backend is not used for promotion.

## Conclusion
The valid-action pipeline is structurally wired, but the real SDK backend is stuck before battle phase. Iteration 009 prevents setup-only runs from being accepted as promotion evidence and emits sidecars that make this failure explicit.
