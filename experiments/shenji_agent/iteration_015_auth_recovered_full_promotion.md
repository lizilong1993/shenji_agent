# Iteration 015 Auth Recovered Full Promotion

## Baseline

- working_dir: `/remote-home/lizilong/shenji_agent`
- initial_head: `b7271b5f998cfba59a8687d8e5d3b81b684b142a`
- initial_status: tracked changes in `scripts/check_sdk_auth_environment.py`, `training/selfplay.py`, and modified Python bytecode under `ai/__pycache__`; many historical untracked local artifacts under `.agent/`, `.tmp_codex/`, local package files, and helper scripts.
- initial_diff_stat:

```text
ai/__pycache__/__init__.cpython-310.pyc | Bin 269 -> 177 bytes
ai/__pycache__/agent.cpython-310.pyc    | Bin 24679 -> 30155 bytes
ai/__pycache__/models.cpython-310.pyc   | Bin 2509 -> 4658 bytes
scripts/check_sdk_auth_environment.py   |   2 +-
training/selfplay.py                    |   2 +-
5 files changed, 2 insertions(+), 2 deletions(-)
```

## Recent Log

```text
b7271b5 docs: finalize iteration 012 auth blocker evidence
17eb19b chore: add full promotion continuation runner
45ad227 docs: finalize iteration 011 auth handoff status
b25ab65 chore: add sdk auth preflight and handoff tools
483a3cf docs: refresh iteration 010 final status
5d29772 docs: finalize sdk auth blocked evidence
0a9c7ff docs: refresh iteration 009 final head
0c62b37 docs: record iteration 009 sdk auth blocker
e7cc4f3 chore: point weights at battle phase diagnostics code
3d3e69a feat: add battle phase valid action diagnostics
1459339 docs: refresh iteration 008 final head
789ce7f docs: finalize iteration 008 blocked evidence
e23a98f fix: keep objective diagnostics out of match stats
1d7cdf3 fix: correct objective policy manifest sha
d3680ae chore: publish objective policy manifest
ecd962e feat: add objective labels and diagnostics
6a7e95d docs: refresh iteration 007 blocked closeout
2055da3 Document Iteration 007 blocked evidence
f4e0dcf Point published weights at diagnostics-capable agent code
ff4804f Expose self-play action type diagnostics
30d9f7b Retrain situation model with normalized training features
77849cd Keep situation normalization consistent across training and inference
850df7c docs: finalize Iteration 006 reports with model quality evidence
374f2be chore: update manifest SHA to current HEAD after smoke validation
5c3931f fix: use correct full SHA from git rev-parse for manifest
2d2d5fb chore: bump manifest to latest HEAD
1e16101 fix: correct source_commit_sha in model manifest
006bcfc chore: bump manifest source_commit_sha for selfplay diagnostics fix
ca2157c chore: bump manifest source_commit_sha to point to latest code
33c6b71 feat: add feature normalizer, tactical model, and training-compatible situation features
57708f6 docs: add iteration 005 report and update final blocked report
677f69d fix: rebalance city scoring, fix situation model feature normalization, add policy score tests
9ba20f3 chore: bump manifest to final report commit
10608cd docs: final blocked report with move routing fix evidence
49039ca fix: correct manifest source_commit_sha to real HEAD
44ac552 chore: bump manifest source_commit_sha to 1860b86
1860b86 fix: passthrough move route diagnostics and update manifest commit
c59846b docs: finalize iteration 003 report with root cause and fix summary
9e36d05 fix: add move route diagnostics, fallback strategy, and context-aware occupy
```

## Auth Preflight

- command: `python3.10 scripts/check_sdk_auth_environment.py --output experiments/shenji_agent/sdk_auth_environment_check_iteration_015.json`
- exit_code: `0`
- mac_match: `true`
- blocker_reason: `null`
- auth_environment_ready: `true`
- bound_mac: `02:42:ac:11:00:02`
- current_mac: `02:42:ac:11:00:02`
- engine_config_path: `/remote-home/lizilong/.local/lib/python3.10/site-packages/train_env/env/authenticate/.engine_config`
- latest_auth_status: `mac_match_unverified`
- latest_battle_phase_entered: `true`

The root executor initially resolved `~` to `/root`, but the SDK auth config present for this project is the user-home SDK path under `/remote-home/lizilong/.local/...`. The diagnostics were adjusted to prefer explicit `SDK_ENGINE_CONFIG`, then the current HOME SDK path, then the repository owner HOME SDK path when root is operating on the user-owned repository. No `.engine_config` contents were changed.
