# V3C Phase 0 — Start State

**Date:** 2026-05-15

## Current state

| Property | Value |
|----------|-------|
| Repo path | `/remote-home/lizilong/shenji_agent` |
| Branch | `main` |
| HEAD | `11895c9` (`formalize tactical outcome ablation evaluation`) |
| V3B commit | `11895c9` |
| SDK path | `land_wargame_sdk/land_wargame_train_env-4.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` |
| SDK installed | Yes (`land_wargame_train_env 4.1.0` at `~/.local/lib/python3.10/site-packages/train_env/`) |
| Python | `/usr/bin/python3` (3.10.12) |
| Docker | Not available on this server |

## MAC addresses

| Interface | Current MAC |
|-----------|-------------|
| lo | `00:00:00:00:00:00` |
| eth0 | `02:42:ac:11:00:07` |

## Engine config (SDK auth)

| Property | Value |
|----------|-------|
| Path | `~/.local/lib/python3.10/site-packages/train_env/env/authenticate/.engine_config` |
| Exists | Yes |
| Bound MAC | `02:42:ac:11:00:02` |
| Timestamp | `1777517431836071613` (base64 decoded) |
| Decode success | Yes |

## Current replay blocker

| Check | Status |
|-------|--------|
| SDK installed | OK |
| Engine config present | OK |
| MAC match | **FAIL** — required `02:42:ac:11:00:02`, current `02:42:ac:11:00:07` |
| Auth environment ready | `false` |
| Blocker reason | `mac_mismatch` |
| Docker available | No |
| Battle phase can be entered | No without MAC fix |

## V3B outcome

V3B completed formal tactical ablation infrastructure and ran 30 games (15 pairs × 3 scenarios × 5 seeds, 0 errors), but outcome delta was 0.0 because the game environment never progressed past deployment stage due to MAC mismatch.

## Working tree

```
 D __pycache__/mock_train_env.cpython-310.pyc
 D ai/__pycache__/__init__.cpython-310.pyc
 D ai/__pycache__/agent.cpython-310.pyc
 D ai/__pycache__/models.cpython-310.pyc
 D ai/__pycache__/trt_utils.cpython-310.pyc
?? goals/V3B_Formal_Tactical_Outcome_Ablation_via_SDK_Replay.goal.md
?? goals/codex_goal_v3c_sdk_auth_replay_recovery.md
?? selective_tactical_restore/
```

## Backup branches

Existing:
- `backup/before_v3c_sdk_auth_recovery_20260515_1308` (this round)
- `backup/before_v3b_formal_tactical_ablation_20260515_1115`
- `backup/before_v3a_intent_checkpoint_fix_20260514_184612`
- `backup/before_v3a_tactical_ablation_20260514_151649`
- `backup/before_selective_tactical_asset_restore_20260514_134407`
- plus earlier branches

## Auth check summary

```
auth_environment_ready: false
blocker_reason: mac_mismatch
current_macs: [{"interface": "eth0", "mac": "02:42:ac:11:00:07"}]
engine_config.bound_mac: "02:42:ac:11:00:02"
engine_config.exists: true
mac_match: false
```

Recommendation from diagnostic:
> Run in an authorized environment whose non-loopback MAC is `02:42:ac:11:00:02`, or request a regenerated `.engine_config` from the SDK provider for the actual runtime MAC.
