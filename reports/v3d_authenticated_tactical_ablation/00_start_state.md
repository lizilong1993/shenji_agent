# V3D Authenticated Tactical Outcome Ablation — Start State

Date: 2026-05-15

## Record

| Field | Value |
|---|---|
| Current branch | `main` |
| HEAD | `729b1d0` |
| V3C commit | `729b1d0` — "diagnose sdk auth replay blocker and recover mac binding for v3c" |
| Worktree status | Clean (only __pycache__ and untracked goal files) |
| Auth/license-sensitive files committed | None detected in V3C commit |
| SDK auth state | `SDK_REPLAY_AUTH_RECOVERED` (per V3C) |
| Current MAC | `02:42:ac:11:00:07` |
| V3B prior blocker | SDK MAC mismatch — `.engine_config` bound to `02:42:ac:11:00:02`, current MAC was `02:42:ac:11:00:07`. Replay stuck in deployment phase. |
| V3C smoke result | Authentication passed, deployment+battle phases entered, tactical invocations: on=3, off=0. 87/87 tests passed. |

## Existing backup branches

- `backup/before_v3c_sdk_auth_recovery_20260515_130809` (pre-V3C)
- `backup/before_v3b_formal_tactical_ablation_20260515_1115` (pre-V3B formal)
- `backup/before_v3a_intent_checkpoint_fix_20260514_184612`
- `backup/before_v3a_tactical_ablation_20260514_151649`
- `backup/before_selective_tactical_asset_restore_20260514_134407`

## Infrastructure available

- `scripts/run_v3b_formal_tactical_ablation.py`
- `scripts/v3b_pairing.py`
- `scripts/v3b_metrics.py`

## V3D objective

Rerun formal tactical-on/off paired ablation under authenticated SDK replay.
Do not train. Do not hand-edit archived artifacts. Do not claim unsupported outcome contribution.
