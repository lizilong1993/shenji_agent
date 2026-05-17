
# V3C SDK Authentication / Replay Environment Recovery

## Objective

V3B completed the formal tactical ablation infrastructure but outcome-level evidence remains weak or inconclusive because the SDK authentication MAC binding blocks normal replay.

Known blocker:

- SDK requires MAC: `02:42:ac:11:00:02`
- Current MAC: `02:42:ac:11:00:07`
- Replay remains in deployment phase
- Tactical on/off score delta is 0.0 because battle/evaluation does not progress

This round must diagnose whether the SDK environment can be restored to an authenticated, battle-progressing state.

Do not train.
Do not modify tactical claims.
Do not edit archived league sidecars.
Do not claim tactical outcome contribution.
Do not run long multi-seed ablations.
Only diagnose SDK authentication and run minimal replay smoke if unblocked.

## Phase 1 — Snapshot

Run:

```bash
pwd
git status --short
git log --oneline --decorate -n 20
ip link
python --version
````

Create:

```text
reports/v3c_sdk_auth_recovery/00_start_state.md
```

Record:

* current branch;
* HEAD;
* V3B commit `11895c9`;
* SDK path;
* current MAC addresses;
* required MAC from V3B logs;
* current replay blocker.

Create backup branch:

```bash
git branch backup/before_v3c_sdk_auth_recovery_$(date +%Y%m%d_%H%M%S)
```

## Phase 2 — Locate SDK auth source and config

Search:

```bash
grep -RInE 'MAC|mac|license|auth|认证|02:42|deploy|deployment|battle' \
  land_wargame_sdk ai training scripts tests reports 2>/dev/null \
  > reports/v3c_sdk_auth_recovery/sdk_auth_grep.txt || true
```

Inspect:

* SDK wheel location;
* SDK config files;
* environment variables;
* license files;
* any hardcoded MAC requirement;
* V3B logs showing MAC mismatch.

Create:

```text
reports/v3c_sdk_auth_recovery/01_sdk_auth_audit.md
reports/v3c_sdk_auth_recovery/sdk_auth_audit.json
```

Classify blocker as one of:

```text
SDK_MAC_BINDING_CONFIRMED
SDK_LICENSE_FILE_MISSING
SDK_ENV_CONFIG_MISSING
SDK_RUNTIME_DEPENDENCY_MISSING
SDK_AUTH_BLOCKER_UNKNOWN
```

## Phase 3 — Check whether MAC can be matched safely

Do not make destructive network changes.

Audit available interfaces:

```bash
ip link
cat /sys/class/net/*/address 2>/dev/null || true
```

Check whether the required MAC can be provided by:

* running in the original container/network;
* using Docker `--mac-address`;
* using a specific network interface;
* setting SDK-supported env/config;
* obtaining a new SDK license for current MAC.

Create:

```text
reports/v3c_sdk_auth_recovery/02_mac_recovery_options.md
```

Do not spoof MAC unless the environment owner explicitly permits it.

## Phase 4 — Minimal SDK replay smoke

If authentication can be restored safely, run only a tiny smoke:

```bash
python scripts/run_v3b_formal_tactical_ablation.py \
  --seeds 0 \
  --pairs 1 \
  --out reports/v3c_sdk_auth_recovery/minimal_replay_smoke
```

or the smallest equivalent command available.

Required output:

```text
reports/v3c_sdk_auth_recovery/minimal_replay_smoke/
reports/v3c_sdk_auth_recovery/03_replay_smoke.md
reports/v3c_sdk_auth_recovery/replay_smoke.json
```

The smoke must report:

* authenticated or not;
* deployment phase entered;
* battle phase entered;
* at least one legal action executed if possible;
* tactical mode status;
* route metrics if available;
* exact blocker if still blocked.

Do not interpret smoke as outcome contribution.

## Phase 5 — Validate V3B infrastructure remains intact

Run:

```bash
python -m unittest discover -v
python -m compileall paper/scripts scripts ai tests training
```

Create:

```text
reports/v3c_sdk_auth_recovery/05_validation_log.md
```

## Phase 6 — Decision

Create:

```text
reports/v3c_sdk_auth_recovery/FINAL_DECISION.md
reports/v3c_sdk_auth_recovery/sdk_auth_recovery_decision.json
```

Use exactly one final state:

```text
SDK_REPLAY_AUTH_RECOVERED
```

Use if SDK authentication is restored and minimal replay progresses beyond deployment.

```text
SDK_REPLAY_AUTH_BLOCKED_MAC_BINDING
```

Use if MAC binding is confirmed and cannot be changed safely in this environment.

```text
SDK_REPLAY_AUTH_BLOCKED_LICENSE_OR_CONFIG
```

Use if a missing license/config is the blocker.

```text
SDK_REPLAY_AUTH_BLOCKER_UNKNOWN
```

Use if the blocker cannot be isolated.

```text
V3C_SAFE_STOP
```

Use if repo/runtime state prevents safe diagnosis.

## Phase 7 — Evidence package update

Update only claim-boundary evidence files:

```text
paper/v3_evidence_package/V3_READINESS.md
paper/v3_evidence_package/TACTICAL_CONTRIBUTION_STATUS.md
paper/source_evidence.md
```

Allowed wording:

```text
V3B formal tactical ablation infrastructure exists, but outcome-level tactical contribution remains weak or inconclusive because SDK authentication prevents replay from progressing beyond deployment in the current environment.
```

If SDK replay is recovered:

```text
SDK replay authentication was restored at smoke level. Formal outcome ablation must be rerun before any tactical outcome contribution claim.
```

Forbidden:

```text
Tactical improves outcome performance.
V3B proves tactical contribution.
Replay recovery implies promotion readiness.
```

## Phase 8 — Commit

Stage reports and small code/config changes only:

```bash
git add reports/v3c_sdk_auth_recovery
git add paper/v3_evidence_package paper/source_evidence.md 2>/dev/null || true
git add scripts tests 2>/dev/null || true
git commit -m "diagnose sdk authentication blocker for replay"
```

Do not commit large binaries.

## SUCCESS criteria

SUCCESS if:

* SDK auth blocker is audited;
* MAC/license/config cause is classified;
* minimal replay smoke is attempted if safe;
* tests and compileall run;
* final decision is written;
* tactical outcome claims remain conservative.

## SAFE_STOP criteria

SAFE_STOP if:

* SDK license/auth files are unavailable;
* MAC binding cannot be resolved safely;
* replay cannot progress beyond deployment;
* environment owner action is required.

In SAFE_STOP, state exact human action needed, such as:

* run on container with MAC `02:42:ac:11:00:02`;
* obtain SDK license for current MAC;
* provide original SDK-authenticated container;
* provide SDK vendor/auth configuration.

