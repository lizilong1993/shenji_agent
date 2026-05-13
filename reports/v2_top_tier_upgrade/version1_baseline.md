# Version 1 Baseline

## Repository Snapshot

| Field | Value |
|---|---|
| Branch | `main` |
| HEAD | `586ffbc0591e9ee32fb1dd89e99ea5a0d45bbbca` |
| Version 1 commit | `586ffbc version 1 completed` |
| Backup branch | `backup/before_v2_top_tier_upgrade_20260513_072051` |
| Baseline state | `PARTIAL_READY_WITH_BLOCKERS` |

## Version 1 Evidence Package

Version 1 evidence is recorded under `paper/prewrite_package/` and summarized by
`paper/prewrite_package/PAPER_WRITING_READY.md`.

Safe Version 1 claims:

- Closed-loop land-wargame SDK evaluation was implemented.
- Candidate agent uses learned intent/situation components.
- Candidate was evaluated against a rule incumbent.
- Promotion gate or league-style evaluation exists.
- One verified promotion window exists.
- Route empty-rate limitation exists.

## Current Blockers

- Tactical contribution is not proven.
- Route reliability is not proven.
- Two-window hard-stop success is not proven.
- Baseline and ablation evidence is insufficient for strong claims.
- Novelty is weak if framed as LSTM plus GraphSAGE.
- Reproducibility is fragile because checkpoint and validator expectations may be inconsistent.
- Evaluation scope may be too narrow for top-tier claims.

## Forbidden Claims at V2 Start

- Tactical module improves performance.
- Route planner is reliable or route issue is resolved.
- Two-window hard-stop success has been achieved.
- Public SOTA is achieved.
- The manuscript or evidence package is top-tier submission ready.

## Required V2 Rule

These claim restrictions may only be lifted by new executable evidence generated after this
baseline. If the evidence cannot be produced, the V2 outcome must remain claim-bounded and
end in SAFE_STOP or a lower readiness classification.
