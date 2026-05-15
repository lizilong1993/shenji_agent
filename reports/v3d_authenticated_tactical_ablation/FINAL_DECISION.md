# V3D Final Decision

## Decision

```
RESUME_V3_WITH_TACTICAL_DIAGNOSTIC_OR_WEAK_CLAIM
```

## Rationale

Outcome evidence is weak/inconclusive but tactical invocation and action-level effect remain valid.

### Evidence summary

| Requirement | Result |
|---|---|
| Authenticated replay preflight runs | ✅ |
| Formal paired ablation attempted | ✅ (15 pairs) |
| Tactical invocation metrics recorded | ✅ (ON nonzero=351, OFF nonzero=0) |
| Outcome deltas computed | ✅ (mean=-1.27, CI [-50.07, 34.82]) |
| Route metrics recorded | ✅ (0% empty rate all runs) |
| Claim gate decision written | ✅ (WEAK_OR_INCONCLUSIVE) |
| V3 evidence package updated | ✅ (conservative wording) |
| Tests pass, compileall clean | ✅ (87/87) |
| No unsupported tactical claim made | ✅ |

### Permitted wording

> Authenticated paired tactical-on/off evaluation executes successfully and invokes the tactical branch, but outcome-level contribution remains weak or inconclusive under the tested protocol.

### Forbidden wording

- Tactical improves win rate generally
- Tactical is a main performance contribution
- The method is promotion-ready
- Two-window success is achieved
- Route reliability is solved

## Previous decisions

| Round | Decision |
|---|---|
| V3B | RESUME_V3_WITH_TACTICAL_DIAGNOSTIC_OR_WEAK_CLAIM (auth-blocked) |
| V3C | RESUME_V3_AUTH_RECOVERED |
| V3D | RESUME_V3_WITH_TACTICAL_DIAGNOSTIC_OR_WEAK_CLAIM (authenticated) |
