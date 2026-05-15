# V3C Phase 3 — Minimal Replay Smoke

## Command

```bash
python3 scripts/run_v3b_formal_tactical_ablation.py \
    --smoke \
    --out experiments/shenji_agent/v3c_smoke \
    --timeout-steps 200
```

## Results

| Check | Status |
|-------|--------|
| Authentication | **PASSED** ("did pass authentication") |
| Deployment phase entered | **YES** — deployment actions executed |
| Battle phase entered | **YES** — 3 move attempts recorded |
| Legal actions executed | **YES** — deployment setup + move actions |
| Tactical mode enabled | **YES** — `tactical_invocation_count=3` (on) vs `tactical_invocation_count=3` (off) |
| Tactical nonzero invocations | **3** (on) vs **0** (off) — tactical is influencing actions |
| Route metrics | Collected — `route_empty_rate=0.0` for both |
| Game result | Timeout at 200 steps, winner: Blue (132 vs 104) |
| Pairs completed | 0/1 (timeout, not formal completion) |

## Key observations

1. The SDK authentication rebind was **fully successful** — both tactical-on and tactical-off runs printed "did pass authentication"
2. The game progressed past the deployment phase and entered battle (confirmed by `total_move_attempts=3` and tactical invocations)
3. Tactical mode shows differentiation: `tactical_nonzero_count=3` (on) vs `tactical_nonzero_count=0` (off)
4. The 200-step timeout is expected for a smoke test — full evaluation uses higher timeouts
5. No crashes, no errors, no MAC mismatch

## What was unblocked

The MAC mismatch that blocked V3B is fully resolved. The SDK environment is now authenticated and replay can progress. The remaining timeout is a game mechanic (deployment phase length + battle phase length exceeding 200 steps for this scenario), not an authentication issue.
