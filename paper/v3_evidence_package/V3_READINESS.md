# V3 Readiness

**State:** `IN_PROGRESS`

## Claim classification updates (V3A — tactical ablation + intent checkpoint fix)

| Claim | V2 status | V3A status | After V3A-FIX | Change |
|-------|-----------|------------|---------------|--------|
| tactical contribution | forbidden | partially supported (action-level only) | partially supported (action-level only, outcome executable) | upgraded |
| route reliability | forbidden | forbidden (unchanged) | forbidden (unchanged) | — |
| two-window success | forbidden | forbidden (unchanged) | forbidden (unchanged) | — |
| baseline superiority | forbidden | forbidden (unchanged) | forbidden (unchanged) | — |
| method novelty | unsupported | unsupported (unchanged) | unsupported (unchanged) | — |
| reproducibility | partially repaired | same (unchanged) | same (unchanged) | — |
| public SOTA | forbidden | forbidden (unchanged) | forbidden (unchanged) | — |
| top-tier readiness | forbidden | forbidden (unchanged) | forbidden (unchanged) | — |

## Completed this round

### V3A (tactical ablation)
- ✅ Tactical on/off switch implemented in `LearnedPolicy`
- ✅ Action-level diagnostic: tactical produces nonzero variegated scores
- ✅ Smoke ablation: 3 seeds, SMOKE_PASS
- ✅ Multi-seed ablation: 5 seeds, consistent results
- ✅ Tactical contribution decision: ACTION_LEVEL_ONLY

### V3A-FIX (intent checkpoint compatibility)
- ✅ Checkpoint architecture audit completed
- ✅ Historical model definition search completed
- ✅ Strategy B (LegacyIntentModel + LegacySituationModel) implemented
- ✅ LegacyIntentModel loads old `fc_intent` checkpoint into `IntentionLSTM(num_classes=8)` wrapper
- ✅ Output adapted to `[batch, 1, 7]` for policy compatibility
- ✅ LegacySituationModel loads bare `SituationGCN` checkpoint
- ✅ Policy auto-detects old vs new checkpoint format
- ✅ 13 compatibility tests pass
- ✅ Full-agent compatibility smoke: PASS (9/9 phases)
- ✅ Outcome-level tactical ablation: EXECUTABLE (SDK replay needed)

### V3B (formal tactical outcome ablation)
- ✅ Formal replay runner built (`scripts/run_v3b_formal_tactical_ablation.py`)
- ✅ Paired evaluation protocol defined (3 scenarios × 5 seeds)
- ✅ Metrics pipeline with bootstrap CIs built
- ✅ 30/30 games completed, 0 errors
- ✅ Full transparency and documentation
- ⚠️ Outcome-level assessment INCONCLUSIVE — SDK authentication blocks full game replay
- ⚠️ MAC mismatch: `02:42:ac:11:00:07` (current) vs `02:42:ac:11:00:02` (required)

## Remaining blockers

1. **SDK authentication** — formal outcome-level evaluation requires authorized Docker environment with matching MAC.
2. **Route reliability still unproven** — not addressed in this round (blocked by same auth issue).
3. **V3C/V3D evaluation not started** — long-horizon and generalization evaluation not yet attempted.
