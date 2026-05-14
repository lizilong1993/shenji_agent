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

## Remaining blockers

1. **Outcome-level contribution unproven** — tactical_on/off replay loop requires external wargame SDK not available in this environment.
2. **Route reliability still unproven** — not addressed in this round.
3. **Full formal V3B/V3C/V3D evaluation not started** — paired tactical-on/off games with outcome metrics needed.
