# V3 Readiness

**State:** `IN_PROGRESS`

## Claim classification updates (V3A — tactical ablation)

| Claim | V2 status | V3A status | Change |
|-------|-----------|------------|--------|
| tactical contribution | forbidden | partially supported (action-level only) | upgraded |
| route reliability | forbidden | forbidden (unchanged) | — |
| two-window success | forbidden | forbidden (unchanged) | — |
| baseline superiority | forbidden | forbidden (unchanged) | — |
| method novelty | unsupported | unsupported (unchanged) | — |
| reproducibility | partially repaired | same (unchanged) | — |
| public SOTA | forbidden | forbidden (unchanged) | — |
| top-tier readiness | forbidden | forbidden (unchanged) | — |

## Completed this round

- ✅ Tactical on/off switch implemented in `LearnedPolicy`
- ✅ Action-level diagnostic: tactical produces nonzero variegated scores
- ✅ Smoke ablation: 3 seeds, SMOKE_PASS
- ✅ Multi-seed ablation: 5 seeds, consistent results
- ✅ Tactical contribution decision: ACTION_LEVEL_ONLY

## Remaining blockers

1. **Intent/situation model weight mismatch** — `intention_lstm.pth` uses old `fc_intent` architecture, current `MultiHorizonIntentionModel` expects `head` module. Blocks full-policy loading and game-level evaluation.
2. **Outcome-level contribution unproven** — without full-agent replay, win rate/rating impact cannot be measured.
3. **Route reliability still unproven** — not addressed in this round.
