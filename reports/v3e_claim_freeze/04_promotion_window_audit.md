# V3E Promotion and Two-Window Audit

## Latest Sidecar Summary

Inspected: `experiments/shenji_agent/league_20260501_170204/`

| Sidecar | Status |
|---|---|
| `participants.json` | ✅ Candidate (`ai/weights`, challenger) vs Incumbent (`ai/weights_incumbent_rule`, baseline) — **distinct** |
| `rating.json` | ✅ Overall pe=0.87, ci95=[0.80, 0.93]; Complex pe=0.82, ci95=[0.64, 0.95] |
| `promotion.json` | ✅ **accepted=true**, elo_delta_lower_bound=240.8, max_exploiter_win_rate=0.25, learned_path_coverage=0.37, move_path_coverage=0.34, shoot_path_coverage=1.0 |
| `league_state.json` | consecutive_windows=0, champion=`ai/weights`, 8 historical_mains all `ai/weights` |
| `exploitability.json` | aggressive_exploiter wr=0.0, occupy_exploiter wr=0.25 (both ≤0.45) |
| `summary.md` | 100 paired maps, battle_phase_entered=true, candidate_fallback_reasons: `['no_valid_move_candidate']` |

## Q&A

**Q1: How many promotion windows are formally verified?**
One. The latest league (league_20260501_170204) has promotion accepted=true. Many earlier May 2026 leagues also have accepted=true. However, `consecutive_windows=0` in the latest.

**Q2: Is there a two-window hard-stop success?**
No. The program.md requires two consecutive windows. The latest league has consecutive_windows=0. There is evidence of consecutive_windows=1 in one earlier league (league_20260501_085914), but the latest evaluation resets to 0 — possibly due to different configuration or evaluation series. Two-window hard-stop success is **not proven**.

**Q3: Are candidate/incumbent participants distinct?**
Yes. Candidate: `ai/weights` with challenger profile (intent_lstm, situation_gcn, tactical_model). Incumbent: `ai/weights_incumbent_rule` with baseline profile (rule-based, no managed model files).

**Q4: Are rating/promotion sidecars complete?**
Yes. All required sidecars exist: participants.json, rating.json, promotion.json, league_state.json, exploitability.json, match_results.json, summary.md, map_list.json.

**Q5: Are latest sidecars consistent with paper claims?**
The prewrite evidence package correctly states "One verified promotion window exists" (PAPER_WRITING_READY.md). The claim matrix correctly does not claim two-window success.

**Q6: Is any manuscript/report prose overstating promotion status?**
No. Current evidence package (PAPER_WRITING_READY.md, claim_to_evidence_matrix_v3.md) correctly limits promotion claims to single-window only.

## Classification

```
PROMOTION_SINGLE_WINDOW_ONLY
```

Justification: Many single-window promotions exist with accepted=true across multiple May 2026 league runs. The candidate convincingly beats the rule incumbent (rating 0.87, elo_delta_lb 240.8, exploiter wr ≤0.25). However, two consecutive windows are not formally verified in the latest evaluation series. The incumbent was never updated to be the previous candidate (champion always `ai/weights`).

## Forbidden Claims

- Two-window hard-stop success.
- Promotion criteria fully passed.
- Top-tier readiness.
- Method is promotion-ready for publication.

## Allowed Wording

> A single formal promotion window is verified: the learned candidate beats the rule incumbent with rating 0.87 (95% CI [0.80, 0.93]) and passes all promotion gate criteria. Two-window success has not yet been demonstrated.
