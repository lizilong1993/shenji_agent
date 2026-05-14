# Phase 1 — Tactical Contribution Criteria

Three levels of evidence are defined.

## Level 1 — Compatibility only

**Supported if:**
- checkpoint loads
- dataset validates
- tactical branch produces nonzero scores
- compatibility smoke passes

**Status:** Already achieved (commit `7ee858b`).

**Allowed claim:**
> The tactical branch can be loaded and produces nonzero scores under compatibility diagnostics.

**Forbidden claim:**
> The tactical branch improves agent performance.

## Level 2 — Action-level contribution

**Supported only if** tactical-on vs tactical-off replay/evaluation shows:
- `tactical_reached_count > 0`
- `tactical_nonzero_score_count > 0`
- `tactical_changed_action_count > 0`
- `changed_action_rate` is measured
- changed actions are legal
- route empty-rate does not worsen substantially

**Allowed claim:**
> The tactical branch measurably changes selected actions in a bounded diagnostic setting.

## Level 3 — Outcome-level contribution

**Supported only if** tactical-on vs tactical-off evaluation shows:
- win rate improves, or
- rating improves, or
- promotion metric improves, or
- diagnostic objective improves

and the effect is shown across multiple seeds or with a confidence interval.

**Allowed claim:**
> The tactical branch improves the evaluated metric under the tested protocol.

Only Level 3 can support strong tactical contribution in a paper.
