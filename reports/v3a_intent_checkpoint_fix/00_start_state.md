# Phase 0 — Start State

## Timestamp
2026-05-14T18:32 (approximate)

## Branch
main

## HEAD
77f2c26 — "evaluate tactical contribution with on-off ablation"

## V3A Commit Exists
Yes — commit 77f2c26 contains V3A tactical contribution ablation deliverables.

## Worktree Status
Dirty files (all unrelated ignored assets):
- `ai/__pycache__/*.cpython-310.pyc` (4 files, staged modify)
- `selective_tactical_restore/` (untracked directory)

These do not block the intent checkpoint fix workflow.

## Local Tactical Assets Availability
| Asset | Path | Exists | Notes |
|-------|------|--------|-------|
| Intent checkpoint | `ai/weights/intention_lstm.pth` | Yes (13.5 MB, root-owned) | Restored from backup |
| Situation GCN | `ai/weights/situation_gcn.pth` | Yes (125 KB, root-owned) | Restored from backup |
| Tactical model | `ai/weights/tactical_model.pt` | Yes (53 KB, lizilong-owned) | Restored from backup |
| Model manifest | `ai/weights/model_manifest.json` | Yes (root-owned) | Lists all managed files |

## Current Intent Checkpoint Path
`ai/weights/intention_lstm.pth`

## Current Model Manifest
- `intent_lstm`: `intention_lstm.pth`
- `situation_gcn`: `situation_gcn.pth`
- `tactical_model`: `tactical_model.pt`
- `model_manifest.json` generated at `20260428_235813`
- Source commit: `3d3e69a5aa688cf4eeff948db60d507d90bee86c`

## Current Blocker Statement
The restored `ai/weights/intention_lstm.pth` uses an old `fc_intent` architecture, while the current `MultiHorizonIntentionModel` expects a `head` module. This prevents full-agent replay and outcome-level tactical-on vs tactical-off ablation.
