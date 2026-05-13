# Tactical Branch Audit

Date: 2026-05-13

## Decision

Use Path B: tactical removal / demotion from contribution claims.

The tactical branch cannot be repaired with the current repo state because:

- No tactical checkpoint exists under `ai/weights/`.
- No alternate `tactical_model.pt`, `tactical_model.pth`, or similarly named tactical checkpoint
  was found in the workspace.
- No tactical training CSV (`Wargame_Feature_Dataset.csv`) was found in the workspace.
- The candidate manifest still declares `tactical_model.pt`, so validators correctly fail closed.

## Required Diagnostic Fields

| Field | Status |
|---|---|
| checkpoint existence | missing: `ai/weights/tactical_model.pt` |
| manifest entry | present: `tactical_model = tactical_model.pt` |
| model class | exists: `ai.models.TacticalScorer` |
| loader path | exists in `ai/policy.py`, now manifest-resolver driven |
| feature construction | exists: `ai.features.action_feature_vector` and policy tactical scoring path |
| tensor shapes | tactical scorer expects input dimension `32` |
| device placement | policy loader moves loaded tactical model to resolved inference device |
| candidate eligibility | ineligible for tactical claims because checkpoint is absent |
| score aggregation | code path adds sigmoid-scaled tactical contribution when model exists |
| whether branch is reached | not reached in current candidate because `tactical_model` is not loaded |
| whether tactical score is overwritten | not applicable without loaded model |
| nonzero score count | `0` |
| effect on action choice | unmeasured and currently unmeasurable |
| effect on win/loss metrics | unmeasured and currently unmeasurable |

## Executable Evidence

Command:

```bash
python scripts/round005_tactical_diagnostic.py --out reports/v2_top_tier_upgrade/tactical_diagnostics/tactical_diagnostic.json
```

Result:

```text
exit code 2
tactical_nonzero_count=0
```

Output:

```text
reports/v2_top_tier_upgrade/tactical_diagnostics/tactical_diagnostic.json
```

Relevant fields:

```text
tactical_model_exists = false
tactical_model_loaded = false
tactical_nonzero_count = 0
blocker = missing tactical model file: ai/weights/tactical_model.pt
```

Additional search:

```bash
find . -type f \( -name '*tactical*.pt' -o -name '*tactical*.pth' -o -name 'tactical_model.pt' \)
find . -type f \( -name 'train.json' -o -name 'label.csv' -o -name 'Wargame_Feature_Dataset.csv' \)
```

Both searches returned no repair asset or training dataset path.

## Claim Boundary

Forbidden:

- The tactical model improves action selection.
- Tactical scores contributed to the reported policy behavior.
- Tactical-on versus tactical-off ablation supports the method.

Allowed:

- Tactical code exists as an inactive/future-work branch.
- Current candidate uses intent/situation/objective/fallback logic without proven tactical
  contribution.

## Top-Tier Impact

Tactical cannot be part of the V2 main contribution unless a compatible checkpoint and
protocol-matched ablation evidence are later generated. V2 novelty must therefore come from
another source, such as evidence-gated policy composition, route-aware action selection, or a
diagnostic promotion protocol. If no alternate novelty path is proven, top-tier readiness must
remain blocked.
