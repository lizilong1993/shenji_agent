# Reproducibility Contract

Date: 2026-05-13

## Scope

This contract covers repo-local checkpoint and asset resolution for V2 evidence generation.
It does not claim that the external SDK authorization environment is independently portable.

## Canonical Candidate Assets

Canonical candidate artifact directory:

```text
ai/weights
```

The candidate manifest is the source of truth:

```text
ai/weights/model_manifest.json
```

Current canonical entries:

| Manifest key | Required file | Status |
|---|---|---|
| `intent_lstm` | `ai/weights/intention_lstm.pth` | exists |
| `situation_gcn` | `ai/weights/situation_gcn.pth` | exists |
| `situation_feature_normalizer` | `ai/weights/situation_feature_normalizer.json` | exists |
| `tactical_model` | `ai/weights/tactical_model.pt` | missing, fail-closed |

The manifest now matches the restored intent/situation checkpoint filenames. Legacy aliases are
still supported by the resolver so validators can report naming mismatches instead of false
missing-asset failures for older manifests.

## Incumbent Rule Assets

Canonical incumbent artifact directory:

```text
ai/weights_incumbent_rule
```

The incumbent manifest has:

```text
snapshot_role = bootstrap_rule_baseline
```

For this role, learned model checkpoint files are not required. This prevents candidate-only or
rule-incumbent smoke checks from falsely failing on absent incumbent neural checkpoints.

## Resolver Behavior

Shared resolver:

```text
ai/artifacts.py
```

Consumers updated:

```text
ai/policy.py
training/selfplay.py
scripts/run_round005_smoke_pipeline.py
```

Rules:

- Loaders and validators resolve model files from the manifest.
- Restored legacy aliases are accepted only when the manifest names older `.pt` files and the
  corresponding `.pth` file exists.
- Alias resolution is reported as a naming mismatch.
- Missing manifest-declared tactical checkpoint remains a fail-closed blocker.
- Rule incumbent manifests with `snapshot_role=bootstrap_rule_baseline` do not require learned
  checkpoint files.

## Verification

Targeted test command:

```bash
python -m unittest tests.test_v2_asset_contract -v
```

Result:

```text
Ran 6 tests in 0.606s
OK
```

Preflight command:

```bash
python scripts/run_round005_smoke_pipeline.py --out reports/v2_top_tier_upgrade/smoke_preflight_v2
```

Result:

```text
exit code 2
```

The exit code is expected because the candidate manifest still declares a tactical checkpoint
that is absent. The generated preflight sidecar reports no intent/situation naming mismatch and
no incumbent-rule asset blocker.

Output:

```text
reports/v2_top_tier_upgrade/smoke_preflight_v2/smoke_preflight.json
reports/v2_top_tier_upgrade/smoke_preflight_v2/BLOCKED.md
```

## Remaining Reproducibility Blocker

The tactical checkpoint is missing:

```text
ai/weights/tactical_model.pt
```

Until Phase 3 either recovers/trains this checkpoint or demotes tactical from the contribution,
tactical contribution claims remain forbidden and full candidate smoke remains fail-closed.
