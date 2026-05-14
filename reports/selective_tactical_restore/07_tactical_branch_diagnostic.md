# Tactical Branch Diagnostic

Command:

```bash
/remote-home/lizilong/venvs/shenji/bin/python scripts/run_tactical_branch_diagnostic.py --out reports/selective_tactical_restore/tactical_branch_diagnostic.json
```

Result: `TACTICAL_BRANCH_LOADS_AND_PRODUCES_NONZERO_SCORES`

Evidence:

- Tactical checkpoint resolved: true.
- Tactical checkpoint loaded: true.
- Tactical model class: `ai.models.TacticalScorer`.
- Tactical feature builder available: true.
- Tactical candidate count: 16.
- Tactical reached count: 16.
- Tactical nonzero score count: 16.
- Score sample contains nonzero logits.

This is executable compatibility evidence only. It is not a tactical contribution claim and does not replace tactical-on vs tactical-off ablation.
