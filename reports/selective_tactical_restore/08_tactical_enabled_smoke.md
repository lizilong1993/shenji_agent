# Tactical Enabled Smoke

Command:

```bash
/remote-home/lizilong/venvs/shenji/bin/python scripts/run_tactical_enabled_smoke.py --out-dir experiments/shenji_agent/tactical_restore_smoke --report reports/selective_tactical_restore/tactical_enabled_smoke.json
```

Result: `TACTICAL_ENABLED_COMPATIBILITY_SMOKE_PASS`

Metrics:

- Episodes attempted: 1.
- Episodes completed: 1.
- Games attempted: 0.
- Games completed: 0.
- Tactical loaded: true.
- Tactical reached count: 16.
- Tactical nonzero score count: 16.
- Tactical changed action count: not available in this compatibility smoke.
- Route empty count: 0.
- Route empty rate: 0.0.
- Errors: none.

This smoke is explicitly marked `not_formal_evidence=true`; it only confirms restored asset compatibility.
