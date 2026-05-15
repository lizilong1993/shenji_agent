# V3E Validation Log

Date: 2026-05-15

## Commands and Results

### V3E claim boundary tests

```
python3 -m pytest tests/test_v3e_claim_boundaries.py -v
```

Exit code: 0

Results: 21 passed, 0 failed, 1 warning (permission-related pytest cache only)

Key tests:
- Tactical claim gate: not supported ✅
- Forbidden claims (SOTA, two-window, top-tier): correctly forbidden ✅
- Promotion: no two-window claim, single window only ✅
- Route: not claimed as solved, not marked as supported ✅
- Cross-file consistency: aligned across all evidence package files ✅
- Baseline: not overstated ✅

### Compileall

```
python3 -m compileall paper/scripts scripts ai tests training
```

Exit code: 0

All modules compiled without errors.

## Warnings

- pytest cache write permission denied (non-issue, cache missing).
- No other warnings or errors.
