# V3G Manuscript Readiness Update

**Date:** 2026-05-27

## Route evidence update

V3G expands authenticated route coverage from 3 scenarios (V3D, 590 attempts) to 42/43 league scenarios (97.7%, 5,425 attempts):

- 39/42 scenarios (92.9%): 0% empty rate
- 2 map 94 scenarios: ~99% empty (genuine A* pathfinding failure)
- 1 scenario crashed (2120331196, map 212)
- 1 scenario excluded (3531, map 53 — timeout)

## Scenario 1910631192 contradiction resolved

The critical V3F contradiction (0% V3D vs 99% league) is resolved. Under current authenticated SDK replay, scenario 1910631192 shows 0% empty rate (matching V3D). The league's 99% was an auth artifact caused by stale `.engine_config` MAC binding in the venv SDK install.

## SDK diagnostic label revised

V3F attributed empty routes to "SDK candidate generation constraints" (`no_valid_move_candidate`). V3G finds this label is a **misnomer**: the route planner IS always called before the label fires, and the true failure is `A* route planner cannot find a path to any city with any move type, and fallback also fails`. The counts are correct (5,425 real move attempts), but the attribution to candidate generation is inaccurate.

## Implications for manuscript

Route reliability is no longer a blocking uncharacterized risk. The failure is localized to map 94 scenarios and is explainable as terrain/pathfinding constraints rather than SDK bugs. The allowed wording supports a conservative characterization of route behavior.

## Classification

```
READY_FOR_BASELINE_EXPANSION
```

Route evidence is sufficiently bounded. The next goal should be baseline/component ablation coverage and promotion evidence expansion.

## Remaining constraint

Tactical, promotion, SOTA, and top-tier claims remain unchanged from V3F:

- tactical outcome contribution: WEAK_OR_INCONCLUSIVE
- two-window promotion success: FORBIDDEN
- public SOTA: FORBIDDEN
- top-tier readiness: FORBIDDEN
