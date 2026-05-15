# V3B Phase 6 — Route Reliability Audit

**Date:** 2026-05-15

## Result

**ROUTE_RELIABILITY_AUDIT_INCONCLUSIVE** — Route metrics cannot be measured because the game never progresses past deployment stage.

## Measurement attempt

| Metric | Result |
|--------|--------|
| Route empty rate (tactical on) | 0.0 (no move routes attempted) |
| Route empty rate (tactical off) | 0.0 (no move routes attempted) |
| Total move attempts | 0 (all 30 games) |
| Route failure reasons | Not available |
| Fallback move frequency | Not available |
| Learned move success rate | Not available |

## Why route metrics are unavailable

The SDK authentication blocks full game replay. The `.engine_config` binds to MAC `02:42:ac:11:00:02`, while the current environment has MAC `02:42:ac:11:00:07`. Without authentication, the environment stays in deployment stage (stage=1) for all 1800 steps, then ends naturally at `max_step=1800`.

During deployment stage, the agent only generates deployment actions (type 303: equip/get-on, type 333: ready). No Move or Shoot actions are generated, so no route planning occurs. The policy's `move_route_samples` diagnostics remain empty.

## Comparison with prior V3A route-empty metric

Prior V3A reported route-empty rate ≈ 15641/23580 ≈ 0.6633. This was measured through the policy's diagnostic pipeline during actual gameplay in Docker-authorized runs. The current environment cannot replicate these measurements without SDK authorization.

## Answers to explicit questions

| Question | Answer |
|----------|--------|
| Does tactical reduce route failures? | Cannot be assessed — no route failures occur (no routes attempted) |
| Does tactical merely alter actions? | Cannot be assessed — no tactical actions executed in deployment stage |
| Are route failures still dominant? | Cannot be assessed — no gameplay occurred |

## Path forward

Route reliability audit requires either:
1. Running the evaluation in an authorized Docker environment with matching MAC
2. Using the offline policy diagnostic pipeline (V3A approach) with real game observations from prior runs
