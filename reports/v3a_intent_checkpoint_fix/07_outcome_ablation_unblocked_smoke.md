# Phase 7 — Outcome Ablation Unblocked Smoke

Timestamp: 2026-05-14
Overall: **PASS**

## Results

### tactical_on
- Status: **PASS**
- intent_loaded: True
- intent_architecture: LegacyIntentModel
- situation_loaded: True
- tactical_loaded: True
- tactical_mode: on

### tactical_off
- Status: **PASS**
- intent_loaded: True
- intent_architecture: LegacyIntentModel
- situation_loaded: True
- tactical_loaded: True
- tactical_mode: off

### intent_inference_tactical_on
- Status: **PASS**
- top_direction: 0
- confidence: 0.2687
- distribution_length: 7

### intent_inference_tactical_off
- Status: **PASS**
- top_direction: 0
- confidence: 0.2125
- distribution_length: 7

### tactical_score_difference
- Status: **PASS**
- tactical_on_score: -0.031009
- tactical_off_score: 0.0


## Assessment

### Outcome ablation is now: **EXECUTABLE**

Both tactical_mode='on' and tactical_mode='off' configure successfully.
All three models load. Intent inference works with 7-class output.

### What is still needed:
- SDK-backed replay loop to run paired tactical on/off episodes
- Same seed, same scenario, same opponent
- Outcome metrics: win/loss, objective scores, route empty rates
- Multiple seeds for statistical confidence

### What has been unblocked:
- Policy loading no longer crashes on old checkpoints
- Both tactical modes are functional
- Action-level tactical contribution is already proven (V3A)

### Forbidden conclusion
Tactical does NOT improve outcome performance from this smoke.
Formal V3B/V3C/V3D evaluation is needed.
