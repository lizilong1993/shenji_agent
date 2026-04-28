# Training & Model Quality Diagnosis Report

**Date:** 2026-04-28
**Iteration:** 006
**HEAD:** `57708f6`

---

## A. Situation Model Status: Near-Dead Output

### Evidence
- Pre-sigmoid values after inference-time normalization: near-zero per-node scores
- Sigmoid output per node: ~0.484 (all nodes near-identical)
- `situation_score mean (move)`: 0.00054 (from latest league_20260428_204108)
- Per-node variance: near-zero — all operators get the same score regardless of position

### Root Cause: Feature Scaling Mismatch

**Training path** (`training/supervised.py:train_situation_model`):
1. `StateGraphDataset.__getitem__` loads features from CSV files
2. Features pass through `_parse_list_like` → raw float values
3. These raw values go directly to `GraphSituationModel` with NO normalization
4. Model is trained on raw-scale features

**Inference path** (`ai/policy.py:_score_counterfactual`):
1. `operator_to_feature()` produces features with heterogeneous scales:
   - `obj_id` / 10000 ≈ 0.0001
   - `armor` raw ≈ 1-200
   - `blood` / 100 ≈ 1.0
   - `color` raw ≈ 0/1/2
   - `cur_hex` / 10000 ≈ 0.001
2. Lines 383-385: Ad-hoc z-score normalization per batch:
   ```python
   feat_std = np.std(features, axis=0)
   feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
   features = (features - np.mean(features, axis=0)) / feat_std
   ```
3. This per-batch normalization differs from training-time (no normalization)

**Mismatch mechanism**:
- Training: raw features → model sees scale 0-200 for armor, 0-2 for color
- Inference: z-scored features → model sees ~N(0,1) for all dims
- Model weights are trained for raw scale, receive normalized input → garbage output

### Feature Scale Range (from `_scale()` in features.py)
| Field group | Scale factor | Example raw value | After scale |
|---|---|---|---|
| obj_id, cur_hex, car, launcher | /10000 | 500000 | 50.0 |
| blood, max_blood, value | /100 | 100 | 1.0 |
| basic_speed, speed | /20 | 10 | 0.5 |
| time fields | /100 | 100 | 1.0 |
| armor, A1, type, sub_type | raw (1.0) | 200 | 200.0 |
| binary/boolean fields | raw (1.0) | 1 | 1.0 |
| color (faction) | raw (1.0) | 2 | 2.0 |

**Scale range**: 0.0001 (obj_id/10000) to 200 (armor raw) = **6 orders of magnitude**

### Manifest Check
- `model_manifest.json` has NO normalizer entry
- No `situation_feature_normalizer.json` or similar file
- Training code never computes or saves feature stats

### Model Architecture Check
- `GraphSituationModel.forward()` returns `node_scores.mean(dim=0)` — global mean
- Policy.py works around this by calling `self.situation_model.backbone(x, adj)` directly
- But the **backbone was trained with loss on global mean**, so per-node gradients are averaged
- GraphSage with 3 layers can still produce differing per-node scores IF features differ

---

## B. Feature Scaling Mismatch: Detailed Evidence

### Per-Dimension Scale Analysis

38 numeric fields in `operator_to_feature` + 22 frame-level fields + ~48 padding/derived = 108 total

**Unscaled fields (raw value)** — potential large range:
- `color`: 0/1/2
- `type`: 0-9 (unit type enum)
- `sub_type`: 0-15 (sub-type enum)
- `armor`: 0-300 (armor value)
- `A1`: unknown range, potentially 0-1000
- `stack`: 0-10
- `guide_ability`: 0/1
- `move_state`: 0-5 (enum)
- `cur_pos`: 0-6 (hex position)
- `can_to_move`, `flag_force_stop`, `stop`: boolean
- `tire`: 0-100
- `keep`, `on_board`: boolean
- `lose_control`, `target_state`: boolean/enum
- `weapon_unfold_state`: 0-2
- `C2`, `C3`: unknown

**Top 20 highest-magnitude features** (estimated from scaling factors):
1. armor (raw, ~200)
2. A1 (raw, ~1000)
3. type (raw, ~10)
4. sub_type (raw, ~15)
5. color (raw, ~2)
6. obj_id (/10000, ~50)
7. cur_hex (/10000, ~50)
8. stack (raw, ~10)
9. blood (/100, ~1)
10. max_blood (/100, ~1)
11. value (/100, ~1)
12-38: Most scaled to 0-2 range or are binary

**Near-constant features** (typically 0 or 1 across all operators):
- can_to_move, flag_force_stop, stop, keep, on_board, lose_control, car, launcher, weapon_unfold_state, C2, C3

**Problem**: With ad-hoc per-batch z-score, many binary/constant fields have std≈0 → replaced with 1.0 → noise injection. Meanwhile, the high-magnitude raw fields (armor, A1) get normalized differently per batch → inconsistent model input.

---

## C. Intent Model Move Bias Analysis

### Evidence
- `intent_bias (move)`: -0.364 (negative, discourages movement)
- `intent_bias (shoot)`: +0.084 (slightly positive)
- Intent model outputs 7-class softmax distribution
- Horizon-0 (immediate) class 0 = "stay/no-move" or "move in direction 0"

### Root Cause Investigation
The intent model is `MultiHorizonIntentionModel` trained on `IntentSequenceDataset`:
- Labels come from `tank_enemy` entries in intent JSON data
- Each entry: `[enemy_id, enemy_pos, horizon_0_onehot, horizon_1_onehot, ...]`
- Class 0 likely represents "stay" (center direction), classes 1-6 represent hexagonal directions
- If most training examples have class 0 (stay), the model learns to predict stay → negative move bias

### Training Data Distribution
- Need to check label distribution from last training metrics
- `horizon_accuracy` from last training was not preserved in artifacts
- Typical military simulation: units spend ~60-80% time stationary, 20-40% moving

### Fix Options
1. **Class weighting**: Increase weight for move-positive classes (1-6)
2. **Label balancing**: Resample move-positive examples
3. **Score fusion adjustment**: Apply positive offset for move actions in policy scoring
4. **Horizon tuning**: Use longer horizons where movement patterns are clearer

---

## D. Tactical Model Status: Disabled

### Evidence
- `tactical_model_loaded_match_count: 0` in all 5 matches
- `supervised_baseline.yaml`: `tactical.enabled: false`
- Candidate `ai/weights/`: no `tactical_model.pt`
- Incumbent `ai/weights_incumbent_rule/`: HAS `tactical_model.pt` (53KB)
- `LearnedPolicy` has NO tactical model loading code

### Code Status
- `TacticalScorer` model class exists in `ai/models.py`
- `train_tactical_model()` exists in `training/supervised.py`
- `TacticalPositionDataset` exists in `training/data.py`
- Tactical CSV dataset exists: `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv`
- **All code plumbing exists, just disabled by config**

### Why Disabled
- Historical: Initially disabled to focus on intent/situation models
- No technical blocker — the pipeline is complete
- Just needs config change and policy.py integration

### Enablement Plan
1. Set `tactical.enabled: true` in config
2. Train tactical model
3. Load in `LearnedPolicy.configure()`
4. Use in `choose_shoot_candidate()` and `choose_city()` scoring

---

## E. Policy Score Function Analysis

### Current Scoring Components

**City score** (`choose_city`):
```
combined = situation*100 + city_value*0.5 + owner_bonus + occupy_proximity + occupy_bias + intent_bias - distance*0.08
```
- situation_score: ~0.0005 × 100 = 0.05 (negligible)
- city_value × 0.5: max 10 (for value=20)
- owner_bonus: ±25/15
- occupy_proximity: 20
- distance × 0.08: max ~140 (for distance=1750)
- **Distance still dominates** even at 0.08 weight

**Shoot score** (`choose_shoot_candidate`):
```
combined = situation*100 + attack*8 + hit_prob*30 + intent_bias + kill_bonus + profile_bonus + target_value*0.3 - target_hp*0.05
```
- kill_bonus: 75 (dominates when kill possible)
- target_hp × 0.05: max 10 (for 200 HP)
- situation: negligible

### Missing Components
1. **Tactical score** not used at all
2. **Situation score** effectively zero → no learned differentiation
3. **Coordinated action reward** not implemented
4. **Objective progress tracking** not in scoring

### Candidate vs Incumbent Differentiation
Both sides use the same multi-mode route fallback (from Iteration 003-004 fix). The ONLY difference is:
- Candidate: uses `LearnedPolicy` with situation + intent models (both near-dead)
- Incumbent: uses baseline profile (no learned models)
- Since learned models contribute ~0 to scores, **both sides are effectively identical**

---

## F. Summary of Hard Evidence

| Issue | Severity | Evidence | Fixable |
|---|---|---|---|
| Feature scaling mismatch | CRITICAL | 6-orders magnitude range, no training normalizer | Yes |
| Situation model dead | CRITICAL | Output ~0.0005, zero variance | Fix via normalization |
| Tactical model disabled | HIGH | Code exists, config disabled | Yes |
| Intent move bias negative | MEDIUM | -0.364 mean bias | Partially via class weighting |
| Score function distance-dominated | MEDIUM | distance*0.08 still 10x city_value*0.5 | Tuning |
| No normalizer in manifest | HIGH | Manifest has no normalizer fields | Yes |
| GraphSituationModel global mean | LOW | Policy bypasses via .backbone | Architecture change needed for full fix |

---

## G. Recommended Fixes (Priority Order)

1. **Add feature normalizer to training pipeline**
   - Compute mean/std from training data
   - Save as JSON in run_dir
   - Apply during training (optional but recommended)
   - Include in manifest

2. **Load and apply normalizer in inference**
   - Load normalizer in `LearnedPolicy.configure()`
   - Apply in `_score_counterfactual()` consistently
   - Warn if missing

3. **Enable and train tactical model**
   - Set `tactical.enabled: true`
   - Train and save `tactical_model.pt`
   - Load in policy
   - Integrate into shoot/city scoring

4. **Retrain all models**
   - With normalized features
   - With tactical enabled
   - Publish to ai/weights

5. **Improve intent move bias**
   - Add class weighting for move-positive classes
   - Or add positive offset in score fusion

6. **Rebalance score function**
   - Further reduce distance weight if needed
   - Increase situation contribution once model works
