# Situation Model Architecture and Data Blocked Analysis

## Question
After fixing training/inference normalization, is the remaining 0-0 failure still primarily caused by GraphSituationModel node indistinguishability, or by data/objective limits outside the GNN architecture?

## Evidence Summary
- Situation graph dataset has `108` graph samples.
- Each graph has `12` nodes.
- Labels are graph-level only: `label.csv` has columns `filename,label`.
- Label distribution: `72` negative, `36` positive.
- No node-level, city-level, target-level, damage-level, occupy-success, or tactical objective labels are present in WG-StateGraph.
- Normalized retrain increased situation per-node variance from `1.5056e-05` to `0.004074`.
- Model-quality warnings dropped from 3 to 1; only `intent_negative_move_bias` remains.
- Smoke still ended 0-0 because no valid `Shoot` or `Occupy` actions were available in the 50-step smoke horizon.
- Medium 400-step run produced valid/generated `Shoot` actions with learned shoot coverage `1.0`, but every match leg still ended 0-0.

## Required Questions

1. Training data size:
The situation dataset contains only `108` graph samples. This is small for GNN training and too small to reliably learn promotion-quality policy behavior.

2. Label distinction:
Labels have some class distinction (`72` negative / `36` positive), but the labels are graph-level binary outcomes. They do not identify which node, city, route, or target caused the label.

3. Situation label distribution:
`label=0`: 72 samples. `label=1`: 36 samples. Mean label is `0.3333`.

4. Overfit/underfit:
Current normalized retrain situation validation accuracy is `0.5455`, only modestly above chance for a small validation split. This is not strong evidence of a reliable tactical/city evaluator.

5. Validation metric credibility:
Low. The dataset has 108 graph samples, all from the complex bucket. Validation accuracy is split from a small homogeneous graph set and does not validate city-level decisions or game-score impact.

6. Per-node score vs graph-level target conflict:
Yes. The model is trained with a graph-level binary loss. Even if node scores differ after normalization, the loss only supervises their pooled graph score. It does not teach which city or unit should receive the high score.

7. Does graph-level training naturally support node-level city scoring?
No. Per-node city scoring is an inference-time use of a graph-level label. The model can produce different node scores, but there is no supervised target tying a node score to an objective city, capture opportunity, or combat result.

8. Need for node-level/city-level supervision:
Yes. Promotion-quality city scoring likely needs city-level labels, occupy-success labels, route-progress labels, or counterfactual value labels generated from rollouts.

9. Need for tactical/city objective labels:
Yes. The tactical model has 24,080 samples and works as a local action classifier, but it does not solve city objective value. Situation labels need objective-specific supervision.

10. Can current data support promotion-quality policy?
Hard evidence says no. After normalization, learned movement and shooting engage, but medium self-play remains 0-0 across all five paired matches.

11. Minimal next data/algorithm requirement:
Generate or collect rollout-derived labels for city capture progress, score delta, damage/kill delta, and per-operator/per-city counterfactual value. Then train a node/city-level head against those labels. Architecture-only changes are unlikely to fix the absence of such labels.

## Architecture Decision
Do not implement GNN v2 in this iteration as the next primary branch. The explicit architecture symptom from Iteration 006, near-zero node variance, improved by about 270x after applying the normalizer during training. The current blocker is now objective/label mismatch: learned move and shoot paths are active, but generated actions do not change game reward.

## Hard Blocker
Current labels and evaluation scenarios do not provide enough supervised signal to map situation scores to score-producing actions. A GNN v2 may improve representational capacity, but without node/city/score-delta labels it would still optimize the wrong target.
