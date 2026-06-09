import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
sys.path.insert(0, "/remote-home/lizilong/shenji_agent")

from ai.policy import LearnedPolicy
from ai.features import action_feature_vector

policy = LearnedPolicy()
policy.configure(faction=1, artifact_dir="ai/weights", policy_profile="challenger")

print(f"\nIntent loaded:   {policy.intent_model is not None}")
print(f"Situation loaded: {policy.situation_model is not None}")
print(f"Tactical loaded:  {policy.tactical_model is not None}")
print(f"Normalizer:       {bool(policy._feature_normalizer)}")
print(f"Device:           {policy.device}")
print(f"Model type:       {type(policy.intent_model).__name__}")
print(f"Situation type:   {type(policy.situation_model).__name__}")

# Test tactical feature dimension
if policy.tactical_model is not None:
    dummy_op = {"obj_id": 1, "cur_hex": 500, "type": 1, "blood": 100, "attack_level": 5}
    dummy_candidate = {"target_obj_id": 2, "weapon_id": 1, "damage": 30, "hit_prob": 0.8}
    try:
        features = action_feature_vector(dummy_op, dummy_candidate)
        print(f"\nTactical feature dim: {len(features)} (model expects 32)")
    except Exception as e:
        print(f"\nTactical feature generation error: {e}")

# Test intent inference
if policy.intent_model is not None:
    import numpy as np
    policy.enemy_histories[99].append(np.random.randn(108).astype(np.float32))
    result = policy.predict_intent(99)
    print(f"\nIntent inference OK: top_direction={result['top_direction']}, confidence={result['confidence']:.3f}")

print("\nAll models loaded successfully!" if policy.has_models() else "\nSome models FAILED to load")
