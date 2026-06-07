#!/usr/bin/env python3
"""Quick smoke test for all Phase 2+3 fixes."""
import sys
sys.path.insert(0, ".")

def test_ablation_modes():
    from ai.policy import LearnedPolicy, ABLATION_MODES
    print("ABLATION_MODES:", list(ABLATION_MODES.keys()))
    p = LearnedPolicy()
    for mode in ["full", "no_intent", "no_situation", "rule_only"]:
        p.configure(faction=0, artifact_dir=None, policy_profile="baseline", ablation_mode=mode)
        abm = getattr(p, "ablation_mode", "MISSING")
        print("  mode=%s ablation_mode=%s" % (mode, abm))
        assert abm == mode, "Expected %s got %s" % (mode, abm)
    print("B5 ablation mode validation: PASSED")

def test_enhanced_policy():
    from ai.enhanced_policy import EnhancedPolicy
    ep = EnhancedPolicy()
    ep.configure(faction=0, policy_profile="enhanced")
    assert ep.policy_profile == "enhanced"
    assert ep.intent_model is None
    assert ep.situation_model is None
    print("B7 enhanced policy: PASSED")

def test_selfplay_args():
    import training.selfplay as sp
    import argparse
    # Check that the main function exists and the module loads
    assert hasattr(sp, "main")
    # Check the source for ablation-mode arg
    import inspect
    src = inspect.getsource(sp.main)
    assert "--ablation-mode" in src, "--ablation-mode not found in selfplay.main"
    assert "[:20]" in src, "exploiter count not expanded"
    print("B5+B8 selfplay patches: PASSED")

def test_agent_ablation():
    import inspect
    from ai.agent import Agent
    src = inspect.getsource(Agent.setup)
    assert "ablation_mode" in src, "ablation_mode not passed in agent.setup"
    print("B5 agent.py ablation passthrough: PASSED")

def test_rl_baseline():
    from ai.rl_baseline_agent import LearnedHeuristicAgent
    agent = LearnedHeuristicAgent()
    print("B6 RL baseline agent: PASSED")

def test_enhanced_manifest():
    import json
    from pathlib import Path
    manifest = json.loads(Path("ai/weights_incumbent_enhanced/model_manifest.json").read_text())
    assert manifest.get("policy_profile") == "enhanced"
    assert manifest.get("snapshot_role") == "enhanced_rule_baseline"
    print("B7 enhanced manifest: PASSED")

if __name__ == "__main__":
    tests = [
        test_ablation_modes,
        test_enhanced_policy,
        test_selfplay_args,
        test_agent_ablation,
        test_rl_baseline,
        test_enhanced_manifest,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print("FAILED %s: %s" % (test.__name__, e))
            failed += 1
    print("\n=== RESULTS: %d passed, %d failed ===" % (passed, failed))
    sys.exit(1 if failed else 0)
