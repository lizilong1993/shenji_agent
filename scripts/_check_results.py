import json
import sys
import os

latest = sorted(
    [d for d in os.listdir("experiments/shenji_agent") if d.startswith("league_")],
    reverse=True
)[0]
base = f"experiments/shenji_agent/{latest}"
print(f"Analyzing: {base}")

# Participants
p = json.load(open(f"{base}/participants.json"))
for side in ["candidate", "incumbent"]:
    info = p.get(side, {})
    print(f"\n{side}:")
    for key in ["intent_model_loaded", "situation_model_loaded", "tactical_model_loaded",
                "tactical_model_loaded_match_count", "policy_profile", "device"]:
        print(f"  {key}: {info.get(key, 'N/A')}")

# Match results route diagnostics
mr = json.load(open(f"{base}/match_results.json"))
for i, match in enumerate(mr.get("matches", [])[:2]):
    for side in ["candidate"]:
        stats = match.get(side, {}).get("policy_stats", {})
        route_agg = stats.get("move_route_aggregate", {})
        if route_agg:
            total = route_agg.get("total_attempts", 0)
            empty = route_agg.get("empty_route_count", 0)
            by_type = route_agg.get("by_move_type", {})
            rate = f"{empty/total*100:.1f}%" if total > 0 else "N/A"
            print(f"\nMatch {i} route ({side}): total={total}, empty={empty} ({rate})")
            for mt, data in by_type.items():
                mt_total = data.get("total", 0)
                mt_empty = data.get("empty", 0)
                mt_rate = f"{mt_empty/mt_total*100:.1f}%" if mt_total > 0 else "N/A"
                print(f"  {mt}: total={mt_total}, empty={mt_empty} ({mt_rate})")
        
        # Score means
        score_means = stats.get("score_means", {})
        if score_means:
            print(f"\nMatch {i} score means ({side}):")
            for action_type, means in score_means.items():
                print(f"  {action_type}: tactical={means.get('tactical_score', 0):.3f}, "
                      f"intent={means.get('intent_bias', 0):.3f}, "
                      f"situation={means.get('situation_score', 0):.3f}, "
                      f"objective={means.get('objective_score', 0):.3f}")
