#!/usr/bin/env python3
"""Fix has_models() method quotes in policy.py."""
from pathlib import Path

policy_path = Path("ai/policy.py")
text = policy_path.read_text(encoding="utf-8")

old = '''    def has_models(self) -> bool:
        abl = getattr(self, ablation_mode, full)
        if abl == full:
            return self.intent_model is not None and self.situation_model is not None
        if abl == no_intent:
            return self.situation_model is not None
        if abl == no_situation:
            return self.intent_model is not None
        return False  # rule_only'''

new = '''    def has_models(self) -> bool:
        abl = getattr(self, "ablation_mode", "full")
        if abl == "full":
            return self.intent_model is not None and self.situation_model is not None
        if abl == "no_intent":
            return self.situation_model is not None
        if abl == "no_situation":
            return self.intent_model is not None
        return False  # rule_only'''

if old in text:
    text = text.replace(old, new, 1)
    policy_path.write_text(text, encoding="utf-8")
    print("Fixed has_models() quotes")
else:
    print("Could not find old has_models() text")
    # Try to find and show what's there
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'def has_models' in line:
            print(f"Found at line {i+1}:")
            for j in range(i, min(i+10, len(lines))):
                print(f"  {j+1}: {lines[j]}")
            break
