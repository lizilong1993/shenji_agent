import torch
import json

for name in ["intention_lstm.pth", "situation_gcn.pth", "tactical_model.pt"]:
    path = f"/remote-home/lizilong/shenji_agent/ai/weights/{name}"
    print(f"\n=== {name} ===")
    try:
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict):
            print(f"Top-level keys: {list(ckpt.keys())}")
            for k, v in ckpt.items():
                if isinstance(v, dict):
                    sub_keys = list(v.keys())[:5]
                    print(f"  {k}: dict with {len(v)} keys, first few: {sub_keys}")
                    if sub_keys:
                        first_v = v[sub_keys[0]]
                        if hasattr(first_v, 'shape'):
                            print(f"    first value shape: {first_v.shape}")
                elif hasattr(v, 'shape'):
                    print(f"  {k}: tensor shape={v.shape}")
                else:
                    print(f"  {k}: {type(v).__name__} = {v}")
        else:
            print(f"Type: {type(ckpt).__name__}")
    except Exception as e:
        print(f"ERROR: {e}")
