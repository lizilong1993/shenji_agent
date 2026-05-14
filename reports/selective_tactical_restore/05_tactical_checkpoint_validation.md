# Tactical Checkpoint Validation

Command:

```bash
/remote-home/lizilong/venvs/shenji/bin/python scripts/validate_tactical_checkpoint.py --checkpoint ai/weights/tactical_model.pt --out reports/selective_tactical_restore/tactical_checkpoint_validation.json
```

Result: `TACTICAL_CHECKPOINT_COMPATIBLE`

Evidence:

- File exists at `ai/weights/tactical_model.pt`.
- Size: 53,013 bytes.
- SHA256: `9fc261e560112e33c4a79c7b40a47a0ebf1314595ad37970cb3b6eef6209cb04`.
- Loaded with `torch.load(..., weights_only=True)`.
- Top-level payload is a dict with `input_dim` and `model_state`.
- Shape-compatible with `ai.models.TacticalScorer`.
- Input dim: 32.
- Hidden dim: 128.

The experiment checkpoint copy produced the same compatible result and checksum.
