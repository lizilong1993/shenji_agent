# Tactical Checkpoint Validation

Date: 2026-05-14

## Result

**TACTICAL_CHECKPOINT_COMPATIBLE**

## Details

| Property | Value |
|---|---|
| File | `ai/weights/tactical_model.pt` |
| Exists | Yes |
| Size | 53,013 bytes |
| SHA256 | `9fc261e560112e33c4a79c7b40a47a0ebf1314595ad37970cb3b6eef6209cb04` |
| Format | PyTorch ZIP serialization (`.pt`) |
| Source | Quarantine backup (checksum matched) |

### Model Structure

- **Architecture**: `ai.models.TacticalScorer` (3-layer MLP)
- **input_dim**: 32 (TACTICAL_FEATURE_DIM)
- **hidden_dim**: 128
- **output_dim**: 1 (binary classifier)

### State keys found (all expected):

| Key | Shape |
|---|---|
| `model.0.weight` | (128, 32) |
| `model.0.bias` | (128,) |
| `model.3.weight` | (64, 128) |
| `model.3.bias` | (64,) |
| `model.5.weight` | (1, 64) |
| `model.5.bias` | (1,) |

### Validation method

Pickle bytecode analysis of ZIP-format `.pt` file without torch import.
Torch `import torch` hangs in this environment (CUDA library issue).
Model structure verified via `pickletools` opcode disassembly.

### Verdict

Checkpoint is fully compatible with the current `TacticalScorer` class.
