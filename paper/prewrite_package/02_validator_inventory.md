# Validator Inventory

| Validator | Exists at HEAD before recovery | Found in history | Candidate source commit | Safe to restore | Writes formal artifacts | Should restore now |
|---|---:|---:|---|---:|---:|---:|
| `scripts/run_round005_smoke_pipeline.py` | no | yes | `bccc61c4fb34c77f45aac7dbe9e188816b646ef3` | yes, fail-closed preflight | no | yes |
| `scripts/round005_tactical_diagnostic.py` | no | yes | `bccc61c4fb34c77f45aac7dbe9e188816b646ef3` | yes, diagnostic only | no | yes |
| `scripts/round005_route_diagnostic.py` | no | yes | `bccc61c4fb34c77f45aac7dbe9e188816b646ef3` | yes, sidecar reader | no | yes |

These scripts were restored path-specifically. They write reports under `reports/` or a caller-selected output path and do not hand-edit formal league sidecars.
