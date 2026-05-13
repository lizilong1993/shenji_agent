# Claim-to-Evidence Matrix

| Claim ID | Status | Evidence | Observation | Allowed wording | Forbidden wording |
|---|---|---|---|---|---|
| C1 | supported | `training/selfplay.py; experiments/shenji_agent/league_* sidecars` | league directories contain rating/promotion/participants sidecars | implemented and exercised in local SDK evaluation | public benchmark dominance |
| C2 | supported | `ai/models.py; ai/weights/model_manifest.json; ai/weights/*.pth` | intent and situation checkpoints exist | loads learned intent/situation components | learned components drove all decisions |
| C3 | supported | `ai/weights_incumbent_rule/model_manifest.json; participants.json` | incumbent profile baseline/rule manifests present | rule incumbent baseline | strong external baseline |
| C4 | supported | `experiments/shenji_agent/league_*/promotion.json` | promotion.json files present | promotion-gated local league evaluation | fully validated benchmark suite |
| C5 | supported | `experiments/shenji_agent/league_20260429_032914/promotion.json` | accepted=True | one accepted promotion sidecar exists | repeatable multi-window success |
| C6 | forbidden | `formal_artifacts_index.json` | no consecutive two-window proof extracted | two-window success not proven | two-window promotion success |
| C7 | forbidden | `ai/weights/model_manifest.json; validation/tactical_diagnostic.json` | tactical checkpoint exists=False; learned coverage=0.0 | tactical path remains unproven | tactical module improves performance |
| C8 | unsupported | `experiments/shenji_agent/league_20260429_032914/rating.json` | route attempts=0 | route diagnostics are tracked | route planner is reliable |
| C9 | supported | `rating.json policy.move_route_aggregate; validation/route_diagnostic.json` | route attempt evidence is absent/zero in key sidecars | route evidence is insufficient and remains a limitation | route issue resolved |
| C10 | partially supported | `configs/*ablation*.yaml; tests; experiments` | tactical ablation config exists; no complete protocol-matched ablation table extracted | preliminary baseline/ablation scaffolding | comprehensive ablation study |
| C11 | partially supported | `08_novelty_and_positioning.md` | novelty framed around evidence-gated workflow, not LSTM/GraphSAGE | evidence-gated workflow contribution | new LSTM or GraphSAGE algorithm |
| C12 | partially supported | `00_repository_state.md; 01_asset_provenance.md; 11_validation_log.md` | assets and commands documented; auth/local SDK may constrain reruns | repo-local reproducibility package | fully independently reproducible |
| C13 | forbidden | `none` | not_available | no public SOTA claim | SOTA |
| C14 | forbidden | `09_reviewer_risk_register.md` | major evidence gaps remain | ready for conservative manuscript drafting with blockers | top-tier-ready submission package |
