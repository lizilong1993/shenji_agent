# Top-Tier Gap Audit

Date: 2026-05-13

## Summary

The current repository is not a top-tier submission candidate. Version 1 is a conservative
technical-report evidence package with useful closed-loop SDK evaluation artifacts, but the
top-tier blockers are executable and material:

- The candidate checkpoint manifest expects `.pt` files, while restored intent/situation
  checkpoints are `.pth`.
- The tactical model checkpoint is absent, so tactical contribution is unproven.
- Route diagnostics show `15641 / 23580` empty routes, an empty-route rate of
  `0.6633163698049194`, all attributed to `no_valid_move_candidate`.
- Existing promotion artifacts do not prove two consecutive passing windows.
- Baseline and ablation evidence is incomplete and not multi-seed/statistically summarized.
- Novelty remains risky unless reframed as a validated evidence-gated policy protocol or
  upgraded with route-aware/evidence-gated action selection and ablation evidence.

No top-tier claims should be lifted at this audit stage.

## Audit Table

| Dimension | Current status | Required for top-tier | Gap | Planned fix | Can fix in this repo? |
|---|---|---|---|---|---|
| Soundness | Promotion sidecars and validators exist, but sidecar coverage is uneven and two-window proof is absent. Route and tactical gates fail. | Protocol-matched evaluation with clear candidate/incumbent identities, leak checks, deterministic seeds, failure analysis, and statistical confidence. | Missing two-window proof; route and tactical diagnostics fail; deterministic/multi-seed evidence not summarized. | Repair reproducibility first, then run smoke baselines/ablations, route diagnostics, and statistical summaries. | Partly; formal SDK runs may still depend on auth/environment. |
| Novelty | Standard LSTM and GraphSAGE-style models; V1 framing leans on evidence-gated workflow discipline. | Clear method contribution beyond standard models, supported by ablations. | Algorithmic novelty is weak if the contribution is only LSTM plus GraphSAGE. | Prefer a defensible evidence-gated/route-aware policy composition contribution, with ablations showing the gate changes behavior or prevents overclaiming. | Partly; strong novelty needs new evidence and may remain below top-tier threshold. |
| Evidence | Intent/situation checkpoints exist as `.pth`; manifest expects `.pt`. Tactical checkpoint is missing. Existing league artifacts include some accepted windows but incomplete formal proof. | Manifest-resolved assets, executable baselines/ablations, route/tactical diagnostics, multi-seed evaluation, confidence intervals. | Checkpoint contract mismatch; tactical absent; route empty-rate high; baseline/ablation table incomplete; no multi-seed statistical package. | Build asset contract, manifest-driven validators, route/tactical audits, baseline runner, statistical summarizer. | Yes for tooling; outcome evidence depends on SDK execution and actual performance. |
| Significance | Closed-loop land-wargame SDK evaluation is domain-relevant, but scope is local SDK and scenario/map diversity is not yet top-tier documented. | Broad enough evaluation scope and clear relevance to AI-method or embodied/spatial decision-making venues. | Narrow benchmark/protocol and unclear external comparability. | Stratify by scenario/map where artifacts allow; record venue limitations honestly. | Partly; public benchmark/SOTA cannot be created from current evidence alone. |
| Reproducibility | Assets and commands are documented. V1 smoke preflight fails closed due to checkpoint naming mismatch and missing tactical checkpoint. | Reproducible commands, asset contracts, deterministic seeds, clear external dependency and checkpoint provenance. | Manifest, restored files, and validators disagree; external SDK/auth constraints remain. | Make loader/validator paths manifest-driven and fail-closed; add unit tests and contract docs. | Yes for repo-local reproducibility; external auth constraints may remain. |

## Detailed Findings

### 1. Soundness

Evidence inspected:

- `paper/prewrite_package/03_formal_artifact_inventory.md`
- `paper/prewrite_package/04_claim_to_evidence_matrix.md`
- `paper/prewrite_package/11_validation_log.md`
- `experiments/shenji_agent/league_*/promotion.json`
- `experiments/shenji_agent/league_*/rating.json`

Findings:

- Some league directories have all required sidecars, but at least one accepted promotion
  window has missing sidecars and no extracted consecutive two-window proof.
- Existing accepted sidecars do not establish repeatable multi-window success.
- Learned coverage is inconsistent: some suitable league artifacts report `learned_coverage=1.0`,
  while others report `0.0`.
- Statistical confidence intervals and multi-seed summaries are not present in the V1 evidence
  package.

### 2. Novelty

Evidence inspected:

- `paper/prewrite_package/07_methods_package.md`
- `paper/prewrite_package/08_novelty_and_positioning.md`
- `ai/models.py`
- `ai/policy.py`

Findings:

- The model components are not novel enough when described as LSTM and GraphSAGE-style
  classifiers.
- The strongest current novelty angle is the evidence-gated claim and promotion protocol, but
  this still needs executable validators and ablation-style evidence showing that the protocol
  prevents unsupported claims.
- A route-aware or evidence-gated policy composition method is the most feasible repo-local
  upgrade path.

### 3. Evidence

Evidence inspected:

- `ai/weights/model_manifest.json`
- `ai/weights/`
- `paper/prewrite_package/validation/tactical_diagnostic.json`
- `paper/prewrite_package/validation/route_diagnostic.json`
- `configs/*ablation*.yaml`
- `tests/`

Findings:

- Existing candidate checkpoint files are `ai/weights/intention_lstm.pth` and
  `ai/weights/situation_gcn.pth`.
- The candidate manifest expects `intent_lstm.pt`, `situation_gcn.pt`, and
  `tactical_model.pt`.
- Tactical diagnostic fails closed because `ai/weights/tactical_model.pt` does not exist.
- Route diagnostic reports `15641 / 23580` empty routes and only one failure reason:
  `no_valid_move_candidate`.
- Tests exist for previous contracts, but V2-specific asset, route, tactical, baseline,
  statistical, and claim-boundary validators do not yet exist.

### 4. Significance

Evidence inspected:

- `program.md`
- `paper/prewrite_package/09_reviewer_risk_register.md`
- `land_wargame_sdk/Data/Data/scenarios/`
- `land_wargame_sdk/Data/Data/maps/`

Findings:

- The domain has plausible significance as closed-loop land-wargame decision-making.
- The evidence is currently local-SDK-focused and not protocol-matched to public SOTA.
- Scenario and map assets are present, but V1 does not provide a top-tier-grade stratified
  evaluation across them.

### 5. Reproducibility

Evidence inspected:

- `paper/prewrite_package/00_repository_state.md`
- `paper/prewrite_package/01_asset_provenance.md`
- `paper/prewrite_package/11_validation_log.md`
- `scripts/run_round005_smoke_pipeline.py`
- `training/selfplay.py`
- `ai/policy.py`

Findings:

- Reproducibility is currently fragile because validator expectations, manifest names, and
  restored checkpoint filenames disagree.
- The repo contains SDK wheel, scenarios, maps, candidate assets, and incumbent manifest, but
  some formal runs may still depend on SDK authorization and environment constraints.
- V2 must make candidate asset resolution manifest-driven and explicitly fail closed on missing
  tactical assets rather than conflating naming mismatch with missing model evidence.

## Phase 1 Decision

Proceed to Phase 2 before any major experiment. The first concrete blocker is the asset and
checkpoint contract mismatch, because it prevents smoke preflight and makes later route,
tactical, baseline, and statistical evidence unreliable.

Top-tier status after this audit: not achieved.
