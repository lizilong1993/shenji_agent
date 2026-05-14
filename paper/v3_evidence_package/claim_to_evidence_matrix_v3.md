# V3 Claim-to-Evidence Matrix

Updates to the prewrite claim matrix based on V3A evidence.

| Claim ID | Status | Evidence | Observation | Allowed wording | Forbidden wording |
|---|---|---|---|---|---|
| C7 (tactical) | prior: forbidden → V3A: partially supported | `ai/weights/tactical_model.pt`; `reports/v3a_tactical_ablation/action_diagnostic/action_diagnostic.json`; `reports/v3a_tactical_ablation/smoke_ablation.json`; `reports/v3a_tactical_ablation/TACTICAL_CONTRIBUTION_DECISION.md` | tactical checkpoint exists (52K), loads as TacticalScorer, produces nonzero variegated scores across 120 candidates (5 seeds). Action-level contribution supported (changed_action_rate=1.0). Outcome-level contribution unproven (game replay blocked by model mismatch). | tactical branch changes action selection under diagnostic evaluation; outcome contribution limited/unproven | tactical module improves agent performance |
| C10 (ablation) | partially supported | same evidence as above | tactical ablation executed with on/off switch, action diagnostic, smoke, and multiseed evaluations | tactical on/off ablation completed at diagnostic level | comprehensive game-level ablation study |
