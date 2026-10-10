# Shenji intent annotation guide

## Annotation unit

Annotate a temporal segment for one entity, group, company, or side. Do not copy the same label independently onto every frame. Segment boundaries should correspond to a stable task objective or a verified transition.

## Label order

1. Select the subject scope and subject ID.
2. Mark the first step at which the intent is active.
3. Mark the final active step, including a terminal completion step.
4. Choose one primary intent.
5. Add secondary intents only when they are simultaneously supported.
6. Identify the target type and target ID when applicable.
7. Record status, outcome, evidence source, evidence grade, and confidence.

## Three label kinds

1. `commanded` records the hidden task or explicit command assigned before or during execution.
2. `realized` records the behavior actually carried out, whether or not it matches the command.
3. `human_inferred` records an annotator judgment made from a specified partial-observation view.

These labels must not be merged. A failed or switched task may have one commanded segment and several realized segments.

## Evidence grades

1. Grade A uses a direct simulator action, mission command, ownership change, or explicit scenario plan.
2. Grade B uses deterministic multi-event and trajectory criteria, followed by sampled human audit.
3. Grade C is ambiguous from behavior alone and requires an explicit plan tag or human review.

`FEINT` is always Grade C. A failed attack or aborted movement is not sufficient evidence for a feint.

## Boundary rules

Use a minimum stable duration defined by the benchmark configuration for derived labels. Apply hysteresis when a noisy feature crosses a threshold repeatedly. End the segment at completion, abortion, or intent switch. Record the new intent as a separate segment and link both segments through their evidence references.

## Multi-label rule

The primary intent captures the dominant task objective. Secondary labels capture concurrent means or support functions. For example, a company may have primary `SEIZE` with secondary `FLANK` and `FIRE_SUPPORT`. Do not force labels to be mutually exclusive and do not normalize their probabilities to sum to one.

## Audit

Every Grade B rule must be audited on a stratified sample covering maps, policies, sides, success outcomes, and failure outcomes. Disagreements are adjudicated by a second reviewer. Store reviewer identity only in the private annotation ledger, then publish anonymized adjudication metadata.
