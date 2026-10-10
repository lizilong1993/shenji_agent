# Shenji Annotation Protocol v1

## Overview

This document defines the annotation workflow for Grade A, B, and C intent labels in the Shenji dataset. It covers annotator selection, viewing conditions, labeling procedure, quality control, and adjudication.

## 1. Label Grades

| Grade | Definition | Applicable Intents | Methodology |
|-------|-----------|-------------------|-------------|
| **A** | Fully deterministic from simulator truth | SEIZE, FORTIFY, ASSEMBLE, DISPERSE, TARGET_GUIDANCE, FIRE_SUPPORT, ASSAULT, HOLD, RECON | Automatic rules (existing `derive_grade_a_intent_plan`) |
| **B** | Requires pattern recognition, may need human confirmation | BREAKTHROUGH, FLANK, ENCIRCLEMENT, PURSUIT, AMBUSH, INTERDICT, SURVEILLANCE, REINFORCE, ESCORT, REDEPLOY, RETREAT, EVADE | Automated detection + human review |
| **C** | Requires plan tag or full-context human judgment | FEINT | Plan tag + independent human labeling |

## 2. Annotator Requirements

- Minimum 2 independent annotators per Grade B/C segment
- Annotators must have domain familiarity (military tactical reasoning or equivalent wargaming experience)
- Third expert adjudicator for disputed samples
- Annotator training: review 20 benchmark episodes with gold-standard labels before independent work

## 3. Annotator Viewing Conditions

- **Grade B annotators**: See only the replay of ONE side (red or blue, assigned randomly per episode)
- **Grade C (FEINT) annotators**: See the full scenario plan including both sides' mission types
- **For human-inferred benchmark**: annotator sees ONLY the partial observations available to the annotator's assigned side, with the same visibility/occlusion as the simulation. No truth data. No future data.
- Recording per annotation event:
  - `visible_entities`: list of entity IDs visible to the annotator at each step
  - `occlusion_level`: fraction of total entities that were visible
  - `confidence`: annotator's self-reported confidence [0.0, 1.0]
  - `evidence_basis`: free-text rationale for the label
  - `abstained`: boolean, true if annotator chose UNKNOWN/AMBIGUOUS

## 4. Labeling Procedure

### 4.1 Per-segment annotation

Each annotator is presented with:
1. A replay segment (start_step to end_step) of the episode from their assigned side
2. The instruction to label the segment with one or more of 22 intents + 4 status layers
3. The option to mark the segment as:
   - `BACKGROUND` — no tactical intent detected
   - `UNKNOWN` — insufficient information
   - `AMBIGUOUS` — multiple conflicting intents possible
   - One or more specific intents from the 22-class ontology

### 4.2 Session format

Each annotation session produces a row in the `annotations` table:

```json
{
  "annotation_id": "sha256(annotator_id + episode_id + segment_id)",
  "episode_id": "...",
  "annotator_id": "anon_001",
  "assigned_side": "red",
  "segment_id": "...",
  "label_kind": "human_inferred",
  "primary_intent": "ASSAULT",
  "secondary_intents": ["FIRE_SUPPORT"],
  "status_layer": null,
  "start_step": 45,
  "end_step": 120,
  "confidence": 0.85,
  "abstained": false,
  "evidence_basis": "Unit moved directly toward enemy position and engaged with sustained fire",
  "visible_entity_count": 3,
  "total_entity_count": 7,
  "occlusion_level": 0.57,
  "review_status": "pending"
}
```

## 5. Quality Control

### 5.1 Inter-annotator agreement

- Per label category: Krippendorff's alpha
- Per segment: pairwise label F1, time boundary IoU
- Multi-label agreement: Jaccard similarity on label sets
- Minimum thresholds:
  - Krippendorff's alpha ≥ 0.67 for Grade B labels
  - Krippendorff's alpha ≥ 0.80 for Grade C labels
  - Boundary IoU ≥ 0.50 for at least 70% of segments

### 5.2 Adjudication

Disputed samples (below threshold) go to a third expert annotator:
- If 2 of 3 agree: adopt majority label
- If all 3 disagree: label as AMBIGUOUS + flag for ontology review
- Adjudication decisions are recorded in `adjudication_schema_v1.json`

### 5.3 Stratified sampling for quality audit

Quality metrics are reported stratified by:
- Intent category (per 22 classes)
- Map family
- Policy (controller)
- Visibility level (high/medium/low)
- Evidence grade (A/B/C)
- Outcome (success/failure/switch)

## 6. Hard Negative Construction

For each of the following confusion pairs, construct matching segments:

| Pair | Matching criteria |
|------|------------------|
| FEINT vs failed ASSAULT | Similar trajectory and action profile, different plan tag |
| RETREAT vs EVADE | Similar initial movement, different duration and task return |
| FLANK vs obstacle avoidance | Similar path geometry, different enemy presence |
| SURVEILLANCE vs RECON | Similar observation pattern, different target novelty |
| ASSAULT vs SEIZE | Similar approach, different terminal objective |
| HOLD vs FORTIFY | Similar position, different concealment state |

Each matched pair must be reviewed by 2 annotators to confirm the distinction is valid in the MiaoSuan domain.

## 7. Human-Inferred Benchmark Protocol

For the `human_inferred_intent` track:
1. Annotators watch a replay from ONE side's perspective
2. Every N steps (N=10 default), the replay pauses and the annotator records:
   - Current intent judgment (one or more of 22 + 4 status layers)
   - Confidence
   - The entity(ies) forming the judgment basis
3. After the full episode, annotator produces a complete intent_segments table
4. Human-inferred labels are stored with `label_kind=human_inferred` in `intent_segments`
5. Models evaluated on this track use the same partial observation data available to the human annotator

## 8. Data Storage

Annotation data is stored in:
- `Shenji/annotations/annotations_v1/` — per-segment annotation records (JSONL)
- `Shenji/annotations/adjudication_v1/` — adjudication decisions (JSONL)
- `Shenji/annotations/human_inferred_v1/` — human-inferred intent labels (intent_segments format JSONL)
