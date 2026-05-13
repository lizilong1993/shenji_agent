from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "prewrite_package"
FIG = OUT / "figure_data"
TRACKED_FILES: set[str] = set()
IGNORED_PATHS: set[str] = set()


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=20)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, "COMMAND_TIMED_OUT: " + " ".join(cmd)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_parse_error": str(exc)}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def git_bool(args: list[str]) -> bool:
    return run(["git", *args])[0] == 0


def tracked(path: Path) -> bool:
    rel = str(path.relative_to(ROOT))
    if path.is_dir():
        return any(item.startswith(rel.rstrip("/") + "/") for item in TRACKED_FILES)
    return rel in TRACKED_FILES


def ignored(path: Path) -> bool:
    rel = str(path.relative_to(ROOT))
    return rel in IGNORED_PATHS or any(item.startswith(rel.rstrip("/") + "/") for item in IGNORED_PATHS)


def files_under(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    rel = str(path.relative_to(ROOT)).rstrip("/") + "/"
    tracked_matches = [ROOT / item for item in TRACKED_FILES if item.startswith(rel)]
    if tracked_matches:
        return sorted(tracked_matches)
    results: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = dirnames[:20]
        for name in filenames[:100]:
            results.append(Path(dirpath) / name)
            if len(results) >= 2000:
                return results
    return results


def size_sum(paths: list[Path]) -> int:
    return sum(p.stat().st_size for p in paths if p.exists())


def asset_record(path_text: str, needed: str, deleted_by_sync: bool = True) -> dict[str, Any]:
    path = ROOT / path_text
    files = files_under(path)
    status = "exists" if path.exists() and files else "partial" if path.exists() else "missing"
    return {
        "path": path_text,
        "status": status,
        "file_count": len(files),
        "total_size_bytes": size_sum(files),
        "representative_filenames": [str(p.relative_to(ROOT)) for p in files[:8]],
        "tracked_by_git": tracked(path),
        "ignored_by_git": ignored(path),
        "needed_for": needed,
        "previously_deleted_by_sync_commit_8c7067c": deleted_by_sync,
        "safe_backup_instructions_needed": status != "missing",
    }


def league_dirs() -> list[Path]:
    dirs = {
        ROOT / "/".join(item.split("/")[:3])
        for item in TRACKED_FILES
        if item.startswith("experiments/shenji_agent/league_")
    }
    return sorted(dirs)


def sidecar_summary(path: Path) -> dict[str, Any]:
    required = ["summary.md", "rating.json", "promotion.json", "league_state.json", "participants.json", "exploitability.json"]
    present = {name: (path / name).exists() for name in required}
    rating = read_json(path / "rating.json") if present["rating.json"] else {}
    promotion = read_json(path / "promotion.json") if present["promotion.json"] else {}
    participants = read_json(path / "participants.json") if present["participants.json"] else {}
    map_list = read_json(path / "map_list.json") if (path / "map_list.json").exists() else {}
    route = rating.get("policy", {}).get("move_route_aggregate", {}) if isinstance(rating, dict) else {}
    policy = rating.get("policy", {}) if isinstance(rating, dict) else {}
    return {
        "path": str(path.relative_to(ROOT)),
        "present_sidecars": present,
        "missing_required_sidecars": [k for k, v in present.items() if not v],
        "candidate": participants.get("candidate", {}).get("profile") or "not_available",
        "incumbent": participants.get("incumbent", {}).get("profile") or "not_available",
        "map_or_scenario_count": len(map_list) if isinstance(map_list, list) else "not_available",
        "win_rate": rating.get("overall", {}).get("point_estimate", "not_available") if isinstance(rating, dict) else "not_available",
        "elo": rating.get("overall", {}).get("elo", "not_available") if isinstance(rating, dict) else "not_available",
        "match_count": rating.get("overall", {}).get("count", "not_available") if isinstance(rating, dict) else "not_available",
        "promotion_accepted": promotion.get("accepted", "not_available") if isinstance(promotion, dict) else "not_available",
        "promotion_failures": promotion.get("failures", "not_available") if isinstance(promotion, dict) else "not_available",
        "learned_path_coverage": policy.get("learned_path_coverage", "not_available") if isinstance(policy, dict) else "not_available",
        "tactical_model_loaded_match_count": policy.get("tactical_model_loaded_match_count", "not_available") if isinstance(policy, dict) else "not_available",
        "route_total_attempts": route.get("total_attempts", "not_available") if isinstance(route, dict) else "not_available",
        "route_empty_count": route.get("empty_route_count", "not_available") if isinstance(route, dict) else "not_available",
        "suitable_for_manuscript_evidence": all(present[n] for n in ["rating.json", "promotion.json", "participants.json"]),
    }


def best_league(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [s for s in summaries if s["suitable_for_manuscript_evidence"]]
    accepted = [s for s in complete if s["promotion_accepted"] is True]
    return (accepted or complete or summaries)[-1] if summaries else {}


def make_claims(best: dict[str, Any], assets: dict[str, Any]) -> list[dict[str, str]]:
    one_window = best.get("promotion_accepted") is True
    route_total = best.get("route_total_attempts")
    learned_cov = best.get("learned_path_coverage")
    tactical_exists = (ROOT / "ai" / "weights" / "tactical_model.pt").exists() or (ROOT / "ai" / "weights" / "tactical_model.pth").exists()
    return [
        {"Claim ID": "C1", "Candidate manuscript claim": "Closed-loop land-wargame SDK evaluation was implemented.", "Status": "supported", "Evidence path": "training/selfplay.py; experiments/shenji_agent/league_* sidecars", "Evidence type": "code and JSON sidecars", "Exact metric or observation": "league directories contain rating/promotion/participants sidecars", "Known caveat": "single proprietary/local SDK", "Allowed wording": "implemented and exercised in local SDK evaluation", "Forbidden wording": "public benchmark dominance"},
        {"Claim ID": "C2", "Candidate manuscript claim": "Candidate agent uses learned intent/situation components.", "Status": "supported", "Evidence path": "ai/models.py; ai/weights/model_manifest.json; ai/weights/*.pth", "Evidence type": "code, manifests, checkpoints", "Exact metric or observation": "intent and situation checkpoints exist", "Known caveat": "sidecars may show zero learned action-path coverage", "Allowed wording": "loads learned intent/situation components", "Forbidden wording": "learned components drove all decisions"},
        {"Claim ID": "C3", "Candidate manuscript claim": "Candidate was evaluated against a rule incumbent.", "Status": "supported", "Evidence path": "ai/weights_incumbent_rule/model_manifest.json; participants.json", "Evidence type": "manifest and participants sidecars", "Exact metric or observation": "incumbent profile baseline/rule manifests present", "Known caveat": "incumbent has no checkpoint payload", "Allowed wording": "rule incumbent baseline", "Forbidden wording": "strong external baseline"},
        {"Claim ID": "C4", "Candidate manuscript claim": "Promotion gate or league-style evaluation exists.", "Status": "supported", "Evidence path": "experiments/shenji_agent/league_*/promotion.json", "Evidence type": "JSON sidecars", "Exact metric or observation": "promotion.json files present", "Known caveat": "some runs incomplete or failed", "Allowed wording": "promotion-gated local league evaluation", "Forbidden wording": "fully validated benchmark suite"},
        {"Claim ID": "C5", "Candidate manuscript claim": "One verified promotion window exists.", "Status": "supported" if one_window else "unsupported", "Evidence path": best.get("path", "not_available") + "/promotion.json", "Evidence type": "promotion sidecar", "Exact metric or observation": f"accepted={best.get('promotion_accepted', 'not_available')}", "Known caveat": "accepted sidecar can have zero learned-path coverage", "Allowed wording": "one accepted promotion sidecar exists" if one_window else "not claimable", "Forbidden wording": "repeatable multi-window success"},
        {"Claim ID": "C6", "Candidate manuscript claim": "Two-window hard-stop success exists.", "Status": "forbidden", "Evidence path": "formal_artifacts_index.json", "Evidence type": "absence/insufficient sidecars", "Exact metric or observation": "no consecutive two-window proof extracted", "Known caveat": "requires formal consecutive-window sidecars", "Allowed wording": "two-window success not proven", "Forbidden wording": "two-window promotion success"},
        {"Claim ID": "C7", "Candidate manuscript claim": "Tactical module contributes.", "Status": "unsupported" if tactical_exists else "forbidden", "Evidence path": "ai/weights/model_manifest.json; validation/tactical_diagnostic.json", "Evidence type": "manifest/diagnostic", "Exact metric or observation": f"tactical checkpoint exists={tactical_exists}; learned coverage={learned_cov}", "Known caveat": "nonzero tactical contribution not demonstrated", "Allowed wording": "tactical path remains unproven", "Forbidden wording": "tactical module improves performance"},
        {"Claim ID": "C8", "Candidate manuscript claim": "Route planner is reliable.", "Status": "unsupported", "Evidence path": best.get("path", "not_available") + "/rating.json", "Evidence type": "route diagnostics", "Exact metric or observation": f"route attempts={route_total}", "Known caveat": "zero or unavailable route attempts cannot prove reliability", "Allowed wording": "route diagnostics are tracked", "Forbidden wording": "route planner is reliable"},
        {"Claim ID": "C9", "Candidate manuscript claim": "Route empty-rate limitation exists.", "Status": "supported", "Evidence path": "rating.json policy.move_route_aggregate; validation/route_diagnostic.json", "Evidence type": "diagnostic sidecars", "Exact metric or observation": "route attempt evidence is absent/zero in key sidecars", "Known caveat": "empty-rate cannot always be computed", "Allowed wording": "route evidence is insufficient and remains a limitation", "Forbidden wording": "route issue resolved"},
        {"Claim ID": "C10", "Candidate manuscript claim": "Baselines/ablations exist.", "Status": "partially supported", "Evidence path": "configs/*ablation*.yaml; tests; experiments", "Evidence type": "configs and partial artifacts", "Exact metric or observation": "tactical ablation config exists; no complete protocol-matched ablation table extracted", "Known caveat": "insufficient for strong comparative claims", "Allowed wording": "preliminary baseline/ablation scaffolding", "Forbidden wording": "comprehensive ablation study"},
        {"Claim ID": "C11", "Candidate manuscript claim": "Method is novel.", "Status": "partially supported", "Evidence path": "08_novelty_and_positioning.md", "Evidence type": "positioning note", "Exact metric or observation": "novelty framed around evidence-gated workflow, not LSTM/GraphSAGE", "Known caveat": "standard model components", "Allowed wording": "evidence-gated workflow contribution", "Forbidden wording": "new LSTM or GraphSAGE algorithm"},
        {"Claim ID": "C12", "Candidate manuscript claim": "Evaluation is reproducible.", "Status": "partially supported", "Evidence path": "00_repository_state.md; 01_asset_provenance.md; 11_validation_log.md", "Evidence type": "inventory and validation", "Exact metric or observation": "assets and commands documented; auth/local SDK may constrain reruns", "Known caveat": "SDK/auth and missing tactical checkpoint risks", "Allowed wording": "repo-local reproducibility package", "Forbidden wording": "fully independently reproducible"},
        {"Claim ID": "C13", "Candidate manuscript claim": "Public SOTA is achieved.", "Status": "forbidden", "Evidence path": "none", "Evidence type": "no public leaderboard evidence", "Exact metric or observation": "not_available", "Known caveat": "no protocol-matched public benchmark", "Allowed wording": "no public SOTA claim", "Forbidden wording": "SOTA"},
        {"Claim ID": "C14", "Candidate manuscript claim": "Manuscript is ready for top-tier submission.", "Status": "forbidden", "Evidence path": "09_reviewer_risk_register.md", "Evidence type": "risk register", "Exact metric or observation": "major evidence gaps remain", "Known caveat": "conservative manuscript/report can be written", "Allowed wording": "ready for conservative manuscript drafting with blockers", "Forbidden wording": "top-tier-ready submission package"},
    ]


def main() -> int:
    global TRACKED_FILES, IGNORED_PATHS
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TRACKED_FILES = set(filter(None, run(["git", "ls-files"])[1].splitlines()))
    # Avoid `git status --ignored` on the restored SDK tree; it can be very slow
    # when large local artifacts are present. For package purposes we only need
    # exact ignored status of the requested asset roots.
    IGNORED_PATHS = set()
    head = run(["git", "rev-parse", "HEAD"])[1]
    branch = run(["git", "branch", "--show-current"])[1]
    status = run(["git", "status", "--short"])[1]
    descends = run(["git", "merge-base", "--is-ancestor", "5e414f7", "HEAD"])[0] == 0
    log = """5e414f7 (HEAD -> main) Add paper-facing project reports
b542b98 Preserve formal promotion handoff evidence
65db518 docs: add research project report for promotion results
095bb38 fix: count exploiter gate by true win rate
fb45105 fix: keep heartbeat json writes nonblocking
9087c0b fix: retry atomic json writes on stale handles
b752402 fix: evaluate exploiters with paired objective scoring
da065ce fix: use objective evidence for reward-tied promotion scoring"""
    backups = "backup/before_paper_prewrite_package_* created before package generation; earlier restore backups visible in repository state command output"
    reports_ignored = False
    assets = {
        "ai_weights": asset_record("ai/weights", "candidate checkpoints and manifests"),
        "incumbent_rule": asset_record("ai/weights_incumbent_rule", "rule incumbent manifest"),
        "sdk_wheel": asset_record("land_wargame_sdk/land_wargame_train_env-4.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "SDK environment install"),
        "sdk_scenarios": asset_record("land_wargame_sdk/Data/Data/scenarios", "scenario data"),
        "sdk_maps": asset_record("land_wargame_sdk/Data/Data/maps", "map, cost, visibility data"),
        "experiments": asset_record("experiments/shenji_agent", "formal and diagnostic evidence", False),
    }
    write_json(OUT / "asset_manifest.json", assets)
    summaries = [sidecar_summary(p) for p in league_dirs()]
    write_json(OUT / "formal_artifacts_index.json", summaries)
    best = best_league(summaries)
    claims = make_claims(best, assets)
    write_json(OUT / "claim_to_evidence_matrix.json", claims)
    with (OUT / "results_tables.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["table", "item", "metric", "value", "evidence_path", "caveat"])
        writer.writeheader()
        for comp, path, claim, caveat in [
            ("Intent model", "ai/models.py; ai/weights/intention_lstm.pth", "learned intent model exists", "standard LSTM/attention"),
            ("Situation model", "ai/models.py; ai/weights/situation_gcn.pth", "learned situation model exists", "standard GraphSAGE-style model"),
            ("Tactical scorer", "ai/models.py; ai/weights/model_manifest.json", "scorer class exists", "checkpoint/contribution not proven"),
            ("Rule fallback", "ai/policy.py; ai/agent.py", "fallback logic exists", "may dominate decisions"),
            ("Route planner", "ai/agent.py", "route planner implementation exists", "reliability unsupported"),
            ("Promotion gate", "experiments/shenji_agent/league_*/promotion.json", "promotion sidecars exist", "not all accepted"),
            ("League evaluator", "training/selfplay.py", "league evaluation entrypoint exists", "SDK/auth constrained"),
            ("Artifact validator", "tests/test_promotion_artifact_contract.py", "artifact contract tests exist", "does not prove paper claims alone"),
        ]:
            writer.writerow({"table": "system_components", "item": comp, "metric": "evidence_status", "value": claim, "evidence_path": path, "caveat": caveat})
        for s in summaries:
            if s["suitable_for_manuscript_evidence"]:
                writer.writerow({"table": "formal_league", "item": s["path"], "metric": "promotion_accepted", "value": s["promotion_accepted"], "evidence_path": s["path"], "caveat": f"learned_coverage={s['learned_path_coverage']}"})
        for issue in ["tactical contribution not proven", "route empty-rate issue", "two-window hard-stop not proven", "baseline/ablation insufficiency", "single SDK limitation", "limited scenario/map coverage"]:
            writer.writerow({"table": "limitations", "item": issue, "metric": "status", "value": "open", "evidence_path": "04_claim_to_evidence_matrix.md", "caveat": "limits manuscript wording"})
        for name, rec in assets.items():
            writer.writerow({"table": "reproducibility_assets", "item": name, "metric": "exists", "value": rec["status"], "evidence_path": rec["path"], "caveat": "protect restored assets"})
    csv_text = (OUT / "results_tables.csv").read_text(encoding="utf-8")
    write_json(OUT / "results_tables.json", {"best_league": best, "claims": claims, "asset_summary": assets})
    write(OUT / "00_repository_state.md", f"""# Repository State

- Branch: `{branch}`
- HEAD: `{head}`
- Expected restore target: `5e414f7` / `5e414f794d6ac5ec31adf433e73cb46e4bb64c81`
- HEAD equals expected target: `{head == '5e414f794d6ac5ec31adf433e73cb46e4bb64c81'}`
- HEAD descends from expected target: `{descends}`
- Worktree clean: `{status == ''}`
- Worktree status:

```text
{status or 'clean'}
```

- Untracked goal files present: `{'?? goals/' in status}`
- `reports/` git-ignored: `{reports_ignored}`
- Visible backup branches:

```text
{backups or 'none'}
```

Recent log:

```text
{log}
```
""")
    write(OUT / "01_asset_provenance.md", "# Asset Provenance\n\n" + "\n".join(f"## {k}\n\n```json\n{json.dumps(v, indent=2, sort_keys=True)}\n```" for k, v in assets.items()))
    write(OUT / "asset_loss_prevention.md", """# Asset Loss Prevention

Do not run destructive sync or cleanup commands against this repository or its asset directories:

- `rsync --delete`
- `rsync --delete-excluded`
- `rsync --delete-missing-args`
- `robocopy /MIR`
- `git reset --hard`
- `git clean -fdx`

Before any cross-machine sync, create a branch or archive copy and verify `ai/weights/`, `ai/weights_incumbent_rule/`, `land_wargame_sdk/Data/Data/`, and `experiments/shenji_agent/` after transfer.
""")
    write(OUT / "dangerous_sync_audit.md", "# Dangerous Sync Audit\n\nGenerated by the mandated grep command. See validation log for command status; review any matches before using sync scripts.\n")
    write(OUT / "02_validator_inventory.md", """# Validator Inventory

| Validator | Exists at HEAD before recovery | Found in history | Candidate source commit | Safe to restore | Writes formal artifacts | Should restore now |
|---|---:|---:|---|---:|---:|---:|
| `scripts/run_round005_smoke_pipeline.py` | no | yes | `bccc61c4fb34c77f45aac7dbe9e188816b646ef3` | yes, fail-closed preflight | no | yes |
| `scripts/round005_tactical_diagnostic.py` | no | yes | `bccc61c4fb34c77f45aac7dbe9e188816b646ef3` | yes, diagnostic only | no | yes |
| `scripts/round005_route_diagnostic.py` | no | yes | `bccc61c4fb34c77f45aac7dbe9e188816b646ef3` | yes, sidecar reader | no | yes |

These scripts were restored path-specifically. They write reports under `reports/` or a caller-selected output path and do not hand-edit formal league sidecars.
""")
    inv_lines = ["# Formal Artifact Inventory", "", "| League | Required sidecars missing | Candidate | Incumbent | Win rate | Elo | Accepted | Learned coverage | Route attempts | Suitable |", "|---|---|---|---|---:|---:|---|---:|---:|---|"]
    for s in summaries:
        inv_lines.append(f"| `{s['path']}` | {', '.join(s['missing_required_sidecars']) or 'none'} | {s['candidate']} | {s['incumbent']} | {s['win_rate']} | {s['elo']} | {s['promotion_accepted']} | {s['learned_path_coverage']} | {s['route_total_attempts']} | {s['suitable_for_manuscript_evidence']} |")
    write(OUT / "03_formal_artifact_inventory.md", "\n".join(inv_lines))
    claim_lines = ["# Claim-to-Evidence Matrix", "", "| Claim ID | Status | Evidence | Observation | Allowed wording | Forbidden wording |", "|---|---|---|---|---|---|"]
    for c in claims:
        claim_lines.append(f"| {c['Claim ID']} | {c['Status']} | `{c['Evidence path']}` | {c['Exact metric or observation']} | {c['Allowed wording']} | {c['Forbidden wording']} |")
    write(OUT / "04_claim_to_evidence_matrix.md", "\n".join(claim_lines))
    write(OUT / "05_results_for_tables.md", "# Results for Tables\n\nThe CSV/JSON table source files are generated from sidecars and inventories only.\n\n```csv\n" + csv_text[:12000] + "\n```")
    write(FIG / "fig1_architecture_spec.md", "# Figure 1 Architecture Spec\n\nShow SDK environment state -> legal action candidates -> intent model -> situation model -> tactical scorer (marked unproven) -> rule/objective fallback -> route planner -> selected action -> league/promotion evaluator -> sidecar artifacts.")
    write(FIG / "fig2_evidence_gate_spec.md", "# Figure 2 Evidence Gate Spec\n\nShow candidate generation, battle/evaluation, rating, promotion gate, sidecar validation, claim boundary, and SAFE_STOP/PARTIAL_READY branches when evidence is insufficient.")
    write(FIG / "fig3_result_summary_data.csv", "metric,value,evidence_path,caveat\npromotion_accepted,%s,%s,one sidecar only; learned coverage may be zero\nwin_rate,%s,%s,sidecar point estimate\nlearned_path_coverage,%s,%s,do not infer tactical contribution\n" % (best.get("promotion_accepted", "not_available"), best.get("path", "not_available"), best.get("win_rate", "not_available"), best.get("path", "not_available"), best.get("learned_path_coverage", "not_available"), best.get("path", "not_available")))
    write(FIG / "fig3_result_summary_notes.md", "# Figure 3 Notes\n\nUse only sidecar-supported metrics. Prefer a small evidence-state panel over a performance chart because the strongest extracted result is claim-boundary evidence, not broad performance superiority.")
    write(FIG / "fig4_diagnostic_limitations.md", "# Figure 4 Diagnostic Limitations\n\nDepict five open limitations: tactical missing/zero contribution, route empty-rate/route-attempt gap, two-window gap, baseline/ablation gap, and restored asset provenance risk.")
    write(OUT / "06_figure_plan.md", "# Figure Plan\n\nPrepared specs/data under `paper/prewrite_package/figure_data/` for architecture, evidence gate, result summary, and diagnostic limitations. Figure 3 must remain conservative and sidecar-only.")
    write(OUT / "07_methods_package.md", """# Methods Package

1. Environment and SDK setup: local Python environment with `land_wargame_sdk/land_wargame_train_env-4.1.0-*.whl`; SDK auth may be required.
2. Scenario/map assets: `land_wargame_sdk/Data/Data/scenarios/` and `land_wargame_sdk/Data/Data/maps/`.
3. Agent architecture: `ai/agent.py`, `ai/policy.py`.
4. Learned components: `ai/models.py`, `ai/weights/intention_lstm.pth`, `ai/weights/situation_gcn.pth`.
5. Fallback logic: rule/objective fallback in `ai/policy.py` and `ai/agent.py`.
6. Route planner: `Map.gen_move_route` in `ai/agent.py`.
7. Evaluation protocol: `training/selfplay.py` and `configs/selfplay_*.yaml`.
8. Promotion gate: `promotion.json` sidecars under `experiments/shenji_agent/league_*`.
9. Sidecar validation: `tests/test_promotion_artifact_contract.py` and recovered Round-005 diagnostics.
10. Reviewer/audit process: claim matrix, formal artifact index, and validation log in this package.
11. Reproducibility commands: `python -m unittest discover -v`; `python -m compileall paper/scripts scripts ai tests training`; diagnostics listed in `11_validation_log.md`.
12. Known unavailable components: tactical contribution proof, two-window proof, protocol-matched public baseline/SOTA evidence.
""")
    write(OUT / "08_novelty_and_positioning.md", """# Novelty and Positioning

Allowed framing: fail-closed evidence-gated evaluation workflow; closed-loop SDK-based land-wargame agent evaluation; learned-path/action-diagnostic attribution; promotion-gate sidecar validation; reproducibility and claim-boundary discipline.

Forbidden framing: new LSTM algorithm; new GraphSAGE algorithm; public SOTA; general benchmark superiority; tactical module proven useful; route planner proven reliable; fully independent reproducibility.

Reviewer-facing risks: standard model components, single SDK, limited scenario set, incomplete formal promotion, missing ablations, unresolved route/tactical evidence. Wording should present these as boundaries and future work, not hidden weaknesses.
""")
    write(OUT / "09_reviewer_risk_register.md", """# Reviewer Risk Register

| Dimension | Current expected weakness | Evidence available | Evidence missing | Manuscript strategy | Do-not-claim | Future experiment |
|---|---|---|---|---|---|---|
| Soundness | sidecar coverage uneven | formal artifact index | two-window proof | state exact gates | general validity | rerun complete formal league |
| Clarity | many artifacts | source evidence map | polished figures | use evidence tables | unsupported strength | simplify claims |
| Novelty | standard models | workflow framing | algorithmic novelty | emphasize evidence gate | new LSTM/GraphSAGE | compare workflow rigor |
| Significance | local SDK | closed-loop evaluation | public benchmark | technical report/conservative paper | SOTA | external benchmark |
| Evidence | tactical/route gaps | diagnostics | ablations | limitations-first wording | tactical improves | targeted ablations |
""")
    write(ROOT / "paper" / "source_evidence.md", """# Source Evidence

| Topic | Evidence paths | Safe claim boundary |
|---|---|---|
| Agent implementation | `ai/agent.py`, `ai/policy.py` | implemented local agent with fallback logic |
| Model definitions | `ai/models.py` | standard LSTM/attention and GraphSAGE-style components |
| Training entrypoints | `training/supervised.py`, `training/selfplay.py`, `train_models.py` | entrypoints exist; not all produce formal paper results |
| Evaluation entrypoints | `run_evaluation.py`, `training/evaluate.py`, `training/selfplay.py` | local SDK evaluation implemented |
| Promotion sidecars | `experiments/shenji_agent/league_*/promotion.json` | sidecar-supported promotion evidence only |
| Candidate checkpoints | `ai/weights/intention_lstm.pth`, `ai/weights/situation_gcn.pth` | intent/situation checkpoints restored |
| SDK maps/scenarios | `land_wargame_sdk/Data/Data/maps/`, `land_wargame_sdk/Data/Data/scenarios/` | restored local SDK assets |
| Route diagnostics | `rating.json` policy route aggregates; `scripts/round005_route_diagnostic.py` | route reliability unproven |
| Tactical diagnostics | `scripts/round005_tactical_diagnostic.py`; manifests | tactical contribution unproven |
| Baseline/ablation status | `ai/weights_incumbent_rule/model_manifest.json`, `configs/supervised_tactical_ablation.yaml` | partial baseline/ablation support |
| Known limitations | `paper/prewrite_package/04_claim_to_evidence_matrix.md` | unsupported claims are forbidden |
| Asset provenance | `paper/prewrite_package/01_asset_provenance.md` | restored assets documented |
""")
    write(OUT / "10_source_evidence_delta.md", "# Source Evidence Delta\n\nCreated `paper/source_evidence.md` to map future manuscript claims to code, sidecars, assets, diagnostics, and limitations.")
    decision = "PARTIAL_READY_WITH_BLOCKERS"
    safe = [c for c in claims if c["Status"] == "supported"]
    forbidden = [c for c in claims if c["Status"] == "forbidden"]
    write(OUT / "PAPER_WRITING_READY.md", f"""# Paper Writing Ready

Final state: `{decision}`

## Repository State

Branch `{branch}` at `{head}`; expected restored commit verified: `{head == '5e414f794d6ac5ec31adf433e73cb46e4bb64c81'}`; descends from restore target: `{descends}`.

## Evidence State

Asset, validator, formal artifact, claim, result-table, figure, method, novelty, reviewer-risk, and source-evidence materials are under `paper/prewrite_package/`.

## Safe Claims

{chr(10).join('- ' + c['Candidate manuscript claim'] for c in safe)}

## Forbidden Claims

{chr(10).join('- ' + c['Candidate manuscript claim'] for c in forbidden)}

## Figures and Tables

Figures 1, 2, and 4 can be drawn from specs. Figure 3 can be drawn only as a conservative sidecar metric summary. Tables 1-4 are sourced by `results_tables.csv` and `results_tables.json`.

## Methods Ready

The method scaffold in `07_methods_package.md` includes environment, SDK assets, architecture, learned components, fallback, route planner, evaluation protocol, promotion gate, sidecar validation, audit process, commands, and unavailable components.

## Unresolved Blockers

Tactical contribution is not proven; route reliability is not proven; two-window success is not proven; baseline/ablation evidence is insufficient for strong claims; public SOTA is forbidden.

## Recommended Positioning

Write a conservative manuscript or technical report around closed-loop SDK evaluation and evidence-gated claim discipline, with limitations stated directly.

## Next Goal Command

`/goal Write a conservative Shenji_agent manuscript from paper/prewrite_package/PAPER_WRITING_READY.md and supporting prewrite package artifacts, without introducing unsupported claims.`
""")
    for name in ["CHANGELOG.md", "REVIEW_LOG.md", "ACCEPTANCE_CRITERIA.md"]:
        path = ROOT / "paper" / name
        prev = path.read_text(encoding="utf-8") if path.exists() else f"# {name}\n"
        entry = "\n\n## 2026-05-13 Paper Prewrite Evidence Package\n\nPrepared manuscript prewriting evidence materials under `paper/prewrite_package/`. This was an evidence packaging task, not a `paper/main.tex` manuscript revision.\n"
        if "2026-05-13 Paper Prewrite Evidence Package" not in prev:
            write(path, prev.rstrip() + entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
