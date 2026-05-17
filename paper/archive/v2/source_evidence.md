# Source Evidence for NeurIPS-Style Draft

Generated: 2026-05-11 Asia/Shanghai. Updated: v1 revision cycle.

## Local project evidence

| Claim surface | Source path | Evidence used |
| --- | --- | --- |
| Project truth surfaces and remote roots | `.agent/context.md` | Current research truth surface, remote work root, strongest snapshot path, priority list. |
| Stop-rule boundaries | `program.md` | No protocol-matched public leaderboard; 100 paired map items; promotion thresholds; two-window requirement; no public SOTA claim. |
| Project overview | `README.md` | Miaosuan SDK setting, `TrainEnv` loop, action types, data layout. |
| Paper outline and writing constraints | `reports/paper_report_outline.md` | Contribution framing, result summary, warning not to claim public SOTA, limitation list. |
| Formal result report | `reports/experiment_results.md` | Experiment setup, bucket counts, promotion-gate results, behavior coverage, exploiter results, remaining risks. |
| Latest formal run summary | `experiments/shenji_agent/league_20260501_170204/summary.md` | Backend, 100 paired maps, participant metadata, win rates, coverage, phase and auth evidence. |
| Main metrics | `experiments/shenji_agent/league_20260501_170204/rating.json` | Overall and bucketed point estimates, confidence intervals, policy coverage, score means. |
| Promotion verdict | `experiments/shenji_agent/league_20260501_170204/promotion.json` | `accepted=true`, no failures, Elo lower bound, coverage, exploiter cap, battle-phase and selected-action evidence. |
| Participant artifacts | `experiments/shenji_agent/league_20260501_170204/participants.json` | Candidate/incumbent profiles, artifact directories, source commit SHA values, real backend. |
| Exploiter check | `experiments/shenji_agent/league_20260501_170204/exploitability.json` | Aggressive and occupy exploiter win rates against candidate. |
| Machine learning model definitions | `ai/models.py` | LSTM intent model, GraphSAGE-style situation model, tactical scorer. |
| Runtime policy | `ai/policy.py` | Model loading, feature normalizer, objective diagnostics, learned path coverage counters. |
| **Two-window stability gap** | `experiments/shenji_agent/league_20260501_170204/league_state.json` | `consecutive_windows = 0`. The project hard-stop requires two consecutive windows; only one verified window exists. |
| **Move route analysis** | `experiments/shenji_agent/league_20260501_170204/rating.json` policy.move_route_aggregate | 15,641 / 23,580 empty routes (66%); highest in Maneuver (75%) and Fly (82%), lower in Walk (18%). |
| **Score diagnostics** | `experiments/shenji_agent/league_20260501_170204/rating.json` policy.score_means | Intent bias, situation score, tactical score (0.0), objective score, combined score for move and shoot. |

## Session verification notes

| Check | Result |
| --- | --- |
| Project governance files | `AGENTS.md` and `.agent/context.md` exist and were read. |
| PaperOrchestra skill | `.agents/skills/paper-orchestra` target `SKILL.md` is missing. |
| Paper writing fallback | `.orchestra/skills/20-ml-paper-writing/SKILL.md` was readable and used as the ML paper-writing workflow. |
| NeurIPS template | Bundled local template from `.orchestra/skills/20-ml-paper-writing/templates/neurips2025/` was copied into `paper/`. |
| Git closeout | `git status` available; commit will be made after revision cycle. |
| Local rerun capacity | Current local `training/` has only `cache/`, while `scripts/` and `tests/` are empty. Paper reports existing artifacts, not a fresh training/self-play rerun. |

## External citation verification

| Citation key | Verification source | Retrieval method |
| --- | --- | --- |
| `silver2017alphagozero` | DOI `10.1038/nature24270` | Retrieved BibTeX with `Accept: application/x-bibtex` from DOI resolver. |
| `vinyals2019alphastar` | DOI `10.1038/s41586-019-1724-z` | Retrieved BibTeX with `Accept: application/x-bibtex` from DOI resolver. |
| `perolat2022deepnash` | DOI `10.1126/science.add4679` | Retrieved BibTeX with `Accept: application/x-bibtex` from DOI resolver. |
| `hochreiter1997lstm` | DOI `10.1162/neco.1997.9.8.1735` | Retrieved BibTeX through Crossref transform endpoint after DOI resolver TLS failure. |
| `samvelyan2019smac` | arXiv `1902.04043` | Retrieved BibTeX from `https://arxiv.org/bibtex/1902.04043`. |
| `ellis2023smacv2` | arXiv `2212.07489` | Retrieved BibTeX from `https://arxiv.org/bibtex/2212.07489`. |
| `yin2025wgsrbench` | arXiv `2506.10264` | Retrieved BibTeX from `https://arxiv.org/bibtex/2506.10264`. |
| `hamilton2018graphsage` | arXiv `1706.02216` | Retrieved BibTeX from `https://arxiv.org/bibtex/1706.02216`. |
