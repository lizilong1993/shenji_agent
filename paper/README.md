# Shenji NeurIPS-Style Paper Draft

This directory contains all paper-related files for the current draft.

## Files

| Path | Purpose |
| --- | --- |
| `main.tex` | NeurIPS-style manuscript draft. |
| `references.bib` | Verified external BibTeX entries only. |
| `neurips.sty` | Bundled local NeurIPS 2025 style file from PaperOrchestra ML paper-writing templates. |
| `extra_pkgs.tex` | Bundled package list used by the local NeurIPS template. |
| `figures/` | LaTeX figure sources used by `main.tex`. |
| `source_evidence.md` | Traceability notes for project artifacts, citations, and session constraints. |

## Build

From this directory:

```powershell
latexmk -pdf -interaction=nonstopmode main.tex
```

The generated PDF is `main.pdf`.

## Known scope boundary

The draft reports the latest verified local artifacts. It does not claim a fresh rerun of training, self-play, or tests in this session because the current local tree lacks the full `training/`, `scripts/`, and `tests/` implementation surfaces referenced by older project context.
