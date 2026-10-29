# Versions

One folder per version of the notebook: `SPEC.md` and the built notebook exactly as they were.
The newest folder is always the notebook on `main`. Make one with `just version "what changed"`
before merging to main; `just version-check` fails if `main` has moved on without one.
Never edit a version folder.

| Version | Date | What changed | Commit |
|---|---|---|---|
| v01 | 2026-10-20 | Night run 2026-10-19: all 14 issues built (PR #15) | c3f3413 |
| v02 | 2026-10-21 | Review pass 1: your EDA review applied, sections 0–12 (PR #16) | dbaaf2d |
| v03 | 2026-10-22 | Cut pass 2026-10-22 | 890a2df |
| v04 | 2026-10-22 | Review, cuts and order fixes 2026-10-22 (PR #19) | 42dcb55 |
| v05 | 2026-10-26 | SPEC: made with Susanne Suter (SN rules incl. SN-006), port tooling, walkthrough work list | a262aa5 |
| v06 | 2026-10-28 | Susanne's copy as base, clean-up, Own-Data Edition | 33b96d5 |
| v07 | 2026-10-29 | Review fixes: real datasets, embeddings note, independent text check | 83fdb7e |
