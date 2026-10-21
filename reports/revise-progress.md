# Revise progress — review pass 1 (2026-10-21)

Checklist for `reviews/2026-10-21-pass-1.md`. Status: todo · in-progress · done · partly · skipped (+ one line why).
Orchestrator: Fable. Subagents: sonnet/haiku, opus only where marked.

## Phases

| Phase | What | Status | Note |
|---|---|---|---|
| 0 | Callout system (N-034): build.py + lint + SPEC L-001/L-003 + convert all 13 sections | done | gate passed (14 tests); 44 boxes + banner converted, run-only cells collapsed, 3 violins |
| 1 | R-01..R-29 on sections 0–5 (4 subagents: 00+01 · 02+03 · 04 · 05) | in-progress | brief in /tmp/eda-phase-brief.md; gate serialised by /tmp/eda-gate.sh |
| 2 | R-30: N-034..N-049 on sections 6–12 (3 subagents) | in-progress | running concurrently with Phase 1 (disjoint files, gate serialised) |
| 3 | Three read-only reviewers (professor · student · cuts[opus]) | todo | |
| 4 | Apply non-removal suggestions; write reports/cut-proposals.md | todo | |
| 5 | just check + reports/revise-2026-10-21.md | todo | |

## R-items

| Item | Status | Note |
|---|---|---|
| R-01 cut boxes/banner | done | Phase 0: callouts are HTML outputs of hidden cells |
| R-02 collapse run-only cells | todo | |
| R-03 Objective callout | todo | |
| R-04 Takeaway callout, When→do lists | todo | |
| R-05 remove licence text, data/ATTRIBUTION.md | todo | |
| R-06 hide _ref_ solutions | todo | |
| R-07 banner kept | done | content unchanged except "broken files" dropped (R-28) |
| R-08 s01 not a wall of text | todo | |
| R-09 s01 example per checklist point | todo | |
| R-10 s01 labels present & validated, shown | todo | |
| R-11 s01 grid before Q1 | todo | |
| R-12 s02 three 👀 as list | todo | |
| R-13 s02 Your turn box cut | done | Phase 0 |
| R-14 s03 extremes beside stats table | todo | |
| R-15 s03 define every statistic | todo | |
| R-16 s03 Q4 tiny example plots | todo | |
| R-17 s03 grayscale examples | todo | |
| R-18 violin plots (03, 08, 09) | todo | |
| R-19 s03 median photo in extremes | todo | |
| R-20 s03 rewrite two cryptic passages | todo | |
| R-21 s03 Across tasks box cut | done | Phase 0 |
| R-22 s03 Q6 concrete numbers | todo | |
| R-23 s03 bilinear mask heatmap + formula | todo | |
| R-24 s04 Q9 reformulate + fake-model grid | todo | |
| R-25 s04 collapse cell under Q9 | todo | |
| R-26 s05 perceptual hashing shown | todo | |
| R-27 s05 memory-card hunt as steps + visual | todo | |
| R-28 s05 remove Broken and odd files | todo | |
| R-29 s05 Q14 clear or propose cut | todo | |
| R-29b s05 rewrite "six cells" sentence | todo | |
| R-29c s05 rewrite Across tasks box | todo | |
| R-30 sections 6–12 | todo | Phase 2 |
| R-31 shorter overall | todo | Phases 3–4 |
| R-32 interaction proposals (≤3) | todo | Phase 3 |

## Log
- 2026-10-21: fresh run started. Baseline `just check` passed (14 tests, 22 s).
- Phase 0a done: scripts/build.py (markdown→HTML callouts, saved output), lint N-034/N-035/N-040, SPEC L-001/L-003, tests/test_content.py callout test, 00 banner. New dev dep `markdown`.
- Created data/ATTRIBUTION.md and a credits line in README.md (R-05 repo side).
- Phase 1: 00+01, 04, 05 agents finished (subagent Bash was denied mid-run, orchestrator runs the gate: lint/ruff/smoke pass). 02+03 running. Phase 2: 10–12 finished (gate pass); 06+07, 08+09 running.
- Note for the report: tests/test_content.py changed only in Phase 0 (callout test); the 05 agent made no change.
- Open from Phase 0a: scripts/split.py (`just sync`) does not map callout code cells back to markers; 00's four-locations table is a plain table (may be cut by Colab).
