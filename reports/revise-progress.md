# Revise progress — review pass 1 (2026-10-21)

Checklist for `reviews/2026-10-21-pass-1.md`. Status: todo · in-progress · done · partly · skipped (+ one line why).
Orchestrator: Fable (run 1), Opus (runs 2–3). Subagents: sonnet/haiku, opus only where marked.

## Phases

| Phase | What | Status | Note |
|---|---|---|---|
| 0 | Callout system (N-034): build.py + lint + SPEC L-001/L-003 + convert all 13 sections | done | gate passed (14 tests); 44 boxes + banner converted, run-only cells collapsed, 3 violins |
| 1 | R-01..R-29 on sections 0–5 (4 subagents: 00+01 · 02+03 · 04 · 05) | done | run 2: finished/audited by 3 agents; gate passed (14 tests) |
| 2 | R-30: N-034..N-049 on sections 6–12 (3 subagents) | done | run 2: 06+07, 08+09 finished, 10–12 audited; gate passed |
| 3 | Three read-only reviewers (professor · student · cuts[opus]) | done | reports/review-{professor,student,cuts}.md |
| 4 | Apply non-removal suggestions; write reports/cut-proposals.md | done | 6 sonnet agents (00+01 · 02+03 · 04+05 · 06+07 · 08+09 · 10–12); cut-proposals.md has 33 rows |
| 5 | just check + reports/revise-2026-10-21.md | done | run 3: gate passed (35 Q, 14 tests), drift clean |

## R-items

| Item | Status | Note |
|---|---|---|
| R-01 cut boxes/banner | done | Phase 0: callouts are HTML outputs of hidden cells |
| R-02 collapse run-only cells | done | all 13 sections |
| R-03 Objective callout | done | all 13; 00 heading moved to cell 0 in run 2 |
| R-04 Takeaway callout, When→do lists | done | all 13; 00 added, 05 reworded in run 2 |
| R-05 remove licence text, data/ATTRIBUTION.md | done | also hides `license` column in 00's `images.head()` |
| R-06 hide _ref_ solutions | done | 00, 05, 06, 08; 10's scorer stays after `findings` (would leak score) |
| R-07 banner kept | done | content unchanged except "broken files" dropped (R-28) |
| R-08 s01 not a wall of text | done | |
| R-09 s01 example per checklist point | done | 10 points, each with a park example |
| R-10 s01 labels present & validated, shown | done | one photo with class/box/mask + spot-check |
| R-11 s01 grid before Q1 | done | |
| R-12 s02 three 👀 as list | done | |
| R-13 s02 Your turn box cut | done | Phase 0 |
| R-14 s03 extremes beside stats table | done | |
| R-15 s03 define every statistic | done | |
| R-16 s03 Q4 tiny example plots | done | |
| R-17 s03 grayscale examples | done | |
| R-18 violin plots (03, 08, 09) | done | |
| R-19 s03 median photo in extremes | done | |
| R-20 s03 rewrite two cryptic passages | done | |
| R-21 s03 Across tasks box cut | done | Phase 0 |
| R-22 s03 Q6 concrete numbers | done | |
| R-23 s03 bilinear mask heatmap + formula | done | |
| R-24 s04 Q9 reformulate + fake-model grid | done | |
| R-25 s04 collapse cell under Q9 | done | |
| R-26 s05 perceptual hashing shown | done | |
| R-27 s05 memory-card hunt as steps + visual | done | |
| R-28 s05 remove Broken and odd files | done | section, banner, SPEC, FACILITATION, 10/11 wording |
| R-29 s05 Q14 clear or propose cut | done | clear in two cells; also a cut proposal |
| R-29b s05 rewrite "six cells" sentence | done | |
| R-29c s05 rewrite Across tasks box | done | |
| R-30 sections 6–12 | done | 09 N-042 mask-check visual not added (offered) |
| R-31 shorter overall | partly | text tightened (~−27 min); removals wait in cut-proposals.md |
| R-32 interaction proposals (≤3) | done | 3 slider proposals in the report, nothing built (G-12) |

## Log
- 2026-10-21: fresh run started. Baseline `just check` passed (14 tests, 22 s).
- Phase 0a done: scripts/build.py (markdown→HTML callouts, saved output), lint N-034/N-035/N-040, SPEC L-001/L-003, tests/test_content.py callout test, 00 banner. New dev dep `markdown`.
- Created data/ATTRIBUTION.md and a credits line in README.md (R-05 repo side).
- Phase 1: 00+01, 04, 05 agents finished (subagent Bash was denied mid-run, orchestrator runs the gate: lint/ruff/smoke pass). 02+03 running. Phase 2: 10–12 finished (gate pass); 06+07, 08+09 running.
- Note for the report: tests/test_content.py changed only in Phase 0 (callout test); the 05 agent made no change.
- Open from Phase 0a: scripts/split.py (`just sync`) does not map callout code cells back to markers; 00's four-locations table is a plain table (may be cut by Colab).
- Run 2 (orchestrator: Opus, resumed). State found: run 1 committed (9d007c7) with 02+03, 06+07, 08+09 mid-edit; `just check` fails on ruff E501 in 03, 06, 07. Subagent reports from run 1 are lost, so R-item status is re-audited. Gate now run by the orchestrator (`just check`, /tmp is outside the sandbox). Relaunch: 02+03 · 06+07 · 08+09 (finish) and audits 00+01+04+05 · 10–12.
- Run 2 · 06+07 agent done (ruff+lint pass, smoke not yet): all conventions done; E501 fixed; Q15 output now relative drift; Q20 `_QUIZ_KEY` moved into a collapsed check cell; 07 "~6 % pixels" → "few pixels". Cuts: 06 🔴 Q18 ruler ~5 min; 06 LOO-vs-LOGO cell ~2; 07 "different shop" block ~3; 07 Q21 IoU/Dice figure → Go deeper ~2. Facil: 06 start trail-camera download early, pause on Q17; 07 toy matrices 98.0 vs 98.4 % accuracy, recall 0→14/20, compare gap not digits.
- Run 2 · 02+03 agent done (ruff+lint pass): R-02..06, R-12, R-14..R-23 done, R-31 partly. Q6 inputs 640×427, box (100,50,200,150) → 224×224; bilinear heatmap + 6×6 zoom + formula + cat/dog answer. Renamed Q6 `_photo`→`_sample` (shadowed helper). Cuts: 03 Q5 odd-size gallery ~1 min; 03 6×6 zoom ~1. Facil: 02 ~5–6 min, look before reading answers; 03 heaviest, spend time on min/median/max figure, Q6 longest stop.
- Run 2 · 10–12 agent done (ruff+lint pass): all conventions already done; broken-files wording fixed in 10 (title) and 11 (Q5 park answer, Q4 "(section 3)"); hint "(`False` for a file PIL cannot decode)" for `is_readable`. Decision: section 10 `_reference` scorer stays after `findings` (it is a scorer, moving it would leak the score). Cut: 10 🔴 ruler Q33 ~3 min. Facil: 10 plan Q31 first; 11 sample first, Q35 homework if short; 12 read the three takeaways.
- Run 2 · 00+01 agent done (ruff+lint pass): R-02, R-06..R-11 done; fixed now: 00 heading moved to cell 0 before objective (R-03), 00 takeaway added (R-04), `images.head()` → `images.drop(columns="license").head()` (R-05). R-08: V7 duplicate merged (7 sources). R-11: grid now before Q1. Four-locations "table" is a list in a theory callout (open item from Phase 0 resolved). FACILITATION row 1 still mentions the licence and a two-column table → orchestrator fixes. Cut: fold label-your-own list ~0.5 min.
- Run 2 · 08+09 agent done (ruff+lint pass): 08 already complete; 09 finished (marker lists, 4 cells split, COCO-ReM → go-deeper, fill gallery lowest/median/highest, new "always background" pitfall demo, thin-parts 3-step scenario, takeaway). Unverified numbers: 08 "sheep 9 % photos / 22 % boxes", 09 "85 % background / bird ~1 %" (and 07 had "~6 %"). Not done: N-042 visual for the four mask checks (offered). Cuts: 09 🔴 Q30 ~3 min; 08 box-centre heatmap ~1.5. Facil: 8 missing-box hunt ~2 min; 9 pause at pitfall demo, try ✏️ thresholds once, skip Q30 first.
- Run 2 · 04+05 agent done (ruff+lint pass): all R-items done (fixed now: 05 takeaway wording, 05 cell 0 trimmed). R-24 strip shows 92 % at default DOG_SHARE (12 frames hold ≥1 dog), printed paddock-day line gives 98 %. R-28: `is_readable`/`flag_odd_files` kept (cleaning + flags used by 10, 11); no test asserted on the removed part. R-29: Q14 clear in two cells, worked answer moved out of `<details>` (L-005). Atypicality defined in one line. Cuts: Q14 ~3 min; Kaggle stories ~1; 04 clicks table ~2; Q13 ~2. Facil: 4 ask guess before Q9, try DOG_SHARE 0.2/0.5; 5 pause at 64-bit figure, predict Q12 before copies figure, Q14 optional.
- Run 2 · gate after Phases 1–2: `just check` passed (lint 0 problems, 35 Q, 14 tests). Phase 3 reviewers launched (professor sonnet · student sonnet · cuts opus).
- Numbers checked on data/coco_animals_subset.json (638 photos, 1550 boxes, before cleaning): sheep 9.2 % photos / 21.9 % boxes ✓; background ≈ 85 % of pixels ✓; bird 1.0 % of pixels ✓ (07's old "~6 %" was wrong; already replaced by "few pixels").
- Phase 3 done. Estimates: professor ~160 min, cuts 244 careful / 190 lab (core path), vs 90 slot and SPEC 110. Phase 4 declined (with reason, for the report): merging ❓ callout with **Qn.** text (questions inside callouts would leave markdown → lint/numbering risk; needs a build change); shortening 01 checklist examples (R-09 asked them longer); folding 05 🔀 box (R-29c asked it visible); replacing 04 Karpathy quote (needs source check); cross-file moves (04 clicks → 08, 07 IoU/mAP → 08) go to cut-proposals.
- Phase 4 · 00+01 done (ruff+lint): prediction rule + bring-your-data in "Before you start" (now a theory callout); 00 reordered: heading → objective → banner → "run the next three cells now" → Setup/data download → story → `double()` demo (banner-first rejected by lint: cell 0 heading + cell 1 objective); park callouts merged; 00 takeaway 2 lines; ref title "run this cell; open it only if you are stuck"; 01 label example + grid before checklist; link lists → one visible go-deeper before takeaway; "weighted loss → next week". 01 checklist kept verbatim (R-09).
- Phase 4 · 02+03 done (ruff+lint): 💡+answer merged (02 Q2, 03 Q4–Q6); openers cut; Q3 answer 3 lines; ✏️ look-for goals; min/median/max 2 stats per row (~10 in tall, was 17); violin sentence; bilinear formula + "why not round" in a visible go-deeper; Q6 (a)/(b); Q4 "closest to a line: two lines in an L"; section-5 promise reworded to the code's truth (all converted to RGB; `flags` marks non-RGB modes + extreme sizes, grayscale-looking RGB not flagged); shortcut sentence; heading split. Facil: Q4 L-shape stumble.
- Phase 4 · 10–12 done (ruff+lint): ten kinds in one list (five steps, ten entries); tools list → recipe kind; Q32 one header per kind (only the worked unreadable check solved); 10 opening down to 3 cells; 🔴 Q33 heading; 11 numbers "about 130 cats, about 50 sheep" (raw data: 130 / 59) and "can halve the error" softened; Q34 think-only; 12 augmentation+tracking in one go-deeper; 10 `_reference` title renamed. Facil: 10 walk Q32 headers 1–3 together, run the reference cell.
- Phase 4 · 04+05 done (ruff+lint): 💡+answer merged; 04 opener cut; Pitfall moved after Q9 and merged with its answer; Q9 split into Rule A (biggest class, Play Dataset) / Rule B (always sheep, paddock); DOG_SHARE/HASH_THRESHOLD say which cell to re-run (and reset to 8); "1 : 10 : 100 seconds" → clicks only; 05 DCT in plain words (Σ formula dropped, figure kept); orientation sentence moved after Q12; ✍️ flip detector now before the shoot-out; N_ATYPICAL 2; section-5 flags line matches 03; headings split; 🔴 Q14 heading `### 🔴 Optional · …`. Facil: 04 guess Rule A and B separately; 05 write the detector ~5 min before the shoot-out.
- Phase 4 · 08+09 done (ruff+lint): 💡+answer merged; openers cut; Q25 spoiler fixed (toy figure after the answer, title neutral, step 3 vague); 08 theory callout 4 lines, hunt 3 steps; 🔀 jargon in `<details>` (kind kept); "about 9 %/22 %/85 %"; new collapsed "five checks drawn on one toy mask" figure (uses real `mask_shape()`); five checks, rubber band = convex hull; ✏️ thresholds: try IOU_MIN first + conclusion; polygon/RLE → go-deeper `<details>`; 3 sentences rewritten; 🔴 Q30 heading.
- Run 3 (Opus, resumed after the 12:32 watchdog pause): tree clean at 6a0dfdc. The Phase 4 06+07 work is in the files (fold diagram, one name for LOGO, class × camera table) but was never logged. `just check` passed (lint 0 problems, 35 Q, 14 tests), `just drift` clean. reports/revise-2026-10-21.md was already complete and needed no change. Pass finished.
