# Agreements check: is everything agreed with Susanne actually in the notebooks? (2026-10-29)

Independent check of branch `fix/results-lowercase` (HEAD `2bff236`) after `just build` (no drift: both `.ipynb`
equal a fresh build). Evidence comes from the built notebooks themselves, read cell by cell with nbformat, and from
`sections/`, `SPEC.md`, `RULES.md`, `data/` and the tests. Commit messages and earlier summaries were not taken as
evidence. Cell numbers are indices into the built notebooks (T = `eda-teaching-notebook.ipynb`, O =
`eda-own-dataset-notebook.ipynb`). Meeting content is paraphrased; no quotes.

Sources of the agreements: the 23 Oct walkthrough (S1–S23, D1–D5), the 26 Oct sync (EDA part), the vault project
page, `reviews/2026-10-23-walkthrough.md`, `reviews/2026-10-28-susanne-copy.md` (P1–P18), the PRD (Q1–Q11), her
written comments on her 28 Oct copy (1–5, as relayed by Simone) and Simone's 28 Oct decisions. Rule of the check:
her 28 Oct copy is the base, so a 23/26 Oct agreement her copy settled differently is "superseded", not a failure.

What I ran: `just build`, `just lint` (lint script, ruff, `test_version`, `test_editions`, `test_her_rules`: 21
passed), `ruff --select N` on `sections/`. I did **not** run `just smoke` (it writes caches into the repo); the smoke
results are therefore not re-verified here.

## Summary

| Verdict | Count |
|---|---|
| DONE | 50 |
| PARTLY | 12 |
| NOT DONE | 4 |
| SUPERSEDED BY HER COPY | 16 |
| NOT APPLICABLE | 3 |
| **Total** | **85** |

The Teaching Edition is in good shape: her text is intact (a fuzzy diff of every markdown cell against her copy
finds only the approved changes), her box set is the only one used, every task heading has its level dot and every
🔴 heading says "(Optional)". The misses are concentrated in the **Own-Data Edition** and in the **capitals rule**:
the Own-Data notebook still carries the coding exercises she asked to leave out, and three N-058 breaches live only
there, where the lint test does not look for them (it allows `CLASSES` twice and never checks in-place changes or
cross-chapter reads).

## NOT DONE and PARTLY

| # | Agreement | Source | What is wrong | Where | Fix |
|---|---|---|---|---|---|
| 49 | Own-Data notebook **without the exercises** | Her comment (4), 28 Oct | Both coding exercises are still there: the "Optional Coding Exercises" block with the `double` demo, and the `count_boxes_per_class` exercise with its "Coding Exercise:" paragraph, ✏️ Your-turn note, reference cell and ✍️ cell | O 10–12 (Setup); O 130–133 (Boxes per Class vs. Photos per Class) | Tag the four exercise cells Teaching-only; in O, compute `boxes_per_class` directly (the reference body, as a plain cell) and drop the "Coding Exercise:" paragraph from the O variant of the task text |
| 38 | A learner setting is read only in its own section (N-058; the exact case that started it was the hash threshold set in the duplicates part and reused in the last chapter) | 26 Oct sync + project page (capitals) | The split suggestion reads `HASH_THRESHOLD`, the slider of the optional Bit Distance Distribution task; the hint below tells the student to change it there and rerun. Same pattern N-058 was written against | O 189 (`find_copies(..., HASH_THRESHOLD, ...)`), O 196; `sections/09_your_dataset.py` | Give the split suggestion its own setting (e.g. `COPY_THRESHOLD = 5` in that cell) and point the hint at it |
| 20 | Audience in SPEC §2 (CS class that knows ML, 7 weeks of classical image processing, first deep-learning CV lecture) | S19, 23 Oct | SPEC §2 still has the v05 audience text, plus v05 lines that are now false (no Colab sliders, ~4 ✍️ functions, "The Challenge") | `SPEC.md` §2 lines 27–35 | Rewrite §2 from S19 and the current notebook (2 ✍️ exercises in T, sliders used, no Challenge) |
| 40 | `git clone` as a loading option for the student's data | 26 Oct sync (her suggestion; Simone's task 11) | Not offered: the loader knows upload, Drive mount and Drive link only. PRD Q3 did not carry it forward and nothing records dropping it | O 19 (loader box), `sections/01_load_your_dataset.py` | Either add an option D (a `!git clone <url>` line, then the cloned folder as `DATASET_PATH`) or record that it was dropped |
| 36, 45, 50 | Agent clean-up: obsolete or repeated code removed; the Own-Data notebook cleaned up | 26 Oct sync; her comments (1) and (4) | Dead code left: `is_readable` is defined and never called (its only caller, the old Check 2, was deleted by P7); `score_hunt` is defined in O but used only by Teaching-only cells | `sections/04_errors_cleaning.py:120` (T 72, O 80); O 89 | Delete `is_readable`; move `score_hunt` into a Teaching-only cell |
| 37, 46 | Capitals only for settings decided once, never reassigned or changed in place; PEP 8 names; consistent | 26 Oct sync, her comment (2), N-058/SN-008 | (a) `TASKS.append("detection")` changes the learner's setting in place. (b) In O, `CLASSES` and `CLASS_COLORS` are set in Setup (park list) and set again, computed from the data, at load; P9 kept the capitals, but in O they are no longer "decided once". (c) Two capital locals computed from data, `W, H = int(...)`. (Teaching is clean; `results` is lowercase.) | O 24 (`sections/01_load_your_dataset.py:932–933`, `:970–972`); `sections/08_segmentation.py:973` | (a) `tasks = list(TASKS)` + append to the copy, use `tasks` downstream. (b) Either skip the park `CLASSES` assignment in O (tag that setup line Teaching-only) or name the loaded ones lowercase; then tighten `test_capitals_only_for_settings` (no `SET_AT_LOAD` double, no in-place calls). (c) `w, h` |
| 47 | Consistency checks (one term per concept, headings) | Her comment (3), PRD Q6 | In O, cell titles still say "Animals" where the headings were switched to "objects" for any dataset: "Explore the Number of Animals per Image", "Explore Spatial Distributions of Animal Pixels"; the missing-box illustration still says "three animals". O also asks "which mistake is expensive" three times (Your Task in Meet Your Data, the Class Imbalance task, guiding question 1) | O 72, O 176, O 157–158; O 28, 74, 199 | Add O wording swaps for the two cell titles; keep one of the three expensive-mistake prompts in O |
| 62 | P11: heading levels per SN-011 | P11, 28 Oct | Teaching passes. In O, "#### How Each Format Stores Its Labels" sits directly under "## Load Your Dataset" (skips a level) | O 15 | Make it `###` |
| 73 | Q2b: encodings explained very clearly, **a drawn example per format** | PRD Q2b | Each format is explained with text, a table and the same sheep box in numbers; only the format actually loaded is drawn (See How One Image Is Labeled). No figure per format | O 15–17, O 25 | Either one small figure that draws the shared sheep box with the three coordinate conventions marked, or note that the text + live drawing replaces it |
| 75 | Q6: names, numbers, spelling, terms and cross-references fixed directly | PRD Q6 | Spelling (US) and cross-references to her chapter names are done; names and terms are not fully, see 37/46 and 47 | as in 37 and 47 | as in 37 and 47 |
| 79 | Q11: hand-back = PDF of annotated before/after + two Colab links | PRD Q11 | PDF exists but was made on 28 Oct (`33b96d5`), nine commits before HEAD (T-30 trail-camera colors, the embeddings explanation, Own-Data loader fixes); not yet sent (Simone's open task) | `reviews/2026-10-28-susanne-copy/handback/` | Re-run the PDF pipeline if any shot shows a changed cell, then send PDF + both links |
| 28 | SPEC's box definitions and legend brought in line with her box rules (the step-2 item of the 23 Oct work list) | 23 Oct work list | SPEC's header says her copy wins, but L-001 still defines eleven box kinds incl. objective, takeaway, pitfall, questions, across-tasks; L-003 still describes the old banner and legend; §6 still marks sections 1 and 9 "in-progress" | `SPEC.md` L-001, L-003, §6 | Replace L-001/L-003 with SN-015's four kinds and her banner/legend; update §6 status |
| 85 | `results` lowercase like every computed value; `RESULTS` exception gone | Simone, 29 Oct (N-058) | Notebook done (no `RESULTS` anywhere in `sections/`), but `RULES.md` N-058 still names `RESULTS` as the one exception | `RULES.md` N-058 | Drop the exception sentence from N-058 (the rule ledger is evidence-only: Simone decided this today) |

## Full list

### A. Walkthrough, 23 Oct (S-items, D-decisions)

| # | Agreement | Verdict | Evidence |
|---|---|---|---|
| 1 | S1: the setup's ✍️ demo should not read as a jump in topic | SUPERSEDED BY HER COPY | Her copy moved it into its own "Optional Coding Exercises" block after Setup, with her intro text (T 11–13) |
| 2 | S2/N-053: the ✍️ stub + reference fallback is explained once, without a count of how often it comes | DONE | T 11 "Optional Coding Exercises" explains both ways of working and that nothing blocks |
| 3 | S3: split the takeaway that mixed the mechanism with a dataset question | SUPERSEDED BY HER COPY | No takeaway boxes in her lesson; the annotation question is its own task (T 28 "Annotations: What Does an Image Annotation Look Like?") |
| 4 | S4/SN-001: few box kinds (answer toggle, blue theory, grey go-deeper) | DONE | T uses only `callout()` kinds theory 13, recipe 4, answer 37, deeper 2 (her set, SN-015); `test_boxes_are_her_four_kinds` passes |
| 5 | S5/SN-002: questions and code exercises are one Task style, marked by level (meeting decision) | DONE | Every task is a heading with 🟢/🔴; the coding exercise sits inside its task (T 133–136) |
| 6 | Legend wording "Tasks are marked beginner, intermediate, advanced" | SUPERSEDED BY HER COPY | Her banner says activities are 🟢 core · 🔴 optional (T 2), as in her copy |
| 7 | S6/SN-004: Recipe in theory blue | SUPERSEDED BY HER COPY | She kept the Recipe yellow `#FFF8D6` (SN-015) |
| 8 | S7/SN-003: section objectives as plain markdown under the heading | SUPERSEDED BY HER COPY | She removed per-section objectives; chapters open with a motivating paragraph (SN-010/011) |
| 9 | S8: takeaway green vs level dot | SUPERSEDED BY HER COPY | No takeaway boxes in T; O's closing "Takeaway" is plain bold markdown (O 200) |
| 10 | S9/N-054 + D2: box text the size of markdown text (16 px knob) | SUPERSEDED BY HER COPY | The 16 px knob lived on `drafts/susanne-prep`, dropped by Q1; her own boxes are 14 px and all boxes now are 14 px. Not re-checked by eye in Colab |
| 11 | S10: a box that only says what the next cells do becomes plain markdown | DONE | T 36 is plain markdown ("The next code cells calculate …") |
| 12 | S11/N-055: level dot in each task heading | DONE | Script over all headings: every task heading has its dot, no 🟡 left, no numbers |
| 13 | S12 + D5: chapter icons (🏷️ 📦 🎭), sent to her | SUPERSEDED BY HER COPY | Her chapter headings have no icons; the drafted icons were dropped with `drafts/susanne-prep` (Q1) |
| 14 | S13/N-056: section numbers generated, never typed | SUPERSEDED BY HER COPY | She removed all numbers; `test_no_question_or_section_numbers` passes; no "section N" left in visible text |
| 15 | S14/N-057: browse images (class picker, next/previous) instead of a fixed first N | DONE | T 22 "Browse Images Within a Class" (dropdown + Previous/Next); browsers also in T 29, 99, 122, 124, 132, 141, 148, 158, 171 |
| 16 | S15/SN-005: learning objectives with Bloom's verbs | DONE | T 2: explore/analyze/assess, identify/assess, select/justify (her wording) |
| 17 | S16/N-005: level calibration, most tasks 🟡 | SUPERSEDED BY HER COPY | She uses two levels only, 🟢 core and 🔴 optional |
| 18 | S17: keep the across-tasks boxes and the click-count comparison | SUPERSEDED BY HER COPY | Both are absent from her copy; SN-010 keeps them out |
| 19 | S18: notebook replaces lecture 2's slides; timing and scope untouched by us | DONE | Fuzzy diff of every T markdown cell against her copy: the only differences are approved ones (Optional tags, Q5's move of chapter 10, the MCQ heading, T-23, T-24, T-26) |
| 20 | S19: audience into SPEC §2 | NOT DONE | See table above |
| 21 | S20: separate "random pair ≈ 32 bits" from "nearest neighbour ≈ 20", one line on why unrelated photos can be close | SUPERSEDED BY HER COPY | Her rewritten text no longer claims a typical distance for unrelated photos; the histogram is labelled as nearest-neighbor distance (T 76–77). The extra "why" line is not there; optional |
| 22 | S21: labelling exercise as a later idea | DONE | Issue #20 open, label `future` |
| 23 | S22: possible reuse in continued education | NOT APPLICABLE | Information only |
| 24 | S23: her edits win, ported as she made them | DONE | Same diff as 19; no unexplained change to her text |
| 25 | D1: protected content test follows her box set | SUPERSEDED BY HER COPY | v05 content tests retired (Q10); `test_her_rules.py` encodes her rules instead |
| 26 | D3: Task box pale teal | SUPERSEDED BY HER COPY | Tasks are headings, not boxes |
| 27 | D4: Pitfall box stays red | SUPERSEDED BY HER COPY | She made it a blockquote (T 103), kept per T-23 |
| 28 | SPEC box rules / legend in line with her rules | PARTLY | See table above |

### B. Vised sync, 26 Oct (EDA part) and the project page

| # | Agreement | Verdict | Evidence |
|---|---|---|---|
| 29 | Material that does not fit goes to a "lost and found" section (SN-009) | SUPERSEDED BY HER COPY | Her final copy has no such section; the moved checklist ended in "Your Project Dataset" |
| 30 | Embedding explorer: DINO features next to the others, 2-D UMAP, most-similar pairs, per-class unusual images | DONE | T 92–101 (UMAP view, similar and unusual images, within-class similarity) |
| 31 | 3-D UMAP | NOT APPLICABLE | An idea to try, not agreed |
| 32 | Interactive views and toggles as she built them (colour histograms etc.) | DONE | Interactivity matches her copy cell for cell (e.g. T 50 slider, T 98–99 feature switch); nothing of hers removed |
| 33 | One merged split view (random / pairs / camera) | DONE | T 124 dropdown: Random · Stratified · Group by burst · Group by camera |
| 34 | Sample answers in their own colour | DONE | 37 grey closed toggles, `callout("answer")` `#F5F5F5` |
| 35 | Clickable multiple-choice question (SN-007) | DONE | T 126–127 (heading added per P11; Check Answer / Show Solution) |
| 36 | Agent clean-up of obsolete or repeated helper code | PARTLY | See table above (`is_readable`, `score_hunt`) |
| 37 | Capitals PEP 8, consistency is what matters (N-058, SN-008) | PARTLY | See table above (`TASKS.append`, `CLASSES` set twice in O, `W, H`) |
| 38 | A learner setting is read only in its own section (N-058) | NOT DONE | See table above (`HASH_THRESHOLD` in O's split suggestion) |
| 39 | Last chapter runnable without Run all (its own imports) | DONE | Replaced by the Own-Data Edition: its own Setup + loader (O 5–25) |
| 40 | `git clone` loading option | NOT DONE | See table above |
| 41 | A second edition with the dataset chosen at the top, park answers removed | DONE | `eda-own-dataset-notebook.ipynb`; `test_own_edition_has_no_sample_answers_and_no_park_hunts` passes |
| 42 | Configuration for any number of classes | DONE | Classes read from the data; 22-color palette repeating after that (`class_colors`, O 23) |
| 43 | Send her the insects zip | NOT APPLICABLE | Outside the notebook (Simone, done 26 Oct per the meeting note) |
| 44 | Photo screening: no nudity, swimwear or bare torsos (SN-006) | DONE | `scripts/prepare_data.py` `COCO_EXCLUDE` lists 5 ids with reasons; `data/coco_animals_subset.json` has 633 images and none of the 5; the DINOv2 file has 633 vectors |

### C. Her written comments on her copy, 28 Oct

| # | Agreement | Verdict | Evidence |
|---|---|---|---|
| 45 | (1) Code clean-up checks | PARTLY | Same as 36 |
| 46 | (2) Variable names per PEP 8 | PARTLY | Same as 37; `ruff --select N` finds only `W, H` (and sklearn-style `X`) |
| 47 | (3) Consistency checks | PARTLY | See table above (O cell titles, repeated prompt) |
| 48 | (4a) A separate notebook for the student's own dataset with the hints and checklists | DONE | O: *In the park* hints after the tasks, the checklist and resources boxes, report card, guiding questions (O 185–200) |
| 49 | (4b) …without the exercises | NOT DONE | See table above |
| 50 | (4c) …cleaned up | PARTLY | See table above (dead `score_hunt`, `is_readable` in O) |
| 51 | (5) Swimwear and bare-torso photos excluded | DONE | Same as 44 |

### D. Simone's answers to the proposals, 28 Oct (P1–P18)

| # | Agreement | Verdict | Evidence |
|---|---|---|---|
| 52 | P1: one `callout()` helper for her boxes | DONE | All boxes in T go through `callout()` except the Learning Objectives banner and How to Use (documented exceptions); O adds one raw-HTML ✏️ loader box from her chapter-10 HTML |
| 53 | P2: no shared `browse()` helper | DONE | Her widgets kept as built |
| 54 | P3: pHash computed once | DONE | `phash_bits` built once (`sections/04_errors_cleaning.py:152`) |
| 55 | P4: MD5 lines in the 🟢 duplicates task | DONE | T 73 computes `exact_groups` inside the duplicates task |
| 56 | P5: helpers take their table as an argument | DONE | `class_shares(images_by_id, ids)` (T 109) |
| 57 | P6: vectorised `find_copies` | DONE | NumPy block distances (`sections/04_errors_cleaning.py:558–570`) |
| 58 | P7: Own-Data drops old checks 1–3, keeps split suggestion + report card | DONE | O 188–192 |
| 59 | P8: `n_animals` → `n_objects` | DONE | No `n_animals` in `sections/` |
| 60 | P9: `CLASSES` / `CLASS_COLORS` stay capitals | DONE | Kept; but see 37 for O's second assignment |
| 61 | P10: pHash-only match labelled "recompressed" | DONE | `find_copies` docstring and `PLANT_KINDS` |
| 62 | P11: heading levels, MCQ heading, "Your Project Dataset" | PARTLY | T passes; O 15 skips a level (table above) |
| 63 | P12: "(Optional)" on every 🔴 heading | DONE | Heading script: none missing in either edition |
| 64 | P13: T-21…T-26 | DONE | Spot checks: T-23 (✏️ box now markdown, Pitfall blockquote), T-24 (one trail-camera intro paragraph), T-26 (working marker gone) |
| 65 | P14: CV diagram sentence matches the figure | DONE | T 113 and the figure title both say gray = training, yellow = validation |
| 66 | P15: one train/test colour pair; trail-camera colours not the park's | DONE | `LIGHT_GRAY`/`HIGHLIGHT` in both split figures; `CCT_COLORS` distinct from the park palette |
| 67 | P16: "photos" → "images" in printed outputs | DONE | No "photo" in any printed string, title, axis label or legend |
| 68 | P17: only "play dataset" → "Play Dataset" | DONE | No lowercase "play dataset" left |
| 69 | P18: no single hash threshold | DONE | As decided: `PAIR_THRESHOLD = 8` and `HASH_THRESHOLD = 5` both remain |
| 70 | T-17/T-18: 💡 on every summary, title patterns | DONE | Every answer toggle is titled "💡 Sample Answer…" |

### E. PRD decisions (Q1–Q11) and Simone's 28/29 Oct calls

| # | Agreement | Verdict | Evidence |
|---|---|---|---|
| 71 | Q1: her copy is the base; v05 and `drafts/susanne-prep` not merged | DONE | Diff as in 19 |
| 72 | Q2: Own-Data tasks become guiding questions with a blank answer line, no Sample Answers, park only as short hints | DONE | Every O task has "✍️ Your answer" + an *In the park* note, except a few with no fitting hint; no Sample Answer (test passes). The coding exercises are counted under 49 |
| 73 | Q2b: detection and segmentation optional (`TASKS`), encodings explained with a drawn example per format | PARTLY | Optional part done (`TASKS`, skip hook, O 23); drawn example see table above |
| 74 | Q3: formats (folders, CSV, COCO, YOLO, VOC, YOLO-seg, PNG masks), auto-detect, clear errors, empty path = park sample in the chosen format | DONE | `READERS` + `read_dataset` with named errors (O 22); `DATASET_PATH = None` writes the sample in the chosen format (O 24). Smoke per format not re-run by me |
| 75 | Q6: names, numbers, spelling, terms, cross-references fixed directly | PARTLY | See table above |
| 76 | Q4: one ordered cell list, two Editions | DONE | `just build` writes both; no drift |
| 77 | Q5: chapter 10's code leaves T; checklist + resources stay; closing cell links O and lists the guiding questions without answers | DONE | T 195–198 |
| 78 | Q7: a Task is one heading level below its parent | DONE | Heading script: no task breaks it in either edition |
| 79 | Q11: hand-back PDF + two Colab links | PARTLY | See table above |
| 80 | Q8–9: CPU target, precomputed park DINOv2, `COMPUTE_EMBEDDINGS` gate, fallback to colour histograms | DONE | `data/coco_animals_dinov2.npz` (633 × 384); gate + explanation T 95–97 |
| 81 | Q10: smoke on both Editions, one O smoke per format + a broken file, v05 tests retired, tests on her rules | DONE | `tests/test_own_smoke.py` (formats, broken file, real layouts), `tests/test_her_rules.py` (passes). Note: its capitals test misses the breaches in 37/38 |
| 82 | Simone, 28 Oct: Susanne is not asked further questions | DONE | Neither notebook nor the hand-back README asks her anything |
| 83 | Simone, 28 Oct: O includes optional detection and segmentation | DONE | O chapters Object Detection and Semantic Segmentation, skipped with one line when labels lack boxes or masks |
| 84 | SN-010: what she removed stays removed | DONE | No Challenge, Wrap-up, objective boxes, Q-numbers or across-tasks boxes in T |
| 85 | Simone, 29 Oct: `results` lowercase, no `RESULTS` exception | PARTLY | See table above (notebook done, rule text not) |
