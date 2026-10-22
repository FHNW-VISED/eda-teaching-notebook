# Review notes — 2026-10-22 (Simone, sections 0–7 read; 8–9 not yet)

Simone read the notebook after the cut pass (PR #19) and wrote the notes quoted below. The decisions
were taken with him in a `/notebook` grilling session the same day. **Applied 2026-10-22 on `cuts/2026-10-22`**
(he confirmed §8/§9 stay as they are). `just check`: 48 passed. R2's key sits in its own closed toggle right
under Q11's (the key is computed at runtime, so it cannot live inside the static callout).

## Decided: apply

| # | Note (verbatim) | Decision | Where |
|---|---|---|---|
| R1 | "the 'worked answers' to the questions could be made with the toggle as well, so that they are hidden until the toggle is removed — using the same toggle system there is in the 'The reasons and park examples behind each point' section" | **Every worked answer goes in a collapsed `<details>` toggle**, the same markup as §1 line 113: the 💡 text **and** the answer figures and printed keys (Q9's always-sheep model, §5's copy key, §9's planted-masks key, §10's `reveal_answers()`). A figure that is an answer is drawn as HTML output (`<details>` + base64 PNG) so it stays hidden until clicked. Figures a question *reads* (the plot the student must interpret) stay visible. **L-005 is replaced** (SPEC §8), and N-024 applies with no override. Frees ≈ 7 lab min (PR #19, time keeper). | all sections |
| R2 | "why this section? 'Score the reference detector against the planted key, print the answer key' — I do not understand what it does" | **Cut the heading, the cell and its 💡 line.** It scored the notebook's own detector, not the student's. The answer key (copy → kind → source) moves into Q11's worked-answer toggle as a small table, with one line: "variant hashing found all 5 planted copies and nothing else". `RESULTS["hunt5"]` is still written, in a silent collapsed cell (tests read it). | §5 |
| R3 | "make it clear also in the image when there are some other rows but you just show the first 3, for instance you could add some dots to hint at the further rows" | The LOO row of the fold diagram (12 folds, 3 shown) gets a `⋮` under fold 3 and a "folds 4–12 not shown" label. The other three rows show every fold and get nothing. N-051. | §6 fold diagram |
| R4 | "you show the opossum frames twice? is that intended?" | **Not intended.** `show_bursts` took the first three bursts by `seq_id`: a rabbit one and two opossum ones, all at camera 38, on different nights. Same tree and animal, so they read as a repeat. Show **three bursts of three different animals from three different cameras** (deterministic pick). | §6 `show_bursts` |
| R5 | "maybe it is a bit too focused on the data splitting strategies? … why is there so much of it? maybe we would not even need the second dataset, no?" | **Keep CCT20, slim §6.** Why it exists: Susanne's outline item 4 (splits, leakage, CV, group strategies, "Yes"), LOO added by Simone (G-27), and COCO has no real groups, so a real burst leak needs CCT20. **In class:** Q14, the CV callout and diagram, the trail cameras with one burst gallery, the leakage diagram, Q15 with the real-photo leak and the paper's 20.8 % → 41.1 %, and the LOO pitfall callout. **Moved to 🔴 (after the in-class part, one 🔴 heading):** the LOO-vs-LOGO sibling-rate cell, ✍️ `split_by_camera` with its crosstab, and the ruler (Q16, already 🔴). Everything still runs on Run all, so `RESULTS` keys (`cv_schemes` with `logo`, `sibling_rate_*`) are unchanged. Target ≈ 4–5 fewer lab minutes; the time keeper re-measures. | §6 |
| R6 | "in the '✍️ YOUR CODE' functions, it would be better if you also write in the function comment a fictitious return statement, something like ('e.g. {key1: [image1.jpg], key2:……}" | Every ✍️ docstring ends with an `e.g. …` example return in the function's real shape (`find_flipped_copies` → `e.g. ["card_017.jpg", "card_031.jpg"]`, `split_by_camera` → `e.g. {"train": ["59a3….jpg", …], "test": [...]}`, `count_boxes_per_class` → `e.g. {"cat": 214, "dog": 198, …}`, and §0's `double` demo stub). N-050, and port the template's N-050 lint check (`yourcode_stubs`) into this repo's `scripts/lint_notebook.py`. | every ✍️ cell; `scripts/lint_notebook.py` |

## Not reviewed yet

- "I have not checked in detail the object detection and segmentation sections, but they look very nice :)"
  → §8 and §9 wait for his read. ⚠️ §8 is the section furthest over its word budget (1,195 vs 400, PR #19).

## Rules

N-024 tightened (medium), N-050 (lint) and N-051 (review) added to the vault ledger on 2026-10-22, and
`RULES.md` was synced on this branch.

## Cuts decided the same day (Simone, after an honest repetition review; N-047 consent given per item)

Applied on `cuts/2026-10-22`. Time column ≈ 108 lab min (estimate, **not re-timed**; was ≈ 144).

| # | Repetition found | Decision | Effect |
|---|---|---|---|
| C1 | "A lazy model ignores the rare class" taught four times (§1 example, §4 Q9, §7's 1000-frame "always sheep" toy, §9's "background everywhere" demo) | **Remove**: §7 Metrics deleted; ✍️ `ALERT_METRIC` (accuracy / recall / precision on dog, each defined in one line) moved to the end of §4; `box_iou_dice` + one go-deeper (counting error, IoU, Dice, mAP) moved into boxes; §9's background demo cut; §1's example no longer pre-empts Q9 | Sections renumbered: boxes 7, masks 8, Challenge 9, your dataset 10, wrap-up 11. Q17 gone |
| C2 | The Challenge re-runs the hunts of §5, boxes and masks | **Optional**: in class a demo (batch → reference score → answer-key toggles); the Q24 hunt sits under "🔴 Optional · Run the hunt yourself" | FACILITATION row 9: 7 min, demo |
| C3 | Q5 (detective beat) found what Q4's answer already says (longest side capped at 640) | **Remove** the beat and its question | Q5 gone; questions now 1–26 |
| C4 | "Deduplicate before splitting" said four times | **Keep only §5's pitfall**; §5's burst go-deeper, §6's across-tasks line and §10's pitfall go | — |
| R7 | "can you please show the box of the animal, if there is one? since they are hard to see" (trail-camera bursts) | The 512 px copy stores the camera's original size in its JPEG comment; `draw_cct_boxes` draws the manifest's `bbox` in the class colour on the burst gallery and on Q14's real-photo leak figure; the gallery prefers bursts with a box on every frame | Old cached copies without the comment are re-downloaded once |

Not chosen: #4 (merge §1/§2) and #6 (drop "What a label costs").
Tests: §7's six tests and the detective-beat / rabbit / 8-minute-hunt tests are replaced by C1–C4, R1 and R7 tests
(`tests/test_cuts.py`); `tests/test_content.py` only renumbered (alert metric now requires section 04).

## Sections 7–11 re-read through the review lens (same day, all applied on Simone's "go for it")

| # | Problem a student would hit | Fix |
|---|---|---|
| R8 | §8 mask hunt: the job never stated, the five kinds only in words, suspects gallery did not mark the suspect, IoU's role unclear ("does it work for all the five checks or only some of them?") | The job stated; a toy picture of the five kinds; each check with its "suspect when" rule (IoU = check 1 only, `IOU_MIN` its one threshold); Q22 as reading steps; the flagged mask bright with a solid box, others faint (commit 66aa406) |
| R9 | §7: no boxed photo before Q16/Q17 | Gallery of four: sheep flock (15 boxes), lone cat (1), bird flock (13 small), big horse |
| R10 | §7 Q19 printed the planted photo openly | Into an answer toggle |
| R11 | §7 missing-box toy drawing came after the answer | Moved above Q19 |
| R12 | §7 IoU taught without a purpose | One line: used by the 🔴 crowding part and §8's check 1 |
| R13 | §7 cryptic clicks sentence | Plain: masks cost far more clicks, ask for them only when outlines are needed |
| R14 | §8: no masked photo before Q20/Q21 | Gallery of four: close-up cat (49 %), bird flock (4 %), horses' legs, sheep flock |
| R15 | §8 Q20 answer ended on a technical aside | Cut |
| R16 | §9 demo showed one line ("found 14 of 14") | Demo toggle: one line per kind ← check → files |
| R17 | §9 closing line assumed students hunted | A question for the room |
| R18 | §9 cryptic "passed every check of sections 5 to 8" | "Every problem was planted in a photo that was clean before" |
| R19 | §10: check outputs with no hint what to look at in class | One 💡 "look at …" line under each check |
| R20 | §10: `data/cache/my_split.csv` vanishes with the Colab session | Download/Drive instruction under check 4 and in the Takeaway |
| R21 | §10 Q25 mixed a no-answer question with the sample question | Only the sample question (it has an answer) |

## §10 setup, tested on a real outside dataset (R22)

Simone: "have you tested with another dataset? does it work? make the setup as clear as possible … very very clear
and test it out". It had **never** run on anything but the built-in sample, and a wrong path printed one ❌ line and
then silently ran on the park sample.

- Setup rewritten as three steps (get it into Colab · Copy path · paste and Runtime → Run cell and below) with ✅/❌
  path examples; Drive mounts itself for `/content/drive/…`, a `.zip` is unzipped, a lone wrapper folder is
  stepped into; every wrong layout **stops** with what to change; skipped files and empty class folders are listed;
  three images per class are shown first.
- `tests/test_my_dataset.py` (10 tests, in `just check`): right folder, quoted path, wrapper, zip with `__MACOSX`,
  typo, class folder, split folders, a file, no images, skipped files/upper-case/nested.
- End to end on PyTorch's ants/bees (`hymenoptera_data.zip`, 398 images): the zip stops on its train/val folders
  with the right advice; `…/train` runs all four checks (245 images, the one `.gif` flagged as mode P, 0 copies,
  49/196 split); a typo shows the nearest folder's contents; a messy folder (corrupt file, RGBA png, .txt, empty
  class, copy across classes) is reported check by check.
- Not testable outside Colab: the Drive sign-in and the Files-pane upload. Simone checks those in Colab.

## Order review under N-052 (same day; Simone: "fix the high and medium right now")

| # | Problem | Fix |
|---|---|---|
| O1 | §4 the ✏️ `DOG_SHARE` box came before Q8 and pointed at it | Q8 → ✏️ (leave 0.02) → cell → answer; the 0.2 / 0.5 experiment after the pitfall |
| O2 | §5 the ✏️ `HASH_THRESHOLD` box came before Q9; "You run it in Q9 below" | Q9 → ✏️ (leave 8) → cell → answer; the 2 / 14 experiment after the answer; no forward reference |
| O3 | §6 "One bad shuffle" box before Q13 gave its answer away | moved after Q13's answer |
| O4 | §10 a go-deeper box sat between "paste in the next cell" and the ✏️ cell | moved after the reader output |
| O5 | §6 the burst diagram and steps before Q14 printed both sibling rates | moved after the Q14 cell, as the "why" |
| O6 | §5 the flip detector was written before the shoot-out showed why | shoot-out (Q10) → why picture → answer → ✍️ `find_flipped_copies` → key |
| O7 | §2 Q2 asked about a grid 18 cells back | the question now sits right under the grid in §1 (becomes Q1; the checklist question becomes Q2) |

Low items (empty "Exact duplicates" heading, the Challenge's two keys, "checks that claim", a lint for forward
references) are proposed, not applied. Forward question references after the pass: none.

## Picture B's text (R23)

Simone, on a screenshot of §8 Picture B: *"wtf I do not understand what the description you added is for"*. The
figure was right; the text was not: three 8-pt lines over every panel, and a suptitle legend ("solid box: the
mask's own · dashed: the team's") that only concerned column 1, in jargon. Now: column title = the check's name
(13 pt); one line under each column with its rule; a large green **normal** / red **SUSPECT** inside each panel
with the value small beside it; plain legends only where a line needs one (column 1: "solid: tightest box around
the mask · dashed: the box the team drew"; column 5: "dashed: the rubber band"); suptitle names the two rows only.
Pinned by `test_r23_picture_b_text_is_a_name_a_rule_and_a_verdict`.
