# Review notes — 2026-10-22 (Simone, sections 0–7 read; 8–9 not yet)

Simone read the notebook after the cut pass (PR #19) and wrote the notes quoted below. The decisions
were taken with him in a `/notebook` grilling session the same day. **Nothing here is applied yet.** It
is applied in **one batched revise pass** on `cuts/2026-10-22`, together with his §8/§9 review and the
open cut levers of PR #19. Then `just version`.

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
