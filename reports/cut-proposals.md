# Cut proposals — review pass 1 (2026-10-21)

Nothing on this list has been applied. Each row removes a part or a question, or moves one out of the core path, so
under N-047 it waits for Simone's yes. Shortening, merging text and moving depth into go-deeper are already done (see
`reports/revise-2026-10-21.md`).

Minutes are the time keeper's estimates at careful student pace (`reports/review-cuts.md`, ±15 %). Before Phase 4 the core path
(everything except 🔴 and go-deeper) was about 244 min careful / 190 lab pace, against a 90-minute slot and a SPEC budget of 110.
Phase 4 applied the safe rows, which the time keeper put at about −27 min. Reviewers: **C** = time keeper (cuts),
**P** = professor, **S** = student, **A** = a section agent.

To answer, put **yes / no / other** in the last column, or just tell me the numbers.

## Questions and parts leaving the core path

| # | What | Where | Min saved | Reviewer | My recommendation | Simone |
|---|---|---|---|---|---|---|
| 1 | Section 11 in class = folder tree, ✏️ `MY_DATASET_DIR` cell, report card on the sample. Q34, the four "how to read" notes, the guiding questions and Q35 become an **"After the lab"** block | 11, from Q34 to the end | 13 | C, P, FACILITATION | **Yes.** FACILITATION already says "Q35 as homework"; nobody has their own data in class anyway | |
| 2 | Section 9: thin parts (Q28) and the per-class mean-mask heatmaps → 🔴 "More about masks". Keep Q27 fill ratio: the Q29 checks use it | 09, mean-mask heatmap + thin parts | ~5 | C, P, S | **Yes**, but keep Q27 (C proposed moving Q27 too, 8 min) | |
| 3 | Q13 (least-typical photos) → 🔴 next to Q14. Colour finds *unusual*, not *wrong*; its honest answer is "nothing to send back" | 05, "Suspicious labels" | 4 | C, A | **Yes** | |
| 4 | Cut the Q20 rabbit quiz and the `QUIZ_ANSWER` check; ask it aloud as a show of hands instead | 07, "Quick check" | 3 | C, P, S | **Yes** | |
| 5 | Cut Q16 and the CV table; keep the 📘 CV callout and the new fold diagram, which shows the "0 %" answer; `cv_schemes` stays computed in a collapsed cell (a test needs it) | 06, Q16 | 3 | C, P | **Yes** | |
| 6 | Fold Q31 (written plan) into Q32: the Q32 cell now has one header per kind, which *is* the plan | 10, Q31 | 3 | C | **Yes** (renumbering Q32→Q31… follows) | |
| 7 | Pre-write the copy and grayscale checks in Q32, so learners write only the four "two sources disagree" checks (the lesson of the Challenge) | 10, Q32 cell | 3 | C, S | **Yes for the copy checks** (section 5 already taught them) | |
| 8 | Cut Q21 (counting error vs box mistakes); the 🔀 callout above already states its answer. IoU/Dice figure → go-deeper next to mAP | 07, Q21 | 2.5 | C, S | **Yes** | |
| 9 | Move 07's second half (counting error, IoU, Dice, mAP) to the end of section 8, where boxes exist. Section 7 then ends on recall/precision | 07 → 08 | ~1.5 net | P | **Yes if #8 is no**; otherwise moot | |
| 10 | Cut Q24 and the crowding table + gallery; keep the IoU drawing (09 uses it) | 08, Q24 | 2.5 | C | **No, move it to 🔴**: the SPEC lists crowding, and the "count without the boxes" moment works aloud | |
| 11 | Section 3 detective beat: drop the toy figure (the Q4 scatter is the real one); keep the three numbered steps | 03, detective beat | 2 | C, P | **Yes** | |
| 12 | Cut Q5 (odd sizes); same lesson as Q4 | 03, Q5 | 2 | S | **No.** It is the only place the "one dominant value, ask why the others differ" habit is practised | |
| 13 | Cut the 2×2 toy confusion matrices in 07 | 07, opening | 2 | C | **No.** They are the N-049 simulation and the professor's build-up to the 6×6 matrix | |
| 14 | Cut the brightness violin in 03 (the min/median/max figure and the ✏️ extremes show brightness twice) | 03 | 1.5 | C, P | **Your call.** It is your R-18 violin "here"; 08 and 09 keep theirs | |
| 15 | "What a label costs" (clicks table): move it from the end of 04 to the opening of 08 | 04 → 08 | 1.5 | P (C: cut it) | **Move**, not cut: it introduces boxes and masks | |
| 16 | One class grid: section 2's Q2 uses section 1's grid (raised to 6 per class); `show_all_classes` goes. R-11's grid before Q1 stays | 02 | 1.5 | C | **Yes** | |
| 17 | Cut the shop cell in 07 (the paragraph the SPEC asks for stays, with numbers inline) | 07, shop | 1.5 | C | **Yes** | |
| 18 | Cut the answer-key figure in 05's hunt (keep the printed key; the five-kinds figure already shows each copy) | 05, after Q12 | 1 | C, P | **Yes** | |
| 19 | 08 Q23: drop the stacked share bars (the table above has the same numbers) | 08, Q23 | 0.7 | P | **Yes** | |
| 20 | 07: drop the decorative `predictions.head()` table | 07 | 0.3 | P | **Yes** | |
| 21 | 09 ✏️: keep only `IOU_MIN` as ✏️ and hard-code the other four thresholds (SPEC: "at most one ✏️ per section") | 09, thresholds | 2 | S | **Yes**; "stricter / looser" then means one value up and down | |
| 22 | Merge Q10 into Q11 (MD5 exact copies as the first line of the near-duplicate question) | 05 | ~1.5 | C | **Yes** | |
| 23 | Merge Q8 into Q7 (animals per photo with the class counts) | 04 | ~1.5 | C | **Neutral**; Q8 seeds "which label then?" | |
| 24 | Time-box the Challenge hunt to 8 min, then show the reference score and the key | 10, facilitation | ~2 | C | **Yes** (FACILITATION change only) | |

## 🔴 optional parts (not in the core path; they save scroll, not class time)

| # | What | Where | Min if run | Reviewer | My recommendation | Simone |
|---|---|---|---|---|---|---|
| 25 | Cut 🔴 Q14 (embeddings): in class the embeddings file is absent and it runs on the same colour histograms as Q13 | 05 | ~3 | C, P, A | **Yes** | |
| 26 | Cut 🔴 Q30 (boundary–edge alignment) | 09 | ~3 | C, S, A | **Yes** | |
| 27 | Cut 🔴 Q33 (ruler in the Challenge); keep 🔴 Q18 (ruler in section 6), one ruler is enough while G-33 is open | 10 | ~3 | C (P, S: cut both) | **Yes, cut Q33, keep Q18** | |
| 28 | Cut 🔴 Q18 too | 06 | ~5 | P, A | **No** (see #27) | |

## Structural moves (bigger decisions)

| # | What | Min saved | Reviewer | My recommendation | Simone |
|---|---|---|---|---|---|
| 29 | Section 1's sources and checklist as **pre-reading** sent before the lab (N-032); in class only Q1 runs | ~5 | C | **No for the checklist** (you asked for it to be richer, R-09); yes for the links, which are already a go-deeper box | |
| 30 | Section 8 cut down to ✍️ `count_boxes_per_class` + Q22 + Q23; the box-centre heatmap and the missing-box hunt → 🔴 (the Challenge still plants a missing box; `hunt8` keeps running for the tests) | ~8 | C, P | **Maybe.** The hunt is the best "two sources disagree" warm-up for the Challenge; I would move only the heatmap | |
| 31 | Section 3's colour half (modes, histograms, grayscale gallery, ✏️ extremes) → 🔴 | ~6 | C | **No.** It holds your R-14/R-17/R-19 requests | |
| 32 | One "animals sit in the centre" heatmap instead of two: keep 08's box-centre heatmap, drop 09's mean-mask one (part of #2) | 1.5 | C, S (A: the other way round) | **Yes, keep 08's** | |
| 33 | Rewrite the SPEC §6 Time column after your decisions (it adds up to 110; the slot is 90) | — | C, S | **Yes**: I'll do it from your answers | |

**Rejected by the time keeper** (listed so you see them): cutting the whole pet-food-shop block (the SPEC asks for the
paragraph), cutting the Kaggle stories (~20 s), cutting 06's LOO-vs-LOGO computation (a test needs `logo`).

**If you say yes to all my "Yes" rows** (about −50 min, counting #10 as a move to 🔴), the time keeper's arithmetic
gives a core path of about 165 min careful / 130 lab (217 after Phase 4, minus 50). Reaching ~90 at lab pace also needs #30 and a per-section word budget (C §4: about 7,000 visible words in total).
These are estimates, not timings: one timed run-through would settle them.
