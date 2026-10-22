# Review · cuts and time (time keeper, pass 1, 2026-10-21)

The short version: at the pace of a student who actually reads, the core path (everything except 🔴 and go-deeper) takes
about **244 minutes**, or about **190** at a brisk lab pace. That is 2.7× (or 2.1×) the 90-minute slot. Most of the time
goes on reading: about 18,100 visible words, roughly 100 minutes on its own. Shortening text gets back about a fifth of
that. Getting near 90 also takes Simone's yes on about 15 question-level cuts and four structural moves (plan, §4).

## 1. Minutes per section

How I estimated. **Careful pace:** visible markdown (callouts, 💡, worked answers; no go-deeper, no 🔴) at 180 wpm, since
many students read English as a second language. Each figure, gallery or table counts 0.5 min. A prediction question is
1 min, one that needs a calculation 2, ✏️ 1.5, ✍️ 3–4, and the professor's talk 1.5–2.5 per section (taken from
`FACILITATION.md`). **Lab pace:** 250 wpm, 0.3 min per figure, the professor talks while people look; this comes out at
≈ 0.78× careful. Word counts are hand counts, ±15 %. "After plan" is the careful-pace estimate once all of §4 is applied.

| § | SPEC | Now (careful) | Over | Visible words · figs · Qs | Basis of the estimate | After plan |
|---|---|---|---|---|---|---|
| 0 | 5 | 7 | +2 | ~600 · 1 · ✍️ demo | 3.5 read + setup/download/`double()` 2 + talk 1.5 | 5.5 |
| 1 | 5 | 11 | +6 | ~1,050 · 2 · Q1 | 6 read (checklist callout alone ~470 w) + 1 figs + Q1 and its 11-line output 2 + talk 2 | 3 |
| 2 | 8 | 12 | +4 | ~750 · 2 · Q2, Q3, ✏️ | 4 read + 2 grids + Q2 1 + ✏️ 1.5 + Q3 writing 3 + talk 0.5 | 8.5 |
| 3 | 8 | 24 | +16 | ~1,950 · 11 · Q4–Q6, ✏️ | 11 read + 6 figs + Q4/Q5 1 each, Q6 2 + ✏️ 1.5 + talk 1.5 | 9 |
| 4 | 8 | 18 | +10 | ~1,430 · 5 · Q7–Q9, ✏️ | 8 read + 3 figs + Q7, Q8 1 each, Q9 2 + `DOG_SHARE` reruns 1.5 + talk 1.5 | 11 |
| 5 | 15 | 29 | +14 | ~1,870 · 9 · Q10–Q13, ✏️, ✍️ | 10.5 read + 5.5 figs + 4 Qs 4.5 + threshold reruns 2 + ✍️ 4 + talk 2.5 | 17 |
| 6 | 12 | 26 | +14 | ~1,950 · 8 · Q15–Q17, ✍️ | 11 read + 4 figs + download 1 + 3 Qs 4 + ✍️ 4 + talk 2 | 15 |
| 7 | 7 | 19 | +12 | ~1,530 · 7 · Q19–Q21, ✍️, ✏️ | 8.5 read + 3 figs + Q19 1, `ALERT_METRIC` 1, Q20 1.5, Q21 1 + talk 2 | 8.5 |
| 8 | 10 | 23 | +13 | ~1,650 · 9 · Q22–Q25, ✍️ | 9 read + 5 figs + 4 Qs 4 + ✍️ 3 + talk 2 | 7.5 |
| 9 | 10 | 27 | +17 | ~2,100 · 10 · Q26–Q29, ✏️ | 11.5 read + 6 figs + Q26–Q28 3, Q29 2 + thresholds 2 + talk 2 | 14 |
| 10 | 15 | 25 | +10 | ~1,400 · 3 · Q31, Q32 | 8 read (incl. tools list and key) + Q31 3 + hunt 10 + key 2 + talk 2 | 11 |
| 11 | 5 | 18 | +13 | ~1,400 · 5 · Q34, Q35, ✏️ | 8 read + checks 2 + Q34 1 + Q35 writing 5 + talk 1.5 | 4 |
| 12 | 2 | 5 | +3 | ~420 · 1 | 2.5 read + recap 1 + talk 1.5 | 3.5 |
| **Σ** | **110** | **244** (lab ≈ 190) | **+134** | ~18,100 · ~70 · 31 core Qs | 🔴 parts Q14, Q18, Q30, Q33 add ~16 more | **≈118** (lab ≈ 92) |

For scale: 90 minutes at careful pace holds about **7,000 visible words, ~17 questions, 3 ✍️, ~4 ✏️, ~35 figures and an
8-minute Challenge**. The SPEC itself still says "~110 total" (§6) while the review says ~90, so the Time column needs a
rewrite either way.

## 2. Duplicated explanations (keep → the one marked ✔)

1. **"Always sheep" scores high and misses every dog:** 01:51–54 · ✔ 04:160–235 (Q9 demo) · 04:141–145 (Pitfall, which
   answers Q9 before anyone predicts) · 07:11–12 · 07:32–71 (toy matrices) · 07:277–278 · 09:244–247 (pixel version: keep, one line).
2. **Karpathy's same two sentences** ("duplicate examples… corrupted images / labels") in 04:16–17 and ✔ 05:17–18.
3. **Bursts leak / the score measures memory:** 03:316 · 05:727–734 (4 steps) · 06:15–16 · 06:19–29 (callout) ·
   ✔ 06:304–344 (4 steps + diagram) · 06:419–428 · 11:303–305.
4. **Deduplicate before splitting:** ✔ 05:687–690 · 05:745 · 05:753 · 06:608 · ✔ 11:248–251 (SPEC pitfall) · 12:61.
5. **Leave-one-out's weakness:** 06:131–132 · 06:137–144 (go-deeper) · 06:182–191 · ✔ 06:441–449 (Pitfall) · 06:474–476 · 06:616–617.
6. **Say the unit (photo, box, pixel):** 04:244–253 · 08:19–23 · 08:151–153 · ✔ 08:526–533 (Pitfall) · 08:537–538 ·
   09:168–170 · ✔ 09:238–241 · 09:821–822 · 12:46–47 · 12:62–63.
7. **"Animals sit in the centre" heatmaps:** ✔ 08:271–312 (box centres, the SPEC's wording) and 09:355–393 (mean mask).
8. **IoU defined:** 07:365–366 · 07:391–432 (box IoU/Dice figure) · ✔ 08:321–354 (drawing) · 09:598.
9. **Smooth resize breaks masks, use nearest-neighbour:** ✔ 03:513–520 + Q6 · 03:602–608 · 03:610–623 · 09:402–403.
10. **"An odd group means another source → keep, convert or drop":** 03:374–375 · ✔ 03:437–448 · 03:497–499 · 03:627–628.
11. **Plain hashing is blind to flips:** 05:290–291 (Pitfall, before Q12) · 05:299–302 · ✔ 05:467–474.
12. **"Unrelated photos are ~32 bits apart":** ✔ 05:151–152 · 05:205–206 · 05:276–278.
13. **The ten Challenge kinds** are listed three times: 10:29–41 · ✔ 10:44–54 · 10:324–326 (plus 10:339–345).
14. **Most crowded photos gallery:** 04:108–110 and ✔ 08:388–395 (the same photos, with boxes).
15. **Recall reads the row, precision reads the column:** ✔ 07:19–30 · 07:63 · 07:201–207 · 07:240–241 · 07:345–350.
16. **Class grid before judging:** 01:120–146 (3 per class) and 02:120–124 (6 per class), two cells apart.
17. **Small boxes shrink further on resize:** 01:48–50 · 08:267–269 · 09:29–30 · ✔ 09:398–404.

## 3. Proposals (ordered by minutes saved, careful pace)

**Apply** says who decides. *safe* means shortening, or moving into go-deeper or `<details>`, with nothing removed;
the orchestrator can apply these now (Phase 4). *Simone* means a removal, or a question or part leaving the core path,
which needs his yes under N-047.

| # | What | Where | Type | Min | Conf | Apply |
|---|---|---|---|---|---|---|
| 1 | Section 11 in class = folder tree, ✏️ cell and the report card on the sample. Q34, the four "how to read" 💡, Q34's answer, the guiding questions and Q35 become an "After the lab" block (FACILITATION already says "Q35 as homework") | 11:117–376 | move to 🔴 | 13 | high | Simone |
| 2 | Merge each 💡 with the worked answer right under it into one block (22 pairs: 03:309,497,602 · 04:62,113,222 · 05:131,276,467,583 · 06:106,182,420 · 07:190 · 08:151,263,398,517 · 09:235,346,449,714). Takeaways ≤ 3 lines. Cut the "Section N ended with…" openers (02:9, 03:10, 04:10, 06:12, 07:11, 08:10, 09:168) | all | shorten | 9 | high | safe |
| 3 | Section 9: fill-ratio violins + gallery (Q27), the mean-mask heatmap and thin parts (Q28) become one 🔴 "More about masks" part. The `animals` cell stays collapsed (10 needs it); fill keeps its one-line definition in the check list (600) | 09:279–456 | move to 🔴 | 8 | med (SPEC lists heatmaps, thin parts) | Simone |
| 4 | Q13 least-typical gallery → 🔴 beside Q14. Colour finds *unusual*, not *wrong*, and its own answer is "nothing to send back" | 05:545–590 | move to 🔴 | 4 | med (SPEC core bullet) | Simone |
| 5 | 🔀 callouts: put the body inside `<details>` (the kind stays, so the kinds test passes) | 05:737–745 · 06:599–608 · 08:406–420 · 09:458–470 | go-deeper | 3.5 | high | safe |
| 6 | Cut the Q20 quiz (a [PROPOSAL] item) | 07:303–352 | cut | 3 | high | Simone |
| 7 | Cut Q16 and the CV table; keep the 📘 callout. Keep `cv_schemes` filled in a collapsed cell (the test needs kfold/stratified_kfold/loo/logo) | 06:146–191 | cut | 3 | med | Simone |
| 8 | Fold Q31 (written plan) into Q32: the cell's comments already map kind → tool, and the pair moment becomes "read them together" | 10:323–328 | merge | 3 | med | Simone |
| 9 | Pre-write the copy and grayscale checks in the Q32 cell. Learners write only the four two-source checks, which 10:505–510 names as the lesson | 10:334–345 | shorten exercise | 3 | med | Simone |
| 10 | Bilinear formula, "why not round" and the 6×6 zoom → one go-deeper after Q6 (R-23 content kept) | 03:583–599, 610–623 | go-deeper | 2.5 | high | safe |
| 11 | Leakage callout trimmed to the definition + rule (the burst story is told at 304–313); LOO-vs-LOGO printout folded into the Pitfall, drop 474–476; Q15 callout shortened | 06:19–29, 40–43, 451–476 | shorten | 2.5 | high | safe |
| 12 | Cut Q21: the 🔀 callout (362–364) states its answer. IoU/Dice figure → go-deeper next to mAP | 07:386–439 | cut + go-deeper | 2.5 | high | Simone |
| 13 | Cut Q24 and the crowding table/gallery; keep the IoU drawing (09 uses IoU) | 08:356–404 | cut | 2.5 | med (SPEC lists crowding) | Simone |
| 14 | Detective beat merged into Q5: keep the 3 steps, drop the toy figure (the Q4 scatter is the real one) | 03:437–468 | merge/cut | 2 | med (N-043 visual) | Simone |
| 15 | Cut the 2×2 toy matrices; one line back to section 4 stays | 07:32–71 | cut | 2 | med | Simone |
| 16 | "Find a dataset" + "Label your own" → one go-deeper; each checklist example cut to one line (R-08/R-09 content kept) | 01:14–33, 40–72 | go-deeper/shorten | 2 | med | safe |
| 17 | Burst steps → one line pointing to section 6; move the "orientation-sensitive" sentence after Q12 (today it gives Q12 away, N-004) | 05:727–734, 290–291 | shorten/reorder | 1.5 | high | safe |
| 18 | Move the Pitfall after Q9 and merge it with 222–224 (it answers Q9 before the prediction). Shorten 120–132. Replace the 04 quote with the next verbatim lines ("I look for data imbalances and biases. I will typically also pay attention to my own process for classifying the data, which hints at the kinds of architectures we'll eventually explore."), checking them against the source first | 04:14–23, 120–145 | reorder/shorten | 1.5 | high (quote: med) | safe |
| 19 | One list of the ten kinds, each with the step where it goes wrong, instead of three | 10:29–54, 324–326 | merge | 1.5 | high | safe |
| 20 | 08 theory callout trimmed (each point has its own subsection); 09 polygon/RLE storage → `<details>`; 08 hunt story in 3 steps | 08:18–29, 425–438 · 09:9–18, 146–149 | shorten | 1.5 | high | safe |
| 21 | Cut the brightness violin: the min/median/max figure has a brightness row and the ✏️ default is brightness | 03:383–410 | cut | 1.5 | med | Simone |
| 22 | Cut the clicks cell; the 🔀 callout's 1 : 10 : 100 carries the point | 04:255–276 | cut | 1.5 | med | Simone |
| 23 | One class grid: Q2 uses section 1's grid (raise it to 6 per class) and `show_all_classes` goes. R-11's grid before Q1 stays | 02:120–124 | merge | 1.5 | med | Simone |
| 24 | Shop contrast as the one paragraph the SPEC asks for, numbers inline (safe); cut the shop cell 243–254 (Simone) | 07:229–254 | shorten + cut | 1.5 | med | split |
| 25 | Cut the answer-key figure; keep the printed key (the copies figure 427–464 already shows each kind) | 05:528–538 | cut | 1 | med | Simone |
| 26 | Small trims: 00 takeaway → one line (setup mechanics, not a lesson); Q3 worked answer → 3 one-liners; 12 next-week text | 00:391–397 · 02:174–186 · 12:66–75 | shorten | 1 | high | safe |
| 27 | Cut the toy missing-box drawing: its title answers Q25 before it is asked | 08:469–489 | cut | 0.7 | high | Simone |
| 28 | 🔴 parts: cut Q14 (in class the embeddings file is missing and it runs on the same colour histograms as Q13), cut Q30, keep Q18 and cut Q33 (one ruler is enough while G-33 is open) | 05:592–677 · 09:737–817 · 10:512–581 | cut | 0 core (~11 🔴) | high / med / med | Simone |

**Totals:** the safe rows (2, 5, 10, 11, 16–20, 26, and the text half of 24) save ≈ 27 min. The Simone rows save ≈ 53.

**The section agents' cut ideas (Log), checked:**
- *Confirmed:* 03 6×6 zoom (in #10), 07 Q21 IoU/Dice figure → go-deeper (extended to cutting Q21, #12), 09 🔴 Q30, 10 🔴 Q33 and 05 Q14 (all #28),
  the 01 labelling list (extended to both lists, #16), Q13 (as a move to 🔴, #4).
- *Refined:* 06 🔴 Q18 ruler (keep it, cut Q33 instead), 06 LOO-vs-LOGO cell (keep the computation, since the test needs `logo`; fold the text, #11),
  04 clicks (keep the callout, #22), 03 Q5 gallery (keep the gallery, cut the toy, #14), 07 shop (#24).
- *Rejected:* 07 cutting the whole shop block (the SPEC asks for the paragraph), 08 box-centre heatmap (cut 09's instead, #3; 08's
  is the SPEC's wording and simpler), and the Kaggle stories (~50 words, ~20 s; the professor tells them).

## 4. Plan to bring the core path to ~90

1. **Apply the safe rows now** (#2, 5, 10, 11, 16–20, 26, the text of #24): 244 → ~217.
2. **Ask Simone for the itemized removals** (#1, 3, 4, 6–9, 12–15, 21–25, 27). That leaves **21 core questions**: 244 → ~164.
   Same arithmetic per section: 162 bottom-up.
3. **Word budget, safe:** move what is left beyond ≈ 7,000 visible words into go-deeper / `<details>`. Per section:
   0: 400 · 1: 500 · 2: 450 · 3: 700 · 4: 600 · 5: 800 · 6: 800 · 7: 550 · 8: 400 · 9: 650 · 10: 650 · 11: 300 · 12: 250.
   About −18 min.
4. **Four structural moves for Simone** (about −26 min):
   - Section 1's sources and checklist sent as pre-reading (N-032), so in class only the Q1 cell runs.
   - Section 8 cut down to ✍️ `count_boxes_per_class` + Q22 + Q23. The heatmap and the missing-box hunt go to 🔴; the Challenge
     still plants a missing box, and `hunt8` keeps running so the test passes.
   - Section 3's colour half (modes, histograms, gray gallery, ✏️ extremes) → 🔴. This touches his own R-14/R-17/R-19.
   - Merge Q8 into Q7 and Q10 into Q11. Time-box the Challenge hunt to 8 min, then show the reference score and the key.

The result is ≈ **118 min careful ≈ 92 min at lab pace** (the "After plan" column). Without step 4 it is ≈ 140 careful /
≈ 110 lab, which is what the SPEC's "~110" assumes. Once Simone decides, the SPEC §6 Time column should be rewritten from the
"After plan" column.
