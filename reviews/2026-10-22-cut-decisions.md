# Cut decisions — 2026-10-22 (Simone, via the Notebook Cut Planner)

Simone went through `reports/cut-proposals.md` (pass 1) in the interactive Cut Planner and pasted his choices
on 2026-10-22. Nothing below is applied yet; the work is specified in the GitHub issue that links to this file.
Rows refer to `reports/cut-proposals.md` (#) and the planner's ids (e…). Minutes are the time keeper's careful-pace
estimates; lab pace = 0.78 × careful.

**Where this lands:** ≈ 114 min lesson at lab pace (≈ 146 careful), ≈ 141 with optional parts, against the
90-minute slot. **Still ≈ 24 min over at lab pace.** Simone left Q32 as a possible further cut (see Open).

## Decided: apply

| # | Planner id | What | Decision | Note |
|---|---|---|---|---|
| 16 | e16 | §2 one class grid instead of two | **Merge**: Q2 uses section 1's grid (raised to 6 per class); `show_all_classes` goes. The grid before Q1 stays | Simone chose it himself |
| 11 | e11 | §3 detective beat toy figure | **Drop** the toy figure; keep the three numbered steps | preset ★, not looked at |
| 14 | e14 | §3 brightness violin | **Move to 🔴 (optional)**, not cut | Simone's own choice: "Make optional" |
| 15 | e15 | §4 "What a label costs" (clicks table) | **Move** to the opening of section 8 | preset ★ |
| 3 | e3 | §5 Q13 least-typical photos | **Move to 🔴** next to Q14 | preset ★ |
| 18 | e18 | §5 answer-key figure of the hunt | **Cut** the figure; keep the printed key | preset ★ |
| 22 | e22 | §5 Q10 | **Merge** Q10 into Q11 (exact copies become the first line of the near-duplicate question) | preset ★ |
| 5 | e5 | §6 Q16 + CV table | **Cut** Q16 and the table; keep the 📘 callout and the fold diagram; `cv_schemes` stays computed in a collapsed cell (a test needs it) | preset ★ |
| 4 | e4 | §7 Q20 rabbit quiz | **Cut** the quiz and `QUIZ_ANSWER`; ask it aloud as a show of hands (FACILITATION) | preset ★ |
| 8 | e8 | §7 Q21 | **Cut** Q21; the IoU/Dice figure → go-deeper next to mAP | Simone chose it himself |
| 9 | e9 | §7 second half → §8 | **Not done** (moot once Q21 is cut) | covered by #8 |
| 17 | e17 | §7 shop cell | **Cut** the cell; the SPEC's paragraph stays with the numbers inline | preset ★ |
| 10 | e10 | §8 Q24 crowding table + gallery | **Move to 🔴**; keep the IoU drawing (section 9 uses it) | preset ★ |
| 19 | e19 | §8 Q23 stacked share bars | **Drop** the bars (the table has the same numbers) | preset ★ |
| 2, 32 | e2 | §9 "More about masks" | **Move to 🔴**: per-class mean-mask heatmaps + thin parts (Q28). **Keep Q27** (fill ratio) in the lesson: the Q29 checks use it. Section 8's box-centre heatmap stays | Simone chose it himself |
| 21 | e21 | §9 ✏️ thresholds | **Only `IOU_MIN`** stays ✏️; hard-code the other four | preset ★ |
| 26 | e26 | §9 🔴 Q30 | **Remove** | preset ★ |
| 6 | e6 | §10 Q31 | **Fold** Q31 into Q32 (one header per kind = the plan); renumber after | preset ★; see Open for Q32 |
| 7 | e7 | §10 Q32 checks | **Pre-write** the copy and grayscale checks; students write only the four "two sources disagree" checks | preset ★ |
| 24 | e24 | §10 hunt | **Time-box to 8 min**, then show the reference score and the key (FACILITATION) | Simone chose it himself |
| 27 | e27 | §10 🔴 Q33 | **Remove** (keep the ruler in section 6, Q18) | preset ★ |
| 1 | e1 | §11 second half | **"After the lab" block**: in class only the folder tree, the ✏️ `MY_DATASET_DIR` cell and the report card on the sample; Q34, the "how to read" notes, the guiding questions and Q35 move to an "After the lab" block | preset ★ |
| C §4.3 | eW | whole notebook | **Text diet**: move what goes past the per-section word budget into go-deeper (nothing deleted). Budgets: 0: 400 · 1: 500 · 2: 450 · 3: 700 · 4: 600 · 5: 800 · 6: 800 · 7: 550 · 8: 400 · 9: 650 · 10: 650 · 11: 300 · 12: 250 (≈ 7,000 visible words) | preset ★ |
| 33 | — | SPEC §6 Time column | **Rewrite** from these decisions so it adds up to the real plan | follows |

## Decided: keep as is

| # | Planner id | What | Decision | Note |
|---|---|---|---|---|
| 25 | e25 | §5 🔴 Q14 embeddings | **Stays optional** (not cut). Add a short note for students that this is possible | Simone: *"I would still like to at least keep it there with a note for the students, so that they can get an idea that it is possible to do"* |
| 20 | e20 | §7 `predictions.head()` | **Keep** | Simone looked and kept it |
| 30 | e30 | §8 heatmap + missing-box hunt | **Keep both** in the lesson | Simone looked and kept it |
| 29 | e29 | §1 pre-reading | Keep in class | not looked at → as is |
| 12 | e12 | §3 Q5 odd sizes | Keep | not looked at → as is |
| 31 | e31 | §3 colour half | Keep in the lesson | not looked at → as is |
| 23 | e23 | §4 Q8 | Keep separate | not looked at → as is |
| 28 | e28 | §6 🔴 Q18 ruler | Stays optional | not looked at → as is |
| 13 | e13 | §7 toy 2×2 matrices | Keep | not looked at → as is |
| — | — | Sliders (HASH_THRESHOLD, DOG_SHARE, IOU_MIN) | **None** (G-12 "no Colab sliders" stands) | not chosen |

## Open (for the spec issue)

1. **Q32 optional?** Simone, on e6: *"I think I would make Q32 optional but I like it! It's just that it takes quite
   some time for the students IMO"*. Q32 (with Q31 folded in) is the Challenge's core exercise. Not decided.
2. **Still over the slot.** ≈ 114 lab-pace minutes against 90. The remaining big levers in the planner were: §1 as
   pre-reading (#29), §3 colour half → 🔴 (#31), §8 heatmap + hunt → 🔴 (#30), Q32 optional (above).
3. Second review pass (Simone, in person) starts at Splits & leakage, after these cuts are applied.

## Raw input (verbatim)

```
## EDA teaching notebook: ≈ 114 min lesson (lab pace), ≈ 141 with optional, slot 90
Changes:
- §2 Use one class grid instead of two [e16]: One grid
- §3 Drop the toy figure in the detective beat [e11]: Drop it (preset, not looked at)
- §3 Cut the brightness violin [e14]: Make optional
- §4 "What a label costs" (the clicks table) [e15]: Move to 8 (preset, not looked at)
- §5 Move Q13 (least-typical photos) to 🔴 [e3]: Make optional (preset, not looked at)
- §5 Cut the answer-key figure of the hunt [e18]: Cut it (preset, not looked at)
- §5 Merge Q10 into Q11 [e22]: Merge (preset, not looked at)
- §6 Cut Q16 and the cross-validation table [e5]: Cut (preset, not looked at)
- §7 Replace the Q20 rabbit quiz with a show of hands [e4]: Ask aloud (preset, not looked at)
- §7 Cut Q21 (counting error vs box mistakes) [e8]: Cut
- §7 Cut the shop cell [e17]: Cut (preset, not looked at)
- §8 Q24: crowding table and gallery [e10]: Make optional (preset, not looked at)
- §8 Drop the stacked share bars in Q23 [e19]: Drop (preset, not looked at)
- §9 "More about masks" as an optional part [e2]: Heatmaps + thin parts
- §9 One ✏️ threshold instead of five [e21]: One knob (preset, not looked at)
- §9 🔴 Q30: does the outline follow the photo's edges? [e26]: Remove (preset, not looked at)
- §10 Fold Q31 (the written plan) into Q32 [e6]: Fold in (preset, not looked at). Note: I think I would make Q32 optional but I like it! It's just that it takes quite some time for the students IMO
- §10 Pre-write the copy and grayscale checks [e7]: Pre-write two (preset, not looked at)
- §10 Time-box the hunt to 8 minutes [e24]: 8-min box
- §10 🔴 Q33: the ruler on the dirty and cleaned batch [e27]: Remove (preset, not looked at)
- §11 Turn the second half into an "After the lab" block [e1]: After the lab (preset, not looked at)
- §∗ Text diet: trim every section toward the word budget [eW]: Trim the text (preset, not looked at)
Looked at and kept as is:
- §5 🔴 Q14: embeddings [e25]: Stay optional. Note: I would still like to at least keep it there with a note for the students, so that they can get an idea that it is possible to do
- §7 Move the second half of 7 to the end of section 8 [e9]: (covered by another choice)
- §7 Drop the decorative predictions table [e20]: Keep
- §8 Shrink section 8 to its core [e30]: Keep both
Not looked at (left as is): 6
```
