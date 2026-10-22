# Review · cuts and time (time keeper, pass 2, 2026-10-22)

## In plain words

After the cut pass the lesson path takes about **144 minutes at lab pace** (about **183** at careful pace). That is
roughly 30 minutes more than the planner promised (≈ 114 lab / ≈ 146 careful) and **54 minutes over the 90-minute
slot**. The cuts themselves landed about where the planner said they would. The gap comes from the text diet. The
planner assumed every section would come down to its word budget (about 7,050 visible words in all). The notebook
still shows about **10,940** words in class, and the extra ~3,900 words cost about 17 lab minutes on their own. About
1,570 of those words are worked answers, and the SPEC (line 251) says those stay visible. The four levers still on the
table (§1 as pre-reading, §3's colour half, §8's heatmap and hunt, Q26 optional) save about 23 lab minutes together,
which brings the lesson to about 121. To get near 90, Simone would also have to agree to things outside the decision
record (see §4). Every number here is ±15 %, so before deciding anything else, someone should do **one timed
run-through** with a student or two.

## 1. Minutes per section

How I estimated it, same as pass 1. **Careful pace:** visible markdown at 180 wpm, 0.5 min per figure, gallery or
table, 1 min for a prediction question, 2 for one that needs a calculation or has two parts, 1.5 per ✏️, 3–4 per ✍️,
and 1.5–2.5 per section for the professor's talk (taken from `FACILITATION.md`). **Lab pace** is 0.78 × careful.
The one exception is FACILITATION's **8-minute box on the Challenge hunt**. A box is wall-clock time, so it counts as
8 at both paces: it caps the careful estimate (a careful pair would hunt longer), and lab pace does not shrink it.
That is why the lab column adds up to slightly more than 0.78 × 183. Word counts come from the lint, corrected for
§6 and §11 (see §2). The lesson path leaves out 🔴 parts, go-deeper boxes, `<details>` blocks and the §11 "After the
lab" block. The one exception is §10's `<details>` worked answer, which FACILITATION tells the room to open.

| § | SPEC | Now (careful) | Lab | Over SPEC (lab) | Visible words · figs · Qs | Basis of the estimate | Optional / homework (careful) |
|---|---|---|---|---|---|---|---|
| 0 | 5 | 6 | 4.7 | 0 | 400 · 1 · ✍️ demo | 2.2 read + setup/download/`double()` 2 + `images.head()` 0.5 + talk 1.5 | — |
| 1 | 5 | 8 | 6.2 | +1 | 557 · 2 · Q1 | 3.1 read + 2 figs 1 + Q1 and its checklist output 2 + talk 2 | — |
| 2 | 8 | 10.5 | 8.2 | 0 | 545 · 1 (+ §1 grid) · Q2, Q3, ✏️ | 3 read + ✏️ grid 0.5 + scroll back to §1's grid 0.5 + Q2 1 + ✏️ 1.5 + Q3 writing 3 + talk 1 | — |
| 3 | 8 | 19 | 14.8 | +7 | 1,294 · 9 · Q4–Q6, ✏️ | 7.2 read + 9 figs 4.5 + Q4 1, Q5 1, Q6 2 + ✏️ extremes 1.5 + talk 1.5 | violin 1 |
| 4 | 8 | 14.5 | 11.3 | +3 | 963 · 4 · Q7–Q9, ✏️ | 5.4 read + 4 figs 2 + Q7 1, Q8 1, Q9 (two rules) 2 + `DOG_SHARE` reruns 1.5 + talk 1.5 | — |
| 5 | 15 | 22 | 17.2 | +2 | 1,234 · 7 · Q10, Q11, ✏️, ✍️ | 6.9 read + 7 figs 3.5 + Q10 (two guesses) 1.5, Q11 1 + `HASH_THRESHOLD` reruns 2 + ✍️ 4 + key silence 0.5 + talk 2.5 | Q12, Q13: 6 |
| 6 | 12 | 20.5 | 16.0 | +4 | 1,341 · 8 · Q14, Q15, ✍️ | 7.5 read + 8 figs 4 + trail download 1 + Q14 1, Q15 1 + ✍️ `split_by_camera` 4 + talk 2 | Q16: 2.5 |
| 7 | 7 | 13.5 | 10.5 | +4 | 984 · 6 · Q17, ✍️ | 5.5 read + 6 figs 3 (the IoU/Dice figure still renders outside its go-deeper) + Q17 1 + `ALERT_METRIC` 1 + rabbit show of hands 1 + talk 2 | — |
| 8 | 10 | 19 | 14.8 | +5 | 1,195 · 8 · Q18, Q19, Q21, ✍️ | 6.6 read + 8 figs 4 + Q18 1, Q19 1, Q21 1.5 + ✍️ 3 + talk 2 (clicks opener included) | Q20: 3 |
| 9 | 10 | 18 | 14.0 | +4 | 1,099 · 9 · Q22–Q24, ✏️ | 6.1 read + 9 figs 4.5 + Q22 1, Q23 1, Q24 2 + `IOU_MIN` rerun 1.5 + talk 2 | More about masks (Q25): 5 |
| 10 | 15 | 21 | 18.1 | +3 | 689 (+333 in `<details>`) · 2 · Q26 | 3.8 read + worked answer 1.9 + batch look 0.5 + ten headers 2 + walk headers 1–3 1 + **hunt 8 (box)** + score and key 2 + talk 2 | — |
| 11 | 5 | 7 | 5.5 | +0.5 | 390 in class · 5 · ✏️ | 2.2 read + five check outputs 2.5 + ✏️ `MY_DATASET_DIR` 1 + talk 1.5 | After the lab (Q27, Q28): 10.5 |
| 12 | 2 | 4 | 3.1 | +1 | 249 · 1 | 1.4 read + recap 1 + talk 1.5 | — |
| **Σ** | **110** | **183** | **144** | **+34** | ~10,940 · ~63 · 18 lesson Qs | | **28** (🔴 17.5 + homework 10.5) ≈ 22 lab |

**Totals against the plan and the slot:**

- **Lesson, lab pace:** 144 (±15 % → 122–166). Planner: ≈ 114 (+30). Slot: 90 (+54).
- **Lesson, careful:** 183 (156–210). Planner: ≈ 146 (+37).
- **With the optional parts, lab:** 144 + 22 = **166**. Planner: ≈ 141 (+25). Careful: 211.

The low end of the range is still 32 minutes over the slot, so the uncertainty does not change the conclusion.

**Where the 30 lab minutes went.** About 17 come from the text diet stopping at "moved" (~3,900 words over budget).
The remaining ~13 are things the planner counted as saved that still cost time. The clicks table moved to §8 instead
of leaving. Q27 and its fill gallery stayed in §9. The IoU/Dice figure in §7 still renders. §10's worked answer is
read in class. Each of these is small, and the rest is estimation noise.

## 2. Visible words against the budget

Lint (`uv run python scripts/lint_notebook.py`) prints a WARNING for every section except 0 and 12.

| § | Budget | Lint | Corrected | Over (corrected) |
|---|---|---|---|---|
| 0 | 400 | 400 | 400 | 0 |
| 1 | 500 | 557 | 557 | +57 |
| 2 | 450 | 545 | 545 | +95 |
| 3 | 700 | 1,294 | 1,294 | +594 |
| 4 | 600 | 963 | 963 | +363 |
| 5 | 800 | 1,234 | 1,234 | +434 |
| 6 | 800 | 1,112 | **1,341** | +541 |
| 7 | 550 | 984 | 984 | +434 |
| 8 | 400 | 1,195 | 1,195 | +795 |
| 9 | 650 | 1,099 | 1,099 | +449 |
| 10 | 650 | 689 | 689 | +39 |
| 11 | 300 | 1,243 | **390** | +90 |
| 12 | 250 | 249 | 249 | 0 |
| **Σ** | **7,050** | 11,564 | **10,940** | **+3,890** |

Two lint quirks:

- **§6 undercounts by 229 words.** The 🔴 Q16 ruler part has no closing heading, so the lint treats the Across-tasks
  callout and the Takeaway after it as optional. The real 🔴 part is 118 words.
- **§11 counts the After-the-lab block** (853 words). The in-class part is **390** words by the lint's own method. The
  ≈ 330–345 in the progress file looks low, because the ✏️ "Point the notebook at…" callout alone is ~100 words.

The biggest overshoots are §8 (+795), §3 (+594), §6 (+541), §9 (+449), §5 and §7 (+434). About 1,570 of the 3,890
extra words are worked answers, which SPEC line 251 keeps "shown openly", so the budget cannot be reached by moving
text alone. One thing I noticed: §10's worked answer sits in `<details>`, which goes against that same line. That may
well be on purpose (it is the Challenge key), but it is worth a sentence in the report.

## 3. What Q26 costs (the Challenge hunt, old Q32 with Q31 folded in)

If Q26 becomes 🔴 optional, §10 loses the ten-header read (2), the walk through headers 1–3 (1), the 8-minute hunt
box and Q26's own text with the your-turn callout (~180 words, ~1).

- **Saved: 12 min careful, 11 min lab** (8 of them are the box, which does not shrink at lab pace).
- **What is left in §10 (≈ 9 careful / 7 lab):** the data-centric callout, the new batch with its tables and a dozen
  photos, the reference solution's score line ("found 14 of 14"), `reveal_answers()` with the worked answer, and the
  takeaway. That turns §10 into a professor-led demo of the key. The one idea, "four kinds only show when two sources
  disagree", is told rather than found. Students lose the only exercise that combines tools from sections 3 to 9.
- If he goes this way, the SPEC objective for §10 ("Find the planted problems…") and FACILITATION's §10 row need
  rewording.
- A middle option, not in the record: keep Q26, but cut the box to 5 minutes and do kinds 6 and 8 together on screen.
  That saves about 4 lab minutes.

## 4. The remaining levers

| Lever | Careful | Lab | Where it stands in the record | Catch |
|---|---|---|---|---|
| Q26 optional | 12 | 11 | Open question | See §3 |
| §8 box-centre heatmap + missing-box hunt (Q21, toy drawing) → 🔴 | 5.5 | 4.3 | #30: Simone **looked and kept both** | Challenge kind 9 is a stub that leans on §8's hunt; it would need pre-writing |
| §3 colour half (modes, gray gallery, RGB histograms, ✏️ extremes) → 🔴 | 5 | 4 | #31: not looked at | §10 kind 7 and §11 check 2 point back to §3's grayscale; touches R-14/R-17/R-19 |
| §1 as pre-reading (in class: the class grid and Q1 only) | 4.5 | 3.5 | #29: not looked at | The grid has to stay in class, because Q2 now reads it |
| **All four** | **27** | **≈ 23** | | Lesson → **≈ 121 lab** |

Other sizeable levers. **None of these is in the decision record.**

- **Finish the text diet.** eW is in the record, but only partly done. About 2,300 of the extra words are not worked
  answers and could move into go-deeper boxes: ≈ 13 careful / **10 lab**. Moving all 3,890 would give ≈ 22 / 17, but
  that means relaxing SPEC line 251.
- **§9 becomes "after the lab", except Q22 and the pitfall demo** (≈ 4 lab left in class): ≈ 13 careful / **10 lab**.
  Challenge kind 10 (ugly mask) would then need pre-writing.
- **§11 entirely as homework** (show the folder tree only): ≈ 6 careful / **4.5 lab**.
- **One ✍️ becomes run-and-read** (`find_flipped_copies` in §5; FACILITATION gives it about 5 minutes): ≈ 3 careful /
  **2.5 lab**.

Taking the four levers plus all of these gives ≈ 94 lab. Getting near 90 means taking essentially everything on this
page.

## 5. Proposed SPEC Time column

The SPEC column today is "participant minutes", whole numbers, and adds up to "~110 total". Pass 1 read that figure as
lab pace, so I keep the same unit: lab pace, whole minutes, lesson path only. The values are rounded from the Lab
column and add up to the lesson total.

| § | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | Σ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Time | 5 | 6 | 8 | 15 | 11 | 17 | 16 | 11 | 15 | 14 | 18 | 5 | 3 | **144** |

Two follow-ups for whoever edits the SPEC. The note above the table should read something like "~144 total at lab pace
(🔴 parts and the After-the-lab block, ≈ 22 more, not included)". FACILITATION's Min column still carries the old
values that add up to 110, and it should follow the SPEC (N-033).
