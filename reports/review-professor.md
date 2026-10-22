# Professor review (read-only pass, 2026-10-21)

Read as someone who has to say this out loud to ~25 Master's students in 90 minutes while they run it in Colab. Line numbers are the current `sections/*.py`. Minute figures are my own guesses at lab pace, not measurements.

**Verdict in one breath.** The story (park, dog alert, "same photos, new unit") carries the whole thing and I can talk to it. The problem is volume, not logic: my honest aloud-plus-run time is about 160 min (0:7 1:9 2:9 3:15 4:10 5:20 6:19 7:12 8:14 9:19 10:20 11:6 12:3), against a 90-minute slot and a SPEC budget of ~110. The cuts below save roughly 35 min if all are taken, about 11 of them 🔴 parts nobody runs in class. That still leaves ~125, so Simone has one real decision left: which of 8, 9 or 11 becomes "read at home". See item 29.

## 1. Flow when explained aloud

- **0 Setup.** Two things break the opening. The banner is below the Objective box (`00_setup.py:5-19`), so the first thing I read is a one-line objective, not the title. And 9 markdown cells (banner, prerequisites, three park callouts, prose, ...) pass before the first code cell at `:56`. FACILITATION says "start the data cell before the story", but the download cell is at `:358`, after the story, so I cannot. The `double()` demo (`:205-226`) sits between setup and data and delays the download further.
- **1 Image datasets.** I lose the thread at `01_image_datasets.py:39-72`: a 10-bullet, ~33-line callout read aloud, then label cell, then photos, then Q1, then a printed checklist (`:169-216`) that repeats the same 10 bullets again. Two passes over one list. The photos come after the theory, so "show before asking" (R-11) is met for Q1 but not for the checklist itself.
- **2 Become one.** Best flow in the notebook: quote, predict, look, write. Nothing to fix.
- **3 Inspection.** Flows until `:229` (min/median/max figure), then I have three "look at extremes" moments in a row: the figure, the brightness violin (`:389`), the ✏️ `show_extremes` (`:421-429`). The jump into "Across tasks · Resizing" at `:507` is abrupt: boxes and masks have not been introduced as units yet (that is sections 8 and 9), and Q6 then packs a box formula, a mask, nearest vs bilinear, a heatmap, a zoom, a formula and a "why not round" into 7 cells (`:522-623`). It reads like a second section.
- **4 Labels.** Clean: bars, multi-animal, pitfall, Q9 strip. The only wobble is the ending: "clicks per task" (`:237-276`) does not follow from imbalance and the takeaway does not mention it. It is really the opening of section 8.
- **5 Errors.** Long but the ladder (MD5, pHash, variants) holds. I lose it at the end of the memory-card story: the plain-vs-variant shoot-out (`:414-464`) already reveals why variants work, and only then do I ask them to write the flip detector (`:476-513`), which is a one-liner that repeats it. The hunt is told twice (shoot-out, then score and key `:519-538`). The 🔴 Q14 block (`:593-677`) re-runs Q13 on colour histograms when embeddings are absent, which is the usual case.
- **6 Splits.** First half is words (callout `:18-29`, prose, CV callout `:125-134`, go-deeper) and the first real photos arrive at `:296`. Q16's table (`:155-179`) answers "0", a trivial result for a whole cell. After that, trail cameras are the best part of the notebook.
- **7 Metrics.** Toy 2x2 then 6x6 matrix is a good build-up. But the second half (`:354-439`: counting error, IoU, Dice, mAP, Q21) previews sections 8 and 9 before there are any boxes. I have to say "we'll see boxes later" twice.
- **8 Boxes.** Good. Two spoilers: the story steps (`:429-435`) and the toy figure titled "difference 1 → open this photo" (`:486`) both give away Q25's answer before it is asked.
- **9 Masks.** Flow is fine up to the hunt. The four checks (`:595-606`) are only text, but I would be drawing them on the board (FACILITATION says "one picture each"). Thin parts and per-class mask heatmaps repeat points made in sections 3 and 8.
- **10 Challenge.** Seven markdown cells (~60 lines) before the first batch photo (`:302`); the ten kinds are listed three times (`:32-37`, `:44-54`, `:324-326`). Q32 is a nearly empty code cell with comments only (`:335-345`), and 8 checks have to be written from scratch: the biggest blank-page moment. The tools list is a muted grey Go deeper (`:309-317`) but students need it to do the task.
- **11, 12.** Fine as a take-home. 11 re-runs the charts of sections 3-6 on a sample, so in class it is a replay. 12 says the three outcomes a third time (banner, section takeaways, final takeaway).

## 2. Scrolling

- **Biggest offender:** `show_min_median_max(STAT_COLUMNS)` (`03_image_inspection.py:171-200`, called `:230`). 7 statistics + 2 odd rows = 9 rows x 1.9 in = a ~1700 px tall figure. I cannot keep the table above it in view.
- **Gallery stacks:** Q13 shows 24 photos (`05:555`, `N_ATYPICAL=4` x 6 classes) after a scatter nobody will discuss. Q29 shows 8 suspects then 10 key panels (`09:661-666`, `:688`). Q5 shows 12 odd-size photos (`03:493`).
- **Three cells per question** in many places: a blue "❓ Questions" callout with one sentence, then the `**Qn.**` text cell, then the code cell (`04:27-34`, `06:39-48`, `08:83-92`, `09:167-174`, `07:119-126`). The two text cells could be one box.
- **Headings buried** at the bottom of 💡 cells where I scroll past them: `03:237`, `05:140`, `05:476`, `05:545`, `06:478`, `06:537`, `07:256`. Give each its own cell above.
- **Long markdown before code:** `01:39-72` (33-line callout), `06:125-134`, `10:9-54`, `00:29-49`.
- **Section 9 has ~14 figures** and section 3 ~11 for 10 and 8 minutes. Not everything can be talked about.

## 3. Balance of text and pictures

- **0, 1, 10 (opening):** text heavy, almost no pictures. 0 could use one small picture of the four cameras and the three jobs (SPEC says "illustrated paragraph"; there is none). 1's callout is words where the printed checklist already carries the data.
- **2, 4, 7, 8:** about right. In 4 the strip of "sheep ✔/✘" photos is the memorable moment. In 7 the dashed accuracy line vs the dog bar carries it.
- **3:** pictures outrun the talking. Colour modes + 6 RGB histograms + grayscale gallery in one cell (`:326-371`) is three stories at once. The statistic definitions (`:24-38`, 15 lines) are text where the min/median/max picture already shows each one.
- **5:** the pHash diagram (`:155-210`) and the five-kinds figure (`:427-464`) earn their place. The atypicality scatter (`:561-572`) and the Q11 histogram get one sentence each from me.
- **6:** words first, pictures late. The four CV schemes are the one thing that really wants a diagram (coloured folds), and there is none; the leakage message is shown three times (diagram `:316`, bar `:373`, photos `:387`).
- **9:** the 4 checks need a picture, thin parts and per-class heatmaps have too many. Nobody will discuss the Q26 two-panel chart's right half.
- **11:** all text and outputs of old charts.

## 4. Interaction (R-32), at most 3 proposals

My caveat first: "no sliders, they can write code" was about *writing* code. These three are places where the learner only retypes a number (and sometimes mistypes it) to see a trade-off, so a slider shortens the loop. Note that Colab forms do not rerun anything on drag, so the `@param` must sit in the same cell as the plot, with the slow part (hashing, mask measuring) cached in an earlier collapsed cell. The ✏️ cell would use `display-mode: "both"`.

1. **`HASH_THRESHOLD`, `05_errors_cleaning.py:220` (+ Q11 cell `:230-273`).** Slider 0-24, step 1, default 8. Q11 literally tells them to try 2 and 14 and watch the list. Dragging moves the dotted line on the histogram, the pair count and the pair table at once. Split the hashing loop (`:231-238`) into its own cell so only the thresholding reruns. Same variable is reused in 5, 10 and 11, so one definition.
2. **`DOG_SHARE`, `04_labels_distribution.py:154` (+ Q9 cell `:170`).** Slider 0.00-0.50, step 0.01, default 0.02, moved into the Q9 cell. The sheep/dog strip gets more dogs, accuracy falls from 98 % towards 50 %, dog recall stays 0 %. That is the whole point of the pitfall, shown rather than described. Today it is two cells and three retyped values (0.02, 0.2, 0.5).
3. **`IOU_MIN`, `09_masks.py:616` (+ Q29 cell `:636-666`).** Slider 0.50-1.00, step 0.01, default 0.90. "Stricter and looser, rerun this cell and the next" is the section's learning goal, and it is the most error-prone retype (5 constants of mixed types). Cache `mask_shape` for all judged masks in a collapsed cell so a drag only recomputes `CHECKS`; the table shows "planted found" vs "real suspects" move opposite ways. Leave the other four thresholds as plain code.

Not proposed: dropdowns for `CLASS_TO_SHOW` (`02:143`) and `STAT_TO_SHOW` (`03:423`). They would remove typos but save little.

## 5. Concrete suggestions

Minutes are estimated saving at student pace; removals are proposals for Simone only.

1. **[reorder]** `00_setup.py:5-19`: banner first, Objective after it. Move Setup (`:56`) and the data cells (`:238-382`) straight after the banner so the 1-2 min download runs while I tell the park story; move the `double()` demo (`:205-226`) after the data cell.
2. **[shorten]** `00_setup.py:29-49`: merge the two park callouts and the prose into one callout (camera → job → what matters, 3 lines). 0.5 min.
3. **[collapse/go-deeper]** `01_image_datasets.py:14-33`: 11 links → one muted Go deeper at the end of the section (or the repo README). 1 min.
4. **[reorder]** `01:74-146`: label example and the 3-per-class photo grid before the checklist callout (`:39-72`), so I show before I list. Then Q1 and the printout.
5. **[shorten]** `01:39-72`: callout to bold point + one park clause (~14 lines); the data lives in the printed checklist (`:169-214`), which stays. Stop repeating every bullet in both. 1.5 min.
6. **[shorten]** `03:220` / `:230`: `show_min_median_max` for `aspect, megapixels, brightness, colourfulness` only (width, height, contrast stay in the table), keep the two odd rows. Halves the tallest figure. 1.5 min.
7. **[remove]** `03:388-413` brightness violin, because brightness extremes are already shown twice (figure, ✏️ extremes). Proposal only; R-18 asked for violins and `08`/`09` have them. 1 min.
8. **[collapse/go-deeper]** `03:583-599` (6x6 zoom), `:610-623` (bilinear formula, "why not round"): one `<details>`. The 4-panel figure plus the plain sentence "bilinear invents in-between values" is enough aloud. 1.5 min.
9. **[clarify]** `03:522-533`: Q6 asks two questions (box; mask value counts). Split into Q6a/Q6b or show the box answer first, then ask the mask question.
10. **[shorten]** `03:437-503`: keep the numbered detective steps, drop the toy figure (`:451-468`) or cut Q5's gallery from 12 to 6 (`:493`). Grayscale gallery, toy and odd sizes all say "a cluster is a clue". 1 min.
11. **[clarify]** Pull buried headings into their own cell: `03:237`, `05:140`, `05:476`, `05:545`, `06:478`, `06:537`, `07:256`.
12. **[reorder]** `04:237-276` ("What a label costs", clicks table) → open of `08` (before `08:10`), shortened to one table and the 1 : 10 : 100 line. Ends section 4 on the dog, which is where the takeaway already points. 1 min net.
13. **[clarify]** Merge each "❓ Questions" callout with its `**Qn.**` text into one callout cell: `04:27-34`, `06:39-48`, `07:119-126`, `08:83-92`, `09:167-174`, `10:319-333`. Saves a screen per question.
14. **[reorder]** `05:476-513`: put the ✍️ `find_flipped_copies` right after the card is planted (`:362-367`), before the Q12 shoot-out (`:414-420`), so the exercise is a prediction and the five-kinds figure (`:427-464`) is the explanation.
15. **[remove]** `05:528-538` key figure (keep the printed key): the five-kinds figure already shows the same copies. 0.7 min. Also `07:117` `predictions.head()` (decorative table).
16. **[shorten]** `05:555`: `N_ATYPICAL` 4 → 2 (24 → 12 photos) and drop the scatter (`:561-572`). 1.5 min.
17. **[collapse/go-deeper]** `05:593-677` (Q14, embedding helpers, go-deeper): one Go deeper `<details>`, since the usual case is the colour-histogram stand-in that repeats Q13. 3 min if run, ~85 lines of scroll either way.
18. **[add visual]** `06:125-134`: one fold diagram (four rows: k-fold, stratified, LOO, LOGO; test block in magenta) replaces the four bullets. **[remove]** Q16's table cell (`:146-179`), whose answer is "0 because one photo per fold" (the diagram shows it). 2 min.
19. **[shorten]** `06:373-380`: fold the Q17 two-bar chart into the LOO/LOGO cell (`:456-472`) as one four-bar chart (random, by camera, LOO, LOGO); the toy diagram (`:316`) and real photos (`:387`) stay. 1 min.
20. **[reorder]** `07:354-439` (counting error, IoU, Dice, mAP, Q21) → end of section 8 (or the IoU drawing at `08:325`). Section 7 then ends on recall/precision + takeaway. The four IoU introductions (`07:361`, `07:399`, `08:321-325`, `09:598`) become two. 3 min in 7 (it moves, ~1.5 net). Update the "(section 7)" pointer at `09:598`.
21. **[remove]** `07:303-352` Q20 rabbit quiz and `QUIZ_ANSWER` check: it introduces a new class for a row/column drill the matrix already shows, and I can ask it aloud by show of hands. 2 min. Proposal only (it is an interaction).
22. **[reorder]** `08:469-489`: move the toy missing-box figure after Q25's answer and drop "→ open this photo" from its title; also trim step 3 of `08:429-435`. Q25 is currently answered before it is asked.
23. **[shorten]** `08:216-221`: drop the stacked-share bars from Q23 (the table above has the same numbers); keep violin + smallest/median/largest. 0.7 min.
24. **[collapse/go-deeper]** `08:406-420` and `09:458-470` ("what handles each later": weighted/focal loss, tiling, multi-scale, iterative stratification, Dice): jargon students cannot use yet. One shared Go deeper, or move to the Next week part of 12. 1.5 min.
25. **[add visual]** `09:595-606`: one 4-panel schematic of the checks (tight box vs drawn box, fill, a hole, a convex "rubber band"), using one real or toy mask. It pays back time in Q29.
26. **[collapse/go-deeper]** `09:395-451` (thin parts: 3 visuals; the resize argument is already in section 3) and `:356-393` (per-class pixel heatmaps, same story as `08:272-312`). 4 min together. Proposals only; the "bird by pixels" thread survives in Q26.
27. **[shorten]** `10`: Q31 should say "the ten kinds above" instead of re-listing (`:324-326`), and the list at `:44-54` has 9 bullets for "ten kinds" (flipped/rotated share one). **[clarify]** change the `:309-317` tools callout from `go-deeper` to a visible kind (`your-turn` or `theory`). **[add visual]** give Q32 a skeleton: one `# kind` header and a stub line per kind, so the blank page is eight lines to fill. 1 min.
28. **[remove]** both 🔴 rulers, `06:537-596` (Q18) and `10:512-581` (Q33). Never run in class. About 6 min if run and ~100 lines of scroll saved. If kept, say "ruler, not training" once in the cell and shrink.
29. **[collapse/go-deeper]** Section 11 as "take-home": keep the folder layout, the `MY_DATASET_DIR` cell and the 7 guiding questions in class; the four checks (`:127-300`) run on the sample only at home. 5 min. If more time is needed, section 8 or 9 is the next candidate: keep the hunt and the "same photos, new unit" idea, push the rest into reading.
30. **[shorten]** `12:65-98`: merge augmentation and MLOps into one Go deeper, keep the "flipped copies are what augmentation makes on purpose" sentence. 1 min.
31. **[clarify]** Numbers in prose that must match output when I read them aloud: `08:529` ("9 % of photos, 22 % of boxes"), `11:357` ("130 ... 59"), `04:66`, `09:238`. Compute them from the tables or mark "about".
