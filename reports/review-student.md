# Student review (Master's student, ~90-minute lab)

File refs are `NN:line` = `sections/NN_*.py:line`. Overall: the story, the worked answers and the toy pictures help a lot. But I counted 35 questions, 4 functions and 8 edit-cells for 90 minutes (SPEC §6 even adds up to 110), so I would be reading answers instead of thinking by section 6.

## 1. Where I get lost
- I am told to "write down your guess" about 25 times (Q1, Q4, Q5, Q7 ... Q30) and there is nowhere to write. Only Q3 (02:168) and Q35 (11:347) have a cell. I stop guessing by Q8, because the worked answer sits right under the output.
- The "Reference solution — peek only if you are stuck" cells (00:213, 05:479, 06:481, 08:101) look optional, but they define `_ref_*`. If I skip one, `use(...)` crashes with a NameError. The title says the opposite of what I must do.
- 01:190 says "weighted loss? section 4". Section 4 never explains it (04:234 only says "next week's topic"). "Iterative stratification" shows up at 06:603 and is first explained at 08:417.
- 03:375 and 03:499 promise that grayscale and odd-size photos get decided in "section 5". Section 5 ends with "keep everything else" (05:683). I was left waiting for a decision that never comes.
- 05:148 "8×8 lowest frequencies of its DCT": I have never heard of a DCT. The Hamming formula at 05:151 is maths for something that is just "count the differing bits".
- 05:370 Q12: "which of the four steps above (twice, smaller, mirrored, rotated)". The list at 05:298-303 has five items, and the fifth is a file-naming one. The "hunt" (05:305 "find every pair without opening 40 photos") is not a hunt, because the Q12 cell just prints the pairs (05:420).
- 06:42 and many later places say "test half", but the split is 80/20. I kept thinking 50 %.
- 06:544 "leave-one-camera-out", 06:134 "LOGO", 06:570 `GroupKFold`: three names for what looks like one idea.
- Violin plots (03:393, 08:191, 09:311) are never explained. I only learned how to read them from the 💡 under the second one.
- 09:596 says "you have four checks", but the Q29 cell prints five (box IoU, fill, pieces, holes, solidity). I don't know what "rubber band" (solidity) means until later in the same cell.
- 09:616-620: five threshold variables with no hint which one matters. Q29 asks for five (kind → check) pairs before I have seen any output.
- 04:160-167 Q9 mixes two rules ("always the biggest class" on the Play Dataset, "always sheep" on the paddock). I could not tell which rule the first answer was for.
- 11 assumes I have my project dataset ready in Drive (11:25). The "Before you start" box (00:22) never says to bring one. 12:97 mentions it, but only at the very end.

## 2. Where I get bored
- "Always answer sheep, accuracy looks great" is told 4 times: 04 pitfall + Q9 + photo strip, 07:11-71 toy, 07 ALERT_METRIC, 09:244 pitfall. After the second time I skim.
- The burst/leak story is retold at 05:727, 06:19, 06:306, 06:441, 06:431. Same tree, same raccoon, five times.
- Almost every Q has a 💡 and a "Worked answer" that say the same thing (04:222-235, 06:105-116, 07:189-199, 09:714-728). I read it twice.
- Heatmaps: 08:272-312 and 09:355-393 are the same idea (where the animals are) twice.
- Section 3 (631 lines, 8 min) has eight statistics, a colourfulness formula, a violin, extremes, a detective toy, Q5 (same lesson as Q4), a box-rescale, and a bilinear formula (03:611-623) I will never use here.
- 01:15-33 is 11 links I will not click in class. 01:40-72 is 10 bullets, each with a "Park example".
- 05 (753 lines) and 09 (828 lines) are the heavy ones: three hash demos, an embeddings detour, then Canny edges in 09:738.

## 3. Main vs secondary (N-048)
- Good: Objective + Takeaway ("When you see X → do Y") alone give me the lesson of each section. I like that.
- The ❓ Questions callout holds only a preamble ("Predict before you run..."). The actual Q is a plain cell below it (e.g. 04:28 vs 04:33). Reading callouts only, I never see what I am asked. 06:147, 06:347, 07:306 have no callout at all.
- Go-deeper is used for things the Challenge needs: the tool list at 10:309-317 is muted grey, but I cannot do Q32 without it. The 20.8 % → 41.1 % paper numbers (06:431-438) are the punchline of section 6; Q18's answer (06:596) refers to them, and I might have skipped the grey box.
- 🔴 parts have no visual mark. They are a plain heading (05:626, 09:738, 10:513) or a bright blue Theory callout (06:540, the ruler), so they look as important as the rest. In 10 the same ruler is grey go-deeper (10:516). Where does the optional part end? I can't tell.
- 08:407 and 09:458 ("What handles each later") are named "Across tasks" but are really "names to google next week" (weighted loss, focal, tiling, multi-scale, Dice loss ...). That is secondary and should look it.
- 00:30 and 00:39 (the park story) are "Theory" callouts. They are not theory. Theory-blue then stops meaning "concept".
- Main content in plain text, not a callout: the guided-hunt stories (05:294, 08:429, 10:27), the leak steps (06:309). Worked answers look like main text, so I cannot tell skippable from required.

## 4. Unclear sentences (worst 14)
1. 04:71-73 "...Here it is dog that matters: well represented in the photos, yet the *rare* event in the paddock." → "Look at the ratio. Then ask whether the smallest class is the one you care about. Dog is common in this dataset but will be rare in the paddock."
2. 05:541-543 "Every planted copy is a pair the detector reported, and the pairs it reported are only the planted ones: insist on that..." → "The detector found all 5 planted copies and reported nothing else. Test your own detector the same way: plant a few copies and check it finds them and nothing more."
3. 05:147-148 "keep the 8×8 lowest frequencies of its DCT... above the median or not" → "Describe the picture as waves, keep the 64 slowest (the big blobs), and write 1 where a wave is above the median, otherwise 0."
4. 05:151 "distance = Σ [bitᵢ(A) ≠ bitᵢ(B)]" → "Count how many of the 64 bits differ: 0 = same picture, about 32 = unrelated."
5. 06:41-43 "a random test half of 24 photos" → "a random test set of 24 photos (20 %)".
6. 06:46-48 Q15 "which class share moves the most... *relative to the class's own size*" → "For each class compare its share in the test set with its share overall. Which one is off by the biggest fraction of itself, the biggest or smallest class?"
7. 06:147-149 "smallest class share any test fold has" → "In every fold, find the class with the fewest test photos. The table shows the lowest of those shares; 0 % means some fold had no photo of some class."
8. 03:313 Q4 answer: "Neither one point nor a cloud: ... two lines, an L shape" → the options were point/line/cloud. Say "closest to 'a line', but two lines meeting in an L", or add that panel to the hint.
9. 07:306-308 Q20 "one rabbit is shown dog food... Which metric did the shop just lose" → "In a week: 10 rabbits and 100 dogs. The classifier calls one rabbit 'dog' and gets every dog right. Which two per-class numbers fall?" (and mark it 🟡, not 🟢).
10. 07:387-388 Q21 "how many mistakes would mAP see?" mAP does not count mistakes. → "How many individual box mistakes (missed or false boxes) are there?"
11. 08:311-312 "...a flock fills the frame, the one bias the Play Dataset shares with the paddock." → "Bird and sheep boxes spread out more because a flock fills the frame. That is the one way this dataset resembles the paddock."
12. 09:27-28 "The loss hears the horse a thousand times louder." → "The horse adds 1,000 times more pixels to the training error than the bird, so the model mostly learns horses."
13. 09:449-450 "a pixel-counting loss barely notices... and IoU barely moves" → "Thin parts are so few pixels that a model can get all of them wrong and still score well."
14. 03:377-379 "a class a model can recognise from colour alone. That is a shortcut" → "If the colour curves differ per class, a model can cheat by using colour instead of the animal. A dog on green grass might get called 'sheep'."

## 5. Tasks: clear or not?
- Clear and I know when I'm done: ✍️ in 00 (`double`, ✅/❌ line), 05:498 (`find_flipped_copies`), 08:108, 02 Q3 (three lines), 11 `MY_DATASET_DIR`.
- 02:141 ✏️ and 03:421 ✏️: I know what to type, but not what to look for. "Try dog and bird" is not a goal. A question like "which class has the darkest photos?" would give me one.
- 04:148-157 ✏️ `DOG_SHARE` sits above Q9 and says "re-run the Q9 cell". Do I also re-run the ✏️ cell? Say "edit, run this cell, then re-run Q9". Same at 05:213-216 (HASH_THRESHOLD): "watch what enters the list" does not say which cell to re-run.
- 06:500-515 ✍️ `split_by_camera` is one `isin` line, and then I must also choose `TEST_LOCATIONS`. I am never shown which classes are at which camera, so 06:532-535 ("the cameras that see all the coyotes") is a guess.
- 07:269 ✍️ `ALERT_METRIC = ...` is a one-word choice that repeats Q19, and the answer is printed right under it.
- 07:320 ✏️ `QUIZ_ANSWER`: fine, but it is the 2nd edit-cell in one section.
- 08:491 Q25 is answered by the figure just above it (08:486 title: "difference 1 → open this photo").
- 09:615 ✏️ has five variables, SPEC says at most one per section. "Make one stricter and one looser" is the task, but I don't know what to conclude.
- 09:409 Q28 "is it for the same reason as the runner-up?" I can't check my answer without reading the worked answer.
- 10:324 Q31 "write ten lines kind → tool → what it reads": where? 10:330 Q32 "in the next cell" has only the first check; the rest are comments. Done = "found k of n" per kind, which is good, but it is a big jump from the earlier scaffolding. A second worked line would help.
- 11:118 Q34 (predict for your data AND for the sample) has no place to write; 11:344 Q35 has a clear "Your dataset:" slot.
- Q13 (05:580), Q31, Q34: "note", "write" with no cell.

## 6. Concrete suggestions
1. **[clarify]** One rule for predictions: "Think of your answer (no need to type it), then run." Or give one `MY_GUESSES` markdown cell per section. 00:19 or section header.
2. **[clarify]** Rename the reference cells "Reference solution (run me; read only if stuck)" or auto-define them in the setup cell. 00:213, 05:479, 06:481, 08:101.
3. **[clarify]** Make the Q itself part of the ❓ callout (question inside the box, not only the preamble). All sections.
4. **[clarify]** Mark 🔴 parts with a callout or `<details>` from heading to end (05:626-677, 06:537-596, 09:738-817, 10:512-581), and use the same kind in 06 and 10.
5. **[clarify]** Move the 10:309 tool list out of go-deeper into a plain callout (your-turn or recipe). Move the 20.8 → 41.1 sentence (06:434) into the main text of Q17's answer.
6. **[clarify]** Say in 03:375/499 "we decide in section 5" and then do it in 05:682 (one line: "grayscale: convert to RGB; tiny files: keep"), or remove the promise.
7. **[clarify]** Say "test set" (not "test half") 06:105, 06:182, 06:471 etc.; use one name for LOGO (06:134, 06:544, 06:570).
8. **[clarify]** Fix "four checks" vs five (09:596); fix "four steps above" vs five items (05:370); fix Q4 answer vs options (03:313).
9. **[clarify]** Add to "Before you start" (00:22): "Bring your project dataset as a folder per class (section 11)." Move up the Drive mount advice.
10. **[add example]** One sentence at the first violin (03:385): "the wider the shape, the more photos at that value; the dark line is the median."
11. **[add example]** Two more lines of starter code in 10:335-345 (grayscale and missing box) so the Challenge starts with three solved checks, not one.
12. **[add example]** Say in 05:148 what the bits mean in plain words (see 4.3), or move the DCT sentence to a go-deeper and just show the 8×8 figure.
13. **[shorten]** 01:15-33 (two link lists) into one go-deeper line; 01:40-72 to 5 bullets without "Park example" repeating the bullet. ~2 min.
14. **[shorten]** Merge each 💡 and "Worked answer" into one 2-3 line answer (sections 4-9). ~5 min over the notebook.
15. **[shorten]** Cut the "always sheep" story to one place (04 Q9) and refer back from 07:11 and 09:244 in one sentence. ~3 min.
16. **[shorten]** Make the leak story appear once (06:19-29 + 06:309 steps). Delete 06:431 repeat and the 05:727 retelling, or keep one-liners. ~2 min.
17. **[collapse/go-deeper]** 03:611-623 (bilinear formula, "why not round"), 03:35-36 (colourfulness formula). ~2 min.
18. **[collapse/go-deeper]** 08:407-420 and 09:458-470 ("handled later" jargon lists); mention only the names. ~1 min.
19. **[collapse/go-deeper]** 06:146-191 (LOO/Q16 table) and 05:593-677 (embeddings, Q14). Q14 is already 🔴. ~4 min.
20. **[collapse/go-deeper]** 09:395-456 (thin parts, Q28) into a go-deeper with its demo. ~3 min.
21. **[remove]** 03:473-504 Q5 (same lesson as Q4). ~2 min.
22. **[remove]** 07:386-439 Q21 + IoU/Dice plot (the plot has nothing to do with the question; IoU is taught in 08:321 anyway). ~2 min.
23. **[remove]** Either the ✍️ `ALERT_METRIC` (07:268-301) or the quiz (07:304-352). ~2 min.
24. **[remove]** 09:355-393 (second heatmap) or 08:271-312. ~2 min.
25. **[remove]** 05:422-464 ("one photo, five kinds of copy"; it repeats Q12's table). ~2 min.
26. **[remove]** 09:738-817 (Canny edges, Q30) and 10:512-581 (Q33 ruler) from the 90-minute path. ~6 min together.
27. **[shorten]** 09:615-620: keep only `IOU_MIN` as ✏️; hard-code the other four. ~2 min, and it matches the one-✏️ rule.
28. **[clarify]** Time budget: if the lab is 90 min, SPEC §6 needs to lose ~20 min. Suggestions 13-27 are about 33 min.
29. **[clarify]** The "1 : 10 : 100" claim (04:252) is about seconds, but the table under it (04:256-272) counts clicks, so the two may not match. Say which one is meant.
30. **[clarify]** Split buried headings from answers: 05:139-140, 06:477-478, 07:255-256 put "###" inside the worked-answer cell, so the TOC jumps into an answer.
