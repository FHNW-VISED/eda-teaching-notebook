# Is everything Susanne wrote still there? An independent check (2026-10-29)

**Short answer: yes.** Every one of her 227 cells is either still in the Teaching notebook with the same
student-visible text, changed only where an approved item says so, or removed by an approved decision. Nothing
she wrote was lost or reworded without an item. Six differences are **unsure**: they come from approved work,
but no item names their visible effect (table below). No difference is **unapproved**.

| her cells | kept identical | changed, approved | changed, unapproved | unsure | removed, approved | removed, unapproved |
|---|---|---|---|---|---|---|
| 227 | 105 | 85 | 0 | 4 | 33 | 0 |

Plus 2 current cells with no counterpart in her copy that are unsure (and 3 that are approved).
**Printed outputs:** 31 code cells print text; 14 of them differ from her saved Colab run, in 29 places: 27 are
approved items (T-05, T-13, N-21, C-07, C-08, T-10, C-22/R-08) and 2 come from the run environment. Every other
printed number is identical to hers (633 images, 1540 annotations, 241 of 633, 0 byte-identical groups, 3 of 5 vs
5 of 5, the within-class medians from cat 0.144 to zebra 0.549, 400 trail-camera images in 142 bursts, 59
suspects of 1052, …). Figures: 47 in her run, 47 in the current run, the same count in every cell.

"Kept identical" means the text a student sees did not change. The hidden code changed in 117 of the 129 code
cells (box cells now call `callout()`, imports moved to setup, renames); it is unchanged only in her cells 2, 4,
11, 12, 42, 43, 52, 56, 59, 64, 131, 132.

## How this was checked

The script `build/verify_her_text.py` lines up her cells with the current ones (by order, then by content where
cells were inserted, removed or turned from a box into markdown) and compares, for each pair, what a student
reads: markdown in full; the `# @title` line; the text of every box as rendered (her saved Colab output vs the
current run, HTML tags stripped); the Colab form fields (`# @param` lines); every human-readable string literal
(prints, f-strings, figure titles, axis labels, legends, widget labels); and the whole source of cells that are
not collapsed into form view. It also records whether the hidden code changed, and compares printed outputs line
by line. Her browsers print into Output widgets, which Colab keeps in the notebook's widget state; the script reads
them from there, so a browser cell is compared on what it really printed.

What was compared: her copy `her-copy.ipynb` and her Colab run (A100) against the **committed** Teaching notebook
at `a5db27a` and a fresh full-data run of exactly that file (`build/verify-teaching.ipynb`, 0 errors, 31.6 s of
cell time), then again at `2f87af6` (see below). Results: `build/verify-her-text.json` and
`build/verify-her-text-2f87af6.json`. Every difference the script found was read and given a verdict by hand; the verdicts are rules
scoped to her cell numbers inside the script, so any new change elsewhere comes out unclassified. As a check of
the checker, five planted changes (a word in a markdown cell, a word in a Sample Answer box, a word in a print,
an axis label, a printed number) were all reported as unclassified.

Not compared: figure pixels (only how many figures each cell draws), so a colour change inside a figure (T-28,
which is approved) is invisible to this check; and the Own-Data notebook beyond the short note at the end.

While this ran, another session committed `2f87af6` (Own-Data loader on real datasets). In the Teaching notebook
it changes only the hidden code of "🔎 Explore Pixel-Level Class Imbalance" (her cell 172: pixels are now counted per
class within an image). The check was repeated on `2f87af6` with its own full run (0 errors, 31.3 s): the result is
the same as above, and that cell's figure is byte-identical to the one before the commit, so her Sample Answer 173
("about 1% bird pixels against 25% of boxes") still describes it.

## Unsure differences (no unapproved ones were found)

| her cell | before | after | why it is not clearly covered |
|---|---|---|---|
| 14 · "Computer Vision Tasks" (📘 box) | `# @title Computer Vision Tasks` | `# @title 📘 Computer Vision Tasks { display-mode: "form" }` | The 📘 is T-18. The `display-mode: "form"` is not named by any item: T-17 adds it only to cells 26 and 135. C-12 (approved) cites N-034, which says a box lives in a form-view cell, so it is probably intended. Visible effect: Colab no longer shows the cell's HTML source above the box. |
| 68 · untitled 🔍 box (two Kaggle stories) | `# @title` | `# @title 🔍 Go Deeper: Two Kaggle Stories { display-mode: "form" }` | The title is T-18. Form mode: same as row 14. |
| 193 · "Checklist Dataset Mini-Project" (📘 box) | `# @title Checklist Dataset Mini-Project` | `# @title Checklist Dataset Mini-Project { display-mode: "form" }` | Form mode: same as row 14. visible-changes.md does not list it. |
| 133 · "✍️ YOUR CODE" (the `count_boxes_per_class` exercise; its code is visible) | `# ✍️ YOUR CODE` on the first line | a second line `# needs: boxes` | An Edition marker from group 7 step 5: the Own-Data loader reads it to skip box cells. In the Teaching notebook it does nothing, but it is the only marker a student sees, in the one cell they type into. No item names a visible change here. |
| — (new) · "Helpers: Images and Masks", current cell 10 | — | a new helper cell | C-14 approved one shared helpers cell named **"Helpers: Images and Display"**, holding the colour constants and `callout` too. It is named "Helpers: Images and Masks" and holds `load_thumb`, `frame`, `to_rle`, `decode`, `encode` (the constants and `callout` went to setup). The cell is approved; its title is not the one approved. |
| — (new) · "Helpers: Mask Quality Checks", current cell 186 | — | a new helper cell | Her "Simulate Mask Annotation Errors" was split in two so the Own-Data Edition can run the five checks (group 7 step 3). A new helper title appears just before her task; no item names the split. |

## Approved differences, by item

One line per item: her cells (0-based indices into `her-copy.ipynb`) and what changed. The full list, one row per
difference, is in `build/verify-her-text-tables.md`.

- **T-01** · 47 Sample Answer, Color Modes: "637" → "632".
- **T-02** · 57 Sample Answer, Class Distribution: "(130 each)" → "(130 and 129)".
- **T-03** · 60 Sample Answer, Animals per Image: "244 of 638" → "241 of 633".
- **T-04** · 65 Sample Answer, Class Imbalance: "130 of 638" → "130 of 633".
- **T-05 / R-05** · 63: slider default `DOG_SHARE` 0.1 → 0.02; output "(10% dog images …) Accuracy: 90.0%" → "(2% …) 98.0%".
- **T-06** · 73 Sample Answer, Duplicate Images: "among the 638 images" → "633".
- **T-08** · 147, 149, 150, 151, 186: every "IOU" → "IoU" (13 places, the LaTeX formula and the figure title included).
- **T-09** · 186: the table's garbled `<escape><</escape>` → `&lt;` (shows as "<").
- **T-10** · 36, 37, 52, 75, 112, 140, 141, 142, 161, 180, 184, 193: US spelling (colour, grey, neighbour, centre, metres, labelled), 19 places, in her markdown, box text, prints and figure labels.
- **T-11** · 112: "Stratified K-fold" → "Stratified k-fold".
- **T-12 (P17)** · 194 Resources box: "our play dataset" → "our Play Dataset".
- **T-13 (P16)** · 36, 37, 42, 43, 46, 56, 59, 75, 78, 107, 119, 120, 128, 134, 167, 172, 175, 177, 180, 183, 189: "photo(s)" → "image(s)" in printed and drawn text only (62 places); none in her markdown, boxes or titles, as P16 says.
- **T-14** · 130: "from Section 4." → "from Class Distribution: Are All Animal Classes Equally Represented?"; 193 checklist box: "(section 4)" → "(Class Imbalance in Meet the Park)".
- **T-15** · 185, 187: figure titles prefixed "Picture A ·" and "Picture B ·".
- **T-16 (P14)** · 111: "Magenta indicates validation images; gray-blue indicates training images" → "Yellow … gray …"; 112 figure title "yellow = test" → "yellow = validation".
- **T-17** · 26 and 135: form mode added; 26 summary "Sample answer" → "Sample Answer".
- **T-17 via C-12 (P1)** · 135, 139, 142, 146, 150, 153, 156, 160, 169, 173, 178, 181, 190: "💡" added to the Sample Answer summary. Every other word in the 59 box cells compared (her rendered output vs the current one) is unchanged.
- **T-18** · 8, 19, 36, 71, 118, 119, 128, 132, 148, 165, 167, 171, 183: cell titles ("Helpers: …" names, 📘 on four boxes, "Explore Example Images per Class", the reference-solution wording); 148 "IOU Helper Function" → "Helpers: IoU" (with T-08).
- **T-19 (P11)** · 69 `####` → `###`; 121 `###` → `####`; 126 heading → bold paragraph; 192 "## 10. Your project dataset" → "## Your Project Dataset"; plus the new MCQ heading (current cell 126).
- **T-20 (P12)** · 30, 62, 101, 147, 151, 179, 182 and the new MCQ heading: " (Optional)" added.
- **T-21 (P13)** · 101: "Which potential data quality issues should be addressed before training a model?" → "Data Quality: Which Issues Should Be Addressed Before Training? (Optional)"; the ⚠️ Pitfall stays a blockquote (T-23).
- **T-22 (P13)** · 51 "**Go Deeper:**" and 80 "**🔎 Go Deeper (optional):**" → "**Go Deeper (optional):**"; 68 box "Go deeper" → "Go Deeper".
- **T-23 (P13)** · 131: the yellow "✏️ Your turn" box became markdown; the words are hers, word for word (her `<code>`/`<strong>` became backticks and bold).
- **T-24 (P13)** · 117: "(a *burst*)" added to the first paragraph; the duplicate last paragraph removed.
- **T-25 (P13)** · 60: the main-list "Classification limitation" bullet removed; the Go Deeper one stays.
- **T-26 (P13)** · 191 "## Continue here (SUS)" removed.
- **N-18** · 49: form dropdown option "colourfulness" → "colorfulness".
- **N-21 (P8)** · 9 output column `n_animals` → `n_objects`; 15 markdown "(`n_animals`)" → "(`n_objects`)".
- **C-07 (P3)** · 72 output: "Found 1074 candidate pairs" → "1068".
- **C-08 (P4)** · the two MD5 lines ("633 files · 0 groups of byte-identical files") print in 72 instead of 75.
- **C-10** · 133 (visible code): the `boxes = annotations[…]` line removed.
- **C-02, C-22, R-08** · 93–97: new `COMPUTE_EMBEDDINGS` checkbox in 94; no HF Hub warning (no model download); 95 prints "Pretrained visual embeddings: loaded from coco_animals_dinov2.npz (633 images, 384 numbers each)…" instead of "Computed and cached global_dinov2."; 96/97 fall back to color histograms when the vectors are missing. The within-class medians in 98 are identical to hers.
- **Simone, later** · the `COMPUTE_EMBEDDINGS` note (current cell 95) and the closing cell (current 198, the seven guiding questions without answers and the Own-Data link); her chapter-10 cells 195–226 removed from the Teaching notebook.

Changes in string literals that no student reads (hidden code, output unchanged): C-03 (71, `flag_odd_files`
deleted), C-06/N-11 (82, 83, variant names moved), N-12 (72, `MAX_PAIR_BITS`; still prints "within 20 bits"),
C-17 (187, thresholds read from settings with the same values), C-22 (94, GPU check), N-21 and group 7 step 1
(8, 9: the download print moved into `load_play_dataset`; same output).

From the run environment, not the notebook: "Colab: True" → "False" (7), and "Computed and cached UMAP" →
"Loaded UMAP" (95, the UMAP came from the local cache).

Approved but not applied: **T-30** (trail-camera colours not the park's, P15). `CCT_COLORS` still reuses the first
five park colours (the group 5 log says it was left for a later pass).

### Her chapter 10 (cells 195–226) in the Own-Data notebook

Outside the question, a quick look at where her old chapter 10 went (best text match in the committed Own-Data
notebook): found nearly as written are 197, 199, 212–217, 220–226 (similarity 0.76–1.0, the T-14/T-27 edits
explain the rest). Dropped by approved items: 203–211 (C-23/P7), 219 (no Sample Answers in Own-Data, E-map).
Not found and replaced by the new loader text (group 7 step 2): 195 (🎯 Objective), 196, 198, 200, 201, 202
(🔍 "Detection or segmentation project, and big datasets"), 218 (the sample question, "Q25"). The E-map planned
to keep 195, 202 and 218 in some form; if their content matters, it is worth a look in the Own-Data notebook.

## visible-changes.md, tested

Everything it lists for the cells still in the Teaching notebook happened as written. The discrepancies:

**Listed, but not in the Teaching notebook.** It describes groups 1–6 only and was not updated when chapter 10
moved to the Own-Data notebook. Its entries for her cells 195, 197, 201, 202, 203, 205, 206, 207, 209, 210, 213,
218, 219, 220, 222, 223, 224, 225, 226 (group 1 fixes, T-07, T-10, T-14, T-27 and the chapter-10 box titles)
describe cells that are no longer there.

**Small attribution slips.** "trail-camera photos ready: n/400" is printed by "Helpers: Trail Cameras" (her 119),
not by the bursts browser (120). The "Colourfulness" figure heading (39) changes only inside the figure, which this
check cannot see; it follows from the renamed column, which did change.

**Happened, but not listed.**
- Form mode added to the box cells 14, 68 and 193 (they no longer show their HTML source in Colab).
- `# needs: boxes` added to the visible ✍️ YOUR CODE cell (133).
- The new helper cell "Helpers: Mask Quality Checks" before her mask-quality task.
- The `COMPUTE_EMBEDDINGS` markdown note above "Helpers: Pretrained Visual Embeddings".
- Chapter 10 removed from the Teaching notebook, and the new closing cell after "🔗 Resources".
