# Visible changes to Susanne's copy (work-list groups 1–6)

Every change a student can see: markdown, box HTML, headings, cell titles shown in Colab's form view, printed
strings, figure titles, axis labels and legends. One line each: `cell title · before → after · item ID`.
Cell titles are the ones in `sections/` after the change (her copy's cell number in brackets where it helps).
Code that only a helper cell holds, and renamed variables a student never reads, are not listed.

## Group 1 · Breakage

- Read your folder (or build the sample), and show what was found [201] · the cell stopped with
  `NameError: name 'clean_images' is not defined`, and the five chapter-10 cells after it stopped too → the park
  sample is built from all Play Dataset images and the chapter runs to its end · C-24 / R-01
- Read your folder (or build the sample), and show what was found [201] · figure "A first look: are these your
  images and your classes?": every panel said "cannot be opened" → the panels show the images · R-03
- Check 3: exact, recompressed, flipped and rotated copies [210] · the cell would stop with
  `NameError: THUMB_FOR_HASH` and then `TypeError: 'int' object is not iterable` → it runs and draws the copy
  pairs · R-02, R-03
- Every browser (her cells 28, 72, 97, 120, 122, 129, 138, 145, 149, 152, 155, 159, 168, 176) · after a later
  browser had run, the buttons of an earlier one raised `AttributeError` or drew into another cell's output →
  each browser drives only itself (no text changes) · N-13
- 🔎 Explore Color Modes, Grayscale Images, and RGB Distributions [46] · no visible change with six classes (the
  grid now grows a row per three classes) · C-15

## Group 2 · Dead and duplicated code

- 🔎 Inspect Near-Duplicates Using pHash [72] · output now starts with the two MD5 lines
  ("633 files · 0 groups of byte-identical files", and up to five groups) that used to print in the 🔴 Bit Distance
  cell → printed here, in the 🟢 task that asks "How many exact duplicates are found?" · C-08
- 🔎 Inspect Near-Duplicates Using pHash [72] · "Found 1074 candidate pairs within 20 bits." → "Found 1068 candidate
  pairs within 20 bits." (full Play Dataset; the hashes are now taken once, on the 256 px thumbnails, and the pair
  browser shows those distances) · C-07
- 🔴 Bit Distance Distribution (Optional) [75] · the line "633 files · 0 groups of byte-identical files" → no longer
  printed here (it moved up, see above) · C-08
- ✍️ YOUR CODE, `count_boxes_per_class` [133] (code a student reads) · the line
  `boxes = annotations[annotations["image_id"].isin(images["id"])].reset_index(drop=True)` above the `use(...)` call
  → removed (`boxes` is made in the box helper cell) · C-10
- Check 3: exact, recompressed, flipped and rotated copies [210] · a copy found by plain pHash is listed with kind
  "original" → "recompressed" · C-06 / P10

## Group 3 · Names

- Load the manifest, download the images, build the tables [9] · the table printed under the cell has a column
  `n_animals` → `n_objects` · N-21 / P8
- The park does not have its own training data yet… [15] (markdown) · "number of annotated animals (`n_animals`)" →
  "number of annotated animals (`n_objects`)" · N-21 / P8
- Measure Image Dimensions and Other Properties [37] · summary table row `colourfulness` → `colorfulness` · N-18
- 🔎 Inspect Images with Minimum, Median, and Maximum Property Values [39] · figure heading "Colourfulness" →
  "Colorfulness" · N-18
- 🔎 Explore Extreme Image Properties [49] · Colab form dropdown option `colourfulness` → `colorfulness` (and the
  figure title "colourfulness: the 5 lowest, …" when it is picked) · N-18

## Group 4 · Boxes

Every box now comes from one helper, `callout(kind, title, body)`, with her four looks. **The text inside every box
is unchanged** (checked by comparing the rendered text of all boxes before and after); what changes is the frame,
the summary line and the cell titles.

- All 55 boxes without a figure (every 💡 Sample Answer except the mask-error one, every 📘, 🧾 and 🔍 box, and
  chapter 10's 📘 checklist and 🔗 resources) · showed nothing until the cell was run → the build saves each box as
  the cell's output, so it shows in Colab before anything runs · C-12 (N-034)
- 💡 Sample Answer, her cells 99, 102, 108, 113, 116, 123, 135, 139, 142, 146, 150, 153, 156, 160, 169, 173, 178,
  181, 190 · a grey box with a 1 px border all round, padding 12 × 16 px, the page's font → the grey box with the
  6 px left bar (`#9CA3AF`), 12 × 14 px padding and 14 px Google Sans that her other answers use · C-12
- 💡 Sample Answer, her cells 84, 87, 90 · a `<details>` element styled as the box (left bar, no font set) → the
  same `<details>` inside her standard answer box (font and inner margins as her other answers) · C-12
- 💡 Sample Answer, her cells 135, 139, 142, 146, 150, 153, 156, 160, 169, 173, 178, 181, 190 · summary
  "Sample Answer" → "💡 Sample Answer" · T-17
- 💡 Sample Answer, her cell 26 (Dataset Suitability) · summary "💡 Sample answer" → "💡 Sample Answer" · T-17
- 💡 Sample Answer, her cells 26 and 135 · the cell showed its HTML source in Colab → collapsed like the others
  (`{ display-mode: "form" }`) · T-17
- 📘 boxes, her cells 92, 105, 110, 118, 163, 165, 171 · 4 px left bar, padding 14 × 18 px, the page's font, title
  in `<strong>` → her standard theory box: 6 px bar, 12 × 14 px, 14 px Google Sans, title as a bold line · C-12
- 📘 boxes, her cells 61, 70, and 🧾 boxes 55, 67 · title line with 6 px space under it, link colour `#333333` in
  55/67 → title line without the extra space (link colour unchanged, it is in her body HTML) · C-12
- 💡 Sample Answer: Detect Suspicious Segmentation Masks [190] · figure saved and embedded by hand → embedded by
  `callout(..., fig=fig)` at the same place, same size (130 dpi, full width) · C-12
- Cell titles (Colab form view) · "Computer Vision Tasks" → "📘 Computer Vision Tasks"; "Group-Based Splitting and
  Data Leakage" → "📘 Group-Based Splitting and Data Leakage"; "Semantic Segmentation: Masks and Pixels" →
  "📘 Semantic Segmentation: Masks and Pixels"; "Pixel-Level Class Imbalance" (the 📘 box) → "📘 Pixel-Level Class
  Imbalance" · T-18
- Cell titles · "🔎 Sample example photos per class, framed in the class colour" → "🔎 Explore Example Images per
  Class"; "🔎 Helpers: Duplicate Detection and Image Quality" → "Helpers: Duplicate Detection and Image Quality" ·
  T-18
- Cell titles of helper cells · "Data helpers (download with cache, build the tables)" → "Helpers: Data Download and
  Tables"; "Measurement helpers" → "Helpers: Measurement"; "Prepare Trail-Camera Helpers" → "Helpers: Trail
  Cameras"; "Box helpers (draw boxes on thumbnails)" → "Helpers: Bounding Boxes"; "IOU Helper Function" → "Helpers:
  IOU" (→ "Helpers: IoU" in group 5); "Mask helpers and data preparation" → "Helpers: Masks and Data Preparation";
  "Simulate Mask Annotation Errors" → "Helpers: Simulate Mask Annotation Errors"; new cell "Helpers: Images and
  Masks" (group 2) · T-18
- Cell title [132] · "Reference solution — peek only if you are stuck" → "Reference solution — run this cell; open it
  only if you are stuck" (the wording of cell 11) · T-18
- Untitled box cells get a title · [68] → "🔍 Go Deeper: Two Kaggle Stories"; chapter 10: [195] → "🎯 Objective",
  [197] → "✏️ Your Turn: Point the Notebook at Your Dataset", [202] → "🔍 Go Deeper: Detection or Segmentation
  Project, and Big Datasets", [219] → "💡 Worked Answer", [223] → "🔍 Go Deeper: The Split Is Only a Suggestion",
  [226] → "✅ Takeaway" · T-18
- ✏️ Your turn · `count_boxes_per_class` [131] · a yellow box (`#FFF4D6`) → the same text as plain markdown: a bold
  line "**✏️ Your turn · `count_boxes_per_class`**" and the paragraph under it, word for word · T-23 (P13)
