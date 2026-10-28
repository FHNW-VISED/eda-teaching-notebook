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
