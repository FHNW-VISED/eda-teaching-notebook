> **Superseded as the record (2026-10-29):** this list was written by the implementers and is incomplete for the
> chapter-10 move and the later additions. The independent check `verify-her-text.md` compares her copy with the
> built Teaching notebook cell by cell (0 unapproved changes) and is the record.

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

## Group 5 · Text

### Numbers (T-01 … T-07)

- 💡 Sample Answer, Color Modes [47] · "637 images are RGB (three channels)" → "632 images are RGB (three channels)" · T-01
- 💡 Sample Answer, Class Distribution [57] · "Cats and dogs have the most images (130 each)" → "Cats and dogs have the
  most images (130 and 129)" · T-02
- 💡 Sample Answer, Animals per Image [60] · "On the full Play Dataset, 244 of 638 images (38%)" → "On the full Play
  Dataset, 241 of 633 images (38%)" · T-03
- 💡 Sample Answer, Class Imbalance [65] · "the largest class contains 130 of 638 images" → "the largest class contains
  130 of 633 images" · T-04
- 🔎 Explore Accuracy and Dog Recall Under Class Imbalance [63] · Colab slider `DOG_SHARE` starts at 0.1 → 0.02; the
  printout "Rule B — Paddock camera (10% dog images, always 'no dog') · Accuracy: 90.0%" → "(2% dog images, …) ·
  Accuracy: 98.0%", which now matches her task text and her answer (the strip in 64 still draws 1 dog in 12, 92 %) ·
  T-05 / R-05
- 💡 Sample Answer, Duplicate Images [73] · "No byte-identical files were found among the 638 images." → "… among the
  633 images." · T-06
- Four checks run next and end in one report card… [203] (markdown) · "about 130 cats against 50 sheep" → "about 130
  cats against 59 sheep" · T-07
- 1. What is the task… (the guiding questions) [225] (markdown) · "sheep about 50" → "sheep about 59" · T-07

### IoU, spelling, wording (T-08 … T-13)

- 🔴 Bounding Box Overlap [147] (markdown) · heading "Understanding IOU" → "Understanding IoU"; "(IOU)" → "(IoU)";
  formula `\mathrm{IOU}` → `\mathrm{IoU}`; "IOU ranges from 0…" → "IoU ranges…"; "What happens to IOU…" → "IoU";
  "Observe how IOU changes…" → "IoU"; "How does IOU change…" → "IoU" · T-08
- Helpers cell title [148] · "Helpers: IOU" (group 4) → "Helpers: IoU" · T-08
- 🔎 Explore Bounding Box Overlap [149] · cell title "(IOU)" → "(IoU)"; figure title "IOU = … / … = … (…)" →
  "IoU = … / … = … (…)" · T-08
- 💡 Sample Answer, Bounding Box Overlap [150] · "IOU increases…", "IOU decreases…", "a high IOU" → "IoU" (three
  times) · T-08
- 🔴 Crowding: Bounding Box Overlap Across Classes [151] (markdown) · "if their IOU exceeds 0.1" → "if their IoU
  exceeds 0.1" · T-08
- Five automated mask quality checks [186] (markdown table) · "IOU between the annotated box and the tightest box" →
  "IoU between …"; the "Flagged when" cell `IoU <escape><</escape> 0.90` (shown garbled) → "IoU < 0.90" · T-08, T-09
- 🔎 Explore Brightness Distributions by Class [52] · y-axis "brightness (mean grey level)" → "brightness (mean gray
  level)" · T-10
- 🔎 Explore Bit Distance Distribution [75] · print "median nearest-neighbour distance" → "median nearest-neighbor
  distance"; figure title "How close is each photo's nearest neighbour?" → "How close is each image's nearest
  neighbor?" · T-10, T-13
- 🟢 Object Positions: Where Are the Animals? [140] (markdown) · "A bounding box's centre" → "center";
  "bounding-box centres" → "centers"; "box-centre heatmaps" → "box-center heatmaps" · T-10
- 🔎 Explore Object Positions Across Classes [141] · y-axis "box centre y (0 = top)" → "box center y (0 = top)";
  figure title "Where the box centres are · …" → "Where the box centers are · …" · T-10
- 💡 Sample Answer: Object Positions [142] · "Bounding-box centres are concentrated" → "Bounding-box centers are
  concentrated" · T-10
- Visualizing a Missing Annotation [161] (markdown) · "The solid grey boxes" → "The solid gray boxes" · T-10
- 🔎 Explore Spatial Distributions of Animal Pixels [180] · figure title "… full colour = always)" → "… full color =
  always)" · T-10
- Five simulated annotation errors [184] (markdown) · "Grey represents the original mask" → "Gray represents …" · T-10
- Checklist Dataset Mini-Project [193] (box) · "two metres away" → "two meters away"; "a zebra labelled "horse"" →
  "labeled" · T-10
- Check 2 opens every file… [206] (markdown) · "its colour mode", "three colour channels" → "color mode", "color
  channels" · T-10
- Check 2 [207] · cell title "readable? size and colour mode of every file" → "color mode"; print "colour modes:" →
  "color modes:" · T-10
- The guiding questions [225] (markdown) · "colour modes" → "color modes" · T-10
- 🔎 Visualize Cross-Validation Strategies [112] · row label "Stratified K-fold (each fold has one dog)" →
  "Stratified k-fold (each fold has one dog)" · T-11
- 🔗 Resources: Image Datasets and Annotation Tools [194] · "our play dataset is a subset of it" → "our Play Dataset
  is a subset of it" · T-12
- "photo(s)" → "image(s)" in what the cells print and draw (T-13, P16), chapters before Your Project Dataset:
  - 🔎 Inspect Images with Minimum, Median, and Maximum Property Values / 🔎 Explore Extreme Image Properties [39, 49]
    (helper text) · "no photos to show." → "no images to show."; figure title "…the typical (median) photo…" → "…image…"
  - Measure Image Dimensions and Other Properties [37] · "measured 633 photos · …" → "measured 633 images · …"
  - 🔎 Understanding Width–Height Scatter Plot Patterns [42] · panel title "all photos the same size" → "all images the
    same size"
  - 🔎 Explore Image Dimensions and Aspect Ratios by Class [43] · y-axis "photos" → "images"
  - 🔎 Explore Color Modes, Grayscale Images, and RGB Distributions [46] · "photos that look grayscale (mode L, …)" →
    "images that look grayscale (…)"; gallery title "photos that look grayscale" → "images …"; y-axis "photos" →
    "images"; panel titles "cat (130 photos)" → "cat (130 images)"
  - 🔎 Explore Class Distribution [56] · "633 photos · largest class …" → "633 images · …"; y-axis "photos" → "images";
    title "Images per class (one label per photo)" → "(one label per image)"
  - 🔎 Explore the Number of Animals per Image [59] · "241 of 633 photos (38 %) hold more than one animal; the most
    crowded photo has 26." → "images" / "image"; "mean animals per photo:" → "per image:"; x-axis "animals in the
    photo" → "in the image"; y-axis "photos" → "images"; title "Animals per photo" → "Animals per image"; gallery title
    "the most crowded photos" → "the most crowded images"
  - 🔎 Explore Bit Distance Distribution [75] · x-axis "Hamming distance to the closest other photo (bits)" → "image";
    y-axis "photos" → "images"
  - 🔎 Explore Perceptual Hashing (pHash) [78] · panel title "1. Original photo" → "1. Original image"; "Unrelated
    photo:" → "Unrelated image:"
  - 🔎 Compare Random and Stratified Splits [107] · panel titles "random · test (… photos)" → "(… images)"; y-axis
    "share of photos" → "share of images"
  - 🔎 Explore Trail-Camera Image Bursts [120] · progress "trail-camera photos ready: n/400" → "trail-camera images
    ready: n/400"; "400 photos · 142 bursts · cameras …" → "400 images · …"; "photos per class:" → "images per class:";
    "N photos could not be downloaded" → "N images could not be downloaded"
  - 🔎 Compare Class Distributions: Photos vs. Boxes [134] · "633 photos · 1540 boxes" → "633 images · …"; "Ranked by
    photos:" → "Ranked by images:"; "rank … by photos → rank … by boxes" → "by images"; left panel title "Photos per
    class (ranked)" and y-axis "Photos" → "Images per class (ranked)", "Images" (her cell title stays)
  - Box and mask galleries (helpers [128], [167]) · "no photos to show." → "no images to show."
  - 🔎 Explore Pixel-Level Class Imbalance [172] · title "Class distribution among animals: photos vs. boxes vs.
    pixels" → "images vs. boxes vs. pixels"; legend "photos" → "images"
  - 🔎 Explore Mask Sizes and Fill Ratios [175] · y-axis "mask area, share of the photo (log)" → "share of the image"
  - 🔎 Explore Thin Structures in Segmentation Masks [177] · gallery title "the photos with the most thin pixels …" →
    "the images with …"
  - 🔎 Explore Spatial Distributions of Animal Pixels [180] · panel titles "cat · …% of a photo" → "cat · …% of an image"
  - Helpers: Simulate Mask Annotation Errors [183] · "park_masks: N animal masks on M photos. Five of them are ugly.
    Find them." → "… on M images. …"
  - 🔎 Detect Suspicious Segmentation Masks [189] · suspect captions "photo <id> · …" → "image <id> · …"

### Headings, references, boxes (T-14 … T-28)

- 🟢 Boxes per Class vs. Photos per Class [130] (markdown) · "with the image-level distribution from Section 4." →
  "with the image-level distribution from Class Distribution: Are All Animal Classes Equally Represented?" · T-14
- Checklist Dataset Mini-Project [193] (box) · "the animal the alert exists for (section 4)." → "(Class Imbalance in
  Meet the Park)." · T-14
- 🎯 Objective [195] (box) · "Run the checklist of sections 3 to 6 on your own folder-per-class dataset" → "Run the
  checklist of Meet the Park, Common Dataset Errors & Cleaning and Dataset Splitting & Data Leakage on your own
  folder-per-class dataset" · T-14
- 🔍 Go Deeper: Detection or Segmentation Project [202] (box) · "(the box and mask checks are sections 7 and 8)" →
  "(the box and mask checks are Object Detection and Semantic Segmentation)" · T-14
- Four checks run next… [203] (markdown) · "as section 4 did for the park" → "as Class Distribution did for the park"
  · T-14
- 💡 Look at the ratio… [205] (markdown) · "(section 4)." → "(Class Imbalance)." · T-14
- Check 3 is … [209] (markdown) · "Check 3 is section 5's copy hunt:" → "Check 3 is the duplicate checks of Common
  Dataset Errors & Cleaning:"; "your `HASH_THRESHOLD` from section 5" → "from Bit Distance Distribution" · T-14
- Check 4 [213] · print "…add that to the groups (section 6)." → "(Dataset Splitting & Data Leakage)." · T-14
- 💡 Worked Answer [219] (box) · "the `HASH_THRESHOLD` you set in section 5." → "you set in Bit Distance
  Distribution." · T-14
- 💡 How to read the ratio [220] (markdown) · "(section 4's ⚠️ Pitfall)." → "(Class Imbalance: Can High Accuracy Be
  Misleading?)." · T-14
- 💡 Every copy pair is a decision for a human [222] (markdown) · "lower `HASH_THRESHOLD` in section 5 and rerun." →
  "in Bit Distance Distribution and rerun." · T-14
- 🔍 Go Deeper: The Split Is Only a Suggestion [223] (box) · "section 6 showed that" → "Dataset Splitting & Data
  Leakage showed that" · T-14
- The guiding questions [225] (markdown) · Q1 "(section 4)" → "(Meet the Park)"; Q2 "(section 2)" → "(Qualitative
  Inspection)"; Q3 "(section 4)" → "(Meet the Park)"; Q4 "(section 3)" → "(Quantitative Inspection)"; Q5 "removed
  before splitting (section 5's "removed N, kept M"); … ambiguous (section 4)." → "removed before splitting; …
  ambiguous (Meet the Park)."; Q6 "(section 6)" → "(Dataset Splitting & Data Leakage)"; Q7 "…aspect ratio (section
  3's 🔀 Callout)." → "…aspect ratio." · T-14
- 🔎 Explore Five Types of Mask Annotation Errors [185] · figure title "Five simulated annotation errors" → "Picture A
  · Five simulated annotation errors" · T-15
- 🔎 Explore Five Mask Quality Checks [187] · figure title "Five mask quality checks: original (top) vs. modified
  (bottom)" → "Picture B · Five mask quality checks: …" · T-15
- The diagram shows four strategies… [111] (markdown) · "Magenta indicates validation images; gray-blue indicates
  training images." → "Yellow indicates validation images; gray indicates training images." · T-16 (P14)
- 🔎 Visualize Cross-Validation Strategies [112] · figure title "Cross-validation: grey = training, yellow = test, D =
  dog" → "Cross-validation: gray = training, yellow = validation, D = dog" · T-16 (P14), T-10
- 🟢 Duplicate Images: What Can We Detect? [69] · heading level `####` → `###` (same level as the other tasks of the
  chapter) · T-19
- 🟢 Splitting Strategies: What Does Grouping Change? [121] · heading level `###` → `####` (a task under the part
  "Group-Based Splitting: The Trail-Camera Dataset") · T-19
- 🔴 Splitting for Object Detection and Segmentation [124] · the multiple-choice question had its level dot only in a
  code-cell title → a markdown heading "### 🔴 Splitting for Object Detection and Segmentation (Optional)" above it,
  so it shows in the table of contents · T-19, T-20
- From Labels to Bounding Boxes [126] · heading `### From Labels to Bounding Boxes` → bold paragraph "**From Labels to
  Bounding Boxes**" (so the tasks after it are not its siblings; it leaves the table of contents) · T-19
- Your Project Dataset [192] · heading "## 10. Your project dataset" → "## Your Project Dataset" · T-19
- 🔴 headings get "(Optional)" · [30] "🔴 Image Classification: One Label per Image" · [62] "🔴 Class Imbalance: Can
  High Accuracy Be Misleading?" · [147] "🔴 Bounding Box Overlap: Understanding IoU" · [151] "🔴 Crowding: Bounding Box
  Overlap Across Classes" · [179] "🔴 Where Are the Animal Pixels?" · [182] "🔴 Mask Quality: Finding Annotation
  Errors" → each + " (Optional)" · T-20 (P12)
- 🔴 Which potential data quality issues… [101] · heading "### 🔴 Which potential data quality issues should be
  addressed before training a model?" → "### 🔴 Data Quality: Which Issues Should Be Addressed Before Training?
  (Optional)" · T-21, T-20 (the ⚠️ Pitfall stays a blockquote, T-23)
- 🔴 Brightness [51] (markdown) · "**Go Deeper:**" → "**Go Deeper (optional):**" · T-22
- 🔴 Detecting Flipped and Rotated Duplicates [80] (markdown) · "**🔎 Go Deeper (optional):**" → "**Go Deeper
  (optional):**" · T-22
- 🔍 Go Deeper: Two Kaggle Stories [68] (box title) · "🔍 Go deeper · This is ordinary: two Kaggle stories" → "🔍 Go
  Deeper · This is ordinary: two Kaggle stories" · T-22
- Group-Based Splitting: The Trail-Camera Dataset [117] (markdown) · first paragraph "…producing bursts of closely
  related images." → "…producing bursts of closely related images (a *burst*)."; the last paragraph ("The farm park
  has installed motion-triggered cameras… Such a sequence of closely related images is called a burst.") → removed
  · T-24
- 💡 Sample Answer, Animals per Image [60] · the main-list bullet "Classification limitation: A single label cannot
  describe how many animals are present… An image with thirty sheep and one with a single lamb both receive the label
  "sheep". For counting or locating animals, object detection or segmentation would be more suitable." → removed (the
  Go Deeper bullet with the same point stays) · T-25
- Continue here (SUS) [191] · heading "## Continue here (SUS)" → removed · T-26
- On the sample the folders are… [218] (markdown) · "**Q25.** 🟢 On the sample…" → "🟢 On the sample…" · T-27
- 💡 Worked Answer [219] (box summary) · "💡 Worked answer (Q25, on the sample)" → "💡 Worked answer (on the sample)" ·
  T-27
- The guiding questions [224] (markdown) · "**Q26.** 🟢 Double-click the next cell…" → "🟢 Double-click the next
  cell…" · T-27
- 🔎 Explore Leakage Across Splitting Strategies [122] · frames of train / test images blue `#4C78A8` / orange
  `#E76F51` → light gray `#B0B0B0` / yellow `#FFDA00`, the pair of the cross-validation diagram · T-28 (P15)

## Group 6 · Embeddings gate

- Helpers: Pretrained Visual Embeddings [94] · ran `%pip -q install transformers umap-learn bokeh` (pip output, and
  possibly Colab's "restart session" prompt), imported torch and downloaded DINOv2 on every runtime → no install
  (all three are preinstalled in Colab), nothing is downloaded; the cell's form shows a new checkbox
  `COMPUTE_EMBEDDINGS` (unticked) · C-22 / R-08
- Precompute Image Representations and UMAP [95] · "Extracting visual features: 100% …" progress bar, then
  "Computed and cached global_dinov2." (or "Loaded global_dinov2 from cache.") → one line "Pretrained visual
  embeddings: loaded from coco_animals_dinov2.npz (633 images, 384 numbers each), computed beforehand with
  facebook/dinov2-small." (same vectors: the within-class medians in [98] are hers, cat 0.144 … zebra 0.549) · R-08
- Precompute Image Representations and UMAP [95], only when the file cannot be read or does not hold every image and
  there is no GPU · new lines "Pretrained visual embeddings: no precomputed file to read (…)." (or "…holds X of
  these N images, not all of them.") and "Pretrained visual embeddings skipped: on this CPU they would take roughly 2
  minutes. To compute them anyway, tick COMPUTE_EMBEDDINGS in the cell above and run both cells again; until then the
  next cells use the color histograms." · R-08 (SN-019)
- 🔎 Explore Image Representations with UMAP [96] and 🔎 Explore Similar and Unusual Images [97], same case · the
  cells stopped with `KeyError: 'Pretrained visual embeddings'` → they draw the color histograms (the UMAP title says
  "— Color histograms") · R-08
- With `COMPUTE_EMBEDDINGS` ticked, or on a GPU runtime without usable precomputed vectors · as before: progress bar
  and "Computed and cached global_dinov2." (on a GPU runtime it first prints "A GPU is available: computing the
  pretrained visual embeddings now.") · C-22
