# Clean-up, consistency and reuse audit — Susanne's copy (2026-10-28)

This is a static read of `reviews/2026-10-28-susanne-copy/her-copy.ipynb` (227 cells). Nothing was executed except
three small checks against the committed data (pixel shares, pHash distances), which are noted where they are used.
Every number quoted from "her output" comes from her original with outputs
(`raw/work/eda-notebook/2026-10-28-susanne-copy/image_dataset_inspection.ipynb`, run on an A100).

Cell numbers are 0-based indices into `her-copy.ipynb`. The `sections/` split may move them, so every item also
gives the cell's title. "Fix" means apply directly (the decision list allows it); "propose" means Simone decides.
Rules cited: base ledger `RULES.md` (N-xxx) and her overlay (SN-xxx); her copy beats both.

## Summary

| Category | Items | fix | propose | partly both |
|---|---|---|---|---|
| C — Code clean-up | 24 | 17 | 7 | 0 |
| N — Names | 23 | 20 | 3 | 0 |
| T — Text consistency | 30 | 14 | 14 | 2 |
| **Total** | **77** | **51** | **24** | **2** |

Edition map (E): 227 cells, of which 130 `both`, 63 `teaching` (37 of them Sample Answers), 33 `own-only-later`,
1 `drop`.

The five changes that matter most:

1. **Browsers break each other (N-13).** Twelve interactive cells keep their state in module-level globals with the
   same names (`navigate`, `show_page`, `page`, `counter`, `output`, `next_button`, `prev_btn`, …), and three helpers
   reuse a widget's name (`show_boxes`, `show_masks`, `threshold`, `n_pairs`, `a`). Once a later cell has run, the
   earlier browsers either raise `AttributeError` or draw into another cell's output. A student who scrolls back to
   the annotation browser (cell 28) after the segmentation chapter clicks *Next* and gets
   `'function' object has no attribute 'value'`. Rename now; the shared `browse()` helper (C-13) removes the cause.
2. **Run all stops at cell 201 (C-24).** Her own output shows `NameError: name 'clean_images' is not defined`, and
   cell 210 would fail next on `THUMB_FOR_HASH`. Both names lived in v05 cells she removed. These cells move to the
   Own-Data Edition anyway (PRD Q5), but until then the Teaching Edition does not finish a Run all.
3. **`results = {}` shadows the smoke-test hook (N-01).** Cell 134 creates a lowercase `results` dict, so
   `boxes_per_class`, `n_boxes`, `small_box_share` and `max_objects_per_image` never reach `RESULTS`, which the
   smoke run dumps and the content tests read.
4. **Stale numbers in Sample Answers (T-01…T-07).** The Play Dataset now has 633 images, not 638: four Sample Answers
   still say 638 (or 637 RGB, 244 crowded, "130 each"), and the Class Imbalance task's text and answer describe a 2 %
   dog share while the slider default is 10 % (so the output says 90 %, the answer 98 %).
5. **Dead and duplicated helpers, plus one hard park assumption (C-01…C-07, C-15).** `answer_box`, `load_embeddings`
   and `flag_odd_files` are never called; `decode_mask` duplicates `to_rle` + `decode`; `KIND_OF_VARIANT` is defined
   twice and the helper cell depends on its run cell; pHash is computed twice on different inputs. Cell 46 draws
   exactly six class panels, which crashes on an own dataset with seven or more classes.

---

## C — Code clean-up

Behaviour and visible output stay identical unless the item says otherwise (those are all `propose`).

### C-01 · `answer_box` is never called
- **Cell(s):** 7 (Setup (run me first))
- **Now:** `answer_box()` plus `ANSWER_BG, ANSWER_INK = "#F6EFE4", "#5C4027"` (our old brown answer colours) are
  defined but no cell calls them. Her Sample Answers are hand-written HTML in grey `#F5F5F5` (SN-015). Cell 190
  re-implements the same idea (lines + a base64 figure inside a collapsed box) by hand.
- **Instruction:** delete the block from the comment `# --- answer_box(): …` through the end of `answer_box`
  (including the `ANSWER_BG, ANSWER_INK` line). If C-12 is accepted, do not delete it: restyle it into the `callout()`
  helper's "answer" kind instead (grey `#F5F5F5`, border `#9CA3AF`, summary `💡 Sample Answer`).
- **Kind:** fix · **Confidence:** high

### C-02 · `load_embeddings` and `EMBEDDINGS_NAME` are dead
- **Cell(s):** 93 (Helpers: Image Embeddings)
- **Now:** `load_embeddings()` fetches `coco_animals_embeddings.npz` and silently falls back to colour histograms;
  nothing calls it. Her DINOv2 cells (94, 95) replaced it. `cosine_similarity_matrix` in the same cell is used (97).
- **Instruction:** delete `EMBEDDINGS_NAME` and `load_embeddings`; keep `cosine_similarity_matrix`. When the
  precomputed `data/coco_animals_dinov2.npz` is added (PRD Q8–9), reuse its by-id loading pattern
  (`dict(zip(z["ids"], z["embeddings"]))`, check every `images["id"]` is covered), but the fallback must print what
  happened and the minutes it would take (SN-019), not swap representations silently.
- **Kind:** fix · **Confidence:** high

### C-03 · `flag_odd_files` and `EXTREME_SIDE_PX` are dead
- **Cell(s):** 71 (🔎 Helpers: Duplicate Detection and Image Quality)
- **Now:** both served v05's cleaning cell, which she removed. `is_readable` in the same cell is still used (cell 207).
- **Instruction:** delete `EXTREME_SIDE_PX = (64, 6000)` and the whole `flag_odd_files` function.
- **Kind:** fix · **Confidence:** high

### C-04 · Other dead code and unused imports
- **Cell(s):** 36, 81, 107, 120, 155
- **Now:** cell 81 builds `hunt_images` (decodes 45 thumbnails) that nothing reads; cell 107 unpacks `_tr_rand`,
  `_tr_strat` that nothing reads and imports `GroupKFold`, `LeaveOneGroupOut` that nothing uses; cell 36 imports
  `ImageDraw` and `Rectangle` (unused there); cell 120 calls `select_cct_rows(cct_all)` three times; cell 155 assigns
  `quality_images = images[images["class"] == class_selector.value].copy()`, which `update_quality_images()`
  overwrites before anything is drawn.
- **Instruction:** delete the `hunt_images = …` line (81); write `_, _te_rand = …` and `_, _te_strat = …` and drop the
  two unused sklearn names (107); delete `from matplotlib.patches import Rectangle` and `from PIL import ImageDraw`
  (36); in 120 compute `cct_rows = select_cct_rows(cct_all)` once and use it in the three places; delete the initial
  `quality_images = …` line (155).
- **Kind:** fix · **Confidence:** high

### C-05 · `decode_mask` duplicates `to_rle` + `decode`
- **Cell(s):** 28 (🔎 Explore Images and Annotations), 167 (Mask helpers and data preparation)
- **Now:** cell 28 defines `decode_mask(segmentation, height, width)` with the same three-way polygon/RLE logic as
  `to_rle` in cell 167, followed by `mask_utils.decode`. The mask helpers live in the Segmentation chapter, so the
  earlier cell had to copy them. `from pycocotools import mask as mask_utils` is imported in both.
- **Instruction:** move `to_rle`, `decode`, `encode` and the `mask_utils` import from 167 into the shared helpers
  cell (C-14). In 28 delete `decode_mask` and call `decode(to_rle(ann["segmentation"], int(row["height"]),
  int(row["width"]))).astype(bool)`.
- **Kind:** fix · **Confidence:** high

### C-06 · `KIND_OF_VARIANT` defined twice; the helper depends on its run cell
- **Cell(s):** 82 (Helpers: Compare Duplicate Detection Methods), 83 (🔎 Compare Duplicate Detection Methods), 210
- **Now:** 82 defines `KIND_OF_VARIANT` with keys `flip_lr`, `flip_tb`, `rot90`… and `find_copies`, which calls
  `variants()` and `VARIANT_NAMES`. Those two are defined only in run cell 83, which also redefines
  `KIND_OF_VARIANT` with keys `flip`, `rotate90`… So 82's dict is dead, and 82 cannot run without 83.
  `PIL.ImageOps` is imported in both 81 and 83.
- **Instruction:** in 82 delete the first `KIND_OF_VARIANT`; move `variants()`, `VARIANT_NAMES` (as the literal
  tuple `("original", "flip", "rotate90", "rotate180", "rotate270")`, see N-11) and the second `KIND_OF_VARIANT` from
  83 into 82. Cell 83 keeps only the two `find_copies` calls and the three prints. Teaching output is unchanged.
  The 83 mapping labels a pHash-only match `"original"`; the Own copy check (210) prints those labels, where
  `"recompressed"` (82's old value) reads better — Simone decides that one word.
- **Kind:** fix (move) · **Confidence:** high

### C-07 · pHash is computed twice, on different inputs
- **Cell(s):** 72 (🔎 Inspect Near-Duplicates Using pHash), 75 (🔎 Explore Bit Distance Distribution)
- **Now:** 72 hashes the full-size photos into a list `hashes` and compares all 200,000 pairs in a Python loop
  (`hashes[i] - hashes[j]`); 75 hashes 256-px thumbnails again into `HASHES` and uses the vectorised
  `hash_distances`. Two decode passes over 633 JPEGs, and the two cells measure slightly different distances.
- **Instruction:** in 72 compute once
  `phash_bits = hash_bits([phash(load_thumb(p)) for p in images["path"]])`,
  `pair_dist = hash_distances(phash_bits, phash_bits)`, and build `candidate_pairs` from
  `np.triu(pair_dist <= MAX_PAIR_BITS, k=1)` sorted by distance; in 75 delete the `HASHES = …` block and use
  `phash_bits`. Checked locally on the committed data: the print in 72 changes from "1074 candidate pairs" to
  "1068"; pairs within 8 bits stay 1, within 5 bits stay 0, the median nearest-neighbor distance stays 20, so her
  Sample Answers 73 and 76 stay true.
- **Kind:** propose (one printed number changes) · **Confidence:** high

### C-08 · The exact-duplicate count runs in the optional cell, not in the task that asks for it
- **Cell(s):** 69 (🟢 Duplicate Images: What Can We Detect?), 72, 75
- **Now:** the 🟢 task asks "How many exact duplicates are found?", but its run cell 72 never runs MD5. The MD5 lines
  (`exact_groups = exact_duplicate_groups(…)` and two prints) sit in the 🔴 optional cell 75.
- **Instruction:** move those four lines from the top of 75 to the top of 72.
- **Kind:** propose (an output line moves) · **Confidence:** high

### C-09 · `images.set_index("id")` under seven names
- **Cell(s):** 107, 138, 141, 167, 175, 176, 177, 180, 189, 190
- **Now:** `photo_of` (138), `_size` (141, then read again in 175 in another chapter), `image_sizes` (167),
  `image_lookup` (176), `_class_of` (180), `_by_image` (190), plus inline `images.set_index("id")` in 107
  (`class_shares`), 177 and 189.
- **Instruction:** add `images_by_id = images.set_index("id")` to cell 9 right after `build_tables`; replace every
  one of the above with `images_by_id` (columns are selected at the point of use, e.g.
  `images_by_id[["width", "height"]]`). The Own loader sets the same name.
- **Kind:** fix · **Confidence:** high

### C-10 · `ann`, `imgs` and `boxes` re-filter tables that are already filtered
- **Cell(s):** 128 (Box helpers), 129 (🔎 Explore Bounding Boxes), 133 (✍️ YOUR CODE)
- **Now:** 129 builds `ann = annotations[annotations["image_id"].isin(images["id"])]` and `imgs = images[…]`; 133
  builds `boxes` with the same filter. `build_tables` already dropped annotations of missing photos, and every one
  of the 633 images has an annotation (its class comes from one), so both filters keep everything.
- **Instruction:** add `boxes = annotations.copy()` to the end of 128 (a copy, because 137 and 141 add columns); in
  129 use `boxes` and `images` and delete `ann`, `imgs`; in 133 delete the `boxes = …` line.
- **Kind:** fix · **Confidence:** medium (it changes one line of code a learner sees in the ✍️ cell)

### C-11 · `crowding` recomputes `n_animals`
- **Cell(s):** 144 (🔎 Explore Object Counts Across Classes)
- **Now:** `object_counts` groups `boxes` by image and merges back into `images`, which already has `n_animals`
  (the same count of the same annotation rows).
- **Instruction:** replace the `object_counts` block and the merge with
  `crowding = images[["id", "class", "n_animals"]].rename(columns={"n_animals": "n_objects"})`.
- **Kind:** fix · **Confidence:** high

### C-12 · One `callout()` helper for ~70 hand-copied HTML boxes
- **Cell(s):** every 💡 Sample Answer (24, 26, 29, 31, 40, 44, 47, 50, 53, 57, 60, 65, 73, 76, 79, 84, 87, 90, 99, 102,
  108, 113, 116, 123, 135, 139, 142, 146, 150, 153, 156, 160, 169, 173, 178, 181, 190), every 📘 box (14, 34, 61, 70,
  92, 103, 105, 110, 118, 163, 165, 171, 193), every 🧾 Recipe box (16, 33, 55, 67), go-deeper boxes (68, 194)
- **Now:** each cell repeats 8–12 lines of inline CSS. Sample Answers come in three different styles
  (border-left 6 px with the `callout` class; `<details>` with border-left; `<details>` with a 1 px border), theory
  boxes in two, Recipe boxes in two. This is the "lot of code for a small thing" she noticed.
- **Instruction:** add to setup a `callout(kind, title, body_html, fig=None)` helper with four kinds that reproduce
  her colours exactly: `answer` (collapsed `<details>`, `#F5F5F5`, border `#9CA3AF`, summary "💡 Sample Answer"),
  `theory` (`#E3F2FD` / `#0D47A1`), `recipe` (`#FFF8D6` / `#F5C400`), `deeper` (`#F3F4F6` / `#4B5563`). Each cell
  becomes `callout("answer", "Sample Answer", """<ul>…</ul>""")` with her inner HTML copied verbatim. `fig=` embeds
  a figure as PNG inside the box, which replaces the 25 hand-written lines in 190. Saved outputs still show the boxes
  before anything runs (N-034). Visible change: the three answer styles become one.
- **Kind:** propose · **Confidence:** high

### C-13 · One `browse()` helper for twelve hand-copied browsers
- **Cell(s):** 21, 28, 72, 89, 97, 120, 129, 138, 145, 155, 168, 176
- **Now:** each browser is 40–80 lines of buttons, counters, an `Output` and page arithmetic, all in module-level
  globals. This is the root cause of N-13. Only cell 89 keeps its state private (`_prev`, `_out`, `_class_index`).
- **Instruction:** add to setup `browse(n_pages, draw, controls=())`: Previous/Next buttons, an "i of n" label and an
  `Output`, with the page index held in a closure; each cell keeps its own selection logic and passes a
  `draw(page)` function. Visible change: buttons become uniform ("← Previous" / "Next →"); 120, 168, 176 currently
  use chevron icons and 21 a "Group:" slider.
- **Kind:** propose · **Confidence:** medium

### C-14 · Imports repeated in ~40 cells; helpers placed where they happen to be needed first
- **Cell(s):** imports repeated in 21, 28, 71, 72, 78, 81, 83, 86, 89, 94, 95, 96, 97, 107, 119, 120, 122, 124, 129,
  138, 145, 149, 152, 155, 158, 159, 167, 168, 176, 190, 199; `from IPython.display import HTML` in 65 cells
- **Now:** `load_thumb` (36) works only because cell 28 happened to import `Image`; `show_rows` (36) uses `display`
  imported by 21/28; cell 81 uses `BytesIO` imported by optional cell 78; cell 86 uses `ImageOps` from 81/83;
  `random` is re-imported in 158; PIL `Image` is imported as `_Image` in 7 and 21 and as `Image` in five other cells.
- **Instruction:** import once in setup (7): `from IPython.display import HTML, display, clear_output`,
  `import ipywidgets as widgets`, `from PIL import Image, ImageOps`, `from io import BytesIO`,
  `import base64, hashlib, re, shutil, warnings`, `from itertools import combinations`, `import imagehash`, `import cv2`,
  `from pycocotools import mask as mask_utils`, and the sklearn splitters used (`train_test_split`, `KFold`,
  `StratifiedKFold`, `LeaveOneOut`, `GroupShuffleSplit`). Delete the repeats. Keep heavy optional imports where
  they are (torch, transformers, umap, bokeh, `scipy.fft`), behind the C-22 flag. Then add one "Helpers: Images and
  Display" cell right after cell 9 holding `load_thumb`, `frame` (from 36), `to_rle`/`decode`/`encode` (C-05),
  the colour constants (C-18) and, if accepted, `callout` and `browse`. If C-12 is rejected, leave
  `from IPython.display import HTML` in the box cells so each stays self-contained. Run all is unaffected because
  setup always runs first.
- **Kind:** fix · **Confidence:** medium (decision for Simone only on the box cells)

### C-15 · Cell 46 draws exactly six class panels
- **Cell(s):** 46 (🔎 Explore Color Modes, Grayscale Images, and RGB Distributions)
- **Now:** `fig.add_gridspec(2, 4, …)` and `fig.add_subplot(_gs[k // 3, 1 + k % 3])` place the class histograms on a
  2 × 3 block. With seven or more classes `k // 3` reaches row 2 and the cell raises `IndexError`.
- **Instruction:** `n_rows = math.ceil(len(CLASSES) / 3)`; `fig = plt.figure(figsize=(13, 2.3 * n_rows))`;
  `fig.add_gridspec(n_rows, 4, …)`; `ax_modes = fig.add_subplot(_gs[:, 0])` unchanged. With six classes the
  figure is identical.
- **Kind:** fix · **Confidence:** high

### C-16 · Helpers read global tables instead of taking them
- **Cell(s):** 36 (`_closest_to_median`, `show_extremes`, `show_min_median_max` read `image_stats`), 107
  (`class_shares` reads `images`)
- **Now:** they work in both Editions only because the Own loader will reuse the same global names.
- **Instruction:** add a `stats_df` parameter to the three helpers in 36 and a `table` parameter to `class_shares`;
  the calls in 39, 49 and 107 pass `image_stats` / `images` explicitly.
- **Kind:** propose · **Confidence:** medium

### C-17 · Mask-check thresholds written three times
- **Cell(s):** 186 (her table), 187 (🔎 Explore Five Mask Quality Checks), 189 (🔎 Detect Suspicious Segmentation Masks)
- **Now:** 189 defines `BOX_IOU_MIN`, `FILL_RANGE`, `MAX_PIECES`, `MAX_HOLES`, `SOLIDITY_MAX`; 187 hard-codes the
  same values inside `CHECK_RULES` (`v < 0.9`, `0.15 <= v <= 0.95`, `v > 3`, `v > 0`, `v > 0.97`).
- **Instruction:** move the five settings from 189 into the helper cell 183 (next to `SHIFT_PX`); build
  `CHECK_RULES` in 187 from them (`lambda v: v < BOX_IOU_MIN`, caption `f"suspect below {BOX_IOU_MIN}"`, …).
  Her markdown table in 186 stays as it is.
- **Kind:** fix · **Confidence:** high

### C-18 · Colour constants redefined in eight cells
- **Cell(s):** 78, 86, 112, 122, 144, 149, 162, 185, 187, 190
- **Now:** `GREY = "#9CA3AF"` and `YELLOW = "#FFDA00"` are set in 185, 187 and 190; `DARK_GREY` in 187;
  `box_grey = "#B0B0B0"` and `fhnw_yellow = "#FFDA00"` in 149 and 162; `box_edge` in 149; `highlight` and
  `bit_colors` in 78; `split_colors` in 112; `"#B0B0B0"` inline in 144; `FOUND_COLOR`, `MISSED_COLOR` in 86.
- **Instruction:** define once in setup next to `INK`: `GRAY = "#9CA3AF"`, `LIGHT_GRAY = "#B0B0B0"`,
  `DARK_GRAY = "#4B5563"`, `EDGE_GRAY = "#666666"`, `HIGHLIGHT = "#FFDA00"` (keep `FOUND_COLOR`/`MISSED_COLOR` in 86,
  used once). Replace the local names and delete the redefinitions. Values unchanged, so figures are identical.
- **Kind:** fix · **Confidence:** high

### C-19 · JPEG re-encoding written three times
- **Cell(s):** 78 (🔎 Explore Perceptual Hashing), 81 (`plant_hunt5`, "recompressed" branch), 86 (`_reencode`)
- **Instruction:** add `jpeg_copy(im, scale=1.0, quality=55)` to helper cell 71 (resize with the `max(8, …)` guard
  from 81, save to `BytesIO`, reopen as RGB); call it in 78 (`scale=0.7, quality=55`), 81 (`0.7, 55`) and 86
  (`1.0, 25`). Same parameters, same images.
- **Kind:** fix · **Confidence:** medium

### C-20 · `find_copies` loops in Python; the Own copy check re-implements it vectorised
- **Cell(s):** 82 (`find_copies`), 210 (Check 3)
- **Now:** `find_copies` compares pairs in a Python double loop (fine for 45 files, slow for 2,000); cell 210 does the
  same job blockwise with numpy, in 35 lines of its own.
- **Instruction:** rewrite `find_copies` with 210's blockwise minimum over variants (same signature, same returned
  columns `a, b, kind, distance`), and make the Own copy check call it.
- **Kind:** propose · **Confidence:** medium

### C-21 · `show_bursts` re-selects bursts the caller already chose
- **Cell(s):** 119 (Prepare Trail-Camera Helpers), 120 (🔎 Explore Trail-Camera Image Bursts)
- **Now:** 120 sorts all bursts (fully boxed first) and pages through them two at a time, then `show_bursts` re-runs
  its own "different animal, different camera" selection on those two. Because the list is already sorted the same
  way, the drawn order never changes.
- **Instruction:** change `show_bursts(table, n_bursts, …)` to `show_bursts(bursts, size=256, show_boxes=True)`
  drawing the list of burst DataFrames it receives; in 120 pass `selected` directly (drop the `pd.concat`).
- **Kind:** fix · **Confidence:** medium

### C-22 · DINOv2 loads unconditionally; `%pip` installs what Colab has
- **Cell(s):** 94 (Helpers: Pretrained Visual Embeddings), 95 (Precompute Image Representations and UMAP)
- **Now:** 94 runs `%pip -q install transformers umap-learn bokeh` and loads `facebook/dinov2-small` on any device;
  95 computes embeddings for every photo. `pyproject.toml` lists none of these packages, so the smoke environment
  depends on that `%pip`. This is the decided SN-019 / PRD Q8–9 change, recorded here so it is not forgotten.
- **Instruction:** add `COMPUTE_EMBEDDINGS = False` (a setting); load the model and compute only when
  `COMPUTE_EMBEDDINGS or DEVICE == "cuda"`; otherwise read the precomputed park vectors (C-02's pattern), and on
  own data on CPU skip with the estimated minutes and fall back to colour histograms with a printed note. Limit the
  `%pip` line to what Colab lacks: transformers, torch and bokeh are preinstalled in Colab; check `umap-learn`
  against `scripts/colab_pins.py` before dropping it, and add the packages the smoke run needs to `pyproject.toml`.
- **Kind:** fix (decided in the PRD) · **Confidence:** medium (on the exact package list)

### C-23 · The old own-dataset checks re-implement her analysis
- **Cell(s):** 203–212 (Checks 1–3), 199 (`_drive_public_download` repeats `download_file`'s `.part` logic)
- **Now:** Check 1 (204) redoes cell 56, Check 2 (207) redoes 37/46 through `measure_image`, Check 3 (210) redoes
  72/75/82. PRD Q4 says the Own-Data Edition reuses her cells on the student's tables, with no new analysis code.
- **Instruction:** once her chapters run on own tables, drop 203–211 from the Own-Data Edition; keep the split
  suggestion (212–214) and the report card (215–216), rewired to read her variables (`images_per_class`,
  `image_stats`, the duplicate pairs from 72). Reuse `download_file` for the Drive download in 199.
- **Kind:** propose · **Confidence:** medium

### C-24 · Two names her removals broke: Run all stops at cell 201
- **Cell(s):** 201 (Read your folder (or build the sample)…), 210 (Check 3)
- **Now:** `build_my_sample` reads `clean_images[clean_images["flags"] == ""]`; v05's cleaning cell that built
  `clean_images` is gone, and her output shows the `NameError`. Cell 210 uses `THUMB_FOR_HASH`, defined in v05 as
  `256` in a cell she also removed. SN-010: fix only what the removal breaks.
- **Instruction:** in `build_my_sample` use `pool = images` (every downloaded photo is readable; the removed filter
  only dropped unreadable or odd files); add `THUMB_FOR_HASH = 256` to helper cell 71 (or drop it if C-07's
  `load_thumb` default of 256 is used everywhere). Both cells move to the Own-Data Edition (E).
- **Kind:** fix · **Confidence:** high

---

## N — Names

N-058 / SN-008: capitals only for settings decided once; anything computed is lowercase; a setting is never
reassigned or changed in place. `RESULTS` is the one exception. Cells listed are every cell that uses the name.

### N-01 · `results` (lowercase) shadows the `RESULTS` hook
- **Cell(s):** 134 (🔎 Compare Class Distributions: Photos vs. Boxes), 137, 144
- **Now:** 134 runs `results = {}` and writes `results["boxes_per_class"]`, `results["n_boxes"]`; 137 writes
  `results["small_box_share"]`; 144 writes `results["max_objects_per_image"]`. None of them reach `RESULTS`, which the
  smoke test dumps (`tests/test_content.py` needs `boxes_per_class` and `n_boxes`). Every other cell writes `RESULTS`.
- **Instruction:** delete `results = {}` in 134; replace `results[` with `RESULTS[` in 134, 137, 144.
- **Kind:** fix · **Confidence:** high

### N-02 · `COCO_RAW` → `coco_raw`
- **Cell(s):** 9 · **Now:** loaded data in capitals. **Instruction:** rename in 9 (only use).
- **Kind:** fix · **Confidence:** high

### N-03 · `CHANNEL_HISTS` → `channel_hists`
- **Cell(s):** 37, 46, 89, 93 (gone with C-02), 95 · **Now:** computed in 37. **Instruction:** rename in all listed cells.
- **Kind:** fix · **Confidence:** high

### N-04 · `PAIRS` → `candidate_pairs`
- **Cell(s):** 72 · **Now:** computed list in capitals. **Instruction:** rename (or replaced by C-07).
- **Kind:** fix · **Confidence:** high

### N-05 · `HASHES` and `hashes`: two names, two computations
- **Cell(s):** 72 (`hashes`, list of `ImageHash`), 75 (`HASHES`, bit array)
- **Instruction:** one lowercase `phash_bits` computed in 72 and reused in 75 (C-07). If C-07 is rejected, still
  rename `HASHES` → `phash_bits` in 75 and `hashes` → `phash_list` in 72.
- **Kind:** fix · **Confidence:** high

### N-06 · `HUNT5_KEY` and the "hunt5" names
- **Cell(s):** 81, 83
- **Now:** `HUNT5_KEY` is computed (a planted answer key); "hunt5" is v05's section number, which no longer exists
  (`plant_hunt5`, folder `data/cache/hunt5`).
- **Instruction:** `HUNT5_KEY` → `duplicate_key` (81, 83); `plant_hunt5` → `plant_duplicates` (81);
  `HUNT_DIR` → `DUPLICATE_CARD_DIR` (81; the folder name can stay or become `duplicate_card`, the cell rebuilds it).
- **Kind:** fix · **Confidence:** high

### N-07 · `HUNT9_KEY` is computed and then changed in place
- **Cell(s):** 183, 190
- **Now:** `HUNT9_KEY` is a computed DataFrame, and 190 adds columns to it (`HUNT9_KEY["checks"] = …`), which N-058
  forbids for a capital name. Also `HUNT9_KINDS`, `plant_hunt9`, `hunt9_found` carry v05's section number.
- **Instruction:** `HUNT9_KEY` → `mask_error_key` (183, 190); `HUNT9_KINDS` → `MASK_ERROR_KINDS` (183);
  `plant_hunt9` → `plant_mask_errors` (183); `hunt9_found` → `n_errors_found` (190). Keep the `RESULTS["hunt9"]` key
  until the content tests are rewritten (PRD Q10), then rename it with them.
- **Kind:** fix · **Confidence:** high

### N-08 · `CHECKS` → `check_hits`
- **Cell(s):** 189 · **Now:** computed by `run_checks`. **Instruction:** rename in 189 (three uses).
- **Kind:** fix · **Confidence:** high

### N-09 · `CCT_LOCATIONS` → `cct_locations`
- **Cell(s):** 120 · **Instruction:** rename (two uses). **Kind:** fix · **Confidence:** high

### N-10 · `MY_CLASSES`, `MY_COLORS` → `my_classes`, `my_colors`
- **Cell(s):** 201, 204, 207, 210, 213, 216 · **Now:** computed from the student's folders.
- **Instruction:** rename in all six cells (these move to the Own-Data Edition; the loader may replace them, see N-22).
- **Kind:** fix · **Confidence:** high

### N-11 · `VARIANT_NAMES` is computed from a function
- **Cell(s):** 82, 83, 210
- **Now:** `VARIANT_NAMES = list(variants(Image.new("RGB", (4, 4))))` (computed, N-058).
- **Instruction:** write it as the literal setting `VARIANT_NAMES = ("original", "flip", "rotate90", "rotate180",
  "rotate270")` in 82 (C-06).
- **Kind:** fix · **Confidence:** high

### N-12 · Settings written in lowercase or as bare literals
- **Cell(s):** 72, 78, 83, 95, 112, 120, 122, 129, 145, 149, 152, 155, 162, 168, 176
- **Now and instruction:**
  `overlap_iou = 0.1` set in both 149 and 152 → `OVERLAP_IOU = 0.1` once in 148 (Helpers: IoU), used in 149, 152;
  `page_size = 4` in 129, 145, 155, 168, 176 → `PAGE_SIZE = 4` in setup (or inside `browse`, C-13);
  `bursts_per_page = 2` (120) → `BURSTS_PER_PAGE`;
  `settings = dict(n_neighbors=15, …)` inside the loop in 95 → `UMAP_SETTINGS` above the loop;
  `test_size=0.2` literal in 122 → `TEST_SIZE` (defined in 107, same chapter);
  the literals `20` and `8` in 72 (candidate limit, slider default) and `threshold=8` in 83 →
  `MAX_PAIR_BITS = 20` and `PAIR_THRESHOLD = 8` in 71;
  colour names in 78, 112, 149, 162 → C-18's constants.
  `split_colors`, `dogs`, `cv_folds` in 112 are toy data for one figure and may stay lowercase.
- **Kind:** fix · **Confidence:** high

### N-13 · Widget and callback names collide across cells (live bug)
- **Cell(s):** 28, 72, 97, 120, 122, 128, 129, 137, 138, 145, 149, 152, 155, 159, 167, 168, 176
- **Now:** callbacks look names up at click time, so the latest cell to define a name wins:

  | name | defined in | what breaks |
  |---|---|---|
  | `show_boxes` | 28 (checkbox), 122 (checkbox), 128 (function) | after 128, toggling or paging in 28 and 122 raises `AttributeError: 'function' object has no attribute 'value'` |
  | `show_masks` | 28 (checkbox), 167 (function) | after 167, the annotation browser in 28 fails the same way |
  | `navigate` | 28, 72, 97, 168, 176 | 28's and 72's buttons drive 176's browser; 168's "previous" → `page + "previous"` TypeError |
  | `threshold` | 72 (slider), 137 (loop variable, ends as `96`) | after 137, the pair browser in 72 fails on `threshold.value` |
  | `n_pairs` | 97 (input box), 152 (int) | after 152, the similarity browser in 97 fails on `n_pairs.value` |
  | `a` | 149 (reference box), 152 (loop variable) | after 152, moving 149's sliders draws a COCO box instead of the toy box |
  | `show_page`, `change_page`, `update_browser` | 129/138/168/176, 129/138, 72/120 | earlier buttons run the later cell's code |
  | `page`, `page_size`, `counter`, `output`, `previous`, `next_button`, `prev_btn`, `next_btn`, `state`, `page_label`, `mask_toggle`, `box_toggle` | 28, 72, 97, 120, 129, 138, 145, 155, 159, 168, 176 | earlier browsers update or draw into a later cell's widgets |

- **Instruction:** short term, give each browser cell its own prefix on every widget, state and callback name it
  defines: 28 `ann_` (and `show_boxes`/`show_masks` → `ann_box_toggle`/`ann_mask_toggle`), 72 `dup_`, 97 `sim_`,
  120 `burst_`, 122 `split_` (`seed` → `split_seed`, `show_boxes` → `split_box_toggle`), 129 `boxview_`,
  138 `size_`, 145 `crowd_`, 155 `quality_`, 159 `missing_`, 168 `maskview_`, 176 `fill_`; in 137 rename the loop
  variable `threshold` → `side_px`; in 152 rename `a, b` → `box_a, box_b`, `n_pairs` → `n_box_pairs`; in 149 rename
  `a` → `REFERENCE_BOX`. Long term C-13 removes the globals. Helper names `show_boxes` (128) and `show_masks` (167)
  stay.
- **Kind:** fix · **Confidence:** high

### N-14 · `images_per_class` and `photos_per_class` are the same dict
- **Cell(s):** 56, 64, 134, 172
- **Instruction:** keep `images_per_class` (56); delete the recomputation `photos_per_class = …` in 134 and use
  `images_per_class` in 134 and 172.
- **Kind:** fix · **Confidence:** high

### N-15 · `classes` (per-image array) next to `CLASSES` (class list)
- **Cell(s):** 97, 98 · **Instruction:** `classes` → `image_classes` in both cells.
- **Kind:** fix · **Confidence:** high

### N-16 · PIL imported as `_Image` and as `Image`
- **Cell(s):** 7, 19, 21 (`_Image`); 28, 36, 71, 72, 78, 81, 83, 86, 94, 119, 138, 201 (`Image`)
- **Instruction:** import `Image` once in setup (C-14); replace `_Image` with `Image` in 19 and 21.
- **Kind:** fix · **Confidence:** high

### N-17 · Three spellings of matplotlib's patches
- **Cell(s):** 21, 28 (`patches.Rectangle`); 36, 78 (`Rectangle`); 112, 119, 128, 138, 149, 162, 172, 185, 187, 190
  (`mpl.patches.…`)
- **Instruction:** use `mpl.patches.Rectangle` everywhere; delete the `import matplotlib.patches as patches` (21, 28)
  and `from matplotlib.patches import Rectangle` (36, 78) lines.
- **Kind:** fix · **Confidence:** high

### N-18 · British spelling in identifiers and in a column she sees
- **Cell(s):** 36, 37, 46, 49, 115, 185, 187, 190 (and C-18's colour names)
- **Now:** the `image_stats` column `colourfulness` shows up in her output: the summary table row (37), the heading
  "Colourfulness" in the min/median/max figure (39), the dropdown option (49), while her text says "Colorfulness"
  (34). Also `COLOUR_MODES` (36, 46), `_centres` (46), `summarise_folds` (115), `GREY`/`DARK_GREY` (185, 187, 190).
- **Instruction:** `colourfulness` → `colorfulness` in 36 (the stats dict key and two docstrings), 37
  (`STAT_COLUMNS`), 46 (`looks_gray`), 49 (`@param` list); `COLOUR_MODES` → `COLOR_MODES` (36, 46); `_centres` →
  `_centers` (46); `summarise_folds` → `summarize_folds` (115, four uses); grey names via C-18.
- **Kind:** fix · **Confidence:** high

### N-19 · Generic globals that outlive their cell
- **Cell(s):** 94, 95, 96, 124, 190
- **Now:** `model`, `processor` (94, read inside `dinov2_embeddings`), `paths` (95, closed over by the cache key and
  the lambda), `p` and `source` (96), `question`, `options`, `answers`, `check`, `solution`, `feedback` (124, read
  by the button callbacks), `lines`, `img`, `buf` (190).
- **Instruction:** `model` → `dino_model`, `processor` → `dino_processor`, `paths` → `embedding_paths`,
  `p` → `umap_fig`, `source` → `umap_source`, the 124 names → `mcq_question`, `mcq_options`, `mcq_answers`,
  `mcq_check`, `mcq_solution`, `mcq_feedback`; in 190 prefix the scratch names with `_`.
- **Kind:** fix · **Confidence:** medium

### N-20 · `hunt_answer_key` is never read
- **Cell(s):** 158 (🔎 Simulate and Detect a Missing Annotation)
- **Now:** the Sample Answer (160) is static text, so the key is unused; it is still the only record of which box was
  removed.
- **Instruction:** rename to `missing_box_key` and keep it (a content test can use it to prove the hunt is solvable).
- **Kind:** fix · **Confidence:** medium

### N-21 · `n_animals` and `n_objects` count the same thing
- **Cell(s):** 8, 9, 15, 59, 144, 145, 157, 158, 159
- **Now:** `images["n_animals"]` (built in 8) and `crowding["n_objects"]` (144) are the same count; for an own
  dataset "animals" is park wording.
- **Instruction:** pick one column name for both Editions. Recommended: `n_objects` in `build_tables` and every use
  (59, 144, 145, 158, 159), with her text in 15 and 157 mentioning `n_animals` updated to match.
- **Kind:** propose · **Confidence:** medium

### N-22 · `CLASSES` and `CLASS_COLORS` become computed in the Own-Data Edition
- **Cell(s):** 7 and ~60 cells that read them
- **Now:** in the Teaching Edition they are settings; in the Own-Data Edition they come from the student's data, which
  N-058 says must be lowercase.
- **Instruction:** either keep the capitals and document them as "decided once, at load" (both Editions set them in
  one place, never later), or rename to `class_names` / `class_colors` everywhere. Recommended: keep the capitals;
  the rename touches ~60 of her cells for no visible gain.
- **Kind:** propose · **Confidence:** medium

### N-23 · Single-letter and vague constants in the toy-horse cells
- **Cell(s):** 187 · **Now:** `H, W = 60, 84` and `UNDER = {…}`.
- **Instruction:** `TOY_H`, `TOY_W`, `CHECK_CAPTIONS`.
- **Kind:** propose · **Confidence:** low

---

## T — Text consistency

### T-01 · RGB count in the Color Modes answer
- **Cell(s):** 47 (💡 Sample Answer, Color Modes)
- **Now:** "637 images are RGB (three channels) and one is grayscale". Her output: `RGB 632 · L 1`.
- **Instruction:** "637" → "632".
- **Kind:** fix · **Confidence:** high

### T-02 · Cat and dog counts
- **Cell(s):** 57 (💡 Sample Answer, Class Distribution)
- **Now:** "Cats and dogs have the most images (130 each)". Output: cat 130, dog 129.
- **Instruction:** "(130 each)" → "(130 and 129)".
- **Kind:** fix · **Confidence:** high

### T-03 · Crowded-image count
- **Cell(s):** 60 (💡 Sample Answer, Animals per Image)
- **Now:** "244 of 638 images (38%)". Output: "241 of 633 photos (38 %)".
- **Instruction:** "244 of 638" → "241 of 633".
- **Kind:** fix · **Confidence:** high

### T-04 · Rule A count
- **Cell(s):** 65 (💡 Sample Answer, Class Imbalance)
- **Now:** "the largest class contains 130 of 638 images, giving approximately 20% accuracy". Output: 20.5 %.
- **Instruction:** "638" → "633" (20.5 % still reads as "approximately 20%").
- **Kind:** fix · **Confidence:** high

### T-05 · Dog share: text says 2 %, slider default is 10 %
- **Cell(s):** 62 (🔴 Class Imbalance task), 63 (🔎 Explore Accuracy and Dog Recall…), 65
- **Now:** the task says "Of 1,000 images, 20 contain a dog" and the answer "98% accuracy … when 2% of images contain
  dogs", but `DOG_SHARE = 0.1`, so the output prints "10% dog images … Accuracy: 90.0%".
- **Instruction:** set the default to `DOG_SHARE = 0.02` in 63. The strip in 64 is unchanged (it still draws 1 dog in
  12 frames, 92 %).
- **Kind:** fix · **Confidence:** high

### T-06 · Image count in the duplicates answer
- **Cell(s):** 73 (💡 Sample Answer, Duplicate Images)
- **Now:** "No byte-identical files were found among the 638 images". Output: "633 files · 0 groups".
- **Instruction:** "638" → "633".
- **Kind:** fix · **Confidence:** high

### T-07 · Sheep count in the own-dataset text
- **Cell(s):** 203, 225
- **Now:** "about 130 cats against 50 sheep" (203), "sheep about 50" (225). Output: sheep 59, ratio 2.2.
- **Instruction:** "50" → "59" in both.
- **Kind:** fix · **Confidence:** high

Numbers checked and correct: 6, 17 grayscale-looking (47), 2.2× and 59 sheep (57), 26 and the per-class means (60),
3 of 5 / 5 of 5 (84), 0 and 38 differing bits (79, recomputed locally), one pair at 8 bits (73, recomputed locally),
40 LOO runs (116), eight cameras (117), ~85 % background, ~1 % bird pixels, 25 % of boxes (173, recomputed
locally), 44 %–57 % fill and 67 % → 14 % thin (178).

### T-08 · IOU → IoU
- **Cell(s):** 147 (heading "🔴 Bounding Box Overlap: Understanding IOU" and three times in the text and formula), 148
  (title "IOU Helper Function"), 149 (title "…Overlap (IOU)" and the figure title `f"IOU = …"`), 150, 151, 186
  ("IOU between the annotated box…")
- **Instruction:** replace every "IOU" with "IoU" (also `\mathrm{IOU}` → `\mathrm{IoU}`). Code already says "box IoU".
- **Kind:** fix · **Confidence:** high

### T-09 · An escape artefact in the checks table
- **Cell(s):** 186
- **Now:** the cell's source contains `IoU <escape><</escape> 0.90` (a Colab export artefact), in both copies.
- **Instruction:** replace with `IoU &lt; 0.90`.
- **Kind:** fix · **Confidence:** medium

### T-10 · US spelling, one per word
- **Cell(s), visible:** 19 title "framed in the class colour"; 52 y-label "mean grey level"; 75 print
  "nearest-neighbour" and title "nearest neighbour"; 112 title "grey = training"; 140 "A bounding box's centre",
  "bounding-box centres", "box-centre heatmaps"; 141 y-label "box centre y", title "Where the box centres are";
  142 "Bounding-box centres"; 161 "The solid grey boxes"; 180 title "full colour"; 184 "Grey represents";
  193 "labelled", "two metres"; 206 "colour mode", "colour channels"; 207 title and print "colour mode(s)";
  225 "colour modes".
- **Cell(s), comments and docstrings:** 7, 9 ("licence"), 36, 43, 52, 93, 119, 128, 141, 167 ("rasterisation"),
  175 ("relabelled"), 183 ("neighbour"), 185, 187, 190, 201.
- **Instruction:** color, gray, neighbor, center(s), labeled, meters, license, rasterization, relabeled. Identifiers
  are N-18.
- **Kind:** fix · **Confidence:** high

### T-11 · "k-fold" capitalisation
- **Cell(s):** 112 ("K-fold (a fold may miss the dog)", "Stratified K-fold (each fold has one dog)"), 115
  ("Stratified k-fold")
- **Instruction:** "K-fold" only at the start of a label, "k-fold" elsewhere: 112 → "Stratified k-fold (each fold
  has one dog)".
- **Kind:** fix · **Confidence:** medium

### T-12 · Three names for the Play Dataset
- **Cell(s):** "Wildwood animal dataset" (15, 18, 27); "Play Dataset" (6, 57, 58, 60); "COCO subset" (61, 62, 63,
  65); "play dataset" (194)
- **Instruction:** fix the case in 194 ("Play Dataset"). Proposal: one name for the dataset, e.g. "Play Dataset"
  (introduced once in 15 as "the Wildwood animal dataset (our Play Dataset), a subset of COCO"), "COCO subset" only
  where the point is its origin.
- **Kind:** fix (194) / propose (the rest) · **Confidence:** medium

### T-13 · "images" in her prose, "photos" in the outputs
- **Cell(s):** prints and figure titles in 9, 37, 43, 46, 56, 59, 64, 75, 107, 115, 120, 134, 158, 177, 183, 189
- **Instruction:** change "photo(s)" → "image(s)" in output strings so the outputs use her word.
- **Kind:** propose · **Confidence:** low

### T-14 · References to section numbers that no longer exist
- **Cell(s):** 130 (hers): "with the image-level distribution from Section 4" → "from Class Distribution: Are All
  Animal Classes Equally Represented?". Own part: 193 "(section 4)" → "(Class Imbalance in Meet the Park)";
  195 "sections 3 to 6" → "Meet the Park, Common Dataset Errors & Cleaning and Dataset Splitting & Data Leakage";
  202 "sections 7 and 8" → "Object Detection and Semantic Segmentation"; 203 "as section 4 did" → "as Class
  Distribution did"; 205 "(section 4)" → "(Class Imbalance)"; 207 comment "section 3's measure_image" →
  "Quantitative Inspection's measure_image"; 209 "section 5's copy hunt" → "the duplicate checks of Common Dataset
  Errors & Cleaning" and "from section 5" → "from Bit Distance Distribution"; 213 print "(section 6)" → "(Dataset
  Splitting & Data Leakage)"; 219 and 222 "section 5" → "Bit Distance Distribution"; 220 "section 4's ⚠️ Pitfall" →
  "Class Imbalance: Can High Accuracy Be Misleading?"; 223 "section 6 showed" → "Dataset Splitting & Data Leakage
  showed"; 225 "(section 2)" → "(Qualitative Inspection)", "(section 3)" → "(Quantitative Inspection)",
  "(section 4)" → "(Meet the Park)", "(section 6)" → "(Dataset Splitting & Data Leakage)", and delete
  "(section 5's "removed N, kept M")" and "(section 3's 🔀 Callout)", which point at things she removed.
- **Kind:** fix · **Confidence:** high

### T-15 · "Picture A" and "Picture B" are not labelled
- **Cell(s):** 182, 188 (her text refers to Picture A and B); 185, 187 (the figures)
- **Instruction:** prefix the two figure titles: `"Picture A · Five simulated annotation errors"` (185) and
  `"Picture B · Five mask quality checks: original (top) vs. modified (bottom)"` (187). Her text stays.
- **Kind:** fix · **Confidence:** medium

### T-16 · The cross-validation diagram's text describes other colours
- **Cell(s):** 111 (text), 112 (figure)
- **Now:** the text says "Magenta indicates validation images; gray-blue indicates training images"; the figure draws
  training gray `#B0B0B0`, held-out yellow `#FFDA00` and titles itself "grey = training, yellow = test" (her theory
  box 110 says "validation").
- **Instruction:** change the figure title in 112 to "Cross-validation: gray = training, yellow = validation, D = dog"
  and the sentence in 111 to "Yellow indicates validation images; gray indicates training images."
- **Kind:** propose · **Confidence:** high

### T-17 · Sample Answer cells that show their code, and their label
- **Cell(s):** 26 and 135 (no `{ display-mode: "form" }`, so Colab shows the HTML source); 26 summary "💡 Sample
  answer"; 135, 139, 142, 146, 150, 153, 156, 160, 169, 173, 178, 181, 190 summaries without 💡
- **Instruction:** add `{ display-mode: "form" }` to the titles of 26 and 135, and "Sample answer" → "Sample Answer" in
  26 (fix). Proposal: "💡 Sample Answer" on every summary (SN-014); C-12 does it automatically.
- **Kind:** fix (form mode, case) / propose (💡) · **Confidence:** high

### T-18 · Cell titles that break her patterns (SN-014)
- **Cell(s):** 19 "🔎 Sample example photos per class, framed in the class colour" → "🔎 Explore Example Images per
  Class"; 71 "🔎 Helpers: …" → "Helpers: …" (helpers carry no icon); helpers not titled "Helpers: …": 8, 36, 119,
  128, 148, 167, 183 (e.g. "Helpers: Measurement", "Helpers: IoU"); 📘 boxes whose title lacks 📘: 14, 118, 165, 171;
  reference-solution titles differ: 11 "…run this cell; open it only if you are stuck" vs 132 "…peek only if you are
  stuck"; untitled `# @title` box cells: 68, 131, 195, 197, 202, 219, 223, 226.
- **Kind:** propose · **Confidence:** medium

### T-19 · Heading levels (SN-011: a Task is one level below its parent)
- **Cell(s):** 69 "#### 🟢 Duplicate Images…" sits directly under the `##` chapter while its siblings 74–91 are `###`
  → `###`. 121 "### 🟢 Splitting Strategies…" follows the part "### Group-Based Splitting: The Trail-Camera
  Dataset" (117) at the same level → `####`. Tasks 127–157 are siblings of the part "### From Labels to Bounding
  Boxes" (126) → either 126 becomes a bold paragraph or the tasks become `####`. 124's multiple-choice question has
  its level dot only in a code title, so it is missing from the table of contents (N-055) → add a markdown heading
  "### 🔴 Splitting for Object Detection and Segmentation" above it. 192 "## 10. Your project dataset" → "## Your
  Project Dataset" (no numbers, N-056).
- **Kind:** propose · **Confidence:** medium

### T-20 · "(Optional)" on some 🔴 headings only
- **Cell(s):** with it: 51, 74, 77, 80, 88, 91, 114; without it: 30, 62, 101, 147, 151, 179, 182
- **Now:** her legend (cell 2) already says "🔴 optional".
- **Instruction:** drop "(Optional)" everywhere (the dot says it), or add it to all fourteen.
- **Kind:** propose · **Confidence:** medium

### T-21 · One heading not in Title Case
- **Cell(s):** 101 "### 🔴 Which potential data quality issues should be addressed before training a model?"
- **Instruction:** "### 🔴 Data Quality: Which Issues Should Be Addressed Before Training?" (her "Topic: Question?"
  form, SN-012).
- **Kind:** propose · **Confidence:** medium

### T-22 · "Go Deeper" label variants
- **Cell(s):** 51 "**Go Deeper:**", 80 "**🔎 Go Deeper (optional):**", all others "**Go Deeper (optional):**";
  68 box "🔍 Go deeper · …"
- **Instruction:** "**Go Deeper (optional):**" in 51 and 80; "Go Deeper" in 68.
- **Kind:** propose · **Confidence:** high

### T-23 · Box kinds outside her set (SN-001, SN-015)
- **Cell(s):** 101 "⚠️ Pitfall" blockquote; 131 "✏️ Your turn" box in `#FFF4D6`
- **Instruction:** 131 → plain markdown under the task (the task text in 130 already says what to write); 101 keep
  as a blockquote or move into the 📘 checklist (103).
- **Kind:** propose · **Confidence:** low

### T-24 · A paragraph appears twice
- **Cell(s):** 117 (### Group-Based Splitting: The Trail-Camera Dataset)
- **Now:** the first and the last paragraph both open with "The farm park has installed motion-triggered cameras in
  its woodland reserve. Each camera can capture several images of the same animal…".
- **Instruction:** delete the last paragraph and add its definition to the first: "…producing bursts of closely
  related images (a *burst*)."
- **Kind:** propose · **Confidence:** high

### T-25 · A bullet appears twice
- **Cell(s):** 60 (💡 Sample Answer, Animals per Image)
- **Now:** "Classification limitation" is in the main list and again, almost word for word, under Go Deeper.
- **Instruction:** keep the Go Deeper one (it answers the Go Deeper question) and drop the main-list one.
- **Kind:** propose · **Confidence:** medium

### T-26 · Her working marker
- **Cell(s):** 191 "## Continue here (SUS)"
- **Instruction:** remove from both Editions (it marks where she stopped, not content).
- **Kind:** propose · **Confidence:** high

### T-27 · Q-numbers survive in the own-dataset part
- **Cell(s):** 218 "**Q25.**", 219 "Worked answer (Q25…)", 224 "**Q26.**"
- **Instruction:** drop the Q-numbers when these cells move to the Own-Data Edition (SN-012).
- **Kind:** fix · **Confidence:** high

### T-28 · Train/test colours differ between the two split figures
- **Cell(s):** 112 (gray / yellow), 122 (blue `#4C78A8` / orange `#E76F51`)
- **Instruction:** use one pair in both (N-027).
- **Kind:** propose · **Confidence:** medium

### T-29 · Two default hash thresholds in one chapter
- **Cell(s):** 72 (slider default 8), 83 (`threshold=8`), 73 ("At a threshold of 8 bits"), 75
  (`HASH_THRESHOLD = 5`), 86 (reads `HASH_THRESHOLD`)
- **Instruction:** one default, 8, as `PAIR_THRESHOLD` (N-12), and start the slider in 75 at 8 too.
- **Kind:** propose · **Confidence:** medium

### T-30 · Trail-camera animals borrow park class colours
- **Cell(s):** 119 (`CCT_COLORS` = the first five park colours: opossum is drawn in the cat's colour)
- **Instruction:** five colours not used by the park classes (N-027: a colour means one thing).
- **Kind:** propose · **Confidence:** low

---

## E — Edition map

Edition values: `both`, `teaching` (park-only), `own-only-later` (the old own-dataset part, cells 192–226, which PRD
Q5 moves to the Own-Data Edition), `drop`. Needs: what the cell needs from the data (`images`, `boxes`, `masks`,
`cct20`, `park-story`). Every 💡 Sample Answer is `teaching` (the Own-Data Edition has none).

| idx | title / first line | edition | needs | park assumption that blocks reuse | change to make it generic |
|---|---|---|---|---|---|
| 0 | Know Your Images (md) | teaching | park-story | subtitle and goal name the Wildwood Farm Park | own title cell |
| 1 | A note about this notebook (md) | both | — | — | — |
| 2 | Learning Objectives (code) | both | — | — | — |
| 3 | What Do You Need to Know for the Exam and Your Project? (md) | both | — | — | — |
| 4 | How to Use This Notebook (code) | teaching | — | "A separate notebook guides you…" | own variant: "This is the Own-Data Edition" |
| 5 | Setup (md) | both | — | — | — |
| 6 | > ▶️ Run the next three cells, in order. (md) | teaching | park-story | Play Dataset, ~115 MB download | own variant: "run the loader cells" |
| 7 | Setup (run me first) (code) | both | images | `CLASSES`, `CLASS_COLORS` hard-coded to the six park animals; `RESULTS["class_colors"]` | keep palette + `INK` here; set classes and colors after loading (see paragraph) |
| 8 | Data helpers (download with cache, build the tables) (code) | both | images | `download_images` expects COCO rows (`coco_url`); `build_tables` needs annotations to set `class` | loader converts every format to a COCO-style dict; let `build_tables` take image-level labels |
| 9 | Load the manifest, download the images, build the tables (code) | teaching | images, park-story | COCO manifest; smoke sample per park `CLASSES` | replaced by the Own-Data loader cell(s) |
| 10 | Optional Coding Exercises (md) | both | — | — | — |
| 11 | Reference solution — run this cell; open it only if you a… (code) | both | — | — | — |
| 12 | # ✍️ YOUR CODE (code) | both | — | — | — |
| 13 | Meet the Park: Explore Your Dataset (md) | teaching | park-story | enclosures, park ranger | own variant: "Meet Your Data" + the student's task |
| 14 | Computer Vision Tasks (code) | teaching | park-story | dogs in the sheep paddock | own: guiding question 1 (task and expensive mistake) |
| 15 | The park does not have its own training data yet. Before … (md) | teaching | park-story | COCO subset story | own variant keeps the two table bullets only |
| 16 | 🧾 Karpathy's Recipe · Step 1 (code) | both | — | — | — |
| 17 | We won't spend hours exploring the data, but let's spend … (md) | both | images | — | — |
| 18 | 🟢 Images: What Do Images Look Like Across Classes? (md) | both | images | "Wildwood animal dataset" | own text variant (guiding question + blank answer) |
| 19 | 🔎 Sample example photos per class, framed in the class co… (code) | both | images | — | cap columns for many classes |
| 20 | Select an animal class and browse at least twelve images … (md) | both | images | "park's fixed outdoor cameras" | own text variant (guiding question + blank answer) |
| 21 | 🔎 Browse Images Within a Class (code) | both | images (+boxes) | — | hide the box checkbox when there are no boxes |
| 22 | 🟢 Document Your Observations (md) | both | images | dog/park example | own text variant (guiding question + blank answer) |
| 23 | My three observations (double-click to edit): (md) | both | — | — | — |
| 24 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 25 | 🟢 Dataset Suitability: How Suitable Are These Images for … (md) | both | images | "for the Park" | own text variant (guiding question + blank answer) |
| 26 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 27 | 🟢 Annotations: What Does an Image Annotation Look Like? (md) | both | images, boxes, masks | "Wildwood animal dataset" | own text variant (guiding question + blank answer) |
| 28 | 🔎 Explore Images and Annotations (code) | both | images, boxes, masks | `decode_mask` assumes every row has a segmentation | hide box/mask toggles when the columns are empty |
| 29 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 30 | 🔴 Image Classification: One Label per Image (md) | both | images | — | own text variant (guiding question + blank answer) |
| 31 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 32 | Quantitative Inspection (md) | both | — | — | — |
| 33 | 🧾 Recipe · Step 1, continued (code) | both | — | — | — |
| 34 | 📘 Image Properties: What Can We Measure? (code) | both | — | — | — |
| 35 | The next code cells calculate these properties for each i… (md) | both | — | — | — |
| 36 | Measurement helpers (code) | both | images | helpers read global `image_stats` | pass the table in (C-16) |
| 37 | Measure Image Dimensions and Other Properties (code) | both | images | — | — |
| 38 | 🟢 Image Properties: What Do the Statistics Reveal? (md) | both | images | — | own text variant (guiding question + blank answer) |
| 39 | 🔎 Inspect Images with Minimum, Median, and Maximum Proper… (code) | both | images | — | — |
| 40 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 41 | 🟢 Image Dimensions: What Do Their Distributions Reveal? (md) | both | images | "COCO … photographers" vs park cameras | own text variant (guiding question + blank answer) |
| 42 | 🔎 Understanding Width–Height Scatter Plot Patterns (code) | both | — | — | — |
| 43 | 🔎 Explore Image Dimensions and Aspect Ratios by Class (code) | both | images | — | — |
| 44 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 45 | 🟢 Color Modes and Colors: What Do They Reveal? (md) | both | images | — | own text variant (guiding question + blank answer) |
| 46 | 🔎 Explore Color Modes, Grayscale Images, and RGB Distribu… (code) | both | images | grid of exactly six class panels (`k // 3` on 2×4) | C-15 |
| 47 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 48 | 🟢 Image Properties: Are Extreme Values Problematic? (md) | both | images | — | own text variant (guiding question + blank answer) |
| 49 | 🔎 Explore Extreme Image Properties (code) | both | images | — | — |
| 50 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 51 | 🔴 Brightness: How Do Distributions Differ Between Classes… (md) | both | images | — | own text variant (guiding question + blank answer) |
| 52 | 🔎 Explore Brightness Distributions by Class (code) | both | images | — | — |
| 53 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 54 | 🟢 Class Distribution: Are All Animal Classes Equally Repr… (md) | both | images | — | own text variant (guiding question + blank answer) |
| 55 | 🧾 Recipe · Step 1, continued: Imbalances and Biases (code) | both | — | — | — |
| 56 | 🔎 Explore Class Distribution (code) | both | images | — | — |
| 57 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 58 | 🟢 Animals per Image: How Common Are Multiple Animals? (md) | both | boxes | "one of our six animal classes" | own text variant (guiding question + blank answer); skip without boxes |
| 59 | 🔎 Explore the Number of Animals per Image (code) | both | boxes | `n_animals` | skip without boxes |
| 60 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 61 | 📘 Labeling Strategies: One Label or Multiple Animals? (code) | teaching | park-story | dogs among sheep, dog alert | — |
| 62 | 🔴 Class Imbalance: Can High Accuracy Be Misleading? (md) | teaching | park-story | dog recall, sheep paddock | own: guiding question on the expensive class |
| 63 | 🔎 Explore Accuracy and Dog Recall Under Class Imbalance (code) | teaching | park-story | `"dog"` literal, `DOG_SHARE` | — |
| 64 | 🔎 See How the Rule Classifies Real Images (code) | teaching | park-story | `"dog"`, `"sheep"` literals | — |
| 65 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 66 | Common Dataset Errors & Cleaning (md) | both | — | — | — |
| 67 | 🧾 Recipe · Step 1, continued: Dataset Errors (code) | both | — | — | — |
| 68 | (box) 🔍 Go deeper · This is ordinary: two Kaggle stories (code) | both | — | — | — |
| 69 | 🟢 Duplicate Images: What Can We Detect? (md) | both | images | — | own text variant (guiding question + blank answer) |
| 70 | 📘 Detecting Duplicates: MD5 vs. Perceptual Hashing (code) | both | — | — | — |
| 71 | 🔎 Helpers: Duplicate Detection and Image Quality (code) | both | images | — | — |
| 72 | 🔎 Inspect Near-Duplicates Using pHash (code) | both | images | — | — |
| 73 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 74 | 🔴 Bit Distance Distribution (Optional) (md) | both | images | — | own text variant (guiding question + blank answer) |
| 75 | 🔎 Explore Bit Distance Distribution (code) | both | images | — | — |
| 76 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 77 | 🔴 How Does Perceptual Hashing Work? (Optional) (md) | both | — | — | — |
| 78 | 🔎 Explore Perceptual Hashing (pHash) (code) | both | images | — | — |
| 79 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 80 | 🔴 Detecting Flipped and Rotated Duplicates (Optional) (md) | teaching | park-story | planted hunt | — |
| 81 | 🔎 Generate a Dataset with Known Duplicates (code) | teaching | images, park-story | planted copies, `HUNT5_KEY` | — |
| 82 | Helpers: Compare Duplicate Detection Methods (code) | both | images | — | `find_copies` reused by the own copy check (C-20) |
| 83 | 🔎 Compare Duplicate Detection Methods (code) | teaching | park-story | scores against the planted key | — |
| 84 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 85 | Why, on one photo and five kinds of copy: the pHash dista… (md) | both | — | — | — |
| 86 | 🔎 How Do Image Transformations Affect pHash? (code) | both | images | — | — |
| 87 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 88 | 🔴 Identifying Unusual Images (Optional) (md) | both | images | — | own text variant (guiding question + blank answer) |
| 89 | 🔎 Find Unusual Images Using Color Histograms (code) | both | images | — | — |
| 90 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 91 | 🔴 Exploring Image Embeddings (Optional) (md) | both | images | — | own text variant (guiding question + blank answer) |
| 92 | 📘 Image Embeddings, Similarity, and UMAP (code) | both | — | — | — |
| 93 | Helpers: Image Embeddings (code) | both | images | — | — |
| 94 | Helpers: Pretrained Visual Embeddings (code) | both | images | — | `COMPUTE_EMBEDDINGS` gate (C-23) |
| 95 | Precompute Image Representations and UMAP (code) | both | images | — | cache must not reuse park vectors for own data (key already includes paths) |
| 96 | 🔎 Explore Image Representations with UMAP (code) | both | images | — | — |
| 97 | 🔎 Explore Similar and Unusual Images (code) | both | images | — | — |
| 98 | 🔎 Compare Within-Class Image Similarity (code) | both | images | — | — |
| 99 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 100 | From Data Exploration to Cleaning (md) | both | — | — | — |
| 101 | 🔴 Which potential data quality issues should be addressed… (md) | both | — | — | own text variant (guiding question + blank answer) |
| 102 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 103 | 📘 Image Dataset Quality Checklist (code) | both | — | — | — |
| 104 | Dataset Splitting & Data Leakage (md) | both | — | — | — |
| 105 | 📘 Data Leakage and Generalization (code) | both | — | — | — |
| 106 | 🟢 Random vs. Stratified Splitting: How Do Class Distribut… (md) | both | images | — | own text variant (guiding question + blank answer) |
| 107 | 🔎 Compare Random and Stratified Splits (code) | both | images | `class_shares` reads global `images` | — |
| 108 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 109 | 🟢 Cross-Validation: How Do We Choose Validation Sets? (md) | both | — | toy: three cameras | own text variant (guiding question + blank answer) |
| 110 | 📘 Cross-Validation Strategies (code) | both | — | — | — |
| 111 | The diagram shows four strategies using 12 example images… (md) | both | — | "D = dog, the rare class" | keep toy (propose "R = rare class") |
| 112 | 🔎 Visualize Cross-Validation Strategies (code) | both | — | toy labels "D = dog" | same as 111 |
| 113 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 114 | 🔴 Cross-Validation in Practice (Optional) (md) | both | images | — | own text variant (guiding question + blank answer) |
| 115 | 🔎 Compare Cross-Validation Fold Counts (code) | both | images | — | — |
| 116 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 117 | Group-Based Splitting: The Trail-Camera Dataset (md) | teaching | cct20, park-story | trail cameras in the woodland reserve | own: guiding question "what ties your images together?" |
| 118 | Group-Based Splitting and Data Leakage (code) | both | — | bursts as the example | — |
| 119 | Prepare Trail-Camera Helpers (code) | teaching | cct20 | CCT20 manifest | — |
| 120 | 🔎 Explore Trail-Camera Image Bursts (code) | teaching | cct20 | — | — |
| 121 | 🟢 Splitting Strategies: What Does Grouping Change? (md) | teaching | cct20 | — | — |
| 122 | 🔎 Explore Leakage Across Splitting Strategies (code) | teaching | cct20 | — | — |
| 123 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 124 | 🔴 Splitting for Object Detection and Segmentation (code) | teaching | — | quiz with its own solution button | Simone: both? |
| 125 | Object Detection (md) | both | boxes | "the park wants to count" | own text variant (guiding question + blank answer) |
| 126 | From Labels to Bounding Boxes (md) | both | boxes | "COCO format" | own variant names the converted format |
| 127 | 🟢 Exploring Bounding Boxes: What Do You Notice? (md) | both | boxes | — | own text variant (guiding question + blank answer) |
| 128 | Box helpers (draw boxes on thumbnails) (code) | both | boxes | — | — |
| 129 | 🔎 Explore Bounding Boxes (code) | both | boxes | — | — |
| 130 | 🟢 Boxes per Class vs. Photos per Class (md) | both | boxes | "12 sheep", "Section 4" | own text variant (guiding question + blank answer) |
| 131 | (box) ✏️ Your turn · (code) | both | boxes | — | — |
| 132 | Reference solution — peek only if you are stuck (code) | both | boxes | — | — |
| 133 | # ✍️ YOUR CODE (code) | both | boxes | docstring example uses park classes | generic example |
| 134 | 🔎 Compare Class Distributions: Photos vs. Boxes (code) | both | boxes | — | — |
| 135 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 136 | 🟢 Object Sizes: How Small Are the Animals? (md) | both | boxes | — | own text variant (guiding question + blank answer) |
| 137 | 🔎 Explore Object Size Distributions (code) | both | boxes | — | — |
| 138 | 🔎 Explore Bounding Box Sizes in Images (code) | both | boxes | — | — |
| 139 | 💡 Sample Answer: Object Sizes (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 140 | 🟢 Object Positions: Where Are the Animals? (md) | both | boxes | — | own text variant (guiding question + blank answer) |
| 141 | 🔎 Explore Object Positions Across Classes (code) | both | boxes | one row of `len(CLASSES)` panels | wrap rows for many classes |
| 142 | 💡 Sample Answer: Object Positions (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 143 | 🟢 Crowding: How Many Animals Appear Together? (md) | both | boxes | — | own text variant (guiding question + blank answer) |
| 144 | 🔎 Explore Object Counts Across Classes (code) | both | boxes | — | — |
| 145 | 🔎 Explore Crowded Images (code) | both | boxes | — | — |
| 146 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 147 | 🔴 Bounding Box Overlap: Understanding IOU (md) | both | — | — | own text variant (guiding question + blank answer) |
| 148 | IOU Helper Function (code) | both | — | — | — |
| 149 | 🔎 Explore Bounding Box Overlap (IOU) (code) | both | — | — | — |
| 150 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 151 | 🔴 Crowding: Bounding Box Overlap Across Classes (md) | both | boxes | — | own text variant (guiding question + blank answer) |
| 152 | 🔎 Explore Bounding Box Overlap Across Classes (code) | both | boxes | — | — |
| 153 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 154 | 🟢 Bounding Box Quality: How Well Do the Boxes Fit? (md) | both | boxes | — | own text variant (guiding question + blank answer) |
| 155 | 🔎 Inspect Bounding Box Quality (code) | both | boxes | — | — |
| 156 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 157 | 🟢 Missing Annotations: Find the Missing Box (md) | teaching | boxes, park-story | planted missing box (`park_boxes`) | — |
| 158 | 🔎 Simulate and Detect a Missing Annotation (code) | teaching | boxes, park-story | prefers a `"sheep"` image | — |
| 159 | 🔎 Inspect Missing Annotations (code) | teaching | boxes, park-story | — | — |
| 160 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 161 | Visualizing a Missing Annotation (md) | both | — | toy | — |
| 162 | 🔎 Visualize a Missing Bounding Box (code) | both | — | toy | — |
| 163 | 📘 What Matters for Object Detection? (code) | both | — | — | — |
| 164 | Semantic Segmentation: Which Pixels Belong to the Animals? (md) | both | masks | — | — |
| 165 | Semantic Segmentation: Masks and Pixels (code) | both | masks | "Our COCO dataset provides…" | own variant sentence |
| 166 | 🟢 Exploring Segmentation Masks (md) | both | masks | — | own text variant (guiding question + blank answer) |
| 167 | Mask helpers and data preparation (code) | both | masks | builds `masks` from `boxes`; every row needs a segmentation | skip rows without one; skip chapter without masks |
| 168 | 🔎 Explore Segmentation Masks (code) | both | masks | — | — |
| 169 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 170 | 🟢 Pixel-Level Class Imbalance (md) | both | masks | "Do you expect birds…" | own text variant (guiding question + blank answer) |
| 171 | Pixel-Level Class Imbalance (code) | both | — | — | — |
| 172 | 🔎 Explore Pixel-Level Class Imbalance (code) | both | masks | needs `boxes_per_class` from Detection | — |
| 173 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 174 | 🟢 Mask Size, Shape, and Fine Structures (md) | both | masks | legs, tails, beaks | own text variant (guiding question + blank answer) |
| 175 | 🔎 Explore Mask Sizes and Fill Ratios (code) | both | masks | uses `_size` from cell 141 | C-09 |
| 176 | 🔎 Inspect Mask Fill Ratios (code) | both | masks | — | — |
| 177 | 🔎 Explore Thin Structures in Segmentation Masks (code) | both | masks | — | — |
| 178 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 179 | 🔴 Where Are the Animal Pixels? (md) | both | masks | "Wildwood Farm Park" cameras | own text variant (guiding question + blank answer) |
| 180 | 🔎 Explore Spatial Distributions of Animal Pixels (code) | both | masks | — | — |
| 181 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 182 | 🔴 Mask Quality: Finding Annotation Errors (md) | teaching | masks, park-story | planted errors in `park_masks` | own variant: "run the five checks on your masks" |
| 183 | Simulate Mask Annotation Errors (code) | teaching | masks, park-story | planted key `HUNT9_KEY` | — |
| 184 | Five simulated annotation errors The following figure ill… (md) | both | — | toy horse | — |
| 185 | 🔎 Explore Five Types of Mask Annotation Errors (code) | both | — | toy horse | — |
| 186 | Five automated mask quality checks. The next figure compa… (md) | both | — | — | — |
| 187 | 🔎 Explore Five Mask Quality Checks (code) | both | — | toy horse | — |
| 188 | Predict and investigate (md) | teaching | park-story | Picture A/B matching + answer key | own variant: inspect the suspects |
| 189 | 🔎 Detect Suspicious Segmentation Masks (code) | both | masks | runs on `park_masks` | input variable set per edition (teaching: `park_masks`, own: `animals`) |
| 190 | 💡 Sample Answer (code) | teaching | — | Sample Answer (Own edition has none) | — |
| 191 | Continue here (SUS) (md) | drop | — | her "Continue here" marker | remove |
| 192 | 10. Your project dataset (md) | own-only-later | images | "## 10." numbered heading | becomes the Own edition's opening |
| 193 | Checklist Dataset Mini-Project (code) | both | park-story | "Park example" lines, "section 4" | fix references (T-14) |
| 194 | 🔗 Resources: Image Datasets and Annotation Tools (code) | both | — | — | — |
| 195 | (box) 🎯 Objective (code) | own-only-later | — | objective box (kind she removed) | plain markdown |
| 196 | Your project has its own dataset, and it deserves the sam… (md) | own-only-later | images | folder-per-class only | superseded by the loader text |
| 197 | (box) ✏️ Your turn · Point the notebook at your dataset (… (code) | own-only-later | images | — | reuse in the loader instructions |
| 198 | # ✏️ YOUR TURN (code) | own-only-later | images | — | superseded by the loader's path setting |
| 199 | Dataset reader: finds your class folders, or says exactly… (code) | own-only-later | images | — | reuse as the folder-per-class reader |
| 200 | The next cell reads your folder (or, with `None`, builds … (md) | own-only-later | images, park-story | park sample with three planted problems | per-format samples (PRD Q3) |
| 201 | Read your folder (or build the sample), and show what was… (code) | own-only-later | images | `clean_images` (removed) → NameError | C-25 |
| 202 | (box) 🔍 Go deeper · Detection or segmentation project, an… (code) | own-only-later | — | "sections 7 and 8" | T-14 |
| 203 | Four checks run next and end in one report card. Check 1 … (md) | own-only-later | images | "130 cats against 50 sheep" | C-24: drop once her cells run on own data |
| 204 | Check 1: images per class, and the largest : smallest rat… (code) | own-only-later | images | duplicates cell 56 | C-24 |
| 205 | 💡 Look at the ratio: up to 3 : 1 is common; from 10 : 1 a… (md) | own-only-later | — | "section 4" | C-24 |
| 206 | Check 2 opens every file with `is_readable`, then reads i… (md) | own-only-later | — | "colour" | C-24 |
| 207 | Check 2: readable? size and colour mode of every file (code) | own-only-later | images | duplicates cells 37/46 | C-24 |
| 208 | 💡 Look for a second cluster of sizes or any mode that is … (md) | own-only-later | — | — | C-24 |
| 209 | Check 3 is section 5's copy hunt: exact copies by file ha… (md) | own-only-later | — | "section 5", `HASH_THRESHOLD` | C-24 |
| 210 | Check 3: exact, recompressed, flipped and rotated copies (code) | own-only-later | images | `THUMB_FOR_HASH` (removed) → NameError; duplicates 72/75/82 | C-24, C-25 |
| 211 | 💡 Look at the pairs across class folders first: one of th… (md) | own-only-later | — | — | C-24 |
| 212 | Check 4 suggests a split: it groups the copies, then make… (md) | own-only-later | — | — | keep |
| 213 | Check 4: a split suggestion, copies grouped, stratified b… (code) | own-only-later | images | "(section 6)" in a print | keep; read her variables |
| 214 | 💡 Look at "copy pairs across train/test": it must be 0. T… (md) | own-only-later | — | — | keep |
| 215 | The checklist in one place, to paste into your project no… (md) | own-only-later | — | — | keep |
| 216 | The report card for this dataset (code) | own-only-later | images | — | keep; read her variables |
| 217 | After the lab · Read your results and answer the guiding … (md) | own-only-later | — | — | — |
| 218 | Q25. 🟢 On the sample the folders are cat, dog, bird, hors… (md) | own-only-later | park-story | "Q25", park sample problems | drop Q-number (SN-012) |
| 219 | (box) 💡 Worked answer (Q25, on the sample) (code) | own-only-later | park-story | brown "Worked answer" box | drop (no Sample Answers in Own) |
| 220 | 💡 How to read the ratio: (md) | own-only-later | — | "section 4's ⚠️ Pitfall" | T-14 |
| 221 | 💡 How to read the sizes and modes: (md) | own-only-later | — | — | — |
| 222 | 💡 Every copy pair is a decision for a human: (md) | own-only-later | — | "section 5" | T-14 |
| 223 | (box) 🔍 Go deeper · The split is only a suggestion (code) | own-only-later | — | "section 6" | T-14 |
| 224 | The guiding questions (md) | own-only-later | — | "Q26" | drop Q-number; teaching gets a short closing cell without answers (PRD Q5) |
| 225 | 1. What is the task, and which mistake is the expensive o… (md) | own-only-later | park-story | park answers, "sheep about 50", sections 2–6 | park answers become short hints; T-07, T-14 |
| 226 | (box) ✅ Takeaway (code) | own-only-later | — | Takeaway box (kind she removed) | plain markdown or drop |

### What has to change so the shared cells run on any dataset

Most of her analysis cells already work on any `images` / `annotations` pair, because they loop over `CLASSES` and
colour by `CLASS_COLORS`. The work is concentrated in a few places. Setup (7) must stop fixing the six park animals:
it keeps the palette and `INK`, and the class list and class colours are set once, right after the data is loaded
(by cell 9 in the Teaching Edition, by the loader in the Own-Data Edition). The palette needs more than six colours,
because an own dataset can have twenty classes and N-027 wants one colour per class; the current
`MY_COLORS` code (201) silently repeats colours after six. `build_tables` (8) must accept image-level labels for
classification-only data, where there are no annotations to take the class from, and the loader must fill exactly
the columns her cells read: `images` = `id, file, path, width, height, class, n_animals` (see N-21), `annotations` =
`id, image_id, class, bbox, area, segmentation, iscrowd`, plus `images_by_id` (C-09).

The helpers to refactor are: the three `image_stats` readers in 36 and `class_shares` in 107 (take the table, C-16);
the six-panel grid in 46 (C-15); the two one-row-per-class figures in 19 and 141, which need to wrap for many classes;
the annotation browser 28 and the class browser 21, which must hide the box and mask toggles when those columns are
empty; the mask table in 167, which must skip rows without a segmentation; `find_copies` in 82 (C-20); and the mask
checks in 189, which read `park_masks` and should read one variable set per Edition (`park_masks` in teaching, the
student's `animals` in own). Chapters that need boxes (Detection, 125–163, plus 58–59) or masks (Segmentation,
164–181, 189) should be skipped with one printed line when the student's data has none (PRD Q2b). Park-only by the
PRD's rule: the dog-alert story (14, 61–65), the planted hunts (80–84, 157–160, 182–183, 188, 190) and the CCT20
group split (117, 119–122).

### Cells that need an Own-Data text variant

To avoid forty duplicated cells, let the build do the mechanical part: in the Own-Data Edition it drops every cell
titled "💡 Sample Answer" and appends a blank **Your answer:** line after every Task cell. Then only the cells with
park wording need a hand-written Own variant:

- **Front matter and chapter openers:** 0, 4, 6, 13, 14 (becomes guiding question 1: the task and the expensive
  mistake), 15, 125, 126, 165.
- **Tasks with park wording:** 18, 20, 22, 25, 27, 41, 58, 130, 170, 174, 179, 133 (docstring example only).
- **Park-only tasks replaced by one guiding question each:** 62 (which class is the expensive mistake?), 117/121
  (what ties your images together, and which split keeps those groups apart?), 182/188 (run the five checks on your
  masks and inspect the suspects).
- **Tasks that only need the mechanical treatment (no park wording):** 30, 38, 45, 48, 51, 54, 69, 74, 88, 91, 101,
  106, 109, 114, 127, 136, 140, 143, 147, 151, 154, 166.
- **The old own-dataset part (192–226)** becomes the Own-Data Edition's opening and closing: the loader replaces
  196–201, her analysis replaces 203–211 (C-23), 212–216 stay (split suggestion and report card), and the guiding
  questions 224–225 keep the park only as short hints ("in the park this showed …", PRD Q2). The Teaching Edition
  gets one short closing cell that links the Own-Data Edition and lists the guiding questions without answers
  (PRD Q5).
