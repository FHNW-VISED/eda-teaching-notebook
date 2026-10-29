# Work list: Susanne's copy (2026-10-28)

The one list of what is done to her copy, in order. Details per item live in the two audits next to this file:
`2026-10-28-susanne-copy/cleanup-audit.md` (C-, N-, T-items and the Edition map E) and
`2026-10-28-susanne-copy/run-audit.md` (R-items: what breaks when it runs, timings). Cell numbers are indices into
`2026-10-28-susanne-copy/her-copy.ipynb`; every item also names the cell title, so it can be found in `sections/`.

## Rules for whoever applies this

- Her copy is the base. Her visible text changes **only** where an item below says so.
- What she removed stays removed (SN-010). Only what a removal breaks is fixed.
- One commit per group below; `just build` and the smoke run of the Teaching Edition must pass after each group.
- Never edit `reviews/2026-10-28-susanne-copy/` (her copy, the audits) or `versions/`.

## Decisions (Simone, 2026-10-28)

All `fix` items in both audits are applied. The `propose` items were answered as follows:

| # | Item(s) | Answer |
|---|---|---|
| P1 | C-12 one `callout()` helper for her boxes (inner HTML verbatim; one Sample Answer style) | yes |
| P2 | C-13 one `browse()` helper | **no** — N-13's renames fix the bug; her widgets stay as built |
| P3 | C-07 pHash once (1074 → 1068 candidate pairs printed) | yes |
| P4 | C-08 MD5 lines into the 🟢 duplicates task | yes |
| P5 | C-16 helpers take their table as an argument | yes |
| P6 | C-20 vectorised `find_copies` | yes |
| P7 | C-23 Own-Data: drop old checks 1–3, keep split suggestion + report card | yes |
| P8 | N-21 `n_animals` → `n_objects` (her two sentences that name the column follow) | yes |
| P9 | N-22 `CLASSES` / `CLASS_COLORS` stay capitals ("decided once, at load") | keep |
| P10 | C-06 pHash-only match label | "recompressed" |
| P11 | T-19 heading levels per SN-011, MCQ gets a markdown heading, "Your Project Dataset" | yes |
| P12 | T-20 "(Optional)" | add to all 🔴 headings |
| P13 | T-21, T-22, T-23 (131 → plain markdown; 101 Pitfall stays a blockquote), T-24, T-25, T-26 | yes |
| P14 | T-16 CV diagram sentence matches the figure (gray = training, yellow = validation) | yes |
| P15 | T-28 one train/test colour pair; T-30 trail-camera colours not the park's | yes |
| P16 | T-13 "photos" → "images" in printed outputs | yes |
| P17 | T-12 one dataset name | **no** — only "play dataset" → "Play Dataset" (194) |
| P18 | T-29 one hash threshold | **no** |
| — | N-23 toy-horse constant names | no |
| — | T-17 💡 on every summary, T-18 title patterns | yes (C-12 does T-17) |

Earlier decisions (PRD Q1–Q11): CPU target; `COMPUTE_EMBEDDINGS = False`, DINOv2 only on a GPU or when set;
park DINOv2 vectors precomputed as `data/coco_animals_dinov2.npz`; the `%pip` line keeps only what Colab lacks
(`imagehash`; umap-learn, bokeh, transformers and torch are preinstalled, checked against Colab's pip freeze 2026-10-28).

### Run-audit items (R-01 … R-10) and where the two audits disagree

- R-01 vs C-03/C-24: `flag_odd_files` is **deleted** (C-03). Chapter 10's `build_my_sample` is replaced by the
  per-format park sample of the Own-Data loader, so `clean_images` is not rebuilt.
- R-02, R-03: apply (`THUMB_FOR_HASH = (256, 256)`; `load_thumb(..., (192, 192))` / `(256, 256)`), in the cells that
  move to the Own-Data Edition.
- R-05 = T-05, R-06 = T-01…T-07, R-07 = T-14: apply once (the cleanup audit's wording).
- R-08: the gate is `COMPUTE_EMBEDDINGS = False`, computing only on a GPU or when set; the precomputed file is
  `data/coco_animals_dinov2.npz` (ids + embeddings, matched by image id); `load_embeddings` is rewritten to read it
  (C-02's pattern) and print what happened; cells 96/97 fall back to "Color histograms" when the pretrained key is
  missing. No `%pip` for umap-learn/bokeh/transformers (preinstalled in Colab, CPU and GPU, checked 2026-10-28).
- R-09, R-10: inform the Own-Data loader; no change now.
- Runner: plain `nbclient` stalled three times on her notebook. If `just smoke` stalls, switch `tests/test_smoke.py`
  to a `jupyter_client` runner that waits for each execute reply (the audit's `build/run_audit2.py` works).

## Order

1. **Breakage** — C-24, N-01, N-13, C-15, and every R-item marked high.
2. **Dead and duplicated code** — C-01 … C-11, C-17 … C-21, C-14 (imports once; helpers cell after the data load).
3. **Names** — N-02 … N-20, N-21 (P8).
4. **Boxes** — C-12 (P1), T-17, T-18.
5. **Text** — T-01 … T-11, T-14 … T-16, T-19 … T-28 as decided.
6. **Embeddings gate** — C-22, precomputed park vectors.
7. **Editions** — E-map tags, Own-Data loader (section 01), own text variants, closing cell (PRD Q5), C-23.
8. **Tests** — content tests on her rules, Own-Data smoke per format, CPU timing report.

## Log

| Group | Commit | Notes |
|---|---|---|
| 1 · Breakage | `103fd55` | C-24 (flag_odd_files + EXTREME_SIDE_PX deleted, `build_my_sample` samples `pool = images`), R-02 (`THUMB_FOR_HASH = (256, 256)` in Check 3), R-03 (two int sizes → tuples), N-01, N-13 (prefixes as in the audit; also `pages` → `size_pages`, `examples` → `fill_examples`, `bursts` → `burst_list`, 149's `output` → `overlap_output`), C-15 (`math` imported in setup). Smoke (project venv): runs to cell 94 and stops on `import torch`, which her copy needs and the venv does not have; that is fixed by group 6, not here. Checked instead with a jupyter_client run that continues past errors: the only errors are 94 and its cascade 95–98; on the audit venv (torch installed) 0 errors. |
| 2 · Dead and duplicated code | `5b155b3` | C-01 (answer_box deleted; `callout()` comes in group 4), C-02, C-03 (group 1), C-04, C-05, C-06 (P10: "recompressed"), C-07, C-08, C-09, C-10, C-11, C-14, C-16, C-17, C-18, C-19, C-20, C-21. Deviations: **N-16 and N-17 applied here**, because C-14 deletes the `_Image` and `patches` imports they rename. **C-19** became two helpers, `jpeg_bytes` (the file, used by the planted copy in 81) and `jpeg_copy` (the reopened image, 78 and 86): 81 writes the JPEG bytes to the card, so a reopened image would have had to be encoded a second time. **C-14**: the shared helpers cell is "Helpers: Images and Masks" (load_thumb, frame, to_rle, decode, encode); the colour constants sit in setup next to `INK` (C-18's wording); box cells keep their own `HTML` import until group 4 replaces them. **C-07**: `MAX_PAIR_BITS = 20` added to the helper cell here (C-07 uses it); `PAIR_THRESHOLD` follows in group 3. **C-16**: `class_shares(table, ids)` takes the id-indexed table, so it is called with `images_by_id` (C-09). **C-20**: Check 3 keeps its own `ia`, `ib`, `same class` columns on top of `find_copies`; `THUMB_FOR_HASH` (group 1) is gone again with it. Smoke: as group 1 (project venv stops at the torch cell 94); audit venv 0 errors, text outputs identical to group 1 except the moved MD5 lines. |
| 3 · Names | `1bd34f0` | N-02, N-03, N-06 (the cache folder keeps its name `hunt5`: renaming it would leave a stale folder in the shared `data/cache`), N-07 (`RESULTS["hunt9"]` kept, as the item says), N-08, N-09, N-10, N-12 (`PAGE_SIZE` in setup; `OVERLAP_IOU` in the IoU helper; `BURSTS_PER_PAGE`; `UMAP_SETTINGS`; `TEST_SIZE` in the leakage browser; `PAIR_THRESHOLD = 8` in the duplicate helper, also the slider's default, whose max now reads `MAX_PAIR_BITS`), N-14, N-15, N-18, N-19 (190's scratch names `_lines`, `_img`, `_buf`, `_items`), N-20, N-21 (P8; the crowded-images browser now merges only `id` back, since `images` already carries `n_objects`). N-04, N-05, N-11 were done by C-07/C-06 in group 2; N-13, N-16, N-17 in groups 1–2. N-22 kept, N-23 not done (decisions). Smoke: as before (project venv stops at cell 94); audit venv 0 errors, outputs identical to group 2 except the two renamed columns. |
| 4 · Boxes | `e75edd4` | C-12 (P1): `callout(kind, title, body_html, fig=None)` in setup, kinds answer / theory / recipe / deeper with her colours; 56 box cells converted, inner HTML copied verbatim (checked: the rendered text of every box is identical before and after, apart from the summary label); the box title is passed as written, icon included, so her own titles stay. T-17, T-18 (titles chosen for the untitled chapter-10 boxes; those six boxes keep their old HTML, they are outside C-12's list and move to the Own-Data Edition). **T-23's 131 part done here** (the box became markdown), not in group 5. **Build**: `scripts/build.py` now runs the notebook's own `callout()` (taken from the sections source with `ast`) for every cell that is a `# @title` line plus one `callout(...)` call with literal arguments, and saves the HTML as the cell's output (N-034); 55 cells get it; the one with `fig=` (190) and her raw-HTML cells (Learning Objectives, How to Use This Notebook, the six chapter-10 boxes) still show only after running. `tests/test_port.py` fails, but it already failed before this branch (it expects the v05 sections). Smoke: as before; audit venv 0 errors. |
| 5 · Text | `23c2509` | T-01 … T-11, T-12 (only 194, P17), T-13 (P16), T-14 (the clean-up audit's wording, chapter-10 cells included), T-15, T-16 (P14), T-19, T-20 (P12, also on the new 124 heading), T-21, T-22, T-24, T-25, T-26, T-27, T-28 (P15: the leakage browser takes the CV diagram's light gray / yellow). Choices: **T-13** applied to every printed string, figure title, axis label and legend in the chapters before Your Project Dataset (the audit's cell list missed a few, e.g. 42, 78, 172, 175, 180), never to her markdown, her box text or her cell titles ("🔎 Compare Class Distributions: Photos vs. Boxes" stays). **T-19**: 126 became a bold paragraph (one change) rather than nine task headings moving to `####`. **T-14** for 130 drops the sentence's final period after the heading's "?". **T-07** reads "59 sheep" / "sheep about 59" as written in the clean-up audit (R-06 suggested "about 60"). **T-10** also renamed two local variables (`grey` → `gray`, `centred` → `centered`). T-23: 101 keeps its blockquote; 131 was done in group 4. Not done: T-29 (P18 no); T-30 is approved (P15) but outside this run's scope, left for the next pass. Also fixed the long lines group 4 added to `scripts/build.py`. Lint: only the `01_load_your_dataset.py` FAIL is left (N-006's Q25 went with T-27). Smoke: as before; audit venv 0 errors; output diff vs group 4 = exactly the listed text changes plus the 98 % Rule B line. |
| 6 · Embeddings gate | this commit (`git log --grep "group 6"`) | C-22 + R-08 as resolved, with the precomputed `data/coco_animals_dinov2.npz` (committed by the orchestrator in 5af26dc): `COMPUTE_EMBEDDINGS = False` (a Colab checkbox); `load_embeddings(ids)` reads the npz through `fetch_manifest` (local `data/`, in Colab from `DATA_BASE_URL`), matches by image id (works for the smoke subset) and prints what happened; DINOv2 runs only when the box is ticked, or on a GPU runtime when the file is missing or does not cover the ids (GPU = `nvidia-smi` on the path, so torch is never imported on the CPU path); torch, transformers and tqdm are imported inside `dinov2_embeddings`, which now loads the model itself; otherwise one line says it is skipped, the rough minutes (633 images at an assumed 5 images/s on Colab's free CPU ≈ 2 min, `CPU_IMAGES_PER_SECOND`, a guess, not measured) and how to turn it on; 96/97 fall back to "Color histograms" through lowercase `umap_features` / `similarity_features` (N-058: the `@param` settings are not reassigned). On a GPU with the file present the file is used: computing would give the same vectors. `%pip` line removed; `cached_features` kept. **Smoke (project venv, `tests/test_smoke.py` with nbclient, unchanged): 9 passed in 21 s, no stall**, so the runner was not rewritten. Skip branch checked once with the file name pointed elsewhere: no error, cells 96–98 on color histograms. **Full data, CPU, OMP/MKL 2 threads, jupyter_client runner (`build/run_nb.py`), images already cached: 0 errors, 23.6 s of cell time; only one cell over 5 s: Precompute Image Representations and UMAP, 6.0 s (two cold UMAP fits); next: UMAP plot 3.0 s, pHash pairs 2.9 s, Measure Image Dimensions 2.1 s, Setup 2.0 s.** Numbers match the audit: 633 images, 1068 candidate pairs (C-07), 0 MD5 groups, 3 of 5 vs 5 of 5, her within-class medians, 59 suspects of 1052, Check 3 finds the exact and the flipped copy. Ruff: 24 findings remain, all present before this branch (B905/E731/I001/E702 in her cells and chapter 10, E501 in `scripts/build.py` line 6, `scripts/lint_notebook.py`, `tests/test_editions.py`); none added. |
| 7 · Editions (mechanics) | `b75a7af` `e7ff218` `13ed4be` `20aa68a` `b74e3e3` + the "Editions, step 6" commit | Step 1: `load_play_dataset(per_class)` in Helpers: Data Download and Tables (smoke sample in a fixed park-class order, `PARK_CLASSES`); `build_tables` takes the most frequent annotation class, `"(no objects)"` without one. Step 2: `sections/01_load_your_dataset.py` (all `own`): loader text from `build/own_loader_text.md` (go-deeper title written as "🔍 Go Deeper: Not In This List?"), the upload box (moved from chapter 9, adapted), ✏️ settings (`LABEL_FORMAT` is a Colab dropdown; `DATASET_PATH`, `TASKS` plain lines), three helper cells, the load cell, "🔎 See How One Image Is Labeled" (`stored_label()` prints the label as stored). Draft changes: `read_dataset` returns the dataset folder too; YOLO polygon samples write every outline part; unreadable files are left out and counted (report card). Step 3: tags per the E-map; "Helpers: Simulate Mask Annotation Errors" split, its thresholds and `_crude_polygon` go to a shared "Helpers: Mask Quality Checks". Step 4: chapter 9 per Edition; forced text fixes: "Check 4 suggests" → "The next cell suggests", "listed in check 2" → "counted in the report card". Step 5: 18 `# needs:` markers; Helpers: Bounding Boxes and Helpers: Masks and Data Preparation stay unmarked (the toy cells use their functions); `masks_to_check`; the two split cells survive a 4-per-class dataset. Step 6: `tests/test_own_smoke.py` (7 formats + broken YOLO line), runner factored into `run_smoke()` in `tests/test_smoke.py`, `just smoke` runs both. Teaching smoke text outputs identical to before step 1 except chapter 9's removed cells. Ruff: 27 → 17 (no new findings). Still for the writer: the own text variants (title, How to Use, Meet the Park opener, task wording, chapter-9 Checklist, "How to read the ratio" points at the Teaching-only Class Imbalance task, the Takeaway still says "four checks" and "folder-per-class"). |
