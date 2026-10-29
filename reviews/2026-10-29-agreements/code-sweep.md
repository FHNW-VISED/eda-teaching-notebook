# Code sweep: capitals, PEP 8 names, consistency, lint (2026-10-29)

Branch `fix/results-lowercase` at `2bff236`. These are the "code clean-up checks, PEP 8 variable names and consistency
checks" Susanne asked for. Each Edition was analysed as its own program: the code cells of the built notebook in order,
with magic lines and `# @title` lines removed. The notebooks were rebuilt into `build/sweep/` (not the repo root), and
the fresh build is byte-identical to the committed `.ipynb` files, so the sweep reads exactly what students get.

**What "visible" means here.** Almost every code cell is a Colab form: 128 of 130 code cells in the Teaching Edition
and 89 of 92 in the Own-Data Edition show only their title until a student clicks "show code". So there are three
levels in the table below:

- **seen**: a widget label, printed line, figure label or title. Every student sees it.
- **code**: the code of a 🔎 or 💡 cell. A student who opens the cell to see how the figure is made reads it.
- **helper**: a `Helpers:` cell or the Setup cell. Readable, but few students open it.

## Summary

The code is in better shape than the raw counts suggest. The repo's own ruff config passes clean. No module-level name
shadows a builtin, every function is snake_case and the one class (`CocoBuilder`) is CapWords. Spelling is US English
everywhere (color, gray, center, labeled, normalize, -ize): zero UK spellings in identifiers or in anything a student
sees. Most of the script hits are throwaway names like `fig`, `ax` and `c` that every plotting cell reuses, which is
normal notebook style.

What is worth fixing is smaller: one setting the notebook changes behind the student's back (`TASKS`), the interactive
browsers using three naming schemes and two button styles, two names for "bit distance" and two for "image", a handful of
single-letter or reused names in 🔎 cells, a dead helper, and a Setup comment that is no longer true.

| Check | Teaching Edition | Own-Data Edition |
|---|---|---|
| 1 Capitals | 80 UPPER names; 19 hit by the script; 0 must-fix, 1 small (`CCT_COLORS`), 18 accepted (decided once from `SMOKE`/`DATA_DIR`, or a fixed table) | 79 UPPER names; 17 hit; **1 must-fix (`TASKS` is mutated)**, 16 accepted; plus 1 inline setting (`4 if SMOKE else 20`) |
| 1 Lowercase settings | 0 | 0 |
| 2 PEP 8 names | 0 case errors at module level; 0 builtin shadows; 14 single-letter module names + 2 single-letter capitals in a function + 1 `X` param; **5 worth renaming** | 0 case errors; 0 builtin shadows; 16 single-letter + 1 `X` param; **4 worth renaming** |
| 3 Consistency | 46 names bound in more than one cell (about 38 are `fig`/`ax`/`c`/`k`-style throwaways); 2 unused names; 1 repeated import (justified); 0 unused imports; 0 UK spellings; 2 term splits | 45 multi-cell names; 5 unused names (1 false positive); 1 unused import; 0 UK spellings; 2 term splits |
| 4 Ruff | repo config: **0** · `--select N,E,W,F,B,UP,SIM`: 48 (44 E402 by design, 4 real) · without per-file ignores: 1734 (1258 F821 + 472 E501 hidden by design) | same files, same counts (ruff reads `sections/`, not Editions) |

The tables below have 35 rows. 16 are worth doing now (marked **fix**; #17 folds into #16, so 15 separate changes),
11 are optional, and 8 are recorded as accepted so the next sweep does not raise them again. The ruff table adds 4
rows, all already covered or one line each.

## Findings

Cell numbers are the cell index in the built notebook, Teaching / Own. "—" means the cell is not in that Edition.
Rows that change wording in her boxes or titles should go through Susanne first: her text comes from her copy.

### Check 1: capitals

| # | Edition | Cell | Name or text | Problem | Fix | Visible? | Priority |
|---|---|---|---|---|---|---|---|
| 1 | own | — / 20 `✏️ YOUR TURN` (set), — / 24 `Load Your Dataset` (changed) | `TASKS` | The setting the student typed is changed by `TASKS.append("detection")` (`01_load_your_dataset.py:933`). The skip hook and every chapter then read a list the student never wrote. | Keep `TASKS` as typed. In the loader derive once: `tasks = [*TASKS, "detection"] if "segmentation" in TASKS and "detection" not in TASKS else list(TASKS)`, and have the skip hook (`01:920`) and the later `in TASKS` tests read `tasks`. Or test `"detection" in TASKS or "segmentation" in TASKS` where boxes are needed. | code; the effect is seen in the "⏭️ Skipped" lines | **fix** |
| 2 | own | — / 24 `Load Your Dataset` | `load_play_dataset(per_class=4 if SMOKE else 20)` | The park-sample size is a bare number inside a call. Teaching names the same kind of value `SMOKE_IMAGES_PER_CLASS`, which the Own Edition defines (shared cell 8) and never uses. | Add `PARK_SAMPLE_PER_CLASS = 4 if SMOKE else 20` beside the other settings and use it. Move `SMOKE_IMAGES_PER_CLASS` from the shared Helpers cell into the teaching-only load cell 9. | helper | **fix** |
| 3 | teaching | 121 / — `Helpers: Trail Cameras` | `CCT_COLORS = dict(zip(CCT_CLASSES, [...], strict=True))` | Built with a function call where a literal works. | Write it as a literal dict, like `CLASS_COLORS`. | helper | optional |
| 4 | both | many (e.g. 38/50 `THUMB`, 47/59 `N_GRAY`, 144/141 `HEAT_BINS`, 8/8 `IMAGE_CACHE`) | 10 `X if SMOKE else Y` names (9 in Own) and 4 `DATA_DIR / ...` paths (2 in Own) | Not a literal, but decided once from `SMOKE` or `DATA_DIR` and never changed. | Keep. Add to RULES: "a literal, a `# @param`, or chosen once from `SMOKE` / built from `DATA_DIR`". | code | accepted |
| 5 | both | 7/7 `Setup` | `SMOKE`, `IN_COLAB` | Read from the environment, once. | Keep: they are the purest settings in the notebook. | helper | accepted |
| 6 | both | 191/183, 97/101, — / 22 | `CHECK_RULES` (lambdas, f-strings), `UMAP_SETTINGS = dict(...)`, `READERS` (Own: format → reader function) | Fixed tables built once, not from data, never mutated. | Keep. Optional: `UMAP_SETTINGS` as a `{...}` literal. | code / helper | accepted |
| 7 | both | — | `CLASSES`, `CLASS_COLORS` | Assigned twice in Own (Setup + loader), once in Teaching; never mutated. | Within the agreed exception. | helper | accepted |

### Check 2: PEP 8 names

| # | Edition | Cell | Name | Problem | Fix | Visible? | Priority |
|---|---|---|---|---|---|---|---|
| 8 | both | 191/183 `🔎 Explore Five Mask Quality Checks` | `H, W = 60, 84` | Single-letter capitals at module level (look like settings, are a toy image size) and repeat `toy_horse`'s own defaults. | `original = toy_horse()` then `height, width = original.shape`; use `height`, `width` below. | code | **fix** |
| 9 | teaching | 187 / — `Helpers: Simulate Mask Annotation Errors` | `W, H` inside `fits()` (ruff N806) | Capitals for local variables. | `width, height`. | helper | **fix** |
| 10 | both | 99/103 `🔎 Explore Similar and Unusual Images` | `i, j = np.triu_indices(...)` | Single-letter module names; `i` is a loop variable in three other cells. | `first, second = np.triu_indices(...)` (row indices of each pair). | code | **fix** |
| 11 | both | 44/56 `🔎 Explore Image Dimensions and Aspect Ratios by Class` | `s = image_stats[image_stats["class"] == c]` | Single letter for a table. | `class_stats`. | code | **fix** |
| 12 | both | 109/113 `🔎 Compare Random and Stratified Splits`, 175/168 `🔎 Explore Pixel-Level Class Imbalance` | `x = np.arange(len(CLASSES))` | Single letter; `x` means something else in four other cells. | `bar_x`. | code | **fix** |
| 13 | both | 114/118 `🔎 Visualize Cross-Validation Strategies` | `summarize_folds(name, splitter, X, y, groups)` (ruff N803) | Capital parameter. | Keep: `X, y` is the scikit-learn convention students meet everywhere. | code | accepted |
| 14 | both | 194/— `💡 Sample Answer`, —/25 `🔎 See How One Image Is Labeled` | `x, y, w, h = annotation["bbox"]`, `_x, _y, _w, _h` | Single letters. | Keep: the COCO box convention, read as one unit. | code | accepted |
| 15 | both | 73/81, 76/84, 109/113, 180/173 | `_i`, `_j`, `_y`, `_x`, `_m` | Single-letter underscore temporaries. | Keep; optional `_rows_a`, `_rows_b`. | code | accepted |

No module-level name shadows a builtin in either Edition (checked: assignments, loop targets, `with`, imports, function
names, parameters, locals, comprehension and lambda variables). `min=`/`max=`/`format=` hits are keyword arguments.

### Check 3: consistency

| # | Edition | Cell | Name or text | Problem | Fix | Visible? | Priority |
|---|---|---|---|---|---|---|---|
| 16 | both | the 11 browsers: 29/41, 73/81, 90/94, 99/103, 122/—, 132/129, 141/138, 148/145, 158/155, 171/164, 179/172 | Previous: `ann_prev_btn`, `sim_prev_btn`, `burst_prev_btn`, `maskview_prev_btn`, `fill_prev_btn` / `dup_previous`, `boxview_previous`, `size_previous`, `crowd_previous`, `quality_previous` / `_prev`. Next: `*_next_btn` / `*_next_button` / `_next`. Display: `*_output` / `sim_preview` / `_out`. Counter: `*_counter` / `*_page_label`. | Three naming schemes for the same widget roles. The quality browser is also verb-first (`show_quality_page`, `change_quality_page`, `update_quality_images`, `shuffle_quality_images`) where every other one is prefix-first (`crowd_show_page`). | One pattern: `<prefix>_prev_btn`, `<prefix>_next_btn`, `<prefix>_counter`, `<prefix>_output`, callbacks `<prefix>_show_page()` and `<prefix>_navigate(step)`. Rename the quality callbacks to `quality_show_page` etc. | code | **fix** |
| 17 | both | 22/34 `🔎 Browse Images Within a Class`, 90/94 `🔎 Find Unusual Images Using Color Histograms`, 127/— (quiz) | `class_dropdown`, `page_slider`, `box_checkbox`, `out`, `update_pages`; `_prev`, `_next`, `_out`, `_navigate`, `_show_class`; `check_answer`, `show_solution` | The only widget cells without a prefix. `out` and `update_pages` are generic enough that a later cell can silently take the name. | Prefix them: `gallery_*`, `atypical_*`, `mcq_check_answer` / `mcq_show_solution`. | code | **fix** (folds into #16) |
| 18 | both | 122/—, 171/164, 179/172 vs the other 8 browsers | Buttons "Previous" / "Next" with chevron icons vs "← Previous" / "Next →" | Students see two button styles for the same action. | Use "← Previous" / "Next →" in the burst, mask and fill-ratio browsers (8 of 11 already do). | **seen** | **fix** |
| 19 | both | 22/34, 29/41, 122/—, 124/—, 171/164, 179/172 | Checkboxes "Show bounding boxes" (3), "Show boxes" (2), "Bounding boxes" (1); "Show segmentation masks" (1), "Show masks" (2) | Three labels for one control. | "Show bounding boxes" and "Show masks" everywhere (or "Show boxes"/"Show masks"; pick one). | **seen** | **fix** |
| 20 | both | 22/34 and 29/41 (same chapter) | `_has_boxes` and `ann_has_boxes` | The same fact computed twice under two names. | Compute `has_boxes` and `has_masks` once where the tables are built (cell 9 / 24) and use them in both browsers. | code | optional |
| 21 | both | 57/69, 64/—, 9/24, —/189 (ch. 9) | `images_per_class` (dict), `class_counts` (Series), `_counts` (Counter in the loader, `value_counts()` in cell 64) | One concept, four names and three types. Chapter 9 recomputes what chapter 3 stored. | In chapter 9 use `images_per_class` (`min(images_per_class.values())`); in cell 64 use `images_per_class` too. | code | optional |
| 22 | own | — / 24 `Load Your Dataset` then — / 49 `Helpers: Measurement` | `_fmt` | First the label format the sample is written in (a string), then a number-formatting function. | Rename the string `_sample_format`. | helper | **fix** |
| 23 | both | 144/141, 175/168, 191/183 (+ locals in `show_boxes`, `show_masks`) | `own` | Three meanings (one class's boxes, one image's masks, the tight box of a mask), and it reads like the Own-Data Edition's name. | `class_boxes`, `image_masks`, `tight_box`; locals `image_boxes`, `image_masks`. | code | **fix** |
| 24 | both | 178/171 (shared) then 187/— (teaching only) | `masks_to_check` | Teaching: first the real masks, then overwritten by `park_masks` (five planted errors) in a Helpers cell. Own: the real masks. Documented in a comment, so deliberate. | Optional: assign it only in Edition-specific cells, so each Edition binds it once. | helper | optional |
| 25 | both | 20/32 vs 60/72; 65/— vs 193/184 | `_per_class` (6, then a list of Series); `_cols` (6, then a column list) | Same temporary name, different meaning. | `_rows_per_class`; `_n_cols` / `_table_cols`. | code | optional |
| 26 | both | 72/80 `Helpers: Duplicate Detection and Image Quality` | `is_readable()` | Defined, never called in either Edition. | Delete it. | helper | **fix** |
| 27 | teaching | 161 / — `🔎 Simulate and Detect a Missing Annotation` | `missing_box_key` | Never used. The comment says the answer key is kept hidden on purpose. | `park_boxes, _answer_key = simulate_missing_box(...)`. | code | optional |
| 28 | own | shared cells 7, 8, 9, 89 | `GroupShuffleSplit`, `SMOKE_IMAGES_PER_CLASS`, `encode()`, `score_hunt()` | Defined in shared cells, used only by Teaching-only cells, so dead in the Own Edition. (`announce_skip` also shows as unused but is called through the skip hook's text: false positive.) | Harmless. If tidying: move `score_hunt` into the teaching-only hunt cell; leave `encode` (a general mask helper) and the import. | helper | optional |
| 29 | both | 7/7 `Setup` | Comment "Every import the later sections need is made here, once" | Not true: `pandas`, `Path`, `Counter`, `json`, `urllib` are imported in `Helpers: Data Download`; `ListedColormap`, `dctn` in the pHash cell; `html.escape` (Teaching 194); `StratifiedGroupKFold`, `zipfile`, `csv`, `xml.etree` (Own). The heavy ones (torch, transformers, umap, bokeh, google.colab) are rightly imported where used. | Move `pandas`, `Path`, `Counter` into Setup (they are used everywhere) and reword: "The common imports are made here. A few heavy or chapter-specific ones are imported where they are first used." | helper | **fix** |
| 30 | both | 7/7 `Setup` | 11 inline `# noqa: F401` | Lint noise in the cell students are told to run first. | Add `"sections/00_front_matter.py" = ["F821", "F811", "E501", "F401"]` to `per-file-ignores` in `pyproject.toml` and drop the inline comments. | helper | optional |
| 31 | both | 2, 4, 7 | `from IPython.display import HTML` three times | Repeated import. | Keep: cells 2 and 4 run before Setup and must stand alone. | helper | accepted |
| 32 | both | 71/79 (definition), 76/84 x-axis, 79/87 bit figure | "Hamming distance" (3) vs "bit distance" (12 Teaching, 5 Own) | Two names for one number. The cell is titled "Explore Bit Distance Distribution" but its x-axis says "Hamming distance to the closest other image (bits)". | Keep the one definition "Hamming distance (bit distance)" in the theory box, then "bit distance" everywhere: axis "Bit distance to the closest other image", figure note "bit distance = number of differing bits". | **seen** | **fix** (axis/figure); her box: ask |
| 33 | both | 137/134 title, 178/171 axis, 07/08 code | "Photos vs. Boxes", `by_photos`, `photo_share`, `photo_rle`, `n_photos`, `_photo_px`, `_photo()` | "image" is the notebook's word (518 uses vs 26 "photo" in Teaching). Cell 137 counts "Photos vs. Boxes"; cell 175 counts the same unit as "images / boxes / pixels". | Title "Images vs. Boxes"; `by_images`, `image_share`, `image_rle`, `n_images`; prose "photo" in her text: ask. | **seen** (title, axis) + code | optional (title: ask) |
| 34 | both | 189/181, 191/183 | "Picture A" / "Picture B" | Figure labels; the rest of the notebook says "figure" or nothing. | Fine as is; "Figure A/B" if she prefers. | **seen** | optional |
| 35 | both | 7, 43/55, 65/—, 158/155 | `random.seed` + `np.random.seed` (Setup), `np.random.default_rng(SEED)`, `random.Random(SEED)` | Three ways to get seeded randomness. | Optional: `np.random.default_rng(SEED)` everywhere new. | code | optional |

US vs UK spelling: all US, zero UK, in identifiers, strings, comments, titles and markdown of both Editions. The one hit
("analyses", Own front matter) is the plural noun, correct in both. Other term families checked and found consistent:
IoU (no `IOU`), pHash (no `phash`/`PHash`; "perceptual hash" only as its expansion), MD5, near-duplicate, dataset, UMAP,
grayscale, aspect ratio, cross-validation.

No helper is defined twice, and no two function bodies are 80 % or more alike. The closest pairs are the browsers'
page logic (for example `crowd_go_to_page` / `fill_navigate`, 75 %) and the three galleries `show_rows` / `show_boxes` /
`show_masks` (62–65 %). One shared pager helper would remove the hand-written prev/next/counter code from 11 cells, but
that is a refactor, not a clean-up, and the repeated pattern is easy to read cell by cell.

### Check 4: ruff

`uv run ruff check sections scripts tests` (the repo's config): **All checks passed.**

`uv run ruff check --select N,E,W,F,B,UP,SIM sections --output-format concise`: 48 findings. On the command line,
`--select` also drops the config's `ignore = ["E402"]`, so 44 of them are E402 (imports where they are taught), which
the repo allows on purpose. The 4 real ones:

| # | Rule | Where | What | Fix | Visible? |
|---|---|---|---|---|---|
| R1 | SIM102 | `00_front_matter.py:400`, `fetch_manifest()` in `Helpers: Data Download and Tables` | nested `if` | `if not path.exists() and download_file(...) is None: raise ...` (same behaviour: `and` short-circuits) | helper |
| R2 | N803 | `06_splits_leakage.py:281`, `summarize_folds(..., X, y)` | capital parameter | keep (row 13) | code |
| R3, R4 | N806 | `08_segmentation.py:973`, `fits()` | `W`, `H` | `width`, `height` (row 9) | helper |

With the per-file ignores switched off (`--config 'lint.per-file-ignores = {}'`) ruff adds what they hide on purpose:
1258 F821 (names made in an earlier section; one kernel, so expected) and 472 E501. Of the long lines, 424 are her HTML
boxes or markdown, 6 are prose, and **42 are code**, mostly long `print()` messages and sample-answer strings in
`01_load_your_dataset.py`, `04_errors_cleaning.py` and `09_your_dataset.py`. Colab wraps them, so they matter only a
little; break them when a cell is touched anyway. F811 (redefinition) hides nothing: zero.

For students the ruff findings that matter are none of the E402/F821 ones (those are how a sectioned notebook works).
R3/R4 and the long code lines are the only ones a student who opens a cell could notice.

## How the scripts work

All scripts live in `build/` (gitignored scratch) and run with `uv run python build/<script>.py`. Results land in
`build/sweep/*.json`.

- `build/sweep_lib.py`: builds both Editions into `build/sweep/` with `scripts/build.py`'s `build_all(out_dir=...)`,
  reads each `.ipynb`, blanks `%`/`!` magic and `# @title` lines (keeping line numbers), parses every code cell with `ast`,
  and records each cell's index, section, title and whether it is a collapsed form.
- `build/sweep_capitals.py`: collects every module-level binding per Edition (including inside module-level
  `if`/`for`/`try`/`with`), and for each UPPER_CASE name reports more than one binding, a non-literal value (split into
  "derived from other settings" and "computed"), assignment inside a loop, and any mutation anywhere in the Edition
  (`X[k] =`, `X.attr =`, `X.append/extend/update/...`, augmented assignment, `global X` reassigned in a function);
  `CLASSES`/`CLASS_COLORS` exempt; then lists lowercase literal names bound once and read in more than one cell.
- `build/sweep_naming.py`: checks module-level variables (snake_case or UPPER), functions and methods (snake_case),
  classes (CapWords), single-letter module names outside `for` targets, function parameters and locals, and builtin
  shadowing at module level, in functions, comprehensions and lambdas.
- `build/sweep_consistency.py`: lists names bound in more than one cell with each binding's value (widget-like names
  first), names bound to an identical expression, plain `a = b` aliases, functions defined twice or with bodies at least
  80 % alike (difflib on the unparsed body), module-level names never read anywhere in the Edition, imports repeated
  across cells or never used, US/UK spelling pairs in identifiers and in every student-visible string (markdown, code
  string literals, comments, form titles), and term families with more than one variant in visible text.
- Check 4 is ruff itself: the repo command, the wide `--select`, and the same with
  `--config 'lint.per-file-ignores = {}'`, saved as `build/sweep/ruff_wide.txt` and `build/sweep/ruff_noignores.txt`.

Judgement on top of the scripts: every script hit was read in its cell before going into the table, and the accepted
rows say why they stay.
