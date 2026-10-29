# Fix log: the 26 rows of `fix-list.md`, applied (2026-10-29)

Branch `fix/results-lowercase` (worktree `eda-apply-2026-10-28`), three commits on top of `1417922`:
`7f5085d` §1 Own-Data Edition content, `464ec6c` §2 Both Editions, `997208e` §3 Tests, SPEC, rules; this log is
committed after them. Each section passed `just build`, `just lint` and `just smoke` before its commit. Tests were
written before the code they test and seen failing. T = Teaching Edition, O = Own-Data Edition.

## Per row

| # | Status | Where, or what was done differently and why |
|---|---|---|
| 1 | DONE | `00_front_matter.py`: "Optional Coding Exercises" markdown, the `_ref_double` cell and the `double` ✍️ cell tagged `edition="teaching"`. `07_object_detection.py`: the ✏️ `count_boxes_per_class` note, its reference cell and its ✍️ cell tagged Teaching; new O-only hidden cell "Helpers: Count Boxes per Class" (`# needs: boxes`) computes `boxes_per_class` with the reference body; the O task text keeps its Before/Observe questions, and its "Coding Exercise" block became one **Explore:** line (compare with the image-level distribution). Three `own_wording.tsv` rows that swapped words in the now Teaching-only cells were deleted (the build fails on a swap that no longer matches). |
| 2 | DONE | `09_your_dataset.py`: new O-only `# ✏️ YOUR TURN` cell `SPLIT_COPY_THRESHOLD = 8` with a one-line comment, above the split cell; the split reads it, the cell comment and the "Every copy pair" hint point at it ("lower `SPLIT_COPY_THRESHOLD` in the cell above the split suggestion"). The markdown before it now says "The next two cells…" and explains the setting. Value 8 is the row's example (and `PAIR_THRESHOLD`'s value); before, the split silently used the section-4 slider's default of 5, so O may now flag a few more copy pairs. |
| 3 | DONE | `01_load_your_dataset.py`: `tasks_to_run` is derived once in the load cell (classification always; detection if TASKS asks for detection or segmentation; segmentation if asked). The loader (`read_dataset`, sample format, polygons), the summary lines, "See How One Image Is Labeled" and the skip hook read `tasks_to_run`; the hook uses `globals().get("tasks_to_run", ())`, so before the dataset is loaded every box/mask cell skips instead of raising. `TASKS` is never changed. O text that said a chapter runs "if `TASKS` includes `"detection"`" now adds "or `"segmentation"`" (Objects per Image task, Object Detection opener), because segmentation now implies detection without touching TASKS. |
| 4 | DONE | `01_load_your_dataset.py`: `is_git_url()` + `clone_git_repository()` in "Helpers: Find and Open Your Dataset"; `open_dataset()` checks for a git address first, clones once with `git clone --depth 1 --quiet` into `data/cache/my_upload/<repo>` (reused if `.git` is there, a failed clone stops with a plain ❌ message). Upload instructions: an "Option D · Paste the address of a git repository" line, plus "four options" and a (D) example in the ✅ list (the box became markdown in row 10). `DATASET_PATH`'s comment mentions it. Test: `tests/test_own_smoke.py::test_own_edition_runs_on_real_layouts[git-clone]` builds a local bare repo `park_sample_repo.git` from the COCO park sample (no network), written first and seen failing ("folders labels carry no boxes": the bare repo was read as a folder). |
| 5 | DONE | `01_load_your_dataset.py`, end of the loader chapter: one sentence, then O-only "🔎 See One Image in Every Format". It loads one image per park class, picks one with 1–2 polygon objects, draws it once with its box(es) in its park class color, and for folders, CSV, COCO, YOLO boxes, VOC, YOLO segmentation and PNG masks writes it with `write_sample()` into `data/cache/format_demo/<format>` and prints it with `stored_label()` (folder path, CSV row, COCO image + annotation with the polygon start + category, YOLO line, VOC `<object>`, YOLO-seg line start, PNG class indices with pixel counts). It sits after the helpers, not next to the format text above the ✏️ cell, because `write_sample`/`stored_label` are defined in the helper cells that follow the ✏️ cell. |
| 6 | DONE | `03_quantitative_inspection.py` O variant renamed "🟢 Class Imbalance: Is the Expensive Class a Small One?" and its first question refers back to "Your Task: Which Mistake Is Expensive?"; the "How to read the ratio" cross-reference in `09` follows the new title. The question is now asked in Meet Your Data and in the guiding questions only. |
| 7 | DONE | `01`: "How Each Format Stores Its Labels" at `###`, the long cell split (intro, then **Classification** in its own cell; boxes and masks were already separate). `07`: O Object Detection opener cut to two short paragraphs. |
| 8 | DONE | `07`: "Visualizing a Missing Annotation" markdown and its "🔎 Visualize a Missing Bounding Box" cell tagged Teaching. |
| 9 | DONE | `own_wording.tsv`: three swaps, "🔎 Explore the Number of Animals per Image" → Objects, "🔎 Explore Spatial Distributions of Animal Pixels" → Object Pixels, and the mask-check figure label "Wrong animal" → "Wrong object" (the only other visible "animal" outside *In the park* notes and park examples). |
| 10 | DONE | `01`: the ✏️ HTML box is plain markdown in five short cells (lead "**✏️ Your Turn: Point the Notebook at Your Dataset**", options A, B, C + D, last step). `09`: the four 💡 hint lines are bold leads; the two `print("💡 …")` lines lost the icon. No 💡 left in O except in `callout()`'s docstring. |
| 11 | DONE | "Helpers: Read Your Labels (Six Formats, One Table)"; "### After the Lab · Read Your Results and Answer the Guiding Questions"; "#### 🟢 The Guiding Questions" with the dot removed from the body; "🔎 Suggest a Split: Copies Grouped, Stratified by Class"; "🔎 Show the Report Card for Your Dataset"; "center" in `read_yolo`'s docstring. "Not In This List?" kept. |
| 12 | DONE | `04`: `score_hunt` moved from the shared "Helpers: Compare Duplicate Detection Methods" into the Teaching-only "🔎 Compare Duplicate Detection Methods" cell. |
| 13 | DONE | `07`: "### From Labels to Bounding Boxes" restored in both variants, exactly as in her copy (cell 126). |
| 14 | DONE | T6: "Run the next four cells, in order … The third cell downloads the Play Dataset" (Setup, Helpers: Data Download, Load the manifest, Helpers: Images and Masks). |
| 15 | DONE | `09`: the heading cell has a T and an O variant, each with one opener sentence; box and cell title "📘 What Makes a Good Dataset"; Kaggle box title "🔍 Go Deeper · This Is Ordinary: Two Kaggle Stories" (`04`). The Kaggle cell title "🔍 Go Deeper: Two Kaggle Stories" was left as it was: the row names the box title only. |
| 16 | DONE | `04` T86: "The next figure makes five kinds of copy of one image and shows how many bits each copy's pHash differs from the original's, with standard pHash and with transformation-aware pHash." |
| 17 | DONE | `scripts/build.py`: `prerender_html()` saves the HTML of a cell that only imports and shows one `HTML(...)` of a string literal (Learning Objectives, both How to Use boxes). The O ✏️ loader box no longer needs it (row 10). Checked by `test_boxes_show_before_anything_runs` (row 24). |
| 18 | DONE | Burst, mask and fill-ratio browsers: "← Previous" / "Next →" (no chevron icons; 11 of 11 browsers now alike). Checkboxes: "Show bounding boxes" everywhere (split view was "Bounding boxes", mask and fill browsers "Show boxes"), "Show segmentation masks" everywhere (was "Show masks" twice). "Bit distance" on the Bit Distance Distribution x-axis ("Bit distance to the closest other image") and in the pHash figure note ("Bit distance = number of differing bits"); the 📘 box keeps its one definition "Hamming distance (bit distance)". |
| 19 | DONE, two differently | `fits()`: `W, H` → `width, height`, **not** `w, h`: `fits()` already binds `x, y, w, h` from the box, so `w, h` would overwrite the box size. `summarize_folds(..., features, y, ...)`. Toy cell: `TOY_H, TOY_W` (all uses, incl. the axis limits). `i, j` → `first, second` (similar-pairs ranking); `s` → `class_stats`; both `x` → `bar_x`. `own` → `class_boxes` (heatmaps), `image_masks` (pixel shares, `show_masks`), `tight_box` (toy cell), `image_boxes` (`show_boxes`); the same word in O's `write_sample` → `image_anns` (a fourth meaning the row did not list). `_fmt` (string) → `_sample_format`. `PARK_SAMPLE_PER_CLASS = 4 if SMOKE else 20` in "Helpers: Park Sample…". `SMOKE_IMAGES_PER_CLASS` is unused in O, so it moved from the shared download helper into the Teaching-only load cell instead of being deleted (T still uses it). |
| 20 | DONE | `is_readable` deleted. `pandas as pd`, `Path`, `Counter` imported in Setup (removed from the download helper); the comment now says the common imports are made there and a few heavy or chapter-specific ones where first used. |
| 21 | DONE | One scheme `<prefix>_prev_btn`, `_next_btn`, `_show_page`, `_navigate`, plus `_counter` / `_output`: `gallery_*` (class browser), `ann_show_page`, `dup_*`, `atypical_*` (color histograms, was `_prev`, `_next`, `_out`, `_show_class`, `_navigate`), `sim_show_page`/`sim_output`, `burst_show_page` + one `burst_navigate(step)` (was two page functions), `mcq_check_btn`, `mcq_solution_btn`, `mcq_check_answer`, `mcq_show_solution`, `boxview_*`, `size_*`, `crowd_*_btn` + `crowd_navigate(step)`, `overlap_x_slider`/`overlap_y_slider`, `quality_*` (prefix first), `maskview_counter`, `fill_counter`. Widget drive: see below. |
| 22 | DONE | `by_images`, `image_share`, `image_rle`. Her task title "Boxes per Class vs. Photos per Class" unchanged. |
| 23 | DONE | `tests/test_her_rules.py`: `test_capitals_never_changed_in_place` (mutating methods, `X[k] =`, `X.attr =`, augmented assignment) and `test_no_capitals_inside_functions_or_as_parameters`, both Editions. Seen failing first: `TASKS.append` (O); `X` parameter and `W`/`H` inside `fits()` (T and O). |
| 24 | DONE | `test_own_edition_has_no_coding_exercises` (no `✍️ YOUR CODE`, no "Coding Exercise", no `_ref_` in O), `test_learner_settings_are_read_only_in_their_own_section` (settings = `# @param` lines and ✏️ cells, checked in both Editions; asserts HASH_THRESHOLD belongs to section 4), `test_boxes_show_before_anything_runs`. Seen failing first: O's coding cells, O cell 189 reading HASH_THRESHOLD, the Learning Objectives and How to Use cells without output. |
| 25 | DONE | `SPEC.md`: §2 rewritten (S19 audience, form fields allowed with N-058, two ✍️ exercises in T and none in O, no Challenge); L-001 = her four `callout()` kinds + the plain-markdown Task + what the build pre-renders; L-003 = her front matter + theme; §6 statuses 1 and 9 `done`. Two lines the rows made false were brought in line too: L-002 ("under its ✏️ Your turn text (plain markdown)") and §7's "No Colab sliders" (struck, superseded by her copy). |
| 26 | DONE | `bash …/sync_rules.sh .`: RULES.md re-synced (87 rules); N-058 no longer has the RESULTS exception. |

## Gates and runs

- **Gate, after each section** (`just build`, `just lint`, `just smoke`): green three times. Final counts: `just lint`
  = lint 0 problems, ruff clean, **30 tests passed** (`test_version`, `test_editions`, `test_her_rules`; 21 before,
  9 new); `just smoke` = **13 passed** (Teaching once; Own-Data on 7 formats, a broken YOLO file and 4 real layouts,
  the new one being `git-clone`). One smoke failure on the way (a leftover `mcq_check` in the quiz's `display`
  after the row-21 rename) was fixed before the §2 commit; a ruff F821/F811 pass over each Edition's whole code
  (all code cells as one file) found no undefined or redefined name after the renames.
- **Full-data runs** (`reviews/2026-10-28-susanne-copy/handback/run_full.py`, jupyter_client, downloads cached):
  Teaching 130/130 code cells, **0 errors**, 633 images, 33 s of cell time; Own-Data with `DATASET_PATH = None`,
  `LABEL_FORMAT = "coco"`, `TASKS = ["classification", "detection", "segmentation"]`: 89/89 code cells,
  **0 errors**, 120 park images, all three tasks run, no cell skipped, 17 s. Outputs in `build/fixlist/full-*.ipynb`.
- **Widget drive** (row 21; driver `build/widget_drive.py`, full data): after each widget cell it sets every
  dropdown and toggle option, slider and bounded number to min, max and middle, toggles every checkbox twice, and
  presses every enabled button (Next until it disables, then Previous, others 3 times), once as shown and once
  with every slider at its maximum (so buttons disabled at the default, like the duplicate browser's, are
  pressed too). Errors are caught both when raised and when ipywidgets' Output or button dispatcher only shows them
  (a probe notebook with two planted errors was caught 51 times). **Teaching: 16 widget cells, 289 widgets, 1,056
  actions, 0 errors. Own-Data (park sample as COCO, all three tasks): 12 widget cells, 227 widgets, 733 actions,
  0 errors.** Reports: `build/fixlist/widgets-*.json`.

## Teaching Edition: what changed beyond the rows

Cell sources of `eda-teaching-notebook.ipynb` compared before the first commit (built at `1417922`, saved as
`build/fixlist/baseline-teaching.ipynb`) and after the last (`build/fixlist/teaching_diff.py`, difflib over the cell
list): 199 cells before and after, **32 changed blocks, every one mapped to a row**; no Teaching cell was added,
removed or moved. Outputs changed only on cells 2 and 4.

| Cells | What changed | Rows |
|---|---|---|
| 6–9 | T6 "four cells … the third downloads"; Setup imports `pd`, `Path`, `Counter` and its comment; the download helper loses those imports and `SMOKE_IMAGES_PER_CLASS`, which moves into the load cell | 14, 20, 19 |
| 2, 4 | Learning Objectives and How to Use: source unchanged, HTML now saved as output | 17 |
| 22 | class browser: `gallery_*` names | 21 |
| 29 | `ann_show_image` → `ann_show_page` | 21 |
| 44 | `s` → `class_stats` | 19 |
| 69 | Kaggle box title | 15 |
| 72–73 | `is_readable` deleted; duplicate browser `dup_prev_btn`, `dup_next_btn`, `dup_show_page` | 20, 21 |
| 76 | x-axis "Bit distance to the closest other image" | 18 |
| 79 | pHash figure note "Bit distance = number of differing bits" | 18 |
| 83–84 | `score_hunt` moved from the shared helper into the Teaching-only comparison cell | 12 |
| 86 | the plain sentence before the copies figure | 16 |
| 90 | color-histogram browser `atypical_*` | 21 |
| 99 | `i, j` → `first, second`; `sim_show_page`, `sim_output` | 19, 21 |
| 109 | `x` → `bar_x` | 19 |
| 117 | `summarize_folds(…, features, …)` | 19 |
| 122 | burst browser "← Previous" / "Next →", `burst_show_page`, `burst_navigate` | 18, 21 |
| 124 | split view checkbox "Show bounding boxes" | 18 |
| 127 | quiz `mcq_check_btn`, `mcq_solution_btn`, `mcq_check_answer`, `mcq_show_solution` | 21 |
| 129 | "### From Labels to Bounding Boxes" | 13 |
| 131–132 | `show_boxes` local `image_boxes`; box browser `boxview_*` | 19, 21 |
| 137 | `by_images` | 22 |
| 141 | size browser `size_*` | 21 |
| 144 | heatmaps `class_boxes` | 19 |
| 148 | crowding browser `crowd_*_btn`, `crowd_navigate` | 21 |
| 152 | `overlap_x_slider`, `overlap_y_slider` | 21 |
| 158 | quality browser `quality_*` | 21 |
| 170–171 | `show_masks` local `image_masks`; mask browser labels and `maskview_counter` | 19, 18, 21 |
| 175 | `image_rle`, loop `image_masks`, `bar_x` | 22, 19 |
| 178–179 | `image_share`; fill-ratio browser labels and `fill_counter` | 22, 18, 21 |
| 183 | `image_rle` in the heatmap cell | 22 |
| 187 | `fits()`: `width, height` | 19 |
| 191 | toy cell `TOY_H`, `TOY_W`, `tight_box` | 19 |
| 195–196 | "## Your Project Dataset" opener; "📘 What Makes a Good Dataset" (box and cell title) | 15 |

Visible text in Teaching changed only in the rows that say so: 13, 14, 15, 16 (markdown and box titles), 18
(button and checkbox labels, an axis label, a figure note) and 17 (boxes now visible before running). Everything else
is code inside collapsed cells.

## Not done here

Nothing was pushed or merged. `versions/` was not touched, so `just version-check` will report that the newest
version differs from SPEC.md and the notebooks until `just version "…"` is run before the merge.
