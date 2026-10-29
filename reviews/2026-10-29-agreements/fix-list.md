# Fix list from the three checks (2026-10-29)

Sources: `agreements-check.md` (A), `rules-check.md` (R), `code-sweep.md` (C) in this folder. Every row is decided;
apply all of them. T = Teaching Edition, O = Own-Data Edition. Her visible text changes only where a row says so.

## 1. Own-Data Edition content
| # | Fix | From |
|---|---|---|
| 1 | **No coding exercises in O** (her request: "without the exercises"): the "Optional Coding Exercises" block (heading, explanation, `double` reference + ✍️ cell) and the `count_boxes_per_class` exercise (its ✏️ text, "Coding Exercise" lines, reference cell, ✍️ cell) become Teaching-only. In O, whatever later cells need from them (e.g. `count_boxes_per_class` / `boxes_per_class`) comes from one hidden O-only cell that uses the reference implementation. The O variant of that task keeps its questions, without the coding part. | A#49 |
| 2 | Split suggestion gets its **own setting** in its own cell (e.g. `SPLIT_COPY_THRESHOLD = 8`, with a one-line comment), not section 4's `HASH_THRESHOLD`; its hint text says to change it in that cell. | A#38, R N-058 |
| 3 | `TASKS` is never mutated: derive a lowercase `tasks_to_run` (segmentation implies detection); the skip hook and every later reader use `tasks_to_run`. | C, A#37, R |
| 4 | **`git clone` loading option** (Susanne, 26 Oct): `DATASET_PATH` may be a git repository URL (`https://github.com/…`, `https://gitlab…`, or ending in `.git`); the notebook runs `git clone --depth 1` into `data/cache/my_upload/<repo>` once and reads it like a folder. One line in the upload instructions box. Test it with a local bare repo in a smoke case (no network). | A#40 |
| 5 | **A drawn example per format** (promised in the grilling, Q3): one O-only cell in the loader chapter, "🔎 See One Image in Every Format", that writes ONE park image in each format with the existing `write_sample`, draws the image with its box(es) once, and prints that image's label as stored in each format (COCO dict, YOLO line, VOC object, CSV row, folder path; for masks the COCO polygon start, the YOLO-seg line start, the PNG class indices). Runs without the student's data. | A#73 |
| 6 | "Which mistake is expensive" is asked once in Meet Your Data and once in the guiding-questions recap; the O Class Imbalance variant refers back to it instead of asking again. | A#47 |
| 7 | O "How Each Format Stores Its Labels" `####` → `###` (one level below its chapter, SN-011); split that very long cell (O15) into a few cells along its headings (N-036); trim the O Object Detection opener variant to at most two short paragraphs (N-036). | A#62, R |
| 8 | O-only orphan: "Visualizing a Missing Annotation" and its toy cell become Teaching-only (their parent task is Teaching-only). | R |
| 9 | O titles still saying "Animals" (Objects per Image run cell, Explore Spatial Distributions run cell, any other): add swaps to `sections/own_wording.tsv`. | A#47 |
| 10 | O19 "✏️ Your Turn" box → plain markdown (SN-001; not in her box set). 💡 used as a hint icon in O cells (How to read …, Look at …) → plain bold lead, no icon (💡 is her Sample Answer). | R SN-001/015 |
| 11 | Title Case / her title patterns in O-only cells: "Not In This List?" fine; "(six formats, one table)" → "(Six Formats, One Table)"; "After the Lab · Read Your Results and Answer the Guiding Questions"; "The Guiding Questions"; the 🟢 dot moves from the body into a heading line; split-suggestion cell → "🔎 Suggest a Split: Copies Grouped, Stratified by Class"; report card → "🔎 Show the Report Card for Your Dataset". "centre" → "center" in the O22 docstring. | R SN-012/014/016 |
| 12 | Delete `score_hunt` from O if only Teaching cells use it (move it into the Teaching-only cell that does). | A#36 |

## 2. Both Editions
| # | Fix | From |
|---|---|---|
| 13 | Restore her heading "### From Labels to Bounding Boxes" exactly as in her copy (it was demoted to bold). | R |
| 14 | T6 is wrong now: say how many cells to run and which one downloads, matching the current notebook. | R |
| 15 | "## Your Project Dataset" gets a one-sentence opener before its theory box (T and O wording may differ, O via a variant). Theory box title "📘 Theory · What makes a good dataset" → "📘 What Makes a Good Dataset"; its cell title "Checklist Dataset Mini-Project" → "📘 What Makes a Good Dataset". Kaggle box → "🔍 Go Deeper · This Is Ordinary: Two Kaggle Stories". | R SN-011/016 |
| 16 | T86 cryptic sentence ("Why, on one photo and five kinds of copy: …") → one plain sentence saying what the next figure shows. | R N-043 |
| 17 | **Boxes visible before running (N-034):** the build pre-renders the saved output for every cell that only displays one `HTML(r"""…""")` literal (Learning Objectives, How to Use, the O How-to-Use variant, any other), as it does for `callout()` cells. | R |
| 18 | Visible consistency: all browsers use "← Previous" / "Next →"; one label per control ("Show bounding boxes", "Show segmentation masks"); "bit distance" everywhere for the pHash bit count (axis labels too; the 📘 box may define it once as the Hamming distance). | C |
| 19 | Capitals/PEP 8: `W, H` locals → `w, h`; the `X` parameter of `summarize_folds` → `features` (her code, but the rule has no exceptions); single-letter module-level names in 🔎 cells (`H, W = 60, 84` → `TOY_H, TOY_W`; `i, j`, `s`, `x` → descriptive); `own` variable (three meanings) → `class_boxes` / `image_masks` / `tight_box`; `_fmt` string → `_sample_format`; bare `4 if SMOKE else 20` → setting `PARK_SAMPLE_PER_CLASS`; delete the unused `SMOKE_IMAGES_PER_CLASS` if unused in O. | C, R |
| 20 | Dead code: delete `is_readable`. Setup comment that claims every import is made there: move pandas, `Path`, `Counter` into setup and reword the comment to what is true. | C |
| 21 | Widget names: one scheme `<prefix>_prev_btn`, `<prefix>_next_btn`, `<prefix>_show_page`, `<prefix>_navigate` (prefix per browser, as already started); the class gallery, colour-histogram and quiz cells get prefixes too. **After renaming, drive every widget programmatically (as the run audit did: set every dropdown option, slider ends, toggle checkboxes, press every button) in both Editions and report zero errors.** | C |
| 22 | Hidden identifiers `by_photos`, `photo_share`, `photo_rle` → `by_images`, `image_share`, `image_rle`. Her task title "Boxes per Class vs. Photos per Class" stays. | C |

## 3. Tests, SPEC, rules
| # | Fix | From |
|---|---|---|
| 23 | `test_capitals_only_for_settings` also fails on: a capital name mutated (`.append`, `.update`, `.extend`, `[k] =`, `|=`), capital names assigned inside functions, capital function parameters. | A#37 |
| 24 | New tests: O has no `# ✍️ YOUR CODE` cell and no "Coding Exercise" text; no O cell reads a setting defined in another section's ✏️/`@param` cell (at least: `HASH_THRESHOLD` only in section 4); every cell that displays a box has a saved output in the built notebook. | A#49/38, R |
| 25 | SPEC: §2 audience/freedom rewritten to the current notebook (no "~4 ✍️ functions", no Challenge, sliders allowed); L-001 box kinds = her four (SN-015) + the plain-markdown Task; L-003 banner = her front matter; §6 statuses all `done`. | A#20/28 |
| 26 | `RULES.md`: re-sync from the vault (N-058 no longer has the RESULTS exception): `bash /Users/simone.bonato/Desktop/second-brain/.claude/skills/notebook/sync_rules.sh .` | A#85 |
