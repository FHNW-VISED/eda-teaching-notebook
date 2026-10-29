# Rules check: both Editions against SN- and N-rules (2026-10-29)

Branch `fix/results-lowercase`, after `just build` (the build reproduced the committed notebooks exactly; `git status` was clean).
Checked: `eda-teaching-notebook.ipynb` (T, 199 cells) and `eda-own-dataset-notebook.ipynb` (O, 201 cells). Cell numbers are
0-based indices in the built `.ipynb`. "Her copy" = `reviews/2026-10-28-susanne-copy/her-copy.ipynb`. She edited cells
0–190; cells 191 onwards (after her `## Continue here (SUS)` marker) are v05 text she had not reached, so anything there
counts as ours, not her choice.

## Summary

- **Rules checked:** 78 (SN-001…SN-019 without the superseded SN-004 = 18; N-001…N-068 minus 8 process rules = 60).
  Out of scope (notebook-system process): N-015, N-016, N-028, N-030, N-032, N-033, N-047, N-064.
- **Rules with at least one violation:** 11 (SN-001, SN-011, SN-014, SN-015, SN-016, SN-008/N-058, N-025, N-034, N-036, N-043, N-055).
- **Violations:** Teaching Edition **9**, Own-Data Edition **28** (the five 💡 cells counted one by one; the 35 ✍️ answer cells counted once).
- Most of the Own-Data violations are in the cells taken over from the unedited v05 tail (section 9) and in the new
  section 1. The cells she edited are almost clean: where they break a base N-rule, her copy did it on purpose, so they
  are listed as "her choice" and not counted.

## Violations

| # | Rule | Ed. | Cell | Offending text | Fix |
|---|---|---|---|---|---|
| 1 | SN-001 / SN-015 | O | 19 `✏️ Your Turn: Point the Notebook at Your Dataset` | A ✏️ "Your turn" box, `background:#FFF4D6; color:#7A4A00` (the old v05 `your-turn` kind). It is not in her box set (📘 `#E3F2FD`, 🧾 `#FFF8D6`, 💡 `#F5F5F5`, 🔍 `#F3F4F6`). | Make it plain markdown with a bold lead, as Teaching cell 134 does (`**✏️ Your turn · …**`), or a 📘 info box. |
| 2 | SN-015 / SN-001 | T | 197 `🔗 Resources: Image Datasets and Annotation Tools` | The go-deeper grey (`callout("deeper", …)`) with a 🔗 icon. That makes a fifth visual kind; her grey box is 🔍. | `🔍 Go Deeper: Image Datasets and Annotation Tools`, or a plain markdown list. |
| 3 | SN-015 / SN-001 | O | 187 (same cell) | same | same |
| 4 | SN-015 (💡 = Sample Answer) | O | 189 `A Split Suggestion…` | `print("💡 the smallest class has only … test images …")`, `print("💡 a stratified 80/20 hold-out …")` | Drop the 💡 (plain text, or ℹ️ as the loader uses). |
| 5 | SN-015 | O | 190 | `💡 Look at "copy pairs across train/test": it must be 0.` | Bold lead with no 💡: `**Look at "copy pairs across train/test":** …` |
| 6 | SN-015 | O | 194 | `💡 How to read the ratio:` | `**How to read the ratio:**` |
| 7 | SN-015 | O | 195 | `💡 How to read the sizes and modes:` | `**How to read the sizes and modes:**` |
| 8 | SN-015 | O | 196 | `💡 Every copy pair is a decision for a human:` | `**Every copy pair is a decision for a human:**` |
| 9 | N-025 (one icon per kind; SN-001 overrides only the "every part in a Callout" half) | O | 28, 39, 42, 44, 53, 57, 60, 63, 66, 70, 73, 74, 82, 85, 88, 92, 95, 105, 108, 114, 119, 122, 123, 135, 139, 142, 146, 150, 153, 156, 165, 169, 174, 177, 178 | `✍️ **Your answer** (double-click to edit):`. ✍️ already means "write code" in the same Edition (cells 12 and 133: `# ✍️ YOUR CODE`). | Another marker for written answers, e.g. `📝 **Your answer**` or `✏️ **Your answer**`. |
| 10 | SN-016 (Title Case) | T | 69 `🔍 Go Deeper: Two Kaggle Stories` | Box title `🔍 Go Deeper · This is ordinary: two Kaggle stories`. The cell title was retitled, the box title was not. It is a v05 title she left as it was, not one she wrote. | Box title `🔍 Go Deeper: Two Kaggle Stories` (same as the cell title). |
| 11 | SN-016 | O | 77 (same cell) | same | same |
| 12 | SN-016 (+ title consistency) | T | 196 `Checklist Dataset Mini-Project` | Box title `📘 Theory · What makes a good dataset`, not Title Case, with a `Theory ·` prefix that no other 📘 box uses. Every other box cell is titled with its box title. Unedited tail. | Box and cell title `📘 What Makes a Good Dataset?` |
| 13 | SN-016 | O | 186 (same cell) | same | same |
| 14 | SN-016 | O | 18 `🔍 Go Deeper: Not In This List?` | `In` capitalized (a short preposition) | `🔍 Go Deeper: Not in This List?` (cell and box title) |
| 15 | SN-016 | O | 19 | Box title `✏️ Your turn · Point the notebook at your dataset` (sentence case; the cell title is Title Case) | Covered by fix 1. If the box stays, use `Point the Notebook at Your Dataset`. |
| 16 | SN-016 | O | 22 | `Helpers: Read Your Labels (six formats, one table)` | `Helpers: Read Your Labels (Six Formats, One Table)` |
| 17 | SN-016 | O | 193 | `### After the lab · Read your results and answer the guiding questions` | `### After the Lab: Read Your Results and Answer the Guiding Questions` |
| 18 | SN-016 | O | 198 | `#### The guiding questions` | `#### 🟢 The Guiding Questions` (see 24) |
| 19 | SN-016 (US spelling) | O | 22 (hidden helper docstring) | `(centre x, centre y, width, height)` | `center x, center y` |
| 20 | SN-014 | O | 189 | `# @title A Split Suggestion: Copies Grouped, Stratified by Class`. It shows a result table but has no 🔎 and does not start with a verb. | `🔎 Suggest a Split: Copies Grouped, Stratified by Class` |
| 21 | SN-014 (+ SN-016) | O | 192 | `# @title The report card for this dataset`. It shows output, has no 🔎, and is not Title Case. | `🔎 Show the Report Card for This Dataset` |
| 22 | SN-011 | T | 195 | `## Your Project Dataset` is a heading only. The chapter goes straight into a 📘 box, with no motivating question or paragraph. | Add one or two sentences, e.g. "Your project needs a dataset too. What makes one good enough to train on?" |
| 23 | SN-011 | O | 185 | same | same |
| 24 | N-055 / SN-012 | O | 198 | The level dot sits in the body text (`🟢 Double-click the next cell and answer …`), and the heading has none. It is the old `**Q26.** 🟢` with the number cut off. | Move the dot into the heading (fix 18) and drop it from the body. |
| 25 | SN-008 / N-058 | T | 187 `Helpers: Simulate Mask Annotation Errors` | Inside a function: `W, H = int(size.at[a["image_id"], "width"]), int(size.at[a["image_id"], "height"])`. Capitals computed from data. It is in her copy, but her capitals edit renamed results, not these, and Simone's 29 Oct rule has no exception. | `w, h = …` |
| 26 | SN-008 / N-058 | T | 117 `🔎 Compare Cross-Validation Fold Counts` | `def summarize_folds(name, splitter, X, y, groups=None)`: an uppercase parameter (sklearn's `X`). The code is hers. Flagged only because of the no-exception rule. | `ids` (or `x`) |
| 27 | SN-008 / N-058 | O | 121 (same cell) | same | same |
| 28 | SN-008 / N-058 | O | 24 `Load Your Dataset (or the Park Sample)` | `TASKS.append("detection")  # every mask also gives its box`. A learner setting is changed in place. | Keep `TASKS` as typed. Derive a lowercase `tasks = TASKS + ["detection"] if …` and read that downstream (the hook in cell 23, cells 24 and 25). |
| 29 | SN-008 / N-058 | O | 189 (section 9) | `find_copies(…, HASH_THRESHOLD, use_variants=True)`: the slider from Bit Distance Distribution (section 4) is read in Your Project Dataset. Cell 196 also tells the learner to change it there and rerun. | Give section 9 its own setting (e.g. `SPLIT_COPY_BITS = 8`, described above the cell), and point cell 196 at it. |
| 30 | N-034 | T | 2 `Learning Objectives` | An `HTML(r"""<div class="callout callout-banner" …>""")` cell with **no saved output** in the built `.ipynb`. In Colab it shows as an empty collapsed "Learning Objectives" until it is run. The build saves output only for `callout(…)` cells. | Save its HTML output at build time, or rewrite it as a `callout`. |
| 31 | N-034 | T | 4 `How to Use This Notebook` | same: no saved output | same |
| 32 | N-034 | O | 2 | same | same |
| 33 | N-034 | O | 4 | same (the Own-Data wording) | same |
| 34 | N-034 | O | 19 `✏️ Your Turn …` | same: the instructions for loading the dataset are invisible until the cell runs | same (and fix 1) |
| 35 | N-043 | T | 86 | `Why, on one photo and five kinds of copy: the pHash distance to the original, plain vs variant.` This is v05 shorthand ("plain vs variant") kept in her copy, not her wording. | O cell 90 already has the clear version: "The next cell takes one photo, makes five kinds of copy, and compares the pHash distance to the original, plain vs. transformation-aware." |
| 36 | N-036 | O | 15 `#### How Each Format Stores Its Labels` | One cell holds 4 paragraphs, 3 lists and 2 code blocks (131 words plus code). | Split it: the intro paragraphs in one cell, then one cell per task, as cells 16 and 17 already do. |
| 37 | N-036 | O | 125 `## Object Detection` | 4 paragraphs. The added "This chapter runs only if `TASKS` includes `"detection"` …" makes her 3-paragraph opener 4. | Fold the TASKS sentence into her first paragraph, or move it into the chapter's first Task. |

(37 rows = 37 violations: 9 Teaching + 28 Own-Data. Row 9 is one pattern over 35 cells and counts once.)

## Other findings (no ledger rule, but wrong for a student)

- **T 6:** `> ▶️ **Run the next three cells, in order.** … The last cell downloads the Play Dataset`. Her copy had three code
  cells there. The build now has four (7 Setup, 8 Helpers: Data Download, 9 Load + download, 10 Helpers: Images and
  Masks), so "three" is wrong and the download is the third of four, not the last. Fix: "Run the next four cells, in
  order … The third cell downloads …"
- **O 157:** `#### Visualizing a Missing Annotation` belongs to the Teaching-only Task "Missing Annotations: Find the
  Missing Box". In the Own-Data Edition it hangs under "Bounding Box Quality", after that Task's answer cell, and still
  says "animals" where the Edition says "objects". Either tag it Teaching-only or give it a parent there.
- **O 15:** a `####` heading straight under `## Load Your Dataset` (no `###` level). It is not a Task, so SN-011 does not
  cover it, but it is the Edition's only level skip.
- **T 129:** her `### From Labels to Bounding Boxes` heading was demoted to a bold line. This is probably right for
  SN-011 (Tasks one level under the chapter), but it changes her edit. Mention it to her rather than leave it silent.
- **O 200:** a `**Takeaway**` block ("When you … → do …"). It is plain markdown now, so it is not a takeaway *box*
  (SN-015), but the Open item "takeaway keep or drop: her copy decides" is unresolved. It comes from the v05 tail she
  had not reached.

## Per rule: how checked / result

### Her overlay (SN)

- **SN-001** few boxes. Script: every `callout(kind…)` plus every inline `background:` in sources and outputs. Result: 3 violations (O19 ✏️ box, the 🔗 box ×2).
- **SN-002** one Task style, level legend. Script: grep for the legend. The legend reads "Activities are marked 🟢 core · 🔴 optional", her own wording, and no 🟡 appears. Her choice. Pass.
- **SN-003** objectives as markdown, not boxes. Script: grep "Objective", the objective colour `#B9E3EA`. None. Pass.
- **SN-005** Bloom verbs. Read cell 2: explore/analyze/assess, identify/assess, select/justify (her text). Pass.
- **SN-006** photo screening. Read `scripts/prepare_data.py`: `COCO_EXCLUDE` lists 5 ids with reasons, removed after sampling (`keep -= set(COCO_EXCLUDE)`). Script: none of them is in `data/coco_animals_subset.json`. Pass.
- **SN-007** occasional MCQ. Read: one clickable MCQ (T127). Pass.
- **SN-008** capitals PEP 8 (with N-058 and Simone's 29 Oct rule). AST over every code cell: uppercase targets, scope, reassignment, in-place mutation, uppercase parameters, cross-section reads of learner settings. No `RESULTS`, and `results` is lowercase. Result: 5 violations (rows 25–29). `CLASSES`/`CLASS_COLORS` reassigned in O24 are allowed by Simone. `H, W = 60, 84` in T191 is a fixed toy size, allowed.
- **SN-009** lost and found. Her final copy has none. Pass, nothing expected.
- **SN-010** her removals stay removed. Script: no Q-numbers, no objective boxes, no Challenge. The Wrap-up appears only as the O200 Takeaway (see other findings). Pass.
- **SN-011** chapters are themes, Tasks one level below their parent. Script: parent-level check of every dotted heading, plus the first cell of each `##`. All Tasks sit one level below their parent. Result: 2 violations (`## Your Project Dataset` has no opener, T and O).
- **SN-012** dot + Title Case, no number. Script: headings regex, `\bQ\d+\b` scan (0 hits). Result: 1 (O198, row 24).
- **SN-013** predict → explore → interpret. Script: bold labels per Task, compared Task by Task with her copy. Every Teaching Task matches or extends her labels. The new Own-Data Tasks use her "Question:" form (her T26 pattern) or Observe and interpret. Pass.
- **SN-014** cell titles. Script: every `@title` classified (🔎 / Helpers / 💡 / box) and compared with her titles. Her own titles that break the pattern (Setup (run me first), Load the manifest…, Measure…, Precompute…, 🔎 Understanding…, 🔎 How Do…?, 🔴 Splitting…, Reference solution…) are her choice. Result: 2 (O189, O192).
- **SN-015** box set and colours. Script: backgrounds of all box HTML. The theory, recipe, answer and deeper colours are exact. Learning Objectives `#E5E7EB` and How to Use `#F5F5F5` are hers (her choice). Result: 7 (O19, 🔗 ×2, 💡 used outside answers ×5, counted with SN-001 where both apply).
- **SN-016** US spelling, Title Case. Script: British-spelling regex over markdown, box text and code strings and comments (1 real hit: "centre"), and a Title Case checker on headings, `@title`s and box titles. Result: 10 (rows 10–19).
- **SN-017** browse, don't grid. Read: class picker plus Previous/Next with "browse at least twelve images" (T21/22), box-size, crowding, quality and mask browsers. Fixed grids that remain (T20, T47, T60, T180) are in her copy. Pass.
- **SN-018** front matter. Read cells 0–4 of both Editions: AI-assistance note, exam block, How to Use, "Simone Bonato & Susanne Suter, FHNW". Pass.
- **SN-019** GPU vs CPU. Read T95/96/97: `COMPUTE_EMBEDDINGS = False`, the DINOv2 model runs on GPU only unless ticked, with the CPU estimate printed. Pass.

### Base ledger (N), in scope

- **N-001** one storyline, 🔴 optional. Read. Pass.
- **N-002** outcomes plus closing wrap-up. Her copy removed the Wrap-up (SN-010). Her choice.
- **N-003** `**Objective:**` lines. Removed by her (SN-003/SN-010). Her choice.
- **N-004** task → predict → run → reflect. Covered by SN-013. Pass.
- **N-005** level markers. Script: every Task heading has 🟢 or 🔴; no 🟡 (her calibration). Pass.
- **N-006 / N-007** Q-numbers. Overridden by SN-012 (0 Q-numbers). n/a.
- **N-008 / N-035** collapsed run-only cells. Script: every code cell without `@title` is a ✍️/✏️ cell (T13, T136, O12, O20, O133), and all titled cells have `cellView: form`. Leading blank lines before `@title` are in her copy too (Colab's own save). Pass.
- **N-009** hands-on early. Read: setup plus the `double()` demo first. Pass.
- **N-010** visual before theory. Read. Pass.
- **N-011** few metrics. Read: accuracy, recall, IoU, fill ratio, all defined. Pass.
- **N-012** no presupposed result. Read the Before/Observe questions. Sample Answers name numbers, but they are answers, and the "smoke-test subset" caveats are hers. Pass.
- **N-013** reset and seed. Grep: every sampler uses `SEED`/`random_state`. The unseeded calls are the learner's Shuffle buttons. Pass.
- **N-014** one SMOKE flag. Grep: `SMOKE` from `NB_SMOKE`, used in `if SMOKE else`. Pass.
- **N-017** Colab-only text. Grep "locally": only Label Studio "can run locally" (about the tool). Pass.
- **N-018** warn before slow steps. Read T6, T95, T119. Pass.
- **N-019** data handling. Read the download helpers (`.part` file, cache). Pass.
- **N-020** English. Pass.
- **N-021** `%pip`. Script: one `%pip install -q imagehash`, no `!pip`. Pass.
- **N-022** validated settings. Read: bounded sliders and dropdowns; O20's path is checked with a plain ❌ message. Pass.
- **N-023** recipe named and linked at the top. T0/O0. Pass.
- **N-024** answers collapsed. Script: all 💡 boxes are `<details>` (closed); the runtime answer T194 uses the same helper. Pass.
- **N-025** one icon per kind. Script: icon usage. Result: 1 (✍️ double meaning in O, row 9). The 💡 reuse is counted under SN-015.
- **N-026 / N-029 / N-046 / N-053** code-writing cells. Read: plain-markdown ✏️ lead (SN-001 overrides the action Callout), reference cell above each stub, `use(…, reference=…)`, the demo introduced once (T11–13). Pass.
- **N-027** theme, one colour per meaning. Script: `CLASS_COLORS` everywhere; the CCT colours are distinct. T65 uses its own green/red for right/wrong (in her copy). Pass.
- **N-031** one sentence before each run cell. Read: consecutive 🔎 cells (T140→141, T147→148, T161→162) exist in her copy. Her choice.
- **N-034** HTML boxes saved as output. Script: box cells without outputs. Result: 5 (rows 30–34). T194 is a runtime answer and is fine.
- **N-036** no walls of text. Script: paragraph, list and word counts per markdown cell, compared with her copy (T0, T16, T18, T129, T177 are hers). Result: 2 (O15, O125).
- **N-037 / N-038 / N-039** show before asking, park examples, statistics defined with min/median/max. Read T35, T39, T40. Pass.
- **N-040** violins. Script: a `boxplot` in T147 "Object counts by class" is in her copy. Her choice.
- **N-041** example shapes under a shape question. T43 tiny scatters. Pass.
- **N-042** visuals for mechanisms. pHash (T79), IoU (T152), masks (T189/191). Pass.
- **N-043** no cryptic sentences. Read. Result: 1 (T86).
- **N-044** few kinds, no colour that looks like a level dot. Her yellow 🧾 is her choice (SN-015). The O19 yellow ✏️ box is counted under SN-001.
- **N-045** no licence text. Grep: only data-column names in code. Pass.
- **N-048** depth secondary. Read: Go Deeper in answers and 🔍 boxes. Pass.
- **N-049** playful simulation. The "lazy model stamps sheep" strip (T65). Pass.
- **N-050** ✍️ docstring example. Script: T13 `e.g. double(21) → 42`, T136 `e.g. {"cat": 12, …}`. Pass.
- **N-051** "N more not shown". Grep: only the CV diagram labels it (T114). T20's six-per-class grid and T193's `head(12)` table are in her copy, and the text says "six example images". Her choice.
- **N-052** order. Read: no reference to a later numbered item. "below"/"Picture A/B" forward references are her style (T164, T185). Pass. (T6's wrong cell count is under Other findings.)
- **N-054** callout font. Script: box `font-size:14px`, the banner title 22px, `0.92em` only on inline code. Pass.
- **N-055** dot in heading. Result: 1 (O198).
- **N-056** generated section numbers. Script: no numbers in headings or in "section N" text. Pass.
- **N-057** browse. Same as SN-017. Pass.
- **N-058** settings. See SN-008. 5 violations.
- **N-059** legends and colour swatches. Script: colour words in text without a swatch: T113, T164, T188, T35, T51 (and the O twins). All are her sentences (we changed only her "grey" to "gray" and her "magenta" to "yellow" to match the figure). Her choice. Figure legends checked by reading the plotting code of the multi-colour plots. Pass.
- **N-060** settings described. Read: COMPUTE_EMBEDDINGS (T95) and DATASET_PATH/TASKS/LABEL_FORMAT (O19) are described in full. The sliders in her Tasks are described in her words. Pass.
- **N-061 / N-062 / N-063** panel roles, numbers in words, before/after pictures. Read the figure titles in the plot code. Pass, within her design.
- **N-065** schematics. pHash and IoU drawings. Pass.
- **N-066 / N-067** SOTA reference and tuning pass. n/a: no model is trained (L-006).
- **N-068** result cards. Her copy displays raw tables and prints (T84, T117, T140), so a raw `display(DataFrame)` is the notebook's convention. Her choice. The new O189 and O192 follow it.
