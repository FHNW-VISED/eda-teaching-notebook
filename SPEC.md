# SPEC — Know Your Data: Exploring and Preparing Image Datasets at the Wildwood Farm Park

The single source of truth for **what** this notebook is. `AGENTS.md` says how to work,
`RULES.md` says what every notebook must respect, and this file says what *this* one must do.
Written 2026-10-19 from Simone's grilling (decisions G-1…G-33 on the vault page
`wiki/work/projects/EDA Teaching Notebook.md`). Agents never change it, except the `Status` column of §6.
Items marked **[PROPOSAL]** come from research into exemplary notebooks (G-31). Build them, and Simone
may strike them in review.

## 1. Purpose
An in-class lab in Susanne Suter's BSc deep-learning / image-processing course, run **after her
lecture on image datasets** (G-1). Students learn to explore and prepare an image dataset *before*
any training, following the first part of Karpathy's *A Recipe for Training Neural Networks*. They
work on a Play Dataset and then reuse the notebook as a checklist for their own project dataset.
The single question it answers: **"Is this dataset fit for my task, and what must I fix before training?"**

## 2. Audience and level
BSc Data Science / Computer Science students. They know Python and basic ML vocabulary, and have
seen Susanne's lecture on image datasets. No training happens here; that comes the following week.
Default path 🟢/🟡, with optional 🔴 parts.
**Freedom (N-028): low to medium** (G-14, G-21):
- **Most cells:** run, then read one sentence on what to look at.
- **✏️ change a value:** at most one per section, plain variables, no Colab sliders (G-12).
- **✍️ write a small function:** ~4 in the notebook, each with a given signature and run through `use(...)` (N-029), so they never block "Run all".
- **The Challenge** is the most open part, and it never blocks either.

## 2b. Prerequisites
A Google account (Colab) and a browser. Python basics. Susanne's lecture on image datasets.

## 3. Learning outcomes
These appear verbatim in the title banner of `00_setup.py`.
1. Inspect an image dataset qualitatively and quantitatively, and say whether it fits a task.
2. Find and fix common dataset errors (duplicates, flipped/rotated copies, broken files, bad labels or masks) before they leak into training.
3. Choose a split and a metric that match the use case, and explain how imbalance, object size and leakage look different for classification, detection and segmentation.

## 4. Use case and data

### The story (G-16, G-20)
**The Wildwood Farm Park** (fictional) has cameras in four places:
- the **farm paddock**: sheep, horses;
- the **aviary pond**: birds;
- the **savanna enclosure**: zebras;
- the **visitor yard**: cats and dogs wander through.

It wants three things from them, one per task:
1. **Recognise** (classification): "who is in front of this camera?" The case that matters is **a dog in the sheep paddock**, which triggers an alert. Missing that dog is worse than a false alarm → **recall on dog**.
2. **Count** (detection): a daily headcount of sheep, zebras and birds → **counting error**. Far-away birds are tiny boxes.
3. **Outline** (segmentation): how much of each enclosure each animal uses → **IoU per class**. Birds are ~6 % of the pixels.

Later the park adds **trail cameras in its woodland reserve** (the CCT20 slice, section 6 only).
**Pet-food-shop contrast** (one paragraph in section 7): same classes, but a shop webcam recommends
food for customers' pets. Horses and zebras never appear, a false "cat" is cheap, and precision on cat vs dog matters.

### Play Dataset: COCO 2017 animals (G-15)
- **val2017**, classes **cat, dog, bird, horse, sheep, zebra**; images with **exactly one animal class**.
- Measured on the full set: 665 images. Images per class: cat 168, dog 153, horse 116, bird 103, zebra 72, sheep 53. Instances: bird 335, sheep 310, horse 248, zebra 239, dog 187, cat 185. Pixel share: bird ~6 %. 233 images have ≥ 2 animals.
- **Cap at ~600 images** (cat and dog capped at ~130), so COCO + CCT20 ≤ 1,000.
- **Licence:** the annotations are CC BY 4.0. The images keep their Flickr licences (34 % are ND), so **we host only the annotation subset** (`data/coco_animals_subset.json`, ≤ 2 MB, keeping each image's `license` and `flickr_url`). The notebook downloads the images from `images.cocodataset.org` at runtime.
- ⛔ All planted copies and defects are generated **inside the notebook at runtime**, never redistributed.
- Known real problems we can point to: about 2.4 % of val masks are poor, and there are duplicate mask pairs (Singh et al., COCO-ReM, ECCV 2024). Also crowd regions, and people in animal photos.

### Section 6 only: Caltech Camera Traps CCT20 slice (G-19)
- **Classes:** opossum, raccoon, coyote, bobcat, rabbit. **Locations:** 46, 43, 38, 33, 100, 115, 120, 88. Up to 4 **whole sequences** per class-location pair → ~400 images, 142 sequences.
- `data/cct20_slice.csv` (`file_name, url, label, location, seq_id, frame_num, datetime`) is committed. Images are fetched from LILA's public HTTPS bucket at runtime and resized to 512 px. Licence: CDLA-Permissive.
- Marco Willi's HF classroom subset (`marco-willi/camera-trap-cct20`) is an FHNW colleague's and can be built on (G-17). But it has no location or sequence columns, so the slice is built from the original CCT annotations.
- Paper number to cite: error **20.8 % at seen locations vs 41.1 % at unseen ones** (Beery et al., 2018).

### Smoke subset (N-014)
`SMOKE`: 36 COCO images (6 per class) and 30 CCT20 images (bursts from 3 locations). Planted problems are scaled down but always present.

## 5. Environment

| Item | Value |
|---|---|
| Runs on | Google Colab (web), **CPU runtime is enough** (no training) |
| Python | 3.12 |
| Preinstalled in Colab (checked 2026-10-19) | numpy, pandas 2.2.3, matplotlib 3.10.0, pillow 11.3.0, pycocotools 2.0.11, opencv 5.0.0.93, scikit-image 0.25.2, scikit-learn 1.6.1, seaborn 0.13.2 |
| `%pip install` needed | `imagehash` only |
| 🔴 optional model-based search | precomputed embeddings downloaded in one cell **[PROPOSAL]**, so no model runs on CPU |
| Data download | ~115 MB COCO images + CCT20 slice, cached in `data/` |
| Smoke run | `NB_SMOKE=1`, CPU, budget 900 s |
| Dev access in Colab | GitHub token in Colab Secrets, or the repo temporarily public (G-29) |

## 6. Sections
Time = participant minutes (~110 total; Susanne may cut sections 1 and the 🔴 parts first). Status: todo · in-progress · done · blocked.
🧾 = opens with a Karpathy Recipe Callout (2–3 verbatim sentences + link). ✏️ = change a value. ✍️ = write a function via `use()`.

| # | File | Title | Objective (one line) | Level | Time | Status |
|---|---|---|---|---|---|---|
| 0 | sections/00_setup.py | Setup & the park | Run setup, meet the park, see the three outcomes | 🟢 | 5 | done |
| 1 | sections/01_image_datasets.py | Image datasets | Know where datasets come from and what makes a good one | 🟢 | 5 | done |
| 2 | sections/02_become_one.py | Become one with the data 🧾 | Look at many images before computing anything | 🟢 | 8 | done |
| 3 | sections/03_image_inspection.py | Image inspection 🧾 | Measure sizes, aspect ratios, colour modes and colour statistics | 🟢🟡 | 8 | done |
| 4 | sections/04_labels_distribution.py | Labels & class distribution 🧾 | Count classes and see why imbalance matters for the dog alert | 🟢🟡 | 8 | done |
| 5 | sections/05_errors_cleaning.py | Common dataset errors & cleaning 🧾 | Find duplicates, flipped/rotated copies, broken files and suspicious labels; produce a cleaned dataset | 🟡 | 15 | todo |
| 6 | sections/06_splits_leakage.py | Splits & leakage | Choose a split that does not leak, and understand cross-validation | 🟡 | 12 | todo |
| 7 | sections/07_metrics.py | Metrics for the park | Pick the metric that matches the use case | 🟡 | 7 | todo |
| 8 | sections/08_boxes.py | Same data, boxes | See what imbalance, size, position and label quality mean for detection | 🟡 | 10 | todo |
| 9 | sections/09_masks.py | Same data, masks | See what they mean for segmentation; find ugly masks | 🟡 | 10 | todo |
| 10 | sections/10_challenge.py | Data-centric AI in practice: the Challenge | Find the planted problems in a new batch with the tools learned | 🟡🔴 | 15 | todo |
| 11 | sections/11_your_dataset.py | Your project dataset | Run the checklist on your own folder-per-class dataset | 🟢 | 5 | todo |
| 12 | sections/12_wrap_up.py | Wrap-up & next week | Takeaways and where to go next | 🟢 | 2 | todo |

### Section contents
- **00 Setup & the park**
  - A **title banner** (Theme, §8 L-003) with the title, the three outcomes and "Questions are marked 🟢 beginner · 🟡 intermediate · 🔴 advanced (optional)".
  - The park story as a short illustrated paragraph.
  - Setup cell (`# @title`): IN_COLAB, SMOKE, SEED, `%pip install -q imagehash` in Colab, the matplotlib Theme style, **`CLASS_COLORS`** (one colour per class, used everywhere: N-027), `RESULTS = {}` (§9), `use()` (N-029), and the data download with caching.
  - The data is loaded into a tidy table: `images` (id, file, width, height, class, n_animals, license) and `annotations` (image_id, class, bbox, area, segmentation).
- **01 Image datasets**
  - Susanne's dataset sources and labelling-tool links (vault page §2b).
  - **Her "what makes a good dataset" checklist, in English**, as a 📘 Theory Callout.
  - Then apply each bullet to the Play Dataset in one line each (e.g. "2–10 classes: ✅ 6").
- **02 Become one with the data**
  - 🧾 Karpathy: "become one with the data".
  - An image grid per class (✏️ `CLASS_TO_SHOW`, `N_SHOW`).
  - Students write down 3 observations, as a markdown cell to fill in.
  - **[PROPOSAL]** A first 💡 insight sentence after every plot (N-004).
- **03 Image inspection**
  - Width/height scatter, aspect-ratio histogram, colour modes (RGB / L / RGBA counts), RGB histograms per class, brightness.
  - **[PROPOSAL]** a `show_extremes(stat)` gallery: the images at both ends of a statistic.
  - **[PROPOSAL]** a detective beat: a cluster of odd sizes or modes points to a different source.
  - 🔀 Across tasks: resizing distorts aspect ratios; box coordinates must be rescaled; masks need nearest-neighbour resizing.
- **04 Labels & class distribution**
  - Images per class (bar, in `CLASS_COLORS`).
  - Images with several animals: "which label then?"
  - Why the dog alert cares about imbalance (⚠️ Pitfall: accuracy hides a rare class).
  - 🔀 Across tasks: one label per image vs one box per animal vs every pixel; what each costs to annotate.
- **05 Common dataset errors & cleaning**
  - **[PROPOSAL]** Opens with a real Kaggle duplicate story: PetFinder (near-duplicates with different scores; train/test duplicates removed) and Cassava (a duplicated pair with two labels).
  - Then:
    - exact duplicates (file hash);
    - near-duplicates (perceptual hash, ✏️ `HASH_THRESHOLD`);
    - **flipped/rotated copies, found by hashing each image's flipped and rotated variants** (**[PROPOSAL]** a shoot-out: plain hashing misses them, variant hashing catches them);
    - broken or odd files (unreadable, grayscale, CMYK, alpha, extreme sizes);
    - suspicious labels (each class's least typical images for a human to judge).
  - **Guided hunt** (G-25): a planted batch with exact duplicates + a flipped + a rotated + a recompressed copy. ✍️ `find_flipped_copies(image, candidates) -> list[str]`.
  - 🔴 model-based near-duplicate and outlier search on precomputed embeddings.
  - Ends with `clean_images`, the cleaned table, and a summary "removed N, kept M".
- **06 Splits & leakage**
  - On COCO: random vs stratified split, with class shares per split.
  - Cross-validation: **k-fold, stratified k-fold, leave-one-out (useful for very small datasets: every image is tested once; costly and noisy), and leave-one-group-out** (G-27).
  - Then the park's woodland trail cameras (CCT20): bursts and locations.
  - **Leakage shown without a model** (G-26): the share of test images with a burst sibling in training, random split vs split by camera, plus the paper's 20.8 % → 41.1 %.
  - ⚠️ Pitfall: **plain leave-one-out leaks across burst frames just like a random split**, and leave-one-*group*-out (one camera) is the fix.
  - Deduplicate *before* splitting (section 5).
  - ✍️ choose and run a split.
  - 🔴 optional nearest-neighbour classifier on colour features to *see* the score gap (seconds on CPU).
- **07 Metrics for the park**
  - On **prepared predictions** (a fixed CSV generated with a seed; no model): accuracy vs recall on dog for the paddock alert, plus a confusion matrix.
  - The pet-food-shop contrast paragraph.
  - ✍️ pick the alert metric: `ALERT_METRIC = ...` with a check.
  - **[PROPOSAL]** a quick multiple-choice check.
  - 🔀 Across tasks: counting error / mAP for detection, IoU / Dice for segmentation.
- **08 Same data, boxes**
  - Boxes per class (vs images per class: the ranking flips, bird goes from 4th to 1st), box-area distribution (small/medium/large), box-centre heatmap, animals per image, overlap and crowding.
  - Guided hunt: **one image with a missing box** (an image with 3 sheep but only 2 boxes).
  - ✍️ `count_boxes_per_class(annotations) -> dict[str, int]`.
  - Each concept gets its one-line "handled later by" (G-18): weighted/focal loss, tiling, multi-scale, iterative stratification.
- **09 Same data, masks**
  - Pixels per class **including background**; mask-area distribution; per-class pixel heatmaps; thin parts.
  - Guided hunt: **5 ugly masks** (G-25):
    - swollen (dilated);
    - shifted ~15 px;
    - a chunk missing;
    - a mask that belongs to another animal;
    - a crude polygon of a few vertices.
  - Students find them with mask–box agreement (IoU of the mask's own bbox with the annotated bbox), fill ratio (mask area / box area), and pieces and holes (✏️ thresholds). 🔴 boundary–edge alignment.
  - Note: COCO has real poor masks too; students may find genuine ones.
- **10 Data-centric AI in practice: the Challenge**
  - A 📘 Theory Callout on data-centric AI (fix the data before the model) introduces it.
  - "A new batch from the park": ~150 COCO images with ~15 planted problems of 8 types (G-22, Q24 of the grilling):
    - exact duplicates;
    - a flipped copy;
    - a rotated copy;
    - a resized + recompressed near-duplicate;
    - wrong labels;
    - one grayscale image and one unreadable file;
    - one train/test overlap;
    - one image with a missing box;
    - plus 2 ugly masks.
  - Students fill `findings = [...]`. `check_findings(findings)` reports "found k of n" per type, and `reveal_answers()` prints the key (also in a `<details>` toggle **[PROPOSAL]**).
  - 🔴 optional ending (G-33): the nearest-neighbour "ruler" score on the dirty batch vs after the student's cleaning, framed as *a ruler, not training*.
- **11 Your project dataset**
  - **One clearly marked swap-in cell** (`MY_DATASET_DIR = None`, folder-per-class).
  - Rerun the classification checklist on it: counts, sizes, modes, duplicates, flips, a split suggestion.
  - The guiding questions, each answered for the park, with space for the student's answer (G-8).
  - In smoke mode, a tiny generated folder-per-class sample is used.
- **12 Wrap-up & next week**
  - The three outcomes restated as takeaways (N-002).
  - Next week: training. Augmentation links (Neptune, Ultralytics) and MLOps tools (W&B, TensorBoard, Neptune, MLflow) from Susanne's chat.

## 6b. Pitfalls and extensions
- Pitfalls to anticipate:
  - forgetting that COCO images can hold several animals (classification label ambiguity);
  - comparing counts across tasks without saying what is counted;
  - splitting before deduplicating;
  - treating LOO as leakage-proof.
- Extensions (🔴): model-based near-duplicates, boundary–edge alignment, the nearest-neighbour ruler in sections 6 and 10.

## 7. Out of scope / limitations
- **No model training** (that's next week). Only the 🔴 nearest-neighbour "ruler" on colour features.
- **No data augmentation as a technique** (G-6). Augmentation appears only as pre-augmented copies to detect, plus links at the end.
- **No Colab sliders or forms for values** (G-12).
- COCO has no camera groups. Any group structure on COCO would be planted, so real groups come only from CCT20.

## 8. Notebook-specific rules

**L-001 — Callouts** (N-025, G-11, G-32):
- One-cell tables in markdown. **Inside them use HTML only** (`<b>`, `<i>`, `<br>`, `<ul><li>`, `<code>`, `<a>`). No markdown inside the table, and no `style=` (Colab strips it).
- Template:
  ```html
  <table width="100%"><tr><td bgcolor="#E3F2FD"><font color="#0D47A1">
  <b>📘 Theory · Data leakage</b><br>
  Text …
  </font></td></tr></table>
  ```
- Kinds, each written as bgcolor / font colour:
  - 📘 **Theory** `#E3F2FD` / `#0D47A1` (the title names the topic);
  - ✏️ **Your turn** `#FFF4D6` / `#7A4A00`;
  - ❓ **Questions** `#E0F7F4` / `#00564D`;
  - 🔀 **Across tasks** `#EDE7F6` / `#311B92`;
  - ⚠️ **Pitfall** `#FDECEA` / `#8B1A10`;
  - 🧾 **Recipe** `#E8F5E9` / `#1B5E20`.
- Source: Simone, G-11 / G-32; Colab rendering research 2026-10-19.

**L-002 — Code students change or write** (N-026, G-12):
- Every ✏️ cell sits directly under a ✏️ Your turn Callout and starts with `# ✏️ YOUR TURN`.
- Every ✍️ function starts with `# ✍️ YOUR CODE`, has its signature and a docstring stating inputs and outputs, and an `...` body, followed by `use(fn, reference=_ref_fn)` (N-029).
- Source: Simone, G-12, G-21, G-24.

**L-003 — Theme green-blue** (N-027, G-9, G-18):
- Title banner: a full-width table, bgcolor `#0B6E79`, white font, with the title, the three outcomes and the level legend.
- Matplotlib style: background white; text and axes `#21434A`; grid light.
- `CLASS_COLORS` used in *every* plot, box and mask overlay: cat `#E07A5F`, dog `#3D405B`, bird `#F2CC8F`, horse `#81B29A`, sheep `#9C89B8`, zebra `#2A9D8F`.
- Source: Simone, G-9, G-18.

**L-004 — Karpathy quotes**: verbatim, **2–3 sentences**, in a 🧾 Recipe Callout with a link to https://karpathy.github.io/2019/04/25/recipe/, only in sections 2, 3, 4 and 5. Source: Simone, G-6, G-18.

**L-005 — Answers** (G-3, N-024 overridden):
- Worked answers on the Play Dataset are **shown openly** right after each question.
- Only the Challenge hides its key (`reveal_answers()` + `<details>`).
- Source: Susanne, "we answer it exemplarily with our play dataset"; Simone, G-3.

**L-006 — No training**: no neural-network training anywhere. The 🔴 nearest-neighbour "ruler" is the only fitted model, and it is labelled as a ruler. Source: Susanne, 2026-10-19; G-26, G-33.

**L-007 — English** throughout, including translations of Susanne's German material (N-020).

## 9. Acceptance (definition of done for the whole notebook)
- `just check` passes, including `tests/test_content.py` (written from G-28; **the night agent never edits it**).
- **Results contract for the tests:**
  - `00_setup.py` creates `RESULTS = {}`.
  - Sections write named results into it (the keys are listed in `tests/test_content.py`).
  - The smoke test dumps `RESULTS` to `build/smoke-results.json` itself (no cell in the notebook).
  - **Progressive gate:** a content test runs once every section it needs has a status other than `todo`. When you finish a section, set its status to `done`. The night runner checks that you did.
- Runs top to bottom on a fresh kernel. No hidden state. Re-running a section from its first cell gives the same result (N-013).
- Every section: markdown → small code cell → one sentence interpreting the output.
- Outputs are tidy: no warning spam, no huge prints, figures labelled in `CLASS_COLORS`.
- A full (non-smoke) run in Colab by Simone (CPU runtime) finishes with no errors in ≤ 15 min, including downloads.
- `FACILITATION.md` has a row for every section (N-033).

## 10. Open questions
For Simone or Susanne. An agent that hits one records it in its report and continues with the stated default.
- Course name, date and real lab length (default: ~110 min as planned).
- Does Susanne want the 🔴 ruler at all (G-33)? Default: built, marked optional.
- [Lon:23] in her slides: Lones, *How to avoid ML pitfalls*? Default: cite Karpathy only.
- Is her Colab link the car-dataset notebook? Not used by default.
- Callout rendering details (full-width tables) are verified in Simone's first Colab test.
