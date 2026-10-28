# Run audit of Susanne's copy (her-copy.ipynb), 2026-10-28

Cell numbers are 0-based indexes into `reviews/2026-10-28-susanne-copy/her-copy.ipynb` (227 cells, markdown included).
Nothing in the repo was edited. Everything produced by this audit is in `build/` (gitignored).
Side effects, stated plainly: my first few runs used `data/cache` as it is (a symlink into the other checkout, `eda-teaching-notebook/data/cache`) and rewrote
three derived, gitignored items there: `my_split.csv` (its Oct 26 content is gone, the notebook regenerates it), `hunt5/` (regenerated each run) and an `embeddings/`
folder that I created and deleted again. All later runs used isolated folders under `build/cold-*/` that only link to the downloaded `coco` and `cct20` photos. Earlier
in the session I also ran `pkill -f ipykernel_launcher` twice to clear my own hung kernels, which would have stopped any other Jupyter kernel running on this machine at those moments.

## Summary

Both runs completed to the last cell. The smoke run (`NB_SMOKE=1`) and the full-data run give the same result: 6 cells
raise an error, and all six belong to chapter 10 ("Your project dataset", cells 201, 204, 207, 210, 213, 216). Only one
of them is a root cause. Cell 201 fails with `NameError: clean_images`, and the other five fail because 201 never
created `my_files` / `my_stats`. Everything before chapter 10 (cells 0 to 190) runs clean in both modes, with no error
and no warning other than a harmless torch line.

Behind that visible error there are two more defects that only show once the first one is fixed:
`THUMB_FOR_HASH` is never defined (cell 210), and `load_thumb` is called with an int where her version needs a tuple
(cell 201, where the error is swallowed by a `try/except` so every photo in the first-look figure says "cannot be
opened", and cell 210, where it crashes). Separately, her text states 637/638 images and 244 multi-animal photos while
the data has 633 and 241, and the Rule B explanation (2% dogs, 98% accuracy) does not match the slider default (10%, 90%).

I verified the fixes below by applying them to a copy and running smoke and full: zero errors in both, chapter 10 included
(`build/her-copy-fixed.ipynb`, `build/her-copy-fixed2*.ipynb`).

Totals, with every download already cached (so downloads are NOT in these numbers) on an Apple-silicon Mac with
`OMP/MKL/TORCH_NUM_THREADS=2`: full run 49.7 s of cell time (143 code cells), smoke run 20.9 s. A Colab free 2-vCPU
machine will be slower; on this Mac DINOv2-small takes 15 s for the 633 photos with 2 threads and 20 s with 1 thread
(see R-08). I did not measure Colab itself.

Top 5 slow cells on the full run: cell 95 (22.0 s, DINOv2 on CPU plus two UMAP fits), cell 94 (5.4 s, importing torch
and transformers and loading the model from the local Hugging Face cache), cell 96 (3.2 s, Bokeh UMAP plot with 633
embedded thumbnails), cell 75 (2.9 s, bit-distance histogram), cells 72 and 37 (2.2 s each). Only cell 95 is over 20 s.

A tooling note you should know about: plain `nbclient` 0.10 (what `tests/test_smoke.py` uses) stalled on this notebook three times. Twice it stopped
at the widget cell 21 (the kernel had finished the cell, but `nbclient` never saw the idle message). On a third run of the whole notebook in smoke mode, with a 90 s
per-cell timeout and no changes of mine, it was still silent eight minutes after it had passed cell 95 and I stopped it; I did not find which cell it was stuck in. The same
cells ran fine under a small `jupyter_client` runner (`build/run_audit2.py`, which waits for the kernel's reply), and in a few other `nbclient` runs of cell 21 alone, so
the stall is flaky and I could not tie it to one cell. I ran the whole audit with the `jupyter_client` runner. If `just check` stalls once this copy is split into
sections, that is the likely reason, and it needs its own look.

## Items

### R-01 `clean_images` is not defined in chapter 10

- **Cell:** 201, "Read your folder (or build the sample), and show what was found".
- **What happens:** `NameError: name 'clean_images' is not defined`, raised inside `build_my_sample` at
  `pool = clean_images[clean_images["flags"] == ""]`. This is the error in her own saved output. In both of my runs the
  next five cells then fail with `my_files` / `my_stats` not defined (R-04).
- **Cause:** v05 cell 162 ("Find the duplicate groups with variant hashing, then build the cleaned table", in the
  section "Clean the dataset") created `clean_images` and its `flags` column. Her copy removed that section and never
  defines the name anywhere, before or after cell 201. It is used exactly once, in cell 201.
- **Her intent:** build the "park sample" (a few photos per class from photos that have no file problems), used when
  `MY_DATASET_DIR` is `None`.
- **Fix:** in cell 201, insert these four lines directly above `MY_SAMPLE_DIR = DATA_DIR / "cache" / "my_dataset"`:

  ```python
  # The park photos that are fine to sample from: nothing unreadable, not RGB, or extreme in size.
  _odd = flag_odd_files(image_stats)  # helper from cell 71
  clean_images = images.copy()
  clean_images["flags"] = clean_images["id"].map(_odd.set_index("id")["flags"]).fillna("")
  ```

  `flag_odd_files`, `images` and `image_stats` already exist (cells 71, 9, 37). The v05 version also dropped duplicates;
  that is not needed here (verified: Check 3 then finds exactly the two planted copies, in smoke and in full).
  If chapter 10 moves to its own notebook, see R-09 for what it then needs.
- **Confidence:** high (verified by running it).

### R-02 `THUMB_FOR_HASH` is never defined

- **Cell:** 210, "Check 3: exact, recompressed, flipped and rotated copies".
- **What happens:** hidden in her saved output because cell 201 failed first. Once R-01 is fixed, cell 210 stops with
  `NameError: name 'THUMB_FOR_HASH' is not defined` at `variants(load_thumb(p, THUMB_FOR_HASH))`.
- **Cause:** v05 cell 123 ("Hash helpers") defined `THUMB_FOR_HASH = 256`. Her copy replaced that helper cell with cell 71 and
  cell 83 and dropped the constant. Used only in cell 210.
- **Her intent:** hash a 256-pixel thumbnail, the same size her own hash cells use (`load_thumb(path)` with its default
  256 x 256, cells 75, 82, 86).
- **Fix:** in cell 210, add `THUMB_FOR_HASH = (256, 256)  # the same thumbnail size as the hashes in section 5` as the
  first line after the `# @title` line. It must be a tuple, see R-03.
- **Confidence:** high (verified).

### R-03 `load_thumb` now takes a tuple, chapter 10 still passes an int

- **Cell:** 201 (silent) and 210 (crash).
- **What happens:** her `load_thumb(path, size=(256, 256))` (cell 36) calls `im.thumbnail(size)`, which needs a pair. v05's version
  took an int. Chapter 10 still calls it with ints. In cell 201 the call sits in a `try/except Exception` that prints
  "cannot be opened", so no error shows: the first-look figure ("A first look: are these your images and your classes?")
  has "cannot be opened" in every panel, for every class, even though the files are fine. I confirmed this by viewing the
  figure. In cell 210 the call in the figure at the end (`load_thumb(my_files.at[idx, "path"], 256)`) raises
  `TypeError: 'int' object is not iterable` as soon as there is at least one copy pair, which the park sample always has.
- **Cause:** her cell 36 changed the signature of the helper, chapter 10 (written against v05) was not updated. These are the
  only two int calls in the notebook; every other call (cells 119, 122, 128, 138, 167, 190) already passes a tuple.
- **Her intent:** show a small thumbnail of each photo.
- **Fix:** in cell 201 change `load_thumb(_of_class.iloc[k]["path"], 192)` to `load_thumb(_of_class.iloc[k]["path"], (192, 192))`.
  In cell 210 change `load_thumb(my_files.at[idx, "path"], 256)` to `load_thumb(my_files.at[idx, "path"], (256, 256))`.
  Do not change `load_thumb` in cell 36.
- **Confidence:** high (verified; the first-look figure then shows the photos).

### R-04 Cascade: five more NameErrors in chapter 10

- **Cell:** 204 (`my_files`), 207 (`my_files`), 210 (`my_stats`), 213 (`my_files`), 216 (`my_stats`).
- **What happens / Cause:** all follow from cell 201 stopping before it creates `my_files` (and cell 207 before it creates
  `my_stats`). They are not separate defects, and they disappear once R-01 to R-03 are applied.
- **Her intent:** the four checks and the report card.
- **Fix:** none beyond R-01 to R-03.
- **Confidence:** high.

### R-05 The Rule B example and its answer do not match the slider default

- **Cell:** 62 (markdown), 63 (`DOG_SHARE`), 64, 65 (sample answer).
- **What happens:** the markdown says "Of 1,000 images, 20 contain a dog" (2%) and the sample answer says "98% accuracy ...
  when 2% of images contain dogs". The code default is `DOG_SHARE = 0.1`, so the cell prints "Rule B ... (10% dog images ...)
  Accuracy 90.0%", and the strip in cell 64 says "accuracy 92%". A student comparing their estimate (98%) with the
  output (90%) sees a contradiction.
- **Cause:** the slider default was left at 0.1 while the text was written for 0.02.
- **Her intent:** the 2% example, with the slider as a way to explore other shares.
- **Fix:** in cell 63 change `DOG_SHARE = 0.1 ` to `DOG_SHARE = 0.02 ` (keep the `# @param` comment). Output is then 98.0%, and
  cell 64 shows 11 sheep and 1 dog, "accuracy 92%" (the same number as before, since 10% of 12 rounds to 1 dog too). Nothing else changes.
- **Confidence:** high (verified).

### R-06 Hard-coded numbers in her text that the data does not give

- **Cell:** 47, 57, 60, 65, 73 (sample answers), and 203, 225 (chapter 10 text).
- **What happens:** the notebook has 633 photos (632 RGB, 1 grayscale `L`), 1540 annotations, 241 photos with more than one
  animal. Her text says otherwise:
  - cell 47: "637 images are RGB ... and one is grayscale" should be "632 images are RGB ... and one is grayscale";
  - cell 60: "244 of 638 images (38%)" should be "241 of 633 images (38%)";
  - cell 65: "130 of 638 images" should be "130 of 633 images" (20.5%, the text's "approximately 20%" stays right);
  - cell 73: "among the 638 images" should be "among the 633 images";
  - cell 57: "Cats and dogs have the most images (130 each)" should be "Cats and dogs have the most images (130 and 129)";
  - cells 203 and 225: "about 130 cats against 50 sheep" and "sheep about 50" should say "about 60" (59 in the data; 130 : 59 is 2.2 : 1, so the
    "2 : 1 to 3 : 1" stays correct).
- **Cause:** counts written by hand from an earlier version of the dataset or from memory. Her text uses both 637 and 638.
- **Her intent:** quote the dataset's real numbers in the sample answers.
- **Fix:** apply the six replacements above in the HTML of the named cells and the two markdown cells (203, 225). They do not
  change any code. Other numbers in her answers were checked and are right: 17 grayscale-looking, sheep 5.7 / birds 3.4 /
  zebras 3.2 animals per photo, most crowded 26, one near-duplicate pair at 8 bits (a cat pair, none at 5 bits), 0 differing
  bits for the recompressed photo and 38 for the unrelated one, 3 of 5 vs 5 of 5 planted copies, 85% background and
  about 1% bird pixels against 25% of boxes, 40 leave-one-out runs.
- **Confidence:** high.

### R-07 Chapter 10 and one earlier cell point at things her copy no longer has

- **Cell:** 130 ("Section 4"), 193, 195, 202, 203, 205, 209, 213 (printed text), 218, 219, 220, 222, 223, 224, 225.
- **What happens:** nothing breaks, but the text refers to "section 2" to "section 8" and to "Q25" / "Q26", and her copy has no numbered
  headings (only "## 10. Your project dataset") and no Q numbers. Examples: "Run the checklist of sections 3 to 6",
  "(section 5's copy hunt)", "set in section 5", "section 4's Pitfall", "section 3's Callout", "section 5's 'removed N, kept M'"
  (that printed line came from the removed cleaning cell), "(section 6)" in the text printed by cell 213, and "**Q25.**" /
  "**Q26.**" in cells 218, 219 and 224. A student cannot find "section 5".
- **Cause:** chapter 10 was written for v05's numbered sections and Q numbers. She removed both (intended).
- **Her intent:** point from the checks back to the topic where each idea was taught.
- **Fix:** replace each "section N" by the heading name her copy uses, using this map: v05's numbering was 2 "Become one with the data" = her "Meet the Park: Explore Your
  Dataset", 3 "Image inspection" = "Quantitative Inspection", 4 "Labels & class distribution" = the class distribution, animals per image and imbalance
  parts of "Quantitative Inspection" (cells 54 to 65), 5 = "Common
  Dataset Errors & Cleaning", 6 = "Dataset Splitting & Data Leakage", 7 = "Object Detection", 8 = "Semantic Segmentation". Drop
  "Q25." and "Q26." from cells 218, 219, 224 (keep the question text). Delete the clause "(section 5's 'removed N, kept M')" in cell 225,
  and "section 4's Pitfall" / "section 3's Callout" can become "the class distribution part" / "the image properties part".
  Cell 130 says "image-level distribution from Section 4": change to "from the class distribution part above".
  Since chapter 10 will move to its own notebook, do this edit there, not here.
- **Confidence:** medium (I could not check every sentence against the intended target section, but none of the targets exists by number).

### R-08 DINOv2 cells (93 to 95): cost on CPU, dead helper, and a fallback that would crash cells 96 and 97

- **Cell:** 93 (helpers), 94 (pip + model), 95 (precompute), 96 and 97 (use the result).
- **What happens:** on this Mac, with 2 threads, cell 94 takes 5.4 s (imports and model load, files cached) and cell 95 takes 22.0 s: DINOv2-small over 633 photos
  15 s (42 photos/s; 20 s with 1 thread), then two UMAP fits about 7 s (the first call includes numba compilation; a warm fit
  is 4.5 s). Not measured: the `%pip` install of transformers / umap-learn / bokeh and the roughly 90 MB model download on Colab, and Colab's
  slower CPU (a few times slower than this Mac is a guess, not a measurement). Her saved output ran on an A100 where this is quick.
  Two defects in the current structure: (1) `load_embeddings()` and `cosine_similarity_matrix` in cell 93 are v05 helpers; only
  `cosine_similarity_matrix` is used (cell 97). `load_embeddings` is never called, so a precomputed file would be ignored even if it existed.
  (2) The file `data/coco_animals_embeddings.npz` that `load_embeddings` reads does not exist in this repo's `data/` folder, so someone has to
  generate and commit it (v05's route, ids + embeddings keyed by image id). Without it, skipping DINOv2 means `representations`
  and `projections` have no "Pretrained visual embeddings" key, and cells 96 and 97 (default `"Pretrained visual embeddings"`) stop
  with `KeyError`.
- **Cause:** she replaced the precomputed route by a live model call.
- **Her intent:** compare colour histograms with pretrained embeddings.
- **Fix (decision: precomputed vectors; on CPU without them skip DINOv2 unless `COMPUTE_EMBEDDINGS = True`):**
  1. Cell 94: replace the cell body with the code below (no torch at import time, `transformers` installed only when computing):

     ```python
     # @title Helpers: Pretrained Visual Embeddings { display-mode: "form" }
     %pip -q install umap-learn bokeh

     # The pretrained vectors normally come precomputed (see load_embeddings above). Computing them here takes
     # a few minutes on the free CPU runtime, so it only happens on a GPU runtime or when this is set to True.
     COMPUTE_EMBEDDINGS = False  # @param {type:"boolean"}
     MODEL_NAME = "facebook/dinov2-small"


     def dinov2_embeddings(paths, batch_size=16):
         get_ipython().run_line_magic("pip", "-q install transformers")
         import torch
         from tqdm.auto import tqdm
         from transformers import AutoImageProcessor, AutoModel

         device = "cuda" if torch.cuda.is_available() else "cpu"
         processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
         model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
         vectors = []
         for start in tqdm(range(0, len(paths), batch_size), desc="Extracting visual features"):
             batch = []
             for path in paths[start:start + batch_size]:
                 with Image.open(path) as im:
                     batch.append(im.convert("RGB"))
             inputs = processor(images=batch, return_tensors="pt").to(device)
             with torch.inference_mode():
                 features = model(**inputs).last_hidden_state[:, 0]
                 features = torch.nn.functional.normalize(features, dim=1)
             vectors.append(features.cpu().numpy())
         return np.concatenate(vectors).astype("float32")
     ```

  2. Cell 95: add this helper above `paths = image_stats["path"].tolist()`:

     ```python
     def torch_has_gpu():
         try:
             import torch
             return torch.cuda.is_available()
         except ImportError:
             return False
     ```

     and replace the whole `representations = { ... }` block by:

     ```python
     representations = {
         "Color histograms": CHANNEL_HISTS.reshape(
             len(image_stats), -1
         ).astype("float32"),
     }

     _vectors, _source = load_embeddings()  # precomputed file from the repository, if there is one
     if _source == "precomputed":
         representations["Pretrained visual embeddings"] = _vectors
     elif COMPUTE_EMBEDDINGS or torch_has_gpu():
         representations["Pretrained visual embeddings"] = cached_features(
             "global_dinov2",
             lambda: dinov2_embeddings(paths)
         )
     else:
         print("Pretrained visual embeddings skipped: no precomputed file and no GPU. "
               "Set COMPUTE_EMBEDDINGS = True in the helper cell above to compute them here.")
     ```

  3. Cells 96 and 97: directly after the `UMAP_FEATURES = ...` line (cell 96) and the `SIMILARITY_FEATURES = ...` line (cell 97) add
     `if UMAP_FEATURES not in representations:  # the pretrained vectors were skipped` and, indented, `UMAP_FEATURES = "Color histograms"`
     (and the same two lines with `SIMILARITY_FEATURES` in cell 97).
  4. Someone must produce `data/coco_animals_embeddings.npz` (arrays `ids` = `images["id"]`, `embeddings` = float32, one row per image,
     in the notebook's full 633-photo order or any order, `load_embeddings` matches by id) with `dinov2_embeddings` on a GPU and
     commit it; until then every CPU run takes the skip branch. `load_embeddings` itself needs no change.
  I ran this version three ways (smoke skip, full skip, full with `COMPUTE_EMBEDDINGS = True`; the `%pip` line removed in my venv):
  no errors; the skip runs end with cells 96 to 99 on colour histograms only, the compute run reproduces her within-class medians
  (cat 0.144, dog 0.040, bird 0.041, horse 0.112, sheep 0.284, zebra 0.549).
- **Confidence:** high for the code (verified), medium for the Colab timing (not measured).

### R-09 What chapter 10 needs from the earlier cells when it moves to its own notebook

- **Cell:** 192 to 226.
- **What happens:** chapter 10 is not self-contained. A static check of every name the chapter uses (plus the three fixes above) gives this list.
- **Cause:** it was written as the last chapter of one shared kernel.
- **Her intent:** (not hers; the chapter was written for the old build.)
- **Fix:** the new notebook needs the contents of these cells before chapter 10: 2 (`HTML`), 7 (setup: `CLASSES`, `CLASS_COLORS`, `INK`, `IN_COLAB`,
  `SMOKE`, `SEED`, `RESULTS`, `np`, `plt`, `random`), 8 (`Path`, `pd`, `Counter`, `json`, `urllib`, `DATA_DIR`, `download_file`), 28 and 71 (`from PIL import Image`), 36 (`load_thumb`, `measure_image`, `frame`), 71 (`file_md5`, `phash`, `hash_bits`, `hash_distances`, `is_readable`,
  `flag_odd_files`), 75 (`HASH_THRESHOLD = 5`), 81 (`shutil` and `ImageOps` are imported here, and nowhere earlier), 82 (`KIND_OF_VARIANT`), 83 (`VARIANT_NAMES`, `variants`).
  If the park sample stays (R-01), also cells 9 and 37 (`images`, `image_stats`). Several of these hide imports in unrelated cells, so
  copy them as imports at the top of the new notebook rather than copying the cells whole. The chapter also writes `data/cache/my_split.csv` and
  `data/cache/my_dataset/`.
- **Confidence:** medium (static analysis, not tested in a fresh notebook; the chapter itself runs with the cells above all present).

### R-10 Own-dataset routes behave as written (no defect), one thing to watch

- **Cell:** 198, 199, 201.
- **What happens:** with the R-01 to R-03 fixes I ran the chapter on the park sample, on a local class-folder directory, and on a zip with a
  wrapper folder (full mode, `MY_DATASET_DIR` set): the folder and the zip were read, the wrapper folder was stepped into, a `.txt` file
  and a broken JPEG were reported, planted cross-class copies were flagged, the report card printed. The error text for a wrong path
  could not be tested here: on this Mac the nearest existing parent of `/content/nope/x` is `/`, and listing it raised a `PermissionError`.
  On Colab the nearest existing parent of a typo under Drive would be a Drive folder, and `_contents()` in cell 199 calls `rglob("*")`
  on each of its first 8 entries, which can be slow on a large Drive.
- **Cause:** `_contents` walks whole subtrees to count images.
- **Her intent:** tell the student what is in the nearest existing folder.
- **Fix:** none needed now. If a slow listing shows up in the Colab check, count only the first level, e.g. `n = sum(_is_image(g) for g in f.iterdir())` instead of `f.rglob("*")` in `_contents`.
- **Confidence:** low (not testable headless).

## Timing

Wall time of each cell, cache warm (no downloads), 2 threads, cells over 5 s. For context, the next slowest are listed under the table.

| cell | title | smoke s | full s |
|---|---|---|---|
| 94 | Helpers: Pretrained Visual Embeddings | 5.3 | 5.4 |
| 95 | Precompute Image Representations and UMAP | 6.8 | 22.0 |

Next slowest on the full run: 96 Explore Image Representations with UMAP 3.2 s, 75 Explore Bit Distance Distribution 2.9 s, 72 Inspect
Near-Duplicates Using pHash 2.2 s, 37 Measure Image Dimensions and Other Properties 2.2 s, 189 / 175 / 180 (segmentation cells) 1.1 s each. Every other
cell is under 1 s. Sum of cell times: 49.7 s full, 20.9 s smoke. Cell 95 (the only cell over 20 s) is slow because of the DINOv2 forward
passes (15 s of the 22 s, 633 photos on 2 CPU threads) plus the first UMAP fit (numba compilation). In one earlier, cold-disk run cell 120 took 18 s (first read of the 400 cached trail-camera JPEGs) and
a smoke cell 167 took 23 s; neither repeated, so I treat them as disk-cache effects, not notebook cost. Download time is not included anywhere: the
COCO images (107 MB, 633 files) and the trail-camera photos (18 MB, 400 files) came from the local cache.

## Number mismatches

Between my full run and her saved Colab output, every printed result is identical: 633 images, 1540 annotations, per class cat 130 / dog 129 /
bird 115 / horse 118 / sheep 59 / zebra 82, 632 RGB + 1 L, 241 of 633 photos with more than one animal, 1074 candidate pairs within 20 bits,
0 byte-identical groups, 0 pairs within 5 bits, 3 of 5 versus 5 of 5 planted copies, the within-class median similarities, 400 trail-camera photos
in 142 bursts, 1519 masks, 59 suspect masks out of 1052. The only differences are environment lines ("Colab: True" versus False, download progress lines).

The mismatches are between her text and the data, listed in R-06: 637 and 638 (both wrong; it is 633), 244 (it is 241), "130 each" for cats and dogs (it is 130 and 129), "about 50 sheep" (59),
and the Rule B example (R-05).

## Manual Colab checks

Headless execution cannot test any of these. I did exercise the widget callbacks without a browser: after each widget cell I set every dropdown to every
option, every slider to min, max and middle, toggled every checkbox, and pressed every enabled button (about 160 actions on 17 widget cells). No callback raised an error. What it cannot show is how the controls look or feel.

Widgets (click through each once on Colab, with the dropdown and slider extremes): cell 21 (class dropdown, group slider, boxes checkbox), 28
(image browser), 72 (previous/next buttons and threshold slider; the buttons are disabled at the ends, a disabled click would raise an IndexError, but Colab
does not send one), 89, 97 (mode toggle, class picker, previous/next), 120 and 122 (trail-camera bursts and splitting strategies), 129, 138, 145, 149,
155, 159, 168, 176 (bounding box and mask explorers).

The multiple-choice question in cell 124: tick B and D and press "Check Answer" (should say Correct), tick anything else (should say "Not quite"), press
"Show Solution". It is plain ipywidgets Checkbox plus Button, no JavaScript, but the checkbox labels use `width="100%"` and long text, so check that they wrap on a phone-width view.

Colab form fields (`# @param`): cell 49 (`STAT_TO_SHOW` dropdown, `N_IMAGES` slider), 63 (`DOG_SHARE` slider; after R-05 its default is 0.02), 75
(`HASH_THRESHOLD` slider), 96 and 97 (representation dropdowns), 94 (`COMPUTE_EMBEDDINGS`, if R-08 is applied). Change each, rerun the cell, and check that
the downstream cells (64 uses `DOG_SHARE`, 75 feeds chapter 10) still work. Also check the collapsed `display-mode: "form"` cells ("Run me first", helpers) show their title and run.

Colab-only code paths: the `IN_COLAB` branch of cell 7 (`%pip install -q imagehash`, then "imagehash: ok"); cell 8 download from
`raw.githubusercontent.com/FHNW-VISED/eda-teaching-notebook/main/data/` (if the repository is private the students need the `GITHUB_TOKEN` secret, otherwise the manifest
download in cell 9 stops with the "Could not download" error); first-run download time of the 633 COCO photos and 400 trail-camera photos on a classroom network (the
text in cell 6 promises "1 to 2 minutes", check it with 30 students at once, where `images.cocodataset.org` and GitHub may rate-limit).

The embeddings part on the free CPU runtime (R-08): time cell 94 with the `%pip` line (it may show a "restart session" prompt if pip changes numpy or scikit-learn, which would
wipe the earlier cells), cell 95 with `COMPUTE_EMBEDDINGS = True`, and the Bokeh plot in cell 96 (`output_notebook()` loads Bokeh JS from a CDN; check the hover thumbnails
and the legend click). Check whether `umap-learn` is already installed on the Colab image.

The own-dataset chapter, after R-01 to R-03 are applied (its new notebook): the three input routes of cell 198 to 201: a path copied from the Files pane, a zip uploaded to
`/content`, and a Google Drive link (public file: no sign-in; private file: `auth.authenticate_user()` pop-up; mounted Drive: the `drive.mount` prompt on a `/content/drive/...` path).
Also a typo in the path, to read the error text and see that the listing is fast on a big Drive (R-10).

Stub exercises (cells 11 and 12, 132 and 133): run once with the stub untouched (should print "not written yet, using the reference solution"), then write a
correct body and a wrong body (✅ and ❌ lines). `use()` relies on `inspect.getsource`, which works headless but should be confirmed in Colab's cell history.

HTML callouts: the `<details>` "Sample Answer" toggles open and close, and the answer boxes drawn with `answer_box` (a closed toggle holding a figure) render at the right
width in both light and dark Colab themes.
