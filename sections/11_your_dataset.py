# %% [markdown]
# ## 11. Your project dataset

# %% [markdown]
# <!-- callout: objective | Objective -->
# Run the checklist of sections 3 to 6 on your own folder-per-class dataset, and answer the guiding questions
# for it the way we answered them for the park.

# %% [markdown]
# Your project has its own dataset, and it deserves the same twenty minutes of looking before anybody trains
# on it. This section reads **any folder-per-class dataset**: one sub-folder per class, the images inside.
#
# ```
# my_dataset/
# ├── cat/      img_001.jpg, img_002.png, …
# ├── dog/      …
# └── sheep/    …
# ```

# %% [markdown]
# <!-- callout: your-turn | Your turn · Point the notebook at your dataset -->
# The workflow:
#
# 1. **Put your dataset in Colab.** Mount Google Drive (`from google.colab import drive`, then
#    `drive.mount("/content/drive")`) or upload a zip in the Files pane and unzip it. *Colab can now see the
#    folder.*
# 2. **Set `MY_DATASET_DIR`** in the next cell to the folder that holds the class folders, e.g.
#    `"/content/drive/MyDrive/project/my_dataset"`. *Each sub-folder name becomes a class.*
# 3. **Run this section again** from the cell below. *The four checks read your photos.*
# 4. **Or leave it at `None`.** *The cells run on a small sample of the Play Dataset with three planted
#    problems, so you can see what the output looks like first.*

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · Detection or segmentation project, and big datasets -->
# Put all images in one folder inside `MY_DATASET_DIR`. The class counts then say nothing, but sizes, modes,
# copies and the split still do (the box and mask checks are sections 8 and 9). Images deeper inside a class
# folder count, other files are ignored. Datasets above `MY_MAX_IMAGES` photos are sampled down, keeping the
# class shares, so the checks stay at a minute or two.

# %%
# ✏️ YOUR TURN
MY_DATASET_DIR = None  # e.g. "/content/drive/MyDrive/project/my_dataset" (one sub-folder per class)

# %% [markdown]
# The next cell lists every image with its class (its folder name). If `MY_DATASET_DIR` is `None`, or the
# folder cannot be used, it builds a sample in `data/cache/my_dataset/` instead: 20 cleaned photos
# per class, plus **three planted problems** (an exact copy of a cat, a
# mirrored copy of a dog, and a bird photo saved in grayscale).

# %%
# @title Read the folder, or build the sample { display-mode: "form" }
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
MY_SAMPLE_DIR = DATA_DIR / "cache" / "my_dataset"
MY_SAMPLE_PER_CLASS = 4 if SMOKE else 20  # photos per class in the sample (N-014: fewer in smoke mode)
MY_MAX_IMAGES = 2000  # larger datasets are sampled down to this many photos, stratified by class


def list_folder_per_class(root) -> pd.DataFrame:
    """One row per image file under `root`: columns path, file, class (the top-level folder it sits in)."""
    rows = []
    class_dirs = [p for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith((".", "_"))]
    for class_dir in sorted(class_dirs):
        for f in sorted(class_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES and not f.name.startswith("."):
                rows.append({"path": str(f), "file": str(f.relative_to(root)), "class": class_dir.name})
    return pd.DataFrame(rows, columns=["path", "file", "class"])


def build_my_sample(dest: Path, per_class: int, seed: int) -> Path:
    """Folder-per-class copy of a few clean Play Dataset photos, plus an exact copy, a mirror, a grayscale."""
    shutil.rmtree(dest, ignore_errors=True)  # rebuilt from scratch: re-running gives the same sample (N-013)
    rng = random.Random(seed)
    pool = clean_images[clean_images["flags"] == ""]
    picked = {}
    for c in CLASSES:
        of_class = sorted(pool.loc[pool["class"] == c, "path"])
        picked[c] = rng.sample(of_class, min(per_class, len(of_class)))
        (dest / c).mkdir(parents=True, exist_ok=True)
        for p in picked[c]:
            shutil.copyfile(p, dest / c / Path(p).name)
    first = {c: Path(picked[c][0]) for c in CLASSES if picked[c]}
    shutil.copyfile(first["cat"], dest / "cat" / f"{first['cat'].stem}_copy.jpg")
    with Image.open(first["dog"]) as im:
        ImageOps.mirror(im.convert("RGB")).save(dest / "dog" / f"{first['dog'].stem}_b.jpg", quality=92)
    with Image.open(first["bird"]) as im:
        im.convert("L").save(dest / "bird" / first["bird"].name, quality=92)
    return dest


my_source = "sample"
if MY_DATASET_DIR is not None and not SMOKE:
    _root = Path(str(MY_DATASET_DIR)).expanduser()
    if not _root.is_dir():
        print(f"❌ MY_DATASET_DIR = {MY_DATASET_DIR!r} is not a folder. Check the path in the Files pane.")
    elif (_listed := list_folder_per_class(_root)).empty:
        print(f"❌ No images in class sub-folders of {_root}. Expected {_root}/<class>/<image files>.")
    else:
        my_source, my_root, my_files = "yours", _root, _listed
if my_source == "sample":
    if MY_DATASET_DIR is not None:
        print("Running on the Play Dataset sample instead" + (" (smoke mode)." if SMOKE else "."))
    my_root = build_my_sample(MY_SAMPLE_DIR, MY_SAMPLE_PER_CLASS, SEED)
    my_files = list_folder_per_class(my_root)

if len(my_files) > MY_MAX_IMAGES:
    _n_all = len(my_files)
    my_files = pd.concat([g.sample(max(1, round(len(g) * MY_MAX_IMAGES / _n_all)), random_state=SEED)
                          for _, g in my_files.groupby("class")])
    print(f"⚠️ {_n_all} images found: the checks run on a stratified sample of {len(my_files)}.")
my_files = my_files.sort_values(["class", "file"]).reset_index(drop=True)
MY_CLASSES = sorted(my_files["class"].unique())
# Park classes keep their colour; any other class name borrows one from the same palette (L-003).
_palette = list(CLASS_COLORS.values())
MY_COLORS = {c: CLASS_COLORS.get(c, _palette[i % len(_palette)]) for i, c in enumerate(MY_CLASSES)}
print(f"source: {my_source} · {my_root} · {len(my_files)} images in {len(MY_CLASSES)} class folders")

# %% [markdown]
# Four checks run next and end in one report card. Check 1 counts the images per folder, as section 4 did for
# the park: the **ratio** is the
# largest class divided by the smallest (in the park about 130 cats against 50 sheep, 2 : 1 to 3 : 1).

# %%
# @title Check 1: images per class, and the largest : smallest ratio { display-mode: "form" }
my_counts = my_files["class"].value_counts().reindex(MY_CLASSES)
fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(MY_CLASSES) + 2), 3.2))
ax.bar(MY_CLASSES, my_counts.to_numpy(), color=[MY_COLORS[c] for c in MY_CLASSES])
ax.set_ylabel("images")
ax.set_title(f"Images per class folder ({len(my_files)} in total)")
ax.tick_params(axis="x", rotation=30 if len(MY_CLASSES) > 6 else 0)
plt.tight_layout()
plt.show()
my_imbalance = float(my_counts.max() / my_counts.min())
print(f"largest {my_counts.idxmax()} ({my_counts.max()}) · smallest {my_counts.idxmin()} ({my_counts.min()}) "
      f"· ratio {my_imbalance:.1f} : 1")

# %% [markdown]
# Check 2 opens every file with `is_readable`, then reads its size (width × height) and its colour mode (`RGB`
# stores three colour channels, `L` one gray channel).

# %%
# @title Check 2: readable? size and colour mode of every file { display-mode: "form" }
# (section 3's measure_image on a small thumbnail)
_my_rows = []
for _, r in my_files.iterrows():
    ok = is_readable(r["path"])
    stats = measure_image(r["path"], thumb=64)[0] if ok else {}
    _my_rows.append({"file": r["file"], "class": r["class"], "readable": ok, "width": stats.get("width"),
                  "height": stats.get("height"), "mode": stats.get("mode", "unreadable")})
my_stats = pd.DataFrame(_my_rows)
_ok = my_stats[my_stats["readable"]]

fig, ax = plt.subplots(figsize=(5.5, 4))
for c in MY_CLASSES:
    sub = _ok[_ok["class"] == c]
    ax.scatter(sub["width"], sub["height"], s=18, alpha=0.7, color=MY_COLORS[c], label=c)
ax.set_xlabel("width (px)")
ax.set_ylabel("height (px)")
ax.set_title("Image sizes per class")
ax.legend(frameon=False, fontsize=8, ncol=2 if len(MY_CLASSES) > 6 else 1)
plt.tight_layout()
plt.show()
print(f"unreadable files: {int((~my_stats['readable']).sum())}")
print("colour modes:", " · ".join(f"{m} {n}" for m, n in my_stats["mode"].value_counts().items()))
print(f"distinct sizes: {_ok.groupby(['width', 'height']).ngroups} · "
      f"long side from {int(_ok[['width', 'height']].max(axis=1).min())} "
      f"to {int(_ok[['width', 'height']].max(axis=1).max())} px")
display(my_stats[my_stats["mode"] != "RGB"].head(10))

# %% [markdown]
# Check 3 is section 5's copy hunt: exact copies by file hash, and near, flipped and rotated copies by the
# **distance** between perceptual hashes (differing bits out of 64). Pairs at or below your `HASH_THRESHOLD`
# from section 5 count as copies.

# %%
# @title Check 3: exact, recompressed, flipped and rotated copies { display-mode: "form" }
# (all pairs, in blocks to keep memory small)
_readable = my_stats.index[my_stats["readable"]].to_numpy()
_my_paths = my_files.loc[_readable, "path"].tolist()
_md5 = [file_md5(p) for p in _my_paths]
_variant_bits = {v: [] for v in VARIANT_NAMES}
for p in _my_paths:
    for v, im in variants(load_thumb(p, THUMB_FOR_HASH)).items():
        _variant_bits[v].append(phash(im))
_variant_bits = {v: hash_bits(h) for v, h in _variant_bits.items()}
_plain = _variant_bits["original"]
_n = len(_my_paths)
_best = np.full((_n, _n), 65, dtype=np.int16)  # smallest distance over the variants, and which variant
_best_v = np.zeros((_n, _n), dtype=np.int8)
for k, v in enumerate(VARIANT_NAMES):
    for start in range(0, _n, 256):
        d = hash_distances(_variant_bits[v][start:start + 256], _plain)
        better = d < _best[start:start + 256]
        _best[start:start + 256][better] = d[better]
        _best_v[start:start + 256][better] = k

_pairs = []
for i, j in zip(*np.nonzero(np.triu(_best <= HASH_THRESHOLD, k=1)), strict=True):
    a, b = _readable[i], _readable[j]
    kind = "exact" if _md5[i] == _md5[j] else KIND_OF_VARIANT[VARIANT_NAMES[_best_v[i, j]]]
    _pairs.append({"a": my_files.at[a, "file"], "b": my_files.at[b, "file"], "kind": kind,
                   "distance": int(_best[i, j]),
                   "same class": my_files.at[a, "class"] == my_files.at[b, "class"],
                   "ia": a, "ib": b})
my_copies = pd.DataFrame(_pairs, columns=["a", "b", "kind", "distance", "same class", "ia", "ib"])
print(f"{len(my_copies)} copy pairs at HASH_THRESHOLD = {HASH_THRESHOLD}:",
      " · ".join(f"{k} {n}" for k, n in my_copies["kind"].value_counts().items()) or "none")
print(f"pairs whose two files sit in different class folders (one label is wrong): "
      f"{int((~my_copies['same class']).sum())}")
display(my_copies.drop(columns=["ia", "ib"]).head(10))

_show = my_copies.head(4)
if len(_show):
    fig, axes = plt.subplots(len(_show), 2, figsize=(5, 2.4 * len(_show)), squeeze=False)
    for row, (_, pr) in zip(axes, _show.iterrows(), strict=True):
        for ax, idx in zip(row, [pr["ia"], pr["ib"]], strict=True):
            ax.imshow(load_thumb(my_files.at[idx, "path"], 256))
            ax.set_title(f"{my_files.at[idx, 'file']}\n{pr['kind']}", fontsize=8)
            frame(ax, MY_COLORS[my_files.at[idx, "class"]])
    plt.tight_layout()
    plt.show()

# %% [markdown]
# Check 4 suggests a split: it groups the copies, then makes a stratified (every class in both halves, in the
# same shares), group-aware split.

# %%
# @title Check 4: a split suggestion, copies grouped, stratified by class { display-mode: "form" }
# Copies form groups that stay on one side; the split is stratified by class.
import warnings

from sklearn.model_selection import StratifiedGroupKFold

_parent = list(range(len(my_files)))  # union-find: every copy pair joins its two files' groups


def _find(i: int) -> int:
    while _parent[i] != i:
        _parent[i] = _parent[_parent[i]]
        i = _parent[i]
    return i


for _, pr in my_copies.iterrows():
    _parent[_find(pr["ia"])] = _find(pr["ib"])
my_groups = np.array([_find(i) for i in range(len(my_files))])

_smallest = int(my_counts.min())
_n_splits = min(5, max(2, _smallest))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # a class with fewer photos than folds only triggers a warning
    _splitter = StratifiedGroupKFold(n_splits=_n_splits, shuffle=True, random_state=SEED)
    _train_idx, _test_idx = next(_splitter.split(my_files, my_files["class"], groups=my_groups))
my_split = my_files[["file", "class"]].copy()
my_split["split"] = "train"
my_split.loc[_test_idx, "split"] = "test"
_split_copies = sum(my_split.at[pr["ia"], "split"] != my_split.at[pr["ib"], "split"]
                    for _, pr in my_copies.iterrows())
display(pd.crosstab(my_split["class"], my_split["split"]).reindex(MY_CLASSES))
_test_small = int((my_split["split"] == "test").groupby(my_split["class"]).sum().min())
print(f"test share {len(_test_idx) / len(my_files):.0%} · copy pairs across train/test: {_split_copies}")
if _test_small < 10:
    print(f"💡 the smallest class has only {_test_small} test images: prefer stratified {_n_splits}-fold "
          "cross-validation with the same groups, and report the mean and the spread over the folds.")
else:
    print("💡 a stratified 80/20 hold-out with these groups is enough; keep the test half for the end.")
print("If photos share a camera, a session, a burst or a patient, add that to the groups (section 6).")
_split_csv = DATA_DIR / "cache" / "my_split.csv"
my_split.to_csv(_split_csv, index=False)
print(f"split saved to {_split_csv}")

# %% [markdown]
# The checklist in one place, to paste into your project notes:

# %%
# @title The report card for this dataset { display-mode: "form" }
_n_unreadable = int((~my_stats["readable"]).sum())
_n_not_rgb = int(((my_stats["mode"] != "RGB") & my_stats["readable"]).sum())
my_report = [
    ("✅" if my_imbalance < 3 else "⚠️", f"classes: {len(MY_CLASSES)}, imbalance {my_imbalance:.1f} : 1"),
    ("✅" if _n_unreadable == 0 else "⚠️", f"unreadable files: {_n_unreadable}"),
    ("✅" if _n_not_rgb == 0 else "⚠️", f"files not stored as RGB: {_n_not_rgb}"),
    ("✅" if len(my_copies) == 0 else "⚠️", f"copy pairs: {len(my_copies)} "
     f"({int((~my_copies['same class']).sum())} across class folders)"),
    ("✅" if _split_copies == 0 else "⚠️", f"split: {len(_test_idx)} test / {len(_train_idx)} train, "
     f"copy pairs across the split: {_split_copies}"),
]
for mark, line in my_report:
    print(mark, line)
RESULTS["swap_in_ok"] = bool(len(my_files) > 0 and len(my_stats) == len(my_split) == len(my_files))
RESULTS["swap_in"] = {"source": my_source, "n_images": int(len(my_files)), "n_classes": len(MY_CLASSES),
                      "copies": int(len(my_copies)), "not_rgb": _n_not_rgb}

# %% [markdown]
# ### After the lab · Read your results and answer the guiding questions
#
# Everything above runs in class. This block is for you to do afterwards, on your own dataset: it explains how
# to read each check and ends with the seven guiding questions.

# %% [markdown]
# **Q27.** 🟢 Think of what you expect for your dataset: the largest and the smallest class, how many image
# sizes and colour modes, and whether there are copies. (Think of your answers; there is no need to type
# them.) Then compare with the checks above. On the sample the folders are cat, dog, bird, horse, sheep and
# zebra with 20 photos each, plus the three planted problems: which check caught each one?

# %% [markdown]
# <!-- callout: answer | Worked answer (Q27, on the sample) -->
# - **Check 1:** every folder holds the same number of photos, plus the planted copies in cat and dog, so
#   the ratio is close to 1 : 1. That is a property of the sample, not of the park.
# - **Check 2:** all sizes fall in the Play Dataset's few clusters (long side 640 or 500 px). The grayscale
#   bird shows up as one `L` file.
# - **Check 3:** the exact copy of the cat and the mirrored dog show up as an `exact` and a `flipped` pair,
#   both inside one folder. The split keeps each pair on one side.
# - If a check missed one of them, look at the `HASH_THRESHOLD` you set in section 5.

# %% [markdown]
# 💡 How to read the ratio:
#
# - **2 or 3 to 1:** common and harmless if you watch the small class's recall.
# - **10 to 1 and beyond:** accuracy hides the small class completely (section 4's ⚠️ Pitfall).
# - **Always ask:** is the *expensive* class the small one?

# %% [markdown]
# 💡 How to read the sizes and modes:
#
# - **One tight cluster of sizes:** one source.
# - **A second cluster, or one class sitting apart:** a second source, and the model may learn the source
#   instead of the animal (section 3's detective beat).
# - **Every file that is not RGB** is listed in check 2: decide whether to convert it or drop it.

# %% [markdown]
# 💡 Every copy pair is a decision for a human:
#
# - **Keep one file of each pair.**
# - **Look twice at pairs from different class folders:** one of the two labels is wrong.
# - **Many pairs on near-identical shots** (a fixed camera, a microscope) are genuinely different photos:
#   lower `HASH_THRESHOLD` in section 5 and rerun.

# %% [markdown]
# <!-- callout: pitfall | Pitfall · Splitting before deduplicating -->
# If the two files of a copy pair land on different sides of the split, the test set contains photos the
# model has already seen, and the score is too good (section 6). Check 4 therefore keeps every copy group
# together, even before you have removed the copies.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · The split is only a suggestion -->
# The split comes from what the folders show. Only you know what else ties photos together, such as one
# camera, one day, one patient; section 6 showed that a split which ignores such groups reports a better score
# than the model will earn.


# %% [markdown]
# #### The guiding questions
#
# These questions are the notebook in short. Each one has the park's answer, and an empty line for yours.
#
# **Q28.** 🟢 Double-click the next cell and answer each guiding question for your project dataset, using
# the output of the checklist above and what you know about where the data comes from.

# %% [markdown]
# 1. **What is the task, and which mistake is the expensive one?**
#    - *The park:* classification of who is in front of the camera; the expensive mistake is a missed dog
#      in the sheep paddock, so the metric is recall on dog (section 7).
#    - *Your dataset:*
# 2. **Do the images look like what the model will see in use?**
#    - *The park:* only partly. COCO has close-ups, sofas and people; a fixed paddock camera sees small
#      animals on grass (section 2).
#    - *Your dataset:*
# 3. **How balanced are the classes, and is the expensive class a small one?**
#    - *The park:* cat and dog about 130 photos each, sheep about 50, roughly 2 : 1 to 3 : 1. Dog is not
#      rare, but accuracy would still hide a weak recall on dog (section 4).
#    - *Your dataset:*
# 4. **Which sizes, aspect ratios and colour modes are there, and what must be resized or converted?**
#    - *The park:* nearly every photo has a long side of 640 or 500 px, in landscape and portrait; files
#      that are not RGB get converted (section 3).
#    - *Your dataset:*
# 5. **Are there copies, flipped or rotated copies, unreadable or odd files, or doubtful labels?**
#    - *The park:* a few copies, removed before splitting (section 5's "removed N, kept M"); photos with
#      several animals make the label ambiguous (section 4).
#    - *Your dataset:*
# 6. **What ties photos together, and which split keeps those groups apart?**
#    - *The park:* COCO has no groups, so a stratified split after deduplication; the trail cameras have
#      bursts and locations, so they are split by camera (section 6).
#    - *Your dataset:*
# 7. **What must be fixed before training?**
#    - *The park:* drop the copies, convert the files that are not RGB, decide the label of multi-animal
#      photos, and resize without distorting the aspect ratio (section 3's 🔀 Callout).
#    - *Your dataset:*

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **When you get a new folder-per-class dataset** → **do** run the four checks (classes, sizes and modes,
#   copies, split) before anything else; they take a few minutes.
# - **When a check lists files** (not RGB, copy pairs, pairs across class folders) → **do** decide per file:
#   convert, keep one, or fix the label.
# - **When you leave** → **do** bring your answers to the guiding questions and `data/cache/my_split.csv`
#   to next week's training.
