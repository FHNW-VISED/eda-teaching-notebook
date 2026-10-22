# %% [markdown]
# ## 10. Your project dataset

# %% [markdown]
# <!-- callout: objective | Objective -->
# Run the checklist of sections 3 to 6 on your own folder-per-class dataset, and answer the guiding questions
# for it the way we answered them for the park.

# %% [markdown]
# Your project has its own dataset, and it deserves the same twenty minutes of looking before anybody trains
# on it. This section runs the checklist on **any folder-per-class dataset**: one sub-folder per class, the
# images inside. The sub-folder names become the class names.
#
# ```
# /content/drive/MyDrive/project/flowers    ← MY_DATASET_DIR points here
# ├── daisy/      img_001.jpg, img_002.png, …
# ├── rose/       …
# └── tulip/      …
# ```

# %% [markdown]
# <!-- callout: your-turn | Your turn · Point the notebook at your dataset (you change one line) -->
# **Step 1 · Get your dataset into Colab.** Colab cannot see your laptop's folders. Pick one:
#
# - **It is on Google Drive:** nothing to do now. When the path starts with `/content/drive`, the next cell
#   connects Drive for you (a Google sign-in window opens the first time).
# - **It is a `.zip` on your laptop:** click the 📁 **Files** icon in Colab's left bar and drag the zip in.
#   Wait until the upload circle disappears. Colab deletes uploads when the session ends.
#
# **Step 2 · Copy the path.** In the 📁 Files pane, find the folder whose sub-folders are your classes, or
# the `.zip` itself. Right-click it → **Copy path**. (Drive files are under `drive/MyDrive/`.)
#
# **Step 3 · Paste it** between the quotes in the next cell, click into that cell, and run everything from
# there: **Runtime → Run cell and below**.
#
# The next cell then stops with a plain message if the path is not right, and says what to change.
#
# - ✅ `/content/drive/MyDrive/project/flowers`: the folder that holds `daisy/`, `rose/`, `tulip/`.
# - ✅ `/content/flowers.zip`: it is unzipped for you.
# - ❌ `…/flowers/daisy`: one class folder, one level too deep.
# - ❌ a folder that holds `train/` and `val/`: point at `…/train` instead (then at `…/val`, if you like).
#
# Leave it at `None` to see the whole section on a small park sample first.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · Detection or segmentation project, and big datasets -->
# Put all images in one folder inside `MY_DATASET_DIR` (e.g. `…/project/all/`). The class counts then say
# nothing, but sizes, modes, copies and the split still do (the box and mask checks are sections 7 and 8).
# Images deeper inside a class folder count, other files are listed and skipped. Datasets above
# `MY_MAX_IMAGES` photos are sampled down, keeping the class shares, so the checks stay at a minute or two.

# %%
# ✏️ YOUR TURN
MY_DATASET_DIR = None  # paste your path between quotes, e.g. "/content/drive/MyDrive/project/flowers"

# %%
# @title Dataset reader: finds your class folders, or says exactly what to change { display-mode: "form" }
import zipfile

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = {"train", "training", "val", "valid", "validation", "test", "testing", "dev"}
MY_MAX_IMAGES = 2000  # larger datasets are sampled down to this many photos, stratified by class


def _visible(folder: Path) -> list[Path]:
    """The entries of `folder` that are not hidden or system files (.DS_Store, __MACOSX, …)."""
    return sorted(f for f in folder.iterdir() if not f.name.startswith((".", "_")))


def _is_image(f: Path) -> bool:
    return f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES


def _contents(folder: Path, limit: int = 8) -> str:
    """One line per entry: 'name/ (12 images)' for folders, the name for files."""
    lines = []
    for f in _visible(folder)[:limit]:
        n = sum(_is_image(g) for g in f.rglob("*")) if f.is_dir() else None
        lines.append(f"    {f.name}/  ({n} images)" if f.is_dir() else f"    {f.name}")
    more = len(_visible(folder)) - limit
    return "\n".join(lines + ([f"    … and {more} more"] if more > 0 else [])) or "    (empty)"


def find_dataset(path) -> Path:
    """The folder that holds the class folders. Stops with a plain message saying what to change if the
    path cannot be used; unzips a .zip and steps into a lone wrapper folder on its own.

    e.g. find_dataset("/content/flowers.zip") → Path("data/cache/my_upload/flowers/flowers")
    """
    p = Path(str(path).strip().strip("'\"")).expanduser()
    if str(p).startswith("/content/drive") and IN_COLAB and not Path("/content/drive/MyDrive").exists():
        from google.colab import drive  # Step 1, Drive route: connect it on first use

        drive.mount("/content/drive")
    if p.is_file() and p.suffix.lower() == ".zip":
        dest = (DATA_DIR / "cache" / "my_upload" / p.stem).resolve()  # absolute, so messages are pastable
        if not dest.exists():
            with zipfile.ZipFile(p) as z:
                z.extractall(dest)
        print(f"ℹ️ unzipped {p.name} into {dest}")
        p = dest
    if not p.exists():
        parent = next(q for q in p.parents if q.exists())
        raise SystemExit(f"❌ MY_DATASET_DIR: {p} does not exist (a typo, or the upload is not there).\n"
                         f"   The nearest folder that exists is {parent}, which holds:\n{_contents(parent)}\n"
                         "   Copy the path from the Files pane: right-click the folder → Copy path.")
    if p.is_file():
        raise SystemExit(f"❌ MY_DATASET_DIR: {p} is a file. Point at the folder that holds your class "
                         f"folders, or at a .zip. It sits in {p.parent}, which holds:\n{_contents(p.parent)}")
    while True:  # a zip often wraps everything in one more folder: step into it
        subdirs = [d for d in _visible(p) if d.is_dir()]
        loose = [f for f in _visible(p) if _is_image(f)]
        if len(subdirs) == 1 and not loose and any(d.is_dir() for d in _visible(subdirs[0])):
            print(f"ℹ️ {p.name}/ holds only the folder {subdirs[0].name}/: using that.")
            p = subdirs[0]
            continue
        break
    names = {d.name.lower() for d in subdirs}
    if subdirs and names <= SPLIT_NAMES:
        first = p / sorted(d.name for d in subdirs)[0]
        raise SystemExit(f"❌ MY_DATASET_DIR: {p} holds split folders ({', '.join(sorted(names))}), "
                         f"not class folders. Point at one of them, e.g. {first}, and run the section "
                         "again for the other.")
    if not subdirs and loose:
        raise SystemExit(f"❌ MY_DATASET_DIR: {p} holds {len(loose)} images but no class folders. If this "
                         f"is one class folder, point one level up, at {p.parent}. If your project has no "
                         "classes (detection, segmentation), move the images into one folder inside, e.g. "
                         f"{p / 'all'}.")
    if not any(_is_image(f) for f in p.rglob("*")):
        raise SystemExit(f"❌ MY_DATASET_DIR: {p} holds no image files "
                         f"({', '.join(sorted(IMAGE_SUFFIXES))}). "
                         f"It holds:\n{_contents(p)}")
    return p


def list_folder_per_class(root) -> pd.DataFrame:
    """One row per image file under `root`: columns path, file, class (the top-level folder it sits in).

    e.g. a row: {"path": "/content/flowers/rose/img_7.jpg", "file": "rose/img_7.jpg", "class": "rose"}
    """
    rows = []
    for class_dir in (d for d in _visible(Path(root)) if d.is_dir()):
        for f in sorted(class_dir.rglob("*")):
            if _is_image(f) and not f.name.startswith("."):
                rows.append({"path": str(f), "file": str(f.relative_to(root)), "class": class_dir.name})
    return pd.DataFrame(rows, columns=["path", "file", "class"])


def folder_notes(root: Path, files: pd.DataFrame) -> list[str]:
    """Warnings about what the reader skipped or what makes the checks weak (one line each)."""
    root = Path(root)
    notes = []
    loose = [f.name for f in _visible(root) if _is_image(f)]
    if loose:
        notes.append(f"⚠️ {len(loose)} images sit directly in {root.name}/, outside any class folder: "
                     "skipped.")
    others = [f for d in _visible(root) if d.is_dir() for f in d.rglob("*")
              if f.is_file() and not _is_image(f) and not f.name.startswith(".")]
    skipped = Counter(f.suffix.lower() or "(no extension)" for f in others)
    if skipped:
        notes.append("⚠️ skipped files that are not images: "
                     + ", ".join(f"{n} × {ext}" for ext, n in skipped.most_common()))
    empty = [d.name for d in _visible(root) if d.is_dir() and d.name not in set(files["class"])]
    if empty:
        notes.append(f"⚠️ class folders with no images: {', '.join(empty)}.")
    if files["class"].nunique() == 1:
        notes.append("⚠️ only one class folder: check 1 has nothing to compare (fine for a detection "
                     "project).")
    return notes

# %% [markdown]
# The next cell reads your folder (or, with `None`, builds the park sample: 20 cleaned photos per class plus
# **three planted problems**, an exact copy of a cat, a mirrored dog and a grayscale bird). It prints what it
# found and shows three images per class: **check that these are your images and your class names** before
# reading anything else.

# %%
# @title Read your folder (or build the sample), and show what was found { display-mode: "form" }
MY_SAMPLE_DIR = DATA_DIR / "cache" / "my_dataset"
MY_SAMPLE_PER_CLASS = 4 if SMOKE else 20  # photos per class in the sample (N-014: fewer in smoke mode)


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


if MY_DATASET_DIR is not None and not SMOKE:
    my_source, my_root = "yours", find_dataset(MY_DATASET_DIR)
else:
    my_source, my_root = "sample", build_my_sample(MY_SAMPLE_DIR, MY_SAMPLE_PER_CLASS, SEED)
    print("MY_DATASET_DIR is None: running on the park sample. Paste your path above to run on yours.")
my_files = list_folder_per_class(my_root)
for _note in folder_notes(my_root, my_files):
    print(_note)

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
print(f"✅ source: {my_source} · {my_root} · {len(my_files)} images in {len(MY_CLASSES)} class folders: "
      + ", ".join(f"{c} ({n})" for c, n in my_files["class"].value_counts().reindex(MY_CLASSES).items()))

_shown = MY_CLASSES[:8]  # a first look: three images per class (at most eight classes)
fig, axes = plt.subplots(len(_shown), 3, figsize=(7, 2.2 * len(_shown)), squeeze=False)
for r, c in enumerate(_shown):
    _of_class = my_files[my_files["class"] == c].head(3)
    for k in range(3):
        ax = axes[r, k]
        ax.axis("off")
        if k < len(_of_class):
            try:
                ax.imshow(load_thumb(_of_class.iloc[k]["path"], 192))
            except Exception:  # an unreadable file: check 2 lists it
                ax.text(0.5, 0.5, "cannot be opened", ha="center", va="center", transform=ax.transAxes)
            _name = Path(_of_class.iloc[k]["file"]).name
            ax.set_title(f"{c} · {_name if len(_name) <= 24 else _name[:20] + '…'}", fontsize=8)
fig.suptitle("A first look: are these your images and your classes?", color=INK)
fig.tight_layout()
plt.show()

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
# 💡 Look at the ratio: up to 3 : 1 is common; from 10 : 1 accuracy hides the small class (section 4).

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
# 💡 Look for a second cluster of sizes or any mode that is not `RGB`: each is a second source to decide on.

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
# 💡 Look at the pairs across class folders first: one of the two labels is wrong.

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
# 💡 Look at "copy pairs across train/test": it must be 0. The split is written to `data/cache/my_split.csv`,
# and **Colab deletes that folder when the session ends**: download it now (Files pane → right-click the
# file → Download) or copy it to your Drive, or it is gone next week.

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
# **Q25.** 🟢 On the sample the folders are cat, dog, bird, horse, sheep and zebra with 20 photos each, plus
# three planted problems (an exact copy of a cat, a mirrored dog, a grayscale bird). Which check caught each
# one? Then run the checks on your own dataset and compare.

# %% [markdown]
# <!-- callout: answer | Worked answer (Q25, on the sample) -->
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
#   instead of the animal.
# - **Every file that is not RGB** is listed in check 2: decide whether to convert it or drop it.

# %% [markdown]
# 💡 Every copy pair is a decision for a human:
#
# - **Keep one file of each pair.**
# - **Look twice at pairs from different class folders:** one of the two labels is wrong.
# - **Many pairs on near-identical shots** (a fixed camera, a microscope) are genuinely different photos:
#   lower `HASH_THRESHOLD` in section 5 and rerun.

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
# **Q26.** 🟢 Double-click the next cell and answer each guiding question for your project dataset, using
# the output of the checklist above and what you know about where the data comes from.

# %% [markdown]
# 1. **What is the task, and which mistake is the expensive one?**
#    - *The park:* classification of who is in front of the camera; the expensive mistake is a missed dog
#      in the sheep paddock, so the metric is recall on dog (section 4).
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
# - **When you leave** → **do** download `data/cache/my_split.csv` (Colab deletes it) and bring it, with your
#   answers to the guiding questions,
#   to next week's training.
