# %% [markdown]
# ## Your Project Dataset

# %%
# @title Checklist Dataset Mini-Project { display-mode: "form" }
callout("theory", "📘 Theory · What makes a good dataset", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><strong>Does it fit the question?</strong> The images, the classes and the setting must match the job you want the
  model to do.<br><em>Park example:</em> the alert needs a dog seen from a fixed camera in the sheep paddock, not a
  dog in a holiday snapshot taken from two meters away.</li>
<li style="margin:2px 0;"><strong>Enough samples</strong>, <strong>2–10 classes</strong> and <strong>about 500×500 px</strong>: the numbers to check, explained below.</li>
<li style="margin:2px 0;"><strong>Classes, distributions, imbalance, bias.</strong> Know them, and take them into account in training (for
  example with a weighted loss).<br>
  <em>Park example:</em> if 70 % of the photos are sheep and 3 % are dogs, the model sees few dogs to learn from,
  and the dog is the animal the alert exists for (Class Imbalance in Meet the Park).</li>
<li style="margin:2px 0;"><strong>Variability matches the target population</strong>, and <strong>labels are present and validated</strong>: every image has
  a label, and someone checked a sample of them.<br>
  <em>Park example:</em> a photo of a zebra labeled "horse" teaches the model the wrong thing, and nobody notices
  until the zebra headcount is off.</li>
<li style="margin:2px 0;"><strong>Objects are identifiable and separable by eye.</strong></li>
<li style="margin:2px 0;"><strong>What do experts and the literature say</strong> about this data and this task?</li>
<li style="margin:2px 0;"><strong>Get a qualitative and a quantitative sense of the data</strong> (Karpathy, 2019): look at many images, then
  measure.</li>
</ul>
<details style="margin:6px 0 0 0;">
<summary style="cursor:pointer;font-weight:bold;">The reasons and park examples behind each point</summary>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><strong>Enough samples:</strong> a few hundred to a few thousand images, depending on how many classes you have.
  <em>Park example:</em> 100 photos per animal (600 in all) is a fair start; 12 zebra photos is not.</li>
<li style="margin:2px 0;"><strong>2–10 classes:</strong> more classes need more photos each, and similar classes get confused.
  <em>Park example:</em> six animals is fine; splitting sheep and horses into 40 breeds would not be.</li>
<li style="margin:2px 0;"><strong>About 500×500 px:</strong> check the input size of the models you plan to use.
  <em>Park example:</em> a bird across the pond may cover only 20×20 px of a 640×480 photo; shrink the photo to
  224×224 and it is a handful of pixels.</li>
<li style="margin:2px 0;"><strong>Variability:</strong> lighting, viewpoints, backgrounds and seasons should look like the ones
  the model will meet.
  <em>Park example:</em> take pictures of the park in winter and in summer, in sun and at dusk. A model that only
  saw sunny July afternoons will meet snow and low light in January.</li>
<li style="margin:2px 0;"><strong>Separable by eye:</strong> if you cannot tell them apart, the model will struggle too.
  <em>Park example:</em> a bird that is a dark speck over the pond, or sheep standing so close that their outlines
  merge into one woolly shape.</li>
<li style="margin:2px 0;"><strong>Experts and literature:</strong> before trusting COCO's masks, read the audit that found some of them poor; it
  tells you what to look for.</li>
<li style="margin:2px 0;"><strong>Qualitative and quantitative:</strong> look at many images and see how big the animals are and how much of the
  photo they cover. Is it the same for each class? Can you already see imbalances just by looking?</li>
</ul>
</details>
""")


# %%
# @title 🔗 Resources: Image Datasets and Annotation Tools { display-mode: "form" }
callout("deeper", "🔗 Resources: Image Datasets and Annotation Tools", r"""
<p style="margin:6px 0 0 0;">
Where to find existing image datasets and tools for annotating your own images.
</p>

<p style="margin:10px 0 0 0;"><strong>Find a dataset</strong></p>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><a href="https://www.kaggle.com/datasets" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">Kaggle datasets</a>: a huge community collection, good for a first look.</li>
<li style="margin:2px 0;"><a href="https://datasetsearch.research.google.com" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">Google Dataset Search</a>: one search box over many repositories.</li>
<li style="margin:2px 0;"><a href="https://www.v7labs.com/open-datasets" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">V7 open datasets</a>: a curated list of open computer-vision data
  (plus <a href="https://www.v7labs.com/blog" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">their blog</a> with overviews of well-known datasets).</li>
<li style="margin:2px 0;"><a href="https://universe.roboflow.com" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">Roboflow Universe</a>: community datasets, many already labeled with boxes or masks.</li>
<li style="margin:2px 0;"><a href="https://lila.science" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">LILA</a>: wildlife and camera-trap images, the closest match to the park's cameras.</li>
<li style="margin:2px 0;"><a href="https://pytorch.org/vision/stable/datasets.html" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">torchvision datasets</a>: classic datasets available directly through PyTorch.</li>
<li style="margin:2px 0;"><a href="https://cocodataset.org" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">COCO</a>: a general-purpose dataset; our Play Dataset is a subset of it.</li>
</ul>

<p style="margin:10px 0 0 0;"><strong>Label your own images</strong></p>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><a href="https://supervisely.com" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">Supervisely</a>: a labeling platform for boxes, polygons, and masks.</li>
<li style="margin:2px 0;"><a href="https://www.v7labs.com" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">V7</a>: browser-based image annotation tools.</li>
<li style="margin:2px 0;"><a href="https://roboflow.com/annotate" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">Roboflow Annotate</a>: browser-based labeling with export to common formats.</li>
<li style="margin:2px 0;"><a href="https://labelstud.io" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">Label Studio</a>: open-source annotation software that can run locally.</li>
</ul>
""")


# %% edition="own"
# @title 🎯 Objective { display-mode: "form" }
from IPython.display import HTML; HTML(r"""<div class="callout callout-objective" style="background:#B9E3EA;color:#08505C;border-left:6px solid #08505C;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">
<div style="font-weight:bold;">🎯 Objective</div>
<p style="margin:6px 0 0 0;">Run the checklist of Meet the Park, Common Dataset Errors &amp; Cleaning and Dataset Splitting &amp; Data Leakage on your own folder-per-class dataset, and answer the guiding questions
for it the way we answered them for the park.</p>
</div>""")

# %% [markdown] edition="own"
# Your project has its own dataset, and it deserves the same twenty minutes of looking before anybody trains
# on it. This section runs the checklist on **any folder-per-class dataset**: one sub-folder per class, the
# images inside. The sub-folder names become the class names.
#
# ```
# my_dataset/                ← MY_DATASET_DIR points here (or at my_dataset.zip)
# ├── daisy/      img_001.jpg, img_002.png, …
# ├── rose/       …
# └── tulip/      …
# ```

# %% edition="own"
# ✏️ YOUR TURN
MY_DATASET_DIR = None  # paste your path between quotes, e.g. "/content/my_dataset.zip"

# %% [markdown] edition="own"
# The next cell reads your folder (or, with `None`, builds the park sample: 20 cleaned photos per class plus
# **three planted problems**, an exact copy of a cat, a mirrored dog and a grayscale bird). It prints what it
# found and shows three images per class: **check that these are your images and your class names** before
# reading anything else.

# %% edition="own"
# @title Read your folder (or build the sample), and show what was found { display-mode: "form" }
MY_SAMPLE_DIR = DATA_DIR / "cache" / "my_dataset"
MY_SAMPLE_PER_CLASS = 4 if SMOKE else 20  # photos per class in the sample (N-014: fewer in smoke mode)


def build_my_sample(dest: Path, per_class: int, seed: int) -> Path:
    """Folder-per-class copy of a few clean Play Dataset photos, plus an exact copy, a mirror, a grayscale."""
    shutil.rmtree(dest, ignore_errors=True)  # rebuilt from scratch: re-running gives the same sample (N-013)
    rng = random.Random(seed)
    pool = images  # every downloaded photo is readable
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
my_classes = sorted(my_files["class"].unique())
# Park classes keep their color; any other class name borrows one from the same palette (L-003).
_palette = list(CLASS_COLORS.values())
my_colors = {c: CLASS_COLORS.get(c, _palette[i % len(_palette)]) for i, c in enumerate(my_classes)}
print(f"✅ source: {my_source} · {my_root} · {len(my_files)} images in {len(my_classes)} class folders: "
      + ", ".join(f"{c} ({n})" for c, n in my_files["class"].value_counts().reindex(my_classes).items()))

_shown = my_classes[:8]  # a first look: three images per class (at most eight classes)
fig, axes = plt.subplots(len(_shown), 3, figsize=(7, 2.2 * len(_shown)), squeeze=False)
for r, c in enumerate(_shown):
    _of_class = my_files[my_files["class"] == c].head(3)
    for k in range(3):
        ax = axes[r, k]
        ax.axis("off")
        if k < len(_of_class):
            try:
                ax.imshow(load_thumb(_of_class.iloc[k]["path"], (192, 192)))
            except Exception:  # an unreadable file: check 2 lists it
                ax.text(0.5, 0.5, "cannot be opened", ha="center", va="center", transform=ax.transAxes)
            _name = Path(_of_class.iloc[k]["file"]).name
            ax.set_title(f"{c} · {_name if len(_name) <= 24 else _name[:20] + '…'}", fontsize=8)
fig.suptitle("A first look: are these your images and your classes?", color=INK)
fig.tight_layout()
plt.show()

# %% edition="own"
# @title 🔍 Go Deeper: Detection or Segmentation Project, and Big Datasets { display-mode: "form" }
from IPython.display import HTML; HTML(r"""<div class="callout callout-go-deeper" style="background:#F3F4F6;color:#4B5563;border-left:6px solid #4B5563;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">
<div style="font-weight:bold;">🔍 Go deeper · Detection or segmentation project, and big datasets</div>
<p style="margin:6px 0 0 0;">Put all images in one folder inside <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">MY_DATASET_DIR</code> (e.g. <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">…/project/all/</code>). The class counts then say
nothing, but sizes, modes, copies and the split still do (the box and mask checks are Object Detection and Semantic Segmentation).
Images deeper inside a class folder count, other files are listed and skipped. Datasets above
<code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">MY_MAX_IMAGES</code> photos are sampled down, keeping the class shares, so the checks stay at a minute or two.</p>
</div>""")

# %% [markdown] edition="own"
# Four checks run next and end in one report card. Check 1 counts the images per folder, as Class Distribution did for
# the park: the **ratio** is the
# largest class divided by the smallest (in the park about 130 cats against 59 sheep, 2 : 1 to 3 : 1).

# %% edition="own"
# @title Check 1: images per class, and the largest : smallest ratio { display-mode: "form" }
my_counts = my_files["class"].value_counts().reindex(my_classes)
fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(my_classes) + 2), 3.2))
ax.bar(my_classes, my_counts.to_numpy(), color=[my_colors[c] for c in my_classes])
ax.set_ylabel("images")
ax.set_title(f"Images per class folder ({len(my_files)} in total)")
ax.tick_params(axis="x", rotation=30 if len(my_classes) > 6 else 0)
plt.tight_layout()
plt.show()
my_imbalance = float(my_counts.max() / my_counts.min())
print(f"largest {my_counts.idxmax()} ({my_counts.max()}) · smallest {my_counts.idxmin()} ({my_counts.min()}) "
      f"· ratio {my_imbalance:.1f} : 1")

# %% [markdown] edition="own"
# 💡 Look at the ratio: up to 3 : 1 is common; from 10 : 1 accuracy hides the small class (Class Imbalance).

# %% [markdown] edition="own"
# Check 2 opens every file with `is_readable`, then reads its size (width × height) and its color mode (`RGB`
# stores three color channels, `L` one gray channel).

# %% edition="own"
# @title Check 2: readable? size and color mode of every file { display-mode: "form" }
# (Quantitative Inspection's measure_image on a small thumbnail)
_my_rows = []
for _, r in my_files.iterrows():
    ok = is_readable(r["path"])
    stats = measure_image(r["path"], thumb=64)[0] if ok else {}
    _my_rows.append({"file": r["file"], "class": r["class"], "readable": ok, "width": stats.get("width"),
                  "height": stats.get("height"), "mode": stats.get("mode", "unreadable")})
my_stats = pd.DataFrame(_my_rows)
_ok = my_stats[my_stats["readable"]]

fig, ax = plt.subplots(figsize=(5.5, 4))
for c in my_classes:
    sub = _ok[_ok["class"] == c]
    ax.scatter(sub["width"], sub["height"], s=18, alpha=0.7, color=my_colors[c], label=c)
ax.set_xlabel("width (px)")
ax.set_ylabel("height (px)")
ax.set_title("Image sizes per class")
ax.legend(frameon=False, fontsize=8, ncol=2 if len(my_classes) > 6 else 1)
plt.tight_layout()
plt.show()
print(f"unreadable files: {int((~my_stats['readable']).sum())}")
print("color modes:", " · ".join(f"{m} {n}" for m, n in my_stats["mode"].value_counts().items()))
print(f"distinct sizes: {_ok.groupby(['width', 'height']).ngroups} · "
      f"long side from {int(_ok[['width', 'height']].max(axis=1).min())} "
      f"to {int(_ok[['width', 'height']].max(axis=1).max())} px")
display(my_stats[my_stats["mode"] != "RGB"].head(10))

# %% [markdown] edition="own"
# 💡 Look for a second cluster of sizes or any mode that is not `RGB`: each is a second source to decide on.

# %% [markdown] edition="own"
# Check 3 is the duplicate checks of Common Dataset Errors & Cleaning: exact copies by file hash, and near, flipped and rotated copies by the
# **distance** between perceptual hashes (differing bits out of 64). Pairs at or below your `HASH_THRESHOLD`
# from Bit Distance Distribution count as copies.

# %% edition="own"
# @title Check 3: exact, recompressed, flipped and rotated copies { display-mode: "form" }
# (find_copies from Compare Duplicate Detection Methods: all pairs, in blocks to keep memory small)
_readable = my_stats.index[my_stats["readable"]].to_numpy()
_row_of = {my_files.at[i, "file"]: i for i in _readable}
my_copies = find_copies({f: my_files.at[i, "path"] for f, i in _row_of.items()}, HASH_THRESHOLD,
                        use_variants=True)
my_copies["ia"] = my_copies["a"].map(_row_of)
my_copies["ib"] = my_copies["b"].map(_row_of)
my_copies.insert(4, "same class", my_copies["ia"].map(my_files["class"]).to_numpy()
                 == my_copies["ib"].map(my_files["class"]).to_numpy())
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
            ax.imshow(load_thumb(my_files.at[idx, "path"], (256, 256)))
            ax.set_title(f"{my_files.at[idx, 'file']}\n{pr['kind']}", fontsize=8)
            frame(ax, my_colors[my_files.at[idx, "class"]])
    plt.tight_layout()
    plt.show()

# %% [markdown] edition="own"
# 💡 Look at the pairs across class folders first: one of the two labels is wrong.

# %% [markdown] edition="own"
# Check 4 suggests a split: it groups the copies, then makes a stratified (every class in both halves, in the
# same shares), group-aware split.

# %% edition="own"
# @title Check 4: a split suggestion, copies grouped, stratified by class { display-mode: "form" }
# Copies form groups that stay on one side; the split is stratified by class.
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
display(pd.crosstab(my_split["class"], my_split["split"]).reindex(my_classes))
_test_small = int((my_split["split"] == "test").groupby(my_split["class"]).sum().min())
print(f"test share {len(_test_idx) / len(my_files):.0%} · copy pairs across train/test: {_split_copies}")
if _test_small < 10:
    print(f"💡 the smallest class has only {_test_small} test images: prefer stratified {_n_splits}-fold "
          "cross-validation with the same groups, and report the mean and the spread over the folds.")
else:
    print("💡 a stratified 80/20 hold-out with these groups is enough; keep the test half for the end.")
print("If photos share a camera, a session, a burst or a patient, add that to the groups (Dataset Splitting & Data Leakage).")
_split_csv = DATA_DIR / "cache" / "my_split.csv"
my_split.to_csv(_split_csv, index=False)
print(f"split saved to {_split_csv}")

# %% [markdown] edition="own"
# 💡 Look at "copy pairs across train/test": it must be 0. The split is written to `data/cache/my_split.csv`,
# and **Colab deletes that folder when the session ends**: download it now (Files pane → right-click the
# file → Download) or copy it to your Drive, or it is gone next week.

# %% [markdown] edition="own"
# The checklist in one place, to paste into your project notes:

# %% edition="own"
# @title The report card for this dataset { display-mode: "form" }
_n_unreadable = int((~my_stats["readable"]).sum())
_n_not_rgb = int(((my_stats["mode"] != "RGB") & my_stats["readable"]).sum())
my_report = [
    ("✅" if my_imbalance < 3 else "⚠️", f"classes: {len(my_classes)}, imbalance {my_imbalance:.1f} : 1"),
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
RESULTS["swap_in"] = {"source": my_source, "n_images": int(len(my_files)), "n_classes": len(my_classes),
                      "copies": int(len(my_copies)), "not_rgb": _n_not_rgb}

# %% [markdown] edition="own"
# ### After the lab · Read your results and answer the guiding questions
#
# Everything above runs in class. This block is for you to do afterwards, on your own dataset: it explains how
# to read each check and ends with the seven guiding questions.

# %% [markdown] edition="own"
# 🟢 On the sample the folders are cat, dog, bird, horse, sheep and zebra with 20 photos each, plus
# three planted problems (an exact copy of a cat, a mirrored dog, a grayscale bird). Which check caught each
# one? Then run the checks on your own dataset and compare.

# %% edition="own"
# @title 💡 Worked Answer { display-mode: "form" }
from IPython.display import HTML; HTML(r"""<div class="callout callout-answer" style="background:#F6EFE4;color:#5C4027;border-left:6px solid #5C4027;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">

<details style="margin:0;"><summary style="cursor:pointer;font-weight:bold;">💡 Worked answer (on the sample)</summary>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><strong>Check 1:</strong> every folder holds the same number of photos, plus the planted copies in cat and dog, so
  the ratio is close to 1 : 1. That is a property of the sample, not of the park.</li>
<li style="margin:2px 0;"><strong>Check 2:</strong> all sizes fall in the Play Dataset's few clusters (long side 640 or 500 px). The grayscale
  bird shows up as one <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">L</code> file.</li>
<li style="margin:2px 0;"><strong>Check 3:</strong> the exact copy of the cat and the mirrored dog show up as an <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">exact</code> and a <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">flipped</code> pair,
  both inside one folder. The split keeps each pair on one side.</li>
<li style="margin:2px 0;">If a check missed one of them, look at the <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">HASH_THRESHOLD</code> you set in Bit Distance Distribution.</li>
</ul>
</details>
</div>""")

# %% [markdown] edition="own"
# 💡 How to read the ratio:
#
# - **2 or 3 to 1:** common and harmless if you watch the small class's recall.
# - **10 to 1 and beyond:** accuracy hides the small class completely (Class Imbalance: Can High Accuracy Be Misleading?).
# - **Always ask:** is the *expensive* class the small one?

# %% [markdown] edition="own"
# 💡 How to read the sizes and modes:
#
# - **One tight cluster of sizes:** one source.
# - **A second cluster, or one class sitting apart:** a second source, and the model may learn the source
#   instead of the animal.
# - **Every file that is not RGB** is listed in check 2: decide whether to convert it or drop it.

# %% [markdown] edition="own"
# 💡 Every copy pair is a decision for a human:
#
# - **Keep one file of each pair.**
# - **Look twice at pairs from different class folders:** one of the two labels is wrong.
# - **Many pairs on near-identical shots** (a fixed camera, a microscope) are genuinely different photos:
#   lower `HASH_THRESHOLD` in Bit Distance Distribution and rerun.

# %% edition="own"
# @title 🔍 Go Deeper: The Split Is Only a Suggestion { display-mode: "form" }
from IPython.display import HTML; HTML(r"""<div class="callout callout-go-deeper" style="background:#F3F4F6;color:#4B5563;border-left:6px solid #4B5563;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">
<div style="font-weight:bold;">🔍 Go deeper · The split is only a suggestion</div>
<p style="margin:6px 0 0 0;">The split comes from what the folders show. Only you know what else ties photos together, such as one
camera, one day, one patient; Dataset Splitting &amp; Data Leakage showed that a split which ignores such groups reports a better score
than the model will earn.</p>
</div>""")

# %% [markdown] edition="own"
# #### The guiding questions
#
# These questions are the notebook in short. Each one has the park's answer, and an empty line for yours.
#
# 🟢 Double-click the next cell and answer each guiding question for your project dataset, using
# the output of the checklist above and what you know about where the data comes from.

# %% [markdown] edition="own"
# 1. **What is the task, and which mistake is the expensive one?**
#    - *The park:* classification of who is in front of the camera; the expensive mistake is a missed dog
#      in the sheep paddock, so the metric is recall on dog (Meet the Park).
#    - *Your dataset:*
# 2. **Do the images look like what the model will see in use?**
#    - *The park:* only partly. COCO has close-ups, sofas and people; a fixed paddock camera sees small
#      animals on grass (Qualitative Inspection).
#    - *Your dataset:*
# 3. **How balanced are the classes, and is the expensive class a small one?**
#    - *The park:* cat and dog about 130 photos each, sheep about 59, roughly 2 : 1 to 3 : 1. Dog is not
#      rare, but accuracy would still hide a weak recall on dog (Meet the Park).
#    - *Your dataset:*
# 4. **Which sizes, aspect ratios and color modes are there, and what must be resized or converted?**
#    - *The park:* nearly every photo has a long side of 640 or 500 px, in landscape and portrait; files
#      that are not RGB get converted (Quantitative Inspection).
#    - *Your dataset:*
# 5. **Are there copies, flipped or rotated copies, unreadable or odd files, or doubtful labels?**
#    - *The park:* a few copies, removed before splitting; photos with
#      several animals make the label ambiguous (Meet the Park).
#    - *Your dataset:*
# 6. **What ties photos together, and which split keeps those groups apart?**
#    - *The park:* COCO has no groups, so a stratified split after deduplication; the trail cameras have
#      bursts and locations, so they are split by camera (Dataset Splitting & Data Leakage).
#    - *Your dataset:*
# 7. **What must be fixed before training?**
#    - *The park:* drop the copies, convert the files that are not RGB, decide the label of multi-animal
#      photos, and resize without distorting the aspect ratio.
#    - *Your dataset:*

# %% edition="own"
# @title ✅ Takeaway { display-mode: "form" }
from IPython.display import HTML; HTML(r"""<div class="callout callout-takeaway" style="background:#E2F0C4;color:#3A5A0B;border-left:6px solid #3A5A0B;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">
<div style="font-weight:bold;">✅ Takeaway</div>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><strong>When you get a new folder-per-class dataset</strong> → <strong>do</strong> run the four checks (classes, sizes and modes,
  copies, split) before anything else; they take a few minutes.</li>
<li style="margin:2px 0;"><strong>When a check lists files</strong> (not RGB, copy pairs, pairs across class folders) → <strong>do</strong> decide per file:
  convert, keep one, or fix the label.</li>
<li style="margin:2px 0;"><strong>When you leave</strong> → <strong>do</strong> download <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">data/cache/my_split.csv</code> (Colab deletes it) and bring it, with your
  answers to the guiding questions,
  to next week's training.</li>
</ul>
</div>""")
