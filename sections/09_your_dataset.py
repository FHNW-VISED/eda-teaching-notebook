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


# %% [markdown] edition="teaching"
# To run these checks on your own project dataset, open the [Own-Data Edition](https://colab.research.google.com/github/FHNW-VISED/eda-teaching-notebook/blob/main/eda-own-dataset-notebook.ipynb) of this notebook, and use these questions to guide your exploration:
#
# 1. What is the task, and which mistake is the expensive one?
# 2. Do the images look like what the model will see in use?
# 3. How balanced are the classes, and is the expensive class a small one?
# 4. Which sizes, aspect ratios and color modes are there, and what must be resized or converted?
# 5. Are there copies, flipped or rotated copies, unreadable or odd files, or doubtful labels?
# 6. What ties photos together, and which split keeps those groups apart?
# 7. What must be fixed before training?

# %% [markdown] edition="own"
# The next cell suggests a split: it groups the copies, then makes a stratified (every class in both halves, in the
# same shares), group-aware split.

# %% edition="own"
# @title A Split Suggestion: Copies Grouped, Stratified by Class { display-mode: "form" }
# Copies (find_copies from Common Dataset Errors & Cleaning, at your HASH_THRESHOLD) form groups that stay on
# one side; the split is stratified by class.
from sklearn.model_selection import StratifiedGroupKFold

copy_pairs = find_copies(dict(enumerate(images["path"])), HASH_THRESHOLD, use_variants=True)  # a, b = row numbers
_image_class = images["class"].to_numpy()
copy_pairs["same class"] = [bool(_image_class[a] == _image_class[b])
                            for a, b in zip(copy_pairs["a"], copy_pairs["b"], strict=True)]
_parent = list(range(len(images)))  # union-find: every copy pair joins its two images' groups


def _find(i: int) -> int:
    while _parent[i] != i:
        _parent[i] = _parent[_parent[i]]
        i = _parent[i]
    return i


for _, pr in copy_pairs.iterrows():
    _parent[_find(pr["a"])] = _find(pr["b"])
copy_groups = np.array([_find(i) for i in range(len(images))])

class_counts = images["class"].value_counts().reindex(CLASSES)
_n_splits = min(5, max(2, int(class_counts.min())))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # a class with fewer images than folds only triggers a warning
    _splitter = StratifiedGroupKFold(n_splits=_n_splits, shuffle=True, random_state=SEED)
    _train_idx, _test_idx = next(_splitter.split(images, images["class"], groups=copy_groups))
split_suggestion = images[["file", "class"]].copy()
split_suggestion["split"] = "train"
split_suggestion.loc[_test_idx, "split"] = "test"
_split_copies = sum(split_suggestion.at[pr["a"], "split"] != split_suggestion.at[pr["b"], "split"]
                    for _, pr in copy_pairs.iterrows())
display(pd.crosstab(split_suggestion["class"], split_suggestion["split"]).reindex(CLASSES))
_test_small = int((split_suggestion["split"] == "test").groupby(split_suggestion["class"]).sum().min())
print(f"test share {len(_test_idx) / len(images):.0%} · copy pairs across train/test: {_split_copies}")
if _test_small < 10:
    print(f"💡 the smallest class has only {_test_small} test images: prefer stratified {_n_splits}-fold "
          "cross-validation with the same groups, and report the mean and the spread over the folds.")
else:
    print("💡 a stratified 80/20 hold-out with these groups is enough; keep the test half for the end.")
print("If images share a camera, a session, a burst or a patient, add that to the groups (Dataset Splitting & Data Leakage).")
_split_csv = DATA_DIR / "cache" / "my_split.csv"
split_suggestion.to_csv(_split_csv, index=False)
print(f"split saved to {_split_csv}")

# %% [markdown] edition="own"
# 💡 Look at "copy pairs across train/test": it must be 0. The split is written to `data/cache/my_split.csv`,
# and **Colab deletes that folder when the session ends**: download it now (Files pane → right-click the
# file → Download) or copy it to your Drive, or it is gone next week.

# %% [markdown] edition="own"
# The checklist in one place, to paste into your project notes:

# %% edition="own"
# @title The report card for this dataset { display-mode: "form" }
_imbalance = float(class_counts.max() / class_counts.min())
_n_unreadable = len(coco_raw.get("unreadable", []))  # files the loader could not open (left out)
_n_not_rgb = int((image_stats["mode"] != "RGB").sum())
report_card = [
    ("✅" if _imbalance < 3 else "⚠️", f"classes: {len(CLASSES)}, imbalance {_imbalance:.1f} : 1"),
    ("✅" if _n_unreadable == 0 else "⚠️", f"unreadable files: {_n_unreadable}"),
    ("✅" if _n_not_rgb == 0 else "⚠️", f"files not stored as RGB: {_n_not_rgb}"),
    ("✅" if len(copy_pairs) == 0 else "⚠️", f"copy pairs: {len(copy_pairs)} "
     f"({int((~copy_pairs['same class']).sum())} across classes)"),
    ("✅" if _split_copies == 0 else "⚠️", f"split: {len(_test_idx)} test / {len(_train_idx)} train, "
     f"copy pairs across the split: {_split_copies}"),
]
for mark, line in report_card:
    print(mark, line)
results["swap_in_ok"] = bool(len(images) > 0 and len(image_stats) == len(split_suggestion) == len(images))
results["swap_in"] = {"source": dataset_source, "n_images": int(len(images)), "n_classes": len(CLASSES),
                      "copies": int(len(copy_pairs)), "not_rgb": _n_not_rgb}

# %% [markdown] edition="own"
# ### After the lab · Read your results and answer the guiding questions
#
# Everything above ran on your dataset. This block explains how to read the results and ends with the seven
# guiding questions.

# %% [markdown] edition="own"
# 💡 How to read the ratio:
#
# - **2 or 3 to 1:** common and harmless if you watch the small class's recall.
# - **10 to 1 and beyond:** accuracy hides the small class completely (Class Imbalance: Which Mistake Is Expensive in Your Task?).
# - **Always ask:** is the *expensive* class the small one?

# %% [markdown] edition="own"
# 💡 How to read the sizes and modes:
#
# - **One tight cluster of sizes:** one source.
# - **A second cluster, or one class sitting apart:** a second source, and the model may learn the source
#   instead of the object.
# - **Every file that is not RGB** is counted in the report card: decide whether to convert it or drop it.

# %% [markdown] edition="own"
# 💡 Every copy pair is a decision for a human:
#
# - **Keep one file of each pair.**
# - **Look twice at pairs from different class folders:** one of the two labels is wrong.
# - **Many pairs on near-identical shots** (a fixed camera, a microscope) are genuinely different photos:
#   lower `HASH_THRESHOLD` in Bit Distance Distribution and rerun.

# %% edition="own"
# @title 🔍 Go Deeper: The Split Is Only a Suggestion { display-mode: "form" }
callout("deeper", "🔍 Go Deeper · The Split Is Only a Suggestion", r"""
<p style="margin:6px 0 0 0;">The split comes from what the labels show. Only you know what else ties images together, such as one
camera, one day, one patient; Dataset Splitting &amp; Data Leakage showed that a split which ignores such groups reports a better score
than the model will earn.</p>
""")

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

# %% [markdown] edition="own"
# **Takeaway**
#
# - **When you get a new dataset** → **do** run this notebook on it (look, count, measure, find copies, split)
#   before anything else; it takes a few minutes.
# - **When a check lists files** (not RGB, copy pairs, pairs across classes) → **do** decide per file:
#   convert, keep one, or fix the label.
# - **When you leave** → **do** download `data/cache/my_split.csv` (Colab deletes it) and bring it, with your
#   answers to the guiding questions, to next week's training.
