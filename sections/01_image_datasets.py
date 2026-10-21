# %% [markdown]
# ## 1. Image datasets

# %% [markdown]
# <!-- callout: objective | Objective -->
# Know where image datasets come from and what makes a good one, then hold the Play Dataset against that
# checklist.

# %% [markdown]
# The park has no training data yet. Before anyone mounts cameras and hires annotators, ask:
# *does a usable dataset already exist?* Here is where to look, and how to label your own images when
# nothing fits.

# %% [markdown]
# **Find a dataset**
#
# - [Kaggle datasets](https://www.kaggle.com/datasets): a huge community collection, good for a first look.
# - [Google Dataset Search](https://datasetsearch.research.google.com): one search box over many repositories.
# - [V7 open datasets](https://www.v7labs.com/open-datasets): a curated list of open computer-vision data.
# - [Roboflow Universe](https://universe.roboflow.com): community datasets, many already labelled with boxes
#   or masks.
# - [LILA](https://lila.science): wildlife and camera-trap images, the closest match to the park's cameras.
# - [V7 blog on computer-vision datasets](https://www.v7labs.com/blog): overviews of well-known datasets.
# - [torchvision datasets](https://pytorch.org/vision/stable/datasets.html): the classics, one import away.
# - [COCO](https://cocodataset.org): a general-purpose dataset; our Play Dataset is a slice of it.

# %% [markdown]
# **Label your own images**
#
# - [Supervisely](https://supervisely.com): a labelling platform for boxes, polygons and masks.
# - [V7](https://www.v7labs.com): browser labelling, free for education.
# - [Roboflow Annotate](https://roboflow.com/annotate): browser labelling that exports to common formats.
# - [Label Studio](https://labelstud.io): open source, runs on your own machine.

# %% [markdown]
# Having a dataset is not the same as having a *good* one. This is the checklist from the lecture, each point
# with an example from the park:

# %% [markdown]
# <!-- callout: theory | Theory · What makes a good dataset -->
# - **Does it fit the question?** The images, the classes and the setting must match the job you want the
#   model to do.<br>*Park example:* the alert needs a dog seen from a fixed camera in the sheep paddock, not a
#   dog in a holiday snapshot taken from two metres away.
# - **Enough samples:** a few hundred to a few thousand images, depending on how many classes you have.<br>
#   *Park example:* 100 photos per animal (600 in all) is a fair start; 12 zebra photos is not.
# - **2–10 classes.** More classes need more photos each, and similar classes get confused.<br>
#   *Park example:* six animals is fine; splitting sheep and horses into 40 breeds would not be.
# - **About 500×500 px.** Check the input size of the models you plan to use.<br>
#   *Park example:* a bird across the pond may cover only 20×20 px of a 640×480 photo; shrink the photo to
#   224×224 and it is a handful of pixels.
# - **Classes, distributions, imbalance, bias.** Know them, and take them into account in training (for
#   example with a weighted loss).<br>
#   *Park example:* if 70 % of the photos are sheep and 3 % are dogs, a model that always answers "sheep" is
#   right 70 % of the time and never raises the dog alert.
# - **Variability matches the target population.** Lighting, viewpoints, backgrounds and seasons in the data
#   should look like the ones the model will meet.<br>
#   *Park example:* take pictures of the park in winter and in summer, in sun and at dusk. A model that only
#   saw sunny July afternoons will meet snow and low light in January.
# - **Labels are present and validated.** Every image has its label, and someone checked a sample of them.<br>
#   *Park example:* a photo of a zebra labelled "horse" teaches the model the wrong thing, and nobody notices
#   until the zebra headcount is off.
# - **Objects are identifiable and separable by eye.** If you cannot tell them apart, the model will struggle
#   too.<br>
#   *Park example:* a bird that is a dark speck over the pond, or sheep standing so close that their outlines
#   merge into one woolly shape.
# - **What do experts and the literature say** about this data and this task?<br>
#   *Park example:* before trusting COCO's masks, read the audit that found some of them poor; it tells you
#   what to look for.
# - **Get a qualitative and a quantitative sense of the data** (Karpathy, 2019): look at many images, then
#   measure.<br>
#   *Park example:* look at many images and see how big the animals are and how much of the photo they
#   cover. Is it the same for each class? Can you already see imbalances just by looking?

# %% [markdown]
# **What is a label here?** In this notebook a photo comes with a class, a box and a mask. The next cell draws
# one photo from the `images` and `annotations` tables with all three.
#
# **How would you validate labels?** Draw a random sample of N photos, check each label by eye and count the
# mistakes; the share of mistakes in the sample estimates the share in the whole set.

# %%
# @title Show one photo with its class, its box and its mask { display-mode: "form" }
import matplotlib.patches as _patches
from PIL import Image as _Image
from pycocotools import mask as _mask_utils


def _label_mask(segmentation, height: int, width: int) -> np.ndarray:
    """Decode a COCO segmentation (polygons or RLE) into a 0/1 array of the photo's size."""
    if isinstance(segmentation, list):
        rle = _mask_utils.merge(_mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation["counts"], list):
        rle = _mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    return _mask_utils.decode(rle)


_single = images[images["n_animals"] == 1]
_row = (_single if len(_single) else images).sample(1, random_state=SEED).iloc[0]
_ann = annotations[annotations["image_id"] == _row["id"]].iloc[0]
_color = CLASS_COLORS[_ann["class"]]
_photo = np.array(_Image.open(_row["path"]).convert("RGB"))
_m = _label_mask(_ann["segmentation"], int(_row["height"]), int(_row["width"])).astype(bool)

_shown = _photo.copy()
_rgb = np.array([int(_color[i : i + 2], 16) for i in (1, 3, 5)])
_shown[_m] = (0.45 * _shown[_m] + 0.55 * _rgb).astype(np.uint8)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.imshow(_shown)
_bx, _by, _bw, _bh = _ann["bbox"]
ax.add_patch(_patches.Rectangle((_bx, _by), _bw, _bh, fill=False, edgecolor=_color, linewidth=3))
ax.set_title(f"class: {_ann['class']} · box and mask drawn from the tables")
ax.axis("off")
plt.show()
print(f"class = {_ann['class']!r} · box (x, y, width, height) = {[round(v) for v in _ann['bbox']]} px")
print(f"mask covers {int(_m.sum()):,} of {_m.size:,} pixels ({_m.mean():.0%} of the photo)")

# %% [markdown]
# Before judging the dataset, look at it: three photos of each class.

# %%
# @title Three example photos per class, framed in the class colour { display-mode: "form" }
_per_class = 3
fig, axes = plt.subplots(_per_class, len(CLASSES), figsize=(13, 2.2 * _per_class))
for j, c in enumerate(CLASSES):
    _of_class = images[images["class"] == c]
    _picked = _of_class.sample(min(_per_class, len(_of_class)), random_state=SEED)
    for i in range(_per_class):
        ax = axes[i, j]
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(CLASS_COLORS[c])
            spine.set_linewidth(4)
        if i < len(_picked):
            _thumb = _Image.open(_picked.iloc[i]["path"]).convert("RGB")
            _thumb.thumbnail((256, 256))
            ax.imshow(_thumb)
        if i == 0:
            ax.set_title(c, color=INK, fontweight="bold")
fig.tight_layout()
plt.show()

# %% [markdown]
# <!-- callout: questions | Questions -->
# Predict before you run: the park wants to recognise, count and outline six animals from fixed cameras.

# %% [markdown]
# **Q1.** 🟢 Which bullets do you expect the Play Dataset to pass (✅) as it is, and which one worries you
# most for the dog-in-the-paddock alert? Write one line, then run the next cell. It goes through the
# checklist: where the tables from section 0 can answer a bullet it computes the answer (✅ fine, ⚠️ needs
# attention); where only your eyes can, it says so (👀).

# %%
# @title Q1 Apply the checklist to the Play Dataset { display-mode: "form" }
# Every number comes from the `images` table of section 0.
n_images = len(images)
counts = images["class"].value_counts().reindex(CLASSES).fillna(0).astype(int)
n_classes = int((counts > 0).sum())
imbalance = counts.max() / max(counts.min(), 1)
median_w, median_h = int(images["width"].median()), int(images["height"].median())
share_labelled = float((images["n_animals"] > 0).mean())
share_several = float((images["n_animals"] > 1).mean())

checklist = [
    (
        "⚠️",
        "Fits the question?",
        "the park's six animals, but Flickr holiday photos, not fixed cameras → judge in section 2",
    ),
    (
        "✅" if n_images >= 300 else "⚠️",
        "A few hundred to a few thousand samples",
        f"{n_images} images" + ("" if n_images >= 300 else " (a tiny test sample; the full set has ~600)"),
    ),
    ("✅" if 2 <= n_classes <= 10 else "⚠️", "2–10 classes", f"{n_classes}: " + ", ".join(CLASSES)),
    (
        "✅" if min(median_w, median_h) >= 224 else "⚠️",
        "About 500×500 px",
        f"median {median_w}×{median_h} px; most classifiers take 224–512 px, so there is room to resize",
    ),
    (
        "⚠️" if imbalance > 2 else "✅",
        "Classes, distributions, imbalance, bias",
        " · ".join(f"{c} {counts[c]}" for c in CLASSES)
        + f" → largest / smallest = {imbalance:.1f}× (weighted loss? section 4)",
    ),
    ("👀", "Variability matches the target population", "lighting, distance, backgrounds: look in section 2"),
    (
        "✅" if share_labelled == 1 else "⚠️",
        "Labels present",
        f"{share_labelled:.0%} of images have at least one box and mask",
    ),
    (
        "⚠️",
        "Labels validated",
        "COCO-ReM (Singh et al., 2024) found ~2.4 % of val masks poor, plus duplicate masks → sections 5, 9",
    ),
    (
        "👀",
        "Objects identifiable and separable by eye",
        f"{share_several:.0%} of images hold several animals; check the grids in section 2",
    ),
    (
        "📚",
        "What experts and the literature say",
        "Lin et al. 2014 (COCO) and Singh et al. 2024 (COCO-ReM) document the data and its mask errors",
    ),
    ("📓", "Qualitative and quantitative sense (Karpathy 2019)", "sections 2–4 of this notebook"),
]
for status, bullet, finding in checklist:
    print(f"{status} {bullet}: {finding}")

# %% [markdown]
# Read the list top to bottom: the size, class count and resolution pass on their own, and everything that
# needs attention points to a later section of this notebook, which is the order we will follow.
#
# **Worked answer (Q1).** Sample size, number of classes and resolution pass without any work. The bullets
# that matter for the alert are *fit* and *imbalance*: holiday photos of dogs are not a camera pointed at a
# paddock, and on the full set the biggest class has more than twice the images of the smallest, so a model
# can score well while ignoring a rare animal. Section 4 makes that concrete for the dog and section 7 picks
# the metric for it.

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **When you see** a dataset you did not label yourself → **do** run the checklist before you trust it, and
#   note which points only your eyes can answer.
# - **When you see** "labels are present" → **do** look at a few photos with their box and mask, then
#   spot-check a random sample and count the mistakes.
# - **When you see** a ⚠️ or 👀 on the checklist → **do** write down which later check will settle it.
