# %% [markdown]
# ## 1. Image datasets
#
# **Objective:** know where image datasets come from and what makes a good one, then hold the Play Dataset
# against that checklist.
#
# The park has no training data yet. Before anyone mounts cameras and hires annotators, the first question is
# always: *does a usable dataset already exist?* These are the places to look, and the tools to label your own
# images when nothing fits.
#
# **Find a dataset:**
# [Kaggle datasets](https://www.kaggle.com/datasets) ·
# [Google Dataset Search](https://datasetsearch.research.google.com) ·
# [V7 open datasets](https://www.v7labs.com/open-datasets) ·
# [Roboflow Universe](https://universe.roboflow.com) ·
# [LILA](https://lila.science) (wildlife and camera-trap data) ·
# [V7 blog on computer-vision datasets](https://www.v7labs.com/blog) ·
# [torchvision datasets](https://pytorch.org/vision/stable/datasets.html) (the classics, one import away).
#
# **Label your own images:**
# [Supervisely](https://supervisely.com) ·
# [V7](https://www.v7labs.com) (free for education) ·
# [Roboflow Annotate](https://roboflow.com/annotate) ·
# [Label Studio](https://labelstud.io) (open source).
#
# Our Play Dataset is a slice of [COCO](https://cocodataset.org), and the woodland trail-camera images of
# section 6 come from LILA. Part of "finding" a dataset is reading its licence: COCO's annotations are
# CC BY 4.0, but every photo keeps its own Flickr licence, which is why this notebook downloads the images at
# runtime instead of shipping them.
#
# Having a dataset is not the same as having a *good* one. This is the checklist from the lecture:
#
# <table width="100%"><tr><td bgcolor="#E3F2FD"><font color="#0D47A1">
# <b>📘 Theory · What makes a good dataset</b><br>
# <ul>
# <li><b>Does it fit the question?</b> The images, the classes and the setting must match the job you want the
# model to do.</li>
# <li><b>Enough samples:</b> a few hundred to a few thousand images, depending on how many classes you
# have.</li>
# <li><b>2–10 classes.</b></li>
# <li><b>About 500×500 px.</b> Check the input size of the models you plan to use.</li>
# <li><b>Classes, distributions, imbalance, bias.</b> Know them, and take them into account in training
# (for example with a weighted loss).</li>
# <li><b>Variability matches the target population.</b> Lighting, viewpoints, backgrounds and seasons in the
# data should look like the ones the model will meet.</li>
# <li><b>Labels are present and validated.</b></li>
# <li><b>Objects are identifiable and separable by eye.</b> If you cannot tell them apart, the model will
# struggle too.</li>
# <li><b>What do experts and the literature say</b> about this data and this task?</li>
# <li><b>Get a qualitative and a quantitative sense of the data</b> (Karpathy, 2019): look at many images,
# then measure.</li>
# </ul>
# </font></td></tr></table>
#
# The next cell goes through the checklist for the Play Dataset. Where the tables from section 0 can answer
# a bullet it computes the answer (✅ fine, ⚠️ needs attention); where only your eyes can, it says so (👀)
# and points to the section that will look.
#
# <table width="100%"><tr><td bgcolor="#E0F7F4"><font color="#00564D">
# <b>❓ Questions</b><br>
# Predict before you run: the park wants to recognise, count and outline six animals from fixed cameras.
# </font></td></tr></table>
#
# **Q1.** 🟢 Which bullets do you expect the Play Dataset to pass (✅) as it is, and which one worries you
# most for the dog-in-the-paddock alert? Write one line, then run the cell.

# %%
# Q1  Apply the checklist to the Play Dataset. Every number comes from the `images` table of section 0.
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
