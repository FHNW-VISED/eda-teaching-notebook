# %% [markdown]
# ## 1. Image datasets

# %% [markdown]
# <!-- callout: objective | Objective -->
# Know where image datasets come from and what makes a good one, then hold the Play Dataset against that
# checklist.

# %% [markdown]
# The park has no training data yet. Before anyone mounts cameras and hires annotators, ask:
# *does a usable dataset already exist?* First, see what a dataset looks like.

# %%
# @title Photo browser helpers { display-mode: "form" }
import io as _io
import threading as _threading

import ipywidgets as widgets
from IPython.display import display
from PIL import Image as _Image
from PIL import ImageDraw as _ImageDraw
from PIL import ImageFont as _ImageFont
from pycocotools import mask as _mask_utils

_THUMBS: dict[tuple[str, int], _Image.Image] = {}  # (path, size) -> thumbnail, so going back is instant


def _thumb(path, size: int) -> _Image.Image:
    """A cached RGB thumbnail (JPEG decodes at reduced size, so this is fast even the first time)."""
    key = (str(path), size)
    if key not in _THUMBS:
        im = _Image.open(path)
        im.draft("RGB", (size, size))
        im = im.convert("RGB")
        im.thumbnail((size, size))
        _THUMBS[key] = im
    return _THUMBS[key]


def _prefetch(paths, size: int) -> None:
    """Load the next photos in the background while the learner looks at these."""
    _threading.Thread(target=lambda: [_thumb(p, size) for p in paths], daemon=True).start()


def _jpeg(im: _Image.Image) -> bytes:
    buf = _io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))


def _label_mask(segmentation, height: int, width: int) -> np.ndarray:
    """Decode a COCO segmentation (polygons or RLE) into a 0/1 array of the photo's size."""
    if isinstance(segmentation, list):
        rle = _mask_utils.merge(_mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation["counts"], list):
        rle = _mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    return _mask_utils.decode(rle)


def _shuffled(rows: pd.DataFrame) -> list[int]:
    """The rows' index in a fixed random order (seeded: everyone starts on the same photo)."""
    order = sorted(rows.index)
    random.Random(SEED).shuffle(order)
    return order


def _button(text: str, tip: str) -> widgets.Button:
    return widgets.Button(description=text, tooltip=tip, layout=widgets.Layout(width="auto"))


# %% [markdown]
# **What is a label here?** In this notebook a photo comes with a class, a box and a mask. The next cell draws
# one photo from the `images` and `annotations` tables with all three. Pick a class and step through the
# photos with ◀ ▶.
#
# <details><summary>How would you validate labels?</summary>
#
# Draw a random sample of N photos, check each label by eye and count the mistakes; the share of mistakes in
# the sample estimates the share in the whole set.
#
# </details>

# %%
# @title Browse photos with their class, box and mask { display-mode: "form" }
_ONE_SIZE = 640  # longest side shown; the mask and box are drawn at the photo's full size, then shrunk


def _one_photo(row) -> tuple[bytes, str]:
    """The photo with its mask tinted and its box drawn in the class colour, plus a one-line caption."""
    ann = annotations[annotations["image_id"] == row["id"]].iloc[0]
    color = CLASS_COLORS[ann["class"]]
    photo = np.array(_Image.open(row["path"]).convert("RGB"))
    m = _label_mask(ann["segmentation"], int(row["height"]), int(row["width"])).astype(bool)
    photo[m] = (0.45 * photo[m] + 0.55 * np.array(_rgb(color))).astype(np.uint8)
    im = _Image.fromarray(photo)
    x, y, w, h = ann["bbox"]
    _ImageDraw.Draw(im).rectangle([x, y, x + w, y + h], outline=_rgb(color), width=max(3, im.width // 150))
    im.thumbnail((_ONE_SIZE, _ONE_SIZE))
    caption = (
        f"<b style='color:{color}'>class: {ann['class']}</b> · box (x, y, width, height) = "
        f"{[round(v) for v in ann['bbox']]} px · mask covers {m.mean():.0%} of the photo"
    )
    return _jpeg(im), caption


class OnePhotoBrowser:
    """Class picker + ◀ ▶: one photo with its single animal's box and mask."""

    def __init__(self) -> None:
        single = images[images["n_animals"] == 1]
        self.pool = single if len(single) else images
        self.picker = widgets.Dropdown(options=["any class", *CLASSES], description="Class:")
        self.prev, self.next = _button("◀", "previous photo"), _button("▶", "next photo")
        self.where = widgets.Label()
        self.image = widgets.Image(format="jpeg", layout=widgets.Layout(max_width=f"{_ONE_SIZE}px"))
        self.caption = widgets.HTML()
        self.picker.observe(lambda _: self.show(0), names="value")
        self.prev.on_click(lambda _: self.show(self.i - 1))
        self.next.on_click(lambda _: self.show(self.i + 1))
        self.view = widgets.VBox(
            [widgets.HBox([self.picker, self.prev, self.next, self.where]), self.image, self.caption]
        )
        self.show(0)

    def show(self, i: int) -> None:
        cls = self.picker.value
        rows = self.pool if cls == "any class" else self.pool[self.pool["class"] == cls]
        order = _shuffled(rows)
        self.i = i % len(order)
        self.image.value, self.caption.value = _one_photo(images.loc[order[self.i]])
        self.where.value = f"photo {self.i + 1} of {len(order)}"


display(OnePhotoBrowser().view)

# %% [markdown]
# Before judging the dataset, look at it: six photos of each class. **🔀 Other photos** shows six new ones per
# class; ◀ goes back.

# %%
# @title Six example photos per class, framed in the class colour { display-mode: "form" }
_TILE, _GAP, _HEAD = 180, 8, 28  # px: one photo's square, the space between, the class-name row


def _class_grid(page: int, per_class: int = 6) -> bytes:
    """One picture: a column per class, `per_class` photos each, the page-th set of each class's order."""
    width = len(CLASSES) * (_TILE + _GAP) + _GAP
    height = _HEAD + per_class * (_TILE + _GAP) + _GAP
    grid = _Image.new("RGB", (width, height), "white")
    draw = _ImageDraw.Draw(grid)
    font = _ImageFont.load_default(size=16)
    upcoming = []
    for j, c in enumerate(CLASSES):
        order = _shuffled(images[images["class"] == c])
        x0 = _GAP + j * (_TILE + _GAP)
        draw.text((x0 + _TILE // 2, _HEAD // 2), c, fill=_rgb(INK), font=font, anchor="mm")
        for i in range(per_class):
            k = (page * per_class + i) % len(order)
            y0 = _HEAD + _GAP + i * (_TILE + _GAP)
            tile = _thumb(images.loc[order[k], "path"], _TILE)
            grid.paste(tile, (x0 + (_TILE - tile.width) // 2, y0 + (_TILE - tile.height) // 2))
            frame = [x0 - 2, y0 - 2, x0 + _TILE + 1, y0 + _TILE + 1]
            draw.rectangle(frame, outline=_rgb(CLASS_COLORS[c]), width=4)
            upcoming.append(images.loc[order[(k + per_class) % len(order)], "path"])
    _prefetch(upcoming, _TILE)
    return _jpeg(grid)


class ClassGridBrowser:
    """🔀 / ◀ through sets of six photos per class."""

    def __init__(self) -> None:
        self.page = 0
        self.more = _button("🔀 Other photos", "six new photos per class")
        self.back = _button("◀", "previous set")
        self.where = widgets.Label()
        self.image = widgets.Image(format="jpeg", layout=widgets.Layout(max_width="100%"))
        self.more.on_click(lambda _: self.show(self.page + 1))
        self.back.on_click(lambda _: self.show(self.page - 1))
        self.view = widgets.VBox([widgets.HBox([self.back, self.more, self.where]), self.image])
        self.show(0)

    def show(self, page: int) -> None:
        self.page = max(0, page)
        self.image.value = _class_grid(self.page)
        self.where.value = f"set {self.page + 1}"


display(ClassGridBrowser().view)

# %% [markdown]
# <!-- callout: questions | Questions -->
# The park's cameras are fixed, outdoors, and often far from the animal.

# %% [markdown]
# **Q1.** 🟢 Look at the grid above. Which class looks *least* like a fixed park camera's view, and why?
# One line.

# %% [markdown]
# <!-- callout: answer | Worked answer (Q1) -->
# Holiday snapshots, not camera-trap frames. Cats and dogs are the farthest from a
# park camera: indoors, up close, with people and sofas. Sheep, horses and zebras are at least outdoors and
# sometimes far away, closer to what the paddock and savanna cameras will see.

# %% [markdown]
# Having a dataset is not the same as having a *good* one. This is the checklist from the lecture:

# %% [markdown]
# <!-- callout: theory | Theory · What makes a good dataset -->
# - **Does it fit the question?** The images, the classes and the setting must match the job you want the
#   model to do.<br>*Park example:* the alert needs a dog seen from a fixed camera in the sheep paddock, not a
#   dog in a holiday snapshot taken from two metres away.
# - **Enough samples**, **2–10 classes** and **about 500×500 px**: the numbers to check, explained below.
# - **Classes, distributions, imbalance, bias.** Know them, and take them into account in training (for
#   example with a weighted loss).<br>
#   *Park example:* if 70 % of the photos are sheep and 3 % are dogs, the model sees few dogs to learn from,
#   and the dog is the animal the alert exists for (section 4).
# - **Variability matches the target population**, and **labels are present and validated**: every image has
#   a label, and someone checked a sample of them.<br>
#   *Park example:* a photo of a zebra labelled "horse" teaches the model the wrong thing, and nobody notices
#   until the zebra headcount is off.
# - **Objects are identifiable and separable by eye.**
# - **What do experts and the literature say** about this data and this task?
# - **Get a qualitative and a quantitative sense of the data** (Karpathy, 2019): look at many images, then
#   measure.
#
# <details><summary>The reasons and park examples behind each point</summary>
#
# - **Enough samples:** a few hundred to a few thousand images, depending on how many classes you have.
#   *Park example:* 100 photos per animal (600 in all) is a fair start; 12 zebra photos is not.
# - **2–10 classes:** more classes need more photos each, and similar classes get confused.
#   *Park example:* six animals is fine; splitting sheep and horses into 40 breeds would not be.
# - **About 500×500 px:** check the input size of the models you plan to use.
#   *Park example:* a bird across the pond may cover only 20×20 px of a 640×480 photo; shrink the photo to
#   224×224 and it is a handful of pixels.
# - **Variability:** lighting, viewpoints, backgrounds and seasons should look like the ones
#   the model will meet.
#   *Park example:* take pictures of the park in winter and in summer, in sun and at dusk. A model that only
#   saw sunny July afternoons will meet snow and low light in January.
# - **Separable by eye:** if you cannot tell them apart, the model will struggle too.
#   *Park example:* a bird that is a dark speck over the pond, or sheep standing so close that their outlines
#   merge into one woolly shape.
# - **Experts and literature:** before trusting COCO's masks, read the audit that found some of them poor; it
#   tells you what to look for.
# - **Qualitative and quantitative:** look at many images and see how big the animals are and how much of the
#   photo they cover. Is it the same for each class? Can you already see imbalances just by looking?
#
# </details>

# %% [markdown]
# <!-- callout: questions | Questions -->
# Predict before you run: the park wants to recognise, count and outline six animals from fixed cameras.

# %% [markdown]
# **Q2.** 🟢 Which bullets do you expect the Play Dataset to pass (✅) as it is, and which one worries you
# most for the dog-in-the-paddock alert? Write one line, then run the next cell. It goes through the
# checklist: where the tables from section 0 can answer a bullet it computes the answer (✅ fine, ⚠️ needs
# attention); where only your eyes can, it says so (👀).

# %%
# @title Q2 Apply the checklist to the Play Dataset { display-mode: "form" }
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
        + f" → largest / smallest = {imbalance:.1f}× (weighted loss? next week)",
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
        "COCO-ReM (Singh et al., 2024) found ~2.4 % of val masks poor, plus duplicate masks → sections 5, 8",
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
# Every ⚠️ and 👀 names the later section that settles it; that is the order we follow.

# %% [markdown]
# <!-- callout: answer | Worked answer (Q2) -->
# Sample size, class count and resolution pass without work. The two that matter for
# the alert are *fit* (holiday photos of dogs are not a camera on a paddock) and *imbalance* (the biggest
# class has over twice the images of the smallest, so a model can score well while ignoring a rare animal).
# Section 4 makes that concrete for the dog and picks the alert's metric.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · Find a dataset, or label your own -->
# Where to look for a dataset, and how to label your own images:
#
# **Find a dataset**
#
# - [Kaggle datasets](https://www.kaggle.com/datasets): a huge community collection, good for a first look.
# - [Google Dataset Search](https://datasetsearch.research.google.com): one search box over many repositories.
# - [V7 open datasets](https://www.v7labs.com/open-datasets): a curated list of open computer-vision data
#   (plus [their blog](https://www.v7labs.com/blog) with overviews of the well-known ones).
# - [Roboflow Universe](https://universe.roboflow.com): community datasets, many already labelled with boxes
#   or masks.
# - [LILA](https://lila.science): wildlife and camera-trap images, the closest match to the park's cameras.
# - [torchvision datasets](https://pytorch.org/vision/stable/datasets.html): the classics, one import away.
# - [COCO](https://cocodataset.org): a general-purpose dataset; our Play Dataset is a slice of it.
#
# **Label your own images**
#
# - [Supervisely](https://supervisely.com): a labelling platform for boxes, polygons and masks.
# - [V7](https://www.v7labs.com): browser labelling, free for education.
# - [Roboflow Annotate](https://roboflow.com/annotate): browser labelling that exports to common formats.
# - [Label Studio](https://labelstud.io): open source, runs on your own machine.

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **When you see** a dataset you did not label yourself → **do** run the checklist before you trust it, and
#   note which points only your eyes can answer.
# - **When you see** "labels are present" → **do** look at a few photos with their box and mask, then
#   spot-check a random sample and count the mistakes.
# - **When you see** a ⚠️ or 👀 on the checklist → **do** write down which later check will settle it.
