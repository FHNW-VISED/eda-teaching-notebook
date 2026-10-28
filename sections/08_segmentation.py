# %% [markdown] edition="teaching"
#
# ## Semantic Segmentation: Which Pixels Belong to the Animals?
#
# So far, we have explored two ways of describing animals in images:
#
# - **Image classification:** Which animal class is in the image?
# - **Object detection:** Where are the individual animals, represented by bounding boxes?
#
# *Semantic segmentation* goes one step further: it assigns a class to every pixel, distinguishing animal areas from the background.
#
# Our dataset contains segmentation annotations in addition to class labels and bounding boxes. These annotations allow us to explore the actual areas occupied by animals, rather than just the rectangles surrounding them.
#
#

# %% [markdown] edition="own"
# ## Semantic Segmentation: Which Pixels Belong to the Objects?
#
# So far, we have explored two ways of describing objects in images:
#
# - **Image classification:** Which class is in the image?
# - **Object detection:** Where are the individual objects, represented by bounding boxes?
#
# *Semantic segmentation* goes one step further: it assigns a class to every pixel, distinguishing object areas from the background.
#
# This chapter runs only if `TASKS` includes `"segmentation"`, so its cells skip themselves when your labels have no masks.

# %% edition="teaching"
# @title 📘 Semantic Segmentation: Masks and Pixels { display-mode: "form" }
callout("theory", "📘 Semantic Segmentation: Masks and Pixels", r"""
<ul>
<li><strong>Binary mask:</strong> A two-dimensional array indicating which pixels
belong to an object (1) and which do not (0).</li>

<li><strong>Semantic segmentation:</strong> Assigns a class label to every pixel.
Pixels that do not belong to any animal are labeled as <em>background</em>.</li>

<li><strong>Instance segmentation:</strong> Identifies individual objects using
separate masks, even when they belong to the same class.</li>
</ul>

<p style="margin-bottom:0;">
Our COCO dataset provides instance-level segmentation annotations.
We can combine these masks to obtain pixel-level class labels for semantic segmentation.
</p>
""")


# %% edition="own"
# @title 📘 Semantic Segmentation: Masks and Pixels { display-mode: "form" }
callout("theory", "📘 Semantic Segmentation: Masks and Pixels", r"""
<ul>
<li><strong>Binary mask:</strong> A two-dimensional array indicating which pixels
belong to an object (1) and which do not (0).</li>

<li><strong>Semantic segmentation:</strong> Assigns a class label to every pixel.
Pixels that do not belong to any object are labeled as <em>background</em>.</li>

<li><strong>Instance segmentation:</strong> Identifies individual objects using
separate masks, even when they belong to the same class.</li>
</ul>

<p style="margin-bottom:0;">
When you loaded your dataset, the notebook turned your labels into one mask per object.
We combine these masks to obtain pixel-level class labels for semantic segmentation.
PNG label masks store classes, not objects, so touching objects of the same class count as one object here.
</p>
""")

# %% [markdown]
# ### 🟢 Exploring Segmentation Masks
#
# Segmentation masks show which pixels belong to animals, rather than just the rectangular areas defined by bounding boxes.
#
# **Before running the code:**
# - How much of an image do you expect to be covered by animals compared with the background?
#
# **Explore:**
# - Browse images with their segmentation masks overlaid.
# - Compare the masks across different animals and images.
#
# **Observe and interpret:**
# - How much of each image is covered by animals, and how much is background?
# - How do masks represent animal shapes differently from bounding boxes?
# - Which animal characteristics might make accurate segmentation difficult?

# %%
# @title Helpers: Masks and Data Preparation { display-mode: "form" }
MIN_JUDGED_AREA = 32**2  # masks under COCO's "small" limit (32×32 px) are too few pixels to judge their shape
THIN_PX = 7  # parts narrower than this many photo pixels count as thin (about 2 px at 224×224)
BLOB_SHARE = 0.005  # pieces and holes under 0.5 % of the mask (or 16 px) are rasterization noise, not shape


def thin_part(mask: np.ndarray) -> np.ndarray:
    """The pixels of `mask` in parts narrower than THIN_PX: what an opening with a THIN_PX disc removes."""
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (THIN_PX, THIN_PX))
    return mask & (1 - cv2.morphologyEx(mask, cv2.MORPH_OPEN, disc))


def mask_shape(mask: np.ndarray, bbox) -> dict:
    """Shape numbers of one mask (0/1 array) against its annotated box (x, y, w, h).

    box_iou: IoU of the mask's own tight box with the annotated box · fill: mask area / box area ·
    pieces, holes: separate parts and enclosed gaps · solidity: mask area / area of its convex hull (a rubber
    band around it; 1.0 = no dents at all) · thin: share of the mask's pixels in thin parts.
    """
    area = int(mask.sum())
    if area == 0:
        return {"box_iou": 0.0, "fill": 0.0, "pieces": 0, "holes": 0, "solidity": 0.0, "thin": 0.0}
    mx, my, mw, mh = cv2.boundingRect(mask)
    crop = np.pad(mask[my:my + mh, mx:mx + mw], 1)
    min_blob = max(16, BLOB_SHARE * area)
    _, _, parts, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    pieces = int((parts[1:, cv2.CC_STAT_AREA] >= min_blob).sum())
    _, labels, gaps, _ = cv2.connectedComponentsWithStats(1 - crop, connectivity=4)
    outside = labels[0, 0]  # the background that touches the border is not a hole
    holes = sum(int(gaps[i, cv2.CC_STAT_AREA] >= min_blob) for i in range(1, len(gaps)) if i != outside)
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(crop)
    cv2.fillPoly(hull, [cv2.convexHull(np.vstack(contours))], 1)
    w, h = bbox[2], bbox[3]
    return {"box_iou": box_iou_dice((mx, my, mw, mh), tuple(bbox))[0],
            "fill": area / max(w * h, 1),
            "pieces": pieces,
            "holes": holes,
            "solidity": area / max(int(hull.sum()), 1),
            "thin": int(thin_part(crop).sum()) / area}



def show_masks(rows_df: pd.DataFrame, masks_df: pd.DataFrame, title: str, caption=None, cols: int = 4,
               size: int = 384, mark_thin: bool = False, highlight: dict | None = None,
               show_mask: bool = True, show_box: bool = True) -> None:
    """Gallery of photos with optional segmentation masks and bounding boxes."""
    n = len(rows_df)
    if n == 0:
        print(f"{title}: no images to show.")
        return

    cols = min(cols, n)
    n_rows = -(-n // cols)
    fig, axes = plt.subplots(n_rows, cols, figsize=(3.4 * cols, 3.2 * n_rows), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (_, row) in zip(axes.ravel(), rows_df.iterrows(), strict=False):
        thumb = load_thumb(row["path"], (size, size))
        scale = thumb.width / row["width"]
        own = masks_df[masks_df["image_id"] == row["id"]]

        flagged = (highlight or {}).get(row["id"])
        if flagged is not None:
            own = own.assign(_top=own["id"] == flagged).sort_values("_top")

        ax.axis("on")
        ax.imshow(thumb)

        if show_mask:
            overlay = np.zeros((thumb.height, thumb.width, 4))

            for _, m in own.iterrows():
                full = decode(m["rle"])
                small = cv2.resize(full, thumb.size, interpolation=cv2.INTER_NEAREST) > 0
                alpha = 0.6 if flagged is None else (0.8 if m["id"] == flagged else 0.15)
                overlay[small] = (*mpl.colors.to_rgb(CLASS_COLORS[m["class"]]), alpha)

                if mark_thin:
                    thin = cv2.resize(thin_part(full), thumb.size, interpolation=cv2.INTER_NEAREST) > 0
                    overlay[thin] = (*mpl.colors.to_rgb(INK), 0.95)

            ax.imshow(overlay)

        if show_box:
            for _, m in own.iterrows():
                x, y, w, h = m["bbox"]
                if flagged is not None and m["id"] != flagged:
                    continue

                ax.add_patch(mpl.patches.Rectangle(
                    (x * scale, y * scale), w * scale, h * scale,
                    fill=False,
                    lw=2.5 if flagged is not None else 1.5,
                    ls="-" if flagged is not None else "--",
                    edgecolor=CLASS_COLORS[m["class"]]
                ))

        frame(ax, CLASS_COLORS[row["class"]])
        ax.set_title(caption(row) if caption else f"{row['class']} · {len(own)} masks", fontsize=9)

    fig.suptitle(title, color=INK)
    fig.tight_layout()
    display(fig)
    plt.close(fig)


# Prepare masks for subsequent inspection activities.
masks = boxes.loc[
    boxes["segmentation"].notna(),  # annotations without a mask (boxes or class labels only) are skipped
    ["id", "image_id", "class", "bbox", "iscrowd", "segmentation"]
].copy()

masks["rle"] = [
    to_rle(seg, int(images_by_id.at[i, "height"]), int(images_by_id.at[i, "width"]))
    for seg, i in zip(masks["segmentation"], masks["image_id"], strict=True)
]

masks["mask_area"] = [int(mask_utils.area(r)) for r in masks["rle"]]
masks = masks.drop(columns="segmentation")

# %%

# @title 🔎 Explore Segmentation Masks { display-mode: "form" }
# needs: masks
mask_images = images[images["id"].isin(masks["image_id"].unique())].copy()
mask_images = mask_images.sample(frac=1, random_state=SEED).reset_index(drop=True)

maskview_page = 0

maskview_prev_btn = widgets.Button(description="Previous", icon="chevron-left")
maskview_next_btn = widgets.Button(description="Next", icon="chevron-right")
maskview_shuffle_btn = widgets.Button(description="Shuffle", icon="shuffle")

maskview_mask_toggle = widgets.Checkbox(value=True, description="Show masks", indent=False)
maskview_box_toggle = widgets.Checkbox(value=True, description="Show boxes", indent=False)

maskview_page_label = widgets.HTML()
maskview_output = widgets.Output()

def maskview_show_page():
    total_pages = (len(mask_images) + PAGE_SIZE - 1) // PAGE_SIZE
    start = maskview_page * PAGE_SIZE
    selected = mask_images.iloc[start:start + PAGE_SIZE]

    maskview_page_label.value = f"Page {maskview_page + 1} of {total_pages}"
    maskview_prev_btn.disabled = (maskview_page == 0)
    maskview_next_btn.disabled = (maskview_page >= total_pages - 1)

    with maskview_output:
        maskview_output.clear_output(wait=True)
        show_masks(
            selected,
            masks,
            "Segmentation masks",
            caption=lambda r: r["class"],
            show_mask=maskview_mask_toggle.value,
            show_box=maskview_box_toggle.value
        )

def maskview_navigate(change):
    global maskview_page, mask_images

    total_pages = (len(mask_images) + PAGE_SIZE - 1) // PAGE_SIZE

    if change == "previous":
        maskview_page = max(0, maskview_page - 1)
    elif change == "next":
        maskview_page = min(total_pages - 1, maskview_page + 1)
    elif change == "shuffle":
        mask_images = mask_images.sample(frac=1).reset_index(drop=True)
        maskview_page = 0

    maskview_show_page()

maskview_prev_btn.on_click(lambda b: maskview_navigate("previous"))
maskview_next_btn.on_click(lambda b: maskview_navigate("next"))
maskview_shuffle_btn.on_click(lambda b: maskview_navigate("shuffle"))

maskview_mask_toggle.observe(lambda change: maskview_show_page(), names="value")
maskview_box_toggle.observe(lambda change: maskview_show_page(), names="value")

display(
    widgets.HBox([maskview_prev_btn, maskview_next_btn, maskview_shuffle_btn, maskview_page_label]),
    widgets.HBox([maskview_mask_toggle, maskview_box_toggle]),
    maskview_output
)

maskview_show_page()


# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul>
<li><strong>Animal vs. background:</strong> The proportion of animal pixels varies
considerably. Some animals occupy large parts of an image, while others cover
only a small area.</li>

<li><strong>Masks vs. bounding boxes:</strong> Masks follow animal shapes more
closely, excluding background pixels that fall inside bounding boxes.</li>

<li><strong>Segmentation challenges:</strong> Thin structures such as legs, tails,
and beaks, as well as small or partially hidden animals, can be difficult
to segment accurately.</li>
</ul>
""")


# %% [markdown] edition="own"
# ✍️ **Your answer** (double-click to edit):
#
# *In the park:* the share of animal pixels varied a lot between images, and thin parts such as legs, tails, and beaks were the hardest to outline.

# %% [markdown] edition="teaching"
# ### 🟢 Pixel-Level Class Imbalance
#
# In semantic segmentation, class distribution is measured by pixels rather than images or objects.
#
# **Before running the code:**
# - Do you expect background pixels to make up a quarter, half, or most of the dataset?
# - Do you expect birds to be frequent or rare when measured by pixels?
#
# **Explore:**
# - Compare class distributions measured by images, bounding boxes, and pixels, including background.
#
# **Observe and interpret:**
# - How dominant is the background?
# - Which classes change most in relative frequency when counting pixels instead of images or objects?
# - How might these differences affect the training and evaluation of a segmentation model?

# %% [markdown] edition="own"
# ### 🟢 Pixel-Level Class Imbalance
#
# In semantic segmentation, class distribution is measured by pixels, not by images or objects.
#
# **Before running the code:**
# - Do you expect background pixels to make up a quarter, half, or most of your dataset?
# - Do you expect your smallest objects to be frequent or rare when measured by pixels?
#
# **Explore:**
# - Compare class distributions measured by images, bounding boxes, and pixels, including background.
#
# **Observe and interpret:**
# - How dominant is the background?
# - Which classes change most in relative frequency when counting pixels instead of images or objects?
# - How might these differences affect the training and evaluation of a segmentation model?

# %%
# @title 📘 Pixel-Level Class Imbalance { display-mode: "form" }
callout("theory", "📘 Pixel-Level Class Imbalance", r"""
<p>In semantic segmentation, each pixel is assigned a class label,
including background. This changes how we measure class imbalance.</p>

<ul style="margin-bottom:0;">
<li><strong>Background dominance:</strong> Large parts of an image may contain
no animals, resulting in many background pixels.</li>
<li><strong>Object size:</strong> Large animals contribute more labeled pixels
than small animals, even when both count as one object.</li>
<li><strong>Training implications:</strong> With an unweighted pixel-wise loss,
larger regions generally contribute more loss terms. Class weighting or
alternative loss functions can help address this imbalance.</li>
</ul>
""")


# %%

# @title 🔎 Explore Pixel-Level Class Imbalance { display-mode: "form" }
# needs: masks

# Calculate pixel shares
photo_rle = {}  # One merged mask per photo, reused by the heatmaps below
animal_px = dict.fromkeys(CLASSES, 0)

for image_id, own in masks.groupby("image_id"):
    photo_rle[image_id] = mask_utils.merge(list(own["rle"]), intersect=False)
    animal_px[own["class"].iloc[0]] += int(mask_utils.area(photo_rle[image_id]))

total_px = int((images["width"] * images["height"]).sum())

pixels_per_class = {c: animal_px[c] / total_px for c in CLASSES}
pixels_per_class["background"] = (
    total_px - sum(animal_px.values())
) / total_px

RESULTS["pixels_per_class"] = pixels_per_class

BACKGROUND_COLOR = "#D9DEDF"

# Compare class distributions across counting units
units = {
    "images": {
        c: images_per_class[c] / sum(images_per_class.values())
        for c in CLASSES
    },
    "boxes": {
        c: boxes_per_class[c] / sum(boxes_per_class.values())
        for c in CLASSES
    },
    "pixels": {
        c: animal_px[c] / max(sum(animal_px.values()), 1)
        for c in CLASSES
    },
}

fig, axes = plt.subplots(
    1, 2, figsize=(12, 3.8),
    gridspec_kw={"width_ratios": [2.2, 1]}
)

# Left: class distribution among animals only
ax = axes[0]
x = np.arange(len(CLASSES))

for k, (unit, alpha) in enumerate([
    ("images", 0.3),
    ("boxes", 0.6),
    ("pixels", 1.0)
]):
    values = [units[unit][c] for c in CLASSES]

    bars = ax.bar(
        x + (k - 1) * 0.27,
        values,
        width=0.27,
        alpha=alpha,
        color=[CLASS_COLORS[c] for c in CLASSES],
        edgecolor="white"
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.004,
            f"{value:.0%}",
            ha="center",
            va="bottom",
            fontsize=7
        )

ax.set_xticks(x, CLASSES)

for label in ax.get_xticklabels():
    label.set_color(CLASS_COLORS[label.get_text()])

ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set_ylim(0, max(max(v.values()) for v in units.values()) * 1.18)
ax.set_title("Class distribution among animals: images vs. boxes vs. pixels")

legend = [
    mpl.patches.Patch(color=INK, alpha=alpha, label=unit)
    for unit, alpha in [
        ("images", 0.3),
        ("boxes", 0.6),
        ("pixels", 1.0)
    ]
]
ax.legend(handles=legend, frameon=False, fontsize=8)

# Right: pixel distribution including background
ax = axes[1]
left = 0.0

for name in ["background", *CLASSES]:
    share = pixels_per_class[name]

    ax.barh(
        0,
        share,
        left=left,
        color=BACKGROUND_COLOR if name == "background" else CLASS_COLORS[name],
        edgecolor="white"
    )

    if name == "background":
        ax.text(
            left + share / 2,
            0,
            f"Background\n{share:.1%}",
            ha="center",
            va="center",
            fontsize=10,
            color=INK
        )

    left += share

ax.set_xlim(0, 1)
ax.set_yticks([])
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set_title("All pixels, background included")

fig.tight_layout()
display(fig)
plt.close(fig)


# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul>
<li><strong>Background dominance:</strong> About 85% of all pixels belong to the background.
Animals occupy only about 15%.</li>

<li><strong>Class differences:</strong> Birds account for about 1% of all pixels,
despite representing 25% of bounding boxes. They are often small or distant.
Cats and dogs occupy relatively more pixels, partly because they frequently
appear in close-up images.</li>

<li><strong>Implications:</strong> Class frequencies depend on whether we count
images, objects, or pixels. Background dominance and small animal regions
can make segmentation challenging, particularly when using an unweighted
pixel-wise loss.</li>
</ul>
""")


# %% [markdown] edition="own"
# ✍️ **Your answer** (double-click to edit):
#
# *In the park:* about 85% of all pixels were background, and birds made up about 1% of the pixels despite representing 25% of the bounding boxes.

# %% [markdown] edition="teaching"
# ### 🟢 Mask Size, Shape, and Fine Structures
#
# A segmentation mask describes the pixels occupied by an animal. Its **fill ratio** measures how much of its bounding box is covered by the mask:
#
# $$
# \text{Fill ratio} = \frac{\text{Mask area}}{\text{Bounding box area}}
# $$
#
# Masks also reveal fine structures, such as legs, tails, and beaks, which may be difficult to preserve when images are resized.
#
# **Before running the code:**
# - What proportion of its bounding box do you expect a typical animal to fill: about 25%, 50%, or 80%?
# - Which animal class do you expect to have the lowest fill ratio?
# - Which classes might have the most masks containing thin structures?
#
# **Explore:**
# - Compare the distributions of mask sizes and fill ratios across animal classes.
# - Inspect examples with low, typical, and high fill ratios.
# - Compare the proportion of masks with thin regions across classes, including and excluding small masks.
# - Inspect examples with particularly thin regions.
#
# **Observe and interpret:**
# - How do mask sizes and fill ratios differ between classes?
# - What characteristics of an animal might explain a low fill ratio?
# - Why can two animals with similar bounding box sizes have very different mask areas?
# - How does excluding small masks change the frequency of thin regions, and why?
# - What challenges might thin structures create when resizing masks?

# %% [markdown] edition="own"
# ### 🟢 Mask Size, Shape, and Fine Structures
#
# A segmentation mask describes the pixels occupied by an object. Its **fill ratio** measures how much of its bounding box is covered by the mask:
#
# $$
# \text{Fill ratio} = \frac{\text{Mask area}}{\text{Bounding box area}}
# $$
#
# Masks also reveal fine structures, such as thin parts, narrow edges, and small details, which may be difficult to preserve when images are resized.
#
# **Before running the code:**
# - What proportion of its bounding box do you expect a typical object to fill: about 25%, 50%, or 80%?
# - Which class do you expect to have the lowest fill ratio?
# - Which classes might have the most masks containing thin structures?
#
# **Explore:**
# - Compare the distributions of mask sizes and fill ratios across classes.
# - Inspect examples with low, typical, and high fill ratios.
# - Compare the proportion of masks with thin regions across classes, including and excluding small masks.
# - Inspect examples with particularly thin regions.
#
# **Observe and interpret:**
# - How do mask sizes and fill ratios differ between classes?
# - What characteristics of an object might explain a low fill ratio?
# - Why can two objects with similar bounding box sizes have very different mask areas?
# - How does excluding small masks change the frequency of thin regions, and why?
# - What challenges might thin structures create when resizing masks?

# %%
# @title 🔎 Explore Mask Sizes and Fill Ratios { display-mode: "form" }
# needs: masks
animals = masks[masks["iscrowd"] == 0].reset_index(drop=True)
_shape = pd.DataFrame([mask_shape(decode(r), b)
                       for r, b in zip(animals["rle"], animals["bbox"], strict=True)])
animals = pd.concat([animals, _shape], axis=1)
_photo_px = animals["image_id"].map(images_by_id["width"] * images_by_id["height"])
animals["photo_share"] = animals["mask_area"] / _photo_px
# The masks the quality checks run on, with park_masks' columns (the Teaching Edition plants five errors first).
masks_to_check = animals[["id", "image_id", "class", "bbox", "iscrowd", "rle", "mask_area"]]
_fill = animals.groupby("class")["fill"].median().reindex(CLASSES)
RESULTS["median_fill"] = {c: float(_fill[c]) for c in CLASSES}
print(f"{len(animals)} animal masks measured · median fill ratio " + " · ".join(
    f"{c} {_fill[c]:.0%}" for c in CLASSES))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, col, label, log in [(axes[0], "photo_share", "mask area, share of the image (log)", True),
                            (axes[1], "fill", "fill ratio = mask area / box area", False)]:
    data = [animals.loc[animals["class"] == c, col].clip(lower=1e-5).to_numpy() for c in CLASSES]
    if log:  # violins are drawn on log10(value) so the KDE is smooth on a log axis; ticks are relabeled
        data = [np.log10(v) for v in data]
    parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.8)
    for body, c in zip(parts["bodies"], CLASSES, strict=True):
        body.set_facecolor(CLASS_COLORS[c])
        body.set_alpha(0.75)
    parts["cmedians"].set_color(INK)
    ax.set_xticks(range(1, len(CLASSES) + 1), CLASSES)
    if log:
        _ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        ax.set_yticks(np.log10(_ticks), [f"{100 * t:g}%" for t in _ticks])
    else:
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_ylabel(label)
    for lab in ax.get_xticklabels():
        lab.set_color(CLASS_COLORS[lab.get_text()])
axes[0].set_title("How big is a mask?")
axes[1].set_title("How much of its box does it fill?")
fig.tight_layout()
display(fig)
plt.close(fig)


# %%

# @title 🔎 Inspect Mask Fill Ratios { display-mode: "form" }
# needs: masks
# Select examples, excluding very small masks
candidates = (
    animals[animals["mask_area"] >= MIN_JUDGED_AREA]
    .sort_values(["fill", "id"])
    .drop_duplicates("image_id")
    .reset_index(drop=True)
)

n_examples = min(20, len(candidates) // 3)

fill_examples = {
    "Low": candidates.head(n_examples).reset_index(drop=True),
    "Typical": candidates.iloc[
        (len(candidates) - n_examples) // 2:
        (len(candidates) + n_examples) // 2
    ].reset_index(drop=True),
    "High": candidates.tail(n_examples).reset_index(drop=True)
}

fill_category = widgets.ToggleButtons(
    options=["Low", "Typical", "High"],
    description="Fill ratio:"
)
fill_prev_btn = widgets.Button(description="Previous", icon="chevron-left")
fill_next_btn = widgets.Button(description="Next", icon="chevron-right")
fill_mask_toggle = widgets.Checkbox(value=True, description="Show masks", indent=False)
fill_box_toggle = widgets.Checkbox(value=True, description="Show boxes", indent=False)
fill_page_label = widgets.HTML()
fill_output = widgets.Output()

fill_page = 0

def fill_show_page():
    selected = fill_examples[fill_category.value]
    total_pages = max(1, (len(selected) + PAGE_SIZE - 1) // PAGE_SIZE)
    start = fill_page * PAGE_SIZE
    current = selected.iloc[start:start + PAGE_SIZE]

    fill_page_label.value = f"Page {fill_page + 1} of {total_pages}"
    fill_prev_btn.disabled = fill_page == 0
    fill_next_btn.disabled = fill_page >= total_pages - 1

    image_rows = images_by_id.loc[current["image_id"]].reset_index()
    fill_values = dict(zip(current["image_id"], current["fill"]))

    with fill_output:
        fill_output.clear_output(wait=True)
        show_masks(
            image_rows,
            masks,
            f"{fill_category.value} fill ratios",
            caption=lambda r: f"{r['class']} · fill {fill_values[r['id']]:.0%}",
            show_mask=fill_mask_toggle.value,
            show_box=fill_box_toggle.value,
            highlight=dict(zip(current["image_id"], current["id"]))
        )

def fill_navigate(direction):
    global fill_page
    total_pages = max(1, (len(fill_examples[fill_category.value]) + PAGE_SIZE - 1) // PAGE_SIZE)
    fill_page = max(0, min(fill_page + direction, total_pages - 1))
    fill_show_page()

def fill_change_category(change):
    global fill_page
    if change["name"] == "value":
        fill_page = 0
        fill_show_page()

fill_prev_btn.on_click(lambda b: fill_navigate(-1))
fill_next_btn.on_click(lambda b: fill_navigate(1))
fill_category.observe(fill_change_category, names="value")
fill_mask_toggle.observe(lambda change: fill_show_page(), names="value")
fill_box_toggle.observe(lambda change: fill_show_page(), names="value")

display(
    widgets.HBox([fill_category, fill_prev_btn, fill_next_btn, fill_page_label]),
    widgets.HBox([fill_mask_toggle, fill_box_toggle]),
    fill_output
)
fill_show_page()


# %%
# @title 🔎 Explore Thin Structures in Segmentation Masks { display-mode: "form" }
# needs: masks

THIN_SHARE = 0.05  # a mask "has thin parts" when at least this share of its pixels is thin
_big = animals["mask_area"] >= MIN_JUDGED_AREA
_has_thin = animals["thin"] >= THIN_SHARE
thin_by_class = pd.DataFrame({
    "all masks": _has_thin.groupby(animals["class"]).mean(),
    f"masks ≥ {int(MIN_JUDGED_AREA**0.5)}² px": _has_thin[_big].groupby(animals.loc[_big, "class"]).mean(),
}).reindex(CLASSES).fillna(0)
RESULTS["thin_mask_share"] = {c: float(thin_by_class.loc[c, "all masks"]) for c in CLASSES}
display((thin_by_class * 100).round(0).astype(int).astype(str).add(" %").T)

fig, ax = plt.subplots(figsize=(8, 3.2))
_x = np.arange(len(CLASSES))
for k, (col, alpha) in enumerate([(thin_by_class.columns[0], 1.0), (thin_by_class.columns[1], 0.45)]):
    ax.bar(_x + (k - 0.5) * 0.38, thin_by_class[col], width=0.38, alpha=alpha,
           color=[CLASS_COLORS[c] for c in CLASSES], edgecolor="white", label=col)
ax.set_xticks(_x, CLASSES)
for lab in ax.get_xticklabels():
    lab.set_color(CLASS_COLORS[lab.get_text()])
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set_title(f"Masks with ≥ {THIN_SHARE:.0%} of their pixels thinner than {THIN_PX} px: all (solid), "
             "large enough to judge (pale)", fontsize=10)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
display(fig)
plt.close(fig)

N_THIN_SHOW = 2 if SMOKE else 4  # photos with the most thin pixels to show (N-014)
animals["thin_px"] = animals["thin"] * animals["mask_area"]
_most_thin = animals.sort_values(["thin_px", "id"], ascending=[False, True]).drop_duplicates("image_id").head(
    N_THIN_SHOW)
show_masks(images_by_id.loc[list(_most_thin["image_id"])].reset_index(), masks,
           f"the images with the most thin pixels (dark = narrower than {THIN_PX} px)", mark_thin=True,
           cols=N_THIN_SHOW)

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul>
<li><strong>Mask size:</strong> Cats and dogs often occupy larger portions
of an image, while birds tend to have smaller masks.</li>

<li><strong>Fill ratio:</strong> A typical animal fills roughly half its
bounding box. Median fill ratios range from about 44% (birds) to
57% (cats). Shape, pose, and occlusion help explain these differences.
Animals with similar bounding boxes can therefore have very different
mask areas.</li>

<li><strong>Low-fill examples:</strong> Overlapping zebras and partly
occluded horses illustrate how the visible mask area can be much smaller
than the bounding box. Extreme values may also indicate annotation
errors, but are not necessarily incorrect.</li>

<li><strong>Thin structures:</strong> Birds have the highest proportion
of masks with thin regions (67%), but this drops to 14% when small masks
are excluded. Many birds are so small that much of the animal is only
a few pixels wide. For larger animals, such as horses, thin regions
often correspond to legs or other narrow body parts.</li>

<li><strong>Resizing:</strong> Thin structures may disappear when masks
are downsampled, especially with nearest-neighbor interpolation.
Both small objects and fine structures therefore require attention
during preprocessing.</li>
</ul>

<p><strong>Key insight:</strong> Mask geometry depends on both object
size and shape. Comparing all masks with sufficiently large masks
helps distinguish small-object effects from genuinely thin structures.</p>
""")


# %% [markdown] edition="own"
# ✍️ **Your answer** (double-click to edit):
#
# *In the park:* median fill ratios ranged from about 44% (birds) to 57% (cats), and 67% of bird masks had thin regions, falling to 14% once small masks were excluded.

# %% [markdown] edition="teaching"
# ### 🔴 Where Are the Animal Pixels? (Optional)
#
# Segmentation masks reveal not only how much of an image an animal occupies, but also where its pixels are located. By averaging masks across images, we can identify spatial patterns in the dataset.
#
# **Before running the code:**
# - Predict where animal pixels are most likely to occur in an image. Do you expect differences between classes?
#
# **Explore:**
# - Examine the heatmaps showing the average mask for each animal class. All masks are resized to a common grid before averaging.
# - Compare the spatial distributions and the average proportion of animal pixels across classes.
#
# **Observe and interpret:**
# - Where are animal pixels concentrated, and how do the patterns differ between classes?
# - What might explain these patterns?
# - How representative are these spatial distributions of images captured by fixed cameras at the Wildwood Farm Park?

# %% [markdown] edition="own"
# ### 🔴 Where Are the Object Pixels? (Optional)
#
# Segmentation masks reveal not only how much of an image an object occupies, but also where its pixels are located. By averaging masks across images, we can identify spatial patterns in your dataset.
#
# **Before running the code:**
# - Predict where object pixels are most likely to occur in an image. Do you expect differences between classes?
#
# **Explore:**
# - Examine the heatmaps showing the average mask for each class. All masks are resized to a common grid before averaging.
# - Compare the spatial distributions and the average proportion of object pixels across classes.
#
# **Observe and interpret:**
# - Where are object pixels concentrated, and how do the patterns differ between classes?
# - What might explain these patterns?
# - How representative are these spatial distributions of the images your model will see in use?

# %%
# @title 🔎 Explore Spatial Distributions of Animal Pixels { display-mode: "form" }
# needs: masks

# Mean mask per class on a common square: the share of the class's photos in which each spot is animal.
HEAT_RES = 24 if SMOKE else 48  # coarser grid with few photos (N-014)
_class_of = images_by_id["class"]
heat = {c: np.zeros((HEAT_RES, HEAT_RES)) for c in CLASSES}
for image_id, rle in photo_rle.items():
    small = cv2.resize(decode(rle).astype(np.float32), (HEAT_RES, HEAT_RES), interpolation=cv2.INTER_AREA)
    heat[_class_of[image_id]] += small

_ncols = min(6, len(CLASSES))  # at most six classes per row
_nrows = -(-len(CLASSES) // _ncols)
fig, axes = plt.subplots(_nrows, _ncols, figsize=(2.3 * _ncols, 2.8 * _nrows), squeeze=False)
for ax in axes.ravel()[len(CLASSES):]:
    ax.axis("off")
for ax, c in zip(axes.ravel(), CLASSES, strict=False):
    n_photos = max(int((_class_of == c).sum()), 1)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(c, ["white", CLASS_COLORS[c]])
    ax.imshow(heat[c] / n_photos, cmap=cmap, vmin=0, vmax=1, extent=[0, 1, 1, 0])
    ax.set_title(f"{c} · {heat[c].sum() / n_photos / HEAT_RES**2:.0%} of an image", fontsize=9,
                 color=CLASS_COLORS[c])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=7)
    ax.grid(False)
fig.suptitle("Where the animal pixels are (same scale for every class: white = never, full color = always)",
             color=INK)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul>
  <li><strong>Observation:</strong> Animal pixels are concentrated near
      the image center across all classes. Cats and dogs occupy
      relatively large central regions, while birds have a much
      lower pixel frequency throughout the image.</li>
  <li><strong>Interpretation:</strong> These patterns reflect differences
      in object size, framing, and typical image composition.
      They indicate a spatial bias in the dataset.</li>
  <li><strong>Implication:</strong> Animals recorded by fixed cameras
      at the Wildwood Farm Park may appear anywhere in the frame.
      Models trained on centrally positioned animals could exploit
      this bias and generalize less reliably to such images.</li>
</ul>
<p><strong>Limitation:</strong> The heatmaps show how frequently each
pixel location belongs to an animal, not where individual animals
always appear. Averaging also hides variation between images.</p>
""")


# %% [markdown] edition="own"
# ✍️ **Your answer** (double-click to edit):
#
# *In the park:* animal pixels were concentrated near the image center for every class, while the park's fixed cameras may see animals anywhere in the frame.

# %% [markdown] edition="teaching"
# ### 🔴 Mask Quality: Finding Annotation Errors (Optional)
#
# Segmentation masks can contain annotation errors that are difficult to identify by inspecting images individually. In this activity, we deliberately introduce five types of errors into copies of existing COCO masks. The modified annotations are stored in `park_masks`; the original `masks` remain unchanged.
#
# Your challenge is to identify suspicious annotations using automated quality checks.
#
# **Before running the code:**
# - Examine five types of annotation errors and five automated quality checks.
# - Predict which checks could detect each error. Can every error be detected reliably?
#
# **Explore:**
# - Compare the simulated errors in Picture A with the quality checks in Picture B.
# - Predict which checks will flag each error type.
# - Run the automated checks and inspect the strongest suspects.
# - Reveal the answer key to evaluate your predictions.
#
# **Observe and interpret:**
# - Which checks detect which types of errors, and why?
# - Which errors are difficult to detect automatically?
# - Why might a valid COCO mask be flagged as suspicious?

# %% [markdown] edition="own"
# ### 🔴 Mask Quality: Which of Your Masks Look Suspicious? (Optional)
#
# Segmentation masks can contain annotation errors that are hard to see when you inspect images one by one. Five automated checks help you decide which masks to look at first.
#
# **Explore:**
# 1. Study the five types of annotation errors and the five quality checks in the next two figures.
# 2. Run the five checks on your masks in 🔎 Detect Suspicious Segmentation Masks.
# 3. Inspect the strongest suspects one by one.
#
# **Observe and interpret:**
# - Which checks flag the most masks in your dataset?
# - For each suspect you inspected, is it an annotation error or a valid mask of an unusual object?
# - Which masks would you fix, and which would you keep?
#
# ✍️ **Your answer** (double-click to edit):
#
# *In the park:* besides the planted errors, the checks flagged original COCO masks that were valid but unusual, for example because of occlusion or complex shapes, so every suspect needed a look.

# %%
# @title Helpers: Mask Quality Checks { display-mode: "form" }
BOX_IOU_MIN = 0.9  # check 1: suspect when the mask's own box overlaps the annotated box less than this
FILL_RANGE = (0.15, 0.95)  # suspect when mask area / box area is outside this range
MAX_PIECES = 3  # suspect with more separate pieces than this
MAX_HOLES = 0  # suspect with more holes than this
SOLIDITY_MAX = 0.97  # suspect when the mask fills more than this share of its rubber band (no dents at all)


def _crude_polygon(mask: np.ndarray, max_vertices: int = 6) -> list[float]:
    """A polygon of at most `max_vertices` points around the mask (its convex hull, simplified)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.vstack(contours))
    eps = 0.02 * cv2.arcLength(hull, True)
    poly = cv2.approxPolyDP(hull, eps, True)
    while len(poly) > max_vertices:
        eps *= 1.5
        poly = cv2.approxPolyDP(hull, eps, True)
    return [float(v) for v in poly.reshape(-1)]


# %% edition="teaching"
# @title Helpers: Simulate Mask Annotation Errors { display-mode: "form" }

MASK_ERROR_KINDS = ["another animal", "chunk missing", "shifted", "swollen", "crude polygon"]
SHIFT_PX = 15


def _paste(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """`mask` placed at the same coordinates on an empty canvas of another photo's size."""
    out = np.zeros((height, width), np.uint8)
    h, w = min(height, mask.shape[0]), min(width, mask.shape[1])
    out[:h, :w] = mask[:h, :w]
    return out


def plant_mask_errors(animals_df: pd.DataFrame, table: pd.DataFrame,
                seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (park_masks, key): the masks of `animals_df` with one ugly mask of each of MASK_ERROR_KINDS.

    Each kind spoils a different photo's mask, picked in a seeded order among masks large enough to judge,
    and the mask is changed through pycocotools (decode → OpenCV → encode). With no mask fitting a kind's
    condition (a small subset), the first free one is used, so five masks are always planted.
    """
    rng = random.Random(seed)
    size = table.set_index("id")[["width", "height"]]
    pool = animals_df[animals_df["mask_area"] >= MIN_JUDGED_AREA].set_index("id")
    order = sorted(pool.index)
    rng.shuffle(order)
    used, new_rle, key = set(), {}, []

    def fits(kind: str, ann_id: int):
        """The spoiled mask as a 0/1 array (or an RLE), plus a note, or None when this mask does not suit."""
        a = pool.loc[ann_id]
        x, y, w, h = a["bbox"]
        W, H = int(size.at[a["image_id"], "width"]), int(size.at[a["image_id"], "height"])
        if kind == "another animal":  # prefer a neighbor in the same photo: a duplicated mask, as in COCO
            others = pool[pool["image_id"] != a["image_id"]].index
            same = pool[(pool["image_id"] == a["image_id"]) & (pool.index != ann_id)].index
            for b in [*sorted(same), *[i for i in order if i in others]]:
                donor = _paste(decode(pool.at[b, "rle"]), H, W)
                if donor.any() and box_iou_dice(cv2.boundingRect(donor), tuple(a["bbox"]))[0] < 0.3:
                    where = "next to it" if b in same else "from another photo"
                    return donor, f"the mask of a {pool.at[b, 'class']} {where}"
            return None
        mask = decode(a["rle"])
        if kind == "chunk missing":
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            r = float(dist.max())
            if r < max(15, 0.15 * a["mask_area"] ** 0.5):
                return None
            cy, cx = np.unravel_index(int(dist.argmax()), dist.shape)
            out = mask.copy()
            cv2.circle(out, (int(cx), int(cy)), int(0.8 * r), 0, -1)
            return out, f"a disc of radius {int(0.8 * r)} px cut out of the body"
        if kind == "shifted":
            if x + w + SHIFT_PX > W or y + h + SHIFT_PX > H or max(w, h) > 300:
                return None
            move = np.float32([[1, 0, SHIFT_PX], [0, 1, SHIFT_PX]])
            moved = cv2.warpAffine(mask, move, (W, H), flags=cv2.INTER_NEAREST)
            return moved, f"moved {SHIFT_PX} px right and down"
        if kind == "swollen":
            d = max(6, round(0.08 * (w * h) ** 0.5))
            if x < d or y < d or x + w + d > W or y + h + d > H:
                return None
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
            return cv2.dilate(mask, disc), f"grown by {d} px all round"
        if kind == "crude polygon":
            if a["solidity"] > 0.85:  # pick an animal with clear dents, so the shortcut shows
                return None
            poly = _crude_polygon(mask)
            return to_rle([poly], H, W), f"{len(poly) // 2} clicked points"
        raise ValueError(kind)

    for kind in MASK_ERROR_KINDS:
        free = [i for i in order if pool.at[i, "image_id"] not in used and i not in new_rle]
        spoiled = next(((i, out) for i in free if (out := fits(kind, i)) is not None), None)
        if spoiled is None:  # nothing fits the condition: give the first free mask another photo's mask
            i, j = free[0], free[1]
            height, width = decode(pool.at[i, "rle"]).shape
            pasted = _paste(decode(pool.at[j, "rle"]), height, width)
            spoiled = (i, (pasted, "another photo's mask (fallback)"))
        ann_id, (out, note) = spoiled
        new_rle[ann_id] = out if isinstance(out, dict) else encode(out)
        used.add(pool.at[ann_id, "image_id"])
        key.append({"kind": kind, "ann_id": int(ann_id), "image_id": int(pool.at[ann_id, "image_id"]),
                    "class": pool.at[ann_id, "class"], "how": note})
    park = animals_df[["id", "image_id", "class", "bbox", "iscrowd"]].copy()
    park["rle"] = [new_rle.get(i, r) for i, r in zip(animals_df["id"], animals_df["rle"], strict=True)]
    park["mask_area"] = [int(mask_utils.area(r)) for r in park["rle"]]
    return park, pd.DataFrame(key)


park_masks, mask_error_key = plant_mask_errors(animals, images, SEED)
print(f"park_masks: {len(park_masks)} animal masks on {park_masks['image_id'].nunique()} images. "
      "Five of them are ugly. Find them.")
masks_to_check = park_masks  # the masks the quality checks run on


# %% [markdown]
#  **Five simulated annotation errors** The following figure illustrates five possible annotation errors using a toy horse mask. Gray represents the original mask and bounding box (dashed); yellow represents the modified annotation.
#
# Look for differences in mask position, shape, coverage, and correspondence with the original horse.

# %%

# @title 🔎 Explore Five Types of Mask Annotation Errors { display-mode: "form" }

def toy_horse(height: int = 60, width: int = 84) -> np.ndarray:
    """A toy horse mask with an oval body and four legs."""
    yy, xx = np.mgrid[0:height, 0:width]
    horse = ((xx - 40) / 24) ** 2 + ((yy - 32) / 13) ** 2 <= 1
    for lx in (24, 32, 46, 54):
        horse |= (xx >= lx) & (xx < lx + 5) & (yy >= 38) & (yy < 56)
    return horse.astype(np.uint8)


horse = toy_horse()
kx, ky, kw, kh = cv2.boundingRect(horse)
yy, xx = np.mgrid[0:horse.shape[0], 0:horse.shape[1]]

poly = np.array(_crude_polygon(horse), np.int32).reshape(-1, 1, 2)
bite = ((xx - 40) ** 2 + (yy - 30) ** 2 <= 36).astype(np.uint8)
other = (((xx - 66) / 13) ** 2 + ((yy - 18) / 9) ** 2 <= 1).astype(np.uint8)

kinds = {
    "Swollen\n(drawn too generously)": cv2.dilate(horse, np.ones((7, 7), np.uint8)),
    "Shifted\n(correct shape, moved)": np.roll(horse, (4, 9), axis=(0, 1)),
    "Chunk missing\n(a piece cut out)": horse & (1 - bite),
    "Wrong animal\n(another mask pasted)": other,
    "Crude polygon\n(a few quick clicks)": cv2.fillPoly(np.zeros_like(horse), [poly], 1),
}

outline, _ = cv2.findContours(horse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

fig, axes = plt.subplots(1, 5, figsize=(13, 2.7))

for ax, (kind, spoiled) in zip(axes, kinds.items(), strict=True):
    # Original mask in gray
    base = np.zeros((*horse.shape, 4))
    base[horse > 0] = (*mpl.colors.to_rgb(GRAY), 0.35)
    ax.imshow(base)

    # Simulated annotation in yellow
    overlay = np.zeros((*horse.shape, 4))
    overlay[spoiled > 0] = (*mpl.colors.to_rgb(HIGHLIGHT), 0.75)
    ax.imshow(overlay)

    # Original outline
    for contour in outline:
        ax.plot(
            contour[:, 0, 0], contour[:, 0, 1],
            color=GRAY, lw=1.5
        )

    # Original bounding box
    ax.add_patch(mpl.patches.Rectangle(
        (kx - 0.5, ky - 0.5), kw, kh,
        fill=False, lw=1.5, ls="--", edgecolor=GRAY
    ))

    ax.set_xlim(-0.5, horse.shape[1] - 0.5)
    ax.set_ylim(horse.shape[0] - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(kind, fontsize=9)

fig.suptitle(
    "Picture A · Five simulated annotation errors",
    color=INK
)
fig.tight_layout()
display(fig)
plt.close(fig)


# %% [markdown]
# **Five automated mask quality checks.** The next figure compares a valid toy horse mask (top row) with a deliberately modified mask (bottom row). Each check measures a different property and flags potentially suspicious annotations.
#
# | Check | What it measures | Flagged when |
# |---|---|---|
# | **Mask–box agreement** | IoU between the annotated box and the tightest box around the mask | IoU &lt; 0.90 |
# | **Fill ratio** | Mask area divided by bounding box area | Outside 0.15–0.95 |
# | **Connected components** | Number of separate mask regions | More than 3 |
# | **Holes** | Enclosed background regions within the mask | At least 1 |
# | **Solidity** | Mask area divided by its convex hull area | Above 0.97 |
#
# A *convex hull* is the smallest convex shape enclosing the mask, similar to stretching a rubber band around it.
#
# These thresholds are heuristics: a flagged mask is worth inspecting, but is not necessarily incorrect.
#

# %%

# @title 🔎 Explore Five Mask Quality Checks { display-mode: "form" }

# Compare an original toy horse with a deliberately modified version.
H, W = 60, 84
yy, xx = np.mgrid[0:H, 0:W]

original = toy_horse(H, W)
hole = (xx - 40) ** 2 + (yy - 28) ** 2 <= 16
piece = (xx >= 68) & (xx < 76) & (yy >= 30) & (yy < 38)
modified = ((original.astype(bool) | piece) & ~hole).astype(np.uint8)

tx, ty, tw, th = cv2.boundingRect(modified)
rows = {
    "Original": (original, cv2.boundingRect(original)),
    "Modified": (modified, (tx + 5, ty + 3, tw, th))
}

CHECK_RULES = [
    ("1 · Mask–box agreement", f"suspect below {BOX_IOU_MIN}", "box_iou",
     lambda v: v < BOX_IOU_MIN, "IoU {:.2f}".format),
    ("2 · Fill", f"suspect outside {FILL_RANGE[0]}–{FILL_RANGE[1]}", "fill",
     lambda v: not FILL_RANGE[0] <= v <= FILL_RANGE[1], "fill {:.2f}".format),
    ("3 · Pieces", f"suspect above {MAX_PIECES}", "pieces",
     lambda v: v > MAX_PIECES, lambda v: f"{v} piece" + "s" * (v != 1)),
    ("4 · Holes", "suspect with any hole", "holes",
     lambda v: v > MAX_HOLES, lambda v: f"{v} hole" + "s" * (v != 1)),
    ("5 · Solidity", f"suspect above {SOLIDITY_MAX}", "solidity",
     lambda v: v > SOLIDITY_MAX, "solidity {:.2f}".format),
]

UNDER = {
    0: "solid: tightest box around mask\ndashed: annotated box",
    4: "dashed: convex hull"
}

def mask_layer(mask, color, alpha):
    layer = np.zeros((H, W, 4))
    layer[mask > 0] = (*mpl.colors.to_rgb(color), alpha)
    return layer

fig, axes = plt.subplots(2, 5, figsize=(13, 6.2))

for r, (row_name, (mask, box)) in enumerate(rows.items()):
    shape = mask_shape(mask, box)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hull = cv2.convexHull(np.vstack(contours))[:, 0, :]

    n_parts, parts = cv2.connectedComponents(mask, connectivity=8)
    main = max(range(1, n_parts), key=lambda k: int((parts == k).sum()))
    own = cv2.boundingRect(mask)

    for k, (name, rule, key, is_suspect, fmt) in enumerate(CHECK_RULES):
        ax = axes[r, k]

        # Original reference in gray, evaluated mask in yellow
        ax.imshow(mask_layer(original, GRAY, 0.35))
        ax.imshow(mask_layer(mask, HIGHLIGHT, 0.8))

        if k == 0:
            ax.add_patch(mpl.patches.Rectangle(
                (own[0] - 0.5, own[1] - 0.5), own[2], own[3],
                fill=False, lw=2, edgecolor=DARK_GRAY
            ))

        if k in (0, 1):
            ax.add_patch(mpl.patches.Rectangle(
                (box[0] - 0.5, box[1] - 0.5), box[2], box[3],
                fill=False, lw=2, ls="--", edgecolor=GRAY
            ))

        if k == 2:
            ax.imshow(mask_layer(
                (parts > 0) & (parts != main), DARK_GRAY, 1.0
            ))

        if k == 3 and row_name == "Modified":
            ax.imshow(mask_layer(hole, DARK_GRAY, 1.0))

        if k == 4:
            ax.add_patch(mpl.patches.Polygon(
                hull, closed=True, fill=False, lw=2,
                ls="--", edgecolor=DARK_GRAY
            ))

        value = shape[key]
        bad = is_suspect(value)

        ax.text(
            0.02, 0.98,
            "SUSPECT" if bad else "normal",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=13, fontweight="bold",
            color=DARK_GRAY
        )

        ax.text(
            0.98, 0.97, fmt(value),
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9, color=DARK_GRAY
        )

        if r == 0:
            ax.set_title(name, fontsize=12, color=INK, pad=10)
        else:
            under = rule + (f"\n\n{UNDER[k]}" if k in UNDER else "")
            ax.set_xlabel(under, fontsize=9, color=INK)

        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H + 3.5, -14.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    axes[r, 0].set_ylabel(row_name, fontsize=12, color=INK)

fig.suptitle(
    "Picture B · Five mask quality checks: original (top) vs. modified (bottom)",
    color=INK, fontsize=13
)
fig.tight_layout()
display(fig)
plt.close(fig)


# %% [markdown] edition="teaching"
# **Predict and investigate**
#
# Match each of the five annotation errors in Picture A to the quality check(s) in Picture B that you expect to flag it. Write five pairs in the form *error type → check number(s)*. An error may trigger multiple checks, and some checks may detect none.
#
# Then run the automated investigation:
#
# 1. **Counts:** Compare how many masks each check flags.
# 2. **Suspect gallery:** Inspect the highlighted mask in each image. Faded masks represent other animals. Predict whether each suspect is a simulated error or an original COCO annotation.
# 3. **Answer key:** Reveal the planted errors and compare the results with your predictions.

# %%
# @title 🔎 Detect Suspicious Segmentation Masks { display-mode: "form" }
# needs: masks

_judge = masks_to_check[masks_to_check["mask_area"] >= MIN_JUDGED_AREA].reset_index(drop=True)
park_shape = pd.concat([_judge, pd.DataFrame([mask_shape(decode(r), b)
                                              for r, b in zip(_judge["rle"], _judge["bbox"], strict=True)])],
                       axis=1)


def run_checks(shape: pd.DataFrame, box_iou_min: float) -> dict[str, pd.Series]:
    """The five checks as {name: True where a mask is a suspect}; only check 1's threshold is a parameter.

    e.g. run_checks(park_shape, 0.9)["holes"] → a True/False Series, one value per judged mask
    """
    return {
        "box IoU": shape["box_iou"] < box_iou_min,
        "fill": (shape["fill"] < FILL_RANGE[0]) | (shape["fill"] > FILL_RANGE[1]),
        "pieces": shape["pieces"] > MAX_PIECES,
        "holes": shape["holes"] > MAX_HOLES,
        "solidity": shape["solidity"] > SOLIDITY_MAX,
    }


check_hits = run_checks(park_shape, BOX_IOU_MIN)
park_shape["checks"] = [", ".join(name for name, hit in check_hits.items() if hit.iloc[i])
                        for i in range(len(park_shape))]
mask_suspects = park_shape[park_shape["checks"] != ""].copy()
mask_suspects["n_checks"] = mask_suspects["checks"].str.count(",") + 1
mask_suspects = mask_suspects.sort_values(["n_checks", "box_iou", "id"], ascending=[False, True, True])

print(f"{len(_judge)} masks judged · {len(mask_suspects)} suspects "
      f"({len(mask_suspects) / max(len(_judge), 1):.1%}). Suspects per check:")
for name, hit in check_hits.items():
    print(f"  {name:9s} {int(hit.sum())}")
_cols = ["id", "image_id", "class", "checks", "box_iou", "fill", "pieces", "holes", "solidity"]
display(mask_suspects[_cols].head(12).round(2).set_index("id"))

N_SUSPECTS_SHOW = 4 if SMOKE else 8  # suspect photos to show (N-014)
_show = mask_suspects.drop_duplicates("image_id").head(N_SUSPECTS_SHOW).set_index("image_id")


def _suspect_caption(r) -> str:
    s = _show.loc[r["id"]]
    return (f"image {r['id']} · {s['class']} · {s['checks']}\n"
            f"box IoU {s['box_iou']:.2f} · fill {s['fill']:.2f}")


show_masks(images_by_id.loc[list(_show.index)].reset_index(), masks_to_check,
           f"the {len(_show)} strongest suspects: bright mask + solid box = suspect, faint = other animals",
           caption=_suspect_caption, highlight=dict(zip(_show.index, _show["id"], strict=True)))

# %% edition="teaching"

# @title 💡 Sample Answer { display-mode: "form" }

from html import escape

# Identify which planted errors were detected.
_checks_by_id = park_shape.set_index("id")["checks"]

mask_error_key["checks"] = mask_error_key["ann_id"].map(_checks_by_id).fillna("")
mask_error_key["found"] = mask_error_key["checks"] != ""

n_errors_found = int(mask_error_key["found"].sum())

RESULTS["hunt9"] = {
    "planted": len(mask_error_key),
    "found": n_errors_found,
}
RESULTS["mask_suspects_real"] = int(
    (~mask_suspects["id"].isin(mask_error_key["ann_id"])).sum()
)

# Prepare the detection summary.
_lines = [
    f"{'✓' if k['found'] else '✗'} {k['kind']}: "
    f"{k['checks'] or 'no check fired'}"
    for _, k in mask_error_key.iterrows()
]

_lines.append(
    f"Detected {n_errors_found} of {len(mask_error_key)} planted errors; "
    f"{RESULTS['mask_suspects_real']} original COCO annotations "
    "were also flagged."
)

# Compare original and modified annotations.

fig, axes = plt.subplots(
    2,
    len(mask_error_key),
    figsize=(3.2 * len(mask_error_key), 6.2),
    squeeze=False,
)

for col, (_, k) in enumerate(mask_error_key.iterrows()):
    photo = images_by_id.loc[k["image_id"]]
    thumb = load_thumb(photo["path"], (320, 320))

    sx = thumb.width / photo["width"]
    sy = thumb.height / photo["height"]

    for row, source in enumerate([masks, park_masks]):
        ax = axes[row, col]
        color = GRAY if row == 0 else HIGHLIGHT

        annotation = source.loc[
            source["id"] == k["ann_id"]
        ].iloc[0]

        full = decode(annotation["rle"])
        small = cv2.resize(
            full,
            thumb.size,
            interpolation=cv2.INTER_NEAREST,
        ) > 0

        overlay = np.zeros((thumb.height, thumb.width, 4))
        overlay[small] = (*mpl.colors.to_rgb(color), 0.6)

        ax.imshow(thumb)
        ax.imshow(overlay)

        x, y, w, h = annotation["bbox"]
        ax.add_patch(
            mpl.patches.Rectangle(
                (x * sx, y * sy),
                w * sx,
                h * sy,
                fill=False,
                lw=1.5,
                ls="--",
                edgecolor=color,
            )
        )

        frame(ax, color)

        ax.set_title(
            f"{k['class']} · original"
            if row == 0
            else f"{k['kind']}\n{k['checks'] or 'no check fired'}",
            fontsize=9,
            color=INK,
        )

fig.suptitle(
    "Five planted errors: original masks (top) vs. modified masks (bottom)",
    color=INK,
    fontsize=13,
)
fig.tight_layout()

_items = "".join(f"<li>{escape(line)}</li>" for line in _lines)

# The complete answer inside one collapsible gray box; the figure goes where the body says <!-- fig -->.
callout("answer", "💡 Sample Answer", f"""
<p><strong>Which errors were detected?</strong></p>
<ul>{_items}</ul>

<!-- fig -->

<p><strong>Why did these checks detect the errors?</strong></p>
<ul>
  <li><strong>Wrong animal → box IoU:</strong>
      The pasted mask belongs to a different animal and no longer
      aligns with the original bounding box.</li>
  <li><strong>Shifted mask → box IoU:</strong>
      Moving the mask reduces its overlap with the annotated box.</li>
  <li><strong>Swollen mask → box IoU, possibly fill:</strong>
      Enlarging the mask changes its bounding box and may increase
      its fill ratio.</li>
  <li><strong>Missing region → holes:</strong>
      The simulated cut-out creates an enclosed background region
      inside the mask.</li>
  <li><strong>Crude polygon → solidity:</strong>
      Simplifying the shape removes concavities, making the mask
      more similar to its convex hull.</li>
</ul>

<p><strong>What about the other masks?</strong></p>
<ul>
  <li>The connected-components check may detect none of the planted
      errors, but can flag animals whose visible regions are separated
      by occlusion.</li>
  <li>Some flagged original COCO masks are valid but unusual,
      for example because of occlusion or complex shapes.</li>
  <li>Others may contain genuine annotation errors, such as inaccurate
      boundaries or missing regions. Not every flagged original mask
      is therefore a false alarm.</li>
  <li>Some incorrect masks may pass all checks. Detection results
      also depend on the thresholds used.</li>
</ul>

<p><strong>Key insight:</strong> Automated quality checks identify
potential geometric inconsistencies, not necessarily incorrect
annotations. Combining checks helps prioritize visual inspection,
but cannot replace it.</p>
""", fig=fig)

