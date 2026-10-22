# %% [markdown]
# ## 9. Same data, masks
#
# **Objective:** see what imbalance, object size, position and thin parts mean once the park wants an
# *outline* of every animal, and find five ugly masks hidden among the real ones.
#
# The park's third job, **outline** each animal to see how much of the enclosure it uses, needs a **mask**:
# for every animal, the exact pixels it covers. `annotations` already holds one in its `segmentation` column,
# either as a COCO *polygon* (the x, y points an annotator clicked around the animal) or, for crowds, as
# *RLE* (run-length encoding: "300 background pixels, then 25 animal pixels, then …"). `pycocotools`, the
# COCO team's own library, turns both into a 0/1 array the size of the photo. This section keeps the photos
# of `clean_images` and counts what is inside those arrays.
#
# <table width="100%"><tr><td bgcolor="#E3F2FD"><font color="#0D47A1">
# <b>📘 Theory · What changes when the unit is a pixel</b><br>
# A segmentation model makes one decision per <b>pixel</b>: which class is this pixel, or is it background?
# Four things follow:
# <ul>
# <li><b>Imbalance is counted in pixels, and background is a class too.</b> Most of every photo is grass,
# sky and water, and a big horse outweighs a whole flock of far-away birds.</li>
# <li><b>Size is everything.</b> A bird of 10×10 px is 100 decisions; a horse filling the frame is
# 100,000. The loss hears the horse a thousand times louder.</li>
# <li><b>Thin parts</b> (legs, tails, beaks, a bird far away) are few pixels, and they are the first to
# vanish when the photo is shrunk to the network's input size.</li>
# <li><b>Mask quality is hard to see.</b> A box is four numbers; a mask is dozens of clicks, so masks are
# sloppier, and a bad one looks like any other row in a table.</li>
# </ul>
# </font></td></tr></table>
#
# The collapsed cell defines the mask tools: decode and encode RLE, measure a mask's shape, and `show_masks`,
# a gallery that paints each mask in its class colour over the photo, with its annotated box dashed.

# %%
# @title Mask helpers (decode, encode, measure, draw) { display-mode: "form" }
import cv2
from pycocotools import mask as mask_utils

MIN_JUDGED_AREA = 32**2  # masks under COCO's "small" limit (32×32 px) are too few pixels to judge their shape
THIN_PX = 7  # parts narrower than this many photo pixels count as thin (about 2 px at 224×224)
BLOB_SHARE = 0.005  # pieces and holes under 0.5 % of the mask (or 16 px) are rasterisation noise, not shape


def to_rle(segmentation, height: int, width: int) -> dict:
    """A COCO segmentation (polygon list, uncompressed or compressed RLE) as one compressed RLE."""
    if isinstance(segmentation, list):
        return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    if isinstance(segmentation["counts"], list):
        return mask_utils.frPyObjects(segmentation, height, width)
    return segmentation


def decode(rle: dict) -> np.ndarray:
    """RLE → 0/1 uint8 array of the photo's size (C-ordered, so OpenCV accepts it)."""
    return np.ascontiguousarray(mask_utils.decode(rle))


def encode(mask: np.ndarray) -> dict:
    """0/1 array → compressed RLE, the way COCO stores masks."""
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


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
               size: int = 384, mark_thin: bool = False) -> None:
    """Gallery of the photos in `rows_df` with every mask of `masks_df` (column `rle`) painted in its class
    colour and its annotated box dashed. `mark_thin=True` paints the thin parts dark.

    `caption(row) -> str` writes the title above each photo; by default the class and the number of masks.
    """
    n = len(rows_df)
    if n == 0:
        print(f"{title}: no photos to show.")
        return
    cols = min(cols, n)
    n_rows = -(-n // cols)
    fig, axes = plt.subplots(n_rows, cols, figsize=(3.4 * cols, 3.2 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), rows_df.iterrows(), strict=False):
        thumb = load_thumb(row["path"], size)
        scale = thumb.width / row["width"]
        overlay = np.zeros((thumb.height, thumb.width, 4))
        own = masks_df[masks_df["image_id"] == row["id"]]
        for _, m in own.iterrows():
            full = decode(m["rle"])
            small = cv2.resize(full, thumb.size, interpolation=cv2.INTER_NEAREST) > 0  # nearest: section 3
            overlay[small] = (*mpl.colors.to_rgb(CLASS_COLORS[m["class"]]), 0.6)
            if mark_thin:
                thin = cv2.resize(thin_part(full), thumb.size, interpolation=cv2.INTER_NEAREST) > 0
                overlay[thin] = (*mpl.colors.to_rgb(INK), 0.95)
        ax.axis("on")
        ax.imshow(thumb)
        ax.imshow(overlay)
        for _, m in own.iterrows():
            x, y, w, h = m["bbox"]
            ax.add_patch(mpl.patches.Rectangle((x * scale, y * scale), w * scale, h * scale, fill=False,
                                               lw=1.5, ls="--", edgecolor=CLASS_COLORS[m["class"]]))
        frame(ax, CLASS_COLORS[row["class"]])
        ax.set_title(caption(row) if caption else f"{row['class']} · {len(own)} masks", fontsize=9)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    display(fig)
    plt.close(fig)


# %% [markdown]
# The next cell turns every mask of `clean_images` into one compressed RLE (the `rle` column of a new table,
# `masks`) and counts its pixels. RLE is small and fast: the union and the area of masks are computed
# without ever building the full 0/1 array.

# %%
# One row per mask of the photos in clean_images (crowd regions included), as RLE, with its pixel count.
_size = clean_images.set_index("id")[["width", "height"]]
masks = boxes[["id", "image_id", "class", "bbox", "iscrowd", "segmentation"]].copy()
masks["rle"] = [to_rle(seg, int(_size.at[i, "height"]), int(_size.at[i, "width"]))
                for seg, i in zip(masks["segmentation"], masks["image_id"], strict=True)]
masks["mask_area"] = [int(mask_utils.area(r)) for r in masks["rle"]]
masks = masks.drop(columns="segmentation")
print(f"{len(masks)} masks on {masks['image_id'].nunique()} photos "
      f"({int(masks['iscrowd'].sum())} of them crowd regions: one mask for a whole group)")
masks.head(3)

# %% [markdown]
# ### Pixels per class, background included
#
# <table width="100%"><tr><td bgcolor="#E0F7F4"><font color="#00564D">
# <b>❓ Questions</b><br>
# Section 4 counted photos and section 8 counted boxes. A segmentation model counts pixels, and every pixel
# that is not an animal is <i>background</i>, which is a class like any other for the model.
# </font></td></tr></table>
#
# **Q26.** 🟢 What share of all pixels in the Play Dataset do you expect to be background: about a quarter,
# a half, or most of them? And will bird be frequent or rare by pixels? Write down both guesses, then run the
# cell: it merges each photo's masks into one (a pixel belongs to at most one animal) and counts.

# %%
# Q26  Pixels per class including background (shares of all pixels), next to the photo and box shares.
photo_rle = {}  # one merged mask per photo, reused by the heatmaps below
_animal_px = dict.fromkeys(CLASSES, 0)
for image_id, own in masks.groupby("image_id"):
    photo_rle[image_id] = mask_utils.merge(list(own["rle"]), intersect=False)
    _animal_px[own["class"].iloc[0]] += int(mask_utils.area(photo_rle[image_id]))
_total_px = int((clean_images["width"] * clean_images["height"]).sum())
pixels_per_class = {c: _animal_px[c] / _total_px for c in CLASSES}
pixels_per_class["background"] = (_total_px - sum(_animal_px.values())) / _total_px
RESULTS["pixels_per_class"] = pixels_per_class

BACKGROUND_COLOR = "#D9DEDF"  # neutral grey: background is not an animal, so it gets no class colour
_units = {
    "photos": {c: photos_per_class[c] / sum(photos_per_class.values()) for c in CLASSES},
    "boxes": {c: boxes_per_class[c] / sum(boxes_per_class.values()) for c in CLASSES},
    "pixels": {c: _animal_px[c] / max(sum(_animal_px.values()), 1) for c in CLASSES},
}
print(f"{_total_px / 1e6:.0f} million pixels · background {pixels_per_class['background']:.1%} · "
      f"all animals together {1 - pixels_per_class['background']:.1%}")
print("share of ALL pixels:  " + " · ".join(f"{c} {pixels_per_class[c]:.1%}" for c in CLASSES))
print("among animals only, by photos / boxes / pixels:")
for c in CLASSES:
    print(f"  {c:6s} {_units['photos'][c]:5.0%}  {_units['boxes'][c]:5.0%}  {_units['pixels'][c]:5.0%}")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
ax = axes[0]
_x = np.arange(len(CLASSES))
for k, (unit, alpha) in enumerate([("photos", 0.3), ("boxes", 0.6), ("pixels", 1.0)]):
    ax.bar(_x + (k - 1) * 0.27, [_units[unit][c] for c in CLASSES], width=0.27, alpha=alpha,
           color=[CLASS_COLORS[c] for c in CLASSES], edgecolor="white", label=unit)
ax.set_xticks(_x, CLASSES)
for lab in ax.get_xticklabels():
    lab.set_color(CLASS_COLORS[lab.get_text()])
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set_title("Share among the animals: photos (pale) · boxes · pixels (solid)")
_legend = [mpl.patches.Patch(color=INK, alpha=a, label=u) for u, a in [("photos", 0.3), ("boxes", 0.6),
                                                                         ("pixels", 1.0)]]
ax.legend(handles=_legend, frameon=False, fontsize=8)

ax = axes[1]
_left = 0.0
for name in ["background", *CLASSES]:
    share = pixels_per_class[name]
    ax.barh(0, share, left=_left, color=BACKGROUND_COLOR if name == "background" else CLASS_COLORS[name],
            edgecolor="white")
    if share > 0.06:
        ax.text(_left + share / 2, 0, f"{name}\n{share:.0%}", ha="center", va="center", fontsize=8, color=INK)
    _left += share
ax.set_xlim(0, 1)
ax.set_yticks([])
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set_title("All pixels, background included")
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# 💡 Background dwarfs everything: a model that answers "background" for every pixel is right most of the
# time and outlines nothing, the pixel version of section 4's "always sheep". Among the animals, the solid
# bars reorder the classes once more: cat and dog come back, because a pet photo is a close-up.
#
# **Worked answer (Q26).** Most of them: on the full set about 85 % of all pixels are background (the smoke
# subset gives about 80 %). Bird is rare again: first by boxes in section 8, but about 1 % of all pixels
# (6–7 % of the animal pixels), because most birds are far away. Photos, boxes, pixels: the
# same dataset gives three rankings, which is why "bird is rare" needs its unit every time.
#
# ### How big is a mask, and how much of its box does it fill?
#
# A mask always sits inside its box, so a second number comes for free: the **fill ratio**, mask area
# divided by box area. A box around a curled-up cat is mostly cat; a box around a horse is mostly the air
# between its legs. The next cell measures every mask's shape (it decodes each one, so it takes a few
# seconds) and plots mask size and fill ratio per class. Crowd regions are left out: they outline a group,
# not an animal.
#
# **Q27.** 🟡 What fraction of its box does a typical animal fill: about 25 %, 50 % or 80 %? Which class do
# you expect to fill its box least? Guess, then run.

# %%
# Q27  Shape numbers of every (non-crowd) mask, then mask area and fill ratio per class.
animals = masks[masks["iscrowd"] == 0].reset_index(drop=True)
_shape = pd.DataFrame([mask_shape(decode(r), b)
                       for r, b in zip(animals["rle"], animals["bbox"], strict=True)])
animals = pd.concat([animals, _shape], axis=1)
_photo_px = animals["image_id"].map(_size["width"] * _size["height"])
animals["photo_share"] = animals["mask_area"] / _photo_px
_fill = animals.groupby("class")["fill"].median().reindex(CLASSES)
RESULTS["median_fill"] = {c: float(_fill[c]) for c in CLASSES}
print(f"{len(animals)} animal masks measured · median fill ratio " + " · ".join(
    f"{c} {_fill[c]:.0%}" for c in CLASSES))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, col, label, log in [(axes[0], "photo_share", "mask area, share of the photo (log)", True),
                            (axes[1], "fill", "fill ratio = mask area / box area", False)]:
    data = [animals.loc[animals["class"] == c, col].clip(lower=1e-5).to_numpy() for c in CLASSES]
    parts = ax.boxplot(data, tick_labels=CLASSES, patch_artist=True, showfliers=not log, widths=0.6,
                       flierprops={"markersize": 3, "markeredgecolor": INK})
    for patch, c in zip(parts["boxes"], CLASSES, strict=True):
        patch.set_facecolor(CLASS_COLORS[c])
        patch.set_alpha(0.75)
    for med in parts["medians"]:
        med.set_color(INK)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=2))
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

N_EXTREME_MASKS = 2 if SMOKE else 3  # photos with the lowest / highest fill ratio to show (N-014)
_judged = animals[animals["mask_area"] >= MIN_JUDGED_AREA].sort_values(["fill", "id"])
_ends = pd.concat([_judged.drop_duplicates("image_id").head(N_EXTREME_MASKS),
                   _judged.drop_duplicates("image_id", keep="last").tail(N_EXTREME_MASKS)])
_fill_of = dict(zip(_ends["image_id"], _ends["fill"], strict=True))
show_masks(clean_images.set_index("id").loc[list(_ends["image_id"])].reset_index(), masks,
           f"the {N_EXTREME_MASKS} lowest and the {N_EXTREME_MASKS} highest fill ratios",
           caption=lambda r: f"{r['class']} · fill {_fill_of[r['id']]:.0%}", cols=2 * N_EXTREME_MASKS)

# %% [markdown]
# 💡 Fill ratios sit in a narrow band, roughly 40–60 % for every class: animals are not rectangles. That
# band is what makes the fill ratio a quality check. A mask that fills 98 % of its box is a rectangle, not an
# animal, and one that fills 5 % has lost most of the animal.
#
# **Worked answer (Q27).** About half: on the full set the median fill is near 45 % for bird and horse
# (wings, legs, necks) and near 55 % for cat and sheep (compact shapes). The low end of the gallery shows
# what drives it down (spread wings, a diagonal pose); the high end shows close-ups cut by the photo's edge.
# The size plot repeats section 8 in pixels: birds are a sliver of the photo, cats and dogs a large part.
#
# ### Where the animal pixels are: one heatmap per class
#
# Section 8 drew where the box *centres* are. A mask says more: which pixels of the frame are animal. The
# next cell shrinks every photo's merged mask to the same small square and averages them per class, so a
# dark spot means "in many of this class's photos, this spot is animal".

# %%
# Mean mask per class on a common square: the share of the class's photos in which each spot is animal.
HEAT_RES = 24 if SMOKE else 48  # coarser grid with few photos (N-014)
_class_of = clean_images.set_index("id")["class"]
heat = {c: np.zeros((HEAT_RES, HEAT_RES)) for c in CLASSES}
for image_id, rle in photo_rle.items():
    small = cv2.resize(decode(rle).astype(np.float32), (HEAT_RES, HEAT_RES), interpolation=cv2.INTER_AREA)
    heat[_class_of[image_id]] += small

fig, axes = plt.subplots(1, len(CLASSES), figsize=(2.3 * len(CLASSES), 2.8))
for ax, c in zip(axes, CLASSES, strict=True):
    n_photos = max(int((_class_of == c).sum()), 1)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(c, ["white", CLASS_COLORS[c]])
    ax.imshow(heat[c] / n_photos, cmap=cmap, vmin=0, vmax=1, extent=[0, 1, 1, 0])
    ax.set_title(f"{c} · {heat[c].sum() / n_photos / HEAT_RES**2:.0%} of a photo", fontsize=9,
                 color=CLASS_COLORS[c])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=7)
    ax.grid(False)
fig.suptitle("Where the animal pixels are (same scale for every class: white = never, full colour = always)",
             color=INK)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# 💡 Every class is darkest in the centre, but by very different amounts: cat and dog spread a large blob
# over the middle and lower half (close-ups), while bird stays pale everywhere because its pixels are few and
# scattered. A model trained on this learns that the edges of the frame are background, which the paddock
# camera will contradict every day.
#
# ### Thin parts
#
# A leg, a tail, a beak or a far-away bird is only a few pixels wide. When the photo is shrunk to a network's
# input (say 224×224, about a third of a COCO photo), a part of 7 px becomes 2 px, and a mask resized with
# nearest neighbour (section 3) can lose it altogether. The next cell marks, in every mask, the pixels in
# parts narrower than `THIN_PX` = 7 px.
#
# **Q28.** 🟡 Which class has the most masks with thin parts, and is it for the same reason as the
# runner-up? Pick one, then run.

# %%
# Q28  Share of each class's masks with at least 5 % of their pixels in thin parts, and the thinnest masks.
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
show_masks(clean_images.set_index("id").loc[list(_most_thin["image_id"])].reset_index(), masks,
           f"the photos with the most thin pixels (dark = narrower than {THIN_PX} px)", mark_thin=True,
           cols=N_THIN_SHOW)

# %% [markdown]
# 💡 Thin pixels are a tiny share of all pixels, so a pixel-counting loss barely notices when it gets them
# wrong, and IoU barely moves. Yet they are what makes an outline an outline: the legs tell a horse from a
# rock. Look at the gallery: the dark pixels are legs, tails and ears.
#
# **Worked answer (Q28).** Bird, by far, on all masks (about two thirds on the full set), but not because
# birds have long legs: most birds are so small that the *whole bird* is a thin part. Among the masks large
# enough to judge, horse leads (legs), with bird, sheep and zebra next, and cat almost never has one. Two
# different problems hide behind one number: tiny objects and thin parts of big ones.
#
# <table width="100%"><tr><td bgcolor="#EDE7F6"><font color="#311B92">
# <b>🔀 Across tasks · What handles each of these later (next week and beyond)</b><br>
# The same vocabulary as section 8, now for pixels:
# <ul>
# <li><b>Pixel imbalance</b> (background ~85 %, bird ~1 %) → a <b>class-weighted</b> cross-entropy or a
# <b>Dice loss</b>, which scores each class's overlap on its own scale, and reporting <b>mean IoU over
# classes</b> (section 7), so that birds count as much as background.</li>
# <li><b>Tiny animals and thin parts</b> → a higher input resolution, or <b>crops / tiles</b> of the photo,
# so that a 7 px leg is still 7 px for the network; boundary-aware losses weigh the outline more.</li>
# <li><b>Everything in the centre</b> → random crops and shifts when training, so the edge of the frame is
# not always background.</li>
# </ul>
# None of these fixes a mask that is wrong. That is the data side, and the next part is about it.
# </font></td></tr></table>
#
# ### Guided hunt: five ugly masks
#
# The park paid an annotation team per mask, and the masks came back. In the Play Dataset they are `masks`;
# the batch the team sent back is `park_masks`, built by the collapsed cell: `masks` with **five masks
# spoiled at runtime**, one of each kind an annotation team really delivers:
# <ul>
# <li><b>swollen</b>: the outline drawn generously around the animal (the mask grown by a few pixels);</li>
# <li><b>shifted</b>: the right outline, about 15 px off to the side;</li>
# <li><b>a chunk missing</b>: a piece of the body cut out;</li>
# <li><b>another animal's mask</b>: the outline of a different animal pasted onto this one;</li>
# <li><b>a crude polygon</b>: an annotator in a hurry clicked five or six points around the animal.</li>
# </ul>
# Which masks, of which photos, is decided at runtime from `SEED`. They are hidden among hundreds of real
# masks, and you will not look at hundreds of photos.

# %%
# @title Plant the hunt (five ugly masks, seeded) { display-mode: "form" }
HUNT9_KINDS = ["another animal", "chunk missing", "shifted", "swollen", "crude polygon"]
SHIFT_PX = 15


def _paste(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """`mask` placed at the same coordinates on an empty canvas of another photo's size."""
    out = np.zeros((height, width), np.uint8)
    h, w = min(height, mask.shape[0]), min(width, mask.shape[1])
    out[:h, :w] = mask[:h, :w]
    return out


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


def plant_hunt9(animals_df: pd.DataFrame, table: pd.DataFrame,
                seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (park_masks, key): the masks of `animals_df` with one ugly mask of each of HUNT9_KINDS.

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
        if kind == "another animal":  # prefer a neighbour in the same photo: a duplicated mask, as in COCO
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

    for kind in HUNT9_KINDS:
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


park_masks, HUNT9_KEY = plant_hunt9(animals, clean_images, SEED)
print(f"park_masks: {len(park_masks)} animal masks on {park_masks['image_id'].nunique()} photos. "
      "Five of them are ugly. Find them.")

# %% [markdown]
# You have four checks, all computed from the mask and its annotated box, no photo needed:
# <ul>
# <li><b>mask–box agreement</b>: the IoU (section 7) between the box drawn around the mask and the box the
# annotator drew. On real COCO masks it is almost always above 0.95, because the box was made from the
# mask;</li>
# <li><b>fill ratio</b>: mask area / box area, from Q27;</li>
# <li><b>pieces and holes</b>: an animal is usually one piece (two or three when something stands in front
# of it) and almost never has a hole;</li>
# <li><b>solidity</b>: mask area / area of a rubber band stretched around it. Animals have dents (between
# the legs, under the neck), so it is rarely close to 1.</li>
# </ul>
# Only masks of at least 32×32 px are judged: below that, a few pixels of rounding change every number.
#
# <table width="100%"><tr><td bgcolor="#FFF4D6"><font color="#7A4A00">
# <b>✏️ Your turn · Set the thresholds</b><br>
# Each threshold decides when a mask becomes a suspect. Run the hunt with the defaults first. Then make one
# check stricter (for example <code>IOU_MIN = 0.97</code> or <code>SOLIDITY_MAX = 0.9</code>) and one looser,
# rerun this cell and the next, and watch how many <i>real</i> masks join the suspects.
# </font></td></tr></table>

# %%
# ✏️ YOUR TURN
IOU_MIN = 0.9  # suspect when the mask's own box overlaps the annotated box less than this (IoU, 0–1)
FILL_RANGE = (0.15, 0.95)  # suspect when mask area / box area is outside this range
MAX_PIECES = 3  # suspect with more separate pieces than this
MAX_HOLES = 0  # suspect with more holes than this
SOLIDITY_MAX = 0.97  # suspect when the mask fills more than this share of its rubber band (no dents at all)

_ok = (isinstance(IOU_MIN, (int, float)) and 0 <= IOU_MIN <= 1
       and isinstance(FILL_RANGE, tuple) and len(FILL_RANGE) == 2 and 0 <= FILL_RANGE[0] < FILL_RANGE[1]
       and isinstance(MAX_PIECES, int) and MAX_PIECES >= 1 and isinstance(MAX_HOLES, int) and MAX_HOLES >= 0
       and isinstance(SOLIDITY_MAX, (int, float)) and 0 < SOLIDITY_MAX <= 1)
if not _ok:
    raise SystemExit("Thresholds: IOU_MIN and SOLIDITY_MAX are numbers between 0 and 1, FILL_RANGE is a pair "
                     "(low, high) with low < high, MAX_PIECES is a whole number ≥ 1 and MAX_HOLES one ≥ 0.")

# %% [markdown]
# **Q29.** 🟡 Before you run: which check do you expect to catch each of the five kinds? Some kinds may trip
# more than one check, and one check may catch none of the planted masks. Write your guess as five pairs
# (kind → check), then run the cell and compare.

# %%
# Q29  Measure every judged mask of park_masks, flag the ones outside the thresholds, and score the hunt.
_judge = park_masks[park_masks["mask_area"] >= MIN_JUDGED_AREA].reset_index(drop=True)
park_shape = pd.concat([_judge, pd.DataFrame([mask_shape(decode(r), b)
                                              for r, b in zip(_judge["rle"], _judge["bbox"], strict=True)])],
                       axis=1)
CHECKS = {
    "box IoU": park_shape["box_iou"] < IOU_MIN,
    "fill": (park_shape["fill"] < FILL_RANGE[0]) | (park_shape["fill"] > FILL_RANGE[1]),
    "pieces": park_shape["pieces"] > MAX_PIECES,
    "holes": park_shape["holes"] > MAX_HOLES,
    "solidity": park_shape["solidity"] > SOLIDITY_MAX,
}
park_shape["checks"] = [", ".join(name for name, hit in CHECKS.items() if hit.iloc[i])
                        for i in range(len(park_shape))]
mask_suspects = park_shape[park_shape["checks"] != ""].copy()
mask_suspects["n_checks"] = mask_suspects["checks"].str.count(",") + 1
mask_suspects = mask_suspects.sort_values(["n_checks", "box_iou", "id"], ascending=[False, True, True])

print(f"{len(_judge)} masks judged · {len(mask_suspects)} suspects "
      f"({len(mask_suspects) / max(len(_judge), 1):.1%}). Suspects per check:")
for name, hit in CHECKS.items():
    print(f"  {name:9s} {int(hit.sum())}")
_cols = ["id", "image_id", "class", "checks", "box_iou", "fill", "pieces", "holes", "solidity"]
display(mask_suspects[_cols].head(12).round(2).set_index("id"))

N_SUSPECTS_SHOW = 4 if SMOKE else 8  # suspect photos to show (N-014)
_show = mask_suspects.drop_duplicates("image_id").head(N_SUSPECTS_SHOW)
_checks_of = dict(zip(_show["image_id"], _show["checks"], strict=True))
show_masks(clean_images.set_index("id").loc[list(_show["image_id"])].reset_index(), park_masks,
           f"the {len(_show)} strongest suspects (as the team sent them)",
           caption=lambda r: f"{r['class']} · {_checks_of[r['id']]}")

# %% [markdown]
# Now the answer key: the five planted masks, what the cell above said about each, and each one next to the
# real COCO mask it replaced.

# %%
# The key: each planted mask, the checks that fired on it, and the original mask (top) vs the planted one.
_checks_by_id = park_shape.set_index("id")["checks"]
HUNT9_KEY["checks"] = HUNT9_KEY["ann_id"].map(_checks_by_id).fillna("")
HUNT9_KEY["found"] = HUNT9_KEY["checks"] != ""
hunt9_found = int(HUNT9_KEY["found"].sum())
RESULTS["hunt9"] = {"planted": len(HUNT9_KEY), "found": hunt9_found}
RESULTS["mask_suspects_real"] = int((~mask_suspects["id"].isin(HUNT9_KEY["ann_id"])).sum())
for _, k in HUNT9_KEY.iterrows():
    print(f"{'✅' if k['found'] else '❌'} {k['kind']:15s} photo {k['image_id']} ({k['class']}, {k['how']}) "
          f"→ {k['checks'] or 'no check fired'}")
print(f"found {hunt9_found} of {len(HUNT9_KEY)} planted · plus {RESULTS['mask_suspects_real']} real COCO "
      "masks among the suspects")

_by_image = clean_images.set_index("id")
fig, axes = plt.subplots(2, len(HUNT9_KEY), figsize=(3.2 * len(HUNT9_KEY), 6.2), squeeze=False)
for col, (_, k) in enumerate(HUNT9_KEY.iterrows()):
    photo = _by_image.loc[k["image_id"]]
    thumb = load_thumb(photo["path"], 320)
    scale = thumb.width / photo["width"]
    color = mpl.colors.to_rgb(CLASS_COLORS[k["class"]])
    for row, source in enumerate([masks, park_masks]):
        ax = axes[row, col]
        full = decode(source.loc[source["id"] == k["ann_id"], "rle"].iloc[0])
        small = cv2.resize(full, thumb.size, interpolation=cv2.INTER_NEAREST) > 0
        overlay = np.zeros((thumb.height, thumb.width, 4))
        overlay[small] = (*color, 0.6)
        ax.imshow(thumb)
        ax.imshow(overlay)
        x, y, w, h = source.loc[source["id"] == k["ann_id"], "bbox"].iloc[0]
        ax.add_patch(mpl.patches.Rectangle((x * scale, y * scale), w * scale, h * scale, fill=False, lw=1.5,
                                           ls="--", edgecolor=CLASS_COLORS[k["class"]]))
        frame(ax, CLASS_COLORS[k["class"]] if row == 0 else INK)
        ax.set_title(f"{k['class']} · original" if row == 0 else f"{k['kind']}\n{k['checks'] or '—'}",
                     fontsize=9)
fig.suptitle("the five planted masks (top: the real COCO mask, bottom: what the team sent)", color=INK)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# 💡 No check catches everything, and together they catch all five, with real masks on top (on the full
# set a few dozen, about 5 % of the masks judged: a list a person can look through in ten minutes).
# The box check does most of the work because COCO built its boxes *from* the masks: any mask that moved,
# grew or belongs elsewhere stops matching its box. The two others need a shape check: a chunk cut out of
# the middle leaves the box intact but makes a hole, and a crude polygon fills its rubber band completely.
#
# **Worked answer (Q29).** Expect: *another animal* → box IoU (the pasted mask sits around a different
# animal); *shifted* → box IoU (15 px of a 100–300 px box is a large share); *swollen* → box IoU, often fill
# too (a swollen mask can fill more than its box); *chunk missing* → holes; *crude polygon* → solidity (a
# polygon of a few points has no dents). Pieces catches none of the planted five, but look at the real masks
# it flags: animals broken up by a fence or a branch, sometimes correctly. Read the `checks` column of your
# own run against this list: on a different seed, a kind may trip one check more or less.
#
# The real suspects are worth a look too. COCO is not clean: a 2024 re-annotation study found that about
# **2.4 % of COCO's val masks are poor**, with duplicated mask pairs among them (Singh et al., *COCO-ReM*,
# ECCV 2024), and the masks with the highest solidity in the Play Dataset are polygons of 5 to 14 clicked
# points, real crude polygons. Some of your real suspects are genuine annotation errors, not false alarms.
#
# ### 🔴 Optional: does the outline follow the photo's edges?
#
# Every check so far ignored the photo. A good outline runs along the animal's edge, where the colour
# changes; a sloppy one cuts across fur and grass. The next cell finds the photo's edges (Canny, the classic
# edge detector) and measures, for each mask, the share of its outline that lies within 2 px of an edge. It
# does this for the five planted masks and a sample of real ones.
#
# **Q30.** 🔴 Which of the five kinds would this check catch, and which one is it blind to? Think about what
# the outline of each one runs along, then run.

# %%
# Q30  Edge alignment: share of each mask's outline within 2 px of a Canny edge in the photo.
N_EDGE_REAL = 10 if SMOKE else 150  # real masks to compare with (N-014)


def edge_alignment(path, mask: np.ndarray, band: int = 2) -> float:
    """Share of the mask's outline pixels that lie within `band` px of an edge of the photo."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape != mask.shape:
        return float("nan")
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    near = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))) > 0
    outline = (mask - cv2.erode(mask, np.ones((3, 3), np.uint8))) > 0
    return float(near[outline].mean()) if outline.any() else float("nan")


_real_pool = park_shape[~park_shape["id"].isin(HUNT9_KEY["ann_id"])]
_real = _real_pool.sample(min(N_EDGE_REAL, len(_real_pool)), random_state=SEED)
_planted = park_shape[park_shape["id"].isin(HUNT9_KEY["ann_id"])]
_edge_rows = pd.concat([_real, _planted])
_edge_rows["alignment"] = [edge_alignment(_by_image.at[i, "path"], decode(r))
                           for i, r in zip(_edge_rows["image_id"], _edge_rows["rle"], strict=True)]
_real_align = _edge_rows.loc[_real.index, "alignment"]
_kind_of = dict(zip(HUNT9_KEY["ann_id"], HUNT9_KEY["kind"], strict=True))
_planted_align = _edge_rows[_edge_rows["id"].isin(_kind_of)]
RESULTS["edge_alignment"] = {_kind_of[i]: float(a) for i, a in zip(_planted_align["id"],
                                                                  _planted_align["alignment"], strict=True)}
print(f"real masks: median alignment {_real_align.median():.0%} (middle half "
      f"{_real_align.quantile(0.25):.0%}–{_real_align.quantile(0.75):.0%}, n = {_real_align.notna().sum()})")
for kind, a in RESULTS["edge_alignment"].items():
    print(f"  {kind:15s} {a:.0%}  (lower than {(_real_align > a).mean():.0%} of the real masks)")

fig, ax = plt.subplots(figsize=(10, 3.4))
_y = {c: i for i, c in enumerate(CLASSES)}
_jitter = np.random.default_rng(SEED).uniform(-0.25, 0.25, len(_real))
ax.scatter(_edge_rows.loc[_real.index, "alignment"], _real["class"].map(_y) + _jitter, s=14, alpha=0.6,
           c=[CLASS_COLORS[c] for c in _real["class"]], label="real masks")
for _, p in _planted_align.iterrows():
    ax.scatter(p["alignment"], _y[p["class"]], s=160, marker="*", color=CLASS_COLORS[p["class"]],
               edgecolor=INK, linewidth=1.2, zorder=3)
    ax.annotate(_kind_of[p["id"]], (p["alignment"], _y[p["class"]]), textcoords="offset points",
                xytext=(6, 6), fontsize=8, color=INK)
ax.set_yticks(range(len(CLASSES)), CLASSES)
for lab in ax.get_yticklabels():
    lab.set_color(CLASS_COLORS[lab.get_text()])
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set_xlabel("share of the outline within 2 px of a photo edge")
ax.set_title("Does the outline follow the photo's edges? dots: real masks · stars: the planted five")
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# 💡 Real outlines are spread widely on this scale and rarely reach 100 %: a smooth animal on a smooth
# background has stretches without a strong edge, and a busy one has edges everywhere.
#
# **Worked answer (Q30).** Think of where each outline runs. *Shifted*, *swollen* and *crude polygon* run a
# few pixels off the animal's edge, through grass or fur, so they tend to sit low. *Chunk missing* keeps its
# real outline but adds the rim of the hole, which runs through the middle of the body, so it drops by as
# much as the hole is large. *Another animal* is the blind spot: when the pasted mask is a neighbour's from
# the same photo, it is a real, careful outline that follows real edges, just around the wrong animal, and
# it sits among the real masks, and only the box check sees it. When it comes from another photo, its
# outline lands on whatever happens to be there, and it sits low. If a planted mask lands among the real
# ones on your run, look at its photo: a busy background (grass, water, a fence) has edges everywhere, so
# any line lies close to one. That is why this check ranks masks for a human to look at rather than
# judging them.
