# %% [markdown]
# ## 8. Same data, masks

# %% [markdown]
# <!-- callout: objective | Objective -->
# See what imbalance and object size mean once the park wants an *outline* of every animal, and find five
# ugly masks hidden among the real ones.

# %% [markdown]
# A **mask** is the exact pixels an animal covers. `annotations` holds one in its `segmentation` column, and
# `pycocotools` turns it into a 0/1 array the size of the photo.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · How a mask is stored: polygon and RLE -->
# <details><summary>Two storage formats, and why RLE is handy</summary>
#
# - **polygon**: the x, y points an annotator clicked around the animal;
# - **RLE** (run-length encoding, used for crowds): "300 background pixels, then 25 animal pixels, then …".
#
# The cell that builds the `masks` table turns every mask into one compressed RLE and counts its pixels.
# RLE is small and fast: the union and the area of masks are computed without building the full 0/1 array.
#
# </details>
#
# The collapsed cell above the `masks` table defines the mask tools: decode and encode RLE, measure a mask's
# shape, and `show_masks`, a gallery that paints each mask in its class colour over the photo, with its
# annotated box dashed.

# %% [markdown]
# <!-- callout: theory | Theory · What changes when the unit is a pixel -->
# A segmentation model makes one decision per **pixel**: which class is this pixel, or is it background?
#
# - **Imbalance is counted in pixels, background included.** Most of every photo is grass, sky and water.
# - **Size is everything.** A bird of 10×10 px is 100 decisions; a horse filling the frame is 100,000. The
#   horse adds about 1,000 times more pixels to the training error than the bird, so a model mostly learns
#   horses.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · Mask quality, thin parts, and where the animals sit -->
# A box is four numbers; a mask is dozens of clicks, so a bad mask looks like any other row. Legs, tails,
# beaks and far-away birds are few pixels wide, and they are the first to vanish when the photo is shrunk to
# the network's input size. A big horse outweighs a whole flock of far-away birds. The optional part at the
# end of this section measures thin parts, and draws where the animal pixels sit in the frame.

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
               size: int = 384, mark_thin: bool = False, highlight: dict | None = None) -> None:
    """Gallery of the photos in `rows_df` with every mask of `masks_df` (column `rle`) painted in its class
    colour and its annotated box dashed. `mark_thin=True` paints the thin parts dark.
    `highlight={image_id: mask id}` draws that one mask bright with a solid box and fades the others.

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
        flagged = (highlight or {}).get(row["id"])
        if flagged is not None:  # the flagged mask goes last, so it is painted on top
            own = own.assign(_top=own["id"] == flagged).sort_values("_top")
        for _, m in own.iterrows():
            full = decode(m["rle"])
            small = cv2.resize(full, thumb.size, interpolation=cv2.INTER_NEAREST) > 0  # nearest: section 3
            alpha = 0.6 if flagged is None else (0.8 if m["id"] == flagged else 0.15)
            overlay[small] = (*mpl.colors.to_rgb(CLASS_COLORS[m["class"]]), alpha)
            if mark_thin:
                thin = cv2.resize(thin_part(full), thumb.size, interpolation=cv2.INTER_NEAREST) > 0
                overlay[thin] = (*mpl.colors.to_rgb(INK), 0.95)
        ax.axis("on")
        ax.imshow(thumb)
        ax.imshow(overlay)
        for _, m in own.iterrows():
            x, y, w, h = m["bbox"]
            if flagged is not None and m["id"] != flagged:
                continue  # with a highlight, only the suspect's box is drawn
            ax.add_patch(mpl.patches.Rectangle((x * scale, y * scale), w * scale, h * scale, fill=False,
                                               lw=2.5 if flagged is not None else 1.5,
                                               ls="-" if flagged is not None else "--",
                                               edgecolor=CLASS_COLORS[m["class"]]))
        frame(ax, CLASS_COLORS[row["class"]])
        ax.set_title(caption(row) if caption else f"{row['class']} · {len(own)} masks", fontsize=9)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    display(fig)
    plt.close(fig)


# %% [markdown]
# The next cell turns every mask of `clean_images` into one compressed RLE (the `rle` column of a new table,
# `masks`) and counts its pixels.

# %%
# @title Build `masks`: one RLE per mask, with its pixel count { display-mode: "form" }
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
# Four photos with their masks painted, before any number: a close-up pet, a flock of small birds, horses
# with thin legs, and a flock of sheep.

# %%
# @title Four photos with their masks: close-up pet, small birds, thin legs, a flock { display-mode: "form" }
_mimgs = clean_images.set_index("id")
_mper = masks.groupby("image_id").agg(n=("class", "size"), cls=("class", "first"),
                                      px=("mask_area", "sum"))
_mper["share"] = _mper["px"] / (_mimgs.loc[_mper.index, "width"] * _mimgs.loc[_mper.index, "height"])
_mpicks = []
for _cls, _how in [(["cat", "dog"], "largest share"), (["bird"], "most masks"),
                   (["horse"], "largest share"), (["sheep", "zebra"], "most masks")]:
    _c = _mper[_mper["cls"].isin(_cls) & ~_mper.index.isin(_mpicks)]
    if len(_c):
        _key = _c["share"].idxmax() if _how == "largest share" else _c["n"].idxmax()
        _mpicks.append(_key)
show_masks(_mimgs.loc[_mpicks].reset_index(), masks, "same photos, third unit: every pixel of every animal",
           caption=lambda r: f"{r['class']} · animals cover {_mper.at[r['id'], 'share']:.0%} of the photo")

# %% [markdown]
# ### Pixels per class, background included

# %% [markdown]
# <!-- callout: questions -->
# Every pixel that is not an animal is *background*, a class like any other for the model.

# %% [markdown]
# **Q20.** 🟢 What share of all pixels do you expect to be background: a quarter, a half, or most? Is bird
# frequent or rare by pixels? Guess both, then run.

# %%
# @title Q20 Pixels per class including background, next to the photo and box shares { display-mode: "form" }
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
# <!-- callout: answer | Worked answer (Q20) -->
# Most of them: about 85 % of all pixels are background (the smoke subset gives about
# 80 %). Bird is rare again: about 1 % of all pixels (6–7 % of the animal pixels), because most birds are far
# away, while cat and dog come back (a pet photo is a close-up). Photos, boxes, pixels: three rankings of one
# dataset, so "bird is rare" needs its unit.

# %% [markdown]
# ### How big is a mask, and how much of its box does it fill?
#
# A mask sits inside its box, so a second number comes for free: the **fill ratio**, mask area divided by box
# area. A box around a curled-up cat is mostly cat; a box around a horse is mostly the air between its legs.
# The cell plots mask size and fill ratio per class, then shows the photos with the lowest, a typical
# (median) and the highest fill ratio.

# %% [markdown]
# **Q21.** 🟡 What fraction of its box does a typical animal fill: about 25 %, 50 % or 80 %? Which class do
# you expect to fill its box least? Guess, then run.

# %%
# @title Q21 Shape numbers of every mask, then mask area and fill ratio per class { display-mode: "form" }
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
    if log:  # violins are drawn on log10(value) so the KDE is smooth on a log axis; ticks are relabelled
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

N_EXTREME_MASKS = 1 if SMOKE else 2  # photos at each end of the fill ratio to show (N-014)
_judged = (animals[animals["mask_area"] >= MIN_JUDGED_AREA].sort_values(["fill", "id"])
           .drop_duplicates("image_id").reset_index(drop=True))
_ends = pd.concat([_judged.head(N_EXTREME_MASKS), _judged.iloc[[len(_judged) // 2]],
                   _judged.tail(N_EXTREME_MASKS)]).drop_duplicates("image_id")
_fill_of = dict(zip(_ends["image_id"], _ends["fill"], strict=True))
_tags = ["lowest"] * N_EXTREME_MASKS + ["median"] + ["highest"] * N_EXTREME_MASKS
_tag_of = dict(zip(_ends["image_id"], _tags, strict=False))
show_masks(clean_images.set_index("id").loc[list(_ends["image_id"])].reset_index(), masks,
           "lowest · typical (median) · highest fill ratio",
           caption=lambda r: f"{_tag_of[r['id']]}: {r['class']}, fill {_fill_of[r['id']]:.0%}",
           cols=len(_ends))

# %% [markdown]
# <!-- callout: answer | Worked answer (Q21) -->
# About half, in a narrow band of roughly 40–60 % for every class: near 45 % for bird
# and horse (wings, legs, necks), near 55 % for cat and sheep (compact shapes). The low end of the gallery
# shows spread wings and diagonal poses, the high end close-ups cut by the photo's edge. That band is what
# makes fill a quality check: a mask filling 98 % of its box is a rectangle, one filling 5 % lost the animal.
# (Crowd regions are left out: they outline a group, not an animal.)

# %% [markdown]
# <!-- callout: across-tasks | Across tasks · What handles each of these later (next week and beyond) -->
# Imbalance and tiny animals have a model-side answer. None of it fixes a wrong mask: that is the data side.
#
# <details><summary>Names to look up next week</summary>
#
# - **Pixel imbalance** (background about 85 %, bird about 1 %) → a **class-weighted** cross-entropy or a
#   **Dice loss**, which scores each class's overlap on its own scale, and reporting **mean IoU over
#   classes** (section 7), so that birds count as much as background.
# - **Tiny animals and thin parts** → a higher input resolution, or **crops / tiles** of the photo,
#   so that a 7 px leg is still 7 px for the network; boundary-aware losses weigh the outline more.
# - **Everything in the centre** → random crops and shifts when training, so the edge of the frame is
#   not always background.
#
# </details>

# %% [markdown]
# ### Guided hunt: five ugly masks
#
# The park paid an annotation team per mask, and the masks came back as `park_masks`: `masks` with **five
# masks spoiled**, picked from `SEED`, one of each kind a team really delivers:
#
# - **swollen**: the outline drawn too generously;
# - **shifted**: the right outline, about 15 px off;
# - **a chunk missing**: a piece of the body cut out;
# - **another animal's mask** pasted onto this one;
# - **a crude polygon**: five or six clicks in a hurry.
#
# They hide among hundreds of real masks. **Your job is not to look through hundreds of masks.** It is to
# know which check catches which kind, so that a script does the looking and you read its short list.
# In order:
#
# 1. look at the five kinds (first picture);
# 2. look at the five checks (second picture);
# 3. answer the question: which check catches which kind;
# 4. run the hunt and read its output;
# 5. open the answer key.

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
# First, what the five kinds look like. The next picture spoils one toy horse in the five ways: the grey line
# is the real outline, the dashed box is the box the team drew from it, the coloured shape is what they sent.

# %%
# @title The five kinds of ugly mask, drawn on one toy horse { display-mode: "form" }
def toy_horse(height: int = 60, width: int = 84) -> np.ndarray:
    """A toy horse mask (an oval body and four legs), 0/1, used by the two toy pictures of the hunt."""
    yy, xx = np.mgrid[0:height, 0:width]
    horse = ((xx - 40) / 24) ** 2 + ((yy - 32) / 13) ** 2 <= 1
    for lx in (24, 32, 46, 54):
        horse |= (xx >= lx) & (xx < lx + 5) & (yy >= 38) & (yy < 56)
    return horse.astype(np.uint8)


_horse = toy_horse()
_kx, _ky, _kw, _kh = cv2.boundingRect(_horse)  # the team's box, drawn from the real outline
_kyy, _kxx = np.mgrid[0 : _horse.shape[0], 0 : _horse.shape[1]]
_poly = np.array(_crude_polygon(_horse), np.int32).reshape(-1, 1, 2)
_bite = ((_kxx - 40) ** 2 + (_kyy - 30) ** 2 <= 36).astype(np.uint8)  # a disc inside the body
_other = (((_kxx - 66) / 13) ** 2 + ((_kyy - 18) / 9) ** 2 <= 1).astype(np.uint8)  # another animal
_kinds = {
    "swollen\n(drawn too generously)": cv2.dilate(_horse, np.ones((7, 7), np.uint8)),
    "shifted\n(right outline, moved)": np.roll(_horse, (4, 9), axis=(0, 1)),
    "chunk missing\n(a piece cut out)": _horse & (1 - _bite),
    "another animal's mask\n(pasted here)": _other,
    "crude polygon\n(six quick clicks)": cv2.fillPoly(np.zeros_like(_horse), [_poly], 1),
}
_outline, _ = cv2.findContours(_horse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
fig, axes = plt.subplots(1, 5, figsize=(13, 2.7))
for ax, (kind, spoiled) in zip(axes, _kinds.items(), strict=True):
    layer = np.zeros((*_horse.shape, 4))
    layer[spoiled > 0] = (*mpl.colors.to_rgb(CLASS_COLORS["horse"]), 0.85)
    ax.imshow(layer)
    for c in _outline:
        ax.plot(c[:, 0, 0], c[:, 0, 1], color="#9AA5A8", lw=1)
    ax.add_patch(mpl.patches.Rectangle((_kx - 0.5, _ky - 0.5), _kw, _kh, fill=False, lw=1.5, ls="--",
                                       edgecolor=INK))
    ax.set_xlim(-0.5, _horse.shape[1] - 0.5)
    ax.set_ylim(_horse.shape[0] - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(kind, fontsize=9)
fig.suptitle("Five ways a delivered mask goes wrong (grey: real outline · dashed: the team's box)", color=INK)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# Then the five checks, all computed from the mask and its annotated box, each with the rule that makes a mask
# a suspect. The picture after the list draws each check on one toy mask.
#
# 1. **mask–box agreement** (the only check that uses IoU): draw the tightest box around the mask and compare
#    it with the team's box; their IoU (section 7) is 1 when they match. Suspect when it is below `IOU_MIN`
#    (0.9), **the one value you can change**.
# 2. **fill ratio**: mask area / box area, from Q21. Suspect outside 15–95 %.
# 3. **pieces**: separate parts of the mask. Suspect with more than 3.
# 4. **holes**: gaps fully inside the mask. Suspect with any hole.
# 5. **solidity**: mask area / area of its **convex hull**, the "rubber band" around the whole mask. Suspect
#    above 0.97: a real animal has dents (between the legs, under the neck).
#
# <details><summary>What is normal for a real animal</summary>
#
# Box agreement is almost always above 0.95 on real COCO masks, because the box was made from the mask. An
# animal is usually one piece (two or three when something stands in front of it) and almost never has a
# hole. It has dents (between the legs, under the neck), so its solidity is rarely close to 1. Only masks
# of at least 32×32 px are judged: below that, a few pixels of rounding change every number.
#
# </details>

# %%
# @title The five checks drawn on one toy mask { display-mode: "form" }
# A toy "horse" (body, four legs, a hole, a separate piece) and its sloppily drawn box, measured with the same
# mask_shape() that the hunt uses, so the numbers in the titles are the numbers the checks will compute.
_H, _W = 60, 84
_yy, _xx = np.mgrid[0:_H, 0:_W]
_toy = toy_horse(_H, _W).astype(bool)  # the body and four legs
_toy |= (_xx >= 68) & (_xx < 76) & (_yy >= 30) & (_yy < 38)  # a separate piece, away from the body
_hole = (_xx - 40) ** 2 + (_yy - 28) ** 2 <= 16
_toy = (_toy & ~_hole).astype(np.uint8)
_tx, _ty, _tw, _th = cv2.boundingRect(_toy)  # the mask's own tight box
_drawn = (_tx + 5, _ty + 3, _tw, _th)  # the annotator's box, a bit off
_s = mask_shape(_toy, _drawn)


def _layer(mask, color, alpha):
    out = np.zeros((_H, _W, 4))
    out[mask > 0] = (*mpl.colors.to_rgb(color), alpha)
    return out


_n_parts, _parts_lab = cv2.connectedComponents(_toy, connectivity=8)
_part_sizes = {k: int((_parts_lab == k).sum()) for k in range(1, _n_parts)}
_main_part = max(_part_sizes, key=_part_sizes.get)
_contours, _ = cv2.findContours(_toy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
_hull = cv2.convexHull(np.vstack(_contours))[:, 0, :]

fig, axes = plt.subplots(1, 5, figsize=(13, 2.9))
for ax in axes:
    ax.imshow(_layer(_toy, CLASS_COLORS["horse"], 0.35))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
ax = axes[0]  # 1 · mask-box agreement: the mask's own box (solid) against the drawn box (dashed)
ax.add_patch(mpl.patches.Rectangle((_tx - 0.5, _ty - 0.5), _tw, _th, fill=False, lw=2, edgecolor=INK))
ax.add_patch(mpl.patches.Rectangle((_drawn[0] - 0.5, _drawn[1] - 0.5), _drawn[2], _drawn[3], fill=False, lw=2,
                                   ls="--", edgecolor=CLASS_COLORS["sheep"]))
ax.set_title(f"1 · mask-box agreement\nsolid: mask's own box · dashed: drawn\nIoU = {_s['box_iou']:.2f}",
             fontsize=8)
ax = axes[1]  # 2 · fill: mask pixels out of the drawn box's pixels
ax.imshow(_layer(_toy, CLASS_COLORS["horse"], 0.9))
ax.add_patch(mpl.patches.Rectangle((_drawn[0] - 0.5, _drawn[1] - 0.5), _drawn[2], _drawn[3], fill=False, lw=2,
                                   ls="--", edgecolor=CLASS_COLORS["sheep"]))
ax.set_title(f"2 · fill = mask ÷ drawn box\n{int(_toy.sum()):,} ÷ {_drawn[2] * _drawn[3]:,} "
             f"= {_s['fill']:.2f}", fontsize=8)
ax = axes[2]  # 3 · pieces: each connected part, the main one in colour, the others dark
for k in _part_sizes:
    ax.imshow(_layer(_parts_lab == k, CLASS_COLORS["horse"] if k == _main_part else INK, 0.9))
ax.set_title(f"3 · pieces\nseparate parts (dark = extra)\n= {_s['pieces']}", fontsize=8)
ax = axes[3]  # 4 · holes: the enclosed gaps
ax.imshow(_layer(_toy, CLASS_COLORS["horse"], 0.9))
ax.imshow(_layer(_hole, INK, 1.0))
ax.annotate("hole", xy=(40, 28), xytext=(40, 6), ha="center", fontsize=8, color=INK,
            arrowprops={"arrowstyle": "->", "color": INK})
ax.set_title(f"4 · holes\ngaps fully inside the mask\n= {_s['holes']}", fontsize=8)
ax = axes[4]  # 5 · solidity: mask area over the area of its convex hull (the rubber band)
ax.add_patch(mpl.patches.Polygon(_hull, closed=True, fill=False, lw=2, ls="--", edgecolor=INK))
ax.imshow(_layer(_toy, CLASS_COLORS["horse"], 0.9))
ax.set_title(f"5 · solidity = mask ÷ rubber band\n(dashed: the rubber band)\n= {_s['solidity']:.2f}",
             fontsize=8)
for ax in axes:
    ax.set_xlim(-0.5, _W - 0.5)
    ax.set_ylim(_H - 0.5, -0.5)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# **Q22.** 🟡 Match each kind (first picture) to the check you expect to catch it (the numbered list):
# write five pairs, e.g. *shifted → mask–box agreement*. Some kinds trip more than one check, and one check
# may catch none.
#
# Then run the hunt (the cell after the ✏️ box) and read its output in this order:
#
# 1. **The count per check:** how many masks each check flags. Most of them are real COCO masks, not planted.
# 2. **The gallery of the strongest suspects** (the masks that tripped the most checks). In each photo the
#    **bright mask with the solid box** is the suspect; the faint ones are the other animals. Its title gives
#    the checks that fired and its numbers. For each photo, guess: planted, or a real COCO mask?
# 3. **The answer key** below: compare it with your five pairs and your guesses.

# %% [markdown]
# <!-- callout: your-turn | Your turn · `IOU_MIN`, the threshold of check 1 only -->
# `IOU_MIN` tunes **check 1 (mask–box agreement) and nothing else**; checks 2 to 5 keep their fixed rules.
#
# - **First run:** leave it at 0.9 and answer Q22.
# - **After the answer key:** come back, set it to 0.97, rerun this cell, the Q22 cell and the key, and count
#   how many *real* masks join the suspects (a stricter check costs more human looking time). Then try 0.5
#   and see which planted mask you lose.

# %%
# ✏️ YOUR TURN
IOU_MIN = 0.9  # suspect when the mask's own box overlaps the annotated box less than this (IoU, 0–1)

if not (isinstance(IOU_MIN, (int, float)) and 0 <= IOU_MIN <= 1):
    raise SystemExit("IOU_MIN is a number between 0 and 1.")

# %%
# @title Q22 Measure every judged mask, flag the suspects, score the hunt { display-mode: "form" }
FILL_RANGE = (0.15, 0.95)  # suspect when mask area / box area is outside this range
MAX_PIECES = 3  # suspect with more separate pieces than this
MAX_HOLES = 0  # suspect with more holes than this
SOLIDITY_MAX = 0.97  # suspect when the mask fills more than this share of its rubber band (no dents at all)

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
_show = mask_suspects.drop_duplicates("image_id").head(N_SUSPECTS_SHOW).set_index("image_id")


def _suspect_caption(r) -> str:
    s = _show.loc[r["id"]]
    return (f"photo {r['id']} · {s['class']} · {s['checks']}\n"
            f"box IoU {s['box_iou']:.2f} · fill {s['fill']:.2f}")


show_masks(clean_images.set_index("id").loc[list(_show.index)].reset_index(), park_masks,
           f"the {len(_show)} strongest suspects: bright mask + solid box = suspect, faint = other animals",
           caption=_suspect_caption, highlight=dict(zip(_show.index, _show["id"], strict=True)))

# %% [markdown]
# The answer key, closed until you open it: the five planted masks, the checks that fired, and each next to
# the real mask it replaced.

# %%
# @title The answer key: planted masks, checks that fired, original vs planted { display-mode: "form" }
# The key: each planted mask, the checks that fired on it, and the original mask (top) vs the planted one.
_checks_by_id = park_shape.set_index("id")["checks"]
HUNT9_KEY["checks"] = HUNT9_KEY["ann_id"].map(_checks_by_id).fillna("")
HUNT9_KEY["found"] = HUNT9_KEY["checks"] != ""
hunt9_found = int(HUNT9_KEY["found"].sum())
RESULTS["hunt9"] = {"planted": len(HUNT9_KEY), "found": hunt9_found}
RESULTS["mask_suspects_real"] = int((~mask_suspects["id"].isin(HUNT9_KEY["ann_id"])).sum())
_lines = [f"{'✅' if k['found'] else '❌'} {k['kind']:15s} photo {k['image_id']} ({k['class']}, {k['how']}) "
          f"→ {k['checks'] or 'no check fired'}" for _, k in HUNT9_KEY.iterrows()]
_lines.append(f"found {hunt9_found} of {len(HUNT9_KEY)} planted · plus {RESULTS['mask_suspects_real']} "
              "real COCO masks among the suspects")

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
answer_box("The answer key: the five planted masks, the checks that fired, original vs planted", _lines, fig)

# %% [markdown]
# <!-- callout: answer | Worked answer (Q22) -->
# No check catches everything; together they catch all five, plus a few dozen real
# masks (about 5 % of those judged: a list a person can look through in ten minutes). The box check does most
# of the work because COCO built its boxes *from* the masks, so a mask that moved, grew or belongs elsewhere
# stops matching its box. The other two kinds need a shape check. Expect:
#
# - *another animal* → box IoU (the pasted mask sits around a different animal);
# - *shifted* → box IoU (15 px of a 100–300 px box is a large share);
# - *swollen* → box IoU, often fill too (it can fill more than its box);
# - *chunk missing* → holes (the box stays intact, the cut-out leaves a hole);
# - *crude polygon* → solidity (a few points leave no dents, so it fills its rubber band).
#
# Pieces catches none of the five, but flags real animals broken up by a fence or a branch. On a different
# seed, a kind may trip one check more or less: read the `checks` column of your own run.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · COCO's own masks are not clean -->
# Some of your real suspects are genuine annotation errors, not false alarms. A 2024 re-annotation study
# found that about **2.4 % of COCO's val masks are poor**, duplicated mask pairs among them (*COCO-ReM*,
# ECCV 2024). In the Play Dataset, the masks with the highest solidity are polygons of 5 to 14 clicked
# points: real crude polygons.

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **When you see** a class called "rare" or "common" → **do** name the unit (photo, box or pixel); bird is
#   rare by pixels (about 1 %) and common by boxes.
# - **When you see** a mostly-background dataset → **do** report per-class IoU or Dice, not pixel accuracy:
#   "all background" already scores about 85 %.
# - **When you see** a mask that does not match its box, or has a hole, or is rubber-band-tight → **do**
#   open the photo: it is shifted, swollen, cut or crudely clicked.

# %% [markdown]
# ### 🔴 Optional · More about masks
#
# Two more things a mask can tell you: where in the frame the animal pixels are, and which masks have parts
# too thin to survive a resize.

# %% [markdown]
# #### Where the animal pixels are: one heatmap per class
#
# Section 7 drew where the box *centres* are. A mask says more: which pixels of the frame are animal. The
# next cell shrinks every photo's merged mask to the same small square and averages them per class, so a
# dark spot means "in many of this class's photos, this spot is animal".

# %%
# @title Mean mask per class: where the animal pixels are { display-mode: "form" }
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

# %% [markdown]
# #### Thin parts
#
# A leg, a tail, a beak or a far-away bird is only a few pixels wide. What happens when the photo is shrunk:
#
# 1. A COCO photo is about 640 px wide; the network takes 224×224, about a third. A horse's leg of 7 px
#    becomes 2 px.
# 2. A mask resized with nearest neighbour (section 3) samples one pixel in three, so anything under 3 px
#    (a thin tail, a far-away bird's leg) can fall between the samples and vanish.
# 3. The network never sees that part: the label says "no tail" for a horse that has one.
#
# The next cell marks, in every mask, the pixels in parts narrower than `THIN_PX` = 7 px.

# %% [markdown]
# **Q23.** 🔴 Which class has the most masks with thin parts, and is it for the same reason as the
# runner-up? Pick one, then run.

# %%
# @title Q23 Share of each class's masks with thin parts, and the thinnest masks { display-mode: "form" }
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
# <!-- callout: answer | Worked answer (Q23) -->
# Bird, by far, on all masks (about two thirds on the full set), but not because
# birds have long legs: most birds are so small that the *whole bird* is a thin part. Among the masks large
# enough to judge, horse leads (legs), then bird, sheep and zebra, and cat almost never has one: tiny objects
# and thin parts of big ones are two problems behind one number. Thin parts are so few pixels that a model
# can get all of them wrong and still score well.
#
# **When you see** thin parts or tiny animals → **do** resize masks with nearest neighbour, keep a higher
# input resolution or tile, and check that the thin parts survive.
