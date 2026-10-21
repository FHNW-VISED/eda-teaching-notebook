# %% [markdown]
# ## 8. Same data, boxes

# %% [markdown]
# <!-- callout: objective | Objective -->
# See what imbalance, object size, position and crowding mean once the park wants a *headcount*, and find
# the one box an annotation team forgot.

# %% [markdown]
# The park's second job, **count** the sheep, zebras and birds every day, needs a **box per animal**
# instead of one label per photo, and the same photos now tell a different story.
#
# `annotations` already holds one row per animal. Its box is `[x, y, w, h]` in pixels (COCO format: the
# top-left corner, then the width and height). This section looks at the boxes of the photos in
# `clean_images`.

# %% [markdown]
# <!-- callout: theory | Theory · What changes when the unit is a box -->
# A classifier sees one example per photo. A detector sees one example per **box**, so a photo with twelve
# sheep weighs twelve times a photo with one cat. Four things follow, each with its own chart below:
#
# - **Imbalance is counted in boxes**, not photos (a paddock photo shows a flock).
# - **Size matters**: a bird on the far side of the pond is a 12×8 px speck.
# - **Position matters**: animals always in the middle means the sheep at the fence edge is missed.
# - **Crowding matters**: two sheep behind each other may get one box.

# %% [markdown]
# The collapsed cell defines `show_boxes`, a gallery that draws each photo's boxes in its class colour.

# %%
# @title Box helpers (draw boxes on thumbnails) { display-mode: "form" }
def box_area_bucket(w: float, h: float) -> str:
    """COCO's size buckets from a box's width and height in pixels: small < 32², medium < 96², else large."""
    area = w * h
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def show_boxes(rows_df: pd.DataFrame, boxes_df: pd.DataFrame, title: str, caption=None, cols: int = 4,
               size: int = 384) -> None:
    """Gallery of the photos in `rows_df` with their boxes from `boxes_df` drawn in the class colour.

    `caption(row) -> str` writes the title above each photo; by default the class and the number of boxes.
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
        ax.axis("on")
        ax.imshow(thumb)
        own = boxes_df[boxes_df["image_id"] == row["id"]]
        for _, b in own.iterrows():
            x, y, w, h = b["bbox"]
            ax.add_patch(mpl.patches.Rectangle((x * scale, y * scale), w * scale, h * scale, fill=False,
                                               lw=2, edgecolor=CLASS_COLORS[b["class"]]))
        frame(ax, CLASS_COLORS[row["class"]])
        text = caption(row) if caption else f"{row['class']} · {len(own)} boxes"
        ax.set_title(text, fontsize=9)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    display(fig)
    plt.close(fig)


# %% [markdown]
# ### Boxes per class vs photos per class

# %% [markdown]
# <!-- callout: questions -->
# Section 4 ranked the classes by photos: cat and dog on top, sheep at the bottom. Now the same photos are
# counted by animal. *Example:* one photo of 12 sheep adds 1 to the sheep's photo count and 12 to its box
# count.

# %% [markdown]
# **Q22.** 🟡 Will the ranking by *boxes* be the same as the ranking by *photos*? Name the class you expect to
# climb most, then run the cell. It needs your first function of the section.

# %% [markdown]
# <!-- callout: your-turn | Your turn · `count_boxes_per_class` -->
# Write the function in the next cell: given an annotations table (one row per animal, with a
# `class` column), return a dict `{class: number of boxes}` with **every** class in
# `CLASSES` as a key, counts as plain `int`s. The `use()` line checks it
# against the reference and continues with the reference either way.

# %%
# @title Reference solution — peek only if you are stuck { display-mode: "form" }
def _ref_count_boxes_per_class(annotations):
    counts = annotations["class"].value_counts()
    return {c: int(counts.get(c, 0)) for c in CLASSES}


# %%
# ✍️ YOUR CODE
def count_boxes_per_class(annotations):
    """Number of boxes (rows) per class.

    Inputs: `annotations`, a DataFrame with one row per animal and a `class` column.
    Output: a dict {class: int} with every class of CLASSES as a key (0 when a class has no box).
    """
    ...


boxes = annotations[annotations["image_id"].isin(clean_images["id"])].reset_index(drop=True)
boxes_per_class = use(count_boxes_per_class, boxes, reference=_ref_count_boxes_per_class)

# %%
# @title Q22 Boxes per class next to photos per class, both ranked { display-mode: "form" }
photos_per_class = {c: int((clean_images["class"] == c).sum()) for c in CLASSES}
RESULTS["boxes_per_class"] = boxes_per_class
RESULTS["n_boxes"] = int(len(boxes))

_by_photos = sorted(CLASSES, key=lambda c: -photos_per_class[c])
_by_boxes = sorted(CLASSES, key=lambda c: -boxes_per_class[c])
print(f"{len(clean_images)} photos · {len(boxes)} boxes")
print("ranked by photos: " + " > ".join(f"{c} {photos_per_class[c]}" for c in _by_photos))
print("ranked by boxes:  " + " > ".join(f"{c} {boxes_per_class[c]}" for c in _by_boxes))
for c in CLASSES:
    move = _by_photos.index(c) - _by_boxes.index(c)
    if move:
        print(f"  {c}: rank {_by_photos.index(c) + 1} by photos → rank {_by_boxes.index(c) + 1} by boxes "
              f"({'↑' if move > 0 else '↓'}{abs(move)})")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
for ax, counts, order, what in zip(axes, [photos_per_class, boxes_per_class], [_by_photos, _by_boxes],
                                   ["photos", "boxes"], strict=True):
    ax.bar(order, [counts[c] for c in order], color=[CLASS_COLORS[c] for c in order])
    ax.set_title(f"{what} per class (ranked)")
    ax.set_ylabel(what)
    for lab in ax.get_xticklabels():
        lab.set_color(CLASS_COLORS[lab.get_text()])
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# **Worked answer (Q22).** No. The flock classes climb because one photo holds many animals: on the full set
# bird goes from 4th by photos to 1st by boxes and sheep from last to 2nd, while the lone cat and dog drop to
# the bottom. A detector is not short of sheep the way a classifier is. On a small subset, compare shapes,
# not digits.

# %% [markdown]
# ### Box sizes: small, medium, large
#
# A box's **area** is its width × height in pixels. COCO sorts boxes by area:
#
# - **small**: under 32×32 = 1,024 px² (a bird far away on the pond);
# - **medium**: under 96×96 = 9,216 px²;
# - **large**: anything bigger (a horse next to the fence).
#
# The plots use the box's **side** = √(w·h), the side of a square with the same area.

# %% [markdown]
# **Q23.** 🟢 Which of the six classes do you expect to have the smallest boxes? Pick one, then run the cell:
# it buckets every box and shows the smallest, a typical (median) and the largest box of the set.

# %%
# @title Q23 Box size per class, COCO buckets, smallest · median · largest box { display-mode: "form" }
boxes["w"] = boxes["bbox"].map(lambda b: b[2])
boxes["h"] = boxes["bbox"].map(lambda b: b[3])
boxes["box_area"] = boxes["w"] * boxes["h"]
boxes["side"] = np.sqrt(boxes["box_area"])  # the side of a square of the same area, in pixels
boxes["bucket"] = [box_area_bucket(w, h) for w, h in zip(boxes["w"], boxes["h"], strict=True)]
bucket_share = (boxes.groupby("class")["bucket"].value_counts(normalize=True).unstack(fill_value=0)
                .reindex(index=CLASSES, columns=["small", "medium", "large"], fill_value=0))
RESULTS["small_box_share"] = {c: float(bucket_share.loc[c, "small"]) for c in CLASSES}
print("share of each class's boxes that is small / medium / large:")
display((bucket_share * 100).round(0).astype(int).T)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
_sides = [boxes.loc[boxes["class"] == c, "side"].clip(lower=4).to_numpy() for c in CLASSES]
# Violins are drawn on log10(side) so the KDE is smooth on the log axis; ticks are relabelled in pixels.
_parts = ax.violinplot([np.log10(v) for v in _sides], showmedians=True, showextrema=False, widths=0.8)
for body, c in zip(_parts["bodies"], CLASSES, strict=True):
    body.set_facecolor(CLASS_COLORS[c])
    body.set_alpha(0.75)
_parts["cmedians"].set_color(INK)
_parts["cmedians"].set_linewidth(2.5)
_tick_names = [f"{c}\nmedian {np.median(v):.0f} px" for c, v in zip(CLASSES, _sides, strict=True)]
ax.set_xticks(range(1, len(CLASSES) + 1), _tick_names, fontsize=8)
for lab, c in zip(ax.get_xticklabels(), CLASSES, strict=True):
    lab.set_color(CLASS_COLORS[c])
ax.set_yticks(np.log10([4, 10, 32, 100, 300, 1000]), ["4", "10", "32", "100", "300", "1000"])
for thr, name in [(32, "small ↓"), (96, "large ↑")]:
    ax.axhline(np.log10(thr), color=INK, linestyle="--", linewidth=1)
    ax.text(0.5, np.log10(thr), f" {thr} px · {name}", va="bottom", fontsize=8, color=INK)
ax.set_ylabel("box side √(w·h), px (log)")
ax.set_title("Box size per class (bar = median)")

ax = axes[1]
_bottom = np.zeros(len(CLASSES))
for bucket, alpha in [("small", 1.0), ("medium", 0.55), ("large", 0.25)]:
    vals = bucket_share[bucket].to_numpy()
    ax.bar(CLASSES, vals, bottom=_bottom, color=[CLASS_COLORS[c] for c in CLASSES], alpha=alpha,
           label=bucket, edgecolor="white")
    _bottom += vals
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set_title("Share of small (solid) / medium / large (pale) boxes")
ax.legend(frameon=False, fontsize=8, loc="lower right")
for lab in ax.get_xticklabels():
    lab.set_color(CLASS_COLORS[lab.get_text()])
fig.tight_layout()
display(fig)
plt.close(fig)

# The smallest, a typical (median) and the largest box of the whole set: the photo with the box in bold, then
# the box cut out and enlarged (pixels shown as blocks), so a "10 px bird" is seen as the speck it is.
_by_area = boxes.sort_values(["box_area", "id"]).reset_index(drop=True)
_picks = {"smallest": _by_area.iloc[0], "median": _by_area.iloc[len(_by_area) // 2],
          "largest": _by_area.iloc[-1]}
_photo_of = clean_images.set_index("id")
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for col, (tag, b) in enumerate(_picks.items()):
    photo = _photo_of.loc[b["image_id"]]
    thumb = load_thumb(photo["path"], 384)
    scale = thumb.width / photo["width"]
    ax = axes[0, col]
    ax.imshow(thumb)
    for _, o in boxes[boxes["image_id"] == b["image_id"]].iterrows():
        ox, oy, ow, oh = o["bbox"]
        ax.add_patch(mpl.patches.Rectangle((ox * scale, oy * scale), ow * scale, oh * scale, fill=False,
                                           lw=3 if o["id"] == b["id"] else 1,
                                           edgecolor=CLASS_COLORS[o["class"]],
                                           ls="-" if o["id"] == b["id"] else ":"))
    frame(ax, CLASS_COLORS[b["class"]])
    ax.set_title(f"{tag}: {b['class']}, {b['w']:.0f}×{b['h']:.0f} px\n= {b['box_area']:,.0f} px² "
                 f"({b['bucket']})", fontsize=9)
    x, y, w, h = b["bbox"]
    pad = 0.25 * max(w, h) + 4
    x0, y0 = max(0.0, x - pad), max(0.0, y - pad)
    full = Image.open(photo["path"]).convert("RGB")
    crop = full.crop((int(x0), int(y0), int(x + w + pad), int(y + h + pad)))
    ax = axes[1, col]
    ax.imshow(crop, interpolation="nearest")
    ax.add_patch(mpl.patches.Rectangle((x - int(x0), y - int(y0)), w, h, fill=False, lw=2,
                                       edgecolor=CLASS_COLORS[b["class"]]))
    frame(ax, CLASS_COLORS[b["class"]])
    ax.set_title("the box up close", fontsize=9)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# **Worked answer (Q23).** Bird, by a wide margin: a large share of its boxes is *small*, while cat, dog and
# horse are mostly *large*. The **small share** is the number to watch: small boxes are what every detector
# misses first, and here they are the far-away birds the headcount needs (the violin is on a log scale
# because sizes range from a speck to a full frame).

# %% [markdown]
# ### Where the animals are: a box-centre heatmap
#
# A box's **centre** is `(x + w/2, y + h/2)`. Dividing by the photo's width and height puts every centre on
# the same 0–1 square, so a sheep at the left fence has a centre near x = 0.05 whatever the photo's size.
# The cell below draws one heatmap per class: dark = many centres there.

# %%
# @title Box-centre heatmap, one per class in its own colour { display-mode: "form" }
# Box centres on the unit square, one heatmap per class in its own colour (white = no boxes there).
_size = clean_images.set_index("id")[["width", "height"]]
_img_w = boxes["image_id"].map(_size["width"])
_img_h = boxes["image_id"].map(_size["height"])
boxes["cx"] = (boxes["bbox"].map(lambda b: b[0] + b[2] / 2) / _img_w).clip(0, 1)
boxes["cy"] = (boxes["bbox"].map(lambda b: b[1] + b[3] / 2) / _img_h).clip(0, 1)
HEAT_BINS = 6 if SMOKE else 12  # coarser cells with few boxes (N-014)
_edge_share = float(((boxes["cx"] < 0.2) | (boxes["cx"] > 0.8) | (boxes["cy"] < 0.2) | (boxes["cy"] > 0.8))
                    .mean())
RESULTS["edge_box_share"] = _edge_share

fig, axes = plt.subplots(1, len(CLASSES), figsize=(2.3 * len(CLASSES), 2.8))
for ax, c in zip(axes, CLASSES, strict=True):
    own = boxes[boxes["class"] == c]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(c, ["white", CLASS_COLORS[c]])
    heat, _, _ = np.histogram2d(own["cy"], own["cx"], bins=HEAT_BINS, range=[[0, 1], [0, 1]])
    ax.imshow(heat, cmap=cmap, extent=[0, 1, 1, 0], vmin=0)
    ax.set_title(f"{c} · {len(own)} boxes", fontsize=9, color=CLASS_COLORS[c])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=7)
    ax.grid(False)
axes[0].set_ylabel("box centre y (0 = top)")
fig.suptitle(f"Where the box centres are · {_edge_share:.0%} of all boxes are in the outer 20 % of the frame",
             color=INK)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# 💡 The centre is darker in every panel: photographers point the camera at the animal. A park camera does
# not, so a detector that learned "animals live in the middle" misses the sheep at the fence. Bird and sheep
# boxes spread out more because a flock fills the frame: the one way this dataset resembles the paddock.

# %% [markdown]
# ### Animals per photo, and crowding
#
# For the headcount the question is how often boxes of the same class **overlap**. Two sheep standing behind
# each other give two boxes that share pixels. Which box is which is hard for the detector, and easy for the
# annotator to get wrong (one box for two sheep, or none for the hidden one).
#
# Overlap is measured with **IoU** (intersection over union): the shared area divided by the area covered by
# either box. We call two boxes *overlapping* when their IoU is above 0.1.

# %%
# @title IoU of two boxes, drawn: shared area ÷ covered area { display-mode: "form" }
OVERLAP_IOU = 0.1  # two boxes "overlap" when their IoU is above this

_a = (20, 20, 60, 40)
_cases = [("two sheep, one behind the other", (50, 35, 60, 40)),
          ("two sheep, corners just touch", (70, 55, 60, 40))]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, (what, _b) in zip(axes, _cases, strict=True):
    iou = box_iou_dice(_a, _b)[0]
    _ix0, _iy0 = max(_a[0], _b[0]), max(_a[1], _b[1])
    _iw = max(0, min(_a[0] + _a[2], _b[0] + _b[2]) - _ix0)
    _ih = max(0, min(_a[1] + _a[3], _b[1] + _b[3]) - _iy0)
    _inter = _iw * _ih
    _union = _a[2] * _a[3] + _b[2] * _b[3] - _inter
    ax.add_patch(mpl.patches.Rectangle(_a[:2], *_a[2:], color=CLASS_COLORS["sheep"], alpha=0.35))
    ax.add_patch(mpl.patches.Rectangle(_b[:2], *_b[2:], color=CLASS_COLORS["zebra"], alpha=0.35))
    ax.add_patch(mpl.patches.Rectangle((_ix0, _iy0), _iw, _ih, facecolor="none", edgecolor=INK, hatch="////",
                                       lw=1.5))
    ax.set_xlim(10, 140)
    ax.set_ylim(105, 10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(f"{what}\nIoU = {_inter:,} ÷ {_union:,} = {iou:.2f} → "
                 f"{'overlapping' if iou > OVERLAP_IOU else 'not counted'}", fontsize=9)
fig.suptitle("IoU = hatched area (shared) ÷ coloured area (covered by either box)", color=INK)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# **Q24.** 🟡 Which class will have the most overlapping pairs of boxes per photo? *Example:* a photo of 4
# sheep has 6 pairs of boxes (4×3÷2); if two sheep stand behind each other, 1 of those 6 pairs overlaps.
# Decide, then run the cell: it measures every pair of boxes in every photo.

# %%
# @title Q24 Boxes per photo per class, and crowding (overlapping box pairs) { display-mode: "form" }
def overlap_pairs(boxes_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """One row per photo: n_boxes, n_pairs (all pairs) and n_overlapping (pairs with IoU > threshold)."""
    rows = []
    for image_id, own in boxes_df.groupby("image_id"):
        bb = [tuple(b) for b in own["bbox"]]
        pairs = [(i, j) for i in range(len(bb)) for j in range(i + 1, len(bb))]
        n_over = sum(box_iou_dice(bb[i], bb[j])[0] > threshold for i, j in pairs)
        rows.append({"image_id": image_id, "class": own["class"].iloc[0], "n_boxes": len(bb),
                     "n_pairs": len(bb) * (len(bb) - 1) // 2, "n_overlapping": int(n_over)})
    return pd.DataFrame(rows)


crowding = overlap_pairs(boxes, OVERLAP_IOU)
_per_class = crowding.groupby("class").agg(photos=("image_id", "size"), boxes_per_photo=("n_boxes", "mean"),
                                           crowded_photos=("n_overlapping", lambda s: float((s > 0).mean())),
                                           overlapping_pairs=("n_overlapping", "sum")).reindex(CLASSES)
_per_class["boxes_per_photo"] = _per_class["boxes_per_photo"].round(1)
_per_class["crowded_photos"] = (_per_class["crowded_photos"] * 100).round(0).astype(int).astype(str) + " %"
RESULTS["crowded_photo_share"] = float((crowding["n_overlapping"] > 0).mean())
_n_crowded = int((crowding["n_overlapping"] > 0).sum())
print("boxes_per_photo = boxes ÷ photos · crowded_photos = share of photos with ≥ 1 overlapping pair")
print(f"{_n_crowded} of {len(crowding)} photos have at least one pair of boxes with IoU > {OVERLAP_IOU} "
      f"({RESULTS['crowded_photo_share']:.0%}).")
display(_per_class.T)

N_CROWDED_BOXES = 2 if SMOKE else 4  # most crowded photos to show with their boxes (N-014)
_most = crowding.sort_values(["n_overlapping", "n_boxes", "image_id"], ascending=[False, False, True]).head(
    N_CROWDED_BOXES)
_rows = clean_images.set_index("id").loc[_most["image_id"]].reset_index()
_over_of = dict(zip(_most["image_id"], _most["n_overlapping"], strict=True))
show_boxes(_rows, boxes, "the most crowded photos (every box drawn)",
           caption=lambda r: f"{r['n_animals']} × {r['class']} · {_over_of[r['id']]} overlapping pairs",
           cols=N_CROWDED_BOXES)

# %% [markdown]
# **Worked answer (Q24).** The flock classes: sheep and zebra on the full set (herds stand close), then bird.
# Cats and dogs are mostly alone, so they have almost no pairs; on a small subset the leader can change, so
# read the `crowded_photos` row. Try counting the crowded photos' animals *without* the boxes: that is the
# annotator's job, and where boxes pile up a missing or doubled box is easy to make.

# %% [markdown]
# <!-- callout: across-tasks | Across tasks · What handles each of these later (next week and beyond) -->
# Each finding (box imbalance, small boxes, a wide range of sizes, photos with many boxes) has a standard
# answer on the model side, so the data report can say "we saw this, the training recipe handles it".
# None of them fixes a photo whose annotation is wrong: that is the data side, and it is yours.
#
# <details><summary>Names to look up next week</summary>
#
# - **Imbalance in boxes** (bird 1st, cat last) → a **weighted** or **focal loss**, which down-weights
#   the easy, frequent boxes.
# - **Small boxes** (far-away birds) → **tiling**: cut the photo into overlapping crops, so a 10 px bird
#   becomes a 40 px bird.
# - **A wide range of sizes** (a 10 px bird and a 400 px horse in one set) → **multi-scale** training, so
#   the detector looks at several resolutions at once.
# - **Photos with several boxes** (a "sheep" photo holding 12 boxes) → **iterative stratification**, the
#   split of section 6 that balances box counts per class, not only photo counts.
#
# </details>

# %% [markdown]
# ### Guided hunt: the box that is not there
#
# The park's annotation team returned boxes for its first batch of paddock photos. In the Play Dataset, the
# team's boxes are `park_boxes` and the keeper's headcount is the `n_animals` column of `clean_images`.

# %% [markdown]
# The story, in three steps:
#
# 1. The team draws a box around every animal and sends the table back (`park_boxes`).
# 2. On the day, the keeper counted the animals in each photo by eye (`n_animals`).
# 3. One box was dropped from a photo of a small flock, and a detector trained on it is told that animal is
#    *background*. Which photo is decided at runtime from `SEED`; nobody can look at hundreds of photos.

# %%
# @title Plant the hunt (one box removed, seeded) { display-mode: "form" }
def plant_hunt8(boxes_df: pd.DataFrame, table: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    """Return (park_boxes, key): `boxes_df` minus one box of a photo with ≥ 3 animals of one class.

    Sheep photos are preferred (the paddock story); otherwise the class with the most such photos. With no
    photo of 3 or more animals, the most crowded photo is used, so one box is always planted.
    """
    rng = random.Random(seed)
    counts = boxes_df.groupby("image_id").size()
    class_of = table.set_index("id")["class"]
    candidates = [int(i) for i in counts[counts >= 3].index]
    sheep = [i for i in candidates if class_of.get(i) == "sheep"]
    if sheep:
        candidates = sheep
    if not candidates:
        candidates = [int(counts.idxmax())]
    image_id = rng.choice(sorted(candidates))
    own = boxes_df[boxes_df["image_id"] == image_id]
    ann_id = int(rng.choice(sorted(own["id"])))
    park = boxes_df[boxes_df["id"] != ann_id].reset_index(drop=True)
    return park, {"image_id": image_id, "ann_id": ann_id, "class": str(class_of.get(image_id)),
                  "n_animals": int(counts[image_id])}


park_boxes, HUNT8_KEY = plant_hunt8(boxes, clean_images, SEED)
print(f"park_boxes: {len(park_boxes)} boxes for {park_boxes['image_id'].nunique()} photos "
      f"(boxes had {len(boxes)}). One box is missing. Find the photo.")

# %% [markdown]
# **Q25.** 🟡 You have two numbers for every photo: the keeper's headcount (`n_animals`) and the number of
# boxes the team drew for it (the rows of `park_boxes` with that `image_id`). What simple comparison of
# the two points straight at the photo with a missing box? Say it in one sentence, then run the cell.

# %%
# @title Q25 Keeper's headcount vs the team's box count per photo { display-mode: "form" }
box_counts = park_boxes.groupby("image_id").size()
headcount = clean_images.set_index("id")["n_animals"]
missing_boxes = (headcount - box_counts.reindex(headcount.index, fill_value=0))
suspects = missing_boxes[missing_boxes != 0].sort_index()
hunt8_found = int(HUNT8_KEY["image_id"] in suspects.index)
RESULTS["hunt8"] = {"planted": 1, "found": hunt8_found}
print(f"{len(suspects)} photo(s) where the headcount and the box count differ:")
for image_id, diff in suspects.items():
    print(f"  photo {image_id}: headcount {headcount[image_id]}, boxes {int(box_counts.get(image_id, 0))} "
          f"({'missing' if diff > 0 else 'extra'} {abs(int(diff))})")
print(f"planted: photo {HUNT8_KEY['image_id']} ({HUNT8_KEY['n_animals']} × {HUNT8_KEY['class']}, one box "
      f"removed) · found: {'✅' if hunt8_found else '❌'}")

_rows = clean_images.set_index("id").loc[list(suspects.index)].reset_index()
show_boxes(_rows, park_boxes, "the suspect photo(s): one animal has no box",
           caption=lambda r: f"{r['class']} · headcount {r['n_animals']} · "
                             f"{int(box_counts.get(r['id'], 0))} boxes")

# %% [markdown]
# **Worked answer (Q25).** Subtract the number of boxes per photo from the keeper's headcount: one photo has
# one box fewer than animals, and the gallery shows the animal without a frame. Two counts of the same thing
# from two sources cost nothing, which is why annotation teams are asked for a per-image count. Section 9
# repeats the idea between masks and boxes, and the Challenge hides another missing box among other faults.

# %%
# @title What a missing box looks like (toy drawing: headcount 3, boxes 2) { display-mode: "form" }
fig, ax = plt.subplots(figsize=(6, 2))
for k, x in enumerate([2, 5, 8]):
    ax.add_patch(mpl.patches.Ellipse((x, 1), 2.0, 1.2, color=CLASS_COLORS["sheep"], alpha=0.8))
    drawn = k < 2
    ax.add_patch(mpl.patches.Rectangle((x - 1.3, 0.1), 2.6, 1.8, fill=False, lw=2 if drawn else 1.5,
                                       ls="-" if drawn else "--",
                                       edgecolor=CLASS_COLORS["sheep"] if drawn else INK))
    if not drawn:
        ax.text(x, 1.0, "no box", ha="center", va="center", fontsize=9, color=INK)
ax.set_xlim(0, 10)
ax.set_ylim(0, 2)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
ax.set_title("the toy version: keeper counted 3 · team drew 2 boxes · difference 1", fontsize=9)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# <!-- callout: pitfall | Pitfall · Counting different things across tasks -->
# The same class has a different share depending on the unit:
#
# - **Sheep** is about 9 % of the photos but about 22 % of the boxes.
# - **Bird** is rare by photos, common by boxes, and rare again by pixels (section 9).
#
# Whenever you report imbalance, size or a split, say what the unit is: photo, box or pixel. The park's three
# jobs count three different things from the same photos.

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **When you see** a class ranking → **do** say whether you counted photos or boxes; the order can flip
#   (bird 4th by photos, 1st by boxes).
# - **When you see** many boxes under 32×32 px (far-away birds) → **do** report the small share and plan for
#   tiling or a higher input resolution.
# - **When you see** box centres piled in the middle → **do** expect misses at the frame edges, and test on
#   photos where the animal is off-centre.
# - **When you see** the keeper's headcount and the team's box count disagree on a photo → **do** open it:
#   a box is missing or doubled.
