# %% [markdown]
# ## 8. Same data, boxes
#
# **Objective:** see what imbalance, object size, position and label quality mean once the park wants a
# *headcount*, and find one box that an annotator forgot.
#
# Sections 4 to 7 treated each photo as one label. The park's second job, **count** the sheep, zebras and
# birds every day, needs a **box per animal** instead, and the same photos now tell a different story.
# `annotations` already holds one row per animal with its box as `[x, y, w, h]` in pixels (COCO format: the
# top-left corner and the size). This section keeps the photos of `clean_images` and looks at their boxes.
#
# <table width="100%"><tr><td bgcolor="#E3F2FD"><font color="#0D47A1">
# <b>📘 Theory · What changes when the unit is a box</b><br>
# A classifier sees one example per photo. A detector sees one example per <b>box</b>, so the photo with
# twelve sheep weighs twelve times the photo with one cat. Four things follow:
# <ul>
# <li><b>Imbalance is counted in boxes</b>, not photos. The class with the most photos is not the class with
# the most boxes.</li>
# <li><b>Size matters.</b> COCO calls a box <i>small</i> under 32×32 px, <i>medium</i> up to 96×96 px, and
# <i>large</i> above. Detectors score far worse on small boxes, and the park's far-away birds are small.</li>
# <li><b>Position matters.</b> If every animal sits in the middle of the frame, the detector learns
# "look in the middle", and the fence-edge sheep is missed.</li>
# <li><b>Crowding matters.</b> Overlapping boxes of the same class are the hard case for both the detector
# and the annotator, who may simply forget one.</li>
# </ul>
# </font></td></tr></table>
#
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
#
# <table width="100%"><tr><td bgcolor="#E0F7F4"><font color="#00564D">
# <b>❓ Questions</b><br>
# Section 4 ranked the classes by photos: cat and dog on top, sheep at the bottom. Now the same photos are
# counted by animal.
# </font></td></tr></table>
#
# **Q22.** 🟡 Will the ranking by *boxes* be the same as the ranking by *photos*? Name the class you expect to
# climb most, then run the cell. It needs your first function of the section.
#
# <table width="100%"><tr><td bgcolor="#FFF4D6"><font color="#7A4A00">
# <b>✏️ Your turn · <code>count_boxes_per_class</code></b><br>
# Write the function in the next cell: given an annotations table (one row per animal, with a
# <code>class</code> column), return a dict <code>{class: number of boxes}</code> with <b>every</b> class in
# <code>CLASSES</code> as a key, counts as plain <code>int</code>s. The <code>use()</code> line checks it
# against the reference and continues with the reference either way.
# </font></td></tr></table>

# %%
# ✍️ YOUR CODE
def count_boxes_per_class(annotations):
    """Number of boxes (rows) per class.

    Inputs: `annotations`, a DataFrame with one row per animal and a `class` column.
    Output: a dict {class: int} with every class of CLASSES as a key (0 when a class has no box).
    """
    ...


def _ref_count_boxes_per_class(annotations):
    counts = annotations["class"].value_counts()
    return {c: int(counts.get(c, 0)) for c in CLASSES}


boxes = annotations[annotations["image_id"].isin(clean_images["id"])].reset_index(drop=True)
boxes_per_class = use(count_boxes_per_class, boxes, reference=_ref_count_boxes_per_class)

# %%
# Q22  Boxes per class next to photos per class, both ranked, on the photos of clean_images.
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
# 💡 Same photos, two rankings: the flock classes (bird, sheep, zebra) climb because one photo of them holds
# many animals, and cat and dog sink because they are photographed alone. A detector trained on this set is
# *not* short of sheep examples the way a classifier is.
#
# **Worked answer (Q22).** No. On the full set bird climbs from 4th by photos to 1st by boxes, and sheep from
# last to 2nd, while cat and dog drop to the bottom. The exact ranks depend on the photos you have (the smoke
# subset is smaller), so compare the *shapes* of the two charts, not the digits. Say which unit you counted
# every time you say "imbalance": the headcount model and the alert model see different ones.
#
# ### Box sizes: small, medium, large
#
# **Q23.** 🟢 Which class do you expect to have the smallest boxes? Pick one, then run the cell: it buckets
# every box with COCO's thresholds and shows the smallest and largest boxes in the set.

# %%
# Q23  Box area per class on a log scale, COCO's small/medium/large buckets, and the extreme boxes.
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

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axes[0]
_sides = [boxes.loc[boxes["class"] == c, "side"].clip(lower=4).to_numpy() for c in CLASSES]
_parts = ax.boxplot(_sides, tick_labels=CLASSES, patch_artist=True, showfliers=False, widths=0.6)
for patch, c in zip(_parts["boxes"], CLASSES, strict=True):
    patch.set_facecolor(CLASS_COLORS[c])
    patch.set_alpha(0.75)
for med in _parts["medians"]:
    med.set_color(INK)
ax.set_yscale("log")
for thr, name in [(32, "small ↓"), (96, "large ↑")]:
    ax.axhline(thr, color=INK, linestyle="--", linewidth=1)
    ax.text(0.5, thr, f" {thr} px · {name}", va="bottom", fontsize=8, color=INK)
ax.set_ylabel("box side √(w·h), px (log)")
ax.set_title("Box size per class")
for lab in ax.get_xticklabels():
    lab.set_color(CLASS_COLORS[lab.get_text()])

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

N_EXTREME_BOXES = 2 if SMOKE else 4  # photos with the smallest / largest boxes to show (N-014)
_smallest = boxes.sort_values(["box_area", "id"]).drop_duplicates("image_id").head(N_EXTREME_BOXES)
_largest = boxes.sort_values(["box_area", "id"], ascending=[False, True]).drop_duplicates("image_id").head(
    N_EXTREME_BOXES)
_extreme_ids = list(_smallest["image_id"]) + list(_largest["image_id"])
_rows = clean_images.set_index("id").loc[_extreme_ids].reset_index()
_side_of = dict(zip(_extreme_ids, list(_smallest["side"]) + list(_largest["side"]), strict=True))
show_boxes(_rows, boxes, f"the {N_EXTREME_BOXES} smallest and the {N_EXTREME_BOXES} largest boxes",
           caption=lambda r: f"{r['class']} · box side {_side_of[r['id']]:.0f} px", cols=2 * N_EXTREME_BOXES)

# %% [markdown]
# 💡 The box plot is on a log scale because the spread is enormous: a bird can be a few pixels on a pond
# and a horse can fill the frame. The small share is the number to watch: those boxes are the ones every
# detector misses first, and in the park they are exactly the far-away birds the headcount needs.
#
# **Worked answer (Q23).** Bird, by a wide margin: on the full set a large share of its boxes is *small* in
# COCO's sense, while cat, dog and horse are mostly *large*. Whatever subset you run on, look at the
# smallest-box gallery: a box of 10 px is a speck, and resizing the photo to a network's input makes it
# smaller still (section 3).
#
# ### Where the animals are: a box-centre heatmap
#
# Each box has a centre. Dividing it by the photo's width and height puts every centre on the same
# 0–1 square, whatever the photo's size, and a 2-D histogram of those points shows where the annotator
# (and so the detector) expects an animal. The next cell draws one small heatmap per class.

# %%
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
# 💡 The centre of the frame is darker in every panel: photographers point the camera at the animal. A
# park camera does not, so a detector that learned "animals live in the middle" will miss the sheep at the
# fence. The flock classes (sheep, bird, zebra) spread wider because a flock fills the frame, and that is the
# one bias the Play Dataset shares with the paddock.
#
# ### Animals per photo, and crowding
#
# Section 4 counted animals per photo to ask "which label then?". For the headcount the question is
# different: how often do boxes of the same class **overlap**? Two sheep standing behind each other give
# two boxes that share pixels, which is hard for the detector (which one is which?) and easy for the
# annotator to get wrong (one box for two sheep, or no box for the hidden one).
#
# **Q24.** 🟡 Which class will have the most overlapping pairs of boxes per photo? Decide, then run the
# cell: for every photo it computes the overlap (IoU, section 7) of every pair of its boxes.

# %%
# Q24  Boxes per photo per class, and crowding: pairs of boxes in one photo with IoU above a threshold.
OVERLAP_IOU = 0.1  # two boxes "overlap" when their IoU is above this


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
# 💡 Look at the crowded gallery and try to count the animals *without* the boxes: that is the annotator's
# job, and the detector's. Where boxes pile up, a missing or doubled box is a matter of attention, which is
# why the next part plants one.
#
# **Worked answer (Q24).** The flock classes: sheep and zebra on the full set (herds stand close), with
# bird next (flocks on water). Cats and dogs are mostly alone, so they have no pairs at all. On a small
# subset the leader can change, so read the `crowded_photos` row: the share of a class's photos with at
# least one overlapping pair.
#
# <table width="100%"><tr><td bgcolor="#EDE7F6"><font color="#311B92">
# <b>🔀 Across tasks · What handles each of these later (next week and beyond)</b><br>
# Each finding above has a standard answer on the model side. Know the name, so that the data report can
# say "we saw this, the training recipe handles it with …":
# <ul>
# <li><b>Imbalance in boxes</b> (bird 1st, cat last) → a <b>weighted</b> or <b>focal loss</b>, which
# down-weights the easy, frequent boxes and lets the rare class count.</li>
# <li><b>Small boxes</b> (far-away birds) → <b>tiling</b>: cut the photo into overlapping crops and detect in
# each, so a 10 px bird becomes a 40 px bird.</li>
# <li><b>A wide range of sizes</b> (a bird of 10 px and a horse of 400 px in one set) → <b>multi-scale</b>
# training and feature pyramids, so the detector looks at several resolutions at once.</li>
# <li><b>Splitting photos that hold several boxes</b> (a photo is "sheep" <i>and</i> has 12 boxes) →
# <b>iterative stratification</b>, the multi-label version of the stratified split of section 6, which
# balances the box counts per class across the splits, not only the photo counts.</li>
# </ul>
# None of these fixes a photo whose annotation is wrong. That is the data side, and it is yours.
# </font></td></tr></table>
#
# ### Guided hunt: the box that is not there
#
# The park sent its first batch of paddock photos to an annotation team and got the boxes back. The keeper
# also wrote down a headcount per photo on the day. In the Play Dataset the keeper's headcount is the
# `n_animals` column of `clean_images` (counted from every instance in section 0), and the team's boxes are
# `park_boxes`, built by the collapsed cell: it is `boxes` with **one box removed** from a photo of a small
# flock. Which photo, and which animal, is decided at runtime from `SEED`.

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
# **Q25.** 🟡 You cannot look at hundreds of photos. Which two columns, compared per photo, point straight at
# the one with a missing box? Say it in one sentence, then run the cell.

# %%
# Q25  The keeper's headcount vs the team's box count, per photo: any photo where they differ is suspect.
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
# 💡 Two counts of the same thing from two sources, and a photo where they disagree: that is the whole
# trick, and it costs nothing. Without a second count (the keeper's), the only way is a visual check list,
# photo by photo, which is why annotation teams ask for a per-image count in the brief.
#
# **Worked answer (Q25).** Compare the keeper's headcount (`n_animals`) with the number of boxes per photo
# in `park_boxes`. One photo has one box fewer than animals; the gallery shows the animal without a frame.
# In section 9 the same comparison works between masks and boxes, and the Challenge hides another missing
# box among other faults.
#
# <table width="100%"><tr><td bgcolor="#FDECEA"><font color="#8B1A10">
# <b>⚠️ Pitfall · Counting different things across tasks</b><br>
# "Sheep is 9 % of the dataset" is true by photos and false by boxes (22 %). "Bird is rare" is true by
# photos and false by boxes, and true again by pixels (section 9). Whenever you report imbalance, size or a
# split, say what the unit is: photo, box or pixel. The park's three jobs count three different things from
# the same photos.
# </font></td></tr></table>
