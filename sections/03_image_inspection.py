# %% [markdown]
# ## 3. Image inspection
#
# **Objective:** measure sizes, aspect ratios, colour modes and colour statistics, and learn to read a cluster
# of odd values as a clue about where the photos came from.
#
# Your eyes found patterns in section 2. Now the computer looks at every photo once and writes down a handful
# of numbers per image: its size, its shape, how its colours are stored and how bright it is. None of these
# numbers needs a label, so this is the first check you can run on *any* folder of images, including your own
# project dataset in section 11. Karpathy's recipe says why this pays off:
#
# <table width="100%"><tr><td bgcolor="#E8F5E9"><font color="#1B5E20">
# <b>🧾 Recipe · Step 1, continued: search, filter, sort</b><br>
# <i>"Once you get a qualitative sense it is also a good idea to write some simple code to search/filter/sort
# by whatever you can think of (e.g. type of label, size of annotations, number of annotations, etc.) and
# visualize their distributions and the outliers along any axis. The outliers especially almost always uncover
# some bugs in data quality or preprocessing."</i><br> Andrej Karpathy, <a
# href="https://karpathy.github.io/2019/04/25/recipe/">A Recipe for Training Neural Networks</a> (2019).
# </font></td></tr></table>
#
# The next cell (collapsed) defines one function that measures a single photo. The cell after it runs that
# function on every photo in the Play Dataset and collects the numbers in a table called `image_stats`.

# %%
# @title Measurement helpers { display-mode: "form" }
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle
from PIL import ImageDraw

HIST_BINS = 32
CHANNEL_COLORS = {"R": "#C0392B", "G": "#27AE60", "B": "#2980B9"}
COLOUR_MODES = ["RGB", "L", "RGBA", "CMYK"]  # the modes we always report, even when the count is 0


def measure_image(path, thumb: int) -> tuple[dict, np.ndarray]:
    """Open one photo and return (stats, channel_hist).

    stats: width, height, mode (as stored in the file), aspect (width / height), long_side, megapixels,
    brightness (mean grey level 0–255 on a thumbnail), contrast (its standard deviation), colourfulness
    (mean distance of the RGB channels from grey: 0 for a grayscale picture), mean_r / mean_g / mean_b.
    channel_hist: a (3, HIST_BINS) array, the density histogram of the R, G and B channels.
    """
    with Image.open(path) as im:
        mode, (w, h) = im.mode, im.size  # read these before decoding: they describe the file itself
        im.draft("RGB", (thumb, thumb))  # fast reduced-size JPEG decode (no effect on other formats)
        rgb = im.convert("RGB")
        rgb.thumbnail((thumb, thumb))
        arr = np.asarray(rgb, dtype=np.float32)
    grey = arr.mean(axis=2)
    channels = arr.reshape(-1, 3)
    hist = np.stack(
        [np.histogram(channels[:, k], bins=HIST_BINS, range=(0, 256), density=True)[0] for k in range(3)]
    )
    stats = {
        "width": w,
        "height": h,
        "mode": mode,
        "aspect": w / h,
        "long_side": max(w, h),
        "megapixels": w * h / 1e6,
        "brightness": float(grey.mean()),
        "contrast": float(grey.std()),
        "colourfulness": float(np.abs(channels - grey.reshape(-1, 1)).mean()),
        "mean_r": float(channels[:, 0].mean()),
        "mean_g": float(channels[:, 1].mean()),
        "mean_b": float(channels[:, 2].mean()),
    }
    return stats, hist


def show_rows(rows_df: pd.DataFrame, title: str, caption=None, cols: int = 6) -> None:
    """A small gallery of the photos in `rows_df` (rows of image_stats), each framed in its class colour.

    `caption(row) -> str` writes the title above each photo; by default the size and the colour mode.
    """
    if caption is None:

        def caption(row):
            return f"{row['class']} · {row['width']}×{row['height']} {row['mode']}"

    n = len(rows_df)
    if n == 0:
        print(f"{title}: no photos to show.")
        return
    cols = min(cols, n)
    n_rows = -(-n // cols)
    fig, axes = plt.subplots(n_rows, cols, figsize=(2.4 * cols, 2.7 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), rows_df.iterrows(), strict=False):
        ax.axis("on")
        ax.imshow(load_thumb(row["path"]))
        frame(ax, CLASS_COLORS[row["class"]])
        ax.set_title(caption(row), fontsize=8)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    display(fig)
    plt.close(fig)


def show_extremes(stat: str, n: int = 4) -> None:
    """Gallery of the `n` photos with the lowest and the `n` with the highest value of `stat`.

    `stat` is a numeric column of `image_stats` (e.g. "brightness", "aspect", "megapixels", "colourfulness").
    """
    numeric = [c for c in image_stats.columns if pd.api.types.is_numeric_dtype(image_stats[c]) and c != "id"]
    if stat not in numeric:
        raise SystemExit(f"stat must be one of {numeric}, not {stat!r}")
    ordered = image_stats.sort_values(stat)
    n = max(1, min(n, len(ordered) // 2))

    def caption(row):
        return f"{row['class']} · {stat} = {row[stat]:.2f}\n{row['width']}×{row['height']} {row['mode']}"

    show_rows(ordered.head(n), f"lowest {stat}", caption, cols=n)
    show_rows(ordered.tail(n)[::-1], f"highest {stat}", caption, cols=n)


# %%
# Measure every photo once (a few seconds: each JPEG is decoded at thumbnail size) and keep the numbers.
THUMB = 64 if SMOKE else 128  # smaller thumbnails in smoke mode, same code (N-014)

_records, _hists = [], []
for _, _row in images.iterrows():
    _stats, _hist = measure_image(_row["path"], THUMB)
    _records.append({"id": _row["id"], "class": _row["class"], "path": _row["path"], **_stats})
    _hists.append(_hist)
image_stats = pd.DataFrame(_records)
CHANNEL_HISTS = np.stack(_hists)  # (n_images, 3, HIST_BINS)
RESULTS["image_stats_n"] = int(len(image_stats))

_same_size = image_stats[["width", "height"]].to_numpy() == images[["width", "height"]].to_numpy()
_agree = int(_same_size.all(axis=1).sum())
print(f"measured {len(image_stats)} photos · the manifest's width/height matches the file for {_agree}")
STAT_COLUMNS = ["width", "height", "aspect", "megapixels", "brightness", "contrast", "colourfulness"]
image_stats[STAT_COLUMNS].describe().round(2)

# %% [markdown]
# 💡 `describe()` is the quickest sanity check there is: look at `min` and `max` of every column before you
# look at anything else. A `min` width of 0, a `max` of 6000 or a brightness of 254 each tell a story.
#
# ### Sizes and shapes
#
# A model sees every photo at one fixed size, so the first question is how far the photos are from that.
#
# <table width="100%"><tr><td bgcolor="#E0F7F4"><font color="#00564D">
# <b>❓ Questions</b><br>
# Predict before you run. These photos were uploaded to Flickr by hundreds of people and collected by the COCO
# team; the park's photos will come from a few fixed cameras.
# </font></td></tr></table>
#
# **Q4.** 🟢 Do you expect all photos to have the same size? Sketch (in your head) where the dots of a
# width-vs-height scatter would land: one point, a line, a cloud? Then run the cell.

# %%
# Q4  Left: every photo as a dot (width across, height up) in its class colour; the dashed line is a square
#     photo. Right: the aspect ratio width/height, stacked by class.
fig, (ax_sc, ax_hist) = plt.subplots(1, 2, figsize=(13, 4.8))
for c in CLASSES:
    s = image_stats[image_stats["class"] == c]
    ax_sc.scatter(s["width"], s["height"], s=22, alpha=0.65, color=CLASS_COLORS[c], label=c, edgecolor="none")
_lim = float(max(image_stats["width"].max(), image_stats["height"].max())) * 1.08
ax_sc.plot([0, _lim], [0, _lim], "--", color=INK, lw=1)
ax_sc.set_xlim(0, _lim)
ax_sc.set_ylim(0, _lim)
ax_sc.set_xlabel("width (px)")
ax_sc.set_ylabel("height (px)")
ax_sc.set_title("Image sizes")
ax_sc.legend(frameon=False, fontsize=9, markerscale=1.6)

_bins = np.linspace(image_stats["aspect"].min() * 0.98, image_stats["aspect"].max() * 1.02, 28)
ax_hist.hist(
    [image_stats.loc[image_stats["class"] == c, "aspect"] for c in CLASSES],
    bins=_bins,
    stacked=True,
    color=[CLASS_COLORS[c] for c in CLASSES],
    label=CLASSES,
)
for ratio, name in [(3 / 4, "3:4"), (1.0, "1:1"), (4 / 3, "4:3")]:
    ax_hist.axvline(ratio, color=INK, lw=0.8, ls=":")
    ax_hist.text(ratio, ax_hist.get_ylim()[1] * 0.97, name, ha="center", va="top", fontsize=8, color=INK)
ax_hist.set_xlabel("aspect ratio = width / height  (left of 1: portrait, right: landscape)")
ax_hist.set_ylabel("photos")
ax_hist.set_title("Aspect ratios")
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# 💡 The dots do not fill the plane: they sit on a few lines and clusters, because the photos share a common
# longest side. The aspect histogram shows the same thing from the side: a landscape peak, a portrait peak and
# a long tail of panoramas and tall crops. Nothing here is "wrong", but it means a model fed 224×224 squares
# will see most of these photos squashed or cropped (the 🔀 Callout at the end of this section).
#
# **Worked answer (Q4).** Not one size, and not a random cloud either: COCO downloaded its photos from Flickr
# at one standard size, so the longest side is capped at the same value for almost every photo and the dots
# line up along "width = cap" and "height = cap". A dataset from the park's own cameras would look different:
# one dot per camera model (every frame of a camera has the same size), which is both tidier and a leakage
# risk you will meet in section 6 (the size gives away the camera).
#
# ### Colour modes and colours
#
# How a file stores colour is easy to forget and painful to discover during training: a single grayscale (`L`)
# or `RGBA` file in a batch of `RGB` arrays crashes the data loader, or silently gets a channel duplicated or
# dropped. The next cell counts the modes and draws the typical RGB histogram of each class.

# %%
# Colour modes per class (always listing RGB / L / RGBA / CMYK), then the mean R, G, B histogram per class.
_modes = COLOUR_MODES + sorted(set(image_stats["mode"]) - set(COLOUR_MODES))
mode_by_class = pd.crosstab(image_stats["mode"], image_stats["class"])
mode_by_class = mode_by_class.reindex(index=_modes, columns=CLASSES, fill_value=0)
image_stats["looks_gray"] = (image_stats["mode"] == "L") | (image_stats["colourfulness"] < 2.0)
print("colour modes: " + " · ".join(f"{m} {int(mode_by_class.loc[m].sum())}" for m in _modes))
_n_gray = int(image_stats["looks_gray"].sum())
print(f"photos that look grayscale (mode L, or RGB with equal channels): {_n_gray}")

fig = plt.figure(figsize=(13, 4.6))
_gs = fig.add_gridspec(
    2, 4, width_ratios=[1.4, 1, 1, 1], hspace=0.65, wspace=0.3, left=0.05, right=0.99, top=0.86, bottom=0.12
)
ax_modes = fig.add_subplot(_gs[:, 0])
_bottom = np.zeros(len(_modes))
for c in CLASSES:
    ax_modes.bar(_modes, mode_by_class[c].to_numpy(), bottom=_bottom, color=CLASS_COLORS[c], label=c)
    _bottom += mode_by_class[c].to_numpy()
ax_modes.set_ylabel("photos")
ax_modes.set_title("Colour mode as stored in the file", fontsize=10)
ax_modes.legend(frameon=False, fontsize=8)

_edges = np.linspace(0, 256, HIST_BINS + 1)
_centres = (_edges[:-1] + _edges[1:]) / 2
for k, c in enumerate(CLASSES):
    ax = fig.add_subplot(_gs[k // 3, 1 + k % 3])
    _mean_hist = CHANNEL_HISTS[(image_stats["class"] == c).to_numpy()].mean(axis=0)
    for ch_idx, ch in enumerate("RGB"):
        ax.plot(_centres, _mean_hist[ch_idx], color=CHANNEL_COLORS[ch], lw=1.4, label=ch)
    ax.set_title(f"{c} ({int((image_stats['class'] == c).sum())} photos)", color=CLASS_COLORS[c], fontsize=10)
    ax.set_yticks([])
    ax.set_xlim(0, 255)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(CLASS_COLORS[c])
        spine.set_linewidth(1.6)
    if k == 0:
        ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.text(0.67, 0.95, "Mean R, G, B histogram per class (pixel value 0–255)", ha="center", color=INK)
display(fig)
plt.close(fig)

# %% [markdown]
# 💡 Read the histograms as *typical colours of a class*: a class whose curves differ from the others' (a
# strong blue channel for a pond, high values across all three for a sandy savanna) is a class a model can
# recognise from colour alone. That is a shortcut you should know about: a dog on green grass in the paddock
# may be classified as "sheep" because the model learned "green = paddock = sheep".
#
# ### Brightness and the extremes
#
# Brightness is the mean grey level of a photo (0 black, 255 white). Its distribution tells you whether the
# dataset has night shots at all, and its extremes are the fastest route to broken or odd files.

# %%
# Brightness per class as box plots, in class colours. The dotted lines mark "very dark" and "very bright".
fig, ax = plt.subplots(figsize=(9, 3.8))
_data = [image_stats.loc[image_stats["class"] == c, "brightness"] for c in CLASSES]
_box = ax.boxplot(_data, tick_labels=CLASSES, patch_artist=True, widths=0.6, medianprops={"color": INK})
for patch, c in zip(_box["boxes"], CLASSES, strict=True):
    patch.set_facecolor(CLASS_COLORS[c])
    patch.set_alpha(0.85)
for level in (40, 215):
    ax.axhline(level, color=INK, lw=0.8, ls=":")
ax.set_ylim(0, 255)
ax.set_ylabel("brightness (mean grey level)")
ax.set_title("Brightness per class")
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# 💡 Every class covers most of the range, so the dataset has dim and bright photos, but the medians sit
# comfortably in the middle: these are daytime holiday photos. The park's cameras also run at dusk and at
# night, which this data cannot teach.
#
# Numbers point, eyes decide. `show_extremes(stat, n)` shows the `n` photos at both ends of any column of
# `image_stats`, so you can see *what* a very dark, very wide or very grey photo actually is.
#
# <table width="100%"><tr><td bgcolor="#FFF4D6"><font color="#7A4A00">
# <b>✏️ Your turn · Look at the extremes</b><br>
# Set <code>STAT_TO_SHOW</code> to one of <code>brightness, contrast, colourfulness, aspect, megapixels,
# long_side</code> and <code>N_EXTREMES</code> to how many photos per end you want, then run the cell. Try
# <code>"brightness"</code> first, then <code>"colourfulness"</code> (the low end is where grayscale pictures
# hide) and <code>"aspect"</code>.
# </font></td></tr></table>

# %%
# ✏️ YOUR TURN
STAT_TO_SHOW = "brightness"
N_EXTREMES = 4

if not isinstance(N_EXTREMES, int) or N_EXTREMES < 1:
    raise SystemExit(f"N_EXTREMES must be a positive integer, not {N_EXTREMES!r}")
n_extremes = min(N_EXTREMES, 3) if SMOKE else N_EXTREMES  # smoke mode caps the gallery, same code (N-014)
show_extremes(STAT_TO_SHOW, n_extremes)

# %% [markdown]
# 💡 The darkest photos are usually real (an animal in shade, a night shot with flash) and worth keeping: they
# are the kind of frame the park's cameras produce at dusk. The brightest are often snow, sand or an
# over-exposed sky. When one end shows a uniform grey rectangle, a thumbnail-sized file or a photo of a photo,
# you have found a file for section 5's cleaning list.
#
# ### The detective beat: a cluster is a clue
#
# A statistic that most photos share, and a few do not, is more than a curiosity. Photos from one pipeline
# (one camera, one download script, one phone model) come out with the same size and mode. A **cluster of odd
# sizes or modes therefore points to a different source**: another camera, another download, a batch someone
# converted by hand. On the park's cameras you would see it as two size clusters (two camera models) or a
# group of grayscale frames (the infrared night mode). Here we look for the same thing in COCO.
#
# <table width="100%"><tr><td bgcolor="#E0F7F4"><font color="#00564D">
# <b>❓ Questions</b><br>
# Look back at the scatter from Q4 before you run the next cell.
# </font></td></tr></table>
#
# **Q5.** 🟡 Which longest-side value do most photos share, and how many photos do you think do *not* share
# it? Write down a guess, then run the cell: it counts them, lists their sizes and shows them.

# %%
# Q5  The most common longest side, the photos that break the pattern, and the photos that look grayscale.
COMMON_LONG_SIDE = int(image_stats["long_side"].mode().iloc[0])
odd_size = image_stats[image_stats["long_side"] != COMMON_LONG_SIDE].sort_values("long_side")
gray_like = image_stats[image_stats["looks_gray"]]

print(
    f"{len(image_stats) - len(odd_size)} of {len(image_stats)} photos have a longest side of "
    f"{COMMON_LONG_SIDE} px; {len(odd_size)} do not."
)
if len(odd_size):
    _sizes = odd_size.apply(lambda r: f"{r['width']}×{r['height']}", axis=1).value_counts()
    print("odd sizes: " + " · ".join(f"{s} ×{n}" for s, n in _sizes.items()))
else:
    print("every photo in this sample went through the same resize step.")
print(f"{len(gray_like)} photos look grayscale.")

N_ODD = 6 if SMOKE else 12  # how many of the odd ones to show (N-014)
show_rows(odd_size.head(N_ODD), f"photos whose longest side is not {COMMON_LONG_SIDE} px")
show_rows(gray_like.head(N_ODD), "photos that look grayscale")

# %% [markdown]
# 💡 The odd sizes are small, not large: nobody uploaded a 4000-pixel photo here. The photos that fall below
# the common cap were *already* smaller than it when COCO fetched them, which is a hint about their origin
# (old uploads, low-resolution cameras, screenshots). A grayscale photo in a colour dataset is a similar hint,
# and both go on the list of things to decide about in section 5.
#
# **Worked answer (Q5).** On the full Play Dataset almost every photo has a longest side of 640 px, because
# COCO downloaded Flickr's "640 px" rendition, and only a handful fall below it (none may appear in a small
# sample). The number itself matters less than the habit: whenever a column of `image_stats` has one dominant
# value, ask "what made the others different?", because the answer is usually a different source, and a
# different source usually means different labels, lighting or quality. On your own dataset (section 11) this
# is the first question to ask about size and mode.
#
# ### Across tasks: what resizing does to boxes and masks
#
# Everything above was about the photo. The park's second and third jobs (count, outline) attach **boxes** and
# **masks** to the photo, and they are affected by the first thing every training pipeline does: resize.
#
# <table width="100%"><tr><td bgcolor="#EDE7F6"><font color="#311B92">
# <b>🔀 Across tasks · Resizing</b><br>
# <ul>
# <li><b>Classification:</b> squashing a 4:3 photo into a 224×224 square distorts aspect ratios. Usually
# harmless for "which animal", because every photo is distorted the same way. Cropping or padding keeps the
# shape at the cost of pixels or borders.</li>
# <li><b>Detection:</b> box coordinates live in pixels, so a box must be rescaled with the image: x and width
# by <code>new_width / width</code>, y and height by <code>new_height / height</code>. A forgotten rescale is
# the classic "the boxes are all in the top-left corner" bug.</li>
# <li><b>Segmentation:</b> a mask holds class ids, not colours. Resizing it with the usual smooth (bilinear)
# filter invents in-between values that are no class at all; masks need <b>nearest-neighbour</b>
# resizing.</li>
# </ul>
# </font></td></tr></table>
#
# **Q6.** 🟡 A box is stored as `(x, y, w, h)` in pixels on a `W × H` photo, and the photo is resized to
# 224×224. Write the formula for the new box. Then predict: how many distinct values does the mask hold after
# a nearest-neighbour resize, and after a bilinear one? Run the cell to check both.

# %%
# Q6  One photo, one box and one mask, squeezed to 224×224: the box is rescaled; the mask is resized two ways.
TARGET = 224
_polygon = annotations["segmentation"].apply(lambda s: isinstance(s, list))
_ann = annotations[(annotations["iscrowd"] == 0) & _polygon].iloc[0]
_photo = images.loc[images["id"] == _ann["image_id"]].iloc[0]
_color = CLASS_COLORS[_ann["class"]]

with Image.open(_photo["path"]) as _im:
    original = _im.convert("RGB")
W, H = original.size
sx, sy = TARGET / W, TARGET / H
x, y, w, h = _ann["bbox"]
new_box = (x * sx, y * sy, w * sx, h * sy)

mask = Image.new("L", (W, H), 0)
for _poly in _ann["segmentation"]:
    ImageDraw.Draw(mask).polygon(list(zip(_poly[::2], _poly[1::2], strict=True)), fill=1)
squashed = original.resize((TARGET, TARGET), Image.BILINEAR)
mask_nearest = np.asarray(mask.resize((TARGET, TARGET), Image.NEAREST))
mask_bilinear = np.asarray(mask.point(lambda v: v * 255).resize((TARGET, TARGET), Image.BILINEAR)) / 255.0

print(f"photo {W}×{H} → {TARGET}×{TARGET}: scale x by {sx:.3f}, y by {sy:.3f}")
_old = ", ".join(f"{v:.0f}" for v in (x, y, w, h))
_new = ", ".join(f"{v:.1f}" for v in new_box)
print(f"box  (x, y, w, h) = ({_old})  →  ({_new})")
print(
    f"mask values: original {np.unique(np.asarray(mask)).tolist()} · "
    f"nearest {np.unique(mask_nearest).tolist()} · bilinear {len(np.unique(mask_bilinear))} distinct values"
)

fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
axes[0].imshow(original)
axes[0].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=_color, lw=2))
axes[0].set_title(f"original {W}×{H} + box", fontsize=9)
axes[1].imshow(squashed)
axes[1].add_patch(Rectangle(new_box[:2], new_box[2], new_box[3], fill=False, edgecolor=_color, lw=2))
axes[1].set_title(f"resized {TARGET}×{TARGET} + rescaled box", fontsize=9)
axes[2].imshow(mask_nearest, cmap=ListedColormap(["white", _color]), vmin=0, vmax=1)
axes[2].set_title(f"mask, nearest: {len(np.unique(mask_nearest))} values", fontsize=9)
_soft_cmap = LinearSegmentedColormap.from_list("mask", ["white", _color])
axes[3].imshow(mask_bilinear, cmap=_soft_cmap, vmin=0, vmax=1)
axes[3].set_title(f"mask, bilinear: {len(np.unique(mask_bilinear))} values", fontsize=9)
for ax in axes:
    frame(ax, _color, width=2)
fig.suptitle(f"{_ann['class']}: what one resize does to the photo, its box and its mask", color=INK)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# 💡 The box follows the photo as long as you scale x and y separately. The mask does not survive a smooth
# resize: the bilinear version has a soft grey rim of values between 0 and 1, which are not "background" and
# not "the animal". With six classes those in-between values would become ids of *other* animals.
#
# **Worked answer (Q6).** New box = `(x · 224/W, y · 224/H, w · 224/W, h · 224/H)`; x-like values use the
# horizontal factor, y-like values the vertical one, and on a landscape photo the two factors differ, so the
# box is squashed exactly like the photo. Nearest-neighbour resizing keeps the two original values (0 and 1).
# Bilinear resizing produces many distinct values along every edge of the animal, and the number grows with
# the length of the boundary, so thin parts (legs, tails, a bird's beak) suffer most. That is why segmentation
# pipelines resize images with a smooth filter and masks with nearest-neighbour.
#
# **Takeaway.** `image_stats` costs seconds and already answers four of the checklist's questions (size,
# shape, colour mode, colour) for any folder of images, labelled or not. Section 4 adds the labels and asks
# the question every count depends on: how many photos of each class do we have, and does it matter?
