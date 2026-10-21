# %% [markdown]
# ## 3. Image inspection

# %% [markdown]
# <!-- callout: objective | Objective -->
# Measure sizes, aspect ratios, colour modes and colour statistics, and read a cluster of odd values as a clue
# about where the photos came from.

# %% [markdown]
# Now the computer measures every photo once. No label is needed, so this is the first check you can run on
# *any* folder of images, including your own in section 11.

# %% [markdown]
# <!-- callout: recipe | Recipe · Step 1, continued: search, filter, sort -->
# *"Once you get a qualitative sense it is also a good idea to write some simple code to search/filter/sort
# by whatever you can think of (e.g. type of label, size of annotations, number of annotations, etc.) and
# visualize their distributions and the outliers along any axis. The outliers especially almost always uncover
# some bugs in data quality or preprocessing."*
#
# Andrej Karpathy, [A Recipe for Training Neural
# Networks](https://karpathy.github.io/2019/04/25/recipe/) (2019).

# %% [markdown]
# The numbers collected per photo, and how each is measured (brightness, contrast and colour are measured on a
# small thumbnail of the photo):
#
# - **width, height:** the pixel columns and rows stored in the file.
# - **aspect:** width ÷ height. 1 is square, above 1 landscape, below 1 portrait.
# - **long_side:** the larger of width and height. **megapixels:** width × height ÷ 1,000,000.
# - **mode:** how the file stores colour: `RGB` three channels, `L` one grey channel, `RGBA` RGB plus
#   transparency, `CMYK` print inks.
# - **brightness:** the mean grey level (R, G, B averaged per pixel, then over the photo): 0 black, 255 white.
# - **contrast:** the standard deviation of that grey level: low is flat or foggy, high is deep shadow next to
#   bright highlight.
# - **colourfulness** (Hasler & Süsstrunk 2003): `sqrt(σ_rg² + σ_yb²) + 0.3 · sqrt(μ_rg² + μ_yb²)`, with
#   `rg = R − G` and `yb = (R + G)/2 − B` per pixel, and σ, μ their standard deviation and mean.
#   A grayscale picture scores exactly 0.
# - **mean_r, mean_g, mean_b:** the average value of each colour channel.

# %% [markdown]
# One collapsed cell defines the function that measures a single photo; the next one runs it on every photo
# of the Play Dataset and keeps the numbers in a table called `image_stats`.

# %%
# @title Measurement helpers { display-mode: "form" }
from matplotlib.patches import Rectangle
from PIL import ImageDraw

HIST_BINS = 32
CHANNEL_COLORS = {"R": "#C0392B", "G": "#27AE60", "B": "#2980B9"}
COLOUR_MODES = ["RGB", "L", "RGBA", "CMYK"]  # the modes we always report, even when the count is 0


def measure_image(path, thumb: int) -> tuple[dict, np.ndarray]:
    """Open one photo and return (stats, channel_hist).

    stats: width, height, mode (as stored in the file), aspect (width / height), long_side, megapixels,
    brightness (mean grey level 0–255 on a thumbnail), contrast (its standard deviation), colourfulness
    (Hasler & Süsstrunk 2003: 0 for a grayscale picture), mean_r / mean_g / mean_b.
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
    rg = channels[:, 0] - channels[:, 1]  # red minus green
    yb = (channels[:, 0] + channels[:, 1]) / 2 - channels[:, 2]  # yellow (mean of R and G) minus blue
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
        "colourfulness": float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())),
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


def _fmt(value) -> str:
    """A number for a caption: 640 stays 640, 1.33333 becomes 1.333."""
    return f"{value:.4g}"


def _closest_to_median(stat: str):
    """The row of `image_stats` whose value of `stat` is closest to the median (the 'typical' photo)."""
    col = image_stats[stat]
    return image_stats.iloc[int((col - col.median()).abs().to_numpy().argmin())]


def _photo(ax, row, caption: str) -> None:
    """Draw one photo with a caption, framed in its class colour."""
    ax.axis("on")
    ax.imshow(load_thumb(row["path"]))
    frame(ax, CLASS_COLORS[row["class"]])
    ax.set_title(caption, fontsize=8)


def show_extremes(stat: str, n: int = 4) -> None:
    """The `n` photos with the lowest value of `stat`, the median ('average') photo, and the `n` highest.

    `stat` is a numeric column of `image_stats` (e.g. "brightness", "aspect", "megapixels", "colourfulness").
    Three rows: lowest (the minimum first), the typical photo in the middle, highest (the maximum first).
    """
    numeric = [c for c in image_stats.columns if pd.api.types.is_numeric_dtype(image_stats[c]) and c != "id"]
    if stat not in numeric:
        raise SystemExit(f"stat must be one of {numeric}, not {stat!r}")
    ordered = image_stats.sort_values(stat)
    n = max(1, min(n, len(ordered) // 2))

    def caption(tag, row):
        size = f"{row['width']}×{row['height']} {row['mode']}"
        return f"{tag}: {stat} = {_fmt(row[stat])}\n{row['class']} · {size}"

    fig, axes = plt.subplots(3, n, figsize=(2.4 * n, 2.8 * 3), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes[0], ordered.head(n).iterrows(), strict=False):
        _photo(ax, row, caption("lowest", row))
    _photo(axes[1, n // 2], _closest_to_median(stat), caption("median", _closest_to_median(stat)))
    for ax, (_, row) in zip(axes[2], ordered.tail(n)[::-1].iterrows(), strict=False):
        _photo(ax, row, caption("highest", row))
    fig.suptitle(f"{stat}: the {n} lowest, the typical (median) photo, the {n} highest", color=INK)
    fig.tight_layout()
    display(fig)
    plt.close(fig)


def show_min_median_max(stats: list[str]) -> None:
    """Per statistic: the photo with its minimum, a typical one (median) and the maximum; two per row.

    Two extra blocks show photos with an odd value: colourfulness close to 0, stored as `L` or as `RGB`.
    """
    spec = []
    for stat in stats:
        items = [
            ("min", image_stats.loc[image_stats[stat].idxmin()]),
            ("median", _closest_to_median(stat)),
            ("max", image_stats.loc[image_stats[stat].idxmax()]),
        ]
        spec.append((stat, [(f"{tag} = {_fmt(row[stat])}\n{row['class']}", row) for tag, row in items]))
    no_colour = image_stats["colourfulness"] < 1
    for mode, label in (("L", "colourfulness ≈ 0\nstored as L"), ("RGB", "colourfulness ≈ 0\nstored as RGB")):
        odd = image_stats[no_colour & (image_stats["mode"] == mode)].head(3)
        if len(odd):
            items = [(f"{_fmt(r['colourfulness'])}\n{r['class']}", r) for _, r in odd.iterrows()]
            spec.append((label, items))
    n_rows = -(-len(spec) // 2)  # two blocks of three photos side by side, a narrow gap between them
    widths = {"width_ratios": [1, 1, 1, 0.45, 1, 1, 1]}
    fig, axes = plt.subplots(n_rows, 7, figsize=(13, 2.0 * n_rows), squeeze=False, gridspec_kw=widths)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (label, items) in enumerate(spec):
        r, side = divmod(k, 2)
        for j, (caption, row) in enumerate(items):
            _photo(axes[r, side * 4 + j], row, caption)
        axes[r, side * 4].set_ylabel(label, color=INK, fontsize=10, fontweight="bold")
    fig.tight_layout()
    display(fig)
    plt.close(fig)


# %%
# @title Measure every photo once and keep the numbers in image_stats { display-mode: "form" }
# (a few seconds: each JPEG is decoded at thumbnail size)
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
image_stats[STAT_COLUMNS].agg(["min", "median", "mean", "max"]).T.round(2)

# %% [markdown]
# 💡 Read the `min` and `max` of every row before anything else: a width of 0, a maximum of 6000 or a
# brightness of 254 each tell a story. A number is a summary; the photos behind it are the evidence. The next
# cell shows, for every statistic, the photo with the minimum, a typical one (the median) and the maximum.

# %%
# @title Show the min, median and max photo of every statistic { display-mode: "form" }
show_min_median_max(STAT_COLUMNS)

# %% [markdown]
# 💡 Look at the median photo first: that is what a typical photo of this dataset looks like. The extremes
# then show how far the dataset stretches. The last blocks hold photos with *no colour at all*, stored in two
# different ways.

# %% [markdown]
# ### Sizes and shapes
#
# A model sees every photo at one fixed size, so the first question is how far the photos are from that.

# %% [markdown]
# <!-- callout: questions | Questions -->
# Predict before you run. These photos were uploaded to Flickr by hundreds of people and collected by the COCO
# team; the park's photos will come from a few fixed cameras.

# %% [markdown]
# **Q4.** 🟢 Do you expect all photos to have the same size? What shape would the dots of a width-vs-height
# scatter make: one point, a line, or a cloud? Pick the closest one (the real plot may not match any exactly),
# then run the cell.

# %%
# @title Q4 hint: three possible scatter shapes { display-mode: "form" }
# Three tiny example scatters of width (across) against height (up), to pick from.
_rng = np.random.default_rng(SEED)
_line_w = _rng.uniform(300, 640, 40)
_examples = {
    "one point\nall photos the same size": (np.full(40, 640.0), np.full(40, 480.0)),
    "a line\nsizes differ, same shape": (_line_w, _line_w * 0.75),
    "a cloud\nany size, any shape": (_rng.uniform(200, 640, 40), _rng.uniform(200, 640, 40)),
}
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6))
for ax, (title, (xs, ys)) in zip(axes, _examples.items(), strict=True):
    ax.scatter(xs, ys, s=14, color=INK, alpha=0.7, edgecolor="none")
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 700)
    ax.set_xlabel("width (px)", fontsize=8)
    ax.set_ylabel("height (px)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)
fig.tight_layout()
display(fig)
plt.close(fig)

# %%
# @title Q4 Plot image sizes and aspect ratios by class { display-mode: "form" }
# Left: every photo as a dot (width across, height up) in its class colour; the dashed line is a square
# photo. Right: the aspect ratio width/height, stacked by class.
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
# **Worked answer (Q4).** 💡 Closest to a line: two lines meeting in an L. COCO fetched its Flickr photos at
# one standard size, so the longest side is capped: width = cap for landscape photos, height = cap for
# portrait ones (the aspect histogram shows the same two peaks). At the park, each camera model would give one
# dot, and a model could learn "this size = this camera = this animal" (leakage, section 6).

# %% [markdown]
# ### Colour modes and colours
#
# How a file stores colour is easy to forget and painful to discover during training: one grayscale (`L`) or
# `RGBA` file in a batch of `RGB` arrays crashes the data loader, or silently gets a channel duplicated or
# dropped. The next cell counts the modes, shows the photos that look grayscale, and draws the typical RGB
# histogram of each class.

# %%
# @title Count colour modes, show gray photos, plot RGB histograms { display-mode: "form" }
# Colour modes per class (always listing RGB / L / RGBA / CMYK), the grayscale-looking photos, then the mean
# R, G, B histogram per class.
_modes = COLOUR_MODES + sorted(set(image_stats["mode"]) - set(COLOUR_MODES))
mode_by_class = pd.crosstab(image_stats["mode"], image_stats["class"])
mode_by_class = mode_by_class.reindex(index=_modes, columns=CLASSES, fill_value=0)
image_stats["looks_gray"] = (image_stats["mode"] == "L") | (image_stats["colourfulness"] < 2.0)
print("colour modes: " + " · ".join(f"{m} {int(mode_by_class.loc[m].sum())}" for m in _modes))
_n_gray = int(image_stats["looks_gray"].sum())
print(f"photos that look grayscale (mode L, or RGB with colourfulness below 2): {_n_gray}")
N_GRAY = 6 if SMOKE else 12  # how many grayscale-looking photos to show (N-014); there may be none
show_rows(image_stats[image_stats["looks_gray"]].head(N_GRAY), "photos that look grayscale")

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
# 💡 The photos above are the grayscale-looking ones (stored as `L`, or as `RGB` with almost no colour).
# Section 5 keeps them: files stored as `L` get a note in a `flags` column, and every photo is opened with
# `.convert("RGB")`, so a grayscale file becomes three equal channels instead of crashing a batch.
#
# Read the histograms as *typical colours of a class* (a strong blue channel for a pond, high values across
# all three for a sandy savanna). If the colour curves differ per class, a model can cheat by using colour
# instead of the animal: a dog on green grass might be called "sheep".

# %% [markdown]
# ### Brightness and the extremes
#
# The brightness distribution tells you whether the dataset has night shots at all, and its extremes are the
# fastest route to broken or odd files. To read a violin: the wider the shape, the more photos at that value;
# the line marks the median.

# %%
# @title Plot brightness per class as violins { display-mode: "form" }
# Brightness per class as violin plots, in class colours. The dotted lines mark "very dark" and "very bright".
fig, ax = plt.subplots(figsize=(9, 3.8))
_data = [image_stats.loc[image_stats["class"] == c, "brightness"].to_numpy() for c in CLASSES]
_violins = ax.violinplot(_data, widths=0.8, showmedians=True)
for body, c in zip(_violins["bodies"], CLASSES, strict=True):
    body.set_facecolor(CLASS_COLORS[c])
    body.set_alpha(0.85)
_violins["cmedians"].set_color(INK)
ax.set_xticks(range(1, len(CLASSES) + 1), CLASSES)
for level in (40, 215):
    ax.axhline(level, color=INK, lw=0.8, ls=":")
ax.set_ylim(0, 255)
ax.set_ylabel("brightness (mean grey level)")
ax.set_title("Brightness per class")
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# 💡 Every class covers most of the range, but the medians sit in the middle: daytime holiday photos. The
# park's cameras also run at dusk and at night, which this data cannot teach.
#
# Numbers point, eyes decide. `show_extremes(stat, n)` shows the `n` photos at both ends of any column of
# `image_stats`, with the median photo in the middle for comparison.

# %% [markdown]
# <!-- callout: your-turn | Your turn · Look at the extremes -->
# **Look for:** which class has the darkest photos, and are they broken or just dusk?
#
# Set `STAT_TO_SHOW` to one of `brightness, contrast, colourfulness, aspect, megapixels, long_side` and
# `N_EXTREMES` to how many photos per end you want, then run the cell. Try `"brightness"` first, then
# `"colourfulness"` (the low end is where grayscale pictures hide) and `"aspect"`.

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
# resemble the park's dusk frames. The brightest are often snow, sand or an over-exposed sky. A uniform grey
# rectangle or a photo of a screen at either end is a file for section 5's cleaning list.

# %% [markdown]
# ### The detective beat: a cluster is a clue
#
# One camera or download script gives one size and one mode, so a **group of odd sizes or modes points to a
# different source**. At the park:
#
# 1. **Two camera models** → two clusters in the width-vs-height scatter. Check every class is in both.
# 2. **An infrared night mode** → a group of grayscale frames among colour ones. A colour-only model has never
#    seen those, and may fail exactly at night.
# 3. **What to do** → decide per group: keep it, convert it to RGB, or drop it.

# %%
# @title Toy example: what a cluster of odd sizes and modes looks like { display-mode: "form" }
_rng = np.random.default_rng(SEED)
fig, (ax_sizes, ax_modes) = plt.subplots(1, 2, figsize=(8.4, 2.8))
_cameras = [((1920, 1080, 30), INK, "camera A"), ((1280, 720, 12), "#0B6E79", "camera B")]
for (_cx, _cy, _n), _col, _name in _cameras:
    ax_sizes.scatter(_rng.normal(_cx, 15, _n), _rng.normal(_cy, 10, _n), s=14, color=_col, label=_name)
ax_sizes.set_xlabel("width (px)", fontsize=8)
ax_sizes.set_ylabel("height (px)", fontsize=8)
ax_sizes.set_title("two camera models: two size clusters", fontsize=9)
ax_sizes.legend(frameon=False, fontsize=8)
ax_modes.bar(["colour (RGB)", "night frames (L)"], [90, 14], color=[INK, "#9AA5A8"])
ax_modes.set_ylabel("frames", fontsize=8)
ax_modes.set_title("infrared night mode: a few grayscale frames", fontsize=9)
for ax in (ax_sizes, ax_modes):
    ax.tick_params(labelsize=7)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# <!-- callout: questions | Questions -->
# The toy picture above shows both clues. Look back at the real scatter from Q4 before you run the next cell.

# %% [markdown]
# **Q5.** 🟡 Which longest-side value do most photos share, and how many photos do you think do *not* share
# it? Write down a guess, then run the cell: it counts them, lists their sizes and shows them.

# %%
# @title Q5 Find the common longest side and show the photos that break the pattern { display-mode: "form" }
COMMON_LONG_SIDE = int(image_stats["long_side"].mode().iloc[0])
odd_size = image_stats[image_stats["long_side"] != COMMON_LONG_SIDE].sort_values("long_side")

print(
    f"{len(image_stats) - len(odd_size)} of {len(image_stats)} photos have a longest side of "
    f"{COMMON_LONG_SIDE} px; {len(odd_size)} do not."
)
if len(odd_size):
    _sizes = odd_size.apply(lambda r: f"{r['width']}×{r['height']}", axis=1).value_counts()
    print("odd sizes: " + " · ".join(f"{s} ×{n}" for s, n in _sizes.items()))
else:
    print("every photo in this sample went through the same resize step.")

N_ODD = 6 if SMOKE else 12  # how many of the odd ones to show (N-014)
show_rows(odd_size.head(N_ODD), f"photos whose longest side is not {COMMON_LONG_SIDE} px")

# %% [markdown]
# **Worked answer (Q5).** 💡 On the full Play Dataset almost every photo has a longest side of 640 px
# (Flickr's "640 px" rendition); a handful are smaller, never larger: old uploads, low-resolution cameras,
# screenshots (a small sample may show none). Like the grayscale photos they are a group with another source,
# and section 5 keeps them (only extreme sizes get a note in `flags`). The habit: one dominant value → ask
# "what made the others different?".

# %% [markdown]
# ### Across tasks: what resizing does to boxes and masks
#
# Everything above was about the photo. The park's second and third jobs (count, outline) attach **boxes** and
# **masks** to it, and the first thing every training pipeline does is resize.

# %% [markdown]
# <!-- callout: across-tasks | Across tasks · Resizing -->
# - **Classification:** squashing a 4:3 photo into a 224×224 square distorts it, but every photo is distorted
#   the same way, so it is usually harmless. Cropping or padding keeps the shape at the cost of pixels or
#   borders.
# - **Detection:** a box is in pixels, so rescale it with the image: x and width by `new_width / width`, y and
#   height by `new_height / height`. Forget it and "all the boxes sit in the top-left corner".
# - **Segmentation:** a mask holds class ids, not colours. Smooth (bilinear) resizing invents in-between
#   values that are no class; masks need **nearest-neighbour** resizing.

# %% [markdown]
# **Q6.** 🟡 A 640×427 photo is resized to 224×224. Answer both parts, then run the cell to check them:
#
# - **(a) Box:** the photo holds a box `(x, y, w, h) = (100, 50, 200, 150)` in pixels. What is the new box?
# - **(b) Mask:** its mask holds only 0 (background) and 1 (animal). How many distinct values does it hold
#   after a nearest-neighbour resize, and after a bilinear one?

# %%
# @title Q6 Rescale a box, resize a mask two ways and show both as heatmaps { display-mode: "form" }
TARGET = 224
_w0, _h0, _box0 = 640, 427, (100, 50, 200, 150)  # the numbers of the question
_new0 = (_box0[0] * TARGET / _w0, _box0[1] * TARGET / _h0, _box0[2] * TARGET / _w0, _box0[3] * TARGET / _h0)
print(f"question: {_w0}×{_h0} → {TARGET}×{TARGET}, box {_box0} → ({', '.join(f'{v:.1f}' for v in _new0)})")

# One real photo from the Play Dataset, with its box and mask:
_polygon = annotations["segmentation"].apply(lambda s: isinstance(s, list))
_ann = annotations[(annotations["iscrowd"] == 0) & _polygon].iloc[0]
_sample = images.loc[images["id"] == _ann["image_id"]].iloc[0]  # not `_photo`: that is a helper above
_color = CLASS_COLORS[_ann["class"]]

with Image.open(_sample["path"]) as _im:
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

fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), layout="constrained")
axes[0].imshow(original)
axes[0].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=_color, lw=2))
axes[0].set_title(f"original {W}×{H} + box", fontsize=9)
axes[1].imshow(squashed)
axes[1].add_patch(Rectangle(new_box[:2], new_box[2], new_box[3], fill=False, edgecolor=_color, lw=2))
axes[1].set_title(f"resized {TARGET}×{TARGET} + rescaled box", fontsize=9)
axes[2].imshow(mask_nearest, cmap="viridis", vmin=0, vmax=1)
axes[2].set_title(f"mask, nearest: {len(np.unique(mask_nearest))} values", fontsize=9)
_heat = axes[3].imshow(mask_bilinear, cmap="viridis", vmin=0, vmax=1)
axes[3].set_title(f"mask, bilinear: {len(np.unique(mask_bilinear))} values", fontsize=9)
for ax in axes:
    frame(ax, _color, width=2)
fig.colorbar(_heat, ax=axes[2:], fraction=0.025, pad=0.02, label="mask value (0 = background, 1 = animal)")
fig.suptitle(f"{_ann['class']}: what one resize does to the photo, its box and its mask (masks as heatmaps)",
             color=INK)
display(fig)
plt.close(fig)

# Zoom on 6×6 pixels at the animal's edge, with the value of every pixel written in.
_edge = np.argwhere((mask_bilinear > 0.3) & (mask_bilinear < 0.7))
if len(_edge):
    _r0, _c0 = (int(np.clip(v - 3, 0, TARGET - 6)) for v in _edge[len(_edge) // 2])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.9))
    for ax, data, name in ((axes[0], mask_nearest, "nearest"), (axes[1], mask_bilinear, "bilinear")):
        patch = data[_r0 : _r0 + 6, _c0 : _c0 + 6]
        ax.imshow(patch, cmap="viridis", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(patch):
            ink = "white" if v < 0.6 else "black"  # readable on dark and on light cells
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=ink)
        ax.set_title(f"{name}: 6×6 pixels at the edge", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    display(fig)
    plt.close(fig)

# %% [markdown]
# **Worked answer (Q6).** 💡 **(a)** New box = `(x · 224/W, y · 224/H, w · 224/W, h · 224/H)`: here x and w
# times 224/640 = 0.35, y and h times 224/427 = 0.525, giving `(35, 26.2, 70, 78.7)`, squashed like the photo.
# **(b)** Nearest-neighbour keeps the two values (0 and 1); bilinear invents many in-between values along
# every edge (the rim in the heatmap), so thin parts (legs, tails, a beak) suffer most.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · Where the in-between values come from -->
# Bilinear interpolation gives every new pixel a weighted average of the four nearest old pixels,
# `v00 v10 v01 v11`:
#
# `new = (1−a)(1−b)·v00 + a(1−b)·v10 + (1−a)b·v01 + a·b·v11`
#
# `a` and `b` (0 to 1) say how far the new pixel lies from them. At an edge some of the four are 0 and some
# are 1, so the result is a fraction such as 0.25 or 0.5.
#
# **Why not just round the fractions?** With 0 = background and 1 = dog, a pixel of 0.5 is neither. With
# several class ids (0 background, 1 cat, 2 dog), an edge between background (0) and dog (2) averages to 1,
# which is a *cat that is not there*. Nearest-neighbour copies one existing neighbour, so it can only produce
# ids that exist: photos get a smooth filter, masks get nearest-neighbour.

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **When you see** a size, shape or mode that most photos share and a few do not → **do** ask what source
#   made the others different, then keep, convert to RGB, or drop them.
# - **When you read** a statistic → **do** look at the min, median and max photo behind it.
# - **When you resize** photos for training → **do** rescale boxes by the same x and y factors, and resize
#   masks with nearest-neighbour.
