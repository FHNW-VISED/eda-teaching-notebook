# %% [markdown]
# ### Quantitative Inspection
#
# Now that we've explored individual images and annotations, let's examine the dataset as a whole.
#
# We'll use statistics and visualizations to investigate class distributions, image properties, and outliers that may reveal potential data quality issues.

# %%
# @title 🧾 Recipe · Step 1, continued { display-mode: "form" }
callout("recipe", "🧾 Recipe · Step 1, continued: Search, Filter, Sort", r"""
<p style="margin:6px 0 0 0;"><em>
"Once you get a qualitative sense it is also a good idea to write some simple code to search/filter/sort
by whatever you can think of (e.g. type of label, size of annotations, number of annotations, etc.) and
visualize their distributions and the outliers along any axis. The outliers especially almost always uncover
some bugs in data quality or preprocessing."
</em></p>

<p style="margin:6px 0 0 0;">
Andrej Karpathy,
<a href="https://karpathy.github.io/2019/04/25/recipe/"
   target="_blank" rel="noopener"
   style="color:inherit;text-decoration:underline;">A Recipe for Training Neural Networks</a> (2019).
</p>
""")


# %%
# @title 📘 Image Properties: What Can We Measure? { display-mode: "form" }
callout("theory", "📘 Image Properties: What Can We Measure?", r"""
<p style="margin:6px 0;">
We can measure image properties to identify patterns, unusual values, and potential data quality issues.
These measurements do not require class labels and can be applied to any image dataset.
</p>

<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:4px 0;"><strong>Dimensions:</strong> Width and height in pixels, and aspect ratio (width ÷ height).</li>
<li style="margin:4px 0;"><strong>Color mode:</strong> How colors are stored, such as RGB (three channels) or grayscale (one channel).</li>
<li style="margin:4px 0;"><strong>Brightness:</strong> Average pixel intensity, from 0 (black) to 255 (white).</li>
<li style="margin:4px 0;"><strong>Contrast:</strong> Variation in pixel intensity, indicating differences between dark and bright regions.</li>
<li style="margin:4px 0;"><strong>Colorfulness:</strong> How strongly image colors differ from gray.</li>
</ul>
""")


# %% [markdown]
# The next code cells calculate these properties for each image and store the results in a table called `image_stats`.

# %%
# @title Helpers: Measurement { display-mode: "form" }
HIST_BINS = 32
CHANNEL_COLORS = {"R": "#C0392B", "G": "#27AE60", "B": "#2980B9"}
COLOR_MODES = ["RGB", "L", "RGBA", "CMYK"]  # the modes we always report, even when the count is 0


def measure_image(path, thumb: int) -> tuple[dict, np.ndarray]:
    """Open one photo and return (stats, channel_hist).

    stats: width, height, mode (as stored in the file), aspect (width / height), long_side, megapixels,
    brightness (mean gray level 0–255 on a thumbnail), contrast (its standard deviation), colorfulness
    (Hasler & Süsstrunk 2003: 0 for a grayscale picture), mean_r / mean_g / mean_b.
    channel_hist: a (3, HIST_BINS) array, the density histogram of the R, G and B channels.
    """
    with Image.open(path) as im:
        mode, (w, h) = im.mode, im.size  # read these before decoding: they describe the file itself
        im.draft("RGB", (thumb, thumb))  # fast reduced-size JPEG decode (no effect on other formats)
        rgb = im.convert("RGB")
        rgb.thumbnail((thumb, thumb))
        arr = np.asarray(rgb, dtype=np.float32)
    gray = arr.mean(axis=2)
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
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "colorfulness": float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())),
        "mean_r": float(channels[:, 0].mean()),
        "mean_g": float(channels[:, 1].mean()),
        "mean_b": float(channels[:, 2].mean()),
    }
    return stats, hist


def show_rows(rows_df: pd.DataFrame, title: str, caption=None, cols: int = 6) -> None:
    """A small gallery of the photos in `rows_df` (rows of image_stats), each framed in its class color.

    `caption(row) -> str` writes the title above each photo; by default the size and the color mode.
    """
    if caption is None:

        def caption(row):
            return f"{row['class']} · {row['width']}×{row['height']} {row['mode']}"

    n = len(rows_df)
    if n == 0:
        print(f"{title}: no images to show.")
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


def _closest_to_median(stats_df: pd.DataFrame, stat: str):
    """The row of `stats_df` whose value of `stat` is closest to the median (the 'typical' photo)."""
    col = stats_df[stat]
    return stats_df.iloc[int((col - col.median()).abs().to_numpy().argmin())]


def _photo(ax, row, caption: str) -> None:
    """Draw one photo with a caption, framed in its class color."""
    ax.axis("on")
    ax.imshow(load_thumb(row["path"]))
    frame(ax, CLASS_COLORS[row["class"]])
    ax.set_title(caption, fontsize=8)


def show_extremes(stats_df: pd.DataFrame, stat: str, n: int = 4) -> None:
    """The `n` photos with the lowest value of `stat`, the median ('average') photo, and the `n` highest.

    `stat` is a numeric column of `stats_df`, e.g. `image_stats` (e.g. "brightness", "aspect", "megapixels", "colorfulness").
    Three rows: lowest (the minimum first), the typical photo in the middle, highest (the maximum first).
    """
    numeric = [c for c in stats_df.columns if pd.api.types.is_numeric_dtype(stats_df[c]) and c != "id"]
    if stat not in numeric:
        raise SystemExit(f"stat must be one of {numeric}, not {stat!r}")
    ordered = stats_df.sort_values(stat)
    n = max(1, min(n, len(ordered) // 2))

    def caption(tag, row):
        size = f"{row['width']}×{row['height']} {row['mode']}"
        return f"{tag}: {stat} = {_fmt(row[stat])}\n{row['class']} · {size}"

    fig, axes = plt.subplots(3, n, figsize=(2.4 * n, 2.8 * 3), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes[0], ordered.head(n).iterrows(), strict=False):
        _photo(ax, row, caption("lowest", row))
    _photo(axes[1, n // 2], _closest_to_median(stats_df, stat), caption("median", _closest_to_median(stats_df, stat)))
    for ax, (_, row) in zip(axes[2], ordered.tail(n)[::-1].iterrows(), strict=False):
        _photo(ax, row, caption("highest", row))
    fig.suptitle(f"{stat}: the {n} lowest, the typical (median) image, the {n} highest", color=INK)
    fig.tight_layout()
    display(fig)
    plt.close(fig)


def show_min_median_max(stats_df: pd.DataFrame, stats: list[str]) -> None:
    """Show min, median, and max images for each property, with headings above."""

    spec = []
    for stat in stats:
        items = [
            ("Min", stats_df.loc[stats_df[stat].idxmin()]),
            ("Median", _closest_to_median(stats_df, stat)),
            ("Max", stats_df.loc[stats_df[stat].idxmax()]),
        ]
        spec.append((stat, items))

    n_rows = -(-len(spec) // 2)
    fig = plt.figure(figsize=(13, 2.8 * n_rows))
    outer = fig.add_gridspec(n_rows, 2, hspace=0.65, wspace=0.25)

    for k, (stat, items) in enumerate(spec):
        r, c = divmod(k, 2)
        inner = outer[r, c].subgridspec(1, 3, wspace=0.15)

        for j, (tag, row) in enumerate(items):
            ax = fig.add_subplot(inner[0, j])
            _photo(ax, row, f"{tag}: {_fmt(row[stat])}\n{row['class']}")

        # Property heading centered above the three images
        x0 = outer[r, c].get_position(fig).x0
        x1 = outer[r, c].get_position(fig).x1
        y1 = outer[r, c].get_position(fig).y1

        fig.text((x0 + x1) / 2, y1 + 0.025,
                 stat.replace("_", " ").title(),
                 ha="center", va="bottom",
                 fontsize=12, fontweight="bold", color=INK)

    plt.show()
    plt.close(fig)



# %%
# @title Measure Image Dimensions and Other Properties { display-mode: "form" }
# (a few seconds: each JPEG is decoded at thumbnail size)
THUMB = 64 if SMOKE else 128  # smaller thumbnails in smoke mode, same code (N-014)

_records, _hists = [], []
for _, _row in images.iterrows():
    _stats, _hist = measure_image(_row["path"], THUMB)
    _records.append({"id": _row["id"], "class": _row["class"], "path": _row["path"], **_stats})
    _hists.append(_hist)
image_stats = pd.DataFrame(_records)
channel_hists = np.stack(_hists)  # (n_images, 3, HIST_BINS)
RESULTS["image_stats_n"] = int(len(image_stats))

_same_size = image_stats[["width", "height"]].to_numpy() == images[["width", "height"]].to_numpy()
_agree = int(_same_size.all(axis=1).sum())
print(f"measured {len(image_stats)} images · the manifest's width/height matches the file for {_agree}")
STAT_COLUMNS = ["width", "height", "aspect", "megapixels", "brightness", "contrast", "colorfulness"]
image_stats[STAT_COLUMNS].agg(["min", "median", "mean", "max"]).T.round(2)

# %% [markdown]
# #### 🟢 Image Properties: What Do the Statistics Reveal?
#
# The next visualization shows images with the minimum, median, and maximum values for each property.
#
# **Observe:**
# - How do images with minimum and maximum values differ from those near the median?
# - Which extreme values might indicate unusual images or potential data quality issues?

# %%
# @title 🔎 Inspect Images with Minimum, Median, and Maximum Property Values { display-mode: "form" }
show_min_median_max(image_stats, STAT_COLUMNS)

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:4px 0;"><strong>Image dimensions:</strong> Most images are relatively small, with widths up to 640 pixels. Some have unusual aspect ratios, such as a very wide zebra image.</li>
<li style="margin:4px 0;"><strong>Brightness:</strong> Images range from very dark to very bright. Animals may be harder to recognize in extreme lighting conditions.</li>
<li style="margin:4px 0;"><strong>Potential issues:</strong> Extreme aspect ratios and lighting conditions deserve closer inspection. They are not necessarily errors, but may affect model performance.</li>
</ul>
""")


# %% [markdown]
# #### 🟢 Image Dimensions: What Do Their Distributions Reveal?
#
# We've examined individual images and summary statistics. Now let's visualize how image dimensions and aspect ratios vary across the dataset.
#
# The COCO images come from many different photographers, while the park will use a few fixed cameras.
#
# **Observe:**
# - What patterns do you see in the width-versus-height scatter plot?
# - How do aspect ratios differ between animal classes?
# - What might these differences mean when preparing images for a computer vision model?

# %%
# @title 🔎 Understanding Width–Height Scatter Plot Patterns { display-mode: "form" }
# Three tiny example scatters of width (across) against height (up), to pick from.
_rng = np.random.default_rng(SEED)
_line_w = _rng.uniform(300, 640, 40)
_examples = {
    "one point\nall images the same size": (np.full(40, 640.0), np.full(40, 480.0)),
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
# @title 🔎 Explore Image Dimensions and Aspect Ratios by Class { display-mode: "form" }
# Left: every photo as a dot (width across, height up) in its class color; the dashed line is a square
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
ax_hist.set_ylabel("images")
ax_hist.set_title("Aspect ratios")
fig.tight_layout()
display(fig)
plt.close(fig)

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>Image dimensions:</strong> The scatter plot shows horizontal and vertical bands around 500 and 640 pixels. Many images share the same width or height, suggesting standardized image sizes rather than completely random dimensions.</li>
<li><strong>Aspect ratios:</strong> Images include both landscape and portrait orientations, with common ratios around 4:3 and 3:4. The animal classes overlap considerably, although their distributions may differ.</li>
<li><strong>Implications:</strong> Images may require resizing or padding before training. At the park, fixed cameras could produce characteristic image sizes. If these correlate with animal classes, a model might learn camera-specific shortcuts instead of recognizing the animals.</li>
</ul>
""")


# %% [markdown]
# #### 🟢 Color Modes and Colors: What Do They Reveal?
#
# Images can be stored in different color modes, such as RGB (three channels) or grayscale (one channel). However, an RGB image can also appear grayscale if its color channels contain similar values.
#
# The next visualization shows the stored color modes, examples of images that appear grayscale, and RGB pixel-intensity distributions for each animal class.
#
# **Observe:**
# - Which color modes occur in the dataset? Do all images have the same number of channels?
# - How many images appear grayscale, and are they necessarily stored in grayscale mode?
# - What differences do you notice between the RGB distributions of animal classes?
# - Why might these differences matter when preparing images or training a model?
#
# **Go Deeper (optional):**
# - What do the shaded bands (25th–75th percentile) reveal about variation within each class?
# - Could class-specific color patterns become misleading shortcuts for a model?

# %%
# @title 🔎 Explore Color Modes, Grayscale Images, and RGB Distributions { display-mode: "form" }
# Count color modes, show grayscale-looking images, and compare RGB distributions across classes.
_modes = COLOR_MODES + sorted(set(image_stats["mode"]) - set(COLOR_MODES))
mode_by_class = pd.crosstab(image_stats["mode"], image_stats["class"])
mode_by_class = mode_by_class.reindex(index=_modes, columns=CLASSES, fill_value=0)

image_stats["looks_gray"] = (image_stats["mode"] == "L") | (image_stats["colorfulness"] < 2.0)
print("color modes: " + " · ".join(f"{m} {int(mode_by_class.loc[m].sum())}" for m in _modes))
_n_gray = int(image_stats["looks_gray"].sum())
print(f"images that look grayscale (mode L, or RGB with colorfulness below 2): {_n_gray}")

N_GRAY = 6 if SMOKE else 12
show_rows(image_stats[image_stats["looks_gray"]].head(N_GRAY), "images that look grayscale")

_n_rows = math.ceil(len(CLASSES) / 3)
fig = plt.figure(figsize=(13, 2.3 * _n_rows))
_gs = fig.add_gridspec(
    _n_rows, 4, width_ratios=[1.4, 1, 1, 1],
    hspace=0.65, wspace=0.3,
    left=0.05, right=0.99, top=0.86, bottom=0.12
)

ax_modes = fig.add_subplot(_gs[:, 0])
_bottom = np.zeros(len(_modes))

for c in CLASSES:
    ax_modes.bar(
        _modes, mode_by_class[c].to_numpy(),
        bottom=_bottom, color=CLASS_COLORS[c], label=c
    )
    _bottom += mode_by_class[c].to_numpy()

ax_modes.set_ylabel("images")
ax_modes.set_title("Color mode as stored in the file", fontsize=10)
ax_modes.legend(frameon=False, fontsize=8)

_edges = np.linspace(0, 256, HIST_BINS + 1)
_centers = (_edges[:-1] + _edges[1:]) / 2

for k, c in enumerate(CLASSES):
    ax = fig.add_subplot(_gs[k // 3, 1 + k % 3])

    # Per-image RGB histograms for this class
    _hists = channel_hists[(image_stats["class"] == c).to_numpy()]

    _mean_hist = _hists.mean(axis=0)
    _q25 = np.percentile(_hists, 25, axis=0)
    _q75 = np.percentile(_hists, 75, axis=0)

    for ch_idx, ch in enumerate("RGB"):
        ax.plot(
            _centers, _mean_hist[ch_idx],
            color=CHANNEL_COLORS[ch], lw=1.4, label=ch
        )
        ax.fill_between(
            _centers, _q25[ch_idx], _q75[ch_idx],
            color=CHANNEL_COLORS[ch], alpha=0.15,
            linewidth=0
        )

    ax.set_title(
        f"{c} ({len(_hists)} images)",
        color=CLASS_COLORS[c], fontsize=10
    )
    ax.set_yticks([])
    ax.set_xlim(0, 255)
    ax.tick_params(labelsize=7)

    for spine in ax.spines.values():
        spine.set_edgecolor(CLASS_COLORS[c])
        spine.set_linewidth(1.6)

    if k == 0:
        ax.legend(frameon=False, fontsize=7, loc="upper left")

fig.text(
    0.67, 0.95,
    "Mean RGB histograms by class (shading: 25th–75th percentile)",
    ha="center", color=INK
)

display(fig)
plt.close(fig)

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>Color modes:</strong> 632 images are RGB (three channels) and one is grayscale (L, one channel of brightness values from 0 to 255). Converting an L image to RGB duplicates its brightness values across three channels, ensuring consistent model inputs.</li>
<li><strong>Grayscale appearance:</strong> 17 images appear grayscale, but most are stored as RGB. Appearance and storage mode are not necessarily the same.</li>
<li><strong>RGB distributions:</strong> Histograms show how frequently different pixel intensities occur, but not where colors appear. The distributions differ between animal classes.</li>
<li><strong>Implications:</strong> Color differences may reflect backgrounds, lighting, or camera conditions rather than animal features. Consistent preprocessing and evaluation on new camera images are important.</li>
</ul>

<p style="margin:12px 0 4px;"><strong>🔎 Go Deeper (optional)</strong></p>
<ul style="margin:4px 0 0 0;padding-left:22px;">
<li><strong>Variability bands:</strong> The shaded areas show the 25th–75th percentile of pixel-intensity frequencies across images within each class. Wider bands indicate greater variation between images, while narrow bands indicate more consistent distributions.</li>
<li><strong>Potential shortcuts:</strong> A model might associate certain colors or backgrounds with particular animals instead of learning their visual features. Such shortcuts could fail when the animals are photographed in different environments, such as the park.</li>
</ul>
""")


# %% [markdown]
# #### 🟢 Image Properties: Are Extreme Values Problematic?
#
# Extreme values can reveal unusual images or potential data quality issues.
#
# **Explore:**
# 1. Select an image property from the dropdown.
# 2. Choose how many images to inspect and run the cell.
# 3. Try different properties and compare the results.
#
# **Observe:**
# - What makes these images unusual?
# - Are they problematic or representative of realistic conditions?
# - Which images deserve further investigation before training?
# - Which images would you flag for further inspection or cleaning, and why?

# %%
# @title 🔎 Explore Extreme Image Properties { display-mode: "form" }

STAT_TO_SHOW = "brightness" # @param ["brightness", "contrast", "colorfulness", "aspect", "megapixels", "long_side"]
N_IMAGES = 5 # @param {type:"slider", min:1, max:5, step:1}

show_extremes(image_stats, STAT_TO_SHOW, N_IMAGES)

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>Extreme values:</strong> Very dark images may show animals in shadows or at dusk, while bright images may contain snow, sand, or overexposed regions. Unusual aspect ratios can indicate very wide or tall images.</li>
<li><strong>Data quality:</strong> Extreme values are not necessarily errors. Images showing realistic lighting or unusual viewpoints may be valuable for training.</li>
<li><strong>Further inspection:</strong> Uniform gray images, photos of screens, or images where the animal is barely recognizable deserve closer inspection. Depending on the use case, they may need cleaning, exclusion, or special handling.</li>
</ul>
""")


# %% [markdown]
# #### 🔴 Brightness: How Do Distributions Differ Between Classes? (Optional)
#
# A violin plot shows the distribution of values within each class. Wider sections indicate where more images are concentrated. The horizontal dark line marks the median, while the vertical line and its endpoints show the minimum and maximum values.
#
# **Observe:**
# - Which animal classes have the highest and lowest median brightness?
# - Which classes show the greatest variation in brightness?
# - Do the distributions overlap, or are there clear differences between classes?
#
# **Go Deeper (optional):**
# - Could differences in brightness become misleading shortcuts for a classification model?
# - What can the violin plots reveal that the summary statistics alone cannot?

# %%
# @title 🔎 Explore Brightness Distributions by Class { display-mode: "form" }
# Brightness per class as violin plots, in class colors. The dotted lines mark "very dark" and "very bright".
fig, ax = plt.subplots(figsize=(9, 3.8))
_data = [image_stats.loc[image_stats["class"] == c, "brightness"].to_numpy() for c in CLASSES]
_violins = ax.violinplot(_data, widths=0.8, showmedians=True)
for body, c in zip(_violins["bodies"], CLASSES, strict=True):
    body.set_facecolor(CLASS_COLORS[c])
    body.set_alpha(0.85)
_violins["cmedians"].set_color(INK)
for part in ("cmins", "cmaxes", "cbars"):
    _violins[part].set_color(INK)
ax.set_xticks(range(1, len(CLASSES) + 1), CLASSES)
for level in (40, 215):
    ax.axhline(level, color=INK, lw=0.8, ls=":")
ax.set_ylim(0, 255)
ax.set_ylabel("brightness (mean gray level)")
ax.set_title("Brightness per class")
fig.tight_layout()
display(fig)
plt.close(fig)

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>Median brightness:</strong> Some animal classes have higher median brightness than others, suggesting differences in typical lighting or backgrounds.</li>
<li><strong>Variation:</strong> Wider and taller violins indicate greater variation in brightness. Some classes contain both very dark and very bright images.</li>
<li><strong>Overlap:</strong> The distributions overlap substantially, so brightness alone cannot reliably distinguish animal classes.</li>
</ul>
<p style="margin:12px 0 4px;"><strong>🔎 Go Deeper (optional)</strong></p>
<ul style="margin:4px 0 0 0;padding-left:22px;">
<li><strong>Potential shortcuts:</strong> A model might associate certain brightness levels with particular animals instead of recognizing their visual features. This could reduce performance under different lighting conditions.</li>
<li><strong>Beyond summary statistics:</strong> Violin plots reveal where brightness values are concentrated, how widely they vary, and how much the classes overlap. Medians and ranges alone cannot show these patterns.</li>
</ul>
""")


# %% [markdown]
# #### 🟢 Class Distribution: Are All Animal Classes Equally Represented?
#
# The number of images per class can influence how a classification model learns. Classes with fewer examples may be harder to recognize reliably.
#
# The next visualization shows how many images belong to each animal class.
#
# **Observe:**
# - Which classes have the most and fewest images?
# - How large are the differences between classes?
# - Why might class imbalance matter when training and evaluating a classification model?

# %%
# @title 🧾 Recipe · Step 1, continued: Imbalances and Biases { display-mode: "form" }
callout("recipe", "🧾 Recipe · Step 1, continued: Imbalances and Biases", r"""
  <div style="font-style:italic;margin-bottom:6px;">
    “I look for data imbalances and biases.”
  </div>
  <div>
    Andrej Karpathy,
    <a href="https://karpathy.github.io/2019/04/25/recipe/"
       target="_blank" style="color:#333333;">
      A Recipe for Training Neural Networks
    </a> (2019).
  </div>
""")


# %%
# @title 🔎 Explore Class Distribution { display-mode: "form" }
images_per_class = {c: int((images["class"] == c).sum()) for c in CLASSES}
RESULTS["images_per_class"] = images_per_class

_total = sum(images_per_class.values())
_largest = max(images_per_class, key=images_per_class.get)
_smallest = min(images_per_class, key=images_per_class.get)
_ratio = images_per_class[_largest] / max(1, images_per_class[_smallest])
print(f"{_total} images · largest class {_largest} ({images_per_class[_largest]}) · smallest {_smallest} "
      f"({images_per_class[_smallest]}) · ratio {_ratio:.1f}×")

fig, ax = plt.subplots(figsize=(9, 3.8))
_order = sorted(CLASSES, key=images_per_class.get, reverse=True)
ax.bar(_order, [images_per_class[c] for c in _order], color=[CLASS_COLORS[c] for c in _order])
for i, c in enumerate(_order):
    ax.text(i, images_per_class[c], f"{100 * images_per_class[c] / _total:.0f} %", ha="center", va="bottom",
            fontsize=9)
ax.set_ylabel("images")
ax.set_title("Images per class (one label per image)")
ax.set_ylim(0, max(images_per_class.values()) * 1.15)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>Class distribution:</strong> Cats and dogs have the most images (130 and 129), while sheep have the fewest (59). The largest classes contain about 2.2 times as many images as the smallest.</li>
<li><strong>Dataset bias:</strong> People may photograph pets more frequently than sheep. The distribution reflects the Flickr photo collection, not necessarily what the park's cameras will capture.</li>
<li><strong>Implications:</strong> A model may learn common classes better than underrepresented ones. Overall accuracy could hide poor performance on sheep, so performance should also be evaluated per class.</li>
<li><strong>Park use case:</strong> Dogs are common in the Play Dataset but may rarely enter the sheep paddock. Detecting these rare events reliably is important, even if dogs are well represented in the training data. Overall accuracy alone may not reflect this performance.</li>
</ul>
<p style="margin:8px 0 0;"><em>Note: The smaller smoke-test dataset may have a different, more balanced class distribution.</em></p>
""")


# %% [markdown]
# #### 🟢 Animals per Image: How Common Are Multiple Animals?
#
# Each image in the Play Dataset has one class label, but it may contain several animals of that class. The dataset was selected from COCO to include only one of our six animal classes per image.
#
# The next visualization shows the number of annotated animals per image and examples with particularly many animals.
#
# **Observe:**
# - What proportion of images contain more than one annotated animal?
# - Which classes tend to have the most animals per image?
# - What do you notice when inspecting images with many animals?
#
# **Go Deeper (optional):**
# - Why might assigning one label to an entire image be problematic, even when all annotated animals belong to the same class?

# %%
# @title 🔎 Explore the Number of Animals per Image { display-mode: "form" }
_max_n = int(images["n_objects"].max())
_several = images[images["n_objects"] > 1]
print(f"{len(_several)} of {len(images)} images ({100 * len(_several) / len(images):.0f} %) hold more than "
      f"one animal; the most crowded image has {_max_n}.")
print("mean animals per image: " + " · ".join(
    f"{c} {images.loc[images['class'] == c, 'n_objects'].mean():.1f}" for c in CLASSES))

fig, ax = plt.subplots(figsize=(9, 3.8))
_bins = np.arange(0.5, min(_max_n, 10) + 1.5, 1)
_per_class = [images.loc[images["class"] == c, "n_objects"].clip(upper=10) for c in CLASSES]
ax.hist(_per_class, bins=_bins, stacked=True, color=[CLASS_COLORS[c] for c in CLASSES], label=CLASSES)
ax.set_xticks(range(1, min(_max_n, 10) + 1))
ax.set_xticklabels([str(i) if i < 10 else "10+" for i in range(1, min(_max_n, 10) + 1)])
ax.set_xlabel("animals in the image")
ax.set_ylabel("images")
ax.set_title("Animals per image")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
display(fig)
plt.close(fig)

N_CROWDED = 3 if SMOKE else 6  # how many crowded photos to show (N-014)
_crowded = images.sort_values(["n_objects", "id"], ascending=[False, True]).head(N_CROWDED)
show_rows(_crowded, "the most crowded images", lambda r: f"{r['n_objects']} × {r['class']}", cols=N_CROWDED)

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>Multiple animals:</strong> On the full Play Dataset, 241 of 633 images (38%) contain more than one annotated animal. The most crowded image contains 26.</li>
<li><strong>Class differences:</strong> Sheep have the highest average (5.7 animals per image), followed by birds (3.4) and zebras (3.2). Cats and dogs usually appear individually.</li>
<li><strong>Crowded images:</strong> Images with many animals may contain overlapping or very small animals, making individual animals harder to distinguish.</li>
</ul>
<p style="margin:12px 0 4px;"><strong>🔎 Go Deeper (optional)</strong></p>
<ul style="margin:4px 0 0 0;padding-left:22px;">
<li><strong>Classification limitation:</strong> One image label does not describe how many animals are present or where they are located. This matters if the park needs to count animals or monitor their movements; object detection or segmentation would be more suitable.</li>
</ul>
""")


# %% edition="teaching"
# @title 📘 Labeling Strategies: One Label or Multiple Animals? { display-mode: "form" }
callout("theory", "📘 Labeling Strategies: One Label or Multiple Animals?", r"""
  <ul style="margin:6px 0;padding-left:22px;">
    <li><strong>Single-label classification:</strong> Assigns one class to an entire image, even if several animals are present.</li>
    <li><strong>Multi-label classification:</strong> Identifies all animal classes present, such as both sheep and dog.</li>
    <li><strong>Object detection:</strong> Identifies and locates individual animals using bounding boxes, allowing them to be counted.</li>
  </ul>
  <div>
    Our COCO subset excludes images containing multiple selected animal classes.
    This removes examples such as dogs among sheep—exactly the situations the park's dog alert needs to recognize.
  </div>
""")


# %% [markdown] edition="teaching"
# #### 🔴 Class Imbalance: Can High Accuracy Be Misleading? (Optional)
#
# Imagine a camera monitoring a sheep paddock. Most images contain only sheep, while dogs appear rarely. A rule that always predicts the most common class might achieve high accuracy without detecting any dogs.
#
# **Accuracy** is the proportion of correct predictions. **Dog recall** is the proportion of actual dog images correctly identified.
#
# **Explore:**
# 1. Estimate accuracy and dog recall for two rules that ignore the image:
#    - **Rule A — COCO subset:** Always predict the most common animal class.
#    - **Rule B — Paddock camera:** Of 1,000 images, 20 contain a dog. Always predict "no dog".
# 2. Run the code and compare your estimates with the results.
# 3. Change `DOG_SHARE` using the slider and observe how the results change.
#
# **Observe:**
# - Which rule achieves higher accuracy?
# - What is the dog recall for each rule?
# - How does the proportion of dog images affect accuracy and dog recall?
# - Why is accuracy alone insufficient for evaluating the park's dog alert?

# %% edition="teaching"
# @title 🔎 Explore Accuracy and Dog Recall Under Class Imbalance { display-mode: "form" }

DOG_SHARE = 0.02 # @param {type:"slider", min:0.01, max:0.2, step:0.01}

# Rule A: Always predict the most common class in the COCO subset
_counts = images["class"].value_counts()
_majority_class = _counts.idxmax()
_accuracy_a = _counts.max() / len(images)
_dog_recall_a = float(_majority_class == "dog")

# Rule B: Always predict "no dog" in the paddock
_accuracy_b = 1 - DOG_SHARE
_dog_recall_b = 0.0

print(f"Rule A — COCO subset (always '{_majority_class}')")
print(f"  Accuracy:   {_accuracy_a:.1%}")
print(f"  Dog recall: {_dog_recall_a:.1%}")

print(f"\nRule B — Paddock camera ({DOG_SHARE:.0%} dog images, always 'no dog')")
print(f"  Accuracy:   {_accuracy_b:.1%}")
print(f"  Dog recall: {_dog_recall_b:.1%}")

# %% edition="teaching"
# @title 🔎 See How the Rule Classifies Real Images { display-mode: "form" }

# The same day in miniature: 12 real photos, a share DOG_SHARE of them dogs, the rest sheep.
_N_STRIP = 12
_strip_dogs = min(int(round(DOG_SHARE * _N_STRIP)) or int(DOG_SHARE > 0), images_per_class["dog"])
_rng = random.Random(SEED)
_dog_idx = _rng.sample(sorted(images.index[images["class"] == "dog"]), _strip_dogs)
_sheep_idx = _rng.sample(sorted(images.index[images["class"] == "sheep"]),
                         min(_N_STRIP - _strip_dogs, images_per_class["sheep"]))
_strip_idx = _dog_idx + _sheep_idx
_rng.shuffle(_strip_idx)
_strip = images.loc[_strip_idx]

_cols = 6
_rows = -(-len(_strip) // _cols)
fig, axes = plt.subplots(_rows, _cols, figsize=(2.3 * _cols, 2.5 * _rows), squeeze=False)
for ax in axes.ravel():
    ax.axis("off")
for ax, (_, row) in zip(axes.ravel(), _strip.iterrows(), strict=False):
    ax.axis("on")
    ax.imshow(load_thumb(row["path"]))
    frame(ax, CLASS_COLORS[row["class"]])
    _right = row["class"] == "sheep"
    ax.text(0.04, 0.96, "sheep " + ("✔" if _right else "✘"), transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold", color="white",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": CLASS_COLORS["sheep"], "edgecolor": "none"})
    ax.set_title(f"true: {row['class']}", fontsize=8, color="#2E7D32" if _right else "#C62828")
_strip_right = int((_strip["class"] == "sheep").sum())
fig.suptitle(f"A lazy model stamps 'sheep' on {len(_strip)} paddock frames: {_strip_right} right, "
             f"{len(_strip) - _strip_right} wrong", color=INK)
fig.tight_layout()


plt.show()

print(
    f"This strip: accuracy {100 * _strip_right / len(_strip):.0f}%, "
    "recall for dog: 0%. "
    "The dog-in-the-paddock alert never fires."
)

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>Rule A:</strong> In the full COCO subset, the largest class contains 130 of 633 images, giving approximately 20% accuracy. Dog recall is 0% if the rule always predicts cat, or 100% if it always predicts dog.</li>
<li><strong>Rule B:</strong> Always predicting "no dog" gives 98% accuracy but 0% dog recall when 2% of images contain dogs. None of these dogs triggers an alert.</li>
<li><strong>Changing DOG_SHARE:</strong> As the proportion of dog images increases, accuracy decreases, while dog recall remains 0%.</li>
<li><strong>Implications:</strong> High accuracy can be misleading for imbalanced datasets. For the park's dog alert, detecting dogs is essential, so dog recall matters; false alarms must also be considered.</li>
</ul>
<p style="margin:8px 0 0;"><em>Results for Rule A may differ when using the smaller smoke-test subset.</em></p>
""")


