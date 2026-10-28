# %% [markdown]
# ## Common Dataset Errors & Cleaning
#
# Exploring a dataset helps us understand whether it is suitable for a task. But even a suitable dataset may contain errors that affect model training and evaluation.
#
# In this section, we investigate duplicate images, including flipped or rotated copies, and suspicious class labels. These problems can cause data leakage, misleading evaluation results, or incorrect learning.
#
# Our goal is to identify potential issues and prepare a cleaner dataset for splitting into training and test sets.

# %%
# @title 🧾 Recipe · Step 1, continued: Dataset Errors { display-mode: "form" }
callout("recipe", "🧾 Recipe · Step 1, continued: Dataset Errors", r"""
  <div style="font-style:italic;margin-bottom:6px;">
    “One time I discovered that the data contained duplicate examples.
    Another time I found corrupted images / labels.”
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
# @title 🔍 Go Deeper: Two Kaggle Stories { display-mode: "form" }
callout("deeper", "🔍 Go Deeper · This is ordinary: two Kaggle stories", r"""
<details style="margin:6px 0 0 0;">
<summary style="cursor:pointer;font-weight:bold;">Datasets built with care that still shipped duplicates</summary>
<p style="margin:6px 0 0 0;"><strong>PetFinder.my Pawpularity</strong> (2021) shipped near-identical photos with <em>different</em> scores, and <strong>Cassava
Leaf Disease</strong> (2020) identical leaf photos with two different labels. Both were built with care. Yours
will have duplicates too, unless you look.</p>
</details>
""")


# %% [markdown]
# ### 🟢 Duplicate Images: What Can We Detect?
#
# Duplicate images can lead to misleading evaluation results, especially when copies appear in both training and test sets.
#
# - **Exact duplicates:** Identical image files with the same content.
# - **Near-duplicates:** Images showing essentially the same content but differing in compression, resolution, cropping, or orientation.
#
# We use MD5 to detect exact duplicates and perceptual hashing (pHash) to identify potential near-duplicates.
#
# **Before running the code:**
# - Do you expect a curated dataset such as COCO to contain exact duplicates?
# - Can pHash also flag images that are visually similar but not duplicates?
#
# **Explore:**
# 1. Run the duplicate detection and inspect the results.
# 2. Adjust the bit distance threshold to change which image pairs are flagged.
# 3. Browse the flagged pairs, ordered from most to least similar. Increase the threshold if no pairs are found.
#
# **Observe:**
# - How many exact duplicates are found?
# - How does increasing the threshold affect the number of flagged pairs?
# - Are all flagged pairs genuine duplicates?
# - Why could duplicates across training and test sets lead to misleading evaluation results?
#
# **Go Deeper (optional):**
# - Why should flagged pairs be visually inspected before removing images?

# %%
# @title 📘 Detecting Duplicates: MD5 vs. Perceptual Hashing { display-mode: "form" }
callout("theory", "📘 Detecting Duplicates: MD5 vs. Perceptual Hashing", r"""
  <ul style="margin:6px 0;padding-left:22px;">
    <li><strong>MD5:</strong> Computes a hash from file bytes. Identical files have identical hashes, but resizing or recompressing changes the hash.</li>
    <li><strong>Perceptual hashing (pHash):</strong> Represents coarse visual structure using a compact 64-bit hash. Visually similar images often produce similar hashes, even after resizing or recompression.</li>
    <li><strong>Hamming distance (bit distance):</strong> Counts how many bits differ between two hashes (0–64). Smaller distances indicate more similar hashes, but not necessarily more similar image content.</li>
    <li><strong>Bit distance threshold:</strong> Defines the maximum number of differing bits for flagging a pair as a potential near-duplicate.</li>
  </ul>
  <div style="margin-top:8px;">
    <strong>Important limitation:</strong> Different images may have similar hashes,
    while genuine near-duplicates may have larger bit distances.
    Flipped or rotated images may also require additional checks.
    The threshold identifies <strong>candidates for visual inspection</strong>,
    not confirmed duplicates.
  </div>
""")


# %%
# @title Helpers: Duplicate Detection and Image Quality { display-mode: "form" }
MAX_PAIR_BITS = 20  # pairs up to this many differing bits are kept as candidates for the pair browser
PAIR_THRESHOLD = 8  # the bit distance threshold a pair browser and the copy comparison start with


def file_md5(path) -> str:
    """Compute an MD5 hash from file bytes."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def exact_duplicate_groups(paths: list) -> list[list]:
    """Return groups of byte-identical files."""
    groups = {}
    for path in paths:
        groups.setdefault(file_md5(path), []).append(path)
    return [group for group in groups.values() if len(group) > 1]


def phash(image: Image.Image) -> imagehash.ImageHash:
    """Compute a 64-bit perceptual hash."""
    return imagehash.phash(image)


def hash_bits(hashes: list[imagehash.ImageHash]) -> np.ndarray:
    """Convert perceptual hashes to an (n, 64) bit array."""
    return np.array([h.hash.flatten() for h in hashes], dtype=bool)


def hash_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise bit distances between hash arrays."""
    return (a[:, None, :] != b[None, :, :]).sum(axis=-1)


def is_readable(path) -> bool:
    """Check whether an image can be fully decoded."""
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except (OSError, ValueError):
        return False


def jpeg_bytes(im: Image.Image, scale: float = 1.0, quality: int = 55) -> bytes:
    """The JPEG file of `im` resized by `scale` (each side at least 8 px) and saved at `quality`."""
    if scale != 1.0:
        im = im.resize((max(8, int(im.width * scale)), max(8, int(im.height * scale))))
    buf = BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def jpeg_copy(im: Image.Image, scale: float = 1.0, quality: int = 55) -> Image.Image:
    """`im` after a trip through a JPEG file: resized by `scale`, saved at `quality`, opened again as RGB."""
    return Image.open(BytesIO(jpeg_bytes(im, scale, quality))).convert("RGB")


# %%
# @title 🔎 Inspect Near-Duplicates Using pHash { display-mode: "form" }
exact_groups = exact_duplicate_groups(images["path"].tolist())
print(f"{len(images)} files · {len(exact_groups)} groups of byte-identical files")
for _group in exact_groups[:5]:
    print("  ", " = ".join(Path(p).name for p in _group))

# One pHash per image (on its 256 px thumbnail), reused by the bit-distance histogram below
phash_bits = hash_bits([phash(load_thumb(path)) for path in images["path"]])
pair_dist = hash_distances(phash_bits, phash_bits)
_i, _j = np.where(np.triu(pair_dist <= MAX_PAIR_BITS, k=1))
candidate_pairs = sorted(zip(_i.tolist(), _j.tolist(), pair_dist[_i, _j].tolist(), strict=True),
                         key=lambda pair: pair[2])

print(f"Found {len(candidate_pairs)} candidate pairs within {MAX_PAIR_BITS} bits.")

dup_threshold = widgets.IntSlider(
    value=PAIR_THRESHOLD, min=0, max=MAX_PAIR_BITS, step=1,
    description="Bit distance threshold:",
    continuous_update=False,
    style={"description_width": "initial"},
    layout=widgets.Layout(width="450px")
)

dup_previous = widgets.Button(description="← Previous")
dup_next_button = widgets.Button(description="Next →")
dup_summary = widgets.HTML()
dup_counter = widgets.HTML()
dup_output = widgets.Output()

dup_state = {"index": 0}

def dup_update_browser(*_):
    # Only show pairs within the selected threshold
    flagged_pairs = [pair for pair in candidate_pairs if pair[2] <= dup_threshold.value]

    dup_state["index"] = min(dup_state["index"], max(0, len(flagged_pairs) - 1))

    dup_summary.value = (
        f"<b>{len(images)}</b> images checked &nbsp; | &nbsp; "
        f"<b>{len(flagged_pairs)}</b> potential near-duplicate pairs"
    )


    dup_previous.disabled = dup_state["index"] == 0
    dup_next_button.disabled = dup_state["index"] >= len(flagged_pairs) - 1

    with dup_output:
        clear_output(wait=True)

        if not flagged_pairs:
            dup_counter.value = ""
            print("No image pairs found within this threshold. Try increasing it.")
            return

        i, j, distance = flagged_pairs[dup_state["index"]]

        dup_counter.value = (
            f"<b>Pair {dup_state['index'] + 1} of {len(flagged_pairs)}</b>"
        )

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for ax, idx in zip(axes, (i, j)):
            row = images.iloc[idx]
            ax.imshow(load_thumb(row["path"]))
            ax.set_title(f"Class: {row['class']}", fontsize=11)
            ax.axis("off")

        fig.suptitle(f"Bit distance: {distance} (out of 64 bits)", fontsize=12)

        fig.tight_layout()
        plt.show()
        plt.close(fig)

def dup_navigate(step):
    dup_state["index"] += step
    dup_update_browser()

def dup_change_threshold(change):
    dup_state["index"] = 0
    dup_update_browser()

dup_previous.on_click(lambda _: dup_navigate(-1))
dup_next_button.on_click(lambda _: dup_navigate(1))
dup_threshold.observe(dup_change_threshold, names="value")

display(dup_summary)
display(dup_threshold)
display(widgets.HBox([dup_previous, dup_counter, dup_next_button]))
display(dup_output)

dup_update_browser()

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>Exact duplicates:</strong> No byte-identical files were found among the 633 images. MD5 detects identical files, but recompressing an image changes its hash. Even curated datasets should be checked, especially when new images are added or exported.</li>

<li><strong>Near-duplicates:</strong> At a threshold of 8 bits, one pair is flagged: two images of the same cat in front of a brick wall, with slightly different head positions. Most other images have larger bit distances, but visually similar scenes can still produce false positives.</li>
<li><strong>Threshold:</strong> Increasing the threshold flags more pairs, including unrelated images. A lower threshold may miss genuine near-duplicates.</li>
<li><strong>Data leakage:</strong> Similar images across training and test sets may lead to overly optimistic performance estimates because the model has already seen much of the visual content.</li>
<li><strong>Go Deeper:</strong> A small bit distance does not prove duplication. Flagged pairs should be inspected before deciding whether to remove or group them during dataset splitting.</li>
</ul>
<p style="margin:8px 0 0;"><em>Counts may differ for the smaller smoke-test subset or other thresholds.</em></p>
""")


# %% [markdown]
# ### 🔴 Bit Distance Distribution (Optional)
#
# The histogram shows the bit distance between each image and its closest match in the dataset. Smaller distances indicate greater visual similarity.
#
# **Observe:**
# - Where are most bit distances concentrated?
# - How does changing the threshold affect the number of images considered potential duplicates?
# - Why might choosing a suitable threshold be difficult?

# %%
# @title 🔎 Explore Bit Distance Distribution { display-mode: "form" }

HASH_THRESHOLD = 5 # @param {type:"slider", min:0, max:20, step:1}

_dist = pair_dist.copy()  # the bit distances of every pair, from the pHash cell above
np.fill_diagonal(_dist, 99)
nearest_distance = _dist.min(axis=1)  # each photo's distance to its closest other photo
_i, _j = np.where(np.triu(_dist <= HASH_THRESHOLD, k=1))
near_pairs = pd.DataFrame({"a": images["id"].to_numpy()[_i], "b": images["id"].to_numpy()[_j],
                           "distance": _dist[_i, _j]}).sort_values("distance").reset_index(drop=True)
print(f"pairs within {HASH_THRESHOLD} bits: {len(near_pairs)} · median nearest-neighbor distance: "
      f"{np.median(nearest_distance):.0f} bits")

fig, ax = plt.subplots(figsize=(9, 3.4))
_bins = np.arange(-0.5, 41, 2)
ax.hist([nearest_distance[(images["class"] == c).to_numpy()] for c in CLASSES], bins=_bins, stacked=True,
        color=[CLASS_COLORS[c] for c in CLASSES], label=CLASSES)
ax.axvline(HASH_THRESHOLD + 0.5, color=INK, ls=":", lw=1)
ax.text(HASH_THRESHOLD + 1, ax.get_ylim()[1] * 0.9, f"HASH_THRESHOLD = {HASH_THRESHOLD}", fontsize=9)
ax.set_xlabel("Hamming distance to the closest other image (bits)")
ax.set_ylabel("images")
ax.set_title("How close is each image's nearest neighbor?")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
display(fig)
plt.close(fig)


# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>Distribution:</strong> Most images have a nearest-neighbor bit distance well above a strict duplicate threshold. Only a few pairs have very small distances.</li>
<li><strong>Threshold:</strong> Increasing the threshold flags more potential duplicates, while decreasing it flags fewer. The underlying distance distribution remains unchanged.</li>
<li><strong>Trade-off:</strong> A low threshold may miss genuine near-duplicates; a high threshold may flag unrelated images. There is no universally correct threshold, so visual inspection is important.</li>
</ul>
""")


# %% [markdown]
# ### 🔴 How Does Perceptual Hashing Work? (Optional)
#
# The following visualization shows how pHash generates a 64-bit representation of an image and why visually similar images often produce similar hashes.

# %%
# @title 🔎 Explore Perceptual Hashing (pHash) { display-mode: "form" }

from matplotlib.colors import ListedColormap
from scipy.fft import dctn

bit_colors = ListedColormap(["#F0F0F0", "#888888"])

c0 = images["class"].iloc[0]
pool = images[images["class"] != c0]

orig = load_thumb(images["path"].iloc[0])
unrelated = load_thumb((pool if len(pool) else images)["path"].iloc[-1])

near = jpeg_copy(orig, scale=0.7, quality=55)

gray = orig.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
dct = dctn(np.asarray(gray, dtype=float), type=2)

bits = {
    k: phash(im).hash
    for k, im in [("orig", orig), ("near", near), ("other", unrelated)]
}


def draw_bits(ax, values, other=None, title=""):
    """Display an 8×8 binary hash and highlight differing bits."""
    ax.imshow(values, cmap=bit_colors, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    for (r, c), bit in np.ndenumerate(values):
        ax.text(
            c, r, str(int(bit)),
            ha="center", va="center", fontsize=7,
            color="white" if bit else INK
        )

        if other is not None and bit != other[r, c]:
            ax.add_patch(mpl.patches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                fill=False, edgecolor=HIGHLIGHT, linewidth=2.5
            ))

    ax.set_title(title, fontsize=9)


fig, axes = plt.subplots(2, 4, figsize=(10, 5.2))

# How the hash is constructed
axes[0, 0].imshow(orig)
axes[0, 0].set_title("1. Original image", fontsize=9)

axes[0, 1].imshow(gray, cmap="gray")
axes[0, 1].set_title("2. Resize to 32×32, grayscale", fontsize=9)

axes[0, 2].imshow(np.log1p(np.abs(dct)), cmap="Greys")
axes[0, 2].add_patch(mpl.patches.Rectangle(
    (-0.5, -0.5), 8, 8,
    fill=False, edgecolor=HIGHLIGHT, linewidth=2.5
))
axes[0, 2].set_title("3. DCT: select top-left 8×8\n(coarse image structure)", fontsize=9)

draw_bits(
    axes[0, 3], bits["orig"],
    title="4. Above median? 1 / 0\n= 64 bits"
)

for ax in axes[0, :3]:
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# Compare perceptual hashes
d_near = int((bits["orig"] != bits["near"]).sum())
d_other = int((bits["orig"] != bits["other"]).sum())

draw_bits(axes[1, 0], bits["orig"], title="Original")
draw_bits(
    axes[1, 1], bits["near"], bits["orig"],
    f"Re-compressed copy:\n{d_near} bits differ"
)
draw_bits(
    axes[1, 2], bits["other"], bits["orig"],
    f"Unrelated image:\n{d_other} bits differ"
)

axes[1, 3].axis("off")
note = (
    "Yellow = differing bits\n\n"
    "Hamming distance =\n"
    "number of differing bits\n\n"
    f"Near copy: {d_near}\n"
    f"Unrelated: {d_other}"
)
axes[1, 3].text(0, 0.5, note, fontsize=9, va="center")

fig.tight_layout()
display(fig)
plt.close(fig)

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li><strong>Hash construction:</strong> The image is resized, converted to grayscale, and transformed using the DCT. Selected low-frequency coefficients are converted into a 64-bit hash.</li>
<li><strong>Near-duplicate:</strong> The recompressed image produces the same hash (0 differing bits), despite changes to its file representation.</li>
<li><strong>Unrelated image:</strong> The hashes differ in 38 of 64 bits, reflecting different visual structures.</li>
<li><strong>Interpretation:</strong> pHash captures coarse visual similarity rather than semantic content. Small bit distances suggest potential duplicates but do not guarantee them.</li>
</ul>
""")


# %% [markdown]
# ### 🔴 Detecting Flipped and Rotated Duplicates (Optional)
#
# Standard perceptual hashing can identify visually similar images, but may miss flipped or rotated copies.
#
# In this experiment, we create a small dataset containing exact copies, recompressed images, and flipped or rotated versions of existing photos.
#
# **Before running the code:**
# - Predict which duplicates each method will detect: exact copies, recompressed images, flipped images, and rotated images. Compare MD5, standard pHash, and transformation-aware pHash.
#
# **Explore:**
# 1. Generate the example dataset.
# 2. Compare the duplicates detected by the different methods.
# 3. Inspect which transformations each method detects or misses.
#
# **Observe:**
# - Which types of duplicates can MD5 detect?
# - Does standard pHash detect all transformed copies?
# - What additional duplicates are identified when flipped and rotated versions are compared?
#
# **Go Deeper (optional):**
# - What are the computational costs and potential drawbacks of checking multiple transformations?

# %%
# @title 🔎 Generate a Dataset with Known Duplicates { display-mode: "form" }
DUPLICATE_CARD_DIR = DATA_DIR / "cache" / "hunt5"
N_CARD_ORIGINALS = 12 if SMOKE else 40  # distinct photos on the card (N-014: fewer in smoke mode)
PLANT_KINDS = ["exact", "exact", "flipped", "rotated", "recompressed"]  # ≥ 4 planted copies, always present


def plant_duplicates(dest: Path, n_originals: int, seed: int) -> tuple[dict[str, Path], list[dict]]:
    """Write a 'memory card': n_originals distinct photos plus one planted copy per entry of PLANT_KINDS.

    Returns (files, key): files maps the card's neutral file names to paths; key lists the planted copies as
    {"copy": name, "source": name, "kind": kind}. Deterministic for a given seed.
    """
    rng = random.Random(seed)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    chosen = images.sample(n=min(n_originals, len(images)), random_state=seed).sort_values("id")
    sources = [Path(p) for p in chosen["path"]]
    # (bytes, kind or None for an original, index of the source photo)
    entries: list[tuple[bytes, str | None, int]] = [(s.read_bytes(), None, i) for i, s in enumerate(sources)]
    planted: list[tuple[str, int]] = []
    source_idx = rng.sample(range(len(sources)), len(PLANT_KINDS))
    for kind, i in zip(PLANT_KINDS, source_idx, strict=True):
        with Image.open(sources[i]) as im:
            rgb = im.convert("RGB")
        buf = BytesIO()
        if kind == "exact":
            data = sources[i].read_bytes()
        elif kind == "flipped":
            ImageOps.mirror(rgb).save(buf, "JPEG", quality=92)
            data = buf.getvalue()
        elif kind == "rotated":
            rgb.transpose(Image.Transpose.ROTATE_90).save(buf, "JPEG", quality=92)
            data = buf.getvalue()
        else:  # recompressed: smaller and at a lower JPEG quality
            data = jpeg_bytes(rgb, scale=0.7, quality=55)
        entries.append((data, kind, i))
        planted.append((kind, i))
    order = list(range(len(entries)))
    rng.shuffle(order)
    names = {idx: f"card_{k:03d}.jpg" for k, idx in enumerate(order)}
    files, key = {}, []
    for idx, (data, kind, i) in enumerate(entries):
        path = dest / names[idx]
        path.write_bytes(data)
        files[names[idx]] = path
        if kind is not None:
            key.append({"copy": names[idx], "source": names[i], "kind": kind})
    return dict(sorted(files.items())), key


hunt_files, duplicate_key = plant_duplicates(DUPLICATE_CARD_DIR, N_CARD_ORIGINALS, SEED)

print(f"Generated {len(hunt_files)} images, including {len(duplicate_key)} intentionally added duplicates.")

# %%
# @title Helpers: Compare Duplicate Detection Methods { display-mode: "form" }

def variants(img):
    """Generate image transformations for duplicate detection."""
    return {
        "original": img,
        "flip": ImageOps.mirror(img),
        "rotate90": img.rotate(90, expand=True),
        "rotate180": img.rotate(180, expand=True),
        "rotate270": img.rotate(270, expand=True),
    }

VARIANT_NAMES = ("original", "flip", "rotate90", "rotate180", "rotate270")

KIND_OF_VARIANT = {
    "original": "recompressed",
    "flip": "flipped",
    "rotate90": "rotated",
    "rotate180": "rotated",
    "rotate270": "rotated",
}


def find_copies(files: dict[str, Path], threshold: int, use_variants: bool) -> pd.DataFrame:
    """Pairs of files that are copies of each other: columns a, b, kind, distance.

    kind is "exact" (same bytes), "recompressed" (hashes within `threshold`), or, with use_variants=True,
    "flipped" / "rotated" (a flipped or rotated variant of a hashes within `threshold` of b).
    All pairs at once, in blocks of 256 rows so that a few thousand files fit in memory.
    """
    names = list(files)
    variant_names = VARIANT_NAMES if use_variants else ("original",)
    md5s, hashes = [], {v: [] for v in variant_names}
    for n in names:
        md5s.append(file_md5(files[n]))
        for v, im in variants(load_thumb(files[n])).items():
            if v in hashes:
                hashes[v].append(phash(im))
    bits = {v: hash_bits(h) for v, h in hashes.items()}
    plain = bits["original"]
    # best[i, j]: smallest distance between a variant of file i and the plain hash of file j; best_v: which variant
    best = np.full((len(names), len(names)), 65, dtype=np.int16)
    best_v = np.zeros((len(names), len(names)), dtype=np.int8)
    for k, v in enumerate(variant_names):
        for start in range(0, len(names), 256):
            d = hash_distances(bits[v][start:start + 256], plain)
            block, block_v = best[start:start + 256], best_v[start:start + 256]
            better = d < block
            block[better] = d[better]
            block_v[better] = k
    md5_code = pd.factorize(pd.Series(md5s, dtype=object))[0]
    same_bytes = md5_code[:, None] == md5_code[None, :]
    rows = []
    for i, j in zip(*np.nonzero(np.triu(same_bytes | (best <= threshold), k=1)), strict=True):
        if same_bytes[i, j]:
            rows.append({"a": names[i], "b": names[j], "kind": "exact", "distance": 0})
        else:
            rows.append({"a": names[i], "b": names[j], "kind": KIND_OF_VARIANT[variant_names[best_v[i, j]]],
                         "distance": int(best[i, j])})
    return pd.DataFrame(rows, columns=["a", "b", "kind", "distance"])


def score_hunt(found: pd.DataFrame, key: list[dict]) -> int:
    """How many planted copies in `key` appear as a pair (either order) in `found`."""
    pairs = {frozenset((r["a"], r["b"])) for _, r in found.iterrows()}
    return sum(frozenset((k["copy"], k["source"])) in pairs for k in key)


# %%
# @title 🔎 Compare Duplicate Detection Methods { display-mode: "form" }
plain = find_copies(hunt_files, threshold=PAIR_THRESHOLD, use_variants=False)
variant = find_copies(hunt_files, threshold=PAIR_THRESHOLD, use_variants=True)

print(f"Ground truth of intentionally added duplicates:\t{len(duplicate_key)}")
print(f"MD5 + standard pHash detected:\t\t\t{score_hunt(plain, duplicate_key)}")
print(f"MD5 + transformation-aware pHash detected:\t{score_hunt(variant, duplicate_key)}")

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
  <ul>
    <li><strong>MD5:</strong> Detects exact copies because their file bytes are identical.</li>
    <li><strong>Standard pHash:</strong> Can also detect recompressed images, but may miss flipped or rotated copies.</li>
    <li><strong>Transformation-aware pHash:</strong> Detects all five intentionally added duplicates by additionally comparing flipped and rotated versions.</li>
    <li><strong>Trade-off:</strong> Checking more transformations increases computation and may also produce more false positives.</li>
  </ul>
  <p><strong>Interpretation:</strong> In this experiment, standard hashing detects 3 of 5 known duplicates, compared with 5 of 5 for transformation-aware hashing. This does not mean the latter is always more accurate: unrelated images may also be flagged.</p>
""")


# %% [markdown]
# Why, on one photo and five kinds of copy: the pHash distance to the original, plain vs variant.

# %%
# @title 🔎 How Do Image Transformations Affect pHash? { display-mode: "form" }
_src = load_thumb(images["path"].iloc[0])

_copies = {
    "original": _src,
    "exact copy": _src,
    "resized": _src.resize((int(_src.width * 0.6), int(_src.height * 0.6))),
    "flipped": ImageOps.mirror(_src),
    "recompressed": jpeg_copy(_src, scale=1.0, quality=25),
    "rotated": _src.transpose(Image.Transpose.ROTATE_90),
}

_h0 = phash(_src)
FOUND_COLOR, MISSED_COLOR = "#1E7B6E", "#B03A2E"

fig, axes = plt.subplots(
    1, len(_copies),
    figsize=(2.6 * len(_copies), 3.8)
)

for ax, (name, im) in zip(axes, _copies.items(), strict=True):
    ax.imshow(im)
    frame(ax, INK)
    ax.set_title(name, fontsize=10)

    if name == "original":
        ax.text(
            0.5, -0.12, "Reference image",
            transform=ax.transAxes, ha="center", fontsize=9
        )
        continue

    d_plain = phash(im) - _h0
    d_variant = min(phash(v) - _h0 for v in variants(im).values())

    for k, d in enumerate((d_plain, d_variant)):
        detected = d <= HASH_THRESHOLD
        verdict = "✓" if detected else "✗"
        color = FOUND_COLOR if detected else MISSED_COLOR

        ax.text(
            0.5, -0.10 - 0.13 * k,
            f"{d} bits  {verdict}",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=10, fontweight="bold", color=color
        )

fig.suptitle(
    f"Which copies are detected? (Threshold: {HASH_THRESHOLD} bits)",
    fontsize=13, color=INK
)

fig.text(0.5, 0.12, "First row: Standard pHash    |    Second row: Transformation-aware pHash",
         ha="center", fontsize=10, color=INK)

fig.subplots_adjust(bottom=0.30, top=0.82, wspace=0.12)

display(fig)
plt.close(fig)

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
  <ul>
    <li><strong>Standard pHash:</strong> Detects exact, resized, and recompressed copies, but may miss flipped or rotated images because their bit distances are larger.</li>
    <li><strong>Transformation-aware pHash:</strong> Also detects flipped and rotated copies by comparing multiple transformations. Reversing a flip or rotation can reduce the bit distance to zero.</li>
    <li><strong>Trade-off:</strong> Computing additional hashes increases computational cost and may produce more false positives.</li>
  </ul>
  <p><strong>Important:</strong> A bit distance of zero means the perceptual hashes match, not necessarily that the image files are identical.</p>
""")


# %% [markdown]
# ### 🔴 Identifying Unusual Images (Optional)
#
# Images with unusual characteristics may reveal interesting variations or potential data quality issues.
#
# We use color histograms to identify images whose color distributions differ most from others in the same class.
#
# **Explore:**
# 1. Run the code and inspect the histogram distances across classes.
# 2. Browse the two most unusual images per class.
#
# **Observe:**
# - What makes these images unusual (e.g., background, lighting, framing)?
# - Are the images and their labels valid?
# - Why does a high histogram distance not necessarily indicate a data quality issue?
#
# **Go Deeper (optional):**
# - What are the limitations of color histograms for identifying unusual images?

# %%
# @title 🔎 Find Unusual Images Using Color Histograms { display-mode: "form" }
N_ATYPICAL = 1 if SMOKE else 2  # Images shown per class
_features = channel_hists.reshape(len(image_stats), -1)
_class_mean = {c: _features[(image_stats["class"] == c).to_numpy()].mean(axis=0) for c in CLASSES}
image_stats["atypicality"] = [float(np.abs(f - _class_mean[c]).sum())
                              for f, c in zip(_features, image_stats["class"], strict=True)]

fig, ax = plt.subplots(figsize=(9, 3.4))
for k, c in enumerate(CLASSES):
    _vals = image_stats.loc[image_stats["class"] == c, "atypicality"].to_numpy()
    _jitter = np.random.RandomState(SEED + k).uniform(-0.25, 0.25, len(_vals))
    ax.scatter(k + _jitter, _vals, s=14, color=CLASS_COLORS[c], alpha=0.8)

ax.set_xticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES)
ax.set_ylabel("Color histogram distance to class average")
ax.set_title("Which images have unusual color distributions?")
fig.tight_layout()
display(fig)
plt.close(fig)

print("Higher distances indicate unusual colors, not necessarily incorrect labels.")

least_typical = (image_stats.sort_values("atypicality", ascending=False)
                 .groupby("class", sort=False).head(N_ATYPICAL))

# Browse the most unusual images, two per class
_class_index = 0
_prev = widgets.Button(description="← Previous")
_next = widgets.Button(description="Next →")
_out = widgets.Output()

def _show_class():
    c = CLASSES[_class_index]
    subset = least_typical[least_typical["class"] == c]
    with _out:
        clear_output(wait=True)
        show_rows(subset, f"{c.capitalize()} — images flagged for manual review",
                  lambda r: f"Distance: {r['atypicality']:.2f}", cols=2)
    _prev.disabled = _class_index == 0
    _next.disabled = _class_index == len(CLASSES) - 1

def _navigate(step):
    global _class_index
    _class_index = max(0, min(len(CLASSES) - 1, _class_index + step))
    _show_class()

_prev.on_click(lambda _: _navigate(-1))
_next.on_click(lambda _: _navigate(1))

display(widgets.HBox([_prev, _next]), _out)
_show_class()

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
  <ul>
    <li><strong>Unusual images:</strong> High histogram distances may result from different backgrounds, lighting, colors, or framing.</li>
    <li><strong>Data quality:</strong> Unusual images are not necessarily problematic. They may represent valid variations within a class. Images and labels should be checked before making changes.</li>
    <li><strong>Limitations:</strong> Color histograms ignore spatial structure and object identity. They cannot reliably distinguish unusual backgrounds from incorrect labels or other data quality issues.</li>
  </ul>
""")


# %% [markdown]
# ### 🔴 Exploring Image Embeddings (Optional)
#
# Images can be represented by numerical features. While **color histograms** describe color distributions, **pretrained visual embeddings** capture more complex visual patterns. We compare both representations to explore image similarity, visualize class distributions using UMAP, and identify unusual images.
#
# **Before running the code:**
# - Do you expect color histograms and pretrained visual embeddings to identify the same similar images and outliers?
#
# **Explore:**
# 1. Compare the UMAP visualizations of color histograms and pretrained visual embeddings.
# 2. Explore the most similar image pairs and unusual images within each class.
# 3. Compare the median within-class similarities across animal classes.
#
# **Observe and interpret:**
# - How do the class distributions differ between the two representations?
# - Do the two representations identify the same similar image pairs and unusual images?
# - Which classes have the highest and lowest within-class similarity, and what might explain these differences?
# - How useful are these approaches for identifying potential data quality issues?
#
# **Go Deeper (optional):**
# - Why might two images have similar visual content but very different color distributions?

# %%
# @title 📘 Image Embeddings, Similarity, and UMAP { display-mode: "form" }
callout("theory", "📘 Image Embeddings, Similarity, and UMAP", r"""
<p>Different image representations capture different information:</p>

<ul>
  <li><strong>Color histograms</strong> describe color distributions but ignore spatial arrangement and object identity.</li>
  <li><strong>Pretrained visual embeddings</strong> capture learned visual features, such as shapes, textures, and broader visual patterns.</li>
  <li><strong>Cosine similarity</strong> measures how closely two feature vectors align. Higher values indicate greater similarity in the chosen representation.</li>
  <li><strong>UMAP</strong> projects high-dimensional features into two dimensions for visualization. It approximates local relationships, so distances in the plot should not be interpreted as exact similarities.</li>
</ul>

<p><strong>Important:</strong> Similar images are not necessarily duplicates, and unusual images are not necessarily mislabeled. Results depend on the representation and should always be checked against the original images.</p>
""")


# %%
# @title Helpers: Image Embeddings { display-mode: "form" }
EMBEDDINGS_NAME = "coco_animals_dinov2.npz"  # npz keys: ids (n,), embeddings (n, 384); matched by image id


def load_embeddings(ids) -> np.ndarray | None:
    """The precomputed DINOv2 vectors of the images `ids`, in that order, or None; prints what happened.

    Reads data/coco_animals_dinov2.npz; in Colab it is fetched from DATA_BASE_URL like the other data files.
    None when the file cannot be read or does not hold every id (another dataset, for example).
    """
    try:
        z = np.load(fetch_manifest(EMBEDDINGS_NAME))
        by_id = dict(zip(z["ids"].tolist(), z["embeddings"], strict=True))
    except (RuntimeError, OSError, ValueError, KeyError) as err:
        print(f"Pretrained visual embeddings: no precomputed file to read ({err}).")
        return None
    missing = [i for i in ids if i not in by_id]
    if missing:
        print(f"Pretrained visual embeddings: the precomputed file holds {len(ids) - len(missing)} of these "
              f"{len(ids)} images, not all of them.")
        return None
    print(f"Pretrained visual embeddings: loaded from {EMBEDDINGS_NAME} ({len(ids)} images, "
          f"{z['embeddings'].shape[1]} numbers each), computed beforehand with {z['model']}.")
    return np.stack([by_id[i] for i in ids]).astype(np.float32)


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity between every pair of rows (after centering the columns)."""
    centered = vectors - vectors.mean(axis=0)
    norm = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-9
    unit = centered / norm
    return unit @ unit.T


# %%
# @title Helpers: Pretrained Visual Embeddings { display-mode: "form" }
# The park's DINOv2 vectors come precomputed (data/coco_animals_dinov2.npz). The model itself runs only when
# COMPUTE_EMBEDDINGS is ticked, or on a GPU runtime when no precomputed vectors fit the images.
COMPUTE_EMBEDDINGS = False  # @param {type:"boolean"}
MODEL_NAME = "facebook/dinov2-small"
CPU_IMAGES_PER_SECOND = 5  # a rough guess for Colab's free CPU runtime, used only for the time estimate


def has_gpu() -> bool:
    """True on a runtime with an NVIDIA GPU (checked without importing torch)."""
    return shutil.which("nvidia-smi") is not None


def dinov2_embeddings(paths, batch_size=16):
    import torch  # only here: the CPU path never pays for importing torch and transformers
    from tqdm.auto import tqdm
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    dino_model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    vectors = []

    for start in tqdm(range(0, len(paths), batch_size), desc="Extracting visual features"):
        batch = []

        for path in paths[start:start + batch_size]:
            with Image.open(path) as im:
                batch.append(im.convert("RGB"))

        inputs = dino_processor(images=batch, return_tensors="pt").to(device)

        with torch.inference_mode():
            features = dino_model(**inputs).last_hidden_state[:, 0]
            features = torch.nn.functional.normalize(features, dim=1)

        vectors.append(features.cpu().numpy())

    return np.concatenate(vectors).astype("float32")


# %%
# @title Precompute Image Representations and UMAP { display-mode: "form" }
import umap

CACHE_DIR = DATA_DIR / "cache" / "embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

embedding_paths = image_stats["path"].tolist()
embedding_ids = image_stats["id"].tolist()

def cached_features(name, compute):
    key = hashlib.sha256(
        (name + MODEL_NAME + repr(embedding_paths)).encode()
    ).hexdigest()[:16]
    cache_path = CACHE_DIR / f"{name}_{key}.npy"

    if cache_path.exists():
        print(f"Loaded {name} from cache.")
        return np.load(cache_path)

    features = compute()
    np.save(cache_path, features)
    print(f"Computed and cached {name}.")
    return features

representations = {
    "Color histograms": channel_hists.reshape(
        len(image_stats), -1
    ).astype("float32"),
}

if COMPUTE_EMBEDDINGS:
    _vectors = cached_features("global_dinov2", lambda: dinov2_embeddings(embedding_paths))
else:
    _vectors = load_embeddings(embedding_ids)
    if _vectors is None and has_gpu():
        print("A GPU is available: computing the pretrained visual embeddings now.")
        _vectors = cached_features("global_dinov2", lambda: dinov2_embeddings(embedding_paths))
if _vectors is not None:
    representations["Pretrained visual embeddings"] = _vectors
else:
    _minutes = max(1, round(len(embedding_paths) / CPU_IMAGES_PER_SECOND / 60))
    print(f"Pretrained visual embeddings skipped: on this CPU they would take roughly {_minutes} minute{'s' * (_minutes != 1)}. To "
          "compute them anyway, tick COMPUTE_EMBEDDINGS in the cell above and run both cells again; until then "
          "the next cells use the color histograms.")

# Compute or load one UMAP projection per representation
projections = {}
UMAP_SETTINGS = dict(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=SEED
)

for name, features in representations.items():
    key = hashlib.sha256(
        np.ascontiguousarray(features).tobytes()
        + repr(UMAP_SETTINGS).encode()
    ).hexdigest()[:16]

    cache_path = CACHE_DIR / f"umap_{key}.npy"

    if cache_path.exists():
        projections[name] = np.load(cache_path)
        print(f"Loaded UMAP: {name}")
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="n_jobs value 1 overridden"
            )
            projections[name] = umap.UMAP(
                **UMAP_SETTINGS
            ).fit_transform(features)

        np.save(cache_path, projections[name])
        print(f"Computed and cached UMAP: {name}")

print("Ready:", ", ".join(representations))

# %%
# @title 🔎 Explore Image Representations with UMAP { display-mode: "form" }
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.transform import factor_cmap

# Choose representation before running the plot
UMAP_FEATURES = "Pretrained visual embeddings" # @param ["Color histograms", "Pretrained visual embeddings"]
umap_features = UMAP_FEATURES if UMAP_FEATURES in representations else "Color histograms"  # if skipped above

def thumbnail_uri(path):
    im = load_thumb(path)
    im.thumbnail((120, 120))
    buf = BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

projection = projections[umap_features]

umap_source = ColumnDataSource(dict(
    x=projection[:, 0],
    y=projection[:, 1],
    cls=image_stats["class"].tolist(),
    image=[thumbnail_uri(umap_fig) for umap_fig in image_stats["path"]],
))

output_notebook()

umap_fig = figure(
    title=f"UMAP of Image Representations — {umap_features}",
    width=850,
    height=500,
    tools="pan,wheel_zoom,box_zoom,reset,save",
    active_scroll="wheel_zoom"
)

umap_fig.scatter(
    "x", "y",
    source=umap_source,
    size=8,
    alpha=0.8,
    color=factor_cmap(
        "cls",
        palette=[CLASS_COLORS[c] for c in CLASSES],
        factors=list(CLASSES)
    ),
    legend_field="cls"
)

umap_fig.add_tools(HoverTool(tooltips="""
    <div style="width:150px;">
        <img src="@image" width="130"
             style="max-height:130px;object-fit:contain;">
        <div><b>@cls</b></div>
    </div>
"""))

umap_fig.xaxis.axis_label = "UMAP 1"
umap_fig.yaxis.axis_label = "UMAP 2"
umap_fig.legend.title = "Class"
umap_fig.legend.location = "top_right"
umap_fig.legend.click_policy = "hide"

show(umap_fig)

# %%
# @title 🔎 Explore Similar and Unusual Images { display-mode: "form" }
SIMILARITY_FEATURES = "Pretrained visual embeddings" # @param ["Color histograms", "Pretrained visual embeddings"]
similarity_features = SIMILARITY_FEATURES if SIMILARITY_FEATURES in representations else "Color histograms"

features = representations[similarity_features]
image_classes = image_stats["class"].to_numpy()
indices = np.arange(len(image_stats))

sim = cosine_similarity_matrix(features)
np.fill_diagonal(sim, -np.inf)

# Rank all image pairs by cosine similarity
i, j = np.triu_indices(len(sim), k=1)
order = np.argsort(sim[i, j])[::-1]
top_pairs = [(int(i[k]), int(j[k])) for k in order]

# Rank images by average similarity to others in their class
class_similarity = np.array([
    sim[k, (image_classes == c) & (indices != k)].mean()
    for k, c in enumerate(image_classes)
])

outlier_indices = {
    c: np.where(image_classes == c)[0][
        np.argsort(class_similarity[image_classes == c])
    ].tolist()
    for c in CLASSES
}

# Exploration controls
sim_mode = widgets.ToggleButtons(
    options=["Most similar pairs", "Unusual images"],
    description="Explore:"
)
sim_class_picker = widgets.Dropdown(
    options=list(CLASSES),
    description="Class:"
)
sim_n_pairs = widgets.BoundedIntText(
    value=10, min=1, max=min(100, len(top_pairs)),
    description="Pairs:",
    layout=widgets.Layout(width="160px")
)
sim_n_unusual = widgets.IntSlider(
    value=4, min=2, max=10, step=1,
    description="Images:",
    continuous_update=False,
    style={"description_width": "initial"}
)
sim_prev_btn = widgets.Button(description="← Previous")
sim_next_btn = widgets.Button(description="Next →")
sim_counter = widgets.HTML()
sim_preview = widgets.Output()
sim_position = 0

def sim_update_view(change=None):
    global sim_position

    if sim_mode.value == "Most similar pairs":
        total = min(sim_n_pairs.value, len(top_pairs))
        sim_position = max(0, min(sim_position, total - 1))

        a, b = top_pairs[sim_position]
        rows = image_stats.iloc[[a, b]]

        title = (
            f"Pair {sim_position + 1}/{total} · "
            f"cosine similarity: {sim[a, b]:.2f}"
        )
        labels = lambda r: r["class"]
        cols = 2

        sim_prev_btn.disabled = sim_position == 0
        sim_next_btn.disabled = sim_position == total - 1
        sim_counter.value = f"{sim_position + 1}/{total}"

    else:
        c = sim_class_picker.value
        selected = outlier_indices[c][:sim_n_unusual.value]
        rows = image_stats.iloc[selected].copy()
        rows["class_similarity"] = class_similarity[selected]

        title = f"{c.capitalize()} · lowest average similarity within class"
        labels = lambda r: f"Avg. similarity: {r['class_similarity']:.2f}"
        cols = min(4, len(selected))

    with sim_preview:
        clear_output(wait=True)
        show_rows(rows, title, labels, cols=cols)

    is_pairs = sim_mode.value == "Most similar pairs"
    sim_class_picker.layout.display = "none" if is_pairs else ""
    sim_n_pairs.layout.display = "" if is_pairs else "none"
    sim_n_unusual.layout.display = "none" if is_pairs else ""
    sim_prev_btn.layout.display = "" if is_pairs else "none"
    sim_next_btn.layout.display = "" if is_pairs else "none"
    sim_counter.layout.display = "" if is_pairs else "none"

def sim_navigate(step):
    global sim_position
    sim_position = max(0, min(sim_n_pairs.value - 1, sim_position + step))
    sim_update_view()

def sim_reset_view(change=None):
    global sim_position
    sim_position = 0
    sim_update_view()

sim_prev_btn.on_click(lambda _: sim_navigate(-1))
sim_next_btn.on_click(lambda _: sim_navigate(1))

sim_mode.observe(sim_reset_view, names="value")
sim_class_picker.observe(sim_reset_view, names="value")
sim_n_pairs.observe(sim_reset_view, names="value")
sim_n_unusual.observe(sim_reset_view, names="value")

display(widgets.HBox([sim_mode, sim_class_picker, sim_n_pairs, sim_n_unusual]))
display(widgets.HBox([sim_prev_btn, sim_next_btn, sim_counter]))
display(sim_preview)

sim_update_view()

# %%
# @title 🔎 Compare Within-Class Image Similarity { display-mode: "form" }
for c in CLASSES:
    idx = np.where(image_classes == c)[0]
    values = sim[np.ix_(idx, idx)]
    values = values[np.triu_indices(len(idx), k=1)]
    print(f"{c:8s} median similarity: {np.median(values):.3f}")

# %%
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul>
  <li><strong>UMAP:</strong> Pretrained embeddings may group visually similar
  animals more clearly than color histograms. However, overlapping clusters
  and 2D distances should be interpreted cautiously.</li>

  <li><strong>Similar pairs:</strong> Color histograms emphasize similar color
  distributions, while pretrained embeddings capture similarities in
  appearance, shape, and scene content. The closest pairs may therefore differ.</li>

  <li><strong>Unusual images:</strong> The two representations may identify
  different outliers. Unusual colors do not necessarily imply unusual
  visual content, or vice versa.</li>

  <li><strong>Within-class similarity:</strong> Higher median similarity suggests
  more consistent image representations within a class. Differences may reflect
  animal appearance, backgrounds, viewpoints, or dataset composition,
  rather than classification accuracy.</li>

  <li><strong>Data quality:</strong> Both approaches can highlight images worth
  inspecting, but neither establishes labeling errors without further examination.</li>

  <li><strong>Go Deeper:</strong> Two images may depict the same type of animal
  but have different colors due to lighting, backgrounds, or appearance.
  Pretrained embeddings may still recognize similarities in their shapes
  and visual structures.</li>
</ul>

<p><strong>Key takeaway:</strong> Image similarity depends on the chosen
representation. Comparing representations reveals different aspects
of visual diversity and potential data quality issues.</p>
""")


