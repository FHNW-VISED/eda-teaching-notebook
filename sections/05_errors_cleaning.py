# %% [markdown]
# ## 5. Common dataset errors & cleaning
#
# Sections 2 to 4 asked whether the data fits the task; now: is it wrong?

# %% [markdown]
# <!-- callout: objective | Objective -->
# Find duplicates, flipped and rotated copies and suspicious labels in a dataset, and produce a cleaned
# table you can split.

# %% [markdown]
# A duplicate that lands in both train and test makes the score lie, and a wrong label teaches the model a
# wrong rule. Neither shows up in a bar chart, so this section looks for them on purpose.

# %% [markdown]
# <!-- callout: recipe | Recipe · Step 1, continued: the two bugs Karpathy names -->
# *"One time I discovered that the data contained duplicate examples. Another time I found corrupted
# images / labels."*
#
# Andrej Karpathy, [A Recipe for Training Neural
# Networks](https://karpathy.github.io/2019/04/25/recipe/) (2019).

# %% [markdown]
# This is ordinary: **PetFinder.my Pawpularity** (2021) shipped near-identical photos with *different* scores,
# and **Cassava Leaf Disease** (2020) identical leaf photos with two different labels. Both were built with
# care. Yours will have duplicates too, unless you look.

# %% [markdown]
# ### Exact duplicates: the same bytes twice
#
# The cheapest check first. Two files that are byte-for-byte identical have the same **MD5 hash**, so one
# pass over the files finds every exact copy, whatever its name.

# %% [markdown]
# <!-- callout: questions | Questions -->
# COCO val2017 is a curated benchmark that thousands of papers evaluate on. Does that mean it has no exact
# duplicates? Decide before you run.

# %% [markdown]
# **Q10.** 🟢 How many byte-identical pairs do you expect in the Play Dataset: none, a handful, or many? Write
# your guess, then run the next two cells.

# %%
# @title Hash helpers (MD5, perceptual hash, flipped and rotated variants) { display-mode: "form" }
import hashlib
import shutil
from io import BytesIO

import imagehash
from PIL import Image, ImageOps

THUMB_FOR_HASH = 256  # perceptual hashes are computed on a thumbnail: fast, and the same for every file
EXTREME_SIDE_PX = (64, 6000)  # longest side below / above this is "extreme"


def file_md5(path) -> str:
    """MD5 of the file's bytes: identical files share it, a one-pixel change breaks it."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def phash(image: Image.Image) -> imagehash.ImageHash:
    """Perceptual hash of a PIL image (64 bits). `h1 - h2` is the number of bits that differ (0–64)."""
    return imagehash.phash(image)


def variants(image: Image.Image) -> dict[str, Image.Image]:
    """The image as stored plus its mirrored, flipped and rotated versions, keyed by what was done."""
    return {
        "original": image,
        "flip_lr": ImageOps.mirror(image),
        "flip_tb": ImageOps.flip(image),
        "rot90": image.transpose(Image.Transpose.ROTATE_90),
        "rot180": image.transpose(Image.Transpose.ROTATE_180),
        "rot270": image.transpose(Image.Transpose.ROTATE_270),
    }


def hash_bits(hashes: list[imagehash.ImageHash]) -> np.ndarray:
    """Stack ImageHash objects into an (n, 64) boolean matrix, so distances can be computed in one go."""
    return np.array([h.hash.flatten() for h in hashes], dtype=bool)


def hash_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) matrix of Hamming distances between two (n, 64) hash matrices."""
    return (a[:, None, :] != b[None, :, :]).sum(axis=-1)


def exact_duplicate_groups(paths: list) -> list[list]:
    """Groups of paths whose bytes are identical (only groups with 2+ members), in first-seen order."""
    by_md5: dict[str, list] = {}
    for p in paths:
        by_md5.setdefault(file_md5(p), []).append(p)
    return [group for group in by_md5.values() if len(group) > 1]


def is_readable(path) -> bool:
    """True if PIL can open and fully decode the file (sections 10 and 11 use it)."""
    try:
        with Image.open(path) as im:
            im.draft("RGB", (THUMB_FOR_HASH, THUMB_FOR_HASH))  # a reduced decode still reads every byte
            im.load()
        return True
    except Exception:
        return False


def flag_odd_files(stats: pd.DataFrame) -> pd.DataFrame:
    """One row per flagged file of `stats` (rows of image_stats): id, class, flags. Empty if all is well."""
    rows = []
    for _, r in stats.iterrows():
        flags = []
        if not is_readable(r["path"]):
            flags.append("unreadable")
        if r["mode"] != "RGB":
            flags.append(f"mode {r['mode']}")
        if r["long_side"] < EXTREME_SIDE_PX[0] or r["long_side"] > EXTREME_SIDE_PX[1]:
            flags.append(f"extreme size {r['width']}×{r['height']}")
        if flags:
            rows.append({"id": r["id"], "class": r["class"], "flags": ", ".join(flags)})
    return pd.DataFrame(rows, columns=["id", "class", "flags"])


# %%
# @title Q10 MD5 of every file in the Play Dataset; identical bytes share one hash { display-mode: "form" }
exact_groups = exact_duplicate_groups(images["path"].tolist())
print(f"{len(images)} files · {len(exact_groups)} groups of byte-identical files")
for _group in exact_groups[:5]:
    print("  ", " = ".join(Path(p).name for p in _group))

# %% [markdown]
# 💡 **Worked answer (Q10).** COCO val2017 was deduplicated by its authors, so expect none or close to it;
# the park's re-exported memory cards will not be so tidy, so run this one-line check on every folder you
# get. But MD5 only sees identical bytes: re-save a photo at another JPEG quality and the hash changes
# completely. We need a hash that looks at the picture, not the file.

# %% [markdown]
# ### Near-duplicates: the same picture, different bytes

# %% [markdown]
# <!-- callout: theory | Theory · Perceptual hashing -->
# A **perceptual hash** (pHash, `imagehash` package) turns a photo into 64 bits that describe its coarse
# shapes:
#
# - shrink to 32×32 and drop the colour, so size and JPEG noise disappear;
# - describe the picture as waves (a DCT), keep the 64 slowest ones (the big blobs), and write 1 where a wave
#   is above the median, otherwise 0.
#
# A resized or re-compressed copy keeps the same big blobs, so it gets almost the same bits. The **Hamming
# distance** is how many of the 64 bits differ: 0 = same picture, about 32 = unrelated. What counts as "the
# same picture" is a threshold you choose.

# %%
# @title How the 64 bits are made, and why a near copy gets almost the same ones { display-mode: "form" }
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.fft import dctn

HIGHLIGHT = "#E67E22"  # differing bits
_c0 = images["class"].iloc[0]
_pool = images[images["class"] != _c0]
_orig = load_thumb(images["path"].iloc[0], THUMB_FOR_HASH)
_unrelated = load_thumb((_pool if len(_pool) else images)["path"].iloc[-1], THUMB_FOR_HASH)
_buf = BytesIO()
_orig.resize((int(_orig.width * 0.7), int(_orig.height * 0.7))).save(_buf, "JPEG", quality=55)
_near = Image.open(_buf).convert("RGB")  # a smaller, re-compressed copy: different bytes, same picture

_grey = _orig.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
_dct = dctn(np.asarray(_grey, dtype=float), type=2)
_bits = {k: phash(im).hash for k, im in (("orig", _orig), ("near", _near), ("other", _unrelated))}


def _draw_bits(ax, bits, other=None, title=""):
    """An 8×8 grid of 0/1; cells that differ from `other` get an orange frame."""
    ax.imshow(bits, cmap=ListedColormap(["#EAF1F2", "#5B8E96"]), vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for (r, c), b in np.ndenumerate(bits):
        ax.text(c, r, str(int(b)), ha="center", va="center", fontsize=7, color="white" if b else INK)
        if other is not None and b != other[r, c]:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=HIGHLIGHT, lw=2.5))
    ax.set_title(title, fontsize=9)


fig, axes = plt.subplots(2, 4, figsize=(10, 5.2))
axes[0, 0].imshow(_orig)
axes[0, 0].set_title("1. the photo", fontsize=9)
axes[0, 1].imshow(_grey, cmap="gray")
axes[0, 1].set_title("2. shrink to 32×32, grayscale", fontsize=9)
axes[0, 2].imshow(np.log1p(np.abs(_dct)), cmap="viridis")
axes[0, 2].add_patch(Rectangle((-0.5, -0.5), 8, 8, fill=False, ec=HIGHLIGHT, lw=2.5))
axes[0, 2].set_title("3. DCT: keep the top-left 8×8\n(the coarse shapes)", fontsize=9)
_draw_bits(axes[0, 3], _bits["orig"], title="4. above the median? 1 / 0\n= 64 bits")
for ax in axes[0, :3]:
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
_d_near, _d_other = int((_bits["orig"] != _bits["near"]).sum()), int((_bits["orig"] != _bits["other"]).sum())
_draw_bits(axes[1, 0], _bits["orig"], title="original")
_draw_bits(axes[1, 1], _bits["near"], _bits["orig"], f"re-compressed copy:\n{_d_near} bits differ")
_draw_bits(axes[1, 2], _bits["other"], _bits["orig"], f"unrelated photo:\n{_d_other} bits differ")
axes[1, 3].axis("off")
_note = "orange = a bit that differs\nHamming distance =\nnumber of orange cells\n\n"
_note += "near copy: small\nunrelated: about 32"
axes[1, 3].text(0, 0.5, _note, fontsize=10, va="center")
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# <!-- callout: your-turn | Your turn · The duplicate threshold -->
# `HASH_THRESHOLD` is the largest Hamming distance (0–64) at which two photos count as near-duplicates.
# Keep the default `8` for Q11 and look at the closest pairs. Then edit the value (a strict `2`, a loose
# `14`), run this cell, and re-run the Q11 cell below to see what enters the list. Set it back to `8`
# before going on: later cells use it too.

# %%
# ✏️ YOUR TURN
HASH_THRESHOLD = 8

if not isinstance(HASH_THRESHOLD, int) or not 0 <= HASH_THRESHOLD <= 64:
    raise SystemExit(f"HASH_THRESHOLD must be a whole number between 0 and 64, not {HASH_THRESHOLD!r}")

# %% [markdown]
# **Q11.** 🟡 With your threshold, how many pairs of the Play Dataset do you expect to be flagged, and will
# the closest pairs be true copies or just two photos of the same kind of scene? Guess, then run.

# %%
# @title Q11 pHash of every photo and its variants, then the closest pairs { display-mode: "form" }
VARIANT_NAMES = list(variants(Image.new("RGB", (4, 4))))
_per_variant: dict[str, list] = {v: [] for v in VARIANT_NAMES}
for _p in images["path"]:
    _thumb = load_thumb(_p, THUMB_FOR_HASH)
    for _v, _im in variants(_thumb).items():
        _per_variant[_v].append(phash(_im))
VARIANT_HASHES = {v: hash_bits(hs) for v, hs in _per_variant.items()}  # each (n_images, 64)
HASHES = VARIANT_HASHES["original"]

_dist = hash_distances(HASHES, HASHES)
np.fill_diagonal(_dist, 99)
nearest_distance = _dist.min(axis=1)  # each photo's distance to its closest other photo
_i, _j = np.where(np.triu(_dist <= HASH_THRESHOLD, k=1))
near_pairs = pd.DataFrame({"a": images["id"].to_numpy()[_i], "b": images["id"].to_numpy()[_j],
                           "distance": _dist[_i, _j]}).sort_values("distance").reset_index(drop=True)
print(f"pairs within {HASH_THRESHOLD} bits: {len(near_pairs)} · median nearest-neighbour distance: "
      f"{np.median(nearest_distance):.0f} bits")

fig, ax = plt.subplots(figsize=(9, 3.4))
_bins = np.arange(-0.5, 41, 2)
ax.hist([nearest_distance[(images["class"] == c).to_numpy()] for c in CLASSES], bins=_bins, stacked=True,
        color=[CLASS_COLORS[c] for c in CLASSES], label=CLASSES)
ax.axvline(HASH_THRESHOLD + 0.5, color=INK, ls=":", lw=1)
ax.text(HASH_THRESHOLD + 1, ax.get_ylim()[1] * 0.9, f"HASH_THRESHOLD = {HASH_THRESHOLD}", fontsize=9)
ax.set_xlabel("Hamming distance to the closest other photo (bits)")
ax.set_ylabel("photos")
ax.set_title("How close is each photo's nearest neighbour?")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
display(fig)
plt.close(fig)

# The closest pairs, side by side, whether or not they pass the threshold: this is what "near" looks like.
N_PAIRS_TO_SHOW = 2 if SMOKE else 4
_upper = np.where(np.triu(np.ones_like(_dist, dtype=bool), k=1), _dist, 99)  # each pair once
_flat = np.argsort(_upper, axis=None)[:N_PAIRS_TO_SHOW]
_closest = list(zip(*np.unravel_index(_flat, _dist.shape), strict=True))
if len(near_pairs):
    display(near_pairs.head(10))
_rows = pd.concat([image_stats.iloc[[a, b]] for a, b in _closest])
_dist_of = {row_id: _dist[a, b] for a, b in _closest for row_id in images["id"].iloc[[a, b]]}
show_rows(_rows, "closest pairs by perceptual hash (each pair side by side)",
          lambda r: f"{r['class']} · {_dist_of[r['id']]} bits apart", cols=2)

# %% [markdown]
# 💡 **Worked answer (Q11).** Unrelated pictures sit about 30 bits apart, far right of the threshold. Most
# closest pairs on COCO are different photos of a similar scene (two zebras on grass), so the flagged list is
# short: on the full Play Dataset the default catches one true pair, two frames of the same tabby in front of
# the same brick wall (the smoke sample may lack it). A loose threshold adds non-copies, a strict one misses
# re-compressed copies. Keep the default and read the list by eye before deleting.

# %% [markdown]
# <!-- callout: pitfall | Pitfall · A hash distance is not a similarity score -->
# Two burst frames of the same sheep can be 12 bits apart and two unrelated photos 10. The threshold only
# separates "same picture, different file" from everything else; it does not rank how alike two photos are.

# %% [markdown]
# ### A memory card from the park: the guided hunt
#
# The park's intern hands you a memory card of photos. Over the years five things happened to it:
#
# 1. The camera software saved one photo **twice**. Two files, identical bytes.
# 2. Someone exported a **smaller, lower-quality** copy for an e-mail. Different bytes, same picture.
# 3. Someone **mirrored** a photo in a viewer and saved it. The animal now faces the other way: every pixel
#    has moved.
# 4. Someone **rotated** a photo and saved it. Same story, turned on its side.
# 5. Every file got a neutral name (`card_017.jpg`), so nothing says which files belong together.
#
# Find every pair without opening 40 photos one by one. The next cell writes the card into
# `data/cache/hunt5/` (seeded: everyone gets the same one). **The answer key comes after the hunt.**

# %%
# @title Plant the memory card (seeded, at runtime; nothing is redistributed) { display-mode: "form" }
HUNT_DIR = DATA_DIR / "cache" / "hunt5"
N_CARD_ORIGINALS = 12 if SMOKE else 40  # distinct photos on the card (N-014: fewer in smoke mode)
PLANT_KINDS = ["exact", "exact", "flipped", "rotated", "recompressed"]  # ≥ 4 planted copies, always present


def plant_hunt5(dest: Path, n_originals: int, seed: int) -> tuple[dict[str, Path], list[dict]]:
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
            small = rgb.resize((max(8, int(rgb.width * 0.7)), max(8, int(rgb.height * 0.7))))
            small.save(buf, "JPEG", quality=55)
            data = buf.getvalue()
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


# %%
# @title Write the memory card to disk and load its photos { display-mode: "form" }
hunt_files, HUNT5_KEY = plant_hunt5(HUNT_DIR, N_CARD_ORIGINALS, SEED)
hunt_images = {name: load_thumb(path, THUMB_FOR_HASH) for name, path in hunt_files.items()}
print(f"{len(hunt_files)} files on the card in {HUNT_DIR}/ · {len(HUNT5_KEY)} of them are copies of another "
      f"file on the card")

# %% [markdown]
# ### ✍️ Write the flip detector yourself

# %% [markdown]
# A mirrored or rotated copy has moved every pixel, so its plain hash is far from the original's. Before the
# shoot-out, write the one piece plain hashing lacks: compare a photo's hash with the hashes of the other
# photos' flipped and rotated versions.

# %%
# @title Reference solution — run this cell; open it only if you are stuck { display-mode: "form" }
def _ref_find_flipped_copies(image, candidates):
    flipped_hashes = [phash(im) for name, im in variants(image).items() if name != "original"]
    found = []
    for name, cand in candidates.items():
        h = phash(cand)
        if min(h - fh for fh in flipped_hashes) <= HASH_THRESHOLD:
            found.append(name)
    return sorted(found)


# %% [markdown]
# <!-- callout: your-turn | Your turn · `find_flipped_copies` -->
# Fill in the function below: given one photo and a dictionary of candidate photos, return the names of the
# candidates that are a flipped or rotated copy of it. You have `variants(image)`, `phash(image)`, `h1 - h2`
# for the distance and `HASH_THRESHOLD`. The `use()` line checks your answer against the reference on the
# card and continues with the reference either way.

# %%
# ✍️ YOUR CODE
def find_flipped_copies(image, candidates):
    """Return the names of the candidates that are a flipped or rotated copy of `image`.

    image: a PIL image. candidates: dict {name: PIL image} (the image itself is not among them).
    Returns a sorted list of names whose pHash is within HASH_THRESHOLD bits of the pHash of one of
    image's flipped or rotated variants (every variant except "original").
    """
    ...


_flip_source = next(k["source"] for k in HUNT5_KEY if k["kind"] == "flipped")
_others = {name: im for name, im in hunt_images.items() if name != _flip_source}
flipped_copies = use(find_flipped_copies, hunt_images[_flip_source], _others,
                     reference=_ref_find_flipped_copies)
print(f"flipped/rotated copies of {_flip_source}: {flipped_copies}")

# %% [markdown]
# **Q12.** 🟡 The shoot-out. Two detectors run on the card: plain hashing (MD5 plus pHash with your threshold)
# and *variant* hashing, the idea you just wrote, applied to every photo. Which of the first four things
# that happened to the card (twice, smaller, mirrored, rotated) does each one catch? Predict, then run. (The
# fifth, the neutral file names, is why you need a detector at all.)

# %%
# @title Copy finder: plain hashing vs variant hashing { display-mode: "form" }
KIND_OF_VARIANT = {"original": "recompressed", "flip_lr": "flipped", "flip_tb": "flipped",
                   "rot90": "rotated", "rot180": "rotated", "rot270": "rotated"}


def find_copies(files: dict[str, Path], threshold: int, use_variants: bool) -> pd.DataFrame:
    """Pairs of files on a card that are copies of each other: columns a, b, kind, distance.

    kind is "exact" (same bytes), "recompressed" (hashes within `threshold`), or, with use_variants=True,
    "flipped" / "rotated" (a flipped or rotated variant of a hashes within `threshold` of b).
    """
    names = list(files)
    md5s = {n: file_md5(p) for n, p in files.items()}
    thumbs = {n: load_thumb(p, THUMB_FOR_HASH) for n, p in files.items()}
    plain = hash_bits([phash(thumbs[n]) for n in names])
    variant_names = VARIANT_NAMES if use_variants else ["original"]
    # distances[v][i, j]: hash of variant v of file i vs the plain hash of file j
    distances = {v: hash_distances(hash_bits([phash(variants(thumbs[n])[v]) for n in names]), plain)
                 for v in variant_names}
    rows = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            if md5s[a] == md5s[b]:
                rows.append({"a": a, "b": b, "kind": "exact", "distance": 0})
                continue
            best_v = min(variant_names, key=lambda v: distances[v][i, j])
            best_d = int(distances[best_v][i, j])
            if best_d <= threshold:
                rows.append({"a": a, "b": b, "kind": KIND_OF_VARIANT[best_v], "distance": best_d})
    return pd.DataFrame(rows, columns=["a", "b", "kind", "distance"])


def score_hunt(found: pd.DataFrame, key: list[dict]) -> int:
    """How many planted copies in `key` appear as a pair (either order) in `found`."""
    pairs = {frozenset((r["a"], r["b"])) for _, r in found.iterrows()}
    return sum(frozenset((k["copy"], k["source"])) in pairs for k in key)


# %%
# @title Q12 Plain hashing, then variant hashing, on the same card and threshold { display-mode: "form" }
plain_found = find_copies(hunt_files, HASH_THRESHOLD, use_variants=False)
variant_found = find_copies(hunt_files, HASH_THRESHOLD, use_variants=True)
print(f"plain hashing   : {len(plain_found)} pairs → {dict(Counter(plain_found['kind']))}")
print(f"variant hashing : {len(variant_found)} pairs → {dict(Counter(variant_found['kind']))}")
display(variant_found)

# %% [markdown]
# The next cell shows why, on one photo and the five kinds of copy: the pHash distance to the original under
# each, for plain hashing and for variant hashing.

# %%
# @title One photo, five kinds of copy: pHash distance, plain vs variant hashing { display-mode: "form" }
_src = load_thumb(images["path"].iloc[0], THUMB_FOR_HASH)


def _reencode(im, quality):
    buf = BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return Image.open(buf).convert("RGB")


_copies = {
    "original": _src,
    "exact copy": _src,
    "resized": _src.resize((int(_src.width * 0.6), int(_src.height * 0.6))),
    "flipped": ImageOps.mirror(_src),
    "re-encoded": _reencode(_src, 25),
    "rotated": _src.transpose(Image.Transpose.ROTATE_90),
}
_h0 = phash(_src)
FOUND_COLOR, MISSED_COLOR = "#1E7B6E", "#B03A2E"
fig, axes = plt.subplots(1, len(_copies), figsize=(2.3 * len(_copies), 3.3))
for ax, (name, im) in zip(axes, _copies.items(), strict=True):
    ax.imshow(im)
    frame(ax, INK, 2)
    ax.set_title(name, fontsize=9)
    if name == "original":
        ax.set_xlabel("distance to the\noriginal, in bits:", fontsize=8)
        continue
    d_plain = phash(im) - _h0
    d_variant = min(phash(v) - _h0 for v in variants(im).values())
    for k, (label, d) in enumerate((("plain", d_plain), ("variants", d_variant))):
        verdict = "found" if d <= HASH_THRESHOLD else "missed"
        ax.text(0.5, -0.07 - 0.12 * k, f"{label}: {d} → {verdict}", transform=ax.transAxes, ha="center",
                va="top", fontsize=8, color=FOUND_COLOR if d <= HASH_THRESHOLD else MISSED_COLOR)
fig.suptitle(f"Does the copy land within HASH_THRESHOLD = {HASH_THRESHOLD} bits of its original?", color=INK)
fig.subplots_adjust(bottom=0.25, top=0.82, wspace=0.08)
display(fig)
plt.close(fig)

# %% [markdown]
# 💡 **Worked answer (Q12).** Plain hashing catches the byte-identical copies (MD5) and the smaller,
# re-compressed one (pHash survives both), but files the mirrored and rotated copies as new photos: a flip
# moves the hash by dozens of bits. Variant hashing catches them too, because one of the six variants of the
# source hashes like the copy. The price: six hashes per photo instead of one, still seconds for thousands.

# %% [markdown]
# ### The hunt's score and the answer key

# %%
# @title Score the reference detector against the planted key, show each pair { display-mode: "form" }
hunt5_found = score_hunt(variant_found, HUNT5_KEY)
RESULTS["hunt5"] = {"planted": len(HUNT5_KEY), "found": int(hunt5_found),
                    "kinds": dict(Counter(k["kind"] for k in HUNT5_KEY))}
print(f"planted copies found by variant hashing: {hunt5_found} of {len(HUNT5_KEY)}")
print("answer key:")
for _k in HUNT5_KEY:
    print(f"  {_k['copy']}: {_k['kind']} copy of {_k['source']}")

fig, axes = plt.subplots(2, len(HUNT5_KEY), figsize=(2.4 * len(HUNT5_KEY), 5.2), squeeze=False)
for _col, _k in enumerate(HUNT5_KEY):
    for _row, _name in enumerate((_k["source"], _k["copy"])):
        ax = axes[_row, _col]
        ax.imshow(hunt_images[_name])
        frame(ax, INK if _row == 0 else CLASS_COLORS["dog"])
        ax.set_title(f"{_name}\n{'source' if _row == 0 else _k['kind'] + ' copy'}", fontsize=8)
fig.suptitle("the planted copies (top: source, bottom: copy)", color=INK)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# 💡 At the default threshold the detector found all 5 planted copies and reported nothing else. Test your
# own detector the same way: plant a few copies and check it finds them and nothing more.

# %% [markdown]
# ### Suspicious labels: each class's least typical photos

# %% [markdown]
# A wrong label cannot be found by hashing. Without a model, sort each class by how *unlike the rest of its
# class* a photo is, and hand the top of each list to a human. The score (atypicality) is the sum of
# absolute differences between the photo's colour histogram (section 3) and its class's average histogram.
# Most flagged photos will be fine (unusual but correct); the wrong ones are far likelier to be here than in
# a random sample.

# %%
# @title Distance of each photo to its class's mean histogram; list the farthest { display-mode: "form" }
N_ATYPICAL = 1 if SMOKE else 2  # photos shown per class (N-014); 6 classes × 2 = 12 photos
_features = CHANNEL_HISTS.reshape(len(image_stats), -1)
_class_mean = {c: _features[(image_stats["class"] == c).to_numpy()].mean(axis=0) for c in CLASSES}
image_stats["atypicality"] = [float(np.abs(f - _class_mean[c]).sum())
                              for f, c in zip(_features, image_stats["class"], strict=True)]

fig, ax = plt.subplots(figsize=(9, 3.4))
for k, c in enumerate(CLASSES):
    _vals = image_stats.loc[image_stats["class"] == c, "atypicality"].to_numpy()
    _jitter = np.random.RandomState(SEED).uniform(-0.25, 0.25, len(_vals))
    ax.scatter(k + _jitter, _vals, s=14, color=CLASS_COLORS[c], alpha=0.8)
ax.set_xticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES)
ax.set_ylabel("distance to the class's mean histogram")
ax.set_title("How typical is each photo of its class? (high = look at me)")
fig.tight_layout()
display(fig)
plt.close(fig)

least_typical = (image_stats.sort_values("atypicality", ascending=False).groupby("class", sort=False)
                 .head(N_ATYPICAL).sort_values(["class", "atypicality"], ascending=[True, False]))
show_rows(least_typical, "least typical photos per class: does the label fit?",
          lambda r: f"{r['class']} · {r['atypicality']:.1f}", cols=6)

# %% [markdown]
# **Q13.** 🟡 Look at the gallery. For each photo, is the label right? Is the animal even visible? Note the
# ones you would send back to the annotator.

# %% [markdown]
# 💡 **Worked answer (Q13).** Colour cannot tell a cat from a dog, so this list finds photos that are
# *unusual*, not *wrong*: close-ups, night shots, an animal filling a tenth of the frame. Real label errors in
# COCO val are rare at image level (more common in the masks, section 9), so an honest answer is "nothing to
# send back, two to keep an eye on". On a crowd-sourced dataset, this list is where the wrong labels live.

# %%
# @title Embedding helpers (precomputed vectors, or colour histograms as a stand-in) { display-mode: "form" }
EMBEDDINGS_NAME = "coco_animals_embeddings.npz"  # npz keys: ids (n,), embeddings (n, d); matched by image id


def load_embeddings() -> tuple[np.ndarray, str]:
    """(vectors aligned with `images`, source). Source is "precomputed" or "colour histograms (stand-in)".

    Reads data/coco_animals_embeddings.npz, fetching it from DATA_BASE_URL outside smoke mode. Never runs a
    model. Falls back to the colour histograms when the file is missing or does not cover every image.
    """
    path = DATA_DIR / EMBEDDINGS_NAME
    if not path.exists() and not SMOKE:
        download_file(DATA_BASE_URL + EMBEDDINGS_NAME, path, headers=_github_headers())
    if path.exists():
        try:
            z = np.load(path)
            by_id = dict(zip(z["ids"].tolist(), z["embeddings"], strict=True))
            if all(i in by_id for i in images["id"]):
                return np.stack([by_id[i] for i in images["id"]]).astype(np.float32), "precomputed"
        except Exception:
            pass
    return CHANNEL_HISTS.reshape(len(images), -1).astype(np.float32), "colour histograms (stand-in)"


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity between every pair of rows (after centring the columns)."""
    centred = vectors - vectors.mean(axis=0)
    norm = np.linalg.norm(centred, axis=1, keepdims=True) + 1e-9
    unit = centred / norm
    return unit @ unit.T


# %% [markdown]
# ### 🔴 Optional · Nearest pairs and outliers in embedding space

# %% [markdown]
# An **embedding** is a list of numbers that stands for a photo, chosen so that photos with a similar
# *content* get similar numbers (here precomputed; if the file is missing, colour histograms stand in, and
# the plot titles say which). **Similarity** runs from 1 (same direction) down to 0 and below. Two cases:
#
# - **Nearest pair**: two photos with similarity near 1, e.g. two frames of the same sheep seconds apart.
#   Keep one, or keep both on the same side of the split.
# - **Outlier**: a photo with low similarity to the rest of its own class, e.g. a "dog" photo that looks like
#   no other dog. Look at it, then relabel or drop it.
#
# **Q14.** 🔴 Will the closest pairs be the same ones the hash found, and will the outliers overlap with the
# colour-based list from Q13? Guess, then run.

# %%
# @title Q14 Nearest pairs and per-class outliers in embedding space { display-mode: "form" }
embeddings, embedding_source = load_embeddings()
print(f"vectors: {embeddings.shape[0]} × {embeddings.shape[1]} from {embedding_source}")
RESULTS["embedding_source"] = embedding_source

_sim = cosine_similarity_matrix(embeddings)
np.fill_diagonal(_sim, -1)
_i, _j = np.unravel_index(np.argsort(-np.triu(_sim, k=1), axis=None)[:N_PAIRS_TO_SHOW], _sim.shape)
_rows = pd.concat([image_stats.iloc[[a, b]] for a, b in zip(_i, _j, strict=True)])
_sim_of = {row_id: _sim[a, b] for a, b in zip(_i, _j, strict=True) for row_id in images["id"].iloc[[a, b]]}
show_rows(_rows, "Nearest pairs (sim 1 = identical): copies or same scene? Keep one",
          lambda r: f"{r['class']} · sim {_sim_of[r['id']]:.2f}", cols=4)

_class_arr = image_stats["class"].to_numpy()
image_stats["class_similarity"] = [float(_sim[k, (_class_arr == c) & (np.arange(len(_sim)) != k)].mean())
                                   for k, c in enumerate(_class_arr)]
embedding_outliers = (image_stats.sort_values("class_similarity").groupby("class", sort=False)
                      .head(N_ATYPICAL).sort_values(["class", "class_similarity"]))
show_rows(embedding_outliers, "Outliers (low sim to their own class): wrong label? Look, relabel or drop",
          lambda r: f"{r['class']} · sim {r['class_similarity']:.2f}", cols=6)
_overlap = len(set(embedding_outliers["id"]) & set(least_typical["id"]))
print(f"{_overlap} of {len(embedding_outliers)} embedding outliers are also in the colour-based list")

# %% [markdown]
# 💡 **Worked answer (Q14).** The hash finds copies of the same *picture*; the embedding finds photos of the
# same *thing*. The two pair lists overlap on exact and near copies and differ elsewhere, and the two outlier
# lists overlap only partly: colour and content are different notions of "typical".

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · Embeddings, and how to get real ones -->
# <details><summary>Why embeddings find what hashes cannot</summary>
#
# With real embeddings the closest pairs are *semantic* near-duplicates (the same dog, another angle), which
# a hash cannot find, and the outliers are photos whose content does not match the label. With the stand-in
# they are colour matches. To get real embeddings, compute them once on a GPU (any pretrained image model),
# save `ids` and `embeddings` to `data/coco_animals_embeddings.npz`, and the Q14 cell picks them up.
# </details>

# %% [markdown]
# ### Clean the dataset
#
# Time to act. The rule applied below is the one most teams use: keep the first photo of every duplicate
# group (exact, near, flipped or rotated), drop files that do not open, and keep everything else. The result
# is `clean_images`, the table the rest of the notebook uses.
#
# Section 3 left the grayscale and odd-size photos for this section. Photos stored in another mode (L, CMYK…)
# or with an extreme size stay in the table, with a note in a `flags` column; every photo is converted to RGB
# when it is loaded, so none of them breaks a later section.

# %% [markdown]
# <!-- callout: pitfall | Pitfall · Split after cleaning, not before -->
# If you split first and deduplicate each half on its own, a pair with one photo in train and one in test
# survives, and the test score reports how well the model *memorised* that photo. Deduplicate the whole
# table, then split (section 6).

# %%
# @title Find the duplicate groups with variant hashing, then build the cleaned table { display-mode: "form" }
_ids = images["id"].to_numpy()
_drop: dict[int, str] = {}
for _group in exact_groups:  # exact: keep the first file of each group
    _group_ids = [int(images.loc[images["path"] == str(p), "id"].iloc[0]) for p in _group]
    for _gid in sorted(_group_ids)[1:]:
        _drop[_gid] = "exact duplicate"
_best = np.full(_dist.shape, 99)
for _v in VARIANT_NAMES:  # variant hashing across the whole Play Dataset: min distance over the six variants
    _best = np.minimum(_best, hash_distances(VARIANT_HASHES[_v], HASHES))
np.fill_diagonal(_best, 99)
_pi, _pj = np.where(np.triu(np.minimum(_best, _best.T) <= HASH_THRESHOLD, k=1))
for _a, _b in zip(_pi, _pj, strict=True):  # near/flipped/rotated: keep the lower id
    _drop.setdefault(int(max(_ids[_a], _ids[_b])), "near-duplicate (incl. flipped/rotated)")
_odd = flag_odd_files(image_stats)  # files that do not open are dropped silently
_unreadable = _odd.loc[_odd["flags"].str.contains("unreadable"), "id"]
_drop.update({int(i): "unreadable" for i in _unreadable})

clean_images = images[~images["id"].isin(_drop)].copy()
# the non-fatal flags (grayscale, extreme size) stay in a column that sections 10 and 11 read
clean_images["flags"] = clean_images["id"].map(_odd.set_index("id")["flags"]).fillna("")
removed = pd.DataFrame({"id": list(_drop), "reason": list(_drop.values())})
RESULTS["n_clean"] = int(len(clean_images))
RESULTS["n_removed"] = int(len(removed))
print(f"Play Dataset: removed {len(removed)}, kept {len(clean_images)} "
      f"(reasons: {dict(Counter(removed['reason']))})")

# The same rule on the memory card: one file per duplicate group.
_card_drop = set()
for _, _r in variant_found.sort_values(["a", "b"]).iterrows():
    if _r["a"] not in _card_drop:
        _card_drop.add(_r["b"])
print(f"memory card : removed {len(_card_drop)}, kept {len(hunt_files) - len(_card_drop)}")

# %% [markdown]
# 💡 On a curated benchmark the "removed" number is small and the lesson is in the tooling. On the park's own
# uploads, the same cells decide whether you can trust your test score:
#
# 1. A camera fires a burst: three frames of the same sheep, saved as three files.
# 2. A random split puts two frames in train and one in test.
# 3. The test frame is almost a photo the model has already seen, so it scores well by *remembering*.
# 4. In the field it meets new sheep and does worse. Cleaning first would have kept the burst on one side.

# %% [markdown]
# <!-- callout: across-tasks | Across tasks · The same errors, three shapes -->
# - **Classification:** the same dog photo is stored twice, once labelled "dog", once "sheep". Whichever
#   copy lands in the test set, one of them is wrong.
# - **Detection:** a mirrored copy of a photo of two sheep whose boxes were not mirrored (`x' = W − x − w`):
#   both boxes now point at grass (section 8).
# - **Segmentation:** a copied photo whose mask outlines the other animal, or a mask that is swollen or
#   shifted while the label is right (section 9).
#
# Deduplicate once, before splitting, for all three; the label checks differ per task.

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **When you see** a new folder of photos → **do** MD5 for exact copies, then pHash with a threshold you
#   have looked at for near copies.
# - **When you see** that mirrored or rotated copies could exist → **do** hash the six variants of each photo.
# - **When you see** a photo far from the rest of its class → **do** look at it, then relabel or drop it.
# - **When you see** the checks are done → **do** report "removed N, kept M", and split after cleaning.
