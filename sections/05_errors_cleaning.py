# %% [markdown]
# ## 5. Common dataset errors & cleaning
#
# **Objective:** find duplicates, flipped and rotated copies, broken files and suspicious labels in a dataset,
# and produce a cleaned table you can split and train on.
#
# Sections 2 to 4 asked whether the data *fits* the task. This section asks whether the data is *wrong*: the
# same photo stored twice, a copy that someone flipped in a viewer, a file that no longer opens, a label that
# a tired annotator got wrong. None of these show up in a bar chart, and all of them bite later: a duplicate
# that lands in both train and test makes the score lie, and a wrong label teaches the model a wrong rule.
#
# <table width="100%"><tr><td bgcolor="#E8F5E9"><font color="#1B5E20">
# <b>🧾 Recipe · Step 1, continued: the two bugs Karpathy names</b><br>
# <i>"One time I discovered that the data contained duplicate examples. Another time I found corrupted
# images / labels."</i><br>
# Andrej Karpathy, <a href="https://karpathy.github.io/2019/04/25/recipe/">A Recipe for Training Neural
# Networks</a> (2019).
# </font></td></tr></table>
#
# Two stories from Kaggle show how ordinary this is. In the <b>PetFinder.my Pawpularity</b> competition
# (2021), teams found near-identical photos in the training set that carried *different* target scores, and
# asked on the forum whether the same photos also sat in the hidden test set; the hosts answered that
# duplicates between train and test had been removed, and left the pairs inside the training set as they
# were, scores and all. In <b>Cassava Leaf Disease Classification</b> (2020), a competition known for its
# noisy labels, teams reported a pair of identical leaf photos that carried two different disease labels.
# Neither dataset was careless: both were built by people who cared, and both still shipped with duplicates.
# Yours will too, unless you look.
#
# ### Exact duplicates: the same bytes twice
#
# The cheapest check first. Two files that are byte-for-byte identical have the same **MD5 hash**, so one
# pass over the files finds every exact copy, whatever its name.
#
# <table width="100%"><tr><td bgcolor="#E0F7F4"><font color="#00564D">
# <b>❓ Questions</b><br>
# COCO val2017 is a curated benchmark that thousands of papers evaluate on. Does that mean it has no exact
# duplicates? Decide before you run.
# </font></td></tr></table>
#
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
    """True if PIL can open and fully decode the file."""
    try:
        with Image.open(path) as im:
            im.draft("RGB", (THUMB_FOR_HASH, THUMB_FOR_HASH))  # a reduced decode still reads every byte
            im.load()
        return True
    except Exception:
        return False


# %%
# Q10  MD5 of every file in the Play Dataset; identical bytes share one hash.
exact_groups = exact_duplicate_groups(images["path"].tolist())
print(f"{len(images)} files · {len(exact_groups)} groups of byte-identical files")
for _group in exact_groups[:5]:
    print("  ", " = ".join(Path(p).name for p in _group))

# %% [markdown]
# 💡 Whatever the count, this check cost one line and one pass over the files; run it on every folder you are
# handed, because exact copies are the duplicates that slip through most often (a batch uploaded twice, a
# "final" and a "final_v2" export of the same card).
#
# **Worked answer (Q10).** COCO val2017 was deduplicated by its authors, so on the Play Dataset the expected
# answer is none or close to it. That is the exception, not the rule: the park's cameras, like any camera
# upload, will produce exact copies the first time someone re-exports a memory card. And MD5 is blind to
# everything but identical bytes: re-save the same photo at a different JPEG quality and the hash changes
# completely. For that we need a hash that looks at the picture, not the file.
#
# ### Near-duplicates: the same picture, different bytes
#
# <table width="100%"><tr><td bgcolor="#E3F2FD"><font color="#0D47A1">
# <b>📘 Theory · Perceptual hashing</b><br>
# A <b>perceptual hash</b> (here pHash, from the <code>imagehash</code> package) shrinks the photo to a 32×32
# grey thumbnail, takes its low-frequency content (a DCT, as in JPEG), and writes 64 bits saying whether each
# coefficient is above or below the median. Two photos that <i>look</i> the same get almost the same bits
# even when their files differ (resized, re-compressed, slightly cropped). The <b>Hamming distance</b>
# between two hashes is the number of bits that differ: 0 means identical hashes, about 32 means unrelated.
# A threshold on that distance decides what counts as "the same picture", and that threshold is a choice.
# </font></td></tr></table>
#
# <table width="100%"><tr><td bgcolor="#FFF4D6"><font color="#7A4A00">
# <b>✏️ Your turn · The duplicate threshold</b><br>
# <code>HASH_THRESHOLD</code> is the largest Hamming distance (0–64) at which two photos count as
# near-duplicates. Start with the default, look at the closest pairs the next cell shows, then try a strict
# <code>2</code> and a loose <code>14</code> and watch what enters the list.
# </font></td></tr></table>

# %%
# ✏️ YOUR TURN
HASH_THRESHOLD = 8

if not isinstance(HASH_THRESHOLD, int) or not 0 <= HASH_THRESHOLD <= 64:
    raise SystemExit(f"HASH_THRESHOLD must be a whole number between 0 and 64, not {HASH_THRESHOLD!r}")

# %% [markdown]
# **Q11.** 🟡 With your threshold, how many pairs of the Play Dataset do you expect to be flagged, and will
# the closest pairs be true copies or just two photos of the same kind of scene? Guess, then run.

# %%
# Q11  pHash of every photo (and of its flipped/rotated variants, used below), then the closest pairs.
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
# 💡 The histogram sits far to the right of the threshold for almost every photo: unrelated pictures are
# about 30 bits apart, and the threshold only matters for the thin left tail. Look at the pairs: at a few bits
# you see the same scene; at 10 or more you see two different photos that merely share a layout.
#
# **Worked answer (Q11).** On COCO most of the closest pairs are different photos of a similar scene (two
# zebras on grass, a cat on a sofa) at distances above the default threshold, so the flagged list is short.
# Short, not empty: on the full Play Dataset the default threshold catches one true pair, two frames of the
# same tabby cat in front of the same brick wall, looking away in one and at the camera in the other. A
# curated benchmark, and still a burst that slipped through. (The small smoke sample may not contain it.)
# With a loose threshold the list fills up with photos that are *not* copies. That is the trade-off: a strict
# threshold misses re-compressed copies and bursts, a loose one deletes real data. Keep the default for now
# and read the list by eye before deleting anything.
#
# <table width="100%"><tr><td bgcolor="#FDECEA"><font color="#8B1A10">
# <b>⚠️ Pitfall · A hash distance is not a similarity score</b><br>
# Two burst frames of the same sheep can be 12 bits apart and two unrelated photos 10. The threshold only
# separates "same picture, different file" from everything else; it does not rank how alike two photos are.
# And a plain hash is <b>orientation-sensitive</b>: mirror a photo and its hash changes by dozens of bits,
# so a flipped copy looks like a new photo. The next part fixes that.
# </font></td></tr></table>
#
# ### A memory card from the park: the guided hunt
#
# The park's intern hands you a memory card. Some files on it are the same photo twice: the camera software
# saved a copy, someone rotated a photo in a viewer and saved it, a second version was exported at lower
# quality for an e-mail. The next cell writes that card into `data/cache/hunt5/` (seeded, so everyone gets
# the same one) with neutral file names. **The answer key is printed after the hunt, not before.**

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
hunt_files, HUNT5_KEY = plant_hunt5(HUNT_DIR, N_CARD_ORIGINALS, SEED)
hunt_images = {name: load_thumb(path, THUMB_FOR_HASH) for name, path in hunt_files.items()}
print(f"{len(hunt_files)} files on the card in {HUNT_DIR}/ · {len(HUNT5_KEY)} of them are copies of another "
      f"file on the card")

# %% [markdown]
# **Q12.** 🟡 The shoot-out. Two detectors run on the card: plain hashing (MD5 plus pHash with your threshold)
# and *variant* hashing, which also hashes each photo's mirrored, flipped and rotated versions. Predict which
# kinds of copy each one catches, then run.

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
# Q12  Plain hashing first, then variant hashing, on the same card and the same threshold.
plain_found = find_copies(hunt_files, HASH_THRESHOLD, use_variants=False)
variant_found = find_copies(hunt_files, HASH_THRESHOLD, use_variants=True)
print(f"plain hashing   : {len(plain_found)} pairs → {dict(Counter(plain_found['kind']))}")
print(f"variant hashing : {len(variant_found)} pairs → {dict(Counter(variant_found['kind']))}")
display(variant_found)

# %% [markdown]
# 💡 The two detectors agree on the exact and re-compressed copies and disagree on the rest: mirroring or
# rotating a photo moves its hash by dozens of bits, so plain hashing files it as a new photo, while hashing
# the variants brings it back within a few bits of its source.
#
# **Worked answer (Q12).** Plain hashing catches the byte-identical copies (MD5) and the re-compressed one
# (pHash is built to survive re-compression and resizing). It misses the flipped and the rotated copy, which
# variant hashing catches because one of the six variants of the source now hashes like the copy. The price
# is six hashes per photo instead of one, which is still seconds for thousands of photos.
#
# ### ✍️ Write the flip detector yourself
#
# <table width="100%"><tr><td bgcolor="#FFF4D6"><font color="#7A4A00">
# <b>✏️ Your turn · <code>find_flipped_copies</code></b><br>
# Fill in the function below: given one photo and a dictionary of candidate photos, return the names of the
# candidates that are a flipped or rotated copy of it. You have <code>variants(image)</code>,
# <code>phash(image)</code>, <code>h1 - h2</code> for the distance and <code>HASH_THRESHOLD</code>. The
# <code>use()</code> line checks your answer against the reference on the card and continues with the
# reference either way.
# </font></td></tr></table>

# %%
# ✍️ YOUR CODE
def find_flipped_copies(image, candidates):
    """Return the names of the candidates that are a flipped or rotated copy of `image`.

    image: a PIL image. candidates: dict {name: PIL image} (the image itself is not among them).
    Returns a sorted list of names whose pHash is within HASH_THRESHOLD bits of the pHash of one of
    image's flipped or rotated variants (every variant except "original").
    """
    ...


def _ref_find_flipped_copies(image, candidates):
    flipped_hashes = [phash(im) for name, im in variants(image).items() if name != "original"]
    found = []
    for name, cand in candidates.items():
        h = phash(cand)
        if min(h - fh for fh in flipped_hashes) <= HASH_THRESHOLD:
            found.append(name)
    return sorted(found)


_flip_source = next(k["source"] for k in HUNT5_KEY if k["kind"] == "flipped")
_others = {name: im for name, im in hunt_images.items() if name != _flip_source}
flipped_copies = use(find_flipped_copies, hunt_images[_flip_source], _others,
                     reference=_ref_find_flipped_copies)
print(f"flipped/rotated copies of {_flip_source}: {flipped_copies}")

# %% [markdown]
# ### The hunt's score and the answer key

# %%
# Score the reference detector against the planted key, then show every planted pair side by side.
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
# 💡 Every planted copy is a pair the detector reported, and the pairs it reported are only the planted ones:
# that is the result to insist on with your own data too, a detector that finds everything you planted and
# nothing else. Before you trust it on a real folder, plant a few copies and check it the same way.
#
# ### Broken and odd files
#
# A file can be in the table and still be useless: truncated during download, a grayscale or CMYK scan among
# RGB photos, a PNG with an alpha channel, a thumbnail of 40 pixels or a 50-megapixel panorama. The reader
# will either crash on it or silently feed the model something strange. Section 3 already measured the
# modes and sizes; this cell adds "does it even open?" and lists everything odd in one table.

# %%
# Flags per file: unreadable, non-RGB mode, extreme size. Reading every file fully takes a few seconds.
EXTREME_SIDE_PX = (64, 6000)  # longest side below / above this is "extreme"


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


odd_files = flag_odd_files(image_stats)
print(f"{len(odd_files)} of {len(image_stats)} files carry a flag")
display(odd_files.head(10))

# A broken file for comparison: the first half of a JPEG. `is_readable` catches it; a plain `open` would not.
_half = Path(images["path"].iloc[0]).read_bytes()
_broken = HUNT_DIR / "broken_demo.jpg"
_broken.write_bytes(_half[: len(_half) // 2])
print(f"a truncated copy of {Path(images['path'].iloc[0]).name}: readable = {is_readable(_broken)}")

# %% [markdown]
# 💡 Flags are not verdicts. A grayscale photo may be a perfectly good night-camera frame, and a large
# panorama may be the park's best shot of the whole paddock; the table says "look at me", and a human decides
# whether to convert, resize or drop. Only "unreadable" is dropped without discussion.
#
# ### Suspicious labels: each class's least typical photos
#
# A wrong label cannot be found by hashing. What can be done without a model is to sort each class by how
# *unlike the rest of its class* a photo is, using the colour histograms of section 3, and hand the top of
# each list to a human. Most of them will be fine (an unusual but correct photo); the wrong ones, if any, are
# far more likely to be here than in a random sample.

# %%
# Each photo's distance to its own class's mean colour histogram; the farthest few per class go to a human.
N_ATYPICAL = 2 if SMOKE else 4
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
          lambda r: f"{r['class']} · {r['atypicality']:.1f}", cols=N_ATYPICAL * 2)

# %% [markdown]
# **Q13.** 🟡 Look at the gallery. For each photo, is the label right? Is the animal even visible? Note the
# ones you would send back to the annotator; the next section's worked answer says what you typically find.
#
# 💡 Colour cannot tell a cat from a dog, so this list finds photos that are *unusual*, not photos that are
# *wrong*: a black cat on white snow, a horse photographed at night. That is still useful, because an
# unusual photo is where a tired annotator slips, and because a model will struggle on it either way.
#
# **Worked answer (Q13).** On the Play Dataset most of the least typical photos are correctly labelled but
# odd: close-ups, night shots, an animal that fills a tenth of the frame, a person holding the animal. Real
# label errors in COCO val are rare at the image level (they are more common in the masks, section 9), so an
# honest answer is "nothing to send back, two to keep an eye on". On a crowd-sourced dataset, the same list is
# where the wrong labels live. The 🔴 part below replaces colour with a learned representation, which does
# tell a cat from a dog.
#
# ### 🔴 Optional: model-based search on precomputed embeddings
#
# A hash sees pixels; an **embedding** (the feature vector a pretrained network computes for a photo) sees
# content, so two photos of the same dog from different angles land close together, and a photo whose
# embedding sits far from every other photo of its class is the best candidate for a wrong label. No model
# runs here: the vectors are read from a precomputed file if it is available. When it is not, the cell says so
# and runs the same search on the colour histograms of section 3 as a stand-in, so you can see the mechanics.

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
# **Q14.** 🔴 Will the embedding search's closest pairs be the same pairs the hash found, or different ones?
# And will its least typical photos per class overlap with the colour-based list? Guess, then run.

# %%
# Q14  Nearest pairs and per-class outliers in embedding space (or in the stand-in space).
embeddings, embedding_source = load_embeddings()
print(f"vectors: {embeddings.shape[0]} × {embeddings.shape[1]} from {embedding_source}")
RESULTS["embedding_source"] = embedding_source

_sim = cosine_similarity_matrix(embeddings)
np.fill_diagonal(_sim, -1)
_i, _j = np.unravel_index(np.argsort(-np.triu(_sim, k=1), axis=None)[:N_PAIRS_TO_SHOW], _sim.shape)
_rows = pd.concat([image_stats.iloc[[a, b]] for a, b in zip(_i, _j, strict=True)])
_sim_of = {row_id: _sim[a, b] for a, b in zip(_i, _j, strict=True) for row_id in images["id"].iloc[[a, b]]}
show_rows(_rows, f"closest pairs by {embedding_source}",
          lambda r: f"{r['class']} · sim {_sim_of[r['id']]:.2f}", cols=2)

_class_arr = image_stats["class"].to_numpy()
image_stats["class_similarity"] = [float(_sim[k, (_class_arr == c) & (np.arange(len(_sim)) != k)].mean())
                                   for k, c in enumerate(_class_arr)]
embedding_outliers = (image_stats.sort_values("class_similarity").groupby("class", sort=False)
                      .head(N_ATYPICAL).sort_values(["class", "class_similarity"]))
show_rows(embedding_outliers, f"least typical per class by {embedding_source}",
          lambda r: f"{r['class']} · sim {r['class_similarity']:.2f}", cols=N_ATYPICAL * 2)
_overlap = len(set(embedding_outliers["id"]) & set(least_typical["id"]))
print(f"{_overlap} of {len(embedding_outliers)} embedding outliers are also in the colour-based list")

# %% [markdown]
# 💡 With real embeddings the closest pairs are *semantic* near-duplicates (the same dog, another angle),
# which a hash cannot find, and the outliers are photos whose content does not match the label. With the
# stand-in they are colour matches, which is why the cell says which one you are looking at.
#
# **Worked answer (Q14).** The hash finds copies of the same *picture*; the embedding finds photos of the same
# *thing*. The two lists overlap on exact and near copies and differ everywhere else, and the overlap between
# the two outlier lists is partial at best: colour and content are different notions of "typical". To get
# real embeddings, compute them once on a GPU (any pretrained image model), save `ids` and `embeddings` to
# `data/coco_animals_embeddings.npz`, and this cell picks them up.
#
# ### Clean the dataset
#
# Time to act. The rule applied below is the one most teams use: keep the first photo of every duplicate
# group (exact, near, flipped or rotated), drop every unreadable file, and keep everything else with its
# flag so a human can still decide. The result is `clean_images`, the table the rest of the notebook uses.
#
# <table width="100%"><tr><td bgcolor="#FDECEA"><font color="#8B1A10">
# <b>⚠️ Pitfall · Split after cleaning, not before</b><br>
# If you split first and deduplicate each half on its own, a pair with one photo in train and one in test
# survives, and the test score reports how well the model <i>memorised</i> that photo. Deduplicate the whole
# table, then split (section 6).
# </font></td></tr></table>

# %%
# Duplicate groups on the Play Dataset with variant hashing, then the cleaned table and its summary line.
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
for _, _r in odd_files.iterrows():
    if "unreadable" in _r["flags"]:
        _drop[int(_r["id"])] = "unreadable"

clean_images = images[~images["id"].isin(_drop)].copy()
clean_images["flags"] = clean_images["id"].map(odd_files.set_index("id")["flags"]).fillna("")
removed = pd.DataFrame({"id": list(_drop), "reason": list(_drop.values())})
RESULTS["n_clean"] = int(len(clean_images))
RESULTS["n_removed"] = int(len(removed))
_n_flagged = int((clean_images["flags"] != "").sum())
print(f"Play Dataset: removed {len(removed)}, kept {len(clean_images)} "
      f"(reasons: {dict(Counter(removed['reason']))}; {_n_flagged} kept with a flag)")

# The same rule on the memory card: one file per duplicate group.
_card_drop = set()
for _, _r in variant_found.sort_values(["a", "b"]).iterrows():
    if _r["a"] not in _card_drop:
        _card_drop.add(_r["b"])
print(f"memory card : removed {len(_card_drop)}, kept {len(hunt_files) - len(_card_drop)}")

# %% [markdown]
# 💡 On a curated benchmark the "removed" number is small and the lesson is in the tooling; on the park's own
# uploads the same six cells will be the difference between a test score you can trust and one you cannot.
#
# <table width="100%"><tr><td bgcolor="#EDE7F6"><font color="#311B92">
# <b>🔀 Across tasks · The same errors, three shapes</b><br>
# <ul>
# <li><b>Classification:</b> a duplicate pair across train and test leaks one label; a wrong label is one
# word to fix.</li>
# <li><b>Detection:</b> a duplicated photo duplicates <i>all</i> its boxes, and a flipped copy whose boxes
# were not mirrored (<code>x' = W − x − w</code>) is worse than no copy at all: the boxes now point at
# grass. Label errors become <i>missing</i> or <i>shifted</i> boxes (section 8).</li>
# <li><b>Segmentation:</b> the same for masks, plus a new error class: masks that are the wrong shape
# (swollen, shifted, crude) while the label is right (section 9).</li>
# </ul>
# Deduplicate once, before splitting, for all three; the label checks differ per task.
# </font></td></tr></table>
#
# **Takeaway.** Four cheap checks before any training: MD5 for exact copies, a perceptual hash with a
# threshold you have looked at for near copies, the same hash on flipped and rotated variants, and a
# "does it open, what mode, what size" pass for broken files. Then sort each class by how unusual its photos
# are and let a human read the top of the list. The output is a table, `clean_images`, and one line you can
# put in a report: removed N, kept M. Section 6 splits that table without leaking.
