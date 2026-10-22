# %% [markdown]
# ## 10. Data-centric AI in practice: the Challenge

# %% [markdown]
# <!-- callout: objective | Objective -->
# Find the problems planted in a new batch of park photos, using only the tools of sections 3 to 9.

# %% [markdown]
# <!-- callout: theory | Theory · Data-centric AI -->
# Two ways to make a model better. **Model-centric:** keep the data, change the model. **Data-centric:** keep
# the model, change the data. The park's alert misses a dog in the sheep paddock because three training photos
# of dogs are labelled "sheep": fix the labels, remove the copies, make the boxes and masks agree.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · Why data-centric -->
# The data is where the cheap gains are. Andrew Ng gave the second approach its name in 2021. Northcutt,
# Athalye and Mueller (2021) found label errors in the *test sets* of ten widely used benchmarks: on
# average at least 3.3 % of the labels, and at least 6 % of ImageNet's validation set. That is enough
# to change which model looks best. A model-centric fix would be a bigger network, another loss or
# more epochs.

# %% [markdown]
# ### A new batch from the park
#
# About 150 photos arrive from the park's cameras, to be checked before next week's training. Five steps made
# the batch, and each can go wrong: ten kinds of problem are planted, **at runtime from `SEED`**:
#
# 1. **The camera operator typed one label per photo.**
#    - **wrong label**: a typo, the wrong class.
# 2. **The intern split the photos into train and test.**
#    - **train/test overlap**: a copy whose two files sit in *different* splits (whatever kind of copy it is),
#      so the test score is too good.
# 3. **The keeper counted the animals per photo.**
#    - **missing box**: an animal the keeper counted, but the annotation team did not annotate.
# 4. **The annotation team drew a box and a mask per animal.**
#    - **ugly mask**: a mask that does not match its animal (section 9).
# 5. **Cameras and uploads saved the files.**
#    - **exact duplicate**: the same file twice.
#    - **flipped copy**: a photo mirrored and saved again.
#    - **rotated copy**: a photo turned by 90° and saved again.
#    - **recompressed copy**: a photo resized and saved at a lower JPEG quality.
#    - **unreadable**: a file that does not open.
#    - **grayscale**: a photo saved without colour.

# %%
# @title Plant the new batch (seeded, at runtime; nothing is redistributed) { display-mode: "form" }
CHALLENGE_DIR = DATA_DIR / "cache" / "challenge"
N_BATCH_ORIGINALS = 24 if SMOKE else 140  # distinct photos in the batch (N-014: fewer in smoke mode)
TEST_SHARE = 0.2
RECOMPRESS_SCALE = 0.7
CHALLENGE_TYPES = ["exact duplicate", "flipped copy", "rotated copy", "recompressed copy",
                   "train/test overlap", "unreadable", "grayscale", "wrong label", "missing box", "ugly mask"]
COPY_KIND = {"exact duplicate": "exact", "flipped copy": "flipped", "rotated copy": "rotated",
             "recompressed copy": "recompressed", "train/test overlap": "recompressed"}
N_COPIES = {"exact duplicate": 1 if SMOKE else 2, "flipped copy": 1, "rotated copy": 1,
            "recompressed copy": 1, "train/test overlap": 1}
N_WRONG_LABELS = 2 if SMOKE else 3
UGLY_MASK_KINDS = ["shifted", "swollen"]  # two of section 9's five kinds


def challenge_pool() -> pd.DataFrame:
    """Rows of clean_images that pass every check of sections 3–9 at the default settings: no flag, colourful,
    no crowd region, headcount = boxes, and every judged mask agrees with its box and has a plain shape."""
    judged = animals[animals["mask_area"] >= MIN_JUDGED_AREA]
    bad_mask = judged[(judged["box_iou"] < 0.93) | (judged["fill"] < 0.15) | (judged["fill"] > 0.95)
                      | (judged["pieces"] > 3) | (judged["holes"] > 0) | (judged["solidity"] > 0.97)]
    crowd = masks.loc[masks["iscrowd"] == 1, "image_id"]
    gray = image_stats.loc[image_stats["looks_gray"], "id"]
    n_boxes = boxes.groupby("image_id").size()
    excluded = set(bad_mask["image_id"]) | set(crowd) | set(gray)
    ok = (clean_images["flags"] == "") & ~clean_images["id"].isin(excluded)
    ok &= clean_images["n_animals"] == clean_images["id"].map(n_boxes)
    return clean_images[ok]


def _jpeg(image: Image.Image, quality: int) -> bytes:
    buf = BytesIO()
    image.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _orientation_gap(path, transpose) -> int:
    """Bits between the pHash of a photo and of its flipped or rotated version (small if symmetric)."""
    thumb = load_thumb(path, THUMB_FOR_HASH)
    return int(phash(thumb) - phash(thumb.transpose(transpose)))


def _move_annotation(rle: dict, bbox, kind: str, width: int, height: int,
                     new_size: tuple) -> tuple[dict, list]:
    """An annotation (RLE mask + box) carried over to an exact, flipped, rotated or resized copy."""
    x, y, w, h = bbox
    if kind == "exact":
        return rle, [x, y, w, h]
    mask = decode(rle)
    if kind == "flipped":
        return encode(mask[:, ::-1]), [width - x - w, y, w, h]
    if kind == "rotated":  # 90° counter-clockwise, like PIL's ROTATE_90
        return encode(np.rot90(mask)), [y, width - x - w, h, w]
    sx, sy = new_size[0] / width, new_size[1] / height  # resized: x and y scale separately (section 3)
    resized = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
    return encode(resized), [x * sx, y * sy, w * sx, h * sy]


def plant_challenge(dest: Path, n_originals: int, seed: int):
    """Write the batch: n_originals photos of challenge_pool() plus planted copies and problems.

    Returns (files, batch, batch_animals, key). files: {name: path}. batch: one row per file (file, class,
    split, n_animals, width, height, path). batch_animals: one row per annotated animal (id, file, class,
    bbox, rle, mask_area). key: one dict per planted problem (type, files, class, detail); for a copy,
    `files` holds the copy and its source. Every problem gets its own photo. Deterministic for a given seed.
    """
    rng = random.Random(seed)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    pool = challenge_pool()
    chosen = pool.sample(n=min(n_originals, len(pool)), random_state=seed).set_index("id").sort_index()
    ids = list(chosen.index)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_test = max(2, round(TEST_SHARE * len(ids)))
    split = {i: "test" if k < n_test else "train" for k, i in enumerate(shuffled)}
    label = dict(chosen["class"])
    anns = animals[animals["image_id"].isin(ids)].set_index("id")
    order = ids[:]
    rng.shuffle(order)
    used: set = set()

    def take(ok=lambda i: True) -> int:
        """The next unused photo in the seeded order that satisfies ok (the next unused one if none does)."""
        free = [i for i in order if i not in used]
        i = next((i for i in free if ok(i)), free[0])
        used.add(i)
        return i

    def resize_safe(i) -> bool:  # masks that stay judged after resizing keep agreeing with their box
        a = anns[anns["image_id"] == i]
        return bool((a.loc[a["mask_area"] * RECOMPRESS_SCALE**2 >= MIN_JUDGED_AREA, "box_iou"] >= 0.95).all())

    in_place = []  # (type, photo id, detail)
    # A missing box: one animal of a photo with several loses its annotation.
    i = take(lambda i: chosen.at[i, "n_animals"] >= 2)
    dropped = rng.choice(sorted(anns.index[anns["image_id"] == i]))
    in_place.append(("missing box", i, f"the keeper counted {chosen.at[i, 'n_animals']}, "
                                       f"the team annotated {chosen.at[i, 'n_animals'] - 1}"))
    # Ugly masks: section 9's planting on the photos still free, two of its five kinds kept.
    free_anns = anns[~anns["image_id"].isin(used)].reset_index()
    spoiled, spoil_key = plant_hunt9(free_anns, chosen.reset_index(), seed)
    spoiled_rle = spoiled.set_index("id")["rle"]
    new_rle = {}
    for _, k in spoil_key[spoil_key["kind"].isin(UGLY_MASK_KINDS)].iterrows():
        new_rle[k["ann_id"]] = spoiled_rle[k["ann_id"]]
        used.add(k["image_id"])
        in_place.append(("ugly mask", k["image_id"], f"{k['kind']} mask of a {k['class']} ({k['how']})"))
    for _ in range(N_WRONG_LABELS):
        i = take()
        label[i] = rng.choice([c for c in CLASSES if c != chosen.at[i, "class"]])
        in_place.append(("wrong label", i, f"labelled {label[i]}, the photo shows a {chosen.at[i, 'class']}"))
    gray_id, unreadable_id = take(), take()
    in_place += [("grayscale", gray_id, "saved without colour (mode L)"),
                 ("unreadable", unreadable_id, "the second half of the file is missing")]
    # Flipped and rotated copies come from photos that do not look the same after the turn (no symmetry).
    flip, turn = Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.ROTATE_90
    condition = {"flipped copy": lambda i: _orientation_gap(chosen.at[i, "path"], flip) >= 20,
                 "rotated copy": lambda i: _orientation_gap(chosen.at[i, "path"], turn) >= 20,
                 "recompressed copy": resize_safe,
                 "train/test overlap": lambda i: split[i] == "train" and resize_safe(i)}
    copies = [(t, take(condition.get(t, lambda i: True))) for t, n in N_COPIES.items() for _ in range(n)]

    # Every file of the batch: the originals (two of them spoiled in place), then the copies.
    entries = []
    for i in ids:
        path, size = Path(chosen.at[i, "path"]), (int(chosen.at[i, "width"]), int(chosen.at[i, "height"]))
        data = path.read_bytes()
        if i == gray_id:
            with Image.open(path) as im:
                data = _jpeg(im.convert("L"), 92)
        elif i == unreadable_id:
            data = data[: len(data) // 2]
        entries.append({"data": data, "source": i, "kind": None, "split": split[i], "size": size})
    for t, i in copies:
        path, (w, h) = Path(chosen.at[i, "path"]), (int(chosen.at[i, "width"]), int(chosen.at[i, "height"]))
        kind = COPY_KIND[t]
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        if kind == "exact":
            data, size = path.read_bytes(), (w, h)
        elif kind == "flipped":
            data, size = _jpeg(ImageOps.mirror(rgb), 92), (w, h)
        elif kind == "rotated":
            data, size = _jpeg(rgb.transpose(Image.Transpose.ROTATE_90), 92), (h, w)
        else:
            size = (max(8, round(w * RECOMPRESS_SCALE)), max(8, round(h * RECOMPRESS_SCALE)))
            data = _jpeg(rgb.resize(size), 55)
        other = "test" if split[i] == "train" else "train"
        entries.append({"data": data, "source": i, "kind": kind, "type": t, "size": size,
                        "split": other if t == "train/test overlap" else split[i]})
    slots = list(range(len(entries)))
    rng.shuffle(slots)
    rows, ann_rows = [], []
    for slot, e in zip(slots, entries, strict=True):
        e["name"] = f"park_{slot:03d}.jpg"
        (dest / e["name"]).write_bytes(e["data"])
        i = e["source"]
        rows.append({"file": e["name"], "class": label[i], "split": e["split"],
                     "n_animals": int(chosen.at[i, "n_animals"]), "width": e["size"][0],
                     "height": e["size"][1], "path": str(dest / e["name"])})
        for ann_id, a in anns[anns["image_id"] == i].iterrows():
            if ann_id == dropped:
                continue
            rle, bbox = _move_annotation(new_rle.get(ann_id, a["rle"]), a["bbox"], e["kind"] or "exact",
                                         int(chosen.at[i, "width"]), int(chosen.at[i, "height"]), e["size"])
            ann_rows.append({"file": e["name"], "class": a["class"],
                             "bbox": [round(float(v), 2) for v in bbox],
                             "rle": rle, "mask_area": int(mask_utils.area(rle))})

    name_of = {(e["source"], e["kind"]): e["name"] for e in entries}
    key = [{"type": t, "files": [name_of[(i, None)]], "class": chosen.at[i, "class"], "detail": d}
           for t, i, d in in_place]
    for t, i in copies:
        copy, source = name_of[(i, COPY_KIND[t])], name_of[(i, None)]
        where = next(e["split"] for e in entries if e["name"] == copy)
        key.append({"type": t, "files": [copy, source], "class": chosen.at[i, "class"],
                    "detail": f"{COPY_KIND[t]} copy of {source} ({split[i]}), the copy is in {where}"})
    key.sort(key=lambda k: (CHALLENGE_TYPES.index(k["type"]), k["files"][0]))
    table = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    annotated = pd.DataFrame(ann_rows).sort_values("file").reset_index(drop=True)
    annotated.insert(0, "id", range(1, len(annotated) + 1))
    files = {r["file"]: Path(r["path"]) for _, r in table.iterrows()}
    return files, table, annotated, key


def show_batch(file_names, title: str, caption=None, cols: int = 5, annotations: bool = False) -> None:
    """Gallery of batch files, each framed in the colour of its label (the `class` column of `batch`).

    annotations=True paints the annotation team's masks in their class colour and draws their boxes dashed.
    A file that does not open shows as an empty frame. `caption(name) -> str` writes the title of each photo.
    """
    names = list(file_names)
    if not names:
        print(f"{title}: no files to show.")
        return
    by_file = batch.set_index("file")
    cols = min(cols, len(names))
    n_rows = -(-len(names) // cols)
    fig, axes = plt.subplots(n_rows, cols, figsize=(2.9 * cols, 3.0 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, name in zip(axes.ravel(), names, strict=False):
        row = by_file.loc[name]
        ax.axis("on")
        if is_readable(row["path"]):
            thumb = load_thumb(row["path"], 256)
            ax.imshow(thumb)
            if annotations:
                scale = thumb.width / row["width"]
                overlay = np.zeros((thumb.height, thumb.width, 4))
                own = batch_animals[batch_animals["file"] == name]
                for _, a in own.iterrows():
                    small = cv2.resize(decode(a["rle"]), thumb.size, interpolation=cv2.INTER_NEAREST) > 0
                    overlay[small] = (*mpl.colors.to_rgb(CLASS_COLORS[a["class"]]), 0.55)
                ax.imshow(overlay)
                for _, a in own.iterrows():
                    x, y, w, h = a["bbox"]
                    ax.add_patch(mpl.patches.Rectangle((x * scale, y * scale), w * scale, h * scale,
                                                       fill=False, lw=1.5, ls="--",
                                                       edgecolor=CLASS_COLORS[a["class"]]))
        else:
            ax.text(0.5, 0.5, "cannot be opened", ha="center", va="center", transform=ax.transAxes, color=INK)
        frame(ax, CLASS_COLORS[row["class"]])
        ax.set_title(caption(name) if caption else f"{name}\n{row['class']} · {row['split']}", fontsize=8)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    display(fig)
    plt.close(fig)


# %% [markdown]
# The next cell writes the batch (a few seconds) and loads three objects: `batch_files` (name → path), `batch`
# (one row per file: label as typed, split, the keeper's headcount `n_animals`, size) and `batch_animals`
# (one row per annotated animal: class, box, mask as RLE).

# %%
# @title Write the batch, then show its tables and a first look at a dozen photos { display-mode: "form" }
batch_files, batch, batch_animals, _CHALLENGE_KEY = plant_challenge(CHALLENGE_DIR, N_BATCH_ORIGINALS, SEED)
RESULTS["challenge_types"] = len({k["type"] for k in _CHALLENGE_KEY})
print(f"{len(batch_files)} files in {CHALLENGE_DIR}/ · {(batch['split'] == 'test').sum()} in test · "
      f"{len(batch_animals)} annotated animals")
print(f"planted: {len(_CHALLENGE_KEY)} problems of {RESULTS['challenge_types']} kinds")
display(batch.drop(columns="path").head())
show_batch(batch["file"].head(12), "the first dozen files of the batch (frame = the label as typed)", cols=6)

# %% [markdown]
# Nothing in the tables says "problem": that is what a real delivery looks like.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · Your tools, and where they came from -->
# - `exact_duplicate_groups(paths)` and `find_copies(files, threshold, use_variants=True)` (section 5): pairs
#   of files that are copies, and what kind of copy;
# - `is_readable(path)` (helper defined in section 5's toolbox): does the file open at all? (`False` for a
#   file PIL cannot decode);
# - `measure_image(path, thumb)[0]` (section 3): `mode` and `colourfulness` of a photo (0 for grayscale);
# - `batch["n_animals"]` against the number of rows per file in `batch_animals` (section 8);
# - `mask_shape(decode(rle), bbox)` (section 9): `box_iou`, `fill`, `pieces`, `holes`, `solidity`;
# - `show_batch(names, title, annotations=True)`: look at any files, with their boxes and masks.
#
# The planted problems sit in photos that passed every check of sections 5 to 9 at the default settings.

# %% [markdown]
# Two collapsed cells follow: the scoring helpers, and a reference solution that prints only its score.

# %%
# @title Score helpers: check_findings, reveal_answers { display-mode: "form" }
def check_findings(findings, verbose: bool = True) -> dict:
    """Score a list of (file, kind) pairs against the planted problems: found k of n per kind.

    A planted problem is found when one pair names its kind and one of its files (for a copy, the copy or its
    source). Returns {"found", "planted", "per_type": {kind: [found, planted]}, "unmatched"}.
    """
    if not isinstance(findings, (list, tuple)):
        raise SystemExit("findings must be a list of (file, kind) pairs, "
                         "e.g. [('park_999.jpg', 'grayscale')]")
    pairs, skipped = set(), []
    for item in findings:
        if (isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(v, str) for v in item)
                and item[1].strip().lower() in CHALLENGE_TYPES):
            pairs.add((Path(item[0].strip()).name, item[1].strip().lower()))
        else:
            skipped.append(item)
    hit = [any(kind == k["type"] and name in k["files"] for name, kind in pairs) for k in _CHALLENGE_KEY]
    matched = {(name, kind) for name, kind in pairs
               if any(kind == k["type"] and name in k["files"] for k in _CHALLENGE_KEY)}
    per_type = {t: [sum(h for h, k in zip(hit, _CHALLENGE_KEY, strict=True) if k["type"] == t),
                    sum(k["type"] == t for k in _CHALLENGE_KEY)] for t in CHALLENGE_TYPES}
    score = {"found": int(sum(hit)), "planted": len(_CHALLENGE_KEY), "per_type": per_type,
             "unmatched": len(pairs) - len(matched)}
    if verbose:
        for t, (k, n) in per_type.items():
            print(f"  {'✅' if k == n else '⬜'} {t:19s} found {k} of {n}")
        print(f"found {score['found']} of {score['planted']} planted problems · {score['unmatched']} of your "
              "findings match no planted problem")
        if skipped:
            print(f"⚠️ skipped {len(skipped)} entries that are not (file, kind) pairs with a kind from "
                  f"CHALLENGE_TYPES, e.g. {skipped[0]!r}")
    return score


def reveal_answers() -> None:
    """Print the answer key, one line per planted problem, and show the files with each problem."""
    for k in _CHALLENGE_KEY:
        print(f"{k['type']:19s} {' & '.join(k['files'])}: {k['detail']}")
    type_of = {k["files"][0]: k["type"] for k in _CHALLENGE_KEY}
    in_annotations = {"wrong label", "missing box", "ugly mask"}
    file_problems = [k["files"][0] for k in _CHALLENGE_KEY if k["type"] not in in_annotations]
    show_batch(file_problems, "problems in the files (copies, unreadable, grayscale)",
               caption=lambda n: f"{type_of[n]}\n{n}", cols=6)
    annotation_problems = [k["files"][0] for k in _CHALLENGE_KEY if k["type"] in in_annotations]
    show_batch(annotation_problems, "problems in the labels and annotations (frame = the label as typed)",
               caption=lambda n: f"{type_of[n]}\n{n} · labelled {batch.set_index('file').at[n, 'class']}",
               cols=6, annotations=True)


# %%
# @title Reference solution (collapsed) — open its code only if you are stuck { display-mode: "form" }
_REF_HASH_THRESHOLD = 8  # section 5's default
_REF_IOU_MIN = 0.9  # section 9's default
_TYPE_OF_COPY = {"exact": "exact duplicate", "flipped": "flipped copy", "rotated": "rotated copy",
                 "recompressed": "recompressed copy"}


def _reference_findings() -> list[tuple[str, str]]:
    """One check per kind, with the tools of sections 3–9; returns (file, kind) pairs."""
    found = []
    readable = {n: p for n, p in batch_files.items() if is_readable(p)}
    found += [(n, "unreadable") for n in batch_files if n not in readable]
    split_of = batch.set_index("file")["split"]
    for _, r in find_copies(readable, _REF_HASH_THRESHOLD, use_variants=True).iterrows():
        across = split_of[r["a"]] != split_of[r["b"]]
        found.append((r["b"], "train/test overlap" if across else _TYPE_OF_COPY[r["kind"]]))
    for n, p in readable.items():
        stats = measure_image(p, 64)[0]
        if stats["mode"] != "RGB" or stats["colourfulness"] < 2.0:
            found.append((n, "grayscale"))
    per_file = batch_animals.groupby("file")
    annotated_class = per_file["class"].agg(lambda s: s.mode().iloc[0])
    n_annotated = per_file.size()
    for _, r in batch.iterrows():
        if r["file"] in annotated_class.index and annotated_class[r["file"]] != r["class"]:
            found.append((r["file"], "wrong label"))
        if int(n_annotated.get(r["file"], 0)) != r["n_animals"]:
            found.append((r["file"], "missing box"))
    for _, a in batch_animals[batch_animals["mask_area"] >= MIN_JUDGED_AREA].iterrows():
        if mask_shape(decode(a["rle"]), a["bbox"])["box_iou"] < _REF_IOU_MIN:
            found.append((a["file"], "ugly mask"))
    return sorted(set(found))


_reference = _reference_findings()
_reference_score = check_findings(_reference, verbose=False)
RESULTS["challenge_reference"] = {"found": _reference_score["found"], "planted": _reference_score["planted"],
                                  "unmatched": _reference_score["unmatched"]}
print(f"reference solution: found {_reference_score['found']} of {_reference_score['planted']} planted "
      f"problems with {len(_reference)} findings ({_reference_score['unmatched']} false alarms)")

# %% [markdown]
# <!-- callout: questions | Questions -->
# The ten headers in the next cell are your plan: one per kind, with its tool and what it reads.

# %% [markdown]
# **Q26.** 🟡 The hunt. Six checks are written for you: kinds 1 to 5 and 7. Write the other four under their
# headers: the kinds where two sources disagree. Before you do, predict: which kinds could you never spot in
# the photo itself, however long you looked? Then collect what you find in `findings` and run the score.
# Rerun both as often as you like.

# %%
# Q26  Your hunt: one header per kind = the plan. Kinds 1-5 and 7 are written; write 6, 8, 9 and 10.

# 1. unreadable — tool: is_readable — reads: the files (written for you)
cant_open = sorted(name for name, path in batch_files.items() if not is_readable(path))
print("unreadable:", cant_open)

# 2. exact duplicate — tool: find_copies, kind "exact" — reads: the files (written for you)
opens = {n: p for n, p in batch_files.items() if n not in cant_open}
pairs = find_copies(opens, HASH_THRESHOLD, use_variants=True)  # its "kind" column names the sort of copy


def of_kind(kind):
    return pairs.loc[pairs["kind"] == kind, ["a", "b"]].values.tolist()


print("exact duplicate:", of_kind("exact"))

# 3. flipped copy — tool: find_copies, kind "flipped" — reads: the files (written for you)
print("flipped copy:", of_kind("flipped"))

# 4. rotated copy — tool: find_copies, kind "rotated" — reads: the files (written for you)
print("rotated copy:", of_kind("rotated"))

# 5. recompressed copy — tool: find_copies, kind "recompressed" — reads: the files (written for you)
print("recompressed copy:", of_kind("recompressed"))

# 6. train/test overlap — tool: the same pairs, then batch["split"] of both files — reads: files + batch


# 7. grayscale — tool: measure_image(path, 64)[0]["mode"] and ["colourfulness"] — reads: the files (written)
looks_gray = [n for n, p in opens.items()
              if (s := measure_image(p, 64)[0])["mode"] != "RGB" or s["colourfulness"] < 2.0]
print("grayscale:", looks_gray)

# 8. wrong label — tool: batch["class"] vs the class of the file's animals — reads: batch + batch_animals


# 9. missing box — tool: batch["n_animals"] vs batch_animals.groupby("file").size() — reads: both tables


# 10. ugly mask — tool: mask_shape(decode(a["rle"]), a["bbox"])["box_iou"] per row a — reads: batch_animals


# %% [markdown]
# <!-- callout: your-turn | Your turn · Your findings -->
# Write one `(file, kind)` pair per problem into `findings`, with the kind spelled as in `CHALLENGE_TYPES`.
#
# - For a copy, name either of its two files.
# - Build the list from your variables, e.g. `[(f, "unreadable") for f in cant_open] + [...]`.
# - `check_findings` prints "found k of n" per kind: n problems of that kind were planted, and k of them are
#   named by one of your pairs. It also counts the pairs that match nothing; it never says which ones.

# %%
# ✏️ YOUR TURN
# One (file, kind) pair per problem; kind is one of CHALLENGE_TYPES. For a copy, either of its files counts.
findings = [
    # ("park_999.jpg", "grayscale"),  (the format; there is no park_999.jpg)
]

challenge_score = check_findings(findings)

# %% [markdown]
# 💡 Read the ⬜ lines as a to-do list: each is a kind you have not caught yet, and its header in the Q26
# cell names a tool. The reference solution above found every problem with this notebook's tools alone.

# %% [markdown]
# ### The answer key
#
# Open the worked answer, then run the next cell to see the files of your batch.

# %% [markdown]
# <details><summary><b>Worked answer (Q26): which check finds which kind</b></summary>
#
# - **Unreadable**: `is_readable(path)` on every file. Run it first, because every other check that opens the
#   photo stumbles on this file.
# - **Exact, flipped, rotated and recompressed copies**: `find_copies(..., use_variants=True)` on the files
#   that open. MD5 finds the exact ones; the perceptual hash of each photo's six variants finds the
#   recompressed, flipped and rotated ones. Plain hashing would miss two of the four.
# - **Train/test overlap**: the same pairs, then `batch["split"]` of both files. A pair across the splits is
#   leakage whatever kind of copy it is (section 6). In the planted batch it is a recompressed copy, so it
#   hides among the near-duplicates.
# - **Grayscale**: `measure_image` gives `mode` "L" and a `colourfulness` of 0 (section 3).
# - **Wrong label**: no tool looks at the pixels and says "this is a dog". But the annotation team's animals
#   carry their own class, so a file whose label disagrees with the class of its animals is a wrong label
#   (or a wrong annotation; a human decides).
# - **Missing box**: the keeper's headcount `n_animals` vs the number of animals annotated per file
#   (section 8).
# - **Ugly mask**: `mask_shape(...)["box_iou"]` below 0.9 (section 9). The two planted kinds, a shifted and a
#   swollen mask, both stop matching their box.
#
# The four kinds that cannot be seen in the pixels are train/test overlap, wrong label, missing box and ugly
# mask. Each is a disagreement between two sources: two splits, a label and its annotation, a headcount and
# the boxes, a mask and its box. That is the data-centric habit: get a second source and compare.
#
# **A finding that matches nothing** is the wrong kind for the right file (a copy across the splits is a
# *train/test overlap*, not a recompressed copy), or a false alarm.
#
# **Looking or comparing?** Looking works for the copies and the grayscale photo, if you happen to see both
# files. The wrong labels, the missing box and the ugly masks need two sources side by side: nobody spots
# them by scrolling through 150 photos.
# </details>

# %%
# @title Show the answer key: every planted problem, then its files { display-mode: "form" }
reveal_answers()

# %% [markdown]
# 💡 Which problems did you find by looking, and which only by comparing?

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **When you see** a new batch → **do** run the same short list every time: does it open, is it a copy
#   (also across the splits), does its mode say colour, does the label agree with the annotation, does the
#   headcount agree with the boxes, does the mask agree with its box.
# - **When you see** a problem that no pixel shows (wrong label, missing box, ugly mask, overlap) → **do** get
#   a second source and compare the two.
# - **When you remove files** → **do** write down "removed N, kept M" and why.
# - **When you finish here** → **do** run the same list on your own dataset in section 11.
