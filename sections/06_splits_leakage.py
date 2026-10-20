# %% [markdown]
# ## 6. Splits & leakage
#
# **Objective:** choose a split that does not leak, and understand what k-fold, stratified, leave-one-out
# and leave-one-group-out cross-validation each buy you.
#
# Section 5 ended with `clean_images`, a table without duplicates. The next decision is the one that decides
# whether next week's test score means anything: **which photos does the model never see?** The held-out
# part has to look like the photos the park's cameras will take *after* training, not like the ones they
# already took. If a test photo has a near-twin in training, the score reports memory, not recognition.
#
# <table width="100%"><tr><td bgcolor="#E3F2FD"><font color="#0D47A1">
# <b>📘 Theory · Data leakage</b><br>
# <b>Leakage</b> is any way information about the test photos reaches the model during training. The
# obvious route is a duplicate in both halves (section 5). The quiet route is <b>a photo that is not a copy
# but comes from the same moment</b>: two frames of one burst, two shots of the same animal in front of the
# same bush, two photos from one camera that always sees the same tree. The model learns the bush, the
# score goes up, and the first camera in a new place breaks it. The rule: <b>split by whatever will be new
# at deployment</b> (a camera, a day, a patient, a farm), not by photo.
# </font></td></tr></table>
#
# ### Random vs stratified split on the Play Dataset
#
# The simplest split shuffles the photos and cuts them 80 / 20. A **stratified** split cuts *each class*
# 80 / 20, so both halves keep the class shares of the whole table. The code below does both on
# `clean_images` and compares the class shares of the test halves.
#
# <table width="100%"><tr><td bgcolor="#E0F7F4"><font color="#00564D">
# <b>❓ Questions</b><br>
# Sheep is the smallest class, with a fifth of the photos of cat. Think of a random 20 % test half.
# </font></td></tr></table>
#
# **Q15.** 🟡 In a random split, which class share will move the most between train and test: the largest
# class or the smallest? Does the stratified split fix that completely, or only mostly? Decide, then run.

# %%
# Random split vs stratified split of clean_images (80 / 20), then the class share of each test half.
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    StratifiedKFold,
    train_test_split,
)

TEST_SIZE = 0.2
_ids = clean_images["id"].to_numpy()
_y = clean_images["class"].to_numpy()
_tr_rand, _te_rand = train_test_split(_ids, test_size=TEST_SIZE, random_state=SEED)
_tr_strat, _te_strat = train_test_split(_ids, test_size=TEST_SIZE, random_state=SEED, stratify=_y)


def class_shares(ids) -> pd.Series:
    """Share of each class among the given image ids of clean_images (in CLASSES order, sums to 1)."""
    counts = clean_images.set_index("id").loc[list(ids), "class"].value_counts()
    return counts.reindex(CLASSES).fillna(0) / max(len(ids), 1)


shares = pd.DataFrame(
    {
        "whole table": class_shares(_ids),
        "random · test": class_shares(_te_rand),
        "stratified · test": class_shares(_te_strat),
    }
)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, col in zip(axes, ["random · test", "stratified · test"], strict=True):
    x = np.arange(len(CLASSES))
    ax.bar(x - 0.2, shares["whole table"], width=0.4, color="#DDE5E7", label="whole table")
    ax.bar(x + 0.2, shares[col], width=0.4, color=[CLASS_COLORS[c] for c in CLASSES], label=col)
    ax.set_xticks(x, CLASSES)
    ax.set_title(f"{col} ({len(_te_rand)} photos)")
    ax.set_ylabel("share of photos")
    ax.legend(frameon=False)
plt.tight_layout()
plt.show()
_drift = (shares.sub(shares["whole table"], axis=0).abs() * 100).round(1)
print("largest class-share drift from the whole table, in percentage points:")
print(f"  random split: {_drift['random · test'].max():.1f} ({_drift['random · test'].idxmax()})   "
      f"stratified split: {_drift['stratified · test'].max():.1f} ({_drift['stratified · test'].idxmax()})")
RESULTS["split_drift_random_pp"] = float(_drift["random · test"].max())
RESULTS["split_drift_stratified_pp"] = float(_drift["stratified · test"].max())

# %% [markdown]
# 💡 The grey bars are the whole table and the coloured bars the test half: with a random cut the small
# classes wobble the most in relative terms, and the stratified cut sits on top of the grey bars up to the
# rounding that whole photos force.
#
# **Worked answer (Q15).** In absolute percentage points the largest classes can move the most, but what
# hurts is the *relative* move: a class of 50 photos that loses or gains five in the test half changes its
# share by 10 %, and with a rare class like the paddock's dog the test half may end up with almost none of
# the one class the alert is about. Stratification removes that wobble up to rounding (you cannot put 10.6
# sheep in the test half), so the shares match to within one photo per class. It does nothing about the
# content of the photos, which is the problem the rest of this section is about.
#
# ### Cross-validation: four ways to reuse one dataset
#
# One split gives one score, and with a few hundred photos that score moves by several points when you
# change the seed. **Cross-validation** rotates the test half around the table so every photo is tested
# once, and reports the mean (and spread) over the folds. Four schemes, each with its reason:
#
# <table width="100%"><tr><td bgcolor="#E3F2FD"><font color="#0D47A1">
# <b>📘 Theory · k-fold, stratified k-fold, leave-one-out, leave-one-group-out</b><br>
# <ul>
# <li><b>k-fold:</b> cut the table into k pieces, test on each piece once, train on the other k−1.
# Five or ten folds is the usual choice.</li>
# <li><b>stratified k-fold:</b> the same, but every piece keeps the class shares. Use it whenever classes are
# imbalanced, which is always.</li>
# <li><b>leave-one-out (LOO):</b> k equals the number of photos. Every photo is tested once against a model
# trained on all the others. Useful when the dataset is <b>very small</b> (a few dozen photos) and every
# training example counts; otherwise it costs n trainings, and the n scores are each 0 or 1, so the mean is
# noisy.</li>
# <li><b>leave-one-group-out (LOGO):</b> the folds are <b>groups</b> that must stay together (one camera, one
# day, one patient). Each fold holds out one whole group, which is the only scheme here that tests what
# happens at a <i>new</i> camera.</li>
# </ul>
# </font></td></tr></table>
#
# The cell below runs the first three on the Play Dataset and reports, for each, how many folds there are,
# how big a test fold is, and the smallest class share any test fold has (0 means some fold tests a class
# it never sees). LOO runs on a small slice of the table, because that is where it belongs.
#
# **Q16.** 🟡 Before running: how many folds does leave-one-out make on the Play Dataset, and what is the
# smallest class share a LOO test fold can have?

# %%
# k-fold, stratified k-fold and leave-one-out on the Play Dataset: folds, test size, smallest class share.
K_FOLDS = 5
LOO_N = 12 if SMOKE else 40  # leave-one-out on a slice: a "very small dataset" (N-014: same code, fewer rows)


def summarise_folds(name: str, splitter, X, y, groups=None) -> dict:
    """One row per CV scheme: folds, test-fold size and the smallest class share across the test folds."""
    sizes, min_share = [], 1.0
    for _, test_idx in splitter.split(X, y, groups):
        sizes.append(len(test_idx))
        counts = pd.Series(y[test_idx]).value_counts(normalize=True).reindex(sorted(set(y))).fillna(0)
        min_share = min(min_share, float(counts.min()))
    return {"scheme": name, "folds": len(sizes), "test fold size": f"{min(sizes)}–{max(sizes)}",
            "smallest class share in a test fold": round(min_share, 3)}


_loo_ids = _ids[:LOO_N]
cv_rows = [
    summarise_folds("kfold", KFold(K_FOLDS, shuffle=True, random_state=SEED), _ids, _y),
    summarise_folds("stratified_kfold", StratifiedKFold(K_FOLDS, shuffle=True, random_state=SEED), _ids, _y),
    summarise_folds("loo", LeaveOneOut(), _loo_ids, _y[:LOO_N]),
]
cv_schemes = [r["scheme"] for r in cv_rows]
display(pd.DataFrame(cv_rows).set_index("scheme"))
print(f"(loo ran on the first {LOO_N} photos: {LOO_N} folds, {LOO_N} trainings for one score)")

# %% [markdown]
# 💡 Plain k-fold already drops the smallest class's share in some fold below its overall share; stratified
# k-fold holds it steady; LOO's test folds are one photo each, so the share column reads 0 whatever you do.
#
# **Worked answer (Q16).** LOO makes one fold per photo, so as many folds as rows, and it would cost that many
# trainings next week. Each test fold is a single photo, so its smallest class share is always 0: five of
# six classes are absent from every fold. That is harmless for the mean score but shows why LOO tells you
# nothing per class. The fourth scheme, leave-one-group-out, needs groups, and the Play Dataset has none:
# COCO photos come from thousands of unrelated Flickr accounts. The park's next cameras do have groups.
#
# ### The woodland trail cameras
#
# The park has just hung **trail cameras in its woodland reserve**. A trail camera fires on motion and takes
# a **burst** of three frames a second apart, every night, from the same tree, for years. We use a slice of
# the Caltech Camera Traps dataset (Beery et al., 2018) as a stand-in: eight camera locations, five
# animals (opossum, raccoon, coyote, bobcat, rabbit), whole bursts kept together. `data/cct20_slice.csv`
# lists the photos; the next cell downloads them from LILA's public bucket and shrinks them to 512 px, so
# **the first run takes a minute** and afterwards the cell is instant. The five woodland classes borrow the
# park's six colours so every plot keeps one colour per class.

# %%
# @title Trail-camera helpers (manifest, download + resize to 512 px, burst gallery) { display-mode: "form" }
CCT_DIR = DATA_DIR / "cache" / "cct20"
CCT_SIDE = 512
CCT_CLASSES = ["opossum", "raccoon", "coyote", "bobcat", "rabbit"]
CCT_COLORS = dict(zip(CCT_CLASSES, [CLASS_COLORS[c] for c in CLASSES], strict=False))
SMOKE_CCT_LOCATIONS = 3
SMOKE_CCT_IMAGES = 30


def select_cct_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The slice to use: everything, or in smoke mode whole bursts from 3 locations up to 30 photos."""
    if not SMOKE:
        return table
    keep = []
    locations = sorted(table["location"].unique())[:SMOKE_CCT_LOCATIONS]
    per_location = SMOKE_CCT_IMAGES // SMOKE_CCT_LOCATIONS
    for loc in locations:
        n = 0
        at_loc = table[table["location"] == loc].sort_values(["seq_id", "frame_num"])
        for _, burst in at_loc.groupby("seq_id", sort=True):
            if n + len(burst) > per_location:
                continue
            keep.append(burst)
            n += len(burst)
    return pd.concat(keep).reset_index(drop=True)


def fetch_cct_image(row: dict) -> tuple[str, Path | None]:
    """Download one trail-camera photo and store a 512 px copy; returns (file_name, path or None)."""
    dest = CCT_DIR / row["file_name"]
    if dest.exists() and dest.stat().st_size > 0:
        return row["file_name"], dest
    raw = download_file(row["url"], CCT_DIR / "raw" / row["file_name"])
    if raw is None:
        return row["file_name"], None
    with Image.open(raw) as im:
        im.draft("RGB", (CCT_SIDE, CCT_SIDE))
        small = im.convert("RGB")
        small.thumbnail((CCT_SIDE, CCT_SIDE))
        small.save(dest, "JPEG", quality=90)
    raw.unlink(missing_ok=True)
    return row["file_name"], dest


def download_cct(table: pd.DataFrame, workers: int = 16) -> pd.DataFrame:
    """Fetch every photo of `table` (parallel, cached), return the rows whose photo is on disk with a path."""
    CCT_DIR.mkdir(parents=True, exist_ok=True)
    paths, done = {}, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, path in pool.map(fetch_cct_image, table.to_dict("records")):
            done += 1
            if path is not None:
                paths[name] = str(path)
            print(f"\r  trail-camera photos ready: {done}/{len(table)}", end="", flush=True)
    print()
    out = table[table["file_name"].isin(paths)].copy()
    out["path"] = out["file_name"].map(paths)
    return out.sort_values(["location", "seq_id", "frame_num"]).reset_index(drop=True)


def show_bursts(table: pd.DataFrame, n_bursts: int, size: int = 256) -> None:
    """One row per burst: its frames side by side, framed in the class colour, titled with the location."""
    bursts = [b for _, b in table.groupby("seq_id", sort=True) if len(b) > 1][:n_bursts]
    cols = max(len(b) for b in bursts)
    fig, axes = plt.subplots(len(bursts), cols, figsize=(2.6 * cols, 2.1 * len(bursts)), squeeze=False)
    for r, burst in enumerate(bursts):
        for c in range(cols):
            ax = axes[r, c]
            if c < len(burst):
                row = burst.iloc[c]
                ax.imshow(load_thumb(row["path"], size))
                frame(ax, CCT_COLORS[row["label"]])
                ax.set_title(f"camera {row['location']} · {row['label']} · frame {row['frame_num']}",
                             fontsize=8)
            else:
                ax.axis("off")
    plt.tight_layout()
    plt.show()


# %%
# Load the slice, download and shrink the photos (cached), then look at a few bursts.
cct_all = pd.read_csv(fetch_manifest("cct20_slice.csv"))
cct = download_cct(select_cct_rows(cct_all))
if len(cct) < len(select_cct_rows(cct_all)):
    print(f"  ⚠️ {len(select_cct_rows(cct_all)) - len(cct)} photos could not be downloaded; run again.")
CCT_LOCATIONS = sorted(cct["location"].unique().tolist())
print(f"{len(cct)} photos · {cct['seq_id'].nunique()} bursts · cameras {CCT_LOCATIONS}")
print("photos per class: " + " · ".join(f"{c} {int((cct['label'] == c).sum())}" for c in CCT_CLASSES))
show_bursts(cct, n_bursts=3)

# %% [markdown]
# 💡 The frames of a burst differ by one second and a few steps of the animal; everything else, the tree,
# the ground and the exposure, is identical. If one frame is in test and its siblings in train, the model is
# asked to recognise a photo it has almost seen.
#
# ### Leakage, measured without a model
#
# No training is needed to see the leak. For any split, count **the share of test photos that have a burst
# sibling (same `seq_id`) in training**. A random split scatters the frames of each burst over both halves;
# a split **by camera** keeps every burst, and every tree, on one side.
#
# **Q17.** 🟡 Predict the sibling share of a random 80 / 20 split of the trail-camera photos: about 0 %,
# about 20 %, or most of the test photos? And of a split by camera?

# %%
# Share of test photos with a burst sibling in training: random split vs split by camera.
def sibling_rate(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Share of test photos whose seq_id (burst) also appears in the training table (0 = no leak)."""
    if len(test) == 0:
        return 0.0
    return float(test["seq_id"].isin(set(train["seq_id"])).mean())


_tr, _te = train_test_split(cct.index.to_numpy(), test_size=TEST_SIZE, random_state=SEED)
sibling_random = sibling_rate(cct.loc[_tr], cct.loc[_te])

_gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
_tr_g, _te_g = next(_gss.split(cct, groups=cct["location"]))
sibling_camera = sibling_rate(cct.iloc[_tr_g], cct.iloc[_te_g])
_held_out = sorted(cct.iloc[_te_g]["location"].unique().tolist())

RESULTS["sibling_rate_random"] = sibling_random
RESULTS["sibling_rate_by_camera"] = sibling_camera
print(f"random split    : {sibling_random:5.1%} of the {len(_te)} test photos have a burst sibling in train")
print(f"split by camera : {sibling_camera:5.1%} of the {len(_te_g)} test photos (held out: {_held_out})")

fig, ax = plt.subplots(figsize=(5.5, 3))
ax.bar(["random split", "split by camera"], [sibling_random, sibling_camera],
       color=[CLASS_COLORS["cat"], CLASS_COLORS["zebra"]])
ax.set_ylabel("test photos with a burst sibling in train")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
plt.tight_layout()
plt.show()

# %% [markdown]
# 💡 With bursts of three frames, a random split gives most test photos a sibling in training; splitting by
# camera brings the share to zero by construction, because a burst never spans two cameras.
#
# **Worked answer (Q17).** A random split leaks for most test photos: a frame lands in test, and with high
# probability at least one of its two siblings landed in train. The split by camera leaks for none. The same
# effect shows in the paper the data comes from: Beery et al. (2018) report a classification **error of
# 20.8 % at cameras seen in training and 41.1 % at cameras never seen**, doubling at the first new tree. A
# random split would have reported the 20.8 % and the park would have discovered the 41.1 % in the field.
#
# <table width="100%"><tr><td bgcolor="#FDECEA"><font color="#8B1A10">
# <b>⚠️ Pitfall · Leave-one-out leaks like a random split</b><br>
# LOO sounds strict (train on everything but one photo), but on burst data it is the <i>worst</i> case: the
# held-out frame's siblings are always in training. The fix is <b>leave-one-<i>group</i>-out</b> with the
# camera as the group: every fold holds out one whole camera, so the sibling share is zero in every fold
# and the score is the one a new camera will see. The same holds for any "one item at a time" scheme: the
# unit you leave out must be the unit that will be new.
# </font></td></tr></table>
#
# The cell below checks that claim fold by fold: leave-one-out and leave-one-group-out on the trail-camera
# photos, each reporting the mean sibling share over its folds.

# %%
# Leave-one-out vs leave-one-group-out (camera) on the trail cameras: sibling share per fold.
def mean_sibling_rate(splitter, table: pd.DataFrame, groups=None) -> float:
    """Mean over the folds of the share of test photos with a burst sibling in the fold's training part."""
    rates = [sibling_rate(table.iloc[tr], table.iloc[te]) for tr, te in splitter.split(table, groups=groups)]
    return float(np.mean(rates))


loo_sibling = mean_sibling_rate(LeaveOneOut(), cct)
logo_sibling = mean_sibling_rate(LeaveOneGroupOut(), cct, groups=cct["location"])
cv_schemes += ["logo"]
RESULTS["cv_schemes"] = cv_schemes
RESULTS["sibling_rate_loo"] = loo_sibling
RESULTS["sibling_rate_logo"] = logo_sibling
print(f"leave-one-out        : {len(cct)} folds, sibling share {loo_sibling:.1%}")
print(f"leave-one-group-out  : {cct['location'].nunique()} folds (one per camera), sibling share "
      f"{logo_sibling:.1%}")
print(f"cross-validation schemes run in this notebook: {cv_schemes}")

# %% [markdown]
# 💡 LOO's sibling share is the share of photos that belong to a burst of more than one frame, which is
# nearly all of them; LOGO by camera is zero in every fold.
#
# ### ✍️ Choose and run a split
#
# <table width="100%"><tr><td bgcolor="#FFF4D6"><font color="#7A4A00">
# <b>✏️ Your turn · <code>split_by_camera</code></b><br>
# Fill in the function: given the trail-camera table and a list of camera locations to hold out, return a
# dictionary <code>{"train": [...], "test": [...]}</code> of <code>file_name</code> values, every photo of a
# held-out camera in test and every other photo in train, each list sorted. Then choose
# <code>TEST_LOCATIONS</code> (any cameras from <code>CCT_LOCATIONS</code>). The <code>use()</code> line
# checks your function against the reference and continues with the reference either way.
# </font></td></tr></table>

# %%
# ✍️ YOUR CODE
def split_by_camera(table, test_locations):
    """Split the trail-camera table by camera.

    table: a DataFrame with columns file_name and location (cct). test_locations: list of location ids.
    Returns {"train": sorted file_names of every other camera, "test": sorted file_names of test_locations}.
    """
    ...


def _ref_split_by_camera(table, test_locations):
    in_test = table["location"].isin(test_locations)
    return {"train": sorted(table.loc[~in_test, "file_name"]),
            "test": sorted(table.loc[in_test, "file_name"])}


TEST_LOCATIONS = [CCT_LOCATIONS[0], CCT_LOCATIONS[-1]]  # choose: any cameras from CCT_LOCATIONS
_bad = [loc for loc in TEST_LOCATIONS if loc not in CCT_LOCATIONS]
if _bad or not TEST_LOCATIONS or len(set(TEST_LOCATIONS)) >= len(CCT_LOCATIONS):
    raise SystemExit(f"TEST_LOCATIONS must be a non-empty subset of {CCT_LOCATIONS} that leaves cameras for "
                     f"training; unknown: {_bad}")

my_split = use(split_by_camera, cct, TEST_LOCATIONS, reference=_ref_split_by_camera)
_train_t = cct[cct["file_name"].isin(my_split["train"])]
_test_t = cct[cct["file_name"].isin(my_split["test"])]
print(f"train: {len(_train_t)} photos from cameras {sorted(_train_t['location'].unique().tolist())}")
print(f"test : {len(_test_t)} photos from cameras {sorted(_test_t['location'].unique().tolist())}")
print(f"sibling share of this split: {sibling_rate(_train_t, _test_t):.1%}")
_split_shares = pd.DataFrame({
    "train": _train_t["label"].value_counts(normalize=True).reindex(CCT_CLASSES).fillna(0),
    "test": _test_t["label"].value_counts(normalize=True).reindex(CCT_CLASSES).fillna(0),
}).round(2)
display(_split_shares.T)
_absent = [c for c in CCT_CLASSES if _split_shares.loc[c, "test"] == 0]
if _absent:
    print(f"⚠️ no {', '.join(_absent)} photos in the test cameras: those classes get no score. "
          "Try other cameras.")

# %% [markdown]
# 💡 A split by camera has no siblings across the cut, but it can leave a class without test photos
# (not every animal walks past every camera): you trade a clean score for a less balanced one, and you choose
# the held-out cameras with both in mind.
#
# ### 🔴 Optional: a ruler to see the gap
#
# <table width="100%"><tr><td bgcolor="#E3F2FD"><font color="#0D47A1">
# <b>📘 Theory · The nearest-neighbour ruler (not training)</b><br>
# No network is trained in this notebook. To <i>see</i> the gap the two splits create, we use a
# <b>ruler</b>: a 1-nearest-neighbour classifier on each photo's colour histogram (the 96 numbers section 3
# computed). It has no parameters to learn, runs in seconds on a CPU, and is nowhere near a real model. Its
# only job is to measure the same thing twice: once with k-fold (random, leaks across bursts) and once with
# leave-one-camera-out.
# </font></td></tr></table>
#
# **Q18.** 🔴 Will the ruler score higher with k-fold or with leave-one-camera-out, and roughly by how much?
# Predict, then run.

# %%
# The colour-histogram ruler: 1-NN accuracy under k-fold vs leave-one-camera-out on the trail cameras.
from sklearn.neighbors import KNeighborsClassifier

RULER_THUMB = 64
_features = np.stack([measure_image(p, RULER_THUMB)[1].ravel() for p in cct["path"]])
_labels = cct["label"].to_numpy()


def ruler_accuracy(splitter, groups=None) -> float:
    """Mean 1-NN accuracy over the folds of `splitter` on the colour-histogram features."""
    scores = []
    for tr, te in splitter.split(_features, _labels, groups):
        knn = KNeighborsClassifier(n_neighbors=1).fit(_features[tr], _labels[tr])
        scores.append(float((knn.predict(_features[te]) == _labels[te]).mean()))
    return float(np.mean(scores))


_n_cameras = cct["location"].nunique()
ruler_kfold = ruler_accuracy(KFold(min(K_FOLDS, _n_cameras), shuffle=True, random_state=SEED))
ruler_by_camera = ruler_accuracy(GroupKFold(_n_cameras), groups=cct["location"])
RESULTS["ruler_kfold"] = ruler_kfold
RESULTS["ruler_by_camera"] = ruler_by_camera
print(f"ruler accuracy, k-fold (random)      : {ruler_kfold:.1%}")
print(f"ruler accuracy, leave-one-camera-out : {ruler_by_camera:.1%}")
print(f"chance level with {len(CCT_CLASSES)} classes: {1 / len(CCT_CLASSES):.0%}")

fig, ax = plt.subplots(figsize=(5.5, 3))
ax.bar(["k-fold (random)", "leave-one-camera-out"], [ruler_kfold, ruler_by_camera],
       color=[CLASS_COLORS["cat"], CLASS_COLORS["zebra"]])
ax.axhline(1 / len(CCT_CLASSES), color=INK, linestyle="--", linewidth=1, label="chance")
ax.set_ylabel("1-NN ruler accuracy")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# 💡 The ruler scores by finding the most similar photo in training: under k-fold that photo is usually a
# burst sibling with the same animal in front of the same tree, so it looks good; under leave-one-camera-out
# it has to find a similar photo from another camera and falls towards chance.
#
# **Worked answer (Q18).** The random scheme scores higher, and the gap is large: on the full slice the ruler
# roughly doubles its accuracy when its neighbours are allowed to be siblings. The exact numbers depend on
# the slice and the seed, so read the direction and the size of the gap, not the digits. A real model would
# score higher on both, and the gap would still be there: that is the 20.8 % → 41.1 % of the paper.
#
# <table width="100%"><tr><td bgcolor="#EDE7F6"><font color="#311B92">
# <b>🔀 Across tasks · Splitting for boxes and masks</b><br>
# <ul>
# <li><b>Classification:</b> stratify by the image label; group by camera, day or source.</li>
# <li><b>Detection:</b> the unit is still the <i>image</i> (never split boxes of one photo across halves),
# but one photo carries several labels, so plain stratification does not apply. <i>Iterative
# stratification</i> balances the box counts per class across folds (section 8).</li>
# <li><b>Segmentation:</b> the same, with pixel shares instead of box counts; a rare class with a few
# large masks can end up entirely in one fold if you do not check (section 9).</li>
# </ul>
# In all three, deduplicate first (section 5), then split, then never look at the test half again.
# </font></td></tr></table>
#
# **Takeaway.** A split decides what the score means. Stratify so every class is scored; group by whatever
# will be new when the model is deployed, so the score is the one you will see there; and measure leakage
# before any model with one number, the share of test photos with a sibling in training. Leave-one-out is
# for tiny datasets and is no protection against siblings; leave-one-group-out is. Section 7 picks the
# metric to compute on that test half.
