# %% [markdown]
# ## 6. Splits & leakage
#
# Which photos does the model never see, and does the score on them mean anything?

# %% [markdown]
# <!-- callout: objective | Objective -->
# Choose a split that does not leak, and know what k-fold, stratified, leave-one-out and leave-one-group-out
# (LOGO; here a group = a camera) cross-validation each buy you.

# %% [markdown]
# The next decision decides whether next week's test score means anything: **which photos does the model
# never see?**

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · What the held-out part should look like -->
# The held-out part has to look like the photos the park's cameras take *after* training. If a test photo
# has a near-twin in training, the score reports memory, not recognition.

# %% [markdown]
# <!-- callout: theory | Theory · Data leakage -->
# **Leakage** is any way information about the test photos reaches the model during training: a duplicate in
# both train and test (section 5), or a near-twin from the same moment (a burst, the same tree). The score
# then measures memory.
#
# **Rule:** split by whatever will be *new* at deployment (a camera, a day, a farm), not by photo.

# %% [markdown]
# ### Random vs stratified split on the Play Dataset
#
# A random split shuffles the photos and cuts them 80 / 20. A **stratified** split cuts *each class* 80 / 20,
# so train and test keep the class shares of the whole table. A **class share** = a class's photos / all
# photos in that set (30 cats among 100 = 30 %).

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · One bad shuffle, in numbers -->
# Picture 100 cats and 20 sheep, and a random test set of 24 photos (20 %). On average it holds 20 cats and
# 4 sheep (shares 83 % and 17 %). A bad shuffle gives 22 cats and 2 sheep (92 % and 8 %): the cat share moved
# by 9 points, the sheep share was cut in half.

# %% [markdown]
# **Q14.** 🟡 On the Play Dataset's random 80 / 20 split, for each class compare its share in the test set
# with its share overall. Which class is off by the biggest fraction of itself: the biggest or the smallest?
# Does the stratified split fix that completely, or only mostly? Decide, then run.

# %%
# @title Q14 Random vs stratified 80 / 20 split and the class share of each test set { display-mode: "form" }
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
_whole = shares["whole table"].replace(0, np.nan)
_rel = shares.sub(shares["whole table"], axis=0).abs().div(_whole, axis=0) * 100
print("largest drift relative to the class's own share, in %:")
print(f"  random split: {_rel['random · test'].max():.0f} ({_rel['random · test'].idxmax()})   "
      f"stratified split: {_rel['stratified · test'].max():.0f} ({_rel['stratified · test'].idxmax()})")
RESULTS["split_drift_random_pp"] = float(_drift["random · test"].max())
RESULTS["split_drift_stratified_pp"] = float(_drift["stratified · test"].max())

# %% [markdown]
# 💡 **Worked answer (Q14).** The smallest class moves the most relative to its own size: one photo can halve
# a small share (grey bars = whole table, coloured = test set), and if that class is the dog, the alert is
# scored on a handful of photos. Stratifying fixes it up to rounding (you cannot put 10.6 sheep in test), but
# says nothing about *what is in* the photos: that is the rest of this section.

# %% [markdown]
# ### Cross-validation: four ways to reuse one dataset
#
# One split gives one score that moves by several points with a few hundred photos. **Cross-validation**
# rotates the test part around the table so every photo is tested once, and reports the mean (and spread)
# over the folds.

# %% [markdown]
# <!-- callout: theory | Theory · k-fold, stratified k-fold, leave-one-out, leave-one-group-out -->
# - **k-fold:** k pieces (*folds*), each tested once, trained on the other k−1. *Park:* 5 folds of 100 photos.
# - **stratified k-fold:** every fold keeps the class shares; use it when classes are imbalanced.
# - **leave-one-out (LOO):** k = the number of photos; each photo is tested alone. *Park:* 100 trainings.
# - **leave-one-group-out (LOGO; here a group = a camera):** each fold holds out one whole camera. The only
#   scheme that tests a *new* camera.

# %% [markdown]
# The next cell draws the four schemes on 12 toy photos from 3 cameras (`D` = a dog, the rare class), three
# folds each: magenta photos are tested, grey-blue ones are trained on.

# %%
# @title Diagram: which photos are tested in k-fold, stratified k-fold, LOO and LOGO { display-mode: "form" }
SPLIT_COLORS = {"train": "#8FA8AD", "test": "#B5446E"}  # train: neutral grey-blue · test: the held-out part
_dogs = {2, 3, 6}  # toy photos 0-11, four per camera (A: 0-3, B: 4-7, C: 8-11); the dog is the rare class
_cv_folds = {  # the test photos of three folds per scheme
    "k-fold (a fold may miss the dog)": [{2, 3, 7, 10}, {0, 5, 8, 11}, {1, 4, 6, 9}],
    "stratified k-fold (each fold has one dog)": [{0, 2, 4, 9}, {1, 3, 7, 10}, {5, 6, 8, 11}],
    "LOO (12 folds, the first three)": [{0}, {1}, {2}],
    "LOGO (one camera per fold)": [{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}],
}
fig, axes = plt.subplots(len(_cv_folds), 1, figsize=(8.5, 6.4))
for ax, (name, folds) in zip(axes, _cv_folds.items(), strict=True):
    for i, test in enumerate(folds):
        for p in range(12):
            x = p + 0.35 * (p // 4)  # a small gap between cameras
            ax.add_patch(mpl.patches.FancyBboxPatch(
                (x + 0.06, i + 0.08), 0.88, 0.84, boxstyle="round,pad=0.02",
                facecolor=SPLIT_COLORS["test" if p in test else "train"], edgecolor="none"))
            if p in _dogs:
                ax.text(x + 0.5, i + 0.5, "D", ha="center", va="center", fontsize=8, color="white",
                        fontweight="bold")
    ax.set_xlim(0, 12.7)
    ax.set_ylim(3, 0)
    ax.set_xticks([])
    ax.set_yticks([0.5, 1.5, 2.5], ["fold 1", "fold 2", "fold 3"], fontsize=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(name, loc="left", fontsize=9, color=INK)
axes[-1].set_xticks([2, 6.35, 10.7], ["camera A", "camera B", "camera C"], fontsize=8)
axes[-1].tick_params(length=0)
fig.suptitle("Toy table of 12 photos: every row is one fold, test photos in magenta", color=INK)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# 💡 Read the picture: plain k-fold can leave a fold without the dog, stratified k-fold gives every fold one,
# and a leave-one-out fold holds a single photo, so some class has **0 %** of it. LOGO holds out a whole
# camera.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · When leave-one-out is worth it -->
# <details><summary>LOO on a very small dataset, and the scikit-learn names</summary>
#
# - LOO suits a **very small** dataset (a few dozen photos) where every training example counts.
# - Otherwise it costs n trainings, and each fold scores one photo, 0 or 1, so the mean is noisy and says
#   nothing per class. On a slice of 40 photos: 40 folds of 1 test photo, 40 trainings for one score.
# - Plain k-fold lets a small class dip in some fold; stratified k-fold holds it steady.
# - LOGO needs groups: the Play Dataset has none (COCO photos come from thousands of Flickr accounts), the
#   park's cameras do. scikit-learn: `LeaveOneGroupOut`, or `GroupKFold` with k = the number of groups.
# - The next (collapsed) cell counts the folds of each scheme on the Play Dataset, with LOO on a slice of 40
#   photos.
#
# </details>

# %%
# @title Count the folds of k-fold, stratified k-fold and leave-one-out { display-mode: "form" }
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
            "lowest class share in a test fold": round(min_share, 3)}


_loo_ids = _ids[:LOO_N]
cv_rows = [
    summarise_folds("kfold", KFold(K_FOLDS, shuffle=True, random_state=SEED), _ids, _y),
    summarise_folds("stratified_kfold", StratifiedKFold(K_FOLDS, shuffle=True, random_state=SEED), _ids, _y),
    summarise_folds("loo", LeaveOneOut(), _loo_ids, _y[:LOO_N]),
]
cv_schemes = [r["scheme"] for r in cv_rows]

# %% [markdown]
# ### The woodland trail cameras
#
# The park has just hung **trail cameras in its woodland reserve**. A camera fires on motion and takes a
# **burst** of three frames a second apart, every night, from the same tree, for years.
#
# - **Stand-in data:** a slice of the Caltech Camera Traps dataset (CCT20): eight cameras, five animals
#   (opossum, raccoon, coyote, bobcat, rabbit), whole bursts kept together.
# - **First run:** the next cell downloads the photos and shrinks them to 512 px, so **it takes a minute**;
#   afterwards it is instant.

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
# @title Load the trail-camera slice, download and shrink the photos, show bursts { display-mode: "form" }
cct_all = pd.read_csv(fetch_manifest("cct20_slice.csv"))
cct = download_cct(select_cct_rows(cct_all))
if len(cct) < len(select_cct_rows(cct_all)):
    print(f"  ⚠️ {len(select_cct_rows(cct_all)) - len(cct)} photos could not be downloaded; run again.")
CCT_LOCATIONS = sorted(cct["location"].unique().tolist())
print(f"{len(cct)} photos · {cct['seq_id'].nunique()} bursts · cameras {CCT_LOCATIONS}")
print("photos per class: " + " · ".join(f"{c} {int((cct['label'] == c).sum())}" for c in CCT_CLASSES))
show_bursts(cct, n_bursts=3)

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · Why a burst is a leak -->
# The frames of a burst differ by a second and a few steps; the tree, the ground and the exposure are
# identical. A test frame whose siblings are in train is a photo the model has almost seen.

# %% [markdown]
# ### Leakage, measured without a model
#
# The **sibling rate** is the share of test photos that have a burst sibling (another frame with the same
# `seq_id`) in training. 0 % means no leak; no training is needed to see it.
#
# 1. A burst of three frames from camera A meets a random split: frames 1 and 3 go to train, frame 2 to test.
# 2. Frame 2 has two siblings in training, so it counts toward the sibling rate.
# 3. The model has seen this tree and this raccoon: frame 2 is easy, the score too good.
# 4. A split **by camera** keeps all three frames on one side: nothing leaks.

# %%
# @title Diagram: three bursts under a random split and a split by camera { display-mode: "form" }
_demo = {  # three cameras, one burst of three frames each; where does every frame land?
    "random split": [["train", "test", "train"], ["train", "train", "test"], ["test", "train", "train"]],
    "split by camera": [["train"] * 3, ["train"] * 3, ["test"] * 3],
}
fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
for ax, (name, grid) in zip(axes, _demo.items(), strict=True):
    n_test = sum(f == "test" for burst in grid for f in burst)
    n_leaky = sum(f == "test" for burst in grid if "train" in burst for f in burst)
    for i, burst in enumerate(grid):
        for j, where in enumerate(burst):
            ax.add_patch(mpl.patches.FancyBboxPatch((j + 0.08, i + 0.08), 0.84, 0.84,
                                                    boxstyle="round,pad=0.02",
                                                    facecolor=SPLIT_COLORS[where], edgecolor="none"))
            ax.text(j + 0.5, i + 0.5, f"frame {j + 1}\n{where}", ha="center", va="center", fontsize=8,
                    color="white")
    ax.set_xlim(0, 3)
    ax.set_ylim(3, 0)
    ax.set_xticks([])
    ax.set_yticks([0.5, 1.5, 2.5], ["camera A", "camera B", "camera C"])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{name}: sibling rate {n_leaky / n_test:.0%}", fontsize=10, color=INK)
fig.suptitle("One burst per camera: every frame is coloured by the set it lands in", color=INK)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# **Q15.** 🟡 Take 100 trail-camera photos in bursts of three and a random 80 / 20 split (20 photos in test).
# For one test photo, each of its two siblings lands in training with probability 0.8. Predict the sibling
# rate of the random split: about 0 %, about 20 %, or most of the test photos? And of a split by camera?

# %%
# @title Q15 Test photos with a burst sibling in training: random vs by camera { display-mode: "form" }
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
# The same random split on real photos: a test frame (magenta) next to its siblings in training (grey-blue).

# %%
# @title Q15 on real photos: a test frame next to its siblings in training { display-mode: "form" }
def show_leaks(table: pd.DataFrame, train_idx, test_idx, n_examples: int = 2, size: int = 256) -> None:
    """For a few test photos that have siblings in training, show the whole burst: test vs train frames."""
    in_test = set(test_idx)
    train_bursts = set(table.loc[train_idx, "seq_id"])
    leaked = table.loc[test_idx]
    leaked = leaked[leaked["seq_id"].isin(train_bursts)].drop_duplicates("seq_id").head(n_examples)
    if leaked.empty:
        print("no test photo has a sibling in training in this split.")
        return
    bursts = [table[table["seq_id"] == s] for s in leaked["seq_id"]]
    cols = max(len(b) for b in bursts)
    fig, axes = plt.subplots(len(bursts), cols, figsize=(2.6 * cols, 2.3 * len(bursts)), squeeze=False)
    for r, burst in enumerate(bursts):
        for c in range(cols):
            ax = axes[r, c]
            if c >= len(burst):
                ax.axis("off")
                continue
            row = burst.iloc[c]
            part = "test" if burst.index[c] in in_test else "train"
            ax.imshow(load_thumb(row["path"], size))
            frame(ax, SPLIT_COLORS[part])
            ax.set_title(f"camera {row['location']} · frame {row['frame_num']} · {part}", fontsize=8)
    fig.suptitle("Random split: the test frame and its siblings in training", color=INK)
    fig.tight_layout()
    display(fig)
    plt.close(fig)


show_leaks(cct, _tr, _te)

# %% [markdown]
# 💡 **Worked answer (Q15).** Random split: **most test photos**, about 96 of 100. Split by camera: **0 %**,
# because a burst never spans two cameras.
#
# A test frame is safe only if *both* siblings are in test too, probability about 0.2 × 0.2 = 4 %; the rate
# is a bit lower on the real slice, where some bursts are shorter than three frames.
#
# **In the field:** Beery et al. (2018) report an error of **20.8 % at cameras seen in training and 41.1 % at
# cameras never seen**: it doubles at the first new tree. A random split would have reported the 20.8 % and
# the park would have found the 41.1 % in the field.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · Where this data comes from -->
# The trail-camera photos are a slice of the Caltech Camera Traps data, introduced in Beery, Van Horn and
# Perona (2018), *Recognition in Terra Incognita*, the paper behind the two error rates above. It tests the
# same trained model at cameras it has seen and at cameras it has never seen. `data/cct20_slice.csv` lists
# the photos. The five woodland classes borrow the park's colours, so each class keeps one colour.

# %% [markdown]
# <!-- callout: pitfall | Pitfall · LOO leaks like a random split -->
# LOO sounds strict (train on everything but one photo), but on burst data it is the *worst* case:
#
# 1. Hold out frame 2 of a burst.
# 2. Frames 1 and 3, same tree and animal, are in training: the sibling rate is as high as it can be.
#
# **Fix:** LOGO with the camera as the group: every fold holds out one whole camera, so the sibling rate is
# zero. The unit you leave out must be the unit that will be new.
#
# <details><summary>More on this pitfall</summary>
#
# "As high as it can be" is the share of photos that belong to a burst of more than one frame: nearly all of
# them. With LOGO the score is the one a new camera will see.
#
# </details>

# %% [markdown]
# The next cell checks that claim fold by fold: the mean sibling rate of LOO and LOGO on the trail-camera
# photos.

# %%
# @title LOO vs LOGO (camera): sibling rate per fold { display-mode: "form" }
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
print(f"LOO  : {len(cct)} folds, sibling rate {loo_sibling:.1%}")
print(f"LOGO : {cct['location'].nunique()} folds (one per camera), sibling rate {logo_sibling:.1%}")
print(f"cross-validation schemes run in this notebook: {cv_schemes}")

# %% [markdown]
# ### ✍️ Choose and run a split

# %% [markdown]
# Before you choose the held-out cameras, the next cell counts each class's photos per camera: a class with
# no photos at your test cameras gets no score.

# %%
# @title Photos per class per camera (to keep every class in test) { display-mode: "form" }
display(pd.crosstab(cct["location"], cct["label"]).reindex(columns=CCT_CLASSES, fill_value=0))

# %%
# @title Reference solution — peek only if you are stuck { display-mode: "form" }
def _ref_split_by_camera(table, test_locations):
    in_test = table["location"].isin(test_locations)
    return {"train": sorted(table.loc[~in_test, "file_name"]),
            "test": sorted(table.loc[in_test, "file_name"])}


# %% [markdown]
# <!-- callout: your-turn | Your turn · `split_by_camera` -->
# Fill in `split_by_camera(table, test_locations)`:
#
# - **Input:** the table `cct` (columns `file_name`, `location`) and a list of cameras to hold out.
# - **Output:** `{"train": [...], "test": [...]}` of sorted `file_name` values: held-out cameras in test,
#   all others in train.
# - **Then** choose `TEST_LOCATIONS` (cameras that keep every class in test, see the table above). `use()`
#   checks your function and continues with the reference either way.

# %%
# ✍️ YOUR CODE
def split_by_camera(table, test_locations):
    """Split the trail-camera table by camera.

    table: a DataFrame with columns file_name and location (cct). test_locations: list of location ids.
    Returns {"train": sorted file_names of every other camera, "test": sorted file_names of test_locations}.
    """
    ...


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
print(f"sibling rate of this split: {sibling_rate(_train_t, _test_t):.1%}")
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
# 💡 A split by camera has no siblings across the cut, but it can leave a class without test photos. You
# trade a clean score for a less balanced one: choose the held-out cameras with both in mind.
#
# <details><summary>An example</summary>
#
# Not every animal walks past every camera. Holding out cameras that never see a coyote leaves the test set
# without one (and holding out all the cameras that do leaves training without one).
#
# </details>

# %% [markdown]
# ### 🔴 Optional · A ruler to see the gap

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · The nearest-neighbour ruler -->
# No network is trained in this notebook. To *see* the gap the two splits create, we use a **ruler, not
# training**: a 1-nearest-neighbour classifier on each photo's colour histogram (the 96 numbers of section 3).
# It copies the label of the most similar training photo, has nothing to learn, and runs in seconds on a CPU.
# It measures the same thing twice: with k-fold (random, leaks across bursts) and with LOGO (one camera out).

# %% [markdown]
# **Q16.** 🔴 With 5 animals, guessing scores 20 %. Will the ruler score higher with k-fold or with LOGO, and
# roughly by how much? Predict, then run.

# %%
# @title Q16 Colour-histogram ruler: 1-NN accuracy, k-fold vs LOGO (one camera out) { display-mode: "form" }
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
ruler_by_camera = ruler_accuracy(GroupKFold(_n_cameras), groups=cct["location"])  # k = cameras: LOGO
RESULTS["ruler_kfold"] = ruler_kfold
RESULTS["ruler_by_camera"] = ruler_by_camera
print(f"ruler accuracy, k-fold (random)  : {ruler_kfold:.1%}")
print(f"ruler accuracy, LOGO (by camera) : {ruler_by_camera:.1%}")
print(f"chance level with {len(CCT_CLASSES)} classes: {1 / len(CCT_CLASSES):.0%}")

fig, ax = plt.subplots(figsize=(5.5, 3))
ax.bar(["k-fold (random)", "LOGO (by camera)"], [ruler_kfold, ruler_by_camera],
       color=[CLASS_COLORS["cat"], CLASS_COLORS["zebra"]])
ax.axhline(1 / len(CCT_CLASSES), color=INK, linestyle="--", linewidth=1, label="chance")
ax.set_ylabel("1-NN ruler accuracy")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# 💡 **Worked answer (Q16).** K-fold scores higher, by a large gap: on the full slice the ruler roughly
# doubles its accuracy when its neighbours may be burst siblings (same animal, same tree), while under LOGO
# it must find a similar photo at another camera and falls towards chance. Read the direction and the size of
# the gap, not the digits (they depend on the slice and the seed). A real model scores higher on both and the
# gap stays: it is the 20.8 % → 41.1 % of the paper.

# %% [markdown]
# <!-- callout: across-tasks | Across tasks · Splitting for boxes and masks -->
# - **Classification:** stratify by the image label; group by camera, day or source.
# - **Detection:** the unit is still the *image*, never the box. A paddock photo with 3 sheep and 1 horse
#   cannot have its sheep in train and its horse in test. One photo carries several labels, so plain
#   stratification does not apply: *iterative stratification* balances the box counts per class across folds
#   (section 8).
# - **Segmentation:** the same, with pixel shares instead of box counts. A rare class with a few large masks,
#   the aviary's birds, can end up entirely in one fold if you do not check (section 9).
#
# In all three: deduplicate first (section 5), then split, then never look at the test set again.

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **When you see** classes of very different sizes → **do** stratify the split, so every class is in the
#   test set and scored.
# - **When you see** photos from the same moment, camera or day (a burst, one tree) → **do** split by that
#   group and measure the sibling rate (it should be 0 %) before any model.
# - **When you see** a tiny dataset → **do** use LOO only if every photo counts, and LOGO if the photos come
#   in groups.
# - **When you see** a test score that looks too good → **do** ask what train and test share. Section 7 picks
#   the metric to compute on that test set.
