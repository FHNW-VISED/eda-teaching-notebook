# %% [markdown]
# ## 6. Splits & leakage
#
# Which photos does the model never see, and does the score on them mean anything?

# %% [markdown]
# <!-- callout: objective | Objective -->
# Choose a split that does not leak, and know what k-fold, stratified, leave-one-out and leave-one-group-out
# cross-validation each buy you.

# %% [markdown]
# Section 5 ended with `clean_images`, a table without duplicates. The next decision decides whether next
# week's test score means anything: **which photos does the model never see?**
#
# The held-out part has to look like the photos the park's cameras take *after* training, not like the ones
# they already took. If a test photo has a near-twin in training, the score reports memory, not recognition.

# %% [markdown]
# <!-- callout: theory | Theory · Data leakage -->
# **Leakage** is any way information about the test photos reaches the model during training.
#
# - **The obvious route:** a duplicate in both halves (section 5).
# - **The quiet route:** a photo that is not a copy but comes from the same moment. A woodland trail camera
#   fires a burst of three frames a second apart, so a random split can put frames 1 and 3 of one raccoon in
#   training and frame 2 in test: same tree, same pose, same light.
# - **The cost:** the model learns "this tree", the test score goes up, and the first camera at a new tree
#   breaks it.
#
# **Rule:** split by whatever will be *new* at deployment (a camera, a day, a farm), not by photo.

# %% [markdown]
# ### Random vs stratified split on the Play Dataset
#
# The simplest split shuffles the photos and cuts them 80 / 20. A **stratified** split cuts *each class*
# 80 / 20, so both halves keep the class shares of the whole table.
#
# A **class share** is a class's photos divided by all photos in that half (30 cats among 100 photos = 30 %).

# %% [markdown]
# <!-- callout: questions -->
# Picture 100 cats and 20 sheep, and a random test half of 24 photos. On average it holds 20 cats and 4 sheep
# (shares 83 % and 17 %). A bad shuffle gives 22 cats and 2 sheep (92 % and 8 %): the cat share moved by 9
# points, the sheep share was cut in half.

# %% [markdown]
# **Q15.** 🟡 On the Play Dataset's random 80 / 20 split, which class share moves the most between the whole
# table and the test half, *relative to the class's own size*: the largest class or the smallest? Does the
# stratified split fix that completely, or only mostly? Decide, then run.

# %%
# @title Q15 Random vs stratified 80 / 20 split and the class share of each test half { display-mode: "form" }
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
# 💡 Grey bars are the whole table, coloured bars the test half. Under a random cut the small classes wobble
# most in relative terms; the stratified cut sits on the grey bars up to the rounding that whole photos force.
#
# **Worked answer (Q15).**
#
# - In percentage points the largest classes can move the most, but what hurts is the *relative* move: a rare
#   class like the paddock's dog may end up almost absent from the test half, the one class the alert is about.
# - Stratification removes that wobble up to rounding (you cannot put 10.6 sheep in test), so the shares match
#   within one photo per class.
# - It does nothing about *what is in* the photos, which is the rest of this section.

# %% [markdown]
# ### Cross-validation: four ways to reuse one dataset
#
# One split gives one score, and with a few hundred photos that score moves by several points when you change
# the seed. **Cross-validation** rotates the test part around the table so every photo is tested once, and
# reports the mean (and spread) of the scores over the folds.

# %% [markdown]
# <!-- callout: theory | Theory · k-fold, stratified k-fold, leave-one-out, leave-one-group-out -->
# - **k-fold:** cut the table into k pieces (*folds*), test on each once and train on the other k−1. Usual k
#   is 5 or 10. *Park:* 5 folds of 100 photos = five rounds with 20 test photos each.
# - **stratified k-fold:** the same, but every fold keeps the class shares. Use it whenever classes are
#   imbalanced, which is always. *Park:* otherwise one fold may hold no dog at all.
# - **leave-one-out (LOO):** k = the number of photos; every photo is tested once against a model trained on
#   all the others. *Park:* 100 photos = 100 trainings.
# - **leave-one-group-out (LOGO):** each fold holds out one whole **group** that must stay together (a
#   camera, a day, a patient). The only scheme here that tests a *new* camera. *Park:* 8 cameras = 8 folds.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · When leave-one-out is worth it -->
# <details><summary>LOO on a very small dataset</summary>
#
# - LOO suits a **very small** dataset (a few dozen photos) where every training example counts.
# - Otherwise it costs n trainings, and each fold scores one photo, 0 or 1, so the mean is noisy and says
#   nothing per class.
#
# </details>

# %% [markdown]
# The next cell runs the first three schemes on the Play Dataset and reports, for each, how many folds there
# are, how big a test fold is, and the **smallest class share** any test fold has (0 means some fold tests
# without a class). LOO runs on a slice of 40 photos, because that is where it belongs.
#
# **Q16.** 🟡 Before running: for leave-one-out on that slice of 40 photos, how many folds are there, how many
# photos does each test fold hold, and what is the smallest class share a test fold can have?

# %%
# @title Q16 k-fold, stratified k-fold, leave-one-out: folds and smallest class share { display-mode: "form" }
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
# 💡 Plain k-fold lets the smallest class's share dip below its overall share in some fold; stratified k-fold
# holds it steady; LOO's test folds are one photo each, so its share column reads 0 whatever you do.
#
# **Worked answer (Q16).**
#
# - 40 photos give **40 folds** (one per photo), each with **1 test photo**, so 40 trainings for one score.
# - A one-photo fold holds one class, so the smallest class share is **0**: five of six classes are absent
#   from every fold. Harmless for the mean, but LOO tells you nothing per class.
# - The fourth scheme, LOGO, needs groups. The Play Dataset has none (COCO photos come from thousands of
#   unrelated Flickr accounts); the park's trail cameras do.

# %% [markdown]
# ### The woodland trail cameras
#
# The park has just hung **trail cameras in its woodland reserve**. A camera fires on motion and takes a
# **burst** of three frames a second apart, every night, from the same tree, for years.
#
# - **Stand-in data:** a slice of the Caltech Camera Traps dataset (CCT20): eight camera locations, five
#   animals (opossum, raccoon, coyote, bobcat, rabbit), whole bursts kept together. `data/cct20_slice.csv`
#   lists the photos.
# - **First run:** the next cell downloads the photos and shrinks them to 512 px, so **it takes a minute**;
#   afterwards it is instant.
# - **Colours:** the five woodland classes borrow the park's colours, so every plot keeps one colour per class.

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
# 💡 The frames of a burst differ by one second and a few steps of the animal; the tree, the ground and the
# exposure are identical. If one frame is in test and its siblings are in train, the model is asked to
# recognise a photo it has almost seen.

# %% [markdown]
# ### Leakage, measured without a model
#
# No training is needed to see the leak. For any split, count the **sibling rate**: the share of test photos
# that have a burst sibling (another frame with the same `seq_id`) in training. 0 % means no leak.
#
# 1. A burst of three frames from camera A meets a random split: frame 1 goes to train, frame 2 to test,
#    frame 3 to train.
# 2. Frame 2 has two siblings in training, so it counts toward the sibling rate.
# 3. The model has already seen this tree and this raccoon, so frame 2 is easy and the test score is too good.
# 4. A split **by camera** keeps all three frames of a burst, and its tree, on one side: nothing leaks.

# %%
# @title Diagram: three bursts under a random split and a split by camera { display-mode: "form" }
SPLIT_COLORS = {"train": "#8FA8AD", "test": "#B5446E"}  # train: neutral grey-blue · test: the held-out part
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
            ax.add_patch(mpl.patches.FancyBboxPatch((j + 0.08, i + 0.08), 0.84, 0.84, boxstyle="round,pad=0.02",
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
fig.suptitle("One burst per camera: every frame is coloured by the half it lands in", color=INK)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# **Q17.** 🟡 Take 100 trail-camera photos in bursts of three and a random 80 / 20 split (20 photos in test).
# For one test photo, each of its two siblings lands in training with probability 0.8. Predict the sibling
# rate of the random split: about 0 %, about 20 %, or most of the test photos? And of a split by camera?

# %%
# @title Q17 Test photos with a burst sibling in training: random vs by camera { display-mode: "form" }
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
# Now the same random split on real photos: a test frame (magenta frame) next to its siblings in training
# (grey-blue frame).

# %%
# @title Q17 on real photos: a test frame next to its siblings in training { display-mode: "form" }
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
# 💡 With bursts of three frames, a random split gives most test photos a sibling in training; splitting by
# camera brings the rate to zero by construction, because a burst never spans two cameras.
#
# **Worked answer (Q17).**
#
# - **Random split: most photos.** A test frame leaks unless *both* siblings are also in test, which has
#   probability about 0.2 × 0.2 = 4 %. So about 96 of 100 test photos have a sibling in train (fewer on the
#   real slice, where some bursts are shorter).
# - **Split by camera: 0 %.** A burst never spans two cameras.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · What a leaky split costs in the field -->
# <details><summary>The paper this data comes from</summary>
#
# Beery et al. (2018) report a classification **error of 20.8 % at cameras seen in training and 41.1 % at
# cameras never seen**: the error doubles at the first new tree. A random split would have reported the 20.8 %
# and the park would have found the 41.1 % in the field.
#
# </details>

# %% [markdown]
# <!-- callout: pitfall | Pitfall · Leave-one-out leaks like a random split -->
# LOO sounds strict (train on everything but one photo), but on burst data it is the *worst* case:
#
# 1. Hold out frame 2 of a burst.
# 2. Frames 1 and 3, same tree and same animal, are in training, so the sibling rate is as high as it can be.
#
# **Fix:** leave-one-*group*-out with the camera as the group. Every fold holds out one whole camera, so the
# sibling rate is zero in every fold and the score is the one a new camera will see.
# The unit you leave out must be the unit that will be new.

# %% [markdown]
# The next cell checks that claim fold by fold: leave-one-out and leave-one-group-out on the trail-camera
# photos, each reporting the mean sibling rate over its folds.

# %%
# @title Leave-one-out vs leave-one-group-out (camera): sibling rate per fold { display-mode: "form" }
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
print(f"leave-one-out        : {len(cct)} folds, sibling rate {loo_sibling:.1%}")
print(f"leave-one-group-out  : {cct['location'].nunique()} folds (one per camera), sibling rate "
      f"{logo_sibling:.1%}")
print(f"cross-validation schemes run in this notebook: {cv_schemes}")

# %% [markdown]
# 💡 LOO's sibling rate is the share of photos that belong to a burst of more than one frame, nearly all of
# them; LOGO by camera is zero in every fold.
#
# ### ✍️ Choose and run a split

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
# - **Input:** the trail-camera table `cct` (columns `file_name`, `location`) and a list of cameras to hold out.
# - **Output:** a dictionary `{"train": [...], "test": [...]}` of `file_name` values, every photo of a
#   held-out camera in test and every other photo in train, each list sorted.
# - **Then** choose `TEST_LOCATIONS` (any cameras from `CCT_LOCATIONS`). The `use()` line checks your function
#   against the reference and continues with the reference either way.

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
# 💡 A split by camera has no siblings across the cut, but it can leave a class without test photos: not
# every animal walks past every camera, so holding out the two cameras that see all the coyotes leaves the
# test half without a coyote. You trade a clean score for a less balanced one, and you choose the held-out
# cameras with both in mind.
#
# ### 🔴 Optional: a ruler to see the gap

# %% [markdown]
# <!-- callout: theory | Theory · The nearest-neighbour ruler (not training) -->
# No network is trained in this notebook. To *see* the gap the two splits create, we use a **ruler**: a
# 1-nearest-neighbour classifier on each photo's colour histogram (the 96 numbers of section 3). It copies the
# label of the most similar training photo, has nothing to learn, and runs in seconds on a CPU. It measures
# the same thing twice: with k-fold (random, leaks across bursts) and with leave-one-camera-out.

# %% [markdown]
# **Q18.** 🔴 With 5 animals, guessing scores 20 %. Will the ruler score higher with k-fold or with
# leave-one-camera-out, and roughly by how much? Predict, then run.

# %%
# @title Q18 Colour-histogram ruler: 1-NN accuracy, k-fold vs leave-one-camera-out { display-mode: "form" }
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
# 💡 Under k-fold the most similar training photo is usually a burst sibling (same animal, same tree), so the
# ruler looks good; under leave-one-camera-out it must find a similar photo at another camera and falls
# towards chance.
#
# **Worked answer (Q18).** The random scheme scores higher, and the gap is large: on the full slice the ruler
# roughly doubles its accuracy when its neighbours may be siblings. Read the direction and the size of the
# gap, not the digits (they depend on the slice and the seed). A real model scores higher on both and the gap
# stays: it is the 20.8 % → 41.1 % of the paper.

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
# In all three: deduplicate first (section 5), then split, then never look at the test half again.

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **When you see** classes of very different sizes → **do** stratify the split, so every class is in the
#   test half and scored.
# - **When you see** photos from the same moment, camera or day (a burst, one tree) → **do** split by that
#   group and measure the sibling rate (it should be 0 %) before any model.
# - **When you see** a tiny dataset → **do** use leave-one-out only if every photo counts, and
#   leave-one-group-out if the photos come in groups.
# - **When you see** a test score that looks too good → **do** ask what train and test share. Section 7 picks
#   the metric to compute on that test half.
