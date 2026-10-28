# %% [markdown]
# ## Dataset Splitting & Data Leakage
#
# How can we split an image dataset so that model performance on unseen images provides a reliable estimate of generalization?
#
# Before training a model, we need to decide which images belong in the training, validation, and test sets. Different splitting strategies affect class distributions and the reliability of model evaluation.

# %%
# @title 📘 Data Leakage and Generalization { display-mode: "form" }
from IPython.display import HTML, display

display(HTML("""
<div style="background:#E3F2FD; border-left:4px solid #0D47A1;
            padding:14px 18px; border-radius:6px; color:#0D47A1;">

<strong>📘 Data Leakage and Generalization</strong>

<p>A test set should provide an independent estimate of how well a model
performs on unseen data.</p>

<p><strong>Data leakage</strong> occurs when information that should be
unavailable during training or model selection influences the model or
its evaluation. For example:</p>

<ul>
  <li><strong>Duplicate images:</strong> Exact or near-duplicates appear
  in both training and test sets.</li>
  <li><strong>Related images:</strong> Images from the same animal, camera,
  or recording session appear in different splits, potentially making
  the test set less independent.</li>
  <li><strong>Preprocessing:</strong> Data-dependent transformations,
  such as normalization using dataset statistics, are fitted using
  test data.</li>
</ul>

<p><strong>Key principle:</strong> Choose a splitting strategy that reflects
the intended application. If the model must generalize to new animals,
cameras, or locations, these groups should be kept separate across splits.</p>

</div>
"""))

# %% [markdown]
# ### 🟢 Random vs. Stratified Splitting: How Do Class Distributions Differ?
#
# A *random split* assigns images to training and test sets without explicitly preserving class proportions. A *stratified split* aims to maintain similar class distributions across both sets.
#
# **Before running the code:**
# - Which classes do you expect to show the largest differences between their overall proportion and their proportion in a random test set?
#
# **Explore:**
# 1. Compare the class distributions of the random and stratified 80/20 splits.
#
# **Observe and interpret:**
# - Which class shows the largest relative deviation in the random test set?
# - Does stratification preserve class proportions exactly? Why or why not?
# - Why might stratification be particularly useful for smaller classes?

# %%
# @title 🔎 Compare Random and Stratified Splits { display-mode: "form" }

TEST_SIZE = 0.2
_ids = images["id"].to_numpy()
_y = images["class"].to_numpy()
_, _te_rand = train_test_split(_ids, test_size=TEST_SIZE, random_state=SEED)
_, _te_strat = train_test_split(_ids, test_size=TEST_SIZE, random_state=SEED, stratify=_y)


def class_shares(table: pd.DataFrame, ids) -> pd.Series:
    """Class proportions for the given image IDs, in CLASSES order; `table` is indexed by image id."""
    counts = table.loc[list(ids), "class"].value_counts()
    return counts.reindex(CLASSES).fillna(0) / max(len(ids), 1)


shares = pd.DataFrame(
    {
        "whole table": class_shares(images_by_id, _ids),
        "random · test": class_shares(images_by_id, _te_rand),
        "stratified · test": class_shares(images_by_id, _te_strat),
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

# %%
# @title 💡 Sample Answer { display-mode: "form" }
from IPython.display import HTML, display

display(HTML("""
<details style="background:#F5F5F5; border:1px solid #9CA3AF;
                border-radius:6px; padding:12px 16px; color:#374151;">
<summary style="cursor:pointer;"><strong>💡 Sample Answer</strong></summary>
<ul>
<li><strong>Before running:</strong> Smaller classes may show larger relative deviations because a few images can substantially change their proportions.</li>
<li><strong>Random split:</strong> Class proportions can differ from the full dataset. The class with the largest relative deviation depends on the random split and should be identified from the results.</li>
<li><strong>Stratified split:</strong> Class proportions are preserved more closely, but not necessarily exactly because image counts must be whole numbers.</li>
<li><strong>Smaller classes:</strong> Stratification helps ensure that underrepresented classes are represented proportionally in both sets, although very small classes may still have too few test images for reliable evaluation.</li>
</ul>
<p><strong>Key takeaway:</strong> Stratification improves class balance across splits, but does not prevent data leakage from duplicate or related images.</p>
</details>
"""))

# %% [markdown]
# ### 🟢 Cross-Validation: How Do We Choose Validation Sets?
#
# A single train/test split provides only one estimate of model performance, which depends on the selected images. *Cross-validation* evaluates a model across multiple splits, providing an estimate of its average performance and variability.
#
# We compare four strategies-k-fold, stratified k-fold, leave-one-out (LOO), and leave-one-group-out (LOGO)-using 12 example images from three cameras.
#
# **Before running the code:**
# - How might different splitting strategies affect class balance and the independence of validation images?
#
# **Explore:**
# 1. Compare which images are used for training and validation across the four strategies.
#
# **Observe and interpret:**
# - How do k-fold and stratified k-fold differ in their class distributions?
# - What distinguishes leave-one-out from k-fold?
# - Which strategy evaluates generalization to an unseen camera, and why?
#
# **Go Deeper (optional):**
# - What are the computational disadvantages of leave-one-out cross-validation?

# %%
# @title 📘 Cross-Validation Strategies { display-mode: "form" }
from IPython.display import HTML, display

display(HTML("""
<div style="background:#E3F2FD; border-left:4px solid #0D47A1;
            padding:14px 18px; border-radius:6px; color:#0D47A1;">

<strong>📘 Cross-Validation Strategies</strong>

<p>Cross-validation evaluates a model across multiple training and validation
splits to estimate its performance and variability.</p>

<ul>
<li><strong>K-fold:</strong> Divide the dataset into k folds. Each fold is
used once for validation, while the remaining k−1 folds are used for training.</li>

<li><strong>Stratified k-fold:</strong> Similar to k-fold, but aims to preserve
class proportions in each fold, particularly useful for imbalanced datasets.</li>

<li><strong>Leave-one-out (LOO):</strong> Each validation fold contains a single
image. This maximizes the training data per fold but requires one training run
per image, making it computationally expensive.</li>

<li><strong>Leave-one-group-out (LOGO):</strong> Each fold holds out an entire
group, such as all images from one camera, to evaluate generalization to unseen
groups. It requires meaningful group identifiers.</li>
</ul>

<p><strong>Key principle:</strong> Choose a cross-validation strategy that
reflects the intended application. Stratification helps preserve class balance,
while group-based splitting prevents related images from appearing in both
training and validation sets.</p>

</div>
"""))

# %% [markdown]
# The diagram shows four strategies using 12 example images from three cameras (D = dog, the rare class). Magenta indicates validation images; gray-blue indicates training images.

# %%
# @title 🔎 Visualize Cross-Validation Strategies { display-mode: "form" }

split_colors = {"train": LIGHT_GRAY, "test": HIGHLIGHT}
dogs = {2, 3, 6}  # Rare class in a toy dataset of 12 photos

cv_folds = {
    "K-fold (a fold may miss the dog)": [{2, 3, 7, 10}, {0, 5, 8, 11}, {1, 4, 6, 9}],
    "Stratified K-fold (each fold has one dog)": [{0, 2, 4, 9}, {1, 3, 7, 10}, {5, 6, 8, 11}],
    "LOO (12 folds, first three shown)": [{0}, {1}, {2}],
    "LOGO (one camera per fold)": [{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}],
}

fig, axes = plt.subplots(len(cv_folds), 1, figsize=(8.5, 7.2))

for ax, (name, folds) in zip(axes, cv_folds.items(), strict=True):
    for i, test in enumerate(folds):
        for p in range(12):
            x = p + 0.35 * (p // 4)  # Gap between cameras

            ax.add_patch(mpl.patches.FancyBboxPatch(
                (x + 0.06, i + 0.08), 0.88, 0.84,
                boxstyle="round,pad=0.02",
                facecolor=split_colors["test" if p in test else "train"],
                edgecolor="none"
            ))

            if p in dogs:
                ax.text(
                    x + 0.5, i + 0.5, "D",
                    ha="center", va="center",
                    fontsize=8, color=INK, fontweight="bold"
                )

    ticks = [0.5, 1.5, 2.5]
    labels = ["fold 1", "fold 2", "fold 3"]

    if name.startswith("LOO"):
        for p in range(12):
            ax.text(
                p + 0.35 * (p // 4) + 0.5, 3.3, "⋮",
                ha="center", va="center", fontsize=11, color=INK
            )
        ticks.append(3.3)
        labels.append("folds 4–12\nnot shown")

    ax.set_xlim(0, 12.7)
    ax.set_ylim(3.6, 0)
    ax.set_xticks([])
    ax.set_yticks(ticks, labels, fontsize=8)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(name, loc="left", fontsize=9, color=INK)

axes[-1].set_xticks(
    [2, 6.35, 10.7],
    ["camera A", "camera B", "camera C"],
    fontsize=8
)
axes[-1].tick_params(length=0)

fig.suptitle(
    "Cross-validation: grey = training, yellow = test, D = dog",
    color=INK
)

fig.tight_layout()
display(fig)
plt.close(fig)

# %%
# @title 💡 Sample Answer { display-mode: "form" }
from IPython.display import HTML, display

display(HTML("""
<details style="background:#F5F5F5; border:1px solid #9CA3AF;
                border-radius:6px; padding:12px 16px; color:#374151;">
<summary style="cursor:pointer;"><strong>💡 Sample Answer</strong></summary>
<ul>
<li><strong>Before running:</strong> Different strategies may produce different class distributions and levels of independence between training and validation images.</li>
<li><strong>K-fold vs. stratified k-fold:</strong> Standard k-fold may produce folds without examples of a rare class. Stratification aims to preserve class proportions across folds.</li>
<li><strong>Leave-one-out:</strong> Each validation fold contains a single image, unlike k-fold, which validates on a subset of images.</li>
<li><strong>Leave-one-group-out:</strong> Holding out an entire camera evaluates generalization to a camera not seen during training.</li>
<li><strong>Go Deeper:</strong> Leave-one-out requires one training run per image, making it computationally expensive.</li>
</ul>
<p><strong>Key takeaway:</strong> Cross-validation strategies differ in class balance, computational cost, and independence between training and validation data.</p>
</details>
"""))

# %% [markdown]
# ### 🔴 Cross-Validation in Practice (Optional)
#
# How many training runs do different cross-validation strategies require?
#
# **Explore:**
# 1. Compare the number of folds and validation images per fold for k-fold, stratified k-fold, and leave-one-out.
#
# **Observe and interpret:**
# - Which strategy requires the most training runs?
# - Why is leave-one-out evaluated on only a small subset?
# - What are the limitations of evaluating a single image per fold?

# %%
# @title 🔎 Compare Cross-Validation Fold Counts { display-mode: "form" }
K_FOLDS = 5
LOO_N = 12 if SMOKE else 40

def summarise_folds(name: str, splitter, X, y, groups=None) -> dict:
    """Summarize the number and size of validation folds."""
    sizes = [
        len(test_idx)
        for _, test_idx in splitter.split(X, y, groups)
    ]
    return {
        "Strategy": name,
        "Images": len(X),
        "Folds / training runs": len(sizes),
        "Validation images per fold": f"{min(sizes)}–{max(sizes)}",
    }

_loo_ids = _ids[:LOO_N]

cv_rows = [
    summarise_folds(
        "K-fold",
        KFold(K_FOLDS, shuffle=True, random_state=SEED),
        _ids, _y
    ),
    summarise_folds(
        "Stratified k-fold",
        StratifiedKFold(K_FOLDS, shuffle=True, random_state=SEED),
        _ids, _y
    ),
    summarise_folds(
        "Leave-one-out",
        LeaveOneOut(),
        _loo_ids, _y[:LOO_N]
    ),
]

display(pd.DataFrame(cv_rows))

# %%
# @title 💡 Sample Answer { display-mode: "form" }
from IPython.display import HTML, display

display(HTML("""
<details style="background:#F5F5F5; border:1px solid #9CA3AF;
                border-radius:6px; padding:12px 16px; color:#374151;">
<summary style="cursor:pointer;"><strong>💡 Sample Answer</strong></summary>
<ul>
<li><strong>Training runs:</strong> K-fold and stratified k-fold each require 5 training runs. Leave-one-out requires one training run per image (40 in this example, or 12 in smoke mode).</li>
<li><strong>Small subset:</strong> Leave-one-out is demonstrated on fewer images to limit computational cost. On the full dataset, it would require one training run for every image.</li>
<li><strong>Single-image validation:</strong> Each LOO fold produces a prediction for only one image. Individual fold accuracy is therefore either 0 or 1, and per-class performance cannot be meaningfully estimated from a single fold.</li>
</ul>
<p><strong>Key takeaway:</strong> More folds increase computational cost. Leave-one-out uses nearly all available images for training in each fold, but requires substantially more training runs than k-fold.</p>
</details>
"""))

# %% [markdown]
# ### Group-Based Splitting: The Trail-Camera Dataset
#
# The farm park has installed motion-triggered cameras in its woodland reserve. Each camera can capture several images of the same animal within a short period, producing bursts of closely related images.
#
# We use a subset of the [Caltech Camera Traps (CCT20) dataset](https://lila.science/datasets/caltech-camera-traps) containing images from eight cameras and five animal classes: opossum, raccoon, coyote, bobcat, and rabbit.
#
# The following cells load the dataset and display example image bursts. The first run may take a few minutes to download and resize the images.
#
# The farm park has installed motion-triggered cameras in its woodland reserve. Each camera can capture several images of the same animal within a short period. Such a sequence of closely related images is called a burst.

# %%
# @title Group-Based Splitting and Data Leakage { display-mode: "form" }
from IPython.display import HTML, display

display(HTML("""
<div style="background:#E3F2FD; border-left:4px solid #0D47A1; padding:14px 18px; border-radius:6px; color:#0D47A1;">

<strong>📘 Group-Based Splitting and Data Leakage</strong>

<p>
Images captured in the same burst are often very similar. If some appear in training and others in testing, the model may achieve overly optimistic test results. This is a form of <strong>data leakage</strong>.
</p>

<p>
Different splitting strategies address different goals:
</p>

<ul>
  <li><strong>Random splitting:</strong> Assigns images randomly to training and test sets.</li>
  <li><strong>Stratified splitting:</strong> Preserves approximately the same class proportions in both sets, but does not prevent related images from being split.</li>
  <li><strong>Group by burst:</strong> Keeps all images from the same burst together.</li>
  <li><strong>Group by camera:</strong> Keeps all images from the same camera together, allowing evaluation on unseen camera locations.</li>
</ul>

<p>
The <strong>sibling rate</strong> measures the percentage of test images that have at least one image from the same burst in the training set. A higher rate indicates greater potential for leakage.
</p>

</div>
"""))

# %%
# @title Prepare Trail-Camera Helpers { display-mode: "form" }
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


def cct_original_size(path: Path) -> tuple[int, int] | None:
    """The camera's (width, height), stored in the 512 px copy's JPEG comment; None for an older copy."""
    with Image.open(path) as im:
        found = re.match(rb"original (\d+)x(\d+)", im.info.get("comment", b""))
    return (int(found[1]), int(found[2])) if found else None


def fetch_cct_image(row: dict) -> tuple[str, Path | None]:
    """Download one trail-camera photo and store a 512 px copy; returns (file_name, path or None)."""
    dest = CCT_DIR / row["file_name"]
    if dest.exists() and dest.stat().st_size > 0 and cct_original_size(dest):
        return row["file_name"], dest
    raw = download_file(row["url"], CCT_DIR / "raw" / row["file_name"])
    if raw is None:
        return row["file_name"], None
    with Image.open(raw) as im:
        width, height = im.size  # the boxes in the manifest are in these pixels
        im.draft("RGB", (CCT_SIDE, CCT_SIDE))
        small = im.convert("RGB")
        small.thumbnail((CCT_SIDE, CCT_SIDE))
        small.save(dest, "JPEG", quality=90, comment=f"original {width}x{height}".encode())
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


def draw_cct_boxes(ax, row, thumb) -> None:
    """Draw the animal's box(es) from the manifest on a thumbnail, in the class colour (none when empty)."""
    original = cct_original_size(Path(row["path"]))
    if not isinstance(row["bbox"], str) or original is None:
        return
    scale = thumb.width / original[0]
    for x, y, w, h in json.loads(row["bbox"]):
        ax.add_patch(mpl.patches.Rectangle((x * scale, y * scale), w * scale, h * scale, fill=False, lw=2,
                                           edgecolor=CCT_COLORS[row["label"]]))


def show_bursts(bursts: list[pd.DataFrame], size: int = 256, show_boxes: bool = True) -> None:
    """One row per burst in `bursts` (one DataFrame each): its frames side by side, framed in the class colour,
    titled with the location."""
    cols = max(len(b) for b in bursts)
    fig, axes = plt.subplots(len(bursts), cols, figsize=(2.6 * cols, 2.1 * len(bursts)), squeeze=False)
    for r, burst in enumerate(bursts):
        for c in range(cols):
            ax = axes[r, c]
            if c < len(burst):
                row = burst.iloc[c]
                thumb = load_thumb(row["path"], (size, size))
                ax.imshow(thumb)
                if show_boxes:
                    draw_cct_boxes(ax, row, thumb)
                frame(ax, CCT_COLORS[row["label"]])
                ax.set_title(f"camera {row['location']} · {row['label']} · frame {row['frame_num']}",
                             fontsize=8)
            else:
                ax.axis("off")
    plt.tight_layout()
    plt.show()


# %%
# @title 🔎 Explore Trail-Camera Image Bursts { display-mode: "form" }

cct_all = pd.read_csv(fetch_manifest("cct20_slice.csv"))
cct_rows = select_cct_rows(cct_all)
cct = download_cct(cct_rows)

if len(cct) < len(cct_rows):
    print(f"  ⚠️ {len(cct_rows) - len(cct)} photos could not be downloaded; run again.")

CCT_LOCATIONS = sorted(cct["location"].unique().tolist())
print(f"{len(cct)} photos · {cct['seq_id'].nunique()} bursts · cameras {CCT_LOCATIONS}")
print("photos per class: " + " · ".join(f"{c} {int((cct['label'] == c).sum())}" for c in CCT_CLASSES))

# Prepare bursts for browsing
burst_list = [b for _, b in cct.groupby("seq_id", sort=True) if len(b) > 1]
burst_list.sort(key=lambda b: not b["bbox"].notna().all())

bursts_per_page = 2
burst_page = 0

burst_box_toggle = widgets.Checkbox(value=True, description="Show bounding boxes")
burst_prev_btn = widgets.Button(description="Previous", icon="chevron-left")
burst_next_btn = widgets.Button(description="Next", icon="chevron-right")
burst_counter = widgets.HTML()
burst_output = widgets.Output()


def burst_update_browser(change=None):
    start = burst_page * bursts_per_page
    selected = burst_list[start:start + bursts_per_page]

    burst_prev_btn.disabled = burst_page == 0
    burst_next_btn.disabled = start + bursts_per_page >= len(burst_list)
    burst_counter.value = f"Bursts {start + 1}–{start + len(selected)} of {len(burst_list)}"

    with burst_output:
        clear_output(wait=True)
        if selected:
            show_bursts(selected, show_boxes=burst_box_toggle.value)


def burst_previous_page(button):
    global burst_page
    burst_page = max(0, burst_page - 1)
    burst_update_browser()


def burst_next_page(button):
    global burst_page
    burst_page = min((len(burst_list) - 1) // bursts_per_page, burst_page + 1)
    burst_update_browser()


burst_prev_btn.on_click(burst_previous_page)
burst_next_btn.on_click(burst_next_page)
burst_box_toggle.observe(burst_update_browser, names="value")

burst_controls = widgets.HBox([burst_prev_btn, burst_next_btn, burst_counter])
display(widgets.VBox([burst_box_toggle, burst_controls, burst_output]))
burst_update_browser()

# %% [markdown]
# ### 🟢 Splitting Strategies: What Does Grouping Change?
#
# Different splitting strategies determine which images can appear together in training and test sets. The appropriate strategy depends on what the model should generalize to.
#
# **Before running the code:**
# - Which splitting strategies do you expect to prevent images from the same burst appearing in both training and test sets?
#
# **Explore:**
# - Switch between the four splitting strategies. Compare the train/test assignments, camera locations, and reported metrics.
# - Change the random seed. Observe which results change and which remain consistent.
#
# **Observe and interpret:**
# - Which strategies keep images from the same burst together?
# - Which strategies ensure that training and test sets contain different cameras?
# - Why does stratified splitting not necessarily prevent leakage?
# - Why might group-based splitting produce a test share different from the requested 20%?
#
# **Go Deeper (optional):**
# - Which splitting strategy would you choose for a model deployed at existing cameras versus new camera locations? Justify your choice.

# %%
# @title 🔎 Explore Leakage Across Splitting Strategies { display-mode: "form" }

split_strategy = widgets.Dropdown(
    options=["Random", "Stratified", "Group by burst", "Group by camera"],
    description="Strategy:"
)
split_seed = widgets.IntSlider(value=SEED, min=0, max=100, description="Seed:")
split_box_toggle = widgets.Checkbox(value=True, description="Bounding boxes")
split_output = widgets.Output()


def split_update(change=None):
    with split_output:
        clear_output(wait=True)

        method = split_strategy.value
        indices = np.arange(len(cct))

        if method in ["Random", "Stratified"]:
            train, test = train_test_split(
                indices, test_size=0.2, random_state=split_seed.value,
                stratify=cct["label"] if method == "Stratified" else None
            )
        else:
            groups = cct["seq_id"] if method == "Group by burst" else cct["location"]
            train, test = next(
                GroupShuffleSplit(test_size=0.2, random_state=split_seed.value)
                .split(indices, groups=groups)
            )

        assignment = np.full(len(cct), "Train", dtype=object)
        assignment[test] = "Test"

        # Select illustrative bursts from both splits
        bursts = [b for _, b in cct.groupby("seq_id") if len(b) > 1]
        mixed = [b for b in bursts if len(set(assignment[b.index])) > 1]
        intact_train = [b for b in bursts if set(assignment[b.index]) == {"Train"}]
        intact_test = [b for b in bursts if set(assignment[b.index]) == {"Test"}]

        examples = (mixed[:2] + intact_train[:1] + intact_test[:1])[:2]
        if not mixed:
            examples = (intact_train[:1] + intact_test[:1])[:2]

        # Metrics for the entire split
        train_data, test_data = cct.iloc[train], cct.iloc[test]
        sibling_rate = 100 * test_data["seq_id"].isin(train_data["seq_id"]).mean()
        shared_cameras = len(
            set(train_data["location"]) & set(test_data["location"])
        )

        print(
            f"Test images: {len(test)} ({100 * len(test) / len(cct):.1f}%)  ·  "
            f"Sibling rate: {sibling_rate:.1f}%  ·  "
            f"Shared cameras: {shared_cameras}"
        )

        # Display real images with train/test borders
        if examples:
            cols = max(len(b) for b in examples)
            fig, axes = plt.subplots(
                len(examples), cols,
                figsize=(3 * cols, 2.7 * len(examples)),
                squeeze=False
            )

            for r, burst in enumerate(examples):
                for c in range(cols):
                    ax = axes[r, c]
                    if c >= len(burst):
                        ax.axis("off")
                        continue

                    row = burst.iloc[c]
                    part = assignment[burst.index[c]]
                    thumb = load_thumb(row["path"], (256, 256))

                    ax.imshow(thumb)
                    if split_box_toggle.value:
                        draw_cct_boxes(ax, row, thumb)

                    frame(ax, "#4C78A8" if part == "Train" else "#E76F51")
                    ax.set_title(
                        f"Camera {row['location']} · Frame {row['frame_num']} · {part}",
                        fontsize=9
                    )

            fig.suptitle(f"Train/test assignments — {method}")
            plt.tight_layout()
            plt.show()


split_strategy.observe(split_update, names="value")
split_seed.observe(split_update, names="value")
split_box_toggle.observe(split_update, names="value")

display(widgets.HBox([split_strategy, split_seed, split_box_toggle]), split_output)
split_update()

# %%
# @title 💡 Sample Answer { display-mode: "form" }
from IPython.display import HTML, display

display(HTML("""
<details style="background:#F5F5F5; border:1px solid #9CA3AF; border-radius:6px; padding:12px 16px; color:#374151;">
<summary><strong>💡 Sample Answer</strong></summary>

<ul>
  <li><strong>Random splitting:</strong> Images from the same burst can appear in both training and test sets, potentially leading to overly optimistic results.</li>
  <li><strong>Stratified splitting:</strong> Preserves class proportions but does not prevent burst-level leakage.</li>
  <li><strong>Group by burst:</strong> Keeps related images together, resulting in a sibling rate of 0%.</li>
  <li><strong>Group by camera:</strong> Separates camera locations and also prevents burst-level leakage. The number of shared cameras is 0.</li>
</ul>

<p>
Group-based splitting may produce a test share different from 20% because entire groups, rather than individual images, are assigned together.
</p>

<p>
<strong>Choosing a strategy:</strong> Grouping by burst is appropriate for evaluating new observations at existing cameras. Grouping by camera provides a more realistic evaluation when deploying a model at new camera locations.
</p>

</details>
"""))

# %%
# @title 🔴 Splitting for Object Detection and Segmentation { display-mode: "form" }

question = widgets.HTML(
    "<b>Which statements about splitting datasets for object detection "
    "and semantic segmentation are correct? Select all that apply.</b>"
)

options = [
    "A. Bounding boxes from the same image can be assigned to different splits.",
    "B. Stratification is more challenging in object detection because images may contain multiple classes.",
    "C. Individual pixels can be randomly assigned to training and test sets in semantic segmentation.",
    "D. Related images should be kept together in both tasks to avoid data leakage."
]

answers = [
    widgets.Checkbox(
        description=option,
        indent=False,
        layout=widgets.Layout(width="100%")
    )
    for option in options
]

check = widgets.Button(description="Check Answer")
solution = widgets.Button(description="Show Solution")
feedback = widgets.HTML()


def check_answer(button):
    selected = {i for i, answer in enumerate(answers) if answer.value}

    if selected == {1, 3}:
        feedback.value = (
            "<span style='color:green'>✅ Correct!</span> "
            "B and D are the correct answers."
        )
    else:
        feedback.value = (
            "<span style='color:#C2410C'>Not quite. Try again!</span><br>"
            "Think about which data units must stay together during splitting."
        )


def show_solution(button):
    feedback.value = (
        "<b>Correct answers: B and D</b><br><br>"
        "<b>A — False:</b> Bounding boxes from the same image must stay together.<br>"
        "<b>B — True:</b> Images may contain multiple classes, making stratification more complex.<br>"
        "<b>C — False:</b> Pixels from the same image must not be split independently.<br>"
        "<b>D — True:</b> Related images should stay together to prevent data leakage."
    )


check.on_click(check_answer)
solution.on_click(show_solution)

display(
    question,
    *answers,
    widgets.HBox([check, solution]),
    feedback
)
