# %% [markdown] edition="teaching"
# ## Object Detection
#
# In image classification, a model predicts what is present in an image. But what if the park wants to count individual animals and determine where they are?
#
# Object detection extends classification by predicting a class label and a bounding box for each object. This allows us to investigate not only which animals appear, but also how many there are, where they are located, and how large they appear.
#
# In this section, we explore how object size, position, crowding, and class imbalance affect an object detection dataset. We also examine the additional annotation effort required compared with image classification.
#

# %% [markdown] edition="own"
# ## Object Detection
#
# In image classification, a model predicts what is present in an image. But what if you need to count individual objects and know where they are?
#
# This chapter runs only if `TASKS` includes `"detection"`, so its cells skip themselves when your labels have no boxes.
#
# Object detection extends classification by predicting a class label and a bounding box for each object. This allows us to investigate not only which objects appear, but also how many there are, where they are located, and how large they appear.
#
# In this section, we explore how object size, position, crowding, and class imbalance affect your object detection dataset. We also examine the additional annotation effort required compared with image classification.

# %% [markdown] edition="teaching"
# **From Labels to Bounding Boxes**
#
# Object detection datasets contain one annotation per object rather than one label per image. Each annotation specifies the object's class and bounding box.
#
# In our dataset, bounding boxes use the COCO format `[x, y, w, h]`, where `x` and `y` indicate the top-left corner and `w` and `h` specify the width and height in pixels.
#
# Unlike image classification, an image can contain multiple annotated objects, potentially belonging to different classes.
#
# We explore four characteristics of the object detection dataset:
#
# - **Class imbalance:** How are object instances distributed across classes?
# - **Object size:** How large or small are the annotated objects?
# - **Object position:** Where are objects located within images?
# - **Crowding:** How many objects appear together, and how much do they overlap?

# %% [markdown] edition="own"
# **From Labels to Bounding Boxes**
#
# Object detection datasets contain one annotation per object, not one label per image. Each annotation specifies the object's class and bounding box.
#
# When you loaded your dataset, the notebook converted your boxes to the COCO format `[x, y, w, h]`, whatever format you stored them in. Here `x` and `y` indicate the top-left corner and `w` and `h` specify the width and height in pixels.
#
# Unlike image classification, an image can contain multiple annotated objects, potentially belonging to different classes.
#
# We explore four characteristics of the object detection dataset:
#
# - **Class imbalance:** How are object instances distributed across classes?
# - **Object size:** How large or small are the annotated objects?
# - **Object position:** Where are objects located within images?
# - **Crowding:** How many objects appear together, and how much do they overlap?

# %% [markdown]
# ### 🟢 Exploring Bounding Boxes: What Do You Notice?
#
# **Before running the code:**
# - How might the number and size of annotated objects differ between images?
#
# **Explore:**
# - Examine the four example images and their bounding boxes.
#
# **Observe and interpret:**
# - How many objects are annotated in each image?
# - How do the object sizes vary, and why might this matter for object detection?

# %%
# @title Helpers: Bounding Boxes { display-mode: "form" }
def box_area_bucket(w: float, h: float) -> str:
    """COCO's size buckets from a box's width and height in pixels: small < 32², medium < 96², else large."""
    area = w * h
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def show_boxes(rows_df: pd.DataFrame, boxes_df: pd.DataFrame, title: str, caption=None, cols: int = 4,
               size: int = 384) -> None:
    """Gallery of the photos in `rows_df` with their boxes from `boxes_df` drawn in the class color.

    `caption(row) -> str` writes the title above each photo; by default the class and the number of boxes.
    """
    n = len(rows_df)
    if n == 0:
        print(f"{title}: no images to show.")
        return
    cols = min(cols, n)
    n_rows = -(-n // cols)
    fig, axes = plt.subplots(n_rows, cols, figsize=(3.4 * cols, 3.2 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), rows_df.iterrows(), strict=False):
        thumb = load_thumb(row["path"], (size, size))
        scale = thumb.width / row["width"]
        ax.axis("on")
        ax.imshow(thumb)
        own = boxes_df[boxes_df["image_id"] == row["id"]]
        for _, b in own.iterrows():
            x, y, w, h = b["bbox"]
            ax.add_patch(mpl.patches.Rectangle((x * scale, y * scale), w * scale, h * scale, fill=False,
                                               lw=2, edgecolor=CLASS_COLORS[b["class"]]))
        frame(ax, CLASS_COLORS[row["class"]])
        text = caption(row) if caption else f"{row['class']} · {len(own)} boxes"
        ax.set_title(text, fontsize=9)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    display(fig)
    plt.close(fig)


boxes = annotations.copy()  # one row per box; a copy, because later cells add columns to it


# %%
# @title 🔎 Explore Bounding Boxes { display-mode: "form" }
# needs: boxes

boxview_page = 0
boxview_output = widgets.Output()

boxview_previous = widgets.Button(description="← Previous")
boxview_next_button = widgets.Button(description="Next →")
boxview_counter = widgets.HTML()


def boxview_show_page():
    with boxview_output:
        clear_output(wait=True)

        start = boxview_page * PAGE_SIZE
        selected = images.iloc[start:start + PAGE_SIZE]

        show_boxes(
            selected,
            boxes,
            "Bounding boxes: one annotation per animal",
            caption=lambda r: r["class"]
        )

        total_pages = (len(images) + PAGE_SIZE - 1) // PAGE_SIZE
        boxview_counter.value = f"Page {boxview_page + 1} of {total_pages}"
        boxview_previous.disabled = boxview_page == 0
        boxview_next_button.disabled = boxview_page >= total_pages - 1


def boxview_change_page(step):
    global boxview_page
    boxview_page += step
    boxview_show_page()


boxview_previous.on_click(lambda b: boxview_change_page(-1))
boxview_next_button.on_click(lambda b: boxview_change_page(1))

display(widgets.HBox([boxview_previous, boxview_next_button, boxview_counter]), boxview_output)
boxview_show_page()

# %% [markdown] edition="teaching"
# ### 🟢 Boxes per Class vs. Photos per Class
#
# In image classification, class distributions are typically counted by images. In object detection, we can also count individual objects. For example, one image containing 12 sheep contributes one image but 12 bounding boxes.
#
# **Before running the code:**
# - Do you expect the class ranking to change when counting bounding boxes instead of images? Which class might increase the most?
#
# **Coding Exercise:**
#
# In this and subsequent coding exercises, you will implement small functions to explore the dataset. Complete the provided code cells and check your results. If you get stuck, you can reveal the reference solution. The notebook continues to work even if your implementation is incomplete.
#
# - Implement `count_boxes_per_class(annotations)` to return the number of bounding boxes per class.
# - Run the validation and compare the resulting distribution with the image-level distribution from Class Distribution: Are All Animal Classes Equally Represented?
#
# **Observe and interpret:**
# - Which classes change most in their relative frequency?
# - Why might image-level and object-level class imbalance lead to different conclusions?

# %% [markdown] edition="own"
# ### 🟢 Boxes per Class vs. Photos per Class
#
# In image classification, class distributions are typically counted by images. In object detection, we can also count individual objects. For example, one image containing 12 objects contributes one image but 12 bounding boxes.
#
# **Before running the code:**
# - Do you expect the class ranking to change when counting bounding boxes instead of images? Which class might increase the most?
#
# **Coding Exercise:**
#
# In this and subsequent coding exercises, you will implement small functions to explore the dataset. Complete the provided code cells and check your results. If you get stuck, you can reveal the reference solution. The notebook continues to work even if your implementation is incomplete.
#
# - Implement `count_boxes_per_class(annotations)` to return the number of bounding boxes per class.
# - Run the validation and compare the resulting distribution with the image-level distribution from Class Distribution: Are All Classes Equally Represented?
#
# **Observe and interpret:**
# - Which classes change most in their relative frequency?
# - Why might image-level and object-level class imbalance lead to different conclusions?

# %% [markdown]
# **✏️ Your turn · `count_boxes_per_class`**
#
# Write the function in the next cell: given an annotations table (one row per animal, with a `class`
# column), return a dict `{class: number of boxes}` with **every** class in `CLASSES` as a key, counts as
# plain `int`s. `use()` checks it against the reference and continues with the reference either way.


# %%
# @title Reference solution — run this cell; open it only if you are stuck { display-mode: "form" }
def _ref_count_boxes_per_class(annotations):
    counts = annotations["class"].value_counts()
    return {c: int(counts.get(c, 0)) for c in CLASSES}


# %%
# ✍️ YOUR CODE
# needs: boxes
def count_boxes_per_class(annotations):
    """Number of boxes (rows) per class.

    Inputs: `annotations`, a DataFrame with one row per animal and a `class` column.
    Output: a dict {class: int} with every class of CLASSES as a key (0 when a class has no box).

    e.g. {"cat": 12, "dog": 9, "bird": 30, "horse": 7, "sheep": 25, "zebra": 14}
    """
    ...


boxes_per_class = use(count_boxes_per_class, boxes, reference=_ref_count_boxes_per_class)

# %%
# @title 🔎 Compare Class Distributions: Photos vs. Boxes { display-mode: "form" }
# needs: boxes

RESULTS["boxes_per_class"] = boxes_per_class
RESULTS["n_boxes"] = int(len(boxes))

by_photos = sorted(CLASSES, key=lambda c: -images_per_class[c])
by_boxes = sorted(CLASSES, key=lambda c: -boxes_per_class[c])

print(f"{len(images)} images · {len(boxes)} boxes")
print("Ranked by images: " + " > ".join(f"{c} {images_per_class[c]}" for c in by_photos))
print("Ranked by boxes:  " + " > ".join(f"{c} {boxes_per_class[c]}" for c in by_boxes))

for c in CLASSES:
    move = by_photos.index(c) - by_boxes.index(c)
    if move:
        print(
            f"  {c}: rank {by_photos.index(c) + 1} by images → "
            f"rank {by_boxes.index(c) + 1} by boxes "
            f"({'↑' if move > 0 else '↓'}{abs(move)})"
        )

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

for ax, counts, order, what in zip(
    axes,
    [images_per_class, boxes_per_class],
    [by_photos, by_boxes],
    ["Images", "Boxes"],
    strict=True
):
    ax.bar(order, [counts[c] for c in order], color=[CLASS_COLORS[c] for c in order])
    ax.set_title(f"{what} per class (ranked)")
    ax.set_ylabel(what)

    for label in ax.get_xticklabels():
        label.set_color(CLASS_COLORS[label.get_text()])

fig.tight_layout()
display(fig)
plt.close(fig)

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p>
The class ranking changes because some images contain many animals. Bird and sheep move up when counting bounding boxes, while cat and dog move down because they typically appear individually.
</p>

<p>
Image-level counts describe how frequently classes appear in photos, whereas box-level counts describe how many object instances are annotated. Therefore, class imbalance can look very different depending on which unit is counted.
</p>
""")


# %% [markdown] edition="own"
# ✍️ **Your answer** (double-click to edit):
#
# *In the park:* bird and sheep moved up when counting boxes, and cat and dog moved down, because cats and dogs usually appear alone.

# %% [markdown]
# ### 🟢 Object Sizes: How Small Are the Animals?
#
# Bounding-box area is calculated as width × height in pixels. COCO distinguishes three object-size categories:
#
# - **Small:** area < 32² pixels
# - **Medium:** 32² ≤ area < 96² pixels
# - **Large:** area ≥ 96² pixels
#
# To compare differently shaped boxes, we use the square root of their area, `√(w × h)`, representing the side length of a square with the same area.
#
# **Before running the code:**
# - Which animal class do you expect to have the smallest bounding boxes?
#
# **Explore:**
# - Compare the box-size distributions across classes.
# - Use Previous / Next to explore bounding boxes of different sizes and their enlarged crops. The selected box has a solid outline; other annotated objects have dotted outlines.
#
# **Observe and interpret:**
# - Which classes contain the smallest objects?
# - How much do object sizes vary within the same class?
# - Why might small objects be particularly challenging for object detection?
# - What do the enlarged crops reveal about the visual information available for detecting small objects?

# %%
# @title 🔎 Explore Object Size Distributions { display-mode: "form" }
# needs: boxes

boxes["w"] = boxes["bbox"].map(lambda b: b[2])
boxes["h"] = boxes["bbox"].map(lambda b: b[3])
boxes["box_area"] = boxes["w"] * boxes["h"]
boxes["side"] = np.sqrt(boxes["box_area"])
boxes["bucket"] = [
    box_area_bucket(w, h)
    for w, h in zip(boxes["w"], boxes["h"], strict=True)
]

bucket_share = (
    boxes.groupby("class")["bucket"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reindex(index=CLASSES, columns=["small", "medium", "large"], fill_value=0)
)

RESULTS["small_box_share"] = {
    c: float(bucket_share.loc[c, "small"]) for c in CLASSES
}

print("Share of each class's boxes that are small / medium / large (%):")
display((bucket_share * 100).round(0).astype(int).T)

fig, ax = plt.subplots(figsize=(8, 3.8))

sides = [
    boxes.loc[boxes["class"] == c, "side"].clip(lower=4).to_numpy()
    for c in CLASSES
]

# Use log10 to display a wide range of object sizes
parts = ax.violinplot(
    [np.log10(v) for v in sides],
    showmedians=True, showextrema=False, widths=0.8
)

for body, c in zip(parts["bodies"], CLASSES, strict=True):
    body.set_facecolor(CLASS_COLORS[c])
    body.set_alpha(0.75)

parts["cmedians"].set_color(INK)
parts["cmedians"].set_linewidth(2.5)

tick_names = [
    f"{c}\nmedian {np.median(v):.0f} px"
    for c, v in zip(CLASSES, sides, strict=True)
]

ax.set_xticks(range(1, len(CLASSES) + 1), tick_names, fontsize=8)

for label, c in zip(ax.get_xticklabels(), CLASSES, strict=True):
    label.set_color(CLASS_COLORS[c])

ax.set_yticks(
    np.log10([4, 10, 32, 100, 300, 1000]),
    ["4", "10", "32", "100", "300", "1000"]
)

for side_px, name in [(32, "small ↓"), (96, "large ↑")]:
    ax.axhline(np.log10(side_px), color=INK, linestyle="--", linewidth=1)
    ax.text(
        0.5, np.log10(side_px),
        f" {side_px} px · {name}",
        va="bottom", fontsize=8, color=INK
    )

ax.set_ylabel("Box side √(w·h), px (log)")
ax.set_title("Box size per class (bar = median)")

fig.tight_layout()
display(fig)
plt.close(fig)

# %%
# @title 🔎 Explore Bounding Box Sizes in Images { display-mode: "form" }
# needs: boxes

by_area = boxes.sort_values(["box_area", "id"]).reset_index(drop=True)

# Start with the smallest, median, and largest boxes
first_page = [
    by_area.iloc[0],
    by_area.iloc[len(by_area) // 2],
    by_area.iloc[-1]
]

# Additional examples from each COCO size category
examples = []
for bucket in ["small", "medium", "large"]:
    subset = by_area[by_area["bucket"] == bucket]
    if not subset.empty:
        positions = np.linspace(0, len(subset) - 1, min(12, len(subset)), dtype=int)
        examples.extend(subset.iloc[positions].to_dict("records"))

# Remove duplicates while preserving the initial examples
seen = set()
selected_boxes = []
for b in [*first_page, *examples]:
    if b["id"] not in seen:
        selected_boxes.append(b)
        seen.add(b["id"])

size_pages = [
    selected_boxes[i:i + 3]
    for i in range(0, len(selected_boxes), 3)
]

size_page = 0
size_previous = widgets.Button(description="← Previous")
size_next_button = widgets.Button(description="Next →")
size_counter = widgets.HTML()
size_output = widgets.Output()


def size_show_page():
    with size_output:
        clear_output(wait=True)

        selected = size_pages[size_page]
        fig, axes = plt.subplots(2, len(selected), figsize=(3.7 * len(selected), 7), squeeze=False)

        for col, b in enumerate(selected):
            photo = images_by_id.loc[b["image_id"]]
            thumb = load_thumb(photo["path"], (384, 384))

            scale_x = thumb.width / photo["width"]
            scale_y = thumb.height / photo["height"]

            ax = axes[0, col]
            ax.imshow(thumb)

            for _, obj in boxes[boxes["image_id"] == b["image_id"]].iterrows():
                x, y, w, h = obj["bbox"]
                ax.add_patch(mpl.patches.Rectangle(
                    (x * scale_x, y * scale_y),
                    w * scale_x, h * scale_y,
                    fill=False,
                    lw=3 if obj["id"] == b["id"] else 1,
                    edgecolor=CLASS_COLORS[obj["class"]],
                    ls="-" if obj["id"] == b["id"] else ":"
                ))

            frame(ax, CLASS_COLORS[b["class"]])
            ax.set_title(
                f"{b['class']}: {b['w']:.0f}×{b['h']:.0f} px\n"
                f"{b['box_area']:,.0f} px² ({b['bucket']})",
                fontsize=9
            )

            # Enlarged crop around the selected box
            x, y, w, h = b["bbox"]
            pad = 0.25 * max(w, h) + 4
            x0, y0 = max(0, x - pad), max(0, y - pad)

            with Image.open(photo["path"]) as img:
                crop = img.convert("RGB").crop((
                    int(x0), int(y0),
                    int(min(img.width, x + w + pad)),
                    int(min(img.height, y + h + pad))
                ))

            ax = axes[1, col]
            ax.imshow(crop, interpolation="nearest")
            ax.add_patch(mpl.patches.Rectangle(
                (x - int(x0), y - int(y0)),
                w, h, fill=False, lw=2,
                edgecolor=CLASS_COLORS[b["class"]]
            ))
            frame(ax, CLASS_COLORS[b["class"]])
            ax.set_title("Bounding-box crop (enlarged)", fontsize=9)

        fig.tight_layout()
        plt.show()

        size_counter.value = f"Page {size_page + 1} of {len(size_pages)}"
        size_previous.disabled = size_page == 0
        size_next_button.disabled = size_page == len(size_pages) - 1


def size_change_page(step):
    global size_page
    size_page += step
    size_show_page()


size_previous.on_click(lambda b: size_change_page(-1))
size_next_button.on_click(lambda b: size_change_page(1))

display(widgets.HBox([size_previous, size_next_button, size_counter]), size_output)
size_show_page()

# %% edition="teaching"
# @title 💡 Sample Answer: Object Sizes { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p>
Birds have the smallest bounding boxes by a wide margin, while cats, dogs, and horses are mostly represented by large boxes. Object sizes can vary considerably within a class, depending on the animal's distance from the camera.
</p>

<p>
Small objects are particularly challenging because they contain fewer pixels and less visual detail. This may make distant birds difficult to detect reliably.
</p>
""")


# %% [markdown] edition="own"
# ✍️ **Your answer** (double-click to edit):
#
# *In the park:* birds had the smallest boxes by a wide margin, while cats, dogs, and horses were mostly large.

# %% [markdown]
# ### 🟢 Object Positions: Where Are the Animals?
#
# A bounding box's center is `(x + w/2, y + h/2)`. Dividing these coordinates by the image width and height gives normalized positions between 0 and 1, allowing us to compare images of different sizes.
#
# A **heatmap** shows where bounding-box centers occur most frequently. Darker regions indicate more objects.
#
# **Before running the code:**
# - Do you expect animals to appear equally often across all image positions?
#
# **Explore:**
# - Compare the box-center heatmaps across classes.
# - Look for differences in how concentrated or spread out the positions are.
#
# **Observe and interpret:**
# - Where are most animals located within the images?
# - Which classes have more widely distributed positions?
# - Why might a detector trained on these images struggle with animals near the image edges?

# %%
# @title 🔎 Explore Object Positions Across Classes { display-mode: "form" }
# needs: boxes
# Box centers on the unit square, one heatmap per class in its own color (white = no boxes there).
_img_w = boxes["image_id"].map(images_by_id["width"])
_img_h = boxes["image_id"].map(images_by_id["height"])
boxes["cx"] = (boxes["bbox"].map(lambda b: b[0] + b[2] / 2) / _img_w).clip(0, 1)
boxes["cy"] = (boxes["bbox"].map(lambda b: b[1] + b[3] / 2) / _img_h).clip(0, 1)
HEAT_BINS = 6 if SMOKE else 12  # coarser cells with few boxes (N-014)
_edge_share = float(((boxes["cx"] < 0.2) | (boxes["cx"] > 0.8) | (boxes["cy"] < 0.2) | (boxes["cy"] > 0.8))
                    .mean())
RESULTS["edge_box_share"] = _edge_share

_ncols = min(6, len(CLASSES))  # at most six classes per row
_nrows = -(-len(CLASSES) // _ncols)
fig, axes = plt.subplots(_nrows, _ncols, figsize=(2.3 * _ncols, 2.8 * _nrows), squeeze=False)
for ax in axes.ravel()[len(CLASSES):]:
    ax.axis("off")
for ax, c in zip(axes.ravel(), CLASSES, strict=False):
    own = boxes[boxes["class"] == c]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(c, ["white", CLASS_COLORS[c]])
    heat, _, _ = np.histogram2d(own["cy"], own["cx"], bins=HEAT_BINS, range=[[0, 1], [0, 1]])
    ax.imshow(heat, cmap=cmap, extent=[0, 1, 1, 0], vmin=0)
    ax.set_title(f"{c} · {len(own)} boxes", fontsize=9, color=CLASS_COLORS[c])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=7)
    ax.grid(False)
for _row in axes:
    _row[0].set_ylabel("box center y (0 = top)")
fig.suptitle(f"Where the box centers are · {_edge_share:.0%} of all boxes are in the outer 20 % of the frame",
             color=INK)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% edition="teaching"
# @title 💡 Sample Answer: Object Positions { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p>
Bounding-box centers are concentrated near the middle of the images, suggesting
that animals are often photographed centrally. Birds and sheep show a wider
spatial distribution, possibly because multiple animals appear together.
</p>
<p>
A detector trained mainly on centrally positioned animals may perform less
reliably near image edges, where fewer training examples are available.
However, the heatmaps alone do not establish how well the detector performs.
</p>
""")


# %% [markdown] edition="own"
# ✍️ **Your answer** (double-click to edit):
#
# *In the park:* box centers clustered near the middle of the images, and birds and sheep were spread more widely.

# %% [markdown]
# ### 🟢 Crowding: How Many Animals Appear Together?
#
# Some images contain a single animal, while others show several animals close together. Crowded scenes can be challenging for object detection, particularly when animals partially hide one another.
#
# **Before running the code:**
# - Which animal classes do you expect to appear most often in images with many animals?
#
# **Explore:**
# - Compare the distributions of object counts across classes.
# - Use the image browser to compare images with one, a few, and many annotated objects.
#
# **Observe and interpret:**
# - Which classes tend to have more objects per image?
# - How do images with few objects differ from those with many objects?
# - Does a higher object count always mean a more crowded scene? Why or why not?
# - What challenges might these differences pose for object detection?

# %%
# @title 🔎 Explore Object Counts Across Classes { display-mode: "form" }
# needs: boxes

# Annotated objects per image (the images table already counts them)
crowding = images[["id", "class", "n_objects"]]

RESULTS["max_objects_per_image"] = int(crowding["n_objects"].max())

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# Distribution of object counts
count_distribution = crowding["n_objects"].value_counts().sort_index()

axes[0].bar(
    count_distribution.index,
    count_distribution.values,
    color=LIGHT_GRAY
)
axes[0].set_xlabel("Number of objects per image")
axes[0].set_ylabel("Number of images")
axes[0].set_title("Objects per image")

# Object counts by class
data = [
    crowding.loc[crowding["class"] == c, "n_objects"].to_numpy()
    for c in CLASSES
]

bp = axes[1].boxplot(
    data,
    tick_labels=CLASSES,
    showfliers=False,
    patch_artist=True
)

for box, c in zip(bp["boxes"], CLASSES, strict=True):
    box.set_facecolor(CLASS_COLORS[c])
    box.set_alpha(0.75)

for median in bp["medians"]:
    median.set_color(INK)
    median.set_linewidth(2)

for label, c in zip(axes[1].get_xticklabels(), CLASSES, strict=True):
    label.set_color(CLASS_COLORS[c])

axes[1].set_ylabel("Number of objects per image")
axes[1].set_title("Object counts by class")

fig.tight_layout()
plt.show()

# %%
# @title 🔎 Explore Crowded Images { display-mode: "form" }
# needs: boxes

# Sort images from most to fewest annotated objects
crowded_images = crowding.sort_values(
    ["n_objects", "id"], ascending=[False, True]
).reset_index(drop=True)

# Select object count (group counts of 10 or more)
available_counts = sorted(
    int(n) for n in crowding["n_objects"].unique() if n < 10
)

crowd_count_selector = widgets.Dropdown(
    options=(
        [("All", None)]
        + [(str(n), n) for n in available_counts]
        + ([("10 or more", "10+")] if (crowding["n_objects"] >= 10).any() else [])
    ),
    value=None,
    description="Objects:",
    style={"description_width": "initial"}
)

crowd_page = 0

crowd_first = widgets.Button(description="⇤ First")
crowd_previous = widgets.Button(description="← Previous")
crowd_next_button = widgets.Button(description="Next →")
crowd_last = widgets.Button(description="Last ⇥")
crowd_counter = widgets.HTML()
crowd_output = widgets.Output()


def crowd_filtered_images():
    if crowd_count_selector.value is None:
        return crowded_images
    if crowd_count_selector.value == "10+":
        return crowded_images[crowded_images["n_objects"] >= 10]
    return crowded_images[
        crowded_images["n_objects"] == crowd_count_selector.value
    ]


def crowd_show_page():
    filtered = crowd_filtered_images()
    total_pages = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)

    with crowd_output:
        clear_output(wait=True)

        selected = filtered.iloc[
            crowd_page * PAGE_SIZE:(crowd_page + 1) * PAGE_SIZE
        ]

        # Preserve the selected order
        selected_images = selected[["id"]].merge(
            images, on="id", how="left", sort=False
        )

        show_boxes(
            selected_images,
            boxes,
            "Images ordered by number of annotated objects",
            caption=lambda r: f"{r['class']} · {r['n_objects']} objects"
        )

    count_label = (
        "All counts" if crowd_count_selector.value is None
        else "10+ objects" if crowd_count_selector.value == "10+"
        else f"{crowd_count_selector.value} objects"
    )

    crowd_counter.value = (
        f"{count_label} · {len(filtered)} images · "
        f"Page {crowd_page + 1} of {total_pages}"
    )

    crowd_first.disabled = crowd_previous.disabled = crowd_page == 0
    crowd_next_button.disabled = crowd_last.disabled = crowd_page >= total_pages - 1


def crowd_go_to_page(new_page):
    global crowd_page
    total_pages = max(
        1, (len(crowd_filtered_images()) + PAGE_SIZE - 1) // PAGE_SIZE
    )
    crowd_page = max(0, min(new_page, total_pages - 1))
    crowd_show_page()


def crowd_change_count(change):
    if change["name"] == "value":
        crowd_go_to_page(0)


crowd_count_selector.observe(crowd_change_count, names="value")

crowd_first.on_click(lambda b: crowd_go_to_page(0))
crowd_previous.on_click(lambda b: crowd_go_to_page(crowd_page - 1))
crowd_next_button.on_click(lambda b: crowd_go_to_page(crowd_page + 1))
crowd_last.on_click(lambda b: crowd_go_to_page(10**9))

display(
    widgets.VBox([
        crowd_count_selector,
        widgets.HBox([crowd_first, crowd_previous, crowd_next_button, crowd_last, crowd_counter])
    ]),
    crowd_output
)

crowd_show_page()

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p>Images of herd animals, such as sheep or zebras, may contain more annotated objects than images of cats or dogs. Images with many objects often show animals close together, but some animals may still be clearly separated.</p>
<p>A higher object count does not necessarily mean more crowding. Object size, spacing, and occlusion also matter. Closely packed animals can be difficult to detect individually, increasing the risk of missed or duplicate detections.</p>
""")


# %% [markdown] edition="own"
# ✍️ **Your answer** (double-click to edit):
#
# *In the park:* herd animals such as sheep and zebras had more boxes per image than cats or dogs, but a high count did not always mean the animals were packed together.

# %% [markdown]
# ### 🔴 Bounding Box Overlap: Understanding IoU (Optional)
#
# Counting objects tells us how many animals are annotated, but not how closely they appear together. We can measure bounding-box overlap using **Intersection over Union (IoU)**:
#
# $$
# \mathrm{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}}
# $$
#
# IoU ranges from 0 (no overlap) to 1 (identical boxes).
#
# **Before running the code:**
# - What happens to IoU when two boxes move closer together?
#
# **Explore:**
# - Move the X and Y sliders to change how two bounding boxes overlap.
# - Observe how IoU changes and when it exceeds the threshold of 0.1.
#
# **Observe and interpret:**
# - How does IoU change as the boxes move apart?
# - Why is bounding-box overlap only an approximation of how crowded a scene is?

# %%
# @title Helpers: IoU { display-mode: "form" }
OVERLAP_IOU = 0.1  # two boxes count as overlapping above this IoU


def box_iou_dice(a: tuple, b: tuple) -> tuple[float, float]:
    """IoU and Dice of two boxes given as (x, y, w, h)."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a, area_b = a[2] * a[3], b[2] * b[3]
    return inter / (area_a + area_b - inter), 2 * inter / (area_a + area_b)


# %%
# @title 🔎 Explore Bounding Box Overlap (IoU) { display-mode: "form" }


REFERENCE_BOX = (20, 20, 60, 40)


def show_overlap(x_shift, y_shift):
    b = (20 + x_shift, 20 + y_shift, 60, 40)

    iou = box_iou_dice(REFERENCE_BOX, b)[0]

    ix0 = max(REFERENCE_BOX[0], b[0])
    iy0 = max(REFERENCE_BOX[1], b[1])
    iw = max(0, min(REFERENCE_BOX[0] + REFERENCE_BOX[2], b[0] + b[2]) - ix0)
    ih = max(0, min(REFERENCE_BOX[1] + REFERENCE_BOX[3], b[1] + b[3]) - iy0)

    intersection = iw * ih
    union = REFERENCE_BOX[2] * REFERENCE_BOX[3] + b[2] * b[3] - intersection

    fig, ax = plt.subplots(figsize=(7, 3.5))

    for box in [REFERENCE_BOX, b]:
        ax.add_patch(mpl.patches.Rectangle(
            box[:2], *box[2:],
            facecolor=LIGHT_GRAY,
            edgecolor=EDGE_GRAY,
            alpha=0.45,
            linewidth=1.5
        ))

    if intersection > 0:
        ax.add_patch(mpl.patches.Rectangle(
            (ix0, iy0), iw, ih,
            facecolor=HIGHLIGHT,
            edgecolor=INK,
            linewidth=1.5
        ))

    ax.set_xlim(10, 160)
    ax.set_ylim(120, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        f"IoU = {intersection} / {union} = {iou:.2f} "
        f"({'overlapping' if iou > OVERLAP_IOU else 'not counted'})",
        fontsize=11
    )

    plt.show()
    plt.close(fig)


x_slider = widgets.IntSlider(
    value=25, min=0, max=90, step=5,
    description="X shift:",
    continuous_update=False
)

y_slider = widgets.IntSlider(
    value=10, min=0, max=70, step=5,
    description="Y shift:",
    continuous_update=False
)

overlap_output = widgets.interactive_output(
    show_overlap,
    {"x_shift": x_slider, "y_shift": y_slider}
)

display(widgets.VBox([x_slider, y_slider]), overlap_output)

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p>IoU increases as the boxes overlap more strongly, reaching 1 when they are identical. When the boxes move apart, IoU decreases and becomes 0 when they no longer intersect.</p>
<p>Bounding-box overlap is only an approximation of crowding. Boxes can overlap even when the animals themselves do not, and animals can partially occlude one another without their boxes having a high IoU.</p>
""")


# %% [markdown] edition="own"
# ✍️ **Your answer** (double-click to edit):
#
# *Look for:* the slider position where IoU drops to 0, and think of two boxes that overlap although the objects inside them do not touch.

# %% [markdown]
# ### 🔴 Crowding: Bounding Box Overlap Across Classes (Optional)
#
# Images with many animals do not necessarily contain overlapping bounding boxes. Let's investigate how often bounding boxes overlap in our dataset.
#
# We consider two bounding boxes overlapping if their IoU exceeds 0.1.
#
# **Before running the code:**
# - Which animal classes do you expect to have the most overlapping bounding boxes?
#
# **Explore:**
# - Compare the number of overlapping box pairs across classes.
# - Compare the proportion of overlapping pairs relative to all possible pairs.
#
# **Observe and interpret:**
# - Which classes have the most overlapping box pairs?
# - Does this ranking change when considering the proportion of overlapping pairs?
# - Why might bounding-box overlap be challenging for object detection?

# %%
# @title 🔎 Explore Bounding Box Overlap Across Classes { display-mode: "form" }
# needs: boxes


# Count overlapping box pairs in each image
rows = []

for image_id, group in boxes.groupby("image_id"):
    bounding_boxes = group["bbox"].tolist()
    n_boxes = len(bounding_boxes)
    n_box_pairs = n_boxes * (n_boxes - 1) // 2

    n_overlapping = sum(
        box_iou_dice(box_a, box_b)[0] > OVERLAP_IOU
        for box_a, box_b in combinations(bounding_boxes, 2)
    )

    rows.append({
        "image_id": image_id,
        "n_boxes": n_boxes,
        "n_pairs": n_box_pairs,
        "n_overlapping": n_overlapping
    })

overlap_counts = pd.DataFrame(rows)

# Associate each image with its class
overlap_counts = overlap_counts.merge(
    images[["id", "class"]],
    left_on="image_id",
    right_on="id",
    how="left"
)

# Aggregate by class
overlap_by_class = (
    overlap_counts.groupby("class")
    .agg(
        boxes_per_image=("n_boxes", "mean"),
        overlapping_pairs=("n_overlapping", "sum"),
        possible_pairs=("n_pairs", "sum")
    )
    .reindex(CLASSES)
)

overlap_by_class["overlap_share"] = (
    100
    * overlap_by_class["overlapping_pairs"]
    / overlap_by_class["possible_pairs"].replace(0, float("nan"))
)

# Visualize overlap across classes
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

colors = [CLASS_COLORS[c] for c in CLASSES]

axes[0].bar(
    CLASSES,
    overlap_by_class["overlapping_pairs"],
    color=colors
)
axes[0].set_title("Overlapping box pairs")
axes[0].set_ylabel("Number of pairs")

axes[1].bar(
    CLASSES,
    overlap_by_class["overlap_share"],
    color=colors
)
axes[1].set_title("Share of overlapping pairs")
axes[1].set_ylabel("Overlapping pairs (%)")

for ax in axes:
    ax.tick_params(axis="x", rotation=30)
    for label, c in zip(ax.get_xticklabels(), CLASSES, strict=False):
        label.set_color(CLASS_COLORS[c])

fig.tight_layout()
plt.show()

display(
    overlap_by_class[
        ["boxes_per_image", "overlapping_pairs", "overlap_share"]
    ].round(1)
)

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p>Classes with many animals per image may have more overlapping box pairs. However, the ranking can change when comparing the proportion of overlapping pairs, because images with more boxes also have more possible pairs.</p>
<p>Overlapping boxes make it harder to distinguish individual animals and assign accurate bounding boxes. This can lead to missed objects, inaccurate localization, or duplicate detections. The overlap statistics describe bounding boxes, not necessarily actual occlusion between animals.</p>
""")


# %% [markdown] edition="own"
# ✍️ **Your answer** (double-click to edit):
#
# *Look for:* whether the class ranking changes between the number and the proportion of overlapping pairs. Images with many boxes also have many possible pairs.

# %% [markdown]
# ### 🟢 Bounding Box Quality: How Well Do the Boxes Fit?
#
# Bounding boxes should enclose the annotated animal as closely as possible. However, annotations may be too large, too small, or incorrectly positioned.
#
# **Before running the code:**
# - What makes a bounding box a good annotation?
#
# **Explore:**
# - Browse images from different animal classes and inspect their bounding boxes.
# - Check whether each box tightly encloses the animal, covers its visible parts, and corresponds to the correct animal.
# - Look for examples where accurate annotation is difficult.
#
# **Observe and interpret:**
# - How well do the bounding boxes fit the visible animals?
# - Which images or animal characteristics make accurate annotation difficult?
# - How might inaccurate bounding boxes affect object detection?

# %%
# @title 🔎 Inspect Bounding Box Quality { display-mode: "form" }
# needs: boxes

# Browse annotated images by class
quality_class_selector = widgets.Dropdown(
    options=CLASSES,
    value=CLASSES[0],
    description="Class:",
    style={"description_width": "initial"}
)

quality_page = 0

quality_previous = widgets.Button(description="← Previous")
quality_next_button = widgets.Button(description="Next →")
quality_shuffle = widgets.Button(description="Shuffle")
quality_counter = widgets.HTML()
quality_output = widgets.Output()

quality_rng = np.random.default_rng(SEED)


def show_quality_page():
    with quality_output:
        clear_output(wait=True)

        selected = quality_images.iloc[
            quality_page * PAGE_SIZE:(quality_page + 1) * PAGE_SIZE
        ]

        show_boxes(
            selected,
            boxes,
            "Inspect bounding box fit",
            caption=lambda r: f"{r['class']} · Image {r['id']}"
        )

    total_pages = max(1, (len(quality_images) + PAGE_SIZE - 1) // PAGE_SIZE)
    quality_counter.value = f"Page {quality_page + 1} of {total_pages}"
    quality_previous.disabled = quality_page == 0
    quality_next_button.disabled = quality_page >= total_pages - 1


def change_quality_page(step):
    global quality_page
    quality_page += step
    show_quality_page()


def update_quality_images(change=None):
    global quality_page, quality_images
    quality_page = 0

    quality_images = images[
        images["class"] == quality_class_selector.value
    ].sample(frac=1, random_state=SEED).reset_index(drop=True)

    show_quality_page()


def shuffle_quality_images(button):
    global quality_page, quality_images
    quality_page = 0

    quality_images = quality_images.sample(
        frac=1,
        random_state=int(quality_rng.integers(0, 2**31 - 1))
    ).reset_index(drop=True)

    show_quality_page()


quality_class_selector.observe(update_quality_images, names="value")
quality_previous.on_click(lambda b: change_quality_page(-1))
quality_next_button.on_click(lambda b: change_quality_page(1))
quality_shuffle.on_click(shuffle_quality_images)

display(
    widgets.VBox([
        quality_class_selector,
        widgets.HBox([quality_previous, quality_next_button, quality_shuffle, quality_counter])
    ]),
    quality_output
)

update_quality_images()

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p>Bounding boxes do not always fit tightly around animals. For example, cats with long tails may require large boxes even though much of the box contains background. In another image, a cat wearing a hat has only its face annotated rather than its entire body.</p>
<p>Animals are not always fully visible; sometimes only parts of their bodies appear in the image. A reflection of a cat on a car is not annotated, although it looks like a real cat. These examples raise questions about annotation guidelines and which visible objects or parts should be labeled.</p>
<p>For very small animals, it is difficult to judge box accuracy. Partial visibility, occlusion, unusual poses, and unclear boundaries make annotation challenging. Inconsistent annotations can affect how accurately an object detector learns to localize animals.</p>
""")


# %% [markdown] edition="own"
# ✍️ **Your answer** (double-click to edit):
#
# *In the park:* a cat wearing a hat had only its face boxed, and a cat's reflection on a car had no box at all.

# %% [markdown] edition="teaching"
# ### 🟢 Missing Annotations: Find the Missing Box
#
# Bounding-box annotations are not always complete. An annotator may overlook an animal, especially in crowded scenes. During training, an unannotated animal may then be treated as background.
#
# To simulate such an annotation error, we create a modified copy of our annotations (`park_boxes`) and deliberately remove **one bounding box**. The original annotations remain unchanged.
#
# Your task is to identify the affected image and locate the missing box.
#
# **Before running the code:**
# - How could you detect a missing bounding box without manually inspecting every image?
#
# **Explore:**
# - Compare the number of boxes in `park_boxes` with the recorded number of animals per image.
# - Identify images where these counts disagree.
# - Inspect the suspicious image and locate the missing bounding box.
#
# **Observe and interpret:**
# - Which animal is missing a bounding box?
# - Why might this annotation error be difficult to notice?
# - How could missing annotations affect the training and evaluation of an object detector?

# %% edition="teaching"
# @title 🔎 Simulate and Detect a Missing Annotation { display-mode: "form" }

# Simulate an annotation error by removing exactly one bounding box
# from a copy of the original annotations.
# The original boxes dataframe remains unchanged.

def simulate_missing_box(boxes_df, images_df, seed):
    rng = random.Random(seed)

    counts = boxes_df.groupby("image_id").size()
    image_classes = images_df.set_index("id")["class"]

    # Prefer a crowded sheep image
    candidates = counts[counts >= 3].index.tolist()
    sheep_candidates = [
        image_id for image_id in candidates
        if image_classes.get(image_id) == "sheep"
    ]

    if sheep_candidates:
        candidates = sheep_candidates
    if not candidates:
        candidates = [counts.idxmax()]

    image_id = rng.choice(sorted(candidates))
    image_boxes = boxes_df[boxes_df["image_id"] == image_id]
    missing_box_id = rng.choice(sorted(image_boxes["id"]))

    simulated_boxes = boxes_df[
        boxes_df["id"] != missing_box_id
    ].copy()

    answer_key = {
        "image_id": image_id,
        "annotation_id": missing_box_id
    }

    return simulated_boxes, answer_key


# Create the simulated dataset (keep the answer key hidden)
park_boxes, missing_box_key = simulate_missing_box(
    boxes, images, SEED
)

# Compare recorded animal counts with the simulated annotations
box_counts = park_boxes.groupby("image_id").size()

annotation_check = images[["id", "class", "n_objects"]].copy()
annotation_check["n_boxes"] = (
    annotation_check["id"].map(box_counts).fillna(0).astype(int)
)
annotation_check["difference"] = (
    annotation_check["n_objects"] - annotation_check["n_boxes"]
)

suspicious_images = annotation_check[
    annotation_check["difference"] != 0
].copy()

print("Simulation: One bounding box was deliberately removed.")
print(f"Images checked: {len(annotation_check)}")
print(f"Images with mismatched counts: {len(suspicious_images)}")

display(
    suspicious_images[
        ["id", "class", "n_objects", "n_boxes", "difference"]
    ].rename(columns={
        "id": "Image ID",
        "class": "Class",
        "n_objects": "Animals",
        "n_boxes": "Boxes",
        "difference": "Missing boxes"
    })
)

# %% edition="teaching"
# @title 🔎 Inspect Missing Annotations { display-mode: "form" }

missing_image_selector = widgets.Dropdown(
    options=[
        (f"Image {image_id}", image_id)
        for image_id in suspicious_images["id"]
    ],
    description="Image ID:",
    style={"description_width": "initial"}
)

missing_output = widgets.Output()


def missing_inspect(change=None):
    with missing_output:
        missing_output.clear_output(wait=True)

        selected_images = images[
            images["id"] == missing_image_selector.value
        ]

        show_boxes(
            selected_images,
            park_boxes,
            "Inspect the available bounding boxes",
            caption=lambda r: (
                f"{r['class']} · "
                f"{r['n_objects']} animals expected"
            )
        )


missing_image_selector.observe(missing_inspect, names="value")

display(missing_image_selector, missing_output)
missing_inspect()

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p><strong>Finding the error:</strong> Comparing the recorded number of animals with the remaining bounding boxes reveals the affected image. Inspecting that image shows an animal without a bounding box.</p>
<p><strong>Why it matters:</strong> Missing annotations can be difficult to notice in crowded scenes, particularly when animals overlap or are partially hidden. During training, an unannotated animal may be treated as background, potentially confusing the detector. During evaluation, a correct detection of that animal may incorrectly count as a false positive.</p>
""")


# %% [markdown]
# #### Visualizing a Missing Annotation
#
# The illustration below shows a simplified example: three animals are present, but only two have bounding boxes.
#
# The solid gray boxes represent existing annotations. The dashed yellow box indicates the missing annotation.
#
# Consider how this mismatch could affect an object detector trained on these annotations.

# %%
# @title 🔎 Visualize a Missing Bounding Box { display-mode: "form" }

fig, ax = plt.subplots(figsize=(7, 2.5))

for k, x in enumerate([2, 5, 8]):
    # Draw an animal
    ax.add_patch(mpl.patches.Ellipse(
        (x, 1), 2.0, 1.2,
        facecolor=LIGHT_GRAY, alpha=0.45,
        edgecolor="none"
    ))

    # Draw existing annotations or highlight the missing one
    missing = k == 2

    ax.add_patch(mpl.patches.Rectangle(
        (x - 1.3, 0.1), 2.6, 1.8,
        fill=False,
        linewidth=2,
        linestyle="--" if missing else "-",
        edgecolor=HIGHLIGHT if missing else LIGHT_GRAY
    ))

    if missing:
        ax.text(
            x, 1, "Missing box",
            ha="center", va="center",
            fontsize=10, color=INK,
            fontweight="bold"
        )

ax.set_xlim(0, 10)
ax.set_ylim(-0.1, 2.2)
ax.set_aspect("equal")
ax.axis("off")

ax.set_title(
    "3 animals present · 2 boxes annotated · 1 missing",
    fontsize=11, color=INK
)

fig.tight_layout()
display(fig)
plt.close(fig)

# %%
# @title 📘 What Matters for Object Detection? { display-mode: "form" }
callout("theory", "📘 What Matters for Object Detection?", r"""
<p>Beyond image classification, object detection requires us to understand
the characteristics and quality of individual object annotations.</p>

<ul style="margin-bottom:0;">
<li><strong>Class distribution:</strong> Counting images and bounding boxes can reveal different class imbalances.</li>
<li><strong>Object size:</strong> Small objects contain fewer pixels and may be harder to detect.</li>
<li><strong>Object position:</strong> Position biases may affect generalization to less common locations.</li>
<li><strong>Crowding:</strong> Closely spaced or overlapping objects can be difficult to distinguish.</li>
<li><strong>Annotation quality:</strong> Missing or inaccurate boxes can affect training and evaluation.</li>
</ul>

<p style="margin-bottom:0; margin-top:12px;">
<strong>Key principle:</strong> Always consider what is being counted and how
dataset characteristics relate to the intended computer vision task.
</p>
""")


