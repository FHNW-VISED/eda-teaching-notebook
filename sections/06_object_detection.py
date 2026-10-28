# %% [markdown]
# ## Object Detection
#
# In image classification, a model predicts what is present in an image. But what if the park wants to count individual animals and determine where they are?
#
# Object detection extends classification by predicting a class label and a bounding box for each object. This allows us to investigate not only which animals appear, but also how many there are, where they are located, and how large they appear.
#
# In this section, we explore how object size, position, crowding, and class imbalance affect an object detection dataset. We also examine the additional annotation effort required compared with image classification.
#

# %% [markdown]
# ### From Labels to Bounding Boxes
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
# @title Box helpers (draw boxes on thumbnails) { display-mode: "form" }
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
    """Gallery of the photos in `rows_df` with their boxes from `boxes_df` drawn in the class colour.

    `caption(row) -> str` writes the title above each photo; by default the class and the number of boxes.
    """
    n = len(rows_df)
    if n == 0:
        print(f"{title}: no photos to show.")
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


# %%
# @title 🔎 Explore Bounding Boxes { display-mode: "form" }

import ipywidgets as widgets
from IPython.display import display, clear_output

ann = annotations[annotations["image_id"].isin(images["id"])]
imgs = images[images["id"].isin(ann["image_id"])].reset_index(drop=True)

page = 0
page_size = 4
output = widgets.Output()

previous = widgets.Button(description="← Previous")
next_button = widgets.Button(description="Next →")
counter = widgets.HTML()


def show_page():
    with output:
        clear_output(wait=True)

        start = page * page_size
        selected = imgs.iloc[start:start + page_size]

        show_boxes(
            selected,
            ann,
            "Bounding boxes: one annotation per animal",
            caption=lambda r: r["class"]
        )

        total_pages = (len(imgs) + page_size - 1) // page_size
        counter.value = f"Page {page + 1} of {total_pages}"
        previous.disabled = page == 0
        next_button.disabled = page >= total_pages - 1


def change_page(step):
    global page
    page += step
    show_page()


previous.on_click(lambda b: change_page(-1))
next_button.on_click(lambda b: change_page(1))

display(widgets.HBox([previous, next_button, counter]), output)
show_page()

# %% [markdown]
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
# - Run the validation and compare the resulting distribution with the image-level distribution from Section 4.
#
# **Observe and interpret:**
# - Which classes change most in their relative frequency?
# - Why might image-level and object-level class imbalance lead to different conclusions?

# %%
# @title
from IPython.display import HTML; HTML(r"""<div class="callout callout-your-turn" style="background:#FFF4D6;color:#7A4A00;border-left:6px solid #7A4A00;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">
<div style="font-weight:bold;">✏️ Your turn · <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">count_boxes_per_class</code></div>
<p style="margin:6px 0 0 0;">Write the function in the next cell: given an annotations table (one row per animal, with a <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">class</code>
column), return a dict <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">{class: number of boxes}</code> with <strong>every</strong> class in <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">CLASSES</code> as a key, counts as
plain <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">int</code>s. <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">use()</code> checks it against the reference and continues with the reference either way.</p>
</div>""")


# %%
# @title Reference solution — peek only if you are stuck { display-mode: "form" }
def _ref_count_boxes_per_class(annotations):
    counts = annotations["class"].value_counts()
    return {c: int(counts.get(c, 0)) for c in CLASSES}


# %%
# ✍️ YOUR CODE
def count_boxes_per_class(annotations):
    """Number of boxes (rows) per class.

    Inputs: `annotations`, a DataFrame with one row per animal and a `class` column.
    Output: a dict {class: int} with every class of CLASSES as a key (0 when a class has no box).

    e.g. {"cat": 12, "dog": 9, "bird": 30, "horse": 7, "sheep": 25, "zebra": 14}
    """
    ...


boxes = annotations[annotations["image_id"].isin(images["id"])].reset_index(drop=True)
boxes_per_class = use(count_boxes_per_class, boxes, reference=_ref_count_boxes_per_class)

# %%
# @title 🔎 Compare Class Distributions: Photos vs. Boxes { display-mode: "form" }

photos_per_class = {c: int((images["class"] == c).sum()) for c in CLASSES}
results = {}
results["boxes_per_class"] = boxes_per_class
results["n_boxes"] = int(len(boxes))

by_photos = sorted(CLASSES, key=lambda c: -photos_per_class[c])
by_boxes = sorted(CLASSES, key=lambda c: -boxes_per_class[c])

print(f"{len(images)} photos · {len(boxes)} boxes")
print("Ranked by photos: " + " > ".join(f"{c} {photos_per_class[c]}" for c in by_photos))
print("Ranked by boxes:  " + " > ".join(f"{c} {boxes_per_class[c]}" for c in by_boxes))

for c in CLASSES:
    move = by_photos.index(c) - by_boxes.index(c)
    if move:
        print(
            f"  {c}: rank {by_photos.index(c) + 1} by photos → "
            f"rank {by_boxes.index(c) + 1} by boxes "
            f"({'↑' if move > 0 else '↓'}{abs(move)})"
        )

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

for ax, counts, order, what in zip(
    axes,
    [photos_per_class, boxes_per_class],
    [by_photos, by_boxes],
    ["Photos", "Boxes"],
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

# %%
# @title 💡 Sample Answer
from IPython.display import HTML; HTML(r"""
<details style="background:#F5F5F5; border:1px solid #9CA3AF; border-radius:6px; padding:12px 16px; color:#374151;">
<summary><strong>Sample Answer</strong></summary>

<p>
The class ranking changes because some images contain many animals. Bird and sheep move up when counting bounding boxes, while cat and dog move down because they typically appear individually.
</p>

<p>
Image-level counts describe how frequently classes appear in photos, whereas box-level counts describe how many object instances are annotated. Therefore, class imbalance can look very different depending on which unit is counted.
</p>

</details>
""")

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

results["small_box_share"] = {
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

for threshold, name in [(32, "small ↓"), (96, "large ↑")]:
    ax.axhline(np.log10(threshold), color=INK, linestyle="--", linewidth=1)
    ax.text(
        0.5, np.log10(threshold),
        f" {threshold} px · {name}",
        va="bottom", fontsize=8, color=INK
    )

ax.set_ylabel("Box side √(w·h), px (log)")
ax.set_title("Box size per class (bar = median)")

fig.tight_layout()
display(fig)
plt.close(fig)

# %%
# @title 🔎 Explore Bounding Box Sizes in Images { display-mode: "form" }

import ipywidgets as widgets
from IPython.display import display, clear_output

by_area = boxes.sort_values(["box_area", "id"]).reset_index(drop=True)
photo_of = images.set_index("id")

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

pages = [
    selected_boxes[i:i + 3]
    for i in range(0, len(selected_boxes), 3)
]

page = 0
previous = widgets.Button(description="← Previous")
next_button = widgets.Button(description="Next →")
counter = widgets.HTML()
output = widgets.Output()


def show_page():
    with output:
        clear_output(wait=True)

        selected = pages[page]
        fig, axes = plt.subplots(2, len(selected), figsize=(3.7 * len(selected), 7), squeeze=False)

        for col, b in enumerate(selected):
            photo = photo_of.loc[b["image_id"]]
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

        counter.value = f"Page {page + 1} of {len(pages)}"
        previous.disabled = page == 0
        next_button.disabled = page == len(pages) - 1


def change_page(step):
    global page
    page += step
    show_page()


previous.on_click(lambda b: change_page(-1))
next_button.on_click(lambda b: change_page(1))

display(widgets.HBox([previous, next_button, counter]), output)
show_page()

# %%
# @title 💡 Sample Answer: Object Sizes { display-mode: "form" }

from IPython.display import HTML, display

display(HTML("""
<details style="background:#F5F5F5; border:1px solid #9CA3AF; border-radius:6px; padding:12px 16px; color:#374151;">
<summary><strong>Sample Answer</strong></summary>

<p>
Birds have the smallest bounding boxes by a wide margin, while cats, dogs, and horses are mostly represented by large boxes. Object sizes can vary considerably within a class, depending on the animal's distance from the camera.
</p>

<p>
Small objects are particularly challenging because they contain fewer pixels and less visual detail. This may make distant birds difficult to detect reliably.
</p>

</details>
"""))

# %% [markdown]
# ### 🟢 Object Positions: Where Are the Animals?
#
# A bounding box's centre is `(x + w/2, y + h/2)`. Dividing these coordinates by the image width and height gives normalized positions between 0 and 1, allowing us to compare images of different sizes.
#
# A **heatmap** shows where bounding-box centres occur most frequently. Darker regions indicate more objects.
#
# **Before running the code:**
# - Do you expect animals to appear equally often across all image positions?
#
# **Explore:**
# - Compare the box-centre heatmaps across classes.
# - Look for differences in how concentrated or spread out the positions are.
#
# **Observe and interpret:**
# - Where are most animals located within the images?
# - Which classes have more widely distributed positions?
# - Why might a detector trained on these images struggle with animals near the image edges?

# %%
# @title 🔎 Explore Object Positions Across Classes { display-mode: "form" }
# Box centres on the unit square, one heatmap per class in its own colour (white = no boxes there).
_size = images.set_index("id")[["width", "height"]]
_img_w = boxes["image_id"].map(_size["width"])
_img_h = boxes["image_id"].map(_size["height"])
boxes["cx"] = (boxes["bbox"].map(lambda b: b[0] + b[2] / 2) / _img_w).clip(0, 1)
boxes["cy"] = (boxes["bbox"].map(lambda b: b[1] + b[3] / 2) / _img_h).clip(0, 1)
HEAT_BINS = 6 if SMOKE else 12  # coarser cells with few boxes (N-014)
_edge_share = float(((boxes["cx"] < 0.2) | (boxes["cx"] > 0.8) | (boxes["cy"] < 0.2) | (boxes["cy"] > 0.8))
                    .mean())
RESULTS["edge_box_share"] = _edge_share

fig, axes = plt.subplots(1, len(CLASSES), figsize=(2.3 * len(CLASSES), 2.8))
for ax, c in zip(axes, CLASSES, strict=True):
    own = boxes[boxes["class"] == c]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(c, ["white", CLASS_COLORS[c]])
    heat, _, _ = np.histogram2d(own["cy"], own["cx"], bins=HEAT_BINS, range=[[0, 1], [0, 1]])
    ax.imshow(heat, cmap=cmap, extent=[0, 1, 1, 0], vmin=0)
    ax.set_title(f"{c} · {len(own)} boxes", fontsize=9, color=CLASS_COLORS[c])
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(labelsize=7)
    ax.grid(False)
axes[0].set_ylabel("box centre y (0 = top)")
fig.suptitle(f"Where the box centres are · {_edge_share:.0%} of all boxes are in the outer 20 % of the frame",
             color=INK)
fig.tight_layout()
display(fig)
plt.close(fig)

# %%
# @title 💡 Sample Answer: Object Positions { display-mode: "form" }

from IPython.display import HTML, display

display(HTML("""
<details style="background:#F5F5F5; border:1px solid #9CA3AF; border-radius:6px; padding:12px 16px; color:#374151;">
<summary><strong>Sample Answer</strong></summary>
<p>
Bounding-box centres are concentrated near the middle of the images, suggesting
that animals are often photographed centrally. Birds and sheep show a wider
spatial distribution, possibly because multiple animals appear together.
</p>
<p>
A detector trained mainly on centrally positioned animals may perform less
reliably near image edges, where fewer training examples are available.
However, the heatmaps alone do not establish how well the detector performs.
</p>
</details>
"""))

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

# Count annotated objects per image
object_counts = (
    boxes.groupby("image_id")
    .size()
    .rename("n_objects")
    .reset_index()
)

# Include images without annotations
crowding = images[["id", "class"]].merge(
    object_counts, left_on="id", right_on="image_id", how="left"
)
crowding["n_objects"] = crowding["n_objects"].fillna(0).astype(int)

results["max_objects_per_image"] = int(crowding["n_objects"].max())

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# Distribution of object counts
count_distribution = crowding["n_objects"].value_counts().sort_index()

axes[0].bar(
    count_distribution.index,
    count_distribution.values,
    color="#B0B0B0"
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

import ipywidgets as widgets
from IPython.display import display, clear_output

# Sort images from most to fewest annotated objects
crowded_images = crowding.sort_values(
    ["n_objects", "id"], ascending=[False, True]
).reset_index(drop=True)

# Select object count (group counts of 10 or more)
available_counts = sorted(
    int(n) for n in crowding["n_objects"].unique() if n < 10
)

count_selector = widgets.Dropdown(
    options=(
        [("All", None)]
        + [(str(n), n) for n in available_counts]
        + ([("10 or more", "10+")] if (crowding["n_objects"] >= 10).any() else [])
    ),
    value=None,
    description="Objects:",
    style={"description_width": "initial"}
)

page = 0
page_size = 4

first = widgets.Button(description="⇤ First")
previous = widgets.Button(description="← Previous")
next_button = widgets.Button(description="Next →")
last = widgets.Button(description="Last ⇥")
counter = widgets.HTML()
output = widgets.Output()


def get_filtered_images():
    if count_selector.value is None:
        return crowded_images
    if count_selector.value == "10+":
        return crowded_images[crowded_images["n_objects"] >= 10]
    return crowded_images[
        crowded_images["n_objects"] == count_selector.value
    ]


def show_crowded_page():
    filtered = get_filtered_images()
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)

    with output:
        clear_output(wait=True)

        selected = filtered.iloc[
            page * page_size:(page + 1) * page_size
        ]

        # Preserve the selected order
        selected_images = selected[["id", "n_objects"]].merge(
            images, on="id", how="left", sort=False
        )

        show_boxes(
            selected_images,
            boxes,
            "Images ordered by number of annotated objects",
            caption=lambda r: f"{r['class']} · {r['n_objects']} objects"
        )

    count_label = (
        "All counts" if count_selector.value is None
        else "10+ objects" if count_selector.value == "10+"
        else f"{count_selector.value} objects"
    )

    counter.value = (
        f"{count_label} · {len(filtered)} images · "
        f"Page {page + 1} of {total_pages}"
    )

    first.disabled = previous.disabled = page == 0
    next_button.disabled = last.disabled = page >= total_pages - 1


def go_to_page(new_page):
    global page
    total_pages = max(
        1, (len(get_filtered_images()) + page_size - 1) // page_size
    )
    page = max(0, min(new_page, total_pages - 1))
    show_crowded_page()


def change_count(change):
    if change["name"] == "value":
        go_to_page(0)


count_selector.observe(change_count, names="value")

first.on_click(lambda b: go_to_page(0))
previous.on_click(lambda b: go_to_page(page - 1))
next_button.on_click(lambda b: go_to_page(page + 1))
last.on_click(lambda b: go_to_page(10**9))

display(
    widgets.VBox([
        count_selector,
        widgets.HBox([first, previous, next_button, last, counter])
    ]),
    output
)

show_crowded_page()

# %%
# @title 💡 Sample Answer { display-mode: "form" }

from IPython.display import HTML, display

display(HTML("""
<details style="background:#F5F5F5; border:1px solid #9CA3AF; border-radius:6px; padding:12px 16px; color:#374151;">
<summary><strong>Sample Answer</strong></summary>
<p>Images of herd animals, such as sheep or zebras, may contain more annotated objects than images of cats or dogs. Images with many objects often show animals close together, but some animals may still be clearly separated.</p>
<p>A higher object count does not necessarily mean more crowding. Object size, spacing, and occlusion also matter. Closely packed animals can be difficult to detect individually, increasing the risk of missed or duplicate detections.</p>
</details>
"""))


# %% [markdown]
# ### 🔴 Bounding Box Overlap: Understanding IOU
#
# Counting objects tells us how many animals are annotated, but not how closely they appear together. We can measure bounding-box overlap using **Intersection over Union (IOU)**:
#
# $$
# \mathrm{IOU} = \frac{\text{Area of Intersection}}{\text{Area of Union}}
# $$
#
# IOU ranges from 0 (no overlap) to 1 (identical boxes).
#
# **Before running the code:**
# - What happens to IOU when two boxes move closer together?
#
# **Explore:**
# - Move the X and Y sliders to change how two bounding boxes overlap.
# - Observe how IOU changes and when it exceeds the threshold of 0.1.
#
# **Observe and interpret:**
# - How does IOU change as the boxes move apart?
# - Why is bounding-box overlap only an approximation of how crowded a scene is?

# %%
# @title IOU Helper Function { display-mode: "form" }
def box_iou_dice(a: tuple, b: tuple) -> tuple[float, float]:
    """IoU and Dice of two boxes given as (x, y, w, h)."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a, area_b = a[2] * a[3], b[2] * b[3]
    return inter / (area_a + area_b - inter), 2 * inter / (area_a + area_b)


# %%
# @title 🔎 Explore Bounding Box Overlap (IOU) { display-mode: "form" }

import ipywidgets as widgets
from IPython.display import display

overlap_iou = 0.1

box_grey = "#B0B0B0"
box_edge = "#666666"
fhnw_yellow = "#FFDA00"

a = (20, 20, 60, 40)


def show_overlap(x_shift, y_shift):
    b = (20 + x_shift, 20 + y_shift, 60, 40)

    iou = box_iou_dice(a, b)[0]

    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - ix0)
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - iy0)

    intersection = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - intersection

    fig, ax = plt.subplots(figsize=(7, 3.5))

    for box in [a, b]:
        ax.add_patch(mpl.patches.Rectangle(
            box[:2], *box[2:],
            facecolor=box_grey,
            edgecolor=box_edge,
            alpha=0.45,
            linewidth=1.5
        ))

    if intersection > 0:
        ax.add_patch(mpl.patches.Rectangle(
            (ix0, iy0), iw, ih,
            facecolor=fhnw_yellow,
            edgecolor=INK,
            linewidth=1.5
        ))

    ax.set_xlim(10, 160)
    ax.set_ylim(120, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        f"IOU = {intersection} / {union} = {iou:.2f} "
        f"({'overlapping' if iou > overlap_iou else 'not counted'})",
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

output = widgets.interactive_output(
    show_overlap,
    {"x_shift": x_slider, "y_shift": y_slider}
)

display(widgets.VBox([x_slider, y_slider]), output)

# %%
# @title 💡 Sample Answer { display-mode: "form" }

from IPython.display import HTML, display

display(HTML("""
<details style="background:#F5F5F5; border:1px solid #9CA3AF; border-radius:6px; padding:12px 16px; color:#374151;">
<summary><strong>Sample Answer</strong></summary>
<p>IOU increases as the boxes overlap more strongly, reaching 1 when they are identical. When the boxes move apart, IOU decreases and becomes 0 when they no longer intersect.</p>
<p>Bounding-box overlap is only an approximation of crowding. Boxes can overlap even when the animals themselves do not, and animals can partially occlude one another without their boxes having a high IOU.</p>
</details>
"""))

# %% [markdown]
# ### 🔴 Crowding: Bounding Box Overlap Across Classes
#
# Images with many animals do not necessarily contain overlapping bounding boxes. Let's investigate how often bounding boxes overlap in our dataset.
#
# We consider two bounding boxes overlapping if their IOU exceeds 0.1.
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

from itertools import combinations

overlap_iou = 0.1

# Count overlapping box pairs in each image
rows = []

for image_id, group in boxes.groupby("image_id"):
    bounding_boxes = group["bbox"].tolist()
    n_boxes = len(bounding_boxes)
    n_pairs = n_boxes * (n_boxes - 1) // 2

    n_overlapping = sum(
        box_iou_dice(a, b)[0] > overlap_iou
        for a, b in combinations(bounding_boxes, 2)
    )

    rows.append({
        "image_id": image_id,
        "n_boxes": n_boxes,
        "n_pairs": n_pairs,
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
    for label, c in zip(ax.get_xticklabels(), CLASSES):
        label.set_color(CLASS_COLORS[c])

fig.tight_layout()
plt.show()

display(
    overlap_by_class[
        ["boxes_per_image", "overlapping_pairs", "overlap_share"]
    ].round(1)
)

# %%
# @title 💡 Sample Answer { display-mode: "form" }

from IPython.display import HTML, display

display(HTML("""
<details style="background:#F5F5F5; border:1px solid #9CA3AF; border-radius:6px; padding:12px 16px; color:#374151;">
<summary><strong>Sample Answer</strong></summary>
<p>Classes with many animals per image may have more overlapping box pairs. However, the ranking can change when comparing the proportion of overlapping pairs, because images with more boxes also have more possible pairs.</p>
<p>Overlapping boxes make it harder to distinguish individual animals and assign accurate bounding boxes. This can lead to missed objects, inaccurate localization, or duplicate detections. The overlap statistics describe bounding boxes, not necessarily actual occlusion between animals.</p>
</details>
"""))

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

import ipywidgets as widgets
from IPython.display import display, clear_output

# Browse annotated images by class
class_selector = widgets.Dropdown(
    options=CLASSES,
    value=CLASSES[0],
    description="Class:",
    style={"description_width": "initial"}
)

page_size = 4
quality_page = 0

previous = widgets.Button(description="← Previous")
next_button = widgets.Button(description="Next →")
shuffle = widgets.Button(description="Shuffle")
counter = widgets.HTML()
output = widgets.Output()

rng = np.random.default_rng(SEED)
quality_images = images[images["class"] == class_selector.value].copy()


def show_quality_page():
    with output:
        clear_output(wait=True)

        selected = quality_images.iloc[
            quality_page * page_size:(quality_page + 1) * page_size
        ]

        show_boxes(
            selected,
            boxes,
            "Inspect bounding box fit",
            caption=lambda r: f"{r['class']} · Image {r['id']}"
        )

    total_pages = max(1, (len(quality_images) + page_size - 1) // page_size)
    counter.value = f"Page {quality_page + 1} of {total_pages}"
    previous.disabled = quality_page == 0
    next_button.disabled = quality_page >= total_pages - 1


def change_quality_page(step):
    global quality_page
    quality_page += step
    show_quality_page()


def update_quality_images(change=None):
    global quality_page, quality_images
    quality_page = 0

    quality_images = images[
        images["class"] == class_selector.value
    ].sample(frac=1, random_state=SEED).reset_index(drop=True)

    show_quality_page()


def shuffle_quality_images(button):
    global quality_page, quality_images
    quality_page = 0

    quality_images = quality_images.sample(
        frac=1,
        random_state=int(rng.integers(0, 2**31 - 1))
    ).reset_index(drop=True)

    show_quality_page()


class_selector.observe(update_quality_images, names="value")
previous.on_click(lambda b: change_quality_page(-1))
next_button.on_click(lambda b: change_quality_page(1))
shuffle.on_click(shuffle_quality_images)

display(
    widgets.VBox([
        class_selector,
        widgets.HBox([previous, next_button, shuffle, counter])
    ]),
    output
)

update_quality_images()

# %%
# @title 💡 Sample Answer { display-mode: "form" }

from IPython.display import HTML, display

display(HTML("""
<details style="background:#F5F5F5; border:1px solid #9CA3AF; border-radius:6px; padding:12px 16px; color:#374151;">
<summary><strong>Sample Answer</strong></summary>
<p>Bounding boxes do not always fit tightly around animals. For example, cats with long tails may require large boxes even though much of the box contains background. In another image, a cat wearing a hat has only its face annotated rather than its entire body.</p>
<p>Animals are not always fully visible; sometimes only parts of their bodies appear in the image. A reflection of a cat on a car is not annotated, although it looks like a real cat. These examples raise questions about annotation guidelines and which visible objects or parts should be labeled.</p>
<p>For very small animals, it is difficult to judge box accuracy. Partial visibility, occlusion, unusual poses, and unclear boundaries make annotation challenging. Inconsistent annotations can affect how accurately an object detector learns to localize animals.</p>
</details>
"""))

# %% [markdown]
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

# %%
# @title 🔎 Simulate and Detect a Missing Annotation { display-mode: "form" }

import random

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
park_boxes, hunt_answer_key = simulate_missing_box(
    boxes, images, SEED
)

# Compare recorded animal counts with the simulated annotations
box_counts = park_boxes.groupby("image_id").size()

annotation_check = images[["id", "class", "n_animals"]].copy()
annotation_check["n_boxes"] = (
    annotation_check["id"].map(box_counts).fillna(0).astype(int)
)
annotation_check["difference"] = (
    annotation_check["n_animals"] - annotation_check["n_boxes"]
)

suspicious_images = annotation_check[
    annotation_check["difference"] != 0
].copy()

print("Simulation: One bounding box was deliberately removed.")
print(f"Images checked: {len(annotation_check)}")
print(f"Images with mismatched counts: {len(suspicious_images)}")

display(
    suspicious_images[
        ["id", "class", "n_animals", "n_boxes", "difference"]
    ].rename(columns={
        "id": "Image ID",
        "class": "Class",
        "n_animals": "Animals",
        "n_boxes": "Boxes",
        "difference": "Missing boxes"
    })
)

# %%
# @title 🔎 Inspect Missing Annotations { display-mode: "form" }

import ipywidgets as widgets
from IPython.display import display

image_selector = widgets.Dropdown(
    options=[
        (f"Image {image_id}", image_id)
        for image_id in suspicious_images["id"]
    ],
    description="Image ID:",
    style={"description_width": "initial"}
)

output = widgets.Output()


def inspect_annotations(change=None):
    with output:
        output.clear_output(wait=True)

        selected_images = images[
            images["id"] == image_selector.value
        ]

        show_boxes(
            selected_images,
            park_boxes,
            "Inspect the available bounding boxes",
            caption=lambda r: (
                f"{r['class']} · "
                f"{r['n_animals']} animals expected"
            )
        )


image_selector.observe(inspect_annotations, names="value")

display(image_selector, output)
inspect_annotations()

# %%
# @title 💡 Sample Answer { display-mode: "form" }

from IPython.display import HTML, display

display(HTML("""
<details style="background:#F5F5F5; border:1px solid #9CA3AF; border-radius:6px; padding:12px 16px; color:#374151;">
<summary><strong>Sample Answer</strong></summary>
<p><strong>Finding the error:</strong> Comparing the recorded number of animals with the remaining bounding boxes reveals the affected image. Inspecting that image shows an animal without a bounding box.</p>
<p><strong>Why it matters:</strong> Missing annotations can be difficult to notice in crowded scenes, particularly when animals overlap or are partially hidden. During training, an unannotated animal may be treated as background, potentially confusing the detector. During evaluation, a correct detection of that animal may incorrectly count as a false positive.</p>
</details>
"""))

# %% [markdown]
# #### Visualizing a Missing Annotation
#
# The illustration below shows a simplified example: three animals are present, but only two have bounding boxes.
#
# The solid grey boxes represent existing annotations. The dashed yellow box indicates the missing annotation.
#
# Consider how this mismatch could affect an object detector trained on these annotations.

# %%
# @title 🔎 Visualize a Missing Bounding Box { display-mode: "form" }

box_grey = "#B0B0B0"
fhnw_yellow = "#FFDA00"

fig, ax = plt.subplots(figsize=(7, 2.5))

for k, x in enumerate([2, 5, 8]):
    # Draw an animal
    ax.add_patch(mpl.patches.Ellipse(
        (x, 1), 2.0, 1.2,
        facecolor=box_grey, alpha=0.45,
        edgecolor="none"
    ))

    # Draw existing annotations or highlight the missing one
    missing = k == 2

    ax.add_patch(mpl.patches.Rectangle(
        (x - 1.3, 0.1), 2.6, 1.8,
        fill=False,
        linewidth=2,
        linestyle="--" if missing else "-",
        edgecolor=fhnw_yellow if missing else box_grey
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

from IPython.display import HTML, display

display(HTML("""
<div style="background:#E3F2FD; border-left:4px solid #0D47A1;
            padding:14px 18px; border-radius:6px; color:#0D47A1;">

<strong>📘 What Matters for Object Detection?</strong>
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

</div>
"""))
