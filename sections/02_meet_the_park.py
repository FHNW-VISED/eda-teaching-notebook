# %% [markdown]
# ## Meet the Park: Explore Your Dataset
#
# Wildwood Farm Park wants to monitor animals across several enclosures using fixed outdoor cameras. The park has sheep and horses in its paddocks, birds near the aviary pond, zebras in the savanna area, and cats and dogs around the visitor yard.
#
# The park ranger wants to use computer vision for three different tasks:

# %%

# @title Computer Vision Tasks
from IPython.display import HTML
HTML(r"""<div class="callout callout-theory" style="background:#E3F2FD;color:#0D47A1;border-left:6px solid #0D47A1;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">
<div style="font-weight:bold;">📘 Three Computer Vision Tasks</div>
<ol style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><strong>Classification – Recognize:</strong> What animal is shown in the image? The park ranger wants to identify dogs entering the sheep paddock so they can intervene before the sheep are harmed.</li>
<li style="margin:2px 0;"><strong>Object Detection – Count:</strong> How many sheep, zebras, or birds are visible, and where are they located?</li>
<li style="margin:2px 0;"><strong>Segmentation – Monitor Space Usage:</strong> Which areas of the image are occupied by animals? The park ranger wants to understand how animals use the space in their enclosures.</li>
</ol>
</div>""")


# %% [markdown]
# The park does not have its own training data yet. Before collecting and annotating new images, the park ranger should ask: Does a suitable dataset already exist, and can we reuse it?
#
# To investigate this question, we'll use the Wildwood animal dataset, a subset of the [COCO 2017 Dataset](https://cocodataset.org/) containing images of six animal classes: cats, dogs, sheep, horses, birds, and zebras.
#
# Each image is assigned one animal class but may contain multiple animals or other objects.
#
# We'll explore the images and annotations to understand the dataset and assess its suitability for the park's computer vision tasks.
#
# The dataset is organized into two tables:
#
# - `images`: One row per image, including its dimensions, animal class, and number of annotated animals (`n_animals`).
# - `annotations`: One row per annotated animal, including its bounding box and segmentation mask.

# %%

# @title 🧾 Karpathy's Recipe · Step 1 { display-mode: "form" }
from IPython.display import HTML

HTML(r"""
<div class="callout callout-recipe"
     style="background:#FFF8D6;color:#333333;border-left:6px solid #F5C400;
            border-radius:6px;padding:12px 14px;line-height:1.5;
            font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;
            font-size:14px;width:100%;box-sizing:border-box;
            white-space:normal;overflow-wrap:break-word;margin:2px 0;">

<div style="font-weight:bold;">🧾 Recipe · Step 1: Become one with the data</div>

<p style="margin:6px 0 0 0;"><em>
"The first step to training a neural net is to not touch any neural net code at all and instead begin by
thoroughly inspecting your data. This step is critical. I like to spend copious amount of time (measured in
units of hours) scanning through thousands of examples, understanding their distribution and looking for
patterns."
</em></p>

<p style="margin:6px 0 0 0;">
Andrej Karpathy,
<a href="https://karpathy.github.io/2019/04/25/recipe/"
   target="_blank" rel="noopener"
   style="color:inherit;text-decoration:underline;">
A Recipe for Training Neural Networks</a> (2019).
</p>

</div>
""")


# %% [markdown]
# We won't spend hours exploring the data, but let's spend a few minutes getting to know our dataset. We'll start with *qualitative inspection* (looking at images and annotations), followed by *quantitative inspection* (examining statistics, distributions, and outliers).
#
# ### Qualitative Inspection
#
# Explore the images to identify patterns, unusual examples, and potential data quality issues. Begin with an overview across classes, then examine one class at a time.
#
# Each image has an assigned animal class, indicated by its frame color. Pay attention to:
#
# - Backgrounds and lighting
# - Viewpoints, distances, and animal sizes
# - Other objects and potential distractions
# - Whether the assigned class reflects the main content of the image
#
# We'll examine the annotations in more detail afterward.

# %% [markdown]
#
# #### 🟢 Images: What Do Images Look Like Across Classes?
#
# Let's start with a quick visual overview of the Wildwood animal dataset. The next cell displays six example images from each animal class to give you a first impression of the dataset.
#
# **Observe:**
# - How do the images differ across classes?
# - How much variation is there within each class (e.g., backgrounds, viewpoints, animal sizes)?
# - Which images might be challenging for an image classification model, and why?

# %%
# @title 🔎 Sample example photos per class, framed in the class colour { display-mode: "form" }
_per_class = 6
fig, axes = plt.subplots(_per_class, len(CLASSES), figsize=(13, 2.2 * _per_class))
for j, c in enumerate(CLASSES):
    _of_class = images[images["class"] == c]
    _picked = _of_class.sample(min(_per_class, len(_of_class)), random_state=SEED)
    for i in range(_per_class):
        ax = axes[i, j]
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(CLASS_COLORS[c])
            spine.set_linewidth(4)
        if i < len(_picked):
            _thumb = _Image.open(_picked.iloc[i]["path"]).convert("RGB")
            _thumb.thumbnail((256, 256))
            ax.imshow(_thumb)
        if i == 0:
            ax.set_title(c, color=INK, fontweight="bold")
fig.tight_layout()
plt.show()

# %% [markdown]
# Select an animal class and browse at least twelve images to explore variation within the class. Pay attention to backgrounds, lighting, viewpoints, animal sizes, and other objects in the images. Look for recurring patterns and unusual examples. If an animal is difficult to spot, enable the bounding boxes.
#
# **Observe:**
# - How much do images vary within each class?
# - Which image characteristics might make classification challenging?
# - Which classes contain images that are least representative of the park's fixed outdoor cameras?

# %%

# @title 🔎 Browse Images Within a Class { display-mode: "form" }
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ipywidgets as widgets
from IPython.display import display
from PIL import Image as _Image

# Shuffle images once, reproducibly
_gallery = {
    c: images[images["class"] == c]
       .sample(frac=1, random_state=SEED)
       .reset_index(drop=True)
    for c in CLASSES
}

def show_gallery(animal_class, page, show_boxes):
    rows = _gallery[animal_class]
    picked = rows.iloc[page * 6 : (page + 1) * 6]

    fig, axes = plt.subplots(2, 3, figsize=(9, 4.5))

    for ax, (_, row) in zip(axes.flat, picked.iterrows()):
        photo = _Image.open(row["path"]).convert("RGB")
        ax.imshow(photo)

        if show_boxes:
            anns = annotations[annotations["image_id"] == row["id"]]
            for _, ann in anns.iterrows():
                x, y, w, h = ann["bbox"]
                ax.add_patch(
                    patches.Rectangle(
                        (x, y), w, h,
                        fill=False,
                        edgecolor=CLASS_COLORS[ann["class"]],
                        linewidth=2
                    )
                )

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_color(CLASS_COLORS[animal_class])
            spine.set_linewidth(3)

    for ax in list(axes.flat)[len(picked):]:
        ax.axis("off")

    fig.suptitle(animal_class, color=INK, fontweight="bold")
    fig.tight_layout(pad=0.5)
    plt.show()
    plt.close(fig)


class_dropdown = widgets.Dropdown(
    options=CLASSES,
    description="Class:"
)

page_slider = widgets.IntSlider(
    value=0,
    min=0,
    max=0,
    description="Group:",
    continuous_update=False
)

box_checkbox = widgets.Checkbox(
    value=False,
    description="Show bounding boxes",
    indent=False
)

def update_pages(change):
    n = len(_gallery[class_dropdown.value])
    page_slider.max = max(0, (n - 1) // 6)
    page_slider.value = 0

class_dropdown.observe(update_pages, names="value")
update_pages(None)

out = widgets.interactive_output(
    show_gallery,
    {
        "animal_class": class_dropdown,
        "page": page_slider,
        "show_boxes": box_checkbox
    }
)

display(
    widgets.HBox([class_dropdown, page_slider, box_checkbox]),
    out
)


# %% [markdown]
# #### 🟢 Document Your Observations
#
# Karpathy's first step also involves documenting what you discover. Write at least three specific observations about the images you explored in the next markdown cell. For each observation, briefly explain what the park should investigate further.
#
# **Hint:** Be specific rather than vague.
#
# - Vague: "Dog images are different."
# - Specific: "Many dog images are taken indoors, often with people in the background. This may not represent the park's outdoor cameras."
#
# Consider backgrounds, lighting, animal sizes, distances, multiple animals, people, and unusual examples.

# %% [markdown]
# **My three observations** (double-click to edit):
#
# 1. …
# 2. …
# 3. …

# %%

# @title 💡 Sample Answer { display-mode: "form" }
from IPython.display import HTML

HTML(r"""
<div class="callout callout-answer"
     style="background:#F5F5F5;color:#374151;border-left:6px solid #9CA3AF;
            border-radius:6px;padding:12px 14px;line-height:1.5;
            font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;
            font-size:14px;width:100%;box-sizing:border-box;
            white-space:normal;overflow-wrap:break-word;margin:2px 0;">

<details style="margin:0;">
<summary style="cursor:pointer;font-weight:bold;">💡 Sample Answer</summary>

<p style="margin:6px 0 0 0;">
Example observations from different animal classes. Your findings may differ.
</p>
<ol style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:4px 0;"><strong>Setting:</strong> Many dogs appear indoors or on streets, often photographed up close. The park's cameras will capture animals outdoors, sometimes far away.</li>
<li style="margin:4px 0;"><strong>Background shortcuts:</strong> People, hands, or leashes appear in many dog images. A model might associate these features with dogs rather than recognizing the animals themselves.</li>
<li style="margin:4px 0;"><strong>Unusual examples:</strong> Some images contain multiple dogs, costumes, or screens displaying animals. These examples may require closer inspection.</li>
<li style="margin:4px 0;"> <strong>Class boundaries:</strong> One image in the sheep class shows several ibexes. Although related to sheep, ibexes are different animals. This raises questions about how the class is defined.</li>
<li style="margin:4px 0;"><strong>Individual animals vs. groups:</strong> Some sheep are annotated individually, while others appear to be annotated as an entire flock. This may affect counting individual animals.</li>
<li style="margin:4px 0;"><strong>Segmentation quality:</strong> Some bird masks extend beyond the actual animal. Such inaccuracies may affect segmentation model training and evaluation.</li>
<li style="margin:4px 0;"><strong>Label relevance:</strong> Some images are labeled as birds even though the bird is tiny and a large bear dominates the image. This may confuse an image classification model.</li>
<li style="margin:4px 0;"><strong>Non-real animals:</strong> Some bird images show stuffed toys rather than living animals. These may not represent the park's intended use case.</li>
<li style="margin:4px 0;"><strong>Image style and incomplete annotations:</strong> Some horse images have an old photographic appearance, and some legs are missing from the annotations. This raises questions about image representativeness and annotation completeness.</li>
</ol>
<p style="margin:6px 0 0 0;">
Keep your observations in mind when evaluating your own dataset for the mini-project.
</p>

</details>
</div>
""")


# %% [markdown]
#
# #### 🟢 Dataset Suitability: How Suitable Are These Images for the Park?
#
# The park's cameras are fixed, outdoors, and often far from the animals.
#
# **Question:**
# Based on the images you explored, which animal classes seem least representative of the park's camera views, and why? Briefly justify your answer.
#

# %%

# @title 💡 Sample Answer
from IPython.display import HTML

HTML(r"""
<div class="callout callout-answer"
     style="background:#F5F5F5;color:#374151;border-left:6px solid #9CA3AF;
            border-radius:6px;padding:12px 14px;line-height:1.5;
            font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;
            font-size:14px;width:100%;box-sizing:border-box;
            white-space:normal;overflow-wrap:break-word;margin:2px 0;">

<details style="margin:0;">
<summary style="cursor:pointer;font-weight:bold;">💡 Sample answer</summary>
<p style="margin:6px 0 0 0;">
Cats and dogs often appear in indoor, close-up photos with people or furniture,
making them less representative of the park's fixed outdoor cameras.
Sheep, horses, and zebras are more often shown outdoors and at greater distances,
making their images somewhat closer to the park's camera views.
</p>
</details>
</div>
""")


# %% [markdown]
#
# #### 🟢 Annotations: What Does an Image Annotation Look Like?
#
# An image can have different types of annotations, such as class labels, bounding boxes, and segmentation masks.
#
# Let's visualize an example from the Wildwood animal dataset and examine how these annotations represent different information about the animals.
#
#
# **Explore the annotations**
#
# Run the next cell to visualize an image with its class label, bounding boxes, and segmentation masks.
#
# **Observe:**
# - What information does each type of annotation provide?
# - How do bounding boxes and segmentation masks differ?
# - Do the annotations correctly represent the animals in the image?

# %%

# @title 🔎 Explore Images and Annotations { display-mode: "form" }
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image
from pycocotools import mask as mask_utils


def decode_mask(segmentation, height, width):
    """Decode a COCO polygon or RLE segmentation into a binary mask."""
    if isinstance(segmentation, list):
        rle = mask_utils.merge(
            mask_utils.frPyObjects(segmentation, height, width)
        )
    elif isinstance(segmentation["counts"], list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation

    return mask_utils.decode(rle).astype(bool)


# Shuffle images reproducibly
explore_images = images.sample(
    frac=1, random_state=SEED
).reset_index(drop=True)

# Navigation widgets
previous_btn = widgets.Button(description="← Previous")
next_btn = widgets.Button(description="Next →")
counter = widgets.HTML()

image_number_input = widgets.BoundedIntText(
    value=1,
    min=1,
    max=len(explore_images),
    description="Image #:",
    layout=widgets.Layout(width="190px")
)
go_btn = widgets.Button(description="Go")

# Annotation visibility controls
show_boxes = widgets.Checkbox(
    value=True,
    description="Show bounding boxes",
    indent=False,
    layout=widgets.Layout(width="190px")
)
show_masks = widgets.Checkbox(
    value=True,
    description="Show segmentation masks",
    indent=False,
    layout=widgets.Layout(width="210px")
)

output = widgets.Output()
state = {"index": 0}


def show_image(index):
    row = explore_images.iloc[index]
    photo = np.array(Image.open(row["path"]).convert("RGB"))
    shown = photo.copy()

    anns = annotations[annotations["image_id"] == row["id"]]
    overlays = []

    for _, ann in anns.iterrows():
        color = CLASS_COLORS[ann["class"]]

        # Decode and overlay masks only when enabled
        if show_masks.value:
            mask = decode_mask(
                ann["segmentation"],
                int(row["height"]),
                int(row["width"])
            )

            rgb = np.array([
                int(color[i:i+2], 16) for i in (1, 3, 5)
            ])

            shown[mask] = (
                0.45 * shown[mask] + 0.55 * rgb
            ).astype(np.uint8)

        overlays.append((ann, color))

    with output:
        clear_output(wait=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(shown)

        # Draw bounding boxes only when enabled
        if show_boxes.value:
            for ann, color in overlays:
                x, y, w, h = ann["bbox"]
                ax.add_patch(
                    patches.Rectangle(
                        (x, y), w, h,
                        fill=False,
                        edgecolor=color,
                        linewidth=2.5
                    )
                )

        ax.set_title(
            f"Class: {row['class']} | "
            f"Annotated animals: {len(anns)}"
        )
        ax.axis("off")
        plt.show()
        plt.close(fig)

        print(
            f"Image number: {index + 1} | "
            f"Size: {row['width']} × {row['height']} px"
        )

        for ann, _ in overlays:
            print(
                f"{ann['class']}: "
                f"box = {[round(v) for v in ann['bbox']]} px"
            )

    # Update navigation controls
    counter.value = (
        f"<b>Image {index + 1} / {len(explore_images)}</b>"
    )
    image_number_input.value = index + 1

    previous_btn.disabled = index == 0
    next_btn.disabled = index == len(explore_images) - 1


def navigate(change):
    state["index"] += change
    show_image(state["index"])


def go_to_image(_):
    state["index"] = image_number_input.value - 1
    show_image(state["index"])


previous_btn.on_click(lambda _: navigate(-1))
next_btn.on_click(lambda _: navigate(1))
go_btn.on_click(go_to_image)

# Refresh the image immediately when a checkbox changes
show_boxes.observe(
    lambda change: show_image(state["index"]),
    names="value"
)
show_masks.observe(
    lambda change: show_image(state["index"]),
    names="value"
)

# Display interface
display(
    widgets.HBox([previous_btn, counter, next_btn]),
    widgets.HBox([image_number_input, go_btn]),
    widgets.HBox([show_boxes, show_masks]),
    output
)

show_image(0)


# %%
# @title 💡 Sample Answer { display-mode: "form" }
from IPython.display import HTML

HTML(r"""
<div style="background:#F5F5F5;color:#374151;border-left:6px solid #9CA3AF;
            border-radius:6px;padding:12px 14px;line-height:1.5;
            font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;
            font-size:14px;width:100%;box-sizing:border-box;margin:2px 0;">

<details style="margin:0;">
<summary style="cursor:pointer;font-weight:bold;">💡 Sample Answer</summary>
<p style="margin:6px 0 0 0;">
<strong>Class labels</strong> identify the animal category.
<strong>Bounding boxes</strong> indicate where animals are located using rectangles,
while <strong>segmentation masks</strong> outline their shapes more precisely.
Annotations are not always accurate: some masks extend beyond the animal,
some body parts are missing, and groups of animals may be annotated inconsistently.
</p>
</details>
</div>
""")

# %% [markdown]
# #### 🔴 Image Classification: One Label per Image
#
# Image classification assigns one label to an entire image, even if it contains multiple animals or different classes. What information is lost, and when might this become a problem?

# %%
# @title 💡 Sample Answer { display-mode: "form" }
from IPython.display import HTML

HTML(r"""
<div style="background:#F5F5F5;color:#374151;border-left:6px solid #9CA3AF;
            border-radius:6px;padding:12px 14px;line-height:1.5;
            font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;
            font-size:14px;width:100%;box-sizing:border-box;margin:2px 0;">

<details style="margin:0;">
<summary style="cursor:pointer;font-weight:bold;">💡 Sample Answer</summary>
<p style="margin:6px 0 0 0;">
A single class label does not indicate how many animals are present,
where they are located, or whether other classes appear in the image.
The labeled animal may also be very small or in the background,
while a much larger object dominates the image.
This can make classification difficult because the label may not reflect
the most prominent image content. Image classification is also unsuitable
when the goal is to locate or count individual animals.
</p>
</details>
</div>
""")
