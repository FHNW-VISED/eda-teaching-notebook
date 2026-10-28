# %% [markdown] edition="teaching"
# ## Meet the Park: Explore Your Dataset
#
# Wildwood Farm Park wants to monitor animals across several enclosures using fixed outdoor cameras. The park has sheep and horses in its paddocks, birds near the aviary pond, zebras in the savanna area, and cats and dogs around the visitor yard.
#
# The park ranger wants to use computer vision for three different tasks:

# %% [markdown] edition="own"
# ## Meet Your Data: Explore Your Dataset
#
# Your project uses computer vision to solve one task, on images you collected or found. Before you train a model, check whether these images and their labels fit that task.
#
# *In the park:* the ranger wanted to recognize dogs in the sheep paddock, count animals, and see how they use their enclosures, all with fixed outdoor cameras.
#
# Computer vision tasks come in three kinds:

# %% edition="teaching"
# @title 📘 Computer Vision Tasks { display-mode: "form" }
callout("theory", "📘 Three Computer Vision Tasks", r"""
<ol style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><strong>Classification – Recognize:</strong> What animal is shown in the image? The park ranger wants to identify dogs entering the sheep paddock so they can intervene before the sheep are harmed.</li>
<li style="margin:2px 0;"><strong>Object Detection – Count:</strong> How many sheep, zebras, or birds are visible, and where are they located?</li>
<li style="margin:2px 0;"><strong>Segmentation – Monitor Space Usage:</strong> Which areas of the image are occupied by animals? The park ranger wants to understand how animals use the space in their enclosures.</li>
</ol>
""")


# %% edition="own"
# @title 📘 Computer Vision Tasks { display-mode: "form" }
callout("theory", "📘 Three Computer Vision Tasks", r"""
<ol style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><strong>Classification – Recognize:</strong> Which class does the image show? One label per image.</li>
<li style="margin:2px 0;"><strong>Object Detection – Count:</strong> How many objects are visible, and where are they located? One box per object.</li>
<li style="margin:2px 0;"><strong>Segmentation – Map Pixels:</strong> Which areas of the image belong to which class? One mask per object.</li>
</ol>
""")

# %% [markdown] edition="teaching"
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
# - `images`: One row per image, including its dimensions, animal class, and number of annotated animals (`n_objects`).
# - `annotations`: One row per annotated animal, including its bounding box and segmentation mask.

# %% [markdown] edition="own"
# Your dataset is organized into the same two tables the park used:
#
# - `images`: One row per image, including its dimensions, class, and number of annotated objects (`n_objects`).
# - `annotations`: One row per annotated object, with its bounding box and segmentation mask if your labels have them.
#
# ### 🟢 Your Task: Which Mistake Is Expensive?
#
# A model makes mistakes, and some of them cost more than others.
#
# **Question:**
# - Which of the three tasks does your project solve?
# - Which mistake would be the expensive one, and for whom?
#
# ✍️ **Your answer** (double-click to edit):
#
# *In the park:* the task was classification, and the expensive mistake was a missed dog in the sheep paddock.

# %%
# @title 🧾 Karpathy's Recipe · Step 1 { display-mode: "form" }
callout("recipe", "🧾 Recipe · Step 1: Become one with the data", r"""
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

# %% [markdown] edition="teaching"
#
# #### 🟢 Images: What Do Images Look Like Across Classes?
#
# Let's start with a quick visual overview of the Wildwood animal dataset. The next cell displays six example images from each animal class to give you a first impression of the dataset.
#
# **Observe:**
# - How do the images differ across classes?
# - How much variation is there within each class (e.g., backgrounds, viewpoints, animal sizes)?
# - Which images might be challenging for an image classification model, and why?

# %% [markdown] edition="own"
# #### 🟢 Images: What Do Images Look Like Across Classes?
#
# Let's start with a quick visual overview of your dataset. The next cell displays six example images from each class to give you a first impression.
#
# **Observe:**
# - How do the images differ across classes?
# - How much variation is there within each class (e.g., backgrounds, viewpoints, object sizes)?
# - Which images might be challenging for an image classification model, and why?

# %%
# @title 🔎 Explore Example Images per Class { display-mode: "form" }
_per_class = 6
_blocks = [CLASSES[k:k + 6] for k in range(0, len(CLASSES), 6)]  # at most six classes side by side
fig, axes = plt.subplots(_per_class * len(_blocks), min(6, len(CLASSES)),
                         figsize=(13, 2.2 * _per_class * len(_blocks)), squeeze=False)
for ax in axes[:, len(_blocks[-1]):][-_per_class:].ravel():  # the last row of classes may be shorter
    ax.axis("off")
for j, c in enumerate(CLASSES):
    _of_class = images[images["class"] == c]
    _picked = _of_class.sample(min(_per_class, len(_of_class)), random_state=SEED)
    for i in range(_per_class):
        ax = axes[(j // 6) * _per_class + i, j % 6]
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(CLASS_COLORS[c])
            spine.set_linewidth(4)
        if i < len(_picked):
            _thumb = Image.open(_picked.iloc[i]["path"]).convert("RGB")
            _thumb.thumbnail((256, 256))
            ax.imshow(_thumb)
        if i == 0:
            ax.set_title(c, color=INK, fontweight="bold")
fig.tight_layout()
plt.show()

# %% [markdown] edition="teaching"
# Select an animal class and browse at least twelve images to explore variation within the class. Pay attention to backgrounds, lighting, viewpoints, animal sizes, and other objects in the images. Look for recurring patterns and unusual examples. If an animal is difficult to spot, enable the bounding boxes.
#
# **Observe:**
# - How much do images vary within each class?
# - Which image characteristics might make classification challenging?
# - Which classes contain images that are least representative of the park's fixed outdoor cameras?

# %% [markdown] edition="own"
# Select a class and browse at least twelve images to explore variation within the class. Pay attention to backgrounds, lighting, viewpoints, object sizes, and other objects in the images. Look for recurring patterns and unusual examples. If an object is difficult to spot and your dataset has boxes, enable the bounding boxes.
#
# **Observe:**
# - How much do images vary within each class?
# - Which image characteristics might make classification challenging?
# - Which classes contain images that look least like the images your model will see in use?

# %%

# @title 🔎 Browse Images Within a Class { display-mode: "form" }
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
        photo = Image.open(row["path"]).convert("RGB")
        ax.imshow(photo)

        if show_boxes:
            anns = annotations[annotations["image_id"] == row["id"]]
            for _, ann in anns.iterrows():
                x, y, w, h = ann["bbox"]
                ax.add_patch(
                    mpl.patches.Rectangle(
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

_has_boxes = bool(annotations["bbox"].notna().any())  # no boxes in the labels: no box checkbox
display(
    widgets.HBox([class_dropdown, page_slider] + ([box_checkbox] if _has_boxes else [])),
    out
)


# %% [markdown] edition="teaching"
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

# %% [markdown] edition="own"
# #### 🟢 Document Your Observations
#
# Karpathy's first step also involves documenting what you discover. Write at least three specific observations about the images you explored in the next markdown cell. For each observation, briefly explain what you should investigate further.
#
# **Hint:** Be specific. *In the park*, for example:
#
# - Vague: "Dog images are different."
# - Specific: "Many dog images are taken indoors, often with people in the background. This may not represent the park's outdoor cameras."
#
# Consider backgrounds, lighting, object sizes, distances, multiple objects, people, and unusual examples.

# %% [markdown]
# **My three observations** (double-click to edit):
#
# 1. …
# 2. …
# 3. …

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
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
""")


# %% [markdown] edition="own"
# *In the park:* many dog photos were taken indoors or on streets, often with people or leashes in the frame, and one photo in the sheep class showed ibexes.

# %% [markdown] edition="teaching"
#
# #### 🟢 Dataset Suitability: How Suitable Are These Images for the Park?
#
# The park's cameras are fixed, outdoors, and often far from the animals.
#
# **Question:**
# Based on the images you explored, which animal classes seem least representative of the park's camera views, and why? Briefly justify your answer.
#

# %% [markdown] edition="own"
# #### 🟢 Dataset Suitability: How Suitable Are These Images for Your Task?
#
# Think about where your model will be used: which camera, which setting, and how far from the objects.
#
# **Question:**
# Based on the images you explored, which classes seem least representative of the images your model will see in use, and why? Briefly justify your answer.

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p style="margin:6px 0 0 0;">
Cats and dogs often appear in indoor, close-up photos with people or furniture,
making them less representative of the park's fixed outdoor cameras.
Sheep, horses, and zebras are more often shown outdoors and at greater distances,
making their images somewhat closer to the park's camera views.
</p>
""")


# %% [markdown] edition="own"
# ✍️ **Your answer** (double-click to edit):
#
# *In the park:* cats and dogs were the least representative, with indoor close-ups next to people or furniture, while the park's cameras are fixed, outdoors, and far from the animals.

# %% [markdown] edition="teaching"
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

# %% [markdown] edition="own"
# #### 🟢 Annotations: What Does an Image Annotation Look Like?
#
# An image can have different types of annotations, such as class labels, bounding boxes, and segmentation masks. Your dataset has the ones you listed in `TASKS`.
#
# Let's visualize an example from your dataset and examine what information each annotation carries.
#
# **Explore the annotations**
#
# Run the next cell to visualize an image with its class label, and with its bounding boxes and segmentation masks if your dataset has them.
#
# **Observe:**
# - What information does each type of annotation provide?
# - If you have boxes or masks, how do they differ?
# - Do the annotations correctly represent the objects in the image?

# %%

# @title 🔎 Explore Images and Annotations { display-mode: "form" }
# Shuffle images reproducibly
ann_images = images.sample(
    frac=1, random_state=SEED
).reset_index(drop=True)

# Navigation widgets
ann_prev_btn = widgets.Button(description="← Previous")
ann_next_btn = widgets.Button(description="Next →")
ann_counter = widgets.HTML()

ann_number_input = widgets.BoundedIntText(
    value=1,
    min=1,
    max=len(ann_images),
    description="Image #:",
    layout=widgets.Layout(width="190px")
)
ann_go_btn = widgets.Button(description="Go")

# Annotation visibility controls (shown only for what the labels contain)
ann_has_boxes = bool(annotations["bbox"].notna().any())
ann_has_masks = bool(annotations["segmentation"].notna().any())
ann_box_toggle = widgets.Checkbox(
    value=ann_has_boxes,
    description="Show bounding boxes",
    indent=False,
    layout=widgets.Layout(width="190px")
)
ann_mask_toggle = widgets.Checkbox(
    value=ann_has_masks,
    description="Show segmentation masks",
    indent=False,
    layout=widgets.Layout(width="210px")
)

ann_output = widgets.Output()
ann_state = {"index": 0}


def ann_show_image(index):
    row = ann_images.iloc[index]
    photo = np.array(Image.open(row["path"]).convert("RGB"))
    shown = photo.copy()

    anns = annotations[annotations["image_id"] == row["id"]]
    overlays = []

    for _, ann in anns.iterrows():
        color = CLASS_COLORS[ann["class"]]

        # Decode and overlay masks only when enabled
        if ann_mask_toggle.value and ann["segmentation"] is not None:
            mask = decode(to_rle(
                ann["segmentation"],
                int(row["height"]),
                int(row["width"])
            )).astype(bool)

            rgb = np.array([
                int(color[i:i+2], 16) for i in (1, 3, 5)
            ])

            shown[mask] = (
                0.45 * shown[mask] + 0.55 * rgb
            ).astype(np.uint8)

        overlays.append((ann, color))

    with ann_output:
        clear_output(wait=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.imshow(shown)

        # Draw bounding boxes only when enabled
        if ann_box_toggle.value:
            for ann, color in overlays:
                if ann["bbox"] is None:
                    continue
                x, y, w, h = ann["bbox"]
                ax.add_patch(
                    mpl.patches.Rectangle(
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

        for ann, _ in (o for o in overlays if o[0]["bbox"] is not None):
            print(
                f"{ann['class']}: "
                f"box = {[round(v) for v in ann['bbox']]} px"
            )

    # Update navigation controls
    ann_counter.value = (
        f"<b>Image {index + 1} / {len(ann_images)}</b>"
    )
    ann_number_input.value = index + 1

    ann_prev_btn.disabled = index == 0
    ann_next_btn.disabled = index == len(ann_images) - 1


def ann_navigate(change):
    ann_state["index"] += change
    ann_show_image(ann_state["index"])


def ann_go_to_image(_):
    ann_state["index"] = ann_number_input.value - 1
    ann_show_image(ann_state["index"])


ann_prev_btn.on_click(lambda _: ann_navigate(-1))
ann_next_btn.on_click(lambda _: ann_navigate(1))
ann_go_btn.on_click(ann_go_to_image)

# Refresh the image immediately when a checkbox changes
ann_box_toggle.observe(
    lambda change: ann_show_image(ann_state["index"]),
    names="value"
)
ann_mask_toggle.observe(
    lambda change: ann_show_image(ann_state["index"]),
    names="value"
)

# Display interface
display(
    widgets.HBox([ann_prev_btn, ann_counter, ann_next_btn]),
    widgets.HBox([ann_number_input, ann_go_btn]),
    widgets.HBox([t for t, has in ((ann_box_toggle, ann_has_boxes), (ann_mask_toggle, ann_has_masks)) if has]),
    ann_output
)

ann_show_image(0)


# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p style="margin:6px 0 0 0;">
<strong>Class labels</strong> identify the animal category.
<strong>Bounding boxes</strong> indicate where animals are located using rectangles,
while <strong>segmentation masks</strong> outline their shapes more precisely.
Annotations are not always accurate: some masks extend beyond the animal,
some body parts are missing, and groups of animals may be annotated inconsistently.
</p>
""")


# %% [markdown] edition="own"
# ✍️ **Your answer** (double-click to edit):
#
# *In the park:* some masks extended beyond the animal, some body parts were missing, and groups of animals were annotated inconsistently.

# %% [markdown]
# #### 🔴 Image Classification: One Label per Image (Optional)
#
# Image classification assigns one label to an entire image, even if it contains multiple animals or different classes. What information is lost, and when might this become a problem?

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<p style="margin:6px 0 0 0;">
A single class label does not indicate how many animals are present,
where they are located, or whether other classes appear in the image.
The labeled animal may also be very small or in the background,
while a much larger object dominates the image.
This can make classification difficult because the label may not reflect
the most prominent image content. Image classification is also unsuitable
when the goal is to locate or count individual animals.
</p>
""")



# %% [markdown] edition="own"
# ✍️ **Your answer** (double-click to edit):
#
# *In the park:* some images labeled bird showed a tiny bird next to a large bear, so the label did not match the most prominent content.
