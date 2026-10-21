# %% [markdown]
# ## 2. Become one with the data

# %% [markdown]
# <!-- callout: objective | Objective -->
# Look at many images before computing anything, and write down what you see.

# %% [markdown]
# Section 1 ended with three 👀 questions that no table can answer. Your eyes are the only tool for them, and
# most people skip this step:
#
# - 👀 Does the data **fit** the park's cameras?
# - 👀 Does its **variability** match what the cameras will see?
# - 👀 Are the animals **identifiable by eye**?
#
# Karpathy puts this step first in his recipe:

# %% [markdown]
# <!-- callout: recipe | Recipe · Step 1: Become one with the data -->
# *"The first step to training a neural net is to not touch any neural net code at all and instead begin by
# thoroughly inspecting your data. This step is critical. I like to spend copious amount of time (measured in
# units of hours) scanning through thousands of examples, understanding their distribution and looking for
# patterns."*
#
# Andrej Karpathy, [A Recipe for Training Neural
# Networks](https://karpathy.github.io/2019/04/25/recipe/) (2019).

# %% [markdown]
# We will not spend hours, but we will spend a few minutes. The first cell draws one row of photos per class,
# framed in the class colour used in every plot of this notebook. Scroll slowly and look at:
#
# - the backgrounds and the lighting,
# - the distance to the animal,
# - anything else in the photo.

# %%
# @title Grid helpers { display-mode: "form" }
from IPython.display import display
from PIL import Image


def load_thumb(path, size: int = 256) -> Image.Image:
    """Open an image as a small RGB thumbnail (fast: JPEG decodes at reduced size)."""
    im = Image.open(path)
    im.draft("RGB", (size, size))
    im = im.convert("RGB")
    im.thumbnail((size, size))
    return im


def sample_rows(class_name: str, n: int) -> pd.DataFrame:
    """A fixed random sample of `n` rows of `images` for one class (seeded: re-runs show the same photos)."""
    of_class = images[images["class"] == class_name]
    n = min(n, len(of_class))
    idx = random.Random(SEED).sample(sorted(of_class.index), n)
    return of_class.loc[sorted(idx)]


def frame(ax, color: str, width: float = 4) -> None:
    """Draw a coloured frame around an image axis and hide ticks and grid."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(width)


def show_grid(class_name: str, n: int, cols: int = 6, label_animals: bool = True) -> None:
    """Show `n` photos of one class in a grid, framed in CLASS_COLORS[class_name]."""
    rows_df = sample_rows(class_name, n)
    n = len(rows_df)
    n_rows = max(1, -(-n // cols))
    fig, axes = plt.subplots(n_rows, cols, figsize=(2.2 * cols, 2.4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, row) in zip(axes, rows_df.iterrows(), strict=False):
        ax.axis("on")
        ax.imshow(load_thumb(row["path"]))
        frame(ax, CLASS_COLORS[class_name])
        if label_animals:
            plural = "s" if row["n_animals"] != 1 else ""
            ax.set_title(f"{row['n_animals']} {class_name}{plural}", fontsize=8)
    fig.suptitle(f"{class_name} · {n} of {int((images['class'] == class_name).sum())} photos", color=INK)
    fig.tight_layout()
    display(fig)
    plt.close(fig)


def show_all_classes(n_per_class: int) -> None:
    """One row per class, `n_per_class` photos each, every row framed in its class colour."""
    fig, axes = plt.subplots(len(CLASSES), n_per_class, figsize=(2.0 * n_per_class, 2.1 * len(CLASSES)))
    axes = np.atleast_2d(axes)
    for r, class_name in enumerate(CLASSES):
        rows_df = sample_rows(class_name, n_per_class)
        for c in range(n_per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(rows_df):
                ax.axis("on")
                ax.imshow(load_thumb(rows_df.iloc[c]["path"]))
                frame(ax, CLASS_COLORS[class_name])
            if c == 0:
                ax.set_ylabel(class_name, color=CLASS_COLORS[class_name], fontsize=12, fontweight="bold")
    fig.tight_layout()
    display(fig)
    plt.close(fig)


# %% [markdown]
# <!-- callout: questions | Questions -->
# Predict before you run: the park's cameras are fixed, outdoors, and often far from the animal.

# %% [markdown]
# **Q2.** 🟢 Which class do you expect to look *least* like a fixed park camera's view, and why? One line,
# then run the cell.

# %%
# @title Q2 Show one row of photos per class { display-mode: "form" }
# A fixed sample of photos per class, framed in the class colour used throughout the notebook.
N_PER_CLASS = 3 if SMOKE else 6  # fewer thumbnails in smoke mode, same code (N-014)
show_all_classes(N_PER_CLASS)

# %% [markdown]
# 💡 Even six photos per class show that these are holiday snapshots, not camera-trap frames: close-ups,
# people, sofas and streets, with the animal usually centred and filling the frame.
#
# **Worked answer (Q2).** Cats and dogs are the farthest from a park camera: they are photographed indoors,
# up close, often with people and furniture. Sheep, horses and zebras are at least outdoors and sometimes far
# away, which is closer to what the paddock and savanna cameras will see.
#
# Now look at more photos of one class: twelve is enough to start noticing patterns.

# %% [markdown]
# <!-- callout: your-turn | Your turn · Pick a class -->
# Set `CLASS_TO_SHOW` to one of `cat, dog, bird, horse, sheep, zebra` and `N_SHOW` to how many photos you
# want, then run the cell. Try at least `"dog"` (the alert class) and `"bird"` (the tiny-box class).

# %%
# ✏️ YOUR TURN
CLASS_TO_SHOW = "dog"
N_SHOW = 12

if CLASS_TO_SHOW not in CLASSES:
    raise SystemExit(f"CLASS_TO_SHOW must be one of {CLASSES}, not {CLASS_TO_SHOW!r}")
if not isinstance(N_SHOW, int) or N_SHOW < 1:
    raise SystemExit(f"N_SHOW must be a positive integer, not {N_SHOW!r}")
n_show = min(N_SHOW, 6) if SMOKE else N_SHOW  # smoke mode caps the grid, same code path (N-014)
show_grid(CLASS_TO_SHOW, n_show)

# %% [markdown]
# 💡 The title of each photo says how many animals of that class it holds: more than one is common, so "one
# label per image" is already a simplification, and section 4 asks which label a photo with several animals
# should get.

# %% [markdown]
# <!-- callout: questions | Questions -->
# Karpathy's step ends with writing things down. Do the same: edit the next markdown cell (double-click it).

# %% [markdown]
# **Q3.** 🟢 Write three observations about the photos you just looked at. Good observations are specific
# ("many dog photos are indoors with a person's legs in the frame"), and each one hints at something the park
# should worry about or check later (fit to the cameras, lighting, distance, several animals, people, odd
# photos).

# %% [markdown]
# **My three observations** (double-click to edit):
#
# 1. …
# 2. …
# 3. …

# %% [markdown]
# **Worked answer (Q3)**, for the dog grid on the full set. Yours will differ; that is the point.
#
# 1. **Setting:** most dog photos are indoors or in a street, taken by the owner from a metre away. A paddock
#    camera sees a small dog far away in grass. The model will have to generalise from close-ups to that
#    (section 11's question "does it fit?").
# 2. **People:** a person, a hand or a leash appears in many photos. A model can learn "person + leash = dog"
#    and miss the stray dog alone in the paddock.
# 3. **Several animals and odd photos:** some photos hold two or three dogs, some show a dog in a costume,
#    on a skateboard or on a screen. Each of these is a labelling decision (sections 4 and 5) before it is a
#    modelling problem.
#
# Keep your three lines: section 11 asks you for the same three on your own dataset, and the rest of this
# notebook turns the observations above into numbers, one at a time.

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **When you see** a pattern in the photos (indoors, people, several animals) → **do** write it down as a
#   specific observation, and turn it into a check later.
# - **When you see** a gap between the photos and the deployment cameras (distance, lighting, setting) →
#   **do** flag it now, before any model exists.
# - **When you see** several animals or an odd subject in one photo → **do** note it as a labelling decision
#   for sections 4 and 5.
