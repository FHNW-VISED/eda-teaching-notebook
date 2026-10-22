# %% [markdown]
# ## 2. Become one with the data

# %% [markdown]
# <!-- callout: objective | Objective -->
# Look at many images before computing anything, and write down what you see.

# %% [markdown]
# Section 1 left three 👀 questions only your eyes can answer (most people skip this step). Karpathy puts
# this step first in his recipe:

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
# <!-- callout: go-deeper | Go deeper · The three 👀 questions from section 1 -->
# - 👀 Does the data **fit** the park's cameras?
# - 👀 Does its **variability** match what the cameras will see?
# - 👀 Are the animals **identifiable by eye**?

# %% [markdown]
# We will not spend hours, but we will spend a few minutes. Scroll back to the class grid of section 1 (six
# photos per class, framed in the class colour used in every plot of this notebook) and look at:
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


# %% [markdown]
# <!-- callout: questions | Questions -->
# The park's cameras are fixed, outdoors, and often far from the animal.

# %% [markdown]
# **Q2.** 🟢 Look at the class grid above (section 1). Which class looks *least* like a fixed park camera's
# view, and why? One line.

# %% [markdown]
# <!-- callout: answer | Worked answer (Q2) -->
# Holiday snapshots, not camera-trap frames. Cats and dogs are the farthest from a
# park camera: indoors, up close, with people and sofas. Sheep, horses and zebras are at least outdoors and
# sometimes far away, closer to what the paddock and savanna cameras will see.

# %% [markdown]
# <!-- callout: your-turn | Your turn · Pick a class -->
# Look at more photos of one class: twelve is enough to start noticing patterns. **Look for:** which class has
# the most photos that a paddock camera would never take?
#
# Set `CLASS_TO_SHOW` (`cat, dog, bird, horse, sheep, zebra`) and `N_SHOW` (how many photos you want), then
# run the cell. Try `"dog"` (the alert class) and `"bird"` (the tiny-box class).

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
# 💡 Each title says how many animals of that class the photo holds. More than one is common (section 4).

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · One label per image is a simplification -->
# A photo with several animals still gets one label, so "one label per image" already hides something
# (section 4).

# %% [markdown]
# <!-- callout: questions | Questions -->
# Karpathy's step ends with writing things down. Edit the next markdown cell (double-click it).

# %% [markdown]
# **Q3.** 🟢 Write three specific observations about the photos you just looked at, each one hinting at
# something the park should check later.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · What a good observation looks like -->
# Specific beats vague: "many dog photos are indoors with a person's legs in the frame", not "dogs look
# different". Good topics: fit to the cameras, lighting, distance, several animals, people, odd photos.

# %% [markdown]
# **My three observations** (double-click to edit):
#
# 1. …
# 2. …
# 3. …

# %% [markdown]
# <!-- callout: answer | Worked answer (Q3) -->
# , for the dog grid on the full set. Yours will differ; that is the point.
#
# 1. **Setting:** most dogs are indoors or in a street, a metre away; the paddock camera sees one far off.
# 2. **People:** a person, hand or leash is in many photos: "person + leash = dog" is a shortcut to learn.
# 3. **Odd photos:** two or three dogs, a costume, a screen: labelling decisions (sections 4 and 5).
#
# Keep your three lines: section 11 asks for the same three on your own dataset.

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **When you see** a pattern in the photos (indoors, people, several animals) → **do** write it down as a
#   specific observation, and turn it into a check later.
# - **When you see** a gap between the photos and the park's cameras (distance, lighting, setting) → **do**
#   flag it now, before any model exists.
# - **When you see** several animals or an odd subject in one photo → **do** note it as a labelling decision
#   for sections 4 and 5.
