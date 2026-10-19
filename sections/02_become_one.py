# %% [markdown]
# ## 2. Become one with the data
#
# **Objective:** look at many images before computing anything, and write down what you see.
#
# Section 1 ended with three 👀 bullets that no table can answer: does the data *fit* the park's cameras, does
# its *variability* match what the cameras will see, and are the animals *identifiable by eye*? The only tool
# for those is your eyes, and this is the step most people skip. Karpathy puts it first in his recipe:
#
# <table width="100%"><tr><td bgcolor="#E8F5E9"><font color="#1B5E20">
# <b>🧾 Recipe · Step 1: Become one with the data</b><br>
# <i>"The first step to training a neural net is to not touch any neural net code at all and instead begin by
# thoroughly inspecting your data. This step is critical. I like to spend copious amount of time (measured in
# units of hours) scanning through thousands of examples, understanding their distribution and looking for
# patterns."</i><br>
# Andrej Karpathy, <a href="https://karpathy.github.io/2019/04/25/recipe/">A Recipe for Training Neural
# Networks</a> (2019).
# </font></td></tr></table>
#
# We will not spend hours, but we will spend a few minutes. The first cell draws one row of photos per class,
# each framed in its class colour (the same colour every plot in this notebook uses for that class). Scroll
# slowly: look at the backgrounds, the distance to the animal, the lighting, and whether anything else is in
# the photo.

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
# <table width="100%"><tr><td bgcolor="#E0F7F4"><font color="#00564D">
# <b>❓ Questions</b><br>
# Predict before you run: the park's cameras are fixed, outdoors, and often far from the animal.
# </font></td></tr></table>
#
# **Q2.** 🟢 Which class do you expect to look *least* like a fixed park camera's view, and why? One line,
# then run the cell.

# %%
# Q2  One row per class: a fixed sample of photos, framed in the class colour used throughout the notebook.
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
# Now pick one class and look at more of it. Twelve photos is enough to start noticing patterns; raise it if
# you have time.
#
# <table width="100%"><tr><td bgcolor="#FFF4D6"><font color="#7A4A00">
# <b>✏️ Your turn · Pick a class</b><br>
# Set <code>CLASS_TO_SHOW</code> to one of <code>cat, dog, bird, horse, sheep, zebra</code> and
# <code>N_SHOW</code> to how many photos you want (up to the class size), then run the cell. Try at least
# <code>"dog"</code> (the alert class) and <code>"bird"</code> (the tiny-box class).
# </font></td></tr></table>

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
#
# <table width="100%"><tr><td bgcolor="#E0F7F4"><font color="#00564D">
# <b>❓ Questions</b><br>
# Karpathy's step ends with writing things down. Do the same: edit the next markdown cell (double-click it).
# </font></td></tr></table>
#
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
