# %% [markdown]
# # Know Your Data: Exploring and Preparing Image Datasets at the Wildwood Farm Park

# %% [markdown]
# <!-- callout: objective | Objective -->
# Run the setup, meet the park, and see the three things you will be able to do at the end.

# %% [markdown]
# <!-- callout: banner | Know Your Data: Exploring and Preparing Image Datasets at the Wildwood Farm Park -->
# **At the end of this notebook you will be able to:**
#
# 1. Inspect an image dataset qualitatively and quantitatively, and say whether it fits a task.
# 2. Find and fix common dataset errors (duplicates, flipped/rotated copies, bad labels or masks) before
#    they leak into training.
# 3. Choose a split and a metric that match the use case, and explain how imbalance, object size and leakage
#    look different for classification, detection and segmentation.
#
# Questions are marked 🟢 beginner · 🟡 intermediate · 🔴 advanced (optional).

# %% [markdown]
# **Before you start:** you need a Google account (this notebook runs in Colab, a CPU runtime is enough),
# Python basics, and the lecture on image datasets. No model is trained here; that is next week.
#
# The notebook follows the first part of Karpathy's
# [*A Recipe for Training Neural Networks*](https://karpathy.github.io/2019/04/25/recipe/):
# know your data before you touch a model.

# %% [markdown]
# ## 0. Setup & the park

# %% [markdown]
# <!-- callout: theory | The Wildwood Farm Park -->
# The park has cameras in four places:
#
# - 🐑🐴 **Farm paddock**: sheep, horses
# - 🐦 **Aviary pond**: birds
# - 🦓 **Savanna enclosure**: zebras
# - 🐱🐶 **Visitor yard**: cats and dogs wander through
#
# It wants three things from them:
#
# 1. **Recognise** (classification): "who is in front of this camera?" The case that matters is **a dog in
#    the sheep paddock**, which triggers an alert. Missing that dog is worse than a false alarm.
# 2. **Count** (detection): a daily headcount of sheep, zebras and birds. Far-away birds are tiny.
# 3. **Outline** (segmentation): how much of each enclosure each animal uses. Birds cover few pixels.

# %% [markdown]
# The park has no training data yet, so we practise on a **Play Dataset**: COCO 2017 photos of the same six
# animals (cat, dog, bird, horse, sheep, zebra), each photo showing one animal class.
#
# Everything you learn here is a checklist you will reuse on your own project dataset in the last section.

# %% [markdown]
# **Run the next cell once.** It installs the one package Colab lacks, fixes the random seed, sets the
# notebook's colours and defines `use()`, the helper that lets you write small functions without ever
# blocking "Run all".

# %%
# @title Setup (run me first) { display-mode: "form" }
import ast
import inspect
import os
import random
import sys
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa: F401  (every later section plots with plt)
import numpy as np

IN_COLAB = "google.colab" in sys.modules
SMOKE = os.environ.get("NB_SMOKE") == "1"  # automated tests: tiny data, CPU, seconds. Never set this by hand.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

if IN_COLAB:  # Jupytext turns the commented magic below into a real `%pip install -q imagehash` (N-021)
    pass
    # %pip install -q imagehash

# --- Theme (SPEC L-003): white background, dark teal text, light grid, one colour per class everywhere ---
CLASSES = ["cat", "dog", "bird", "horse", "sheep", "zebra"]
CLASS_COLORS = {
    "cat": "#E07A5F",
    "dog": "#3D405B",
    "bird": "#F2CC8F",
    "horse": "#81B29A",
    "sheep": "#9C89B8",
    "zebra": "#2A9D8F",
}
INK = "#21434A"
mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": "#DDE5E7",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=[CLASS_COLORS[c] for c in CLASSES]),
        "figure.dpi": 100,
        "font.size": 11,
    }
)

# --- Results every section writes into (SPEC §9) ---
RESULTS = {}
RESULTS["class_colors"] = dict(CLASS_COLORS)


# --- use(): run a learner's function next to the reference solution (N-029) ---
def _is_stub(fn) -> bool:
    """True if fn's body is only a docstring plus `...` or `pass` (the learner has not written it yet)."""
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
        body = node.body
    except (OSError, TypeError, SyntaxError, IndexError, AttributeError):
        return False
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0].value, "value", None), str):
        body = body[1:]  # skip the docstring
    return all(
        isinstance(n, ast.Pass) or (isinstance(n, ast.Expr) and getattr(n.value, "value", None) is Ellipsis)
        for n in body
    )


def _same(a, b) -> bool:
    if hasattr(a, "equals"):  # pandas
        return bool(a.equals(b))
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return bool(np.array_equal(a, b))
        return bool(a == b)
    except Exception:
        return False


def _short(x, n=60) -> str:
    s = repr(x)
    return s if len(s) <= n else s[: n - 1] + "…"


def _diff(mine, ref) -> str:
    """One short line on how `mine` differs from `ref` (e.g. 'cat: 12 vs 14')."""
    if isinstance(mine, dict) and isinstance(ref, dict):
        keys = list(ref) + [k for k in mine if k not in ref]
        bad = [k for k in keys if not _same(mine.get(k, "—"), ref.get(k, "—"))]
        return ", ".join(f"{k}: {_short(mine.get(k, '—'))} vs {_short(ref.get(k, '—'))}" for k in bad[:4])
    if isinstance(mine, (list, tuple, set)) and isinstance(ref, (list, tuple, set)):
        try:
            extra, missing = set(mine) - set(ref), set(ref) - set(mine)
            parts = []
            if missing:
                parts.append(f"missing {_short(sorted(missing, key=str))}")
            if extra:
                parts.append(f"extra {_short(sorted(extra, key=str))}")
            return " · ".join(parts) or f"same items, different order or length ({len(mine)} vs {len(ref)})"
        except TypeError:
            pass
    return f"{_short(mine)} vs {_short(ref)}"


def use(fn, *args, reference, **kwargs):
    """Run `fn(*args, **kwargs)` next to `reference(*args, **kwargs)`; ALWAYS return the reference's result.

    - stub (body is `...`) or NotImplementedError → "using the reference solution";
    - otherwise ✅ when both agree, ❌ with a short diff when they do not.
    The notebook continues with the reference either way, so "Run all" never breaks on your code.
    """
    expected = reference(*args, **kwargs)
    name = getattr(fn, "__name__", "your function")
    if _is_stub(fn):
        print(f"✍️  {name}: not written yet, using the reference solution.")
        return expected
    try:
        mine = fn(*args, **kwargs)
    except NotImplementedError:
        print(f"✍️  {name}: not written yet, using the reference solution.")
        return expected
    except Exception as err:  # a learner's bug must not stop the notebook
        print(f"❌ {name} raised {type(err).__name__}: {err} · using the reference solution.")
        return expected
    if _same(mine, expected):
        print(f"✅ {name} matches the reference solution.")
    else:
        print(f"❌ {name} differs from the reference: {_diff(mine, expected)} · using the reference.")
    return expected


print(f"Colab: {IN_COLAB} · smoke mode: {SMOKE} · seed: {SEED} · imagehash: ", end="")
try:
    import imagehash  # noqa: F401

    print("ok")
except ImportError:
    print("missing (run this cell again)")

# %% [markdown]
# ### How the ✍️ cells work
#
# About four times in this notebook you write a small function. Each one comes with its signature and a
# `use(my_function, …, reference=…)` line that compares your result with the reference solution and
# then continues with the reference, so an unfinished or wrong function never stops the notebook.
# Run the cell below as is, then fill in the body and run it again to see the message change.

# %%
# @title Reference solution — peek only if you are stuck { display-mode: "form" }
def _ref_double(x):
    return 2 * x


# %%
# ✍️ YOUR CODE
def double(x):
    """Return twice x."""
    ...


result = use(double, 21, reference=_ref_double)
print("the notebook continues with:", result)

# %% [markdown]
# ### The Play Dataset
#
# The next cell fetches the annotation subset we prepared (COCO 2017 val, six animal classes, one class per
# photo) and downloads the photos from `images.cocodataset.org`.
#
# **The first run takes a minute or two** (about 115 MB); afterwards the images are cached and the cell is
# instant. The result is two tidy tables, `images` (one row per photo) and `annotations` (one row per
# animal: its box and mask).

# %%
# @title Data helpers (download with cache, build the tables) { display-mode: "form" }
import json
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

# Where the committed manifests live. In Colab they are fetched from the repository; locally they are read
# from ./data. ⬇️ DATA_BASE_URL: change this if the repository moves or you work on a fork.
DATA_BASE_URL = "https://raw.githubusercontent.com/FHNW-VISED/eda-teaching-notebook/main/data/"
DATA_DIR = Path("data")
IMAGE_CACHE = DATA_DIR / "cache" / "coco"
SMOKE_IMAGES_PER_CLASS = 6


def _github_headers() -> dict:
    """Authorization header from the Colab secret GITHUB_TOKEN, if the repository is private (SPEC §5)."""
    if not IN_COLAB:
        return {}
    try:
        from google.colab import userdata

        token = userdata.get("GITHUB_TOKEN")
        return {"Authorization": f"token {token}"} if token else {}
    except Exception:
        return {}


def download_file(url: str, dest: Path, headers: dict | None = None, attempts: int = 3) -> Path | None:
    """Download `url` to `dest` once; a half-finished download is written to `.part` first (N-019)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as fh:
                fh.write(resp.read())
            part.rename(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == attempts - 1:
                part.unlink(missing_ok=True)
                return None
    return None


def fetch_manifest(name: str) -> Path:
    """Return the local path of a committed manifest, downloading it from DATA_BASE_URL if needed."""
    path = DATA_DIR / name
    if not path.exists():
        if download_file(DATA_BASE_URL + name, path, headers=_github_headers()) is None:
            raise RuntimeError(
                f"Could not download {name} from {DATA_BASE_URL}. Check DATA_BASE_URL, or add a GITHUB_TOKEN "
                "secret in Colab if the repository is private."
            )
    return path


def download_images(rows: list[dict], workers: int = 16) -> dict[int, Path]:
    """Download every image in `rows` ('id', 'coco_url', 'file_name') into IMAGE_CACHE; one progress line."""
    done, paths = 0, {}

    def one(row):
        return row["id"], download_file(row["coco_url"], IMAGE_CACHE / row["file_name"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for image_id, path in pool.map(one, rows):
            done += 1
            if path is not None:
                paths[image_id] = path
            print(f"\r  images ready: {done}/{len(rows)}", end="", flush=True)
    print()
    return paths


def build_tables(coco: dict, paths: dict[int, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tidy tables from a COCO-format dict: `images` (one row per photo), `annotations` (one per animal)."""
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    lic_name = {lic["id"]: lic["name"] for lic in coco.get("licenses", [])}
    anns = pd.DataFrame(
        {
            "id": a["id"],
            "image_id": a["image_id"],
            "class": cat_name[a["category_id"]],
            "bbox": a["bbox"],
            "area": a["area"],
            "segmentation": a["segmentation"],
            "iscrowd": a["iscrowd"],
        }
        for a in coco["annotations"]
        if a["image_id"] in paths
    )
    per_image_class = anns.groupby("image_id")["class"].first()
    n_animals = anns.groupby("image_id").size()
    imgs = pd.DataFrame(
        {
            "id": im["id"],
            "file": im["file_name"],
            "path": str(paths[im["id"]]),
            "width": im["width"],
            "height": im["height"],
            "class": per_image_class.get(im["id"]),
            "n_animals": int(n_animals.get(im["id"], 0)),
            "license": lic_name.get(im["license"], str(im["license"])),
        }
        for im in coco["images"]
        if im["id"] in paths
    )
    imgs = imgs.sort_values("id").reset_index(drop=True)
    anns = anns.sort_values("id").reset_index(drop=True)
    return imgs, anns


# %%
# @title Load the manifest, download the images, build the tables { display-mode: "form" }
COCO_RAW = json.loads(fetch_manifest("coco_animals_subset.json").read_text())
_cat_name = {c["id"]: c["name"] for c in COCO_RAW["categories"]}
_image_class = {a["image_id"]: _cat_name[a["category_id"]] for a in COCO_RAW["annotations"]}

_rows = COCO_RAW["images"]
if SMOKE:  # same code path, fewer images (N-014)
    _rng = random.Random(SEED)
    _rows = []
    for c in CLASSES:
        _of_class = [im for im in COCO_RAW["images"] if _image_class[im["id"]] == c]
        _rows += sorted(_rng.sample(_of_class, SMOKE_IMAGES_PER_CLASS), key=lambda im: im["id"])

print(f"{len(_rows)} images to fetch from images.cocodataset.org (cached in {IMAGE_CACHE}/)")
_paths = download_images(_rows)
if len(_paths) < len(_rows):
    print(f"  ⚠️ {len(_rows) - len(_paths)} images could not be downloaded and are left out; run again.")

images, annotations = build_tables(COCO_RAW, _paths)
RESULTS["n_images"] = int(len(images))

_counts = Counter(images["class"])
print(f"images: {len(images)} · animals (annotations): {len(annotations)}")
print("per class: " + " · ".join(f"{c} {_counts[c]}" for c in CLASSES))
images.head()

# %% [markdown]
# Each row of `images` is one photo with its size, its class and how many animals it holds; `annotations`
# has one row per animal with its box and mask.
#
# Look at the `n_animals` column: several photos hold more than one animal, which already hints at a
# question we will meet in section 4.
