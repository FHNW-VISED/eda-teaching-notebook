# %% [markdown]
# # Know Your Images
# ## Exploring and Preparing the Wildwood Farm Park Image Dataset
#
# Simone Bonato & Susanne Suter, FHNW
#
# Before training a deep learning model, you need to understand the data it will learn from. As Andrej Karpathy puts it, "become one with the data."
#
# In computer vision, this means looking at your images - not just their labels or statistics. Explore the dataset, inspect representative examples, and look for unusual patterns, class imbalances, labeling errors, and differences in image quality.
#
# Understanding your images helps you make informed decisions about preprocessing, data augmentation, model selection, and evaluation. It also helps you identify potential problems before they affect model performance.
#
# **Your goal:** Get to know your dataset well enough to explain what makes the classification task easy or difficult - and what your model will need to learn.
#
# *Inspired by Andrej Karpathy's [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/), Step 1: Become one with the data.*
#

# %% [markdown]
# **A note about this notebook**
#
# This notebook was developed with AI assistance and subsequently reviewed, edited, and extended by the authors. While we have made every effort to ensure its accuracy, not every detail could be thoroughly verified.
#
# If you encounter errors, unexpected behavior, or anything that seems incorrect or confusing, please let us know. Your feedback is especially valuable as this notebook is being used for the first time.

# %%

# @title Learning Objectives

from IPython.display import HTML

HTML(r"""
<div class="callout callout-banner"
     style="background:#E5E7EB;color:#374151;border-left:6px solid #6B7280;
            border-radius:6px;padding:12px 14px;line-height:1.5;
            font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;
            font-size:14px;width:100%;box-sizing:border-box;
            white-space:normal;overflow-wrap:break-word;margin:2px 0;">

<div style="font-size:22px;font-weight:bold;line-height:1.3;">
Learning Objectives: Know Your Images
</div>

<p style="margin:6px 0 0 0;">
<strong>By the end of this notebook, you will be able to:</strong>
</p>

<ol style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;">
Explore and analyze image datasets qualitatively and quantitatively,
and assess their suitability for a given task.
</li>
<li style="margin:2px 0;">
Identify and assess common dataset issues, including duplicates,
incorrect labels, and invalid annotations.
</li>
<li style="margin:2px 0;">
Select and justify appropriate data splits and evaluation metrics,
considering class imbalance, object size, and data leakage across
computer vision tasks.
</li>
</ol>

<p style="margin:8px 0 0 0;">
Activities are marked 🟢 core · 🔴 optional
</p>

</div>
""")


# %% [markdown]
# **What Do You Need to Know for the Exam and Your Project?**
#
# Focus on understanding the key concepts and applying them to new datasets. You should be able to explore and analyze your own image dataset, identify relevant characteristics and potential problems, interpret observations, and justify decisions about data preparation and evaluation.
#
# You do **not** need to memorize specific dataset statistics, thresholds, code implementations, or individual examples. The activities help you develop the competencies needed for both the exam and your own project.

# %%

# @title How to Use This Notebook
from IPython.display import HTML

HTML(r"""
<div class="callout" style="background:#F5F5F5;color:#374151;border-left:6px solid #9CA3AF;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">
<div style="font-weight:bold;">How to Use This Notebook</div>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;">This notebook is built around interactive activities. Explore the images, experiment with visualizations, and reflect on your observations.</li>
<li style="margin:2px 0;">The notebook runs in Google Colab. A CPU is sufficient for most activities, although computing image embeddings can take longer without a GPU.</li>
<li style="margin:2px 0;">A separate notebook guides you through exploring your own project dataset.</li>
</ul>
</div>
""")


# %% [markdown]
# ### Setup

# %% [markdown]
#
# > ▶️ **Run the next three cells, in order.**
# >
# > Continue reading while they execute. The last cell downloads the Play Dataset (~115 MB), which may take 1–2 minutes on the first run. Subsequent runs use the cache.
#

# %%
# @title Setup (run me first) { display-mode: "form" }
import ast
import inspect
import math  # noqa: F401  (used by later sections)
import os
import random
import sys
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt  # noqa: F401  (every later section plots with plt)
import numpy as np
from PIL import Image as _Image

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


# --- answer_box(): a worked answer computed at runtime, closed until opened (N-024, SPEC L-005) ---
ANSWER_BG, ANSWER_INK = "#F6EFE4", "#5C4027"  # the build's `answer` callout colours


def answer_box(title, lines=(), fig=None, table=None):
    """Draw a collapsed 💡 toggle holding printed lines, a DataFrame and/or matplotlib figure(s)."""
    import base64
    import html
    import io

    from IPython.display import HTML, display

    parts = []
    if lines:
        text = html.escape("\n".join(lines))
        parts.append(f'<pre style="white-space:pre-wrap;margin:6px 0 0 0;font-size:0.92em;">{text}</pre>')
    if table is not None:
        parts.append(table.to_html(border=0))
    for f in fig if isinstance(fig, (list, tuple)) else [fig]:
        if f is None:
            continue
        buf = io.BytesIO()
        f.savefig(buf, format="png", bbox_inches="tight")
        plt.close(f)
        png = base64.b64encode(buf.getvalue()).decode()
        parts.append(f'<img src="data:image/png;base64,{png}" style="max-width:100%;margin-top:6px;">')
    box = (f"background:{ANSWER_BG};color:{ANSWER_INK};border-left:6px solid {ANSWER_INK};border-radius:6px;"
           "padding:12px 14px;line-height:1.5;font-size:14px;width:100%;box-sizing:border-box;margin:2px 0;")
    display(HTML(f'<div style="{box}"><details style="margin:0;"><summary style="cursor:pointer;'
                 f'font-weight:bold;">💡 {html.escape(title)}</summary>{"".join(parts)}</details></div>'))


print(f"Colab: {IN_COLAB} · smoke mode: {SMOKE} · seed: {SEED} · imagehash: ", end="")
try:
    import imagehash  # noqa: F401

    print("ok")
except ImportError:
    print("missing (run this cell again)")

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
images.drop(columns="license").head()  # the licence column stays in the table, not on screen (N-045)


# %% [markdown]
# ### Optional Coding Exercises
#
# Some parts of this notebook let you choose how to work:
#
# - **Explore:** Run the provided code, experiment with it, and interpret the results.
# - **Code it yourself:** Implement the function in the ✍️ cell and compare your result with the reference solution.
#
# Both approaches let you continue through the notebook. If you choose to implement a function, run the reference cell first, then complete the ✍️ cell. An incomplete or incorrect implementation will not prevent you from continuing.
#

# %%
# @title Reference solution — run this cell; open it only if you are stuck { display-mode: "form" }
def _ref_double(x):
    return 2 * x


# %%
# ✍️ YOUR CODE
def double(x):
    """Return twice x.

    e.g. double(21) → 42
    """
    ...


result = use(double, 21, reference=_ref_double)
print("the notebook continues with:", result)
