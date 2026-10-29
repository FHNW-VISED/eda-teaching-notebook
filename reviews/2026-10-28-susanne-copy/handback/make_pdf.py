"""Lay out the annotated before/after screenshots as an A4-landscape PDF for Susanne."""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

HERE = Path(__file__).parent
CHROME = ("/Users/simone.bonato/Library/Caches/ms-playwright/chromium_headless_shell-1234/"
          "chrome-headless-shell-mac-arm64/chrome-headless-shell")

PAGES = [
    # (title, text, before image or None, after image(s), after label)
    ("Answers now match the data", "The Play Dataset has 633 images after the five photos were excluded on 26 October. "
     "Four Sample Answers still quoted 637 or 638 images (and 244 crowded images, \"130 each\"). They now quote "
     "what the notebook prints: 632 RGB, 241 of 633, 130 and 129.",
     "b-numbers.png", ["a-numbers.png"], None),
    ("The dog slider starts where your text is", "The Class Imbalance task and its answer describe 20 dogs in 1,000 "
     "images (2 %, 98 % accuracy), but the slider started at 10 % and printed 90 %. It now starts at 2 %; students "
     "can still move it.",
     "b-dog.png", ["a-dog.png"], None),
    ("Image embeddings: fast on a classroom CPU", "Your DINOv2 comparison is unchanged, and the model is the same. "
     "Your copy ran on an A100. On Colab's free CPU, installing the packages and embedding every image takes minutes, "
     "so the park's DINOv2 vectors are now computed once and loaded (they give your within-class similarities to "
     "three decimals). A short note above the cell explains the COMPUTE_EMBEDDINGS checkbox: the model still runs on a GPU, or on a CPU when it is ticked.",
     "b-emb.png", ["a-emb.png"], None),
    ("The end of the class notebook", "Your marker and the old chapter 10 are gone from the class notebook: chapter 10 "
     "stopped Run all with an error (it used a cleaning step that your copy no longer has). Your checklist and "
     "resources stay, with one opening sentence; a short closing points to the new Own-Data notebook and lists the guiding questions.",
     ["b-ch10.png", "b-ch10err.png"], ["a-ch10.png"], None),
    ("New: the Own-Data notebook", "Built from the same cells as your notebook, so it never drifts from it. Students load "
     "their own dataset first; every analysis after that runs on their images. No Sample Answers: each task ends with a "
     "line for their answer and a one-line hint from the park.",
     None, ["o-title.png"], "Own-Data notebook: the opening"),
    ("One image in every format", "Students see the same park image labeled in each format the notebook "
     "reads, before they load anything, so they can recognize which one their labeling tool exported.",
     None, ["o-every.png"], "Own-Data notebook: Load Your Dataset"),
    ("How students learn the label formats", "One page explains what each task needs and how each format stores a "
     "box or a mask, using the same sheep box in every format so students can compare the numbers.",
     None, ["o-formats.png"], "Own-Data notebook: Load Your Dataset"),
    ("Loading a dataset, in any common format", "Students set three things: where the data is, which tasks their "
     "labels support (classification, + boxes, + masks), and the format (found automatically). Folders, CSV, COCO, "
     "YOLO, Pascal VOC and PNG masks are converted to the same tables your chapters use. The notebook then shows one "
     "image with its labels exactly as stored in their files.",
     None, ["o-load.png"], "Settings, the load summary and one labeled image"),
    ("Same task, two notebooks", "Left: your task in the class notebook, with its Sample Answer. Right: the same task in "
     "the Own-Data notebook, with an answer line and your answer turned into a one-line hint.",
     "t-task.png", ["o-task.png"], "Own-Data notebook"),
    ("Chapters the data cannot support skip themselves", "A classification-only dataset has no boxes, so every Object "
     "Detection cell says it was skipped instead of failing. Run all always reaches the end.",
     None, ["o-skip.png"], "Own-Data notebook, classification only"),
    ("Your multiple-choice question is in the table of contents", "It had its level dot only in a code-cell title, "
     "so Colab's table of contents did not list it. It now has a heading like your other tasks; every 🔴 heading "
     "carries \"(Optional)\".",
     "b-mcq.png", ["a-mcq.png"], None),
]

SUMMARY = """
<p>Thank you for the edited notebook. Your copy is now the notebook: nothing was merged back from the old version,
and everything you removed stays removed. This document shows the biggest changes made on top of it, each with your
version on the left and the new one on the right; the red numbers mark the exact place.</p>
<p><b>The Own-Data notebook</b> follows your request: the same analyses and hints, the checklists, no exercises
and no Sample Answers. Students can load a folder, a zip, a Google Drive link or a git repository.</p>
<p><b>Changed under the hood, not shown:</b> the hidden helper code was cleaned up (unused and repeated helpers
removed, imports in one place); every computed variable is lowercase (including <code>results</code>), only settings are in CAPITALS (PEP 8); browser buttons and checkbox labels are the same everywhere; "bit distance" is used throughout; the twelve image
browsers no longer interfere with each other (clicking an earlier browser after running later chapters crashed or
drew into the wrong cell); all Sample Answers use your gray box through one small helper; spelling is US throughout,
"IOU" is "IoU"; references to section numbers point to your chapter names. Every cell runs on a free Colab CPU
(tested headless, both notebooks).</p>
"""


def page(title, text, before, afters, after_label):
    def fig(src, label):
        return f'<figure><figcaption>{label}</figcaption><img src="img/{src}"></figure>'
    cols = []
    if before:
        befores = before if isinstance(before, list) else [before]
        cols.append('<div class="col">' + "".join(fig(b, "Your copy (28 Oct)") for b in befores) + "</div>")
    cols.append('<div class="col">' + "".join(fig(a, after_label or "New") for a in afters) + "</div>")
    return f'<section><h2>{title}</h2><p class="why">{text}</p><div class="row">{"".join(cols)}</div></section>'


HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
@page { size: A4 landscape; margin: 12mm; }
body { font-family: 'Helvetica Neue', Arial, sans-serif; color: #1f2937; }
section { page-break-after: always; }
h1 { font-size: 22px; margin: 0 0 6px; } h2 { font-size: 18px; margin: 0 0 4px; }
p { font-size: 12.5px; line-height: 1.45; margin: 4px 0 8px; max-width: 250mm; }
.why { color: #374151; }
.row { display: flex; gap: 8mm; align-items: flex-start; }
.col { flex: 1; min-width: 0; }
figure { margin: 0 0 4mm; } figcaption { font-size: 11px; font-weight: bold; color: #D7263D; margin-bottom: 2px; }
img { width: 100%; max-height: 150mm; object-fit: contain; object-position: top left;
      border: 1px solid #d1d5db; border-radius: 4px; }
</style></head><body>
<section><h1>Know Your Images: what changed in your copy</h1>
<p style="color:#6b7280">Simone Bonato · 29 October 2026</p>{summary}</section>
{pages}</body></html>"""


async def main(out: Path):
    html = HTML.replace("{summary}", SUMMARY).replace("{pages}", "".join(page(*p) for p in PAGES))
    src = HERE / "report.html"
    src.write_text(html)
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=CHROME)
        pg = await browser.new_page()
        await pg.goto("file://" + str(src))
        await pg.wait_for_timeout(500)
        await pg.pdf(path=str(out), format="A4", landscape=True, print_background=True)
        await browser.close()
    print("wrote", out)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
