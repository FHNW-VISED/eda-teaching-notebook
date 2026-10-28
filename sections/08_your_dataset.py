# %% [markdown]
# ## 10. Your project dataset

# %%
# @title Checklist Dataset Mini-Project
from IPython.display import HTML; HTML(r"""<div class="callout callout-theory" style="background:#E3F2FD;color:#0D47A1;border-left:6px solid #0D47A1;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">
<div style="font-weight:bold;">📘 Theory · What makes a good dataset</div>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><strong>Does it fit the question?</strong> The images, the classes and the setting must match the job you want the
  model to do.<br><em>Park example:</em> the alert needs a dog seen from a fixed camera in the sheep paddock, not a
  dog in a holiday snapshot taken from two metres away.</li>
<li style="margin:2px 0;"><strong>Enough samples</strong>, <strong>2–10 classes</strong> and <strong>about 500×500 px</strong>: the numbers to check, explained below.</li>
<li style="margin:2px 0;"><strong>Classes, distributions, imbalance, bias.</strong> Know them, and take them into account in training (for
  example with a weighted loss).<br>
  <em>Park example:</em> if 70 % of the photos are sheep and 3 % are dogs, the model sees few dogs to learn from,
  and the dog is the animal the alert exists for (section 4).</li>
<li style="margin:2px 0;"><strong>Variability matches the target population</strong>, and <strong>labels are present and validated</strong>: every image has
  a label, and someone checked a sample of them.<br>
  <em>Park example:</em> a photo of a zebra labelled "horse" teaches the model the wrong thing, and nobody notices
  until the zebra headcount is off.</li>
<li style="margin:2px 0;"><strong>Objects are identifiable and separable by eye.</strong></li>
<li style="margin:2px 0;"><strong>What do experts and the literature say</strong> about this data and this task?</li>
<li style="margin:2px 0;"><strong>Get a qualitative and a quantitative sense of the data</strong> (Karpathy, 2019): look at many images, then
  measure.</li>
</ul>
<details style="margin:6px 0 0 0;">
<summary style="cursor:pointer;font-weight:bold;">The reasons and park examples behind each point</summary>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><strong>Enough samples:</strong> a few hundred to a few thousand images, depending on how many classes you have.
  <em>Park example:</em> 100 photos per animal (600 in all) is a fair start; 12 zebra photos is not.</li>
<li style="margin:2px 0;"><strong>2–10 classes:</strong> more classes need more photos each, and similar classes get confused.
  <em>Park example:</em> six animals is fine; splitting sheep and horses into 40 breeds would not be.</li>
<li style="margin:2px 0;"><strong>About 500×500 px:</strong> check the input size of the models you plan to use.
  <em>Park example:</em> a bird across the pond may cover only 20×20 px of a 640×480 photo; shrink the photo to
  224×224 and it is a handful of pixels.</li>
<li style="margin:2px 0;"><strong>Variability:</strong> lighting, viewpoints, backgrounds and seasons should look like the ones
  the model will meet.
  <em>Park example:</em> take pictures of the park in winter and in summer, in sun and at dusk. A model that only
  saw sunny July afternoons will meet snow and low light in January.</li>
<li style="margin:2px 0;"><strong>Separable by eye:</strong> if you cannot tell them apart, the model will struggle too.
  <em>Park example:</em> a bird that is a dark speck over the pond, or sheep standing so close that their outlines
  merge into one woolly shape.</li>
<li style="margin:2px 0;"><strong>Experts and literature:</strong> before trusting COCO's masks, read the audit that found some of them poor; it
  tells you what to look for.</li>
<li style="margin:2px 0;"><strong>Qualitative and quantitative:</strong> look at many images and see how big the animals are and how much of the
  photo they cover. Is it the same for each class? Can you already see imbalances just by looking?</li>
</ul>
</details>
</div>""")

# %%
# @title 🔗 Resources: Image Datasets and Annotation Tools { display-mode: "form" }

from IPython.display import HTML, display

display(HTML(r"""
<div style="background:#F3F4F6;color:#4B5563;
            border-left:6px solid #4B5563;border-radius:6px;
            padding:12px 14px;line-height:1.5;
            font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;
            font-size:14px;width:100%;box-sizing:border-box;
            white-space:normal;overflow-wrap:break-word;margin:2px 0;">

<div style="font-weight:bold;">🔗 Resources: Image Datasets and Annotation Tools</div>

<p style="margin:6px 0 0 0;">
Where to find existing image datasets and tools for annotating your own images.
</p>

<p style="margin:10px 0 0 0;"><strong>Find a dataset</strong></p>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><a href="https://www.kaggle.com/datasets" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">Kaggle datasets</a>: a huge community collection, good for a first look.</li>
<li style="margin:2px 0;"><a href="https://datasetsearch.research.google.com" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">Google Dataset Search</a>: one search box over many repositories.</li>
<li style="margin:2px 0;"><a href="https://www.v7labs.com/open-datasets" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">V7 open datasets</a>: a curated list of open computer-vision data
  (plus <a href="https://www.v7labs.com/blog" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">their blog</a> with overviews of well-known datasets).</li>
<li style="margin:2px 0;"><a href="https://universe.roboflow.com" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">Roboflow Universe</a>: community datasets, many already labeled with boxes or masks.</li>
<li style="margin:2px 0;"><a href="https://lila.science" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">LILA</a>: wildlife and camera-trap images, the closest match to the park's cameras.</li>
<li style="margin:2px 0;"><a href="https://pytorch.org/vision/stable/datasets.html" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">torchvision datasets</a>: classic datasets available directly through PyTorch.</li>
<li style="margin:2px 0;"><a href="https://cocodataset.org" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">COCO</a>: a general-purpose dataset; our play dataset is a subset of it.</li>
</ul>

<p style="margin:10px 0 0 0;"><strong>Label your own images</strong></p>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><a href="https://supervisely.com" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">Supervisely</a>: a labeling platform for boxes, polygons, and masks.</li>
<li style="margin:2px 0;"><a href="https://www.v7labs.com" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">V7</a>: browser-based image annotation tools.</li>
<li style="margin:2px 0;"><a href="https://roboflow.com/annotate" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">Roboflow Annotate</a>: browser-based labeling with export to common formats.</li>
<li style="margin:2px 0;"><a href="https://labelstud.io" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">Label Studio</a>: open-source annotation software that can run locally.</li>
</ul>

</div>
"""))

# %%
# @title
from IPython.display import HTML; HTML(r"""<div class="callout callout-objective" style="background:#B9E3EA;color:#08505C;border-left:6px solid #08505C;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">
<div style="font-weight:bold;">🎯 Objective</div>
<p style="margin:6px 0 0 0;">Run the checklist of sections 3 to 6 on your own folder-per-class dataset, and answer the guiding questions
for it the way we answered them for the park.</p>
</div>""")

# %% [markdown]
# Your project has its own dataset, and it deserves the same twenty minutes of looking before anybody trains
# on it. This section runs the checklist on **any folder-per-class dataset**: one sub-folder per class, the
# images inside. The sub-folder names become the class names.
#
# ```
# my_dataset/                ← MY_DATASET_DIR points here (or at my_dataset.zip)
# ├── daisy/      img_001.jpg, img_002.png, …
# ├── rose/       …
# └── tulip/      …
# ```

# %%
# @title
from IPython.display import HTML; HTML(r"""<div class="callout callout-your-turn" style="background:#FFF4D6;color:#7A4A00;border-left:6px solid #7A4A00;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">
<div style="font-weight:bold;">✏️ Your turn · Point the notebook at your dataset (you change one line)</div>
<p style="margin:6px 0 0 0;">Colab cannot see your laptop's folders, so the dataset first goes where Colab can read it. Pick <strong>one</strong> of
the three options, then do the last step.</p>
<p style="margin:6px 0 0 0;"><strong>Option A · Upload it to this Colab session</strong> (fastest; gone when the session ends)</p>
<ol style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><strong>Zip the dataset folder</strong> on your laptop: Mac, right-click the folder → <em>Compress</em>; Windows,
   right-click → <em>Send to</em> → <em>Compressed (zipped) folder</em>. Colab uploads files, not folders.</li>
<li style="margin:2px 0;">Click the 📁 <strong>Files</strong> icon in Colab's left bar and <strong>drag the zip into it</strong> (or use the upload icon at
   the top of the pane). Wait until the progress circle at the bottom left disappears.</li>
<li style="margin:2px 0;"><strong>Right-click the zip → Copy path.</strong> It looks like <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">/content/my_dataset.zip</code>; the notebook unzips it.</li>
</ol>
<p style="margin:6px 0 0 0;">⚠️ Colab deletes uploaded files when the session ends (closing the tab for a while, or <em>Disconnect and
delete runtime</em>): next time, upload the zip again.</p>
<p style="margin:6px 0 0 0;"><strong>Option B · Use Google Drive</strong> (stays between sessions; slower the first time)</p>
<ol style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><strong>Put the dataset on your Drive</strong> at <a href="https://drive.google.com" target="_blank" rel="noopener" style="color:inherit;text-decoration:underline;">drive.google.com</a>: <em>New → Folder upload</em>
   for the folder, or <em>File upload</em> for a zip.</li>
<li style="margin:2px 0;"><strong>Connect Drive to Colab:</strong> in the 📁 Files pane, click the <strong>Mount Drive</strong> icon (a folder with the
   Drive logo) and allow access in the window that opens. A <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">drive/</code> folder appears in the pane.</li>
<li style="margin:2px 0;">Open <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">drive/MyDrive/</code>, find the folder whose sub-folders are your classes (or the zip), <strong>right-click →
   Copy path.</strong> It looks like <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">/content/drive/MyDrive/project/my_dataset</code>.</li>
</ol>
<p style="margin:6px 0 0 0;"><strong>Option C · Paste a Google Drive link to a zip</strong> (no mounting, no browsing)</p>
<ol style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;">In Drive, right-click the <strong>zip</strong> → <em>Share</em> → <em>Copy link</em>. It looks like
   <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">https://drive.google.com/file/d/1I9D…/view?usp=drive_link</code>.</li>
<li style="margin:2px 0;">Paste the whole link as the path. If the zip is shared as <em>Anyone with the link</em>, it downloads straight
   away; if it is private, Colab asks you to sign in with the Google account that can open it.</li>
</ol>
<p style="margin:6px 0 0 0;">Links to a Drive <em>folder</em> do not work: zip the folder first, or use option B.</p>
<p style="margin:6px 0 0 0;"><strong>Last step, all options:</strong> paste the path between the quotes in the next cell, click into that cell, and
run everything from there: <strong>Runtime → Run cell and below</strong>. If the path is not right, the cell after it
stops with a plain message that says what to change.</p>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;">✅ <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">/content/my_dataset.zip</code> (A), <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">/content/drive/MyDrive/project/my_dataset</code> (B) or
  <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">https://drive.google.com/file/d/…/view</code> (C): the folder that holds <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">daisy/</code>, <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">rose/</code>, <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">tulip/</code>, or a zip
  of it.</li>
<li style="margin:2px 0;">❌ <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">…/my_dataset/daisy</code>: one class folder, one level too deep.</li>
<li style="margin:2px 0;">❌ a folder that holds <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">train/</code> and <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">val/</code>: point at <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">…/train</code> instead (then at <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">…/val</code>, if you like).</li>
</ul>
<p style="margin:6px 0 0 0;">Leave it at <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">None</code> to see the whole section on a small park sample first.</p>
</div>""")

# %%
# ✏️ YOUR TURN
MY_DATASET_DIR = None  # paste your path between quotes, e.g. "/content/my_dataset.zip"

# %%
# @title Dataset reader: finds your class folders, or says exactly what to change { display-mode: "form" }
import re
import urllib.error
import urllib.request
import zipfile

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = {"train", "training", "val", "valid", "validation", "test", "testing", "dev"}
MY_MAX_IMAGES = 2000  # larger datasets are sampled down to this many photos, stratified by class


def _visible(folder: Path) -> list[Path]:
    """The entries of `folder` that are not hidden or system files (.DS_Store, __MACOSX, …)."""
    return sorted(f for f in folder.iterdir() if not f.name.startswith((".", "_")))


def _is_image(f: Path) -> bool:
    return f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES


def _contents(folder: Path, limit: int = 8) -> str:
    """One line per entry: 'name/ (12 images)' for folders, the name for files."""
    lines = []
    for f in _visible(folder)[:limit]:
        n = sum(_is_image(g) for g in f.rglob("*")) if f.is_dir() else None
        lines.append(f"    {f.name}/  ({n} images)" if f.is_dir() else f"    {f.name}")
    more = len(_visible(folder)) - limit
    return "\n".join(lines + ([f"    … and {more} more"] if more > 0 else [])) or "    (empty)"


def drive_file_id(url: str) -> str:
    """The file id in a Google Drive share link; stops with advice for a folder link or another site.

    e.g. drive_file_id("https://drive.google.com/file/d/1I9De…/view?usp=drive_link") → "1I9De…"
    """
    if "drive.google.com/drive/folders" in url:
        raise SystemExit("❌ MY_DATASET_DIR: this is a link to a Drive *folder*. Zip the folder, upload the "
                         "zip to Drive and share the zip's link, or use option B (Mount Drive, Copy path).")
    found = re.search(r"/file/d/([\w-]+)", url) or re.search(r"[?&]id=([\w-]+)", url)
    if "drive.google.com" not in url or not found:
        raise SystemExit("❌ MY_DATASET_DIR: links work only for a Google Drive file, like "
                         "https://drive.google.com/file/d/<id>/view. For anything else, download it and use "
                         "option A.")
    return found[1]


def _drive_public_download(file_id: str, dest: Path) -> bool:
    """Download a file shared as 'Anyone with the link'. False if Drive answers with a page, not the file."""
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            first = r.read(4)
            if first[:2] != b"PK":  # not a zip: the sign-in page (private file) or a 'not found' page
                return False
            part = dest.with_suffix(".part")  # renamed only when complete, so a cut download is not reused
            with open(part, "wb") as f:
                f.write(first)
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    if f.tell() % (25 << 20) < (1 << 20):
                        print(f"\r  downloading {dest.name}: {f.tell() / 1e6:.0f} MB", end="", flush=True)
            part.rename(dest)
        print(f"\r  downloaded {dest.name}: {dest.stat().st_size / 1e6:.0f} MB")
        return True
    except urllib.error.HTTPError:  # 404: a wrong id, or a file that is gone
        return False
    except OSError as err:  # the connection dropped: say so, and start over next time
        dest.with_suffix(".part").unlink(missing_ok=True)
        raise SystemExit(f"❌ the download from Drive stopped halfway ({err}). Run the cell again: it starts "
                         "over.") from None


def _drive_signed_in_download(file_id: str, folder: Path) -> Path:
    """Colab only: sign in with Google and download a private file through the Drive API."""
    from google.colab import auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    print("ℹ️ the file is not shared publicly: sign in with the Google account that can open it.")
    auth.authenticate_user()
    drive = build("drive", "v3")
    meta = drive.files().get(fileId=file_id, fields="name,mimeType", supportsAllDrives=True).execute()
    if meta["mimeType"] == "application/vnd.google-apps.folder":
        drive_file_id("drive.google.com/drive/folders/")  # stops with the folder advice
    dest = folder / meta["name"]
    part = dest.with_suffix(".part")  # renamed only when complete, so a cut download is not reused
    with open(part, "wb") as f:
        download = MediaIoBaseDownload(f, drive.files().get_media(fileId=file_id, supportsAllDrives=True))
        done = False
        while not done:
            status, done = download.next_chunk()
            print(f"\r  downloading {dest.name}: {status.progress():.0%}", end="", flush=True)
    part.rename(dest)
    print()
    return dest


def fetch_drive_link(url: str) -> Path:
    """Download the zip behind a Google Drive share link (once per session) and return its local path."""
    file_id = drive_file_id(url)
    folder = DATA_DIR / "cache" / "my_upload" / f"drive_{file_id}"
    done = sorted(folder.glob("*.zip")) if folder.exists() else []
    if done:
        return done[0].resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if _drive_public_download(file_id, folder / "dataset.zip"):
        return (folder / "dataset.zip").resolve()
    if IN_COLAB:
        return _drive_signed_in_download(file_id, folder).resolve()
    raise SystemExit("❌ MY_DATASET_DIR: Drive did not hand out this file without a sign-in, so it is "
                     "private (or the link is wrong). In Colab the notebook asks you to sign in; elsewhere, "
                     "open the file in Drive → Share → General access → 'Anyone with the link'.")


def find_dataset(path) -> Path:
    """The folder that holds the class folders. Stops with a plain message saying what to change if the
    path cannot be used; unzips a .zip and steps into a lone wrapper folder on its own.

    e.g. find_dataset("/content/flowers.zip") → Path("data/cache/my_upload/flowers/flowers")
    """
    text = str(path).strip().strip("'\"")
    p = fetch_drive_link(text) if text.startswith(("http://", "https://")) else Path(text).expanduser()
    if str(p).startswith("/content/drive") and IN_COLAB and not Path("/content/drive/MyDrive").exists():
        from google.colab import drive  # Step 1, Drive route: connect it on first use

        drive.mount("/content/drive")
    if p.is_file() and p.suffix.lower() == ".zip":
        dest = (DATA_DIR / "cache" / "my_upload" / p.stem).resolve()  # absolute, so messages are pastable
        if not dest.exists():
            with zipfile.ZipFile(p) as z:
                z.extractall(dest)
        print(f"ℹ️ unzipped {p.name} into {dest}")
        p = dest
    if not p.exists():
        parent = next(q for q in p.parents if q.exists())
        raise SystemExit(f"❌ MY_DATASET_DIR: {p} does not exist (a typo, or the upload is not there).\n"
                         f"   The nearest folder that exists is {parent}, which holds:\n{_contents(parent)}\n"
                         "   Copy the path from the Files pane: right-click the folder → Copy path.")
    if p.is_file():
        raise SystemExit(f"❌ MY_DATASET_DIR: {p} is a file. Point at the folder that holds your class "
                         f"folders, or at a .zip. It sits in {p.parent}, which holds:\n{_contents(p.parent)}")
    while True:  # a zip often wraps everything in one more folder: step into it
        subdirs = [d for d in _visible(p) if d.is_dir()]
        loose = [f for f in _visible(p) if _is_image(f)]
        if len(subdirs) == 1 and not loose and any(d.is_dir() for d in _visible(subdirs[0])):
            print(f"ℹ️ {p.name}/ holds only the folder {subdirs[0].name}/: using that.")
            p = subdirs[0]
            continue
        break
    names = {d.name.lower() for d in subdirs}
    if subdirs and names <= SPLIT_NAMES:
        first = p / sorted(d.name for d in subdirs)[0]
        raise SystemExit(f"❌ MY_DATASET_DIR: {p} holds split folders ({', '.join(sorted(names))}), "
                         f"not class folders. Point at one of them, e.g. {first}, and run the section "
                         "again for the other.")
    if not subdirs and loose:
        raise SystemExit(f"❌ MY_DATASET_DIR: {p} holds {len(loose)} images but no class folders. If this "
                         f"is one class folder, point one level up, at {p.parent}. If your project has no "
                         "classes (detection, segmentation), move the images into one folder inside, e.g. "
                         f"{p / 'all'}.")
    if not any(_is_image(f) for f in p.rglob("*")):
        raise SystemExit(f"❌ MY_DATASET_DIR: {p} holds no image files "
                         f"({', '.join(sorted(IMAGE_SUFFIXES))}). "
                         f"It holds:\n{_contents(p)}")
    return p


def list_folder_per_class(root) -> pd.DataFrame:
    """One row per image file under `root`: columns path, file, class (the top-level folder it sits in).

    e.g. a row: {"path": "/content/flowers/rose/img_7.jpg", "file": "rose/img_7.jpg", "class": "rose"}
    """
    rows = []
    for class_dir in (d for d in _visible(Path(root)) if d.is_dir()):
        for f in sorted(class_dir.rglob("*")):
            if _is_image(f) and not f.name.startswith("."):
                rows.append({"path": str(f), "file": str(f.relative_to(root)), "class": class_dir.name})
    return pd.DataFrame(rows, columns=["path", "file", "class"])


def folder_notes(root: Path, files: pd.DataFrame) -> list[str]:
    """Warnings about what the reader skipped or what makes the checks weak (one line each)."""
    root = Path(root)
    notes = []
    loose = [f.name for f in _visible(root) if _is_image(f)]
    if loose:
        notes.append(f"⚠️ {len(loose)} images sit directly in {root.name}/, outside any class folder: "
                     "skipped.")
    others = [f for d in _visible(root) if d.is_dir() for f in d.rglob("*")
              if f.is_file() and not _is_image(f) and not f.name.startswith(".")]
    skipped = Counter(f.suffix.lower() or "(no extension)" for f in others)
    if skipped:
        notes.append("⚠️ skipped files that are not images: "
                     + ", ".join(f"{n} × {ext}" for ext, n in skipped.most_common()))
    empty = [d.name for d in _visible(root) if d.is_dir() and d.name not in set(files["class"])]
    if empty:
        notes.append(f"⚠️ class folders with no images: {', '.join(empty)}.")
    if files["class"].nunique() == 1:
        notes.append("⚠️ only one class folder: check 1 has nothing to compare (fine for a detection "
                     "project).")
    return notes


# %% [markdown]
# The next cell reads your folder (or, with `None`, builds the park sample: 20 cleaned photos per class plus
# **three planted problems**, an exact copy of a cat, a mirrored dog and a grayscale bird). It prints what it
# found and shows three images per class: **check that these are your images and your class names** before
# reading anything else.

# %%
# @title Read your folder (or build the sample), and show what was found { display-mode: "form" }
MY_SAMPLE_DIR = DATA_DIR / "cache" / "my_dataset"
MY_SAMPLE_PER_CLASS = 4 if SMOKE else 20  # photos per class in the sample (N-014: fewer in smoke mode)


def build_my_sample(dest: Path, per_class: int, seed: int) -> Path:
    """Folder-per-class copy of a few clean Play Dataset photos, plus an exact copy, a mirror, a grayscale."""
    shutil.rmtree(dest, ignore_errors=True)  # rebuilt from scratch: re-running gives the same sample (N-013)
    rng = random.Random(seed)
    pool = clean_images[clean_images["flags"] == ""]
    picked = {}
    for c in CLASSES:
        of_class = sorted(pool.loc[pool["class"] == c, "path"])
        picked[c] = rng.sample(of_class, min(per_class, len(of_class)))
        (dest / c).mkdir(parents=True, exist_ok=True)
        for p in picked[c]:
            shutil.copyfile(p, dest / c / Path(p).name)
    first = {c: Path(picked[c][0]) for c in CLASSES if picked[c]}
    shutil.copyfile(first["cat"], dest / "cat" / f"{first['cat'].stem}_copy.jpg")
    with Image.open(first["dog"]) as im:
        ImageOps.mirror(im.convert("RGB")).save(dest / "dog" / f"{first['dog'].stem}_b.jpg", quality=92)
    with Image.open(first["bird"]) as im:
        im.convert("L").save(dest / "bird" / first["bird"].name, quality=92)
    return dest


if MY_DATASET_DIR is not None and not SMOKE:
    my_source, my_root = "yours", find_dataset(MY_DATASET_DIR)
else:
    my_source, my_root = "sample", build_my_sample(MY_SAMPLE_DIR, MY_SAMPLE_PER_CLASS, SEED)
    print("MY_DATASET_DIR is None: running on the park sample. Paste your path above to run on yours.")
my_files = list_folder_per_class(my_root)
for _note in folder_notes(my_root, my_files):
    print(_note)

if len(my_files) > MY_MAX_IMAGES:
    _n_all = len(my_files)
    my_files = pd.concat([g.sample(max(1, round(len(g) * MY_MAX_IMAGES / _n_all)), random_state=SEED)
                          for _, g in my_files.groupby("class")])
    print(f"⚠️ {_n_all} images found: the checks run on a stratified sample of {len(my_files)}.")
my_files = my_files.sort_values(["class", "file"]).reset_index(drop=True)
MY_CLASSES = sorted(my_files["class"].unique())
# Park classes keep their colour; any other class name borrows one from the same palette (L-003).
_palette = list(CLASS_COLORS.values())
MY_COLORS = {c: CLASS_COLORS.get(c, _palette[i % len(_palette)]) for i, c in enumerate(MY_CLASSES)}
print(f"✅ source: {my_source} · {my_root} · {len(my_files)} images in {len(MY_CLASSES)} class folders: "
      + ", ".join(f"{c} ({n})" for c, n in my_files["class"].value_counts().reindex(MY_CLASSES).items()))

_shown = MY_CLASSES[:8]  # a first look: three images per class (at most eight classes)
fig, axes = plt.subplots(len(_shown), 3, figsize=(7, 2.2 * len(_shown)), squeeze=False)
for r, c in enumerate(_shown):
    _of_class = my_files[my_files["class"] == c].head(3)
    for k in range(3):
        ax = axes[r, k]
        ax.axis("off")
        if k < len(_of_class):
            try:
                ax.imshow(load_thumb(_of_class.iloc[k]["path"], 192))
            except Exception:  # an unreadable file: check 2 lists it
                ax.text(0.5, 0.5, "cannot be opened", ha="center", va="center", transform=ax.transAxes)
            _name = Path(_of_class.iloc[k]["file"]).name
            ax.set_title(f"{c} · {_name if len(_name) <= 24 else _name[:20] + '…'}", fontsize=8)
fig.suptitle("A first look: are these your images and your classes?", color=INK)
fig.tight_layout()
plt.show()

# %%
# @title
from IPython.display import HTML; HTML(r"""<div class="callout callout-go-deeper" style="background:#F3F4F6;color:#4B5563;border-left:6px solid #4B5563;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">
<div style="font-weight:bold;">🔍 Go deeper · Detection or segmentation project, and big datasets</div>
<p style="margin:6px 0 0 0;">Put all images in one folder inside <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">MY_DATASET_DIR</code> (e.g. <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">…/project/all/</code>). The class counts then say
nothing, but sizes, modes, copies and the split still do (the box and mask checks are sections 7 and 8).
Images deeper inside a class folder count, other files are listed and skipped. Datasets above
<code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">MY_MAX_IMAGES</code> photos are sampled down, keeping the class shares, so the checks stay at a minute or two.</p>
</div>""")

# %% [markdown]
# Four checks run next and end in one report card. Check 1 counts the images per folder, as section 4 did for
# the park: the **ratio** is the
# largest class divided by the smallest (in the park about 130 cats against 50 sheep, 2 : 1 to 3 : 1).

# %%
# @title Check 1: images per class, and the largest : smallest ratio { display-mode: "form" }
my_counts = my_files["class"].value_counts().reindex(MY_CLASSES)
fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(MY_CLASSES) + 2), 3.2))
ax.bar(MY_CLASSES, my_counts.to_numpy(), color=[MY_COLORS[c] for c in MY_CLASSES])
ax.set_ylabel("images")
ax.set_title(f"Images per class folder ({len(my_files)} in total)")
ax.tick_params(axis="x", rotation=30 if len(MY_CLASSES) > 6 else 0)
plt.tight_layout()
plt.show()
my_imbalance = float(my_counts.max() / my_counts.min())
print(f"largest {my_counts.idxmax()} ({my_counts.max()}) · smallest {my_counts.idxmin()} ({my_counts.min()}) "
      f"· ratio {my_imbalance:.1f} : 1")

# %% [markdown]
# 💡 Look at the ratio: up to 3 : 1 is common; from 10 : 1 accuracy hides the small class (section 4).

# %% [markdown]
# Check 2 opens every file with `is_readable`, then reads its size (width × height) and its colour mode (`RGB`
# stores three colour channels, `L` one gray channel).

# %%
# @title Check 2: readable? size and colour mode of every file { display-mode: "form" }
# (section 3's measure_image on a small thumbnail)
_my_rows = []
for _, r in my_files.iterrows():
    ok = is_readable(r["path"])
    stats = measure_image(r["path"], thumb=64)[0] if ok else {}
    _my_rows.append({"file": r["file"], "class": r["class"], "readable": ok, "width": stats.get("width"),
                  "height": stats.get("height"), "mode": stats.get("mode", "unreadable")})
my_stats = pd.DataFrame(_my_rows)
_ok = my_stats[my_stats["readable"]]

fig, ax = plt.subplots(figsize=(5.5, 4))
for c in MY_CLASSES:
    sub = _ok[_ok["class"] == c]
    ax.scatter(sub["width"], sub["height"], s=18, alpha=0.7, color=MY_COLORS[c], label=c)
ax.set_xlabel("width (px)")
ax.set_ylabel("height (px)")
ax.set_title("Image sizes per class")
ax.legend(frameon=False, fontsize=8, ncol=2 if len(MY_CLASSES) > 6 else 1)
plt.tight_layout()
plt.show()
print(f"unreadable files: {int((~my_stats['readable']).sum())}")
print("colour modes:", " · ".join(f"{m} {n}" for m, n in my_stats["mode"].value_counts().items()))
print(f"distinct sizes: {_ok.groupby(['width', 'height']).ngroups} · "
      f"long side from {int(_ok[['width', 'height']].max(axis=1).min())} "
      f"to {int(_ok[['width', 'height']].max(axis=1).max())} px")
display(my_stats[my_stats["mode"] != "RGB"].head(10))

# %% [markdown]
# 💡 Look for a second cluster of sizes or any mode that is not `RGB`: each is a second source to decide on.

# %% [markdown]
# Check 3 is section 5's copy hunt: exact copies by file hash, and near, flipped and rotated copies by the
# **distance** between perceptual hashes (differing bits out of 64). Pairs at or below your `HASH_THRESHOLD`
# from section 5 count as copies.

# %%
# @title Check 3: exact, recompressed, flipped and rotated copies { display-mode: "form" }
# (all pairs, in blocks to keep memory small)
_readable = my_stats.index[my_stats["readable"]].to_numpy()
_my_paths = my_files.loc[_readable, "path"].tolist()
_md5 = [file_md5(p) for p in _my_paths]
_variant_bits = {v: [] for v in VARIANT_NAMES}
for p in _my_paths:
    for v, im in variants(load_thumb(p, THUMB_FOR_HASH)).items():
        _variant_bits[v].append(phash(im))
_variant_bits = {v: hash_bits(h) for v, h in _variant_bits.items()}
_plain = _variant_bits["original"]
_n = len(_my_paths)
_best = np.full((_n, _n), 65, dtype=np.int16)  # smallest distance over the variants, and which variant
_best_v = np.zeros((_n, _n), dtype=np.int8)
for k, v in enumerate(VARIANT_NAMES):
    for start in range(0, _n, 256):
        d = hash_distances(_variant_bits[v][start:start + 256], _plain)
        better = d < _best[start:start + 256]
        _best[start:start + 256][better] = d[better]
        _best_v[start:start + 256][better] = k

_pairs = []
for i, j in zip(*np.nonzero(np.triu(_best <= HASH_THRESHOLD, k=1)), strict=True):
    a, b = _readable[i], _readable[j]
    kind = "exact" if _md5[i] == _md5[j] else KIND_OF_VARIANT[VARIANT_NAMES[_best_v[i, j]]]
    _pairs.append({"a": my_files.at[a, "file"], "b": my_files.at[b, "file"], "kind": kind,
                   "distance": int(_best[i, j]),
                   "same class": my_files.at[a, "class"] == my_files.at[b, "class"],
                   "ia": a, "ib": b})
my_copies = pd.DataFrame(_pairs, columns=["a", "b", "kind", "distance", "same class", "ia", "ib"])
print(f"{len(my_copies)} copy pairs at HASH_THRESHOLD = {HASH_THRESHOLD}:",
      " · ".join(f"{k} {n}" for k, n in my_copies["kind"].value_counts().items()) or "none")
print(f"pairs whose two files sit in different class folders (one label is wrong): "
      f"{int((~my_copies['same class']).sum())}")
display(my_copies.drop(columns=["ia", "ib"]).head(10))

_show = my_copies.head(4)
if len(_show):
    fig, axes = plt.subplots(len(_show), 2, figsize=(5, 2.4 * len(_show)), squeeze=False)
    for row, (_, pr) in zip(axes, _show.iterrows(), strict=True):
        for ax, idx in zip(row, [pr["ia"], pr["ib"]], strict=True):
            ax.imshow(load_thumb(my_files.at[idx, "path"], 256))
            ax.set_title(f"{my_files.at[idx, 'file']}\n{pr['kind']}", fontsize=8)
            frame(ax, MY_COLORS[my_files.at[idx, "class"]])
    plt.tight_layout()
    plt.show()

# %% [markdown]
# 💡 Look at the pairs across class folders first: one of the two labels is wrong.

# %% [markdown]
# Check 4 suggests a split: it groups the copies, then makes a stratified (every class in both halves, in the
# same shares), group-aware split.

# %%
# @title Check 4: a split suggestion, copies grouped, stratified by class { display-mode: "form" }
# Copies form groups that stay on one side; the split is stratified by class.
import warnings

from sklearn.model_selection import StratifiedGroupKFold

_parent = list(range(len(my_files)))  # union-find: every copy pair joins its two files' groups


def _find(i: int) -> int:
    while _parent[i] != i:
        _parent[i] = _parent[_parent[i]]
        i = _parent[i]
    return i


for _, pr in my_copies.iterrows():
    _parent[_find(pr["ia"])] = _find(pr["ib"])
my_groups = np.array([_find(i) for i in range(len(my_files))])

_smallest = int(my_counts.min())
_n_splits = min(5, max(2, _smallest))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # a class with fewer photos than folds only triggers a warning
    _splitter = StratifiedGroupKFold(n_splits=_n_splits, shuffle=True, random_state=SEED)
    _train_idx, _test_idx = next(_splitter.split(my_files, my_files["class"], groups=my_groups))
my_split = my_files[["file", "class"]].copy()
my_split["split"] = "train"
my_split.loc[_test_idx, "split"] = "test"
_split_copies = sum(my_split.at[pr["ia"], "split"] != my_split.at[pr["ib"], "split"]
                    for _, pr in my_copies.iterrows())
display(pd.crosstab(my_split["class"], my_split["split"]).reindex(MY_CLASSES))
_test_small = int((my_split["split"] == "test").groupby(my_split["class"]).sum().min())
print(f"test share {len(_test_idx) / len(my_files):.0%} · copy pairs across train/test: {_split_copies}")
if _test_small < 10:
    print(f"💡 the smallest class has only {_test_small} test images: prefer stratified {_n_splits}-fold "
          "cross-validation with the same groups, and report the mean and the spread over the folds.")
else:
    print("💡 a stratified 80/20 hold-out with these groups is enough; keep the test half for the end.")
print("If photos share a camera, a session, a burst or a patient, add that to the groups (section 6).")
_split_csv = DATA_DIR / "cache" / "my_split.csv"
my_split.to_csv(_split_csv, index=False)
print(f"split saved to {_split_csv}")

# %% [markdown]
# 💡 Look at "copy pairs across train/test": it must be 0. The split is written to `data/cache/my_split.csv`,
# and **Colab deletes that folder when the session ends**: download it now (Files pane → right-click the
# file → Download) or copy it to your Drive, or it is gone next week.

# %% [markdown]
# The checklist in one place, to paste into your project notes:

# %%
# @title The report card for this dataset { display-mode: "form" }
_n_unreadable = int((~my_stats["readable"]).sum())
_n_not_rgb = int(((my_stats["mode"] != "RGB") & my_stats["readable"]).sum())
my_report = [
    ("✅" if my_imbalance < 3 else "⚠️", f"classes: {len(MY_CLASSES)}, imbalance {my_imbalance:.1f} : 1"),
    ("✅" if _n_unreadable == 0 else "⚠️", f"unreadable files: {_n_unreadable}"),
    ("✅" if _n_not_rgb == 0 else "⚠️", f"files not stored as RGB: {_n_not_rgb}"),
    ("✅" if len(my_copies) == 0 else "⚠️", f"copy pairs: {len(my_copies)} "
     f"({int((~my_copies['same class']).sum())} across class folders)"),
    ("✅" if _split_copies == 0 else "⚠️", f"split: {len(_test_idx)} test / {len(_train_idx)} train, "
     f"copy pairs across the split: {_split_copies}"),
]
for mark, line in my_report:
    print(mark, line)
RESULTS["swap_in_ok"] = bool(len(my_files) > 0 and len(my_stats) == len(my_split) == len(my_files))
RESULTS["swap_in"] = {"source": my_source, "n_images": int(len(my_files)), "n_classes": len(MY_CLASSES),
                      "copies": int(len(my_copies)), "not_rgb": _n_not_rgb}

# %% [markdown]
# ### After the lab · Read your results and answer the guiding questions
#
# Everything above runs in class. This block is for you to do afterwards, on your own dataset: it explains how
# to read each check and ends with the seven guiding questions.

# %% [markdown]
# **Q25.** 🟢 On the sample the folders are cat, dog, bird, horse, sheep and zebra with 20 photos each, plus
# three planted problems (an exact copy of a cat, a mirrored dog, a grayscale bird). Which check caught each
# one? Then run the checks on your own dataset and compare.

# %%
# @title
from IPython.display import HTML; HTML(r"""<div class="callout callout-answer" style="background:#F6EFE4;color:#5C4027;border-left:6px solid #5C4027;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">

<details style="margin:0;"><summary style="cursor:pointer;font-weight:bold;">💡 Worked answer (Q25, on the sample)</summary>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><strong>Check 1:</strong> every folder holds the same number of photos, plus the planted copies in cat and dog, so
  the ratio is close to 1 : 1. That is a property of the sample, not of the park.</li>
<li style="margin:2px 0;"><strong>Check 2:</strong> all sizes fall in the Play Dataset's few clusters (long side 640 or 500 px). The grayscale
  bird shows up as one <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">L</code> file.</li>
<li style="margin:2px 0;"><strong>Check 3:</strong> the exact copy of the cat and the mirrored dog show up as an <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">exact</code> and a <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">flipped</code> pair,
  both inside one folder. The split keeps each pair on one side.</li>
<li style="margin:2px 0;">If a check missed one of them, look at the <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">HASH_THRESHOLD</code> you set in section 5.</li>
</ul>
</details>
</div>""")

# %% [markdown]
# 💡 How to read the ratio:
#
# - **2 or 3 to 1:** common and harmless if you watch the small class's recall.
# - **10 to 1 and beyond:** accuracy hides the small class completely (section 4's ⚠️ Pitfall).
# - **Always ask:** is the *expensive* class the small one?

# %% [markdown]
# 💡 How to read the sizes and modes:
#
# - **One tight cluster of sizes:** one source.
# - **A second cluster, or one class sitting apart:** a second source, and the model may learn the source
#   instead of the animal.
# - **Every file that is not RGB** is listed in check 2: decide whether to convert it or drop it.

# %% [markdown]
# 💡 Every copy pair is a decision for a human:
#
# - **Keep one file of each pair.**
# - **Look twice at pairs from different class folders:** one of the two labels is wrong.
# - **Many pairs on near-identical shots** (a fixed camera, a microscope) are genuinely different photos:
#   lower `HASH_THRESHOLD` in section 5 and rerun.

# %%
# @title
from IPython.display import HTML; HTML(r"""<div class="callout callout-go-deeper" style="background:#F3F4F6;color:#4B5563;border-left:6px solid #4B5563;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">
<div style="font-weight:bold;">🔍 Go deeper · The split is only a suggestion</div>
<p style="margin:6px 0 0 0;">The split comes from what the folders show. Only you know what else ties photos together, such as one
camera, one day, one patient; section 6 showed that a split which ignores such groups reports a better score
than the model will earn.</p>
</div>""")

# %% [markdown]
# #### The guiding questions
#
# These questions are the notebook in short. Each one has the park's answer, and an empty line for yours.
#
# **Q26.** 🟢 Double-click the next cell and answer each guiding question for your project dataset, using
# the output of the checklist above and what you know about where the data comes from.

# %% [markdown]
# 1. **What is the task, and which mistake is the expensive one?**
#    - *The park:* classification of who is in front of the camera; the expensive mistake is a missed dog
#      in the sheep paddock, so the metric is recall on dog (section 4).
#    - *Your dataset:*
# 2. **Do the images look like what the model will see in use?**
#    - *The park:* only partly. COCO has close-ups, sofas and people; a fixed paddock camera sees small
#      animals on grass (section 2).
#    - *Your dataset:*
# 3. **How balanced are the classes, and is the expensive class a small one?**
#    - *The park:* cat and dog about 130 photos each, sheep about 50, roughly 2 : 1 to 3 : 1. Dog is not
#      rare, but accuracy would still hide a weak recall on dog (section 4).
#    - *Your dataset:*
# 4. **Which sizes, aspect ratios and colour modes are there, and what must be resized or converted?**
#    - *The park:* nearly every photo has a long side of 640 or 500 px, in landscape and portrait; files
#      that are not RGB get converted (section 3).
#    - *Your dataset:*
# 5. **Are there copies, flipped or rotated copies, unreadable or odd files, or doubtful labels?**
#    - *The park:* a few copies, removed before splitting (section 5's "removed N, kept M"); photos with
#      several animals make the label ambiguous (section 4).
#    - *Your dataset:*
# 6. **What ties photos together, and which split keeps those groups apart?**
#    - *The park:* COCO has no groups, so a stratified split after deduplication; the trail cameras have
#      bursts and locations, so they are split by camera (section 6).
#    - *Your dataset:*
# 7. **What must be fixed before training?**
#    - *The park:* drop the copies, convert the files that are not RGB, decide the label of multi-animal
#      photos, and resize without distorting the aspect ratio (section 3's 🔀 Callout).
#    - *Your dataset:*

# %%
# @title
from IPython.display import HTML; HTML(r"""<div class="callout callout-takeaway" style="background:#E2F0C4;color:#3A5A0B;border-left:6px solid #3A5A0B;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">
<div style="font-weight:bold;">✅ Takeaway</div>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><strong>When you get a new folder-per-class dataset</strong> → <strong>do</strong> run the four checks (classes, sizes and modes,
  copies, split) before anything else; they take a few minutes.</li>
<li style="margin:2px 0;"><strong>When a check lists files</strong> (not RGB, copy pairs, pairs across class folders) → <strong>do</strong> decide per file:
  convert, keep one, or fix the label.</li>
<li style="margin:2px 0;"><strong>When you leave</strong> → <strong>do</strong> download <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">data/cache/my_split.csv</code> (Colab deletes it) and bring it, with your
  answers to the guiding questions,
  to next week's training.</li>
</ul>
</div>""")
