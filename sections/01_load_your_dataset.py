# %% [markdown] edition="own"
# ## Load Your Dataset
#
# In class you explored the Wildwood Farm Park. This notebook runs the same checks on **your** dataset: every cell
# after this chapter reads your images and labels instead of the park's. Where it helps, a short *In the park* note
# reminds you what the same cell showed there, so you know what to look for.
#
# > ▶️ **Run the cells of this chapter in order.** When your dataset is loaded, use *Runtime → Run all* or go chapter by
# > chapter. Chapters your labels cannot support (for example Object Detection without boxes) skip themselves with one
# > line saying why.

# %% edition="own"
# @title 📘 What Your Dataset Must Contain { display-mode: "form" }
callout("theory", "📘 What Your Dataset Must Contain", r"""
<p style="margin:6px 0 0 0;">Each computer vision task needs a different kind of label:</p>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><strong>Classification:</strong> one class per image. <em>"This image shows a sheep."</em></li>
<li style="margin:2px 0;"><strong>Object detection:</strong> a <strong>box</strong> around every object, with its class. <em>"There is a sheep inside this rectangle."</em></li>
<li style="margin:2px 0;"><strong>Semantic segmentation:</strong> a <strong>mask</strong> of every object, with its class: the exact pixels it covers.</li>
</ul>
<p style="margin:6px 0 0 0;">Labels build on each other: from a mask we compute its box, and from the boxes in an image we take the most
frequent class as the image's class. So a dataset with masks can run all three chapters, a dataset with boxes the
first two, and a dataset with class labels only the classification chapters.</p>
""")


# %% [markdown] edition="own"
# #### How Each Format Stores Its Labels
#
# You do not convert anything yourself. Tell the notebook which tasks your labels support, and it reads any of the
# formats below and converts them internally. If your labeling tool offers several exports, **COCO JSON** is the safest
# choice: it holds boxes and masks in one file.
#
# To compare the formats, every example below describes **the same box**: a sheep in a 640 × 480 image, whose box
# starts 120 pixels from the left edge and 80 from the top, and is 200 pixels wide and 150 pixels high.
#
# **Classification**
#
# - **One folder per class** (`"folders"`): the folder name is the class.
#
#   ```
#   my_dataset/
#   ├── horse/   img_001.jpg, img_002.jpg, …
#   └── sheep/   img_101.jpg, …
#   ```
#
# - **A CSV file** (`"csv"`): one row per image, with a `file` column (the path from the CSV's folder) and a `label`
#   column.
#
#   ```
#   file,label
#   images/img_101.jpg,sheep
#   ```

# %% [markdown] edition="own"
# **Object detection (boxes).** A box is always four numbers. The formats differ in *which* four:
#
# | Format | One file per | The four numbers | Our sheep |
# |---|---|---|---|
# | **COCO JSON** (`"coco"`) | dataset | x, y of the top-left corner, width, height, in pixels | `[120, 80, 200, 150]` |
# | **YOLO** (`"yolo"`) | image (`.txt`) | x, y of the **center**, width, height, as **fractions** of the image size (0 to 1) | `0.3438 0.3229 0.3125 0.3125` |
# | **Pascal VOC** (`"voc"`) | image (`.xml`) | x, y of the top-left corner, x, y of the **bottom-right** corner, in pixels | `120 80 320 230` |
#
# - **COCO JSON:** one file lists the images, the annotations and the class names. Each annotation points to its image
#   (`image_id`) and its class (`category_id`):
#
#   ```json
#   {"images":      [{"id": 1, "file_name": "img_101.jpg", "width": 640, "height": 480}],
#    "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "bbox": [120, 80, 200, 150]}],
#    "categories":  [{"id": 2, "name": "sheep"}]}
#   ```
#
# - **YOLO:** `labels/img_101.txt` has one line per object, starting with the class **index**; the class names are in
#   `data.yaml` (`names: [horse, sheep]`, so index 1 is sheep). An image with no objects has an empty or no `.txt` file.
#
#   ```
#   1 0.3438 0.3229 0.3125 0.3125
#   ```
#
# - **Pascal VOC:** `labels/img_101.xml` has one `<object>` per object:
#
#   ```xml
#   <object><name>sheep</name>
#     <bndbox><xmin>120</xmin><ymin>80</ymin><xmax>320</xmax><ymax>230</ymax></bndbox></object>
#   ```

# %% [markdown] edition="own"
# **Semantic segmentation (masks)**
#
# - **COCO JSON** (`"coco"`): each annotation also has a `segmentation`, either a **polygon** (the outline as
#   x1, y1, x2, y2, … in pixels) or an **RLE** (a compressed pixel mask, as most tools export for filled areas).
# - **YOLO segmentation** (`"yolo"`): like YOLO boxes, but after the class index comes the outline,
#   `x1 y1 x2 y2 x3 y3 …`, as fractions 0 to 1. A line with 5 numbers is a box, a longer line an outline.
# - **PNG masks** (`"png-masks"`): one PNG per image in a folder named `masks/`, with the image's name
#   (`masks/img_101.png`). Each pixel stores a class **index**: 0 = background, 255 = ignore (not labeled). The names are
#   in `classes.txt`, one per line, starting with index 0 (`background`). ⚠️ A PNG mask stores classes, not objects: two
#   sheep that touch become one object here.

# %% edition="own"
# @title 🔍 Go Deeper: Not In This List? { display-mode: "form" }
callout("deeper", "🔍 Go Deeper: Not In This List?", r"""
<p style="margin:6px 0 0 0;">Most labeling tools (Roboflow, Label Studio, CVAT, Supervisely, V7) export COCO JSON or YOLO: choose one of those
when you export. Datasets from Kaggle or Hugging Face usually come as one of these formats as well. Videos must first
be saved as image frames. Labels for several tasks at once (for example boxes in YOLO and masks as PNG) are not
combined: load the one that covers the most tasks.</p>
""")


# %% edition="own"
# @title ✏️ Your Turn: Point the Notebook at Your Dataset { display-mode: "form" }
HTML(r"""<div class="callout callout-your-turn" style="background:#FFF4D6;color:#7A4A00;border-left:6px solid #7A4A00;border-radius:6px;padding:12px 14px;line-height:1.5;font-family:'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif;font-size:14px;width:100%;box-sizing:border-box;white-space:normal;overflow-wrap:break-word;margin:2px 0;">
<div style="font-weight:bold;">✏️ Your turn · Point the notebook at your dataset</div>
<p style="margin:6px 0 0 0;">Colab cannot see your laptop's folders, so the dataset first goes where Colab can read it. Pick <strong>one</strong> of
the three options, then do the last step.</p>
<p style="margin:6px 0 0 0;"><strong>Option A · Upload it to this Colab session</strong> (fastest; gone when the session ends)</p>
<ol style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;"><strong>Zip the dataset folder</strong> (your images and your label files) on your laptop: Mac, right-click the folder → <em>Compress</em>; Windows,
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
<li style="margin:2px 0;">Open <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">drive/MyDrive/</code>, find the folder that holds your dataset (or the zip), <strong>right-click →
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
<p style="margin:6px 0 0 0;"><strong>Last step, all options:</strong> paste the path between the quotes of <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">DATASET_PATH</code> in the next cell, set <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">TASKS</code> and <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">LABEL_FORMAT</code>, click into that cell, and
run everything from there: <strong>Runtime → Run cell and below</strong>. If the path is not right, the cell after it
stops with a plain message that says what to change.</p>
<ul style="margin:6px 0 0 0;padding-left:22px;">
<li style="margin:2px 0;">✅ <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">/content/my_dataset.zip</code> (A), <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">/content/drive/MyDrive/project/my_dataset</code> (B) or
  <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">https://drive.google.com/file/d/…/view</code> (C): the folder that holds your images and labels, or a zip of it. A zip may hold any of the formats
  above, with or without one wrapper folder around it.</li>
<li style="margin:2px 0;">❌ <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">…/my_dataset/daisy</code>: one class folder, one level too deep (one folder per class).</li>
<li style="margin:2px 0;">❌ a folder that holds <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">train/</code> and <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">val/</code>: point at <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">…/train</code> instead (then at <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">…/val</code>, if you like). This holds for one folder per class; the other formats read every split together.</li>
</ul>
<p style="margin:6px 0 0 0;">Leave it at <code style="font-family:'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;border-radius:3px;">None</code> to see the notebook on a small park sample first, written in the format you chose.</p>
</div>""")

# %% edition="own"
# ✏️ YOUR TURN
DATASET_PATH = None  # your folder, .zip, or a Google Drive link to a .zip; None = the park sample
TASKS = ["classification"]  # add "detection" if you have boxes, and "segmentation" if you have masks
LABEL_FORMAT = "auto"  # @param ["auto", "folders", "csv", "coco", "yolo", "voc", "png-masks"]

# %% edition="own"
# @title Helpers: Find and Open Your Dataset { display-mode: "form" }
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
        raise SystemExit("❌ DATASET_PATH: this is a link to a Drive *folder*. Zip the folder, upload the "
                         "zip to Drive and share the zip's link, or use option B (Mount Drive, Copy path).")
    found = re.search(r"/file/d/([\w-]+)", url) or re.search(r"[?&]id=([\w-]+)", url)
    if "drive.google.com" not in url or not found:
        raise SystemExit("❌ DATASET_PATH: links work only for a Google Drive file, like "
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
    raise SystemExit("❌ DATASET_PATH: Drive did not hand out this file without a sign-in, so it is "
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
        raise SystemExit(f"❌ DATASET_PATH: {p} does not exist (a typo, or the upload is not there).\n"
                         f"   The nearest folder that exists is {parent}, which holds:\n{_contents(parent)}\n"
                         "   Copy the path from the Files pane: right-click the folder → Copy path.")
    if p.is_file():
        raise SystemExit(f"❌ DATASET_PATH: {p} is a file. Point at the folder that holds your class "
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
    if subdirs and names <= SPLIT_NAMES:  # train/, val/, test/ with the class folders inside: read them together
        print(f"ℹ️ {p.name}/ holds the splits {', '.join(sorted(d.name for d in subdirs))}: they are read together, "
              "so the checks also find copies across your splits.")
    if not subdirs and loose:
        raise SystemExit(f"❌ DATASET_PATH: {p} holds {len(loose)} images but no class folders. If this "
                         f"is one class folder, point one level up, at {p.parent}. If your project has no "
                         "classes (detection, segmentation), move the images into one folder inside, e.g. "
                         f"{p / 'all'}.")
    if not any(_is_image(f) for f in p.rglob("*")):
        raise SystemExit(f"❌ DATASET_PATH: {p} holds no image files "
                         f"({', '.join(sorted(IMAGE_SUFFIXES))}). "
                         f"It holds:\n{_contents(p)}")
    return p


def class_folders(root) -> list[Path]:
    """The class folders under `root`: its sub-folders, or those inside train/, val/, test/ split folders."""
    subdirs = [d for d in _visible(Path(root)) if d.is_dir()]
    if subdirs and {d.name.lower() for d in subdirs} <= SPLIT_NAMES:
        return [c for split in subdirs for c in _visible(split) if c.is_dir()]
    return subdirs


def list_folder_per_class(root) -> pd.DataFrame:
    """One row per image file under `root`: columns path, file, class (the folder it sits in, inside the split
    folder if there is one).

    e.g. a row: {"path": "/content/flowers/rose/img_7.jpg", "file": "rose/img_7.jpg", "class": "rose"}
    """
    rows = []
    for class_dir in class_folders(root):
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
    others = [f for d in class_folders(root) for f in d.rglob("*")
              if f.is_file() and not _is_image(f) and not f.name.startswith(".")]
    skipped = Counter(f.suffix.lower() or "(no extension)" for f in others)
    if skipped:
        notes.append("⚠️ skipped files that are not images: "
                     + ", ".join(f"{n} × {ext}" for ext, n in skipped.most_common()))
    empty = sorted({d.name for d in class_folders(root)} - set(files["class"]))
    if empty:
        notes.append(f"⚠️ class folders with no images: {', '.join(empty)}.")
    if files["class"].nunique() == 1:
        notes.append("⚠️ only one class folder: the class comparisons have nothing to compare (fine for a "
                     "detection project).")
    return notes


# %% edition="own"
# @title Helpers: Read Your Labels (six formats, one table) { display-mode: "form" }
# Every format is converted to the same COCO-style dict the park uses, then build_tables() makes the
# `images` and `annotations` tables, so every later cell runs unchanged on your data.
import csv
import xml.etree.ElementTree as ET

LABEL_FORMATS = ("folders", "csv", "coco", "yolo", "voc", "png-masks")
FORMAT_HAS = {  # what each format can carry
    "folders": {"classification"},
    "csv": {"classification"},
    "coco": {"classification", "detection", "segmentation"},
    "yolo": {"classification", "detection", "segmentation"},
    "voc": {"classification", "detection"},
    "png-masks": {"classification", "detection", "segmentation"},
}
MASK_DIRS = ("masks", "mask", "segmentationclass", "segmentationobject", "segmentation")


def is_mask_dir(name: str) -> bool:
    """A folder of PNG masks: masks/, SegmentationClass/, or any name with "mask" in it (PedMasks/)."""
    return name.lower() in MASK_DIRS or "mask" in name.lower()


def is_yolo_label(path: Path) -> bool:
    """True if the first line with content is numbers only (a YOLO line), not text like "# Annotation …"."""
    for line in path.read_text(errors="ignore").splitlines():
        if line.strip():
            return all(re.fullmatch(r"-?\d+(\.\d+)?(e-?\d+)?", v) for v in line.split())
    return True  # an empty label file is a YOLO image without objects


class CocoBuilder:
    """Collects images and annotations into a COCO-style dict (boxes in pixels: x, y, width, height)."""

    def __init__(self):
        self.images, self.annotations, self.paths, self.categories = [], [], {}, {}
        self.unreadable = []  # files that cannot be opened as images: left out, and counted in the report card

    def category(self, name) -> int:
        return self.categories.setdefault(str(name), len(self.categories) + 1)

    def image(self, path: Path, root: Path) -> int | None:
        try:
            with Image.open(path) as im:  # reads the file header only
                width, height = im.size
        except OSError:  # not an image after all, or a broken file
            self.unreadable.append(str(path.relative_to(root)))
            return None
        image_id = len(self.images) + 1
        self.images.append({"id": image_id, "file_name": str(path.relative_to(root)), "width": width,
                            "height": height, "license": "your dataset"})
        self.paths[image_id] = path
        return image_id

    def annotation(self, image_id: int | None, name, bbox=None, segmentation=None, area=None) -> None:
        if image_id is None:  # its image could not be opened
            return
        if area is None:
            area = float(bbox[2] * bbox[3]) if bbox is not None else None
        self.annotations.append({"id": len(self.annotations) + 1, "image_id": image_id,
                                 "category_id": self.category(name), "bbox": bbox, "area": area,
                                 "segmentation": segmentation, "iscrowd": 0})

    def result(self) -> tuple[dict, dict]:
        cats = [{"id": i, "name": n} for n, i in self.categories.items()]
        if self.unreadable:
            print(f"⚠️ {len(self.unreadable)} files cannot be opened as images and are left out, e.g. "
                  f"{self.unreadable[0]}.")
        return {"images": self.images, "annotations": self.annotations, "categories": cats,
                "unreadable": self.unreadable}, self.paths


def _images_under(root: Path, skip_dirs=lambda name: False) -> list[Path]:
    """Every image file under `root`, leaving out hidden folders and the folders `skip_dirs(name)` picks."""
    found = []
    for f in sorted(root.rglob("*")):
        parts = [p.lower() for p in f.relative_to(root).parts[:-1]]
        if _is_image(f) and not any(p.startswith((".", "__")) or skip_dirs(p) for p in parts):
            found.append(f)
    return found


def _by_stem(files: list[Path]) -> dict[str, Path]:
    return {f.stem: f for f in files}


def _polygon_bbox(xy: list[float]) -> list[float]:
    xs, ys = xy[0::2], xy[1::2]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def _polygon_area(xy: list[float]) -> float:
    xs, ys = np.array(xy[0::2]), np.array(xy[1::2])
    return float(abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))) / 2)


def detect_format(root: Path) -> str:
    """Guess the label format from the files under `root` (the student confirms it in the printout)."""
    for j in sorted(root.rglob("*.json")):
        try:
            head = json.loads(j.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(head, dict) and {"images", "annotations", "categories"} <= set(head):
            return "coco"
    if any(d.is_dir() and is_mask_dir(d.name) for d in root.rglob("*")):
        return "png-masks"
    if any("<annotation" in x.read_text(errors="ignore")[:500] for x in sorted(root.rglob("*.xml"))[:5]):
        return "voc"
    txts = [t for t in root.rglob("*.txt") if t.name.lower() not in ("classes.txt", "obj.names", "readme.txt")]
    if txts and all(is_yolo_label(t) for t in txts[:20]):
        return "yolo"
    if any(root.rglob("*.csv")):
        return "csv"
    return "folders"


def read_folders(root: Path) -> tuple[dict, dict]:
    """One folder per class: data/<class>/<image>."""
    root = find_dataset(root)
    files = list_folder_per_class(root)
    for note in folder_notes(root, files):
        print(note)
    coco = CocoBuilder()
    for _, row in files.iterrows():
        coco.annotation(coco.image(Path(row["path"]), root), row["class"])
    return coco.result()


def read_csv(root: Path) -> tuple[dict, dict]:
    """A .csv with one row per image: a file column (file, filename, image or path) and a label column."""
    table = sorted(root.rglob("*.csv"))[0]
    rows = list(csv.DictReader(table.open(newline="", encoding="utf-8-sig")))
    if not rows:
        raise SystemExit(f"❌ {table.name} is empty.")
    cols = {c.lower().strip(): c for c in rows[0]}
    file_col = next((cols[c] for c in ("file", "filename", "file_name", "image", "path") if c in cols), None)
    label_col = next((cols[c] for c in ("label", "class", "category") if c in cols), None)
    if file_col is None or label_col is None:
        raise SystemExit(f"❌ {table.name} needs a column 'file' and a column 'label'. It has: "
                         f"{', '.join(rows[0])}.")
    by_name = {f.name: f for f in _images_under(root)}
    coco, missing = CocoBuilder(), []
    for row in rows:
        path = table.parent / row[file_col]
        path = path if path.exists() else by_name.get(Path(row[file_col]).name)
        if path is None:
            missing.append(row[file_col])
            continue
        coco.annotation(coco.image(path, root), row[label_col].strip())
    if missing:
        print(f"⚠️ {len(missing)} rows of {table.name} name a file that is not there, e.g. {missing[0]}: skipped.")
    return coco.result()


def read_coco(root: Path) -> tuple[dict, dict]:
    """COCO JSON: images, annotations (bbox = x, y, width, height in pixels; segmentation = polygons or
    RLE) and categories, with the image files next to the JSON or in a folder below it."""
    found = [j for j in sorted(root.rglob("*.json"))
             if {"images", "annotations", "categories"} <= set(json.loads(j.read_text()))]
    if len(found) > 1:
        print(f"ℹ️ {len(found)} COCO files found; reading {found[0].relative_to(root)} "
              f"(the others: {', '.join(str(j.relative_to(root)) for j in found[1:])}).")
    raw = json.loads(found[0].read_text())
    by_name = {f.name: f for f in _images_under(root)}
    names = {c["id"]: c["name"] for c in raw["categories"]}
    coco, new_id, missing = CocoBuilder(), {}, 0
    for im in raw["images"]:
        path = found[0].parent / im["file_name"]
        path = path if path.exists() else by_name.get(Path(im["file_name"]).name)
        if path is None:
            missing += 1
            continue
        new_id[im["id"]] = coco.image(path, root)  # None if it cannot be opened: its annotations are skipped
    for a in raw["annotations"]:
        if a["image_id"] in new_id:
            coco.annotation(new_id[a["image_id"]], names[a["category_id"]], bbox=a.get("bbox"),
                            segmentation=a.get("segmentation") or None, area=a.get("area"))
    if missing:
        print(f"⚠️ {missing} images listed in {found[0].name} are not in the folder: skipped.")
    return coco.result()


def _yolo_names(root: Path) -> dict[int, str]:
    """Class names from data.yaml (names: [...] or an indented list or 0: name lines) or classes.txt."""
    for y in sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml")):
        text = y.read_text()
        inline = re.search(r"^names:\s*\[(.*?)\]", text, re.M | re.S)
        if inline:
            return {i: n.strip().strip("'\"") for i, n in enumerate(inline[1].split(","))}
        block = re.search(r"^names:\s*\n((?:[ \t]+.*\n?)+)", text, re.M)
        if block:
            lines = [ln.strip() for ln in block[1].splitlines() if ln.strip()]
            if all(ln.startswith("- ") for ln in lines):
                return {i: ln[2:].strip().strip("'\"") for i, ln in enumerate(lines)}
            pairs = [ln.split(":", 1) for ln in lines if ":" in ln]
            return {int(k): v.strip().strip("'\"") for k, v in pairs}
    for name in ("classes.txt", "obj.names"):
        for t in root.rglob(name):
            return dict(enumerate(ln.strip() for ln in t.read_text().splitlines() if ln.strip()))
    return {}


def read_yolo(root: Path) -> tuple[dict, dict]:
    """YOLO: one .txt per image with the same name. Each line is a class index, then either a box
    (centre x, centre y, width, height) or a polygon (x1 y1 x2 y2 …), all as fractions 0–1 of the image."""
    names = _yolo_names(root)
    if not names:
        print("⚠️ no data.yaml or classes.txt found: classes are named by their index (class_0, class_1, …).")
    images = _by_stem(_images_under(root))
    coco, unmatched = CocoBuilder(), 0
    labels = [t for t in sorted(root.rglob("*.txt")) if t.name not in ("classes.txt", "obj.names", "README.txt")]
    for t in labels:
        path = images.pop(t.stem, None)
        if path is None:
            unmatched += 1
            continue
        image_id = coco.image(path, root)
        if image_id is None:
            continue
        width, height = coco.images[-1]["width"], coco.images[-1]["height"]
        for n, line in enumerate(t.read_text().splitlines(), 1):
            values = line.split()
            if not values:
                continue
            if len(values) == 5:
                k, cx, cy, w, h = int(values[0]), *map(float, values[1:])
                bbox = [(cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height]
                coco.annotation(image_id, names.get(k, f"class_{k}"), bbox=bbox)
            elif len(values) >= 7 and len(values) % 2 == 1:
                k, xy = int(values[0]), [float(v) for v in values[1:]]
                xy = [v * (width if i % 2 == 0 else height) for i, v in enumerate(xy)]
                coco.annotation(image_id, names.get(k, f"class_{k}"), bbox=_polygon_bbox(xy),
                                segmentation=[xy], area=_polygon_area(xy))
            else:
                raise SystemExit(f"❌ {t.name}, line {n}, has {len(values)} numbers. YOLO needs 5 for a box "
                                 "(class cx cy w h) or an odd number from 7 up for a polygon (class x1 y1 x2 y2 x3 "
                                 "y3 …). A 6th number is usually a confidence score: remove it.")
    for path in images.values():  # images without a label file: no objects in them
        coco.image(path, root)
    if unmatched:
        print(f"⚠️ {unmatched} label files have no image with the same name: skipped.")
    return coco.result()


def read_voc(root: Path) -> tuple[dict, dict]:
    """Pascal VOC: one .xml per image with <object><name> and <bndbox> xmin, ymin, xmax, ymax in pixels."""
    images = _by_stem(_images_under(root))
    coco = CocoBuilder()
    for x in sorted(root.rglob("*.xml")):
        tree = ET.parse(x).getroot()
        if tree.tag != "annotation":
            continue
        name = (tree.findtext("filename") or x.stem).strip()
        path = images.pop(Path(name).stem, None) or images.pop(x.stem, None)
        if path is None:
            print(f"⚠️ {x.name}: no image named {name}: skipped.")
            continue
        image_id = coco.image(path, root)
        if image_id is None:
            continue
        for obj in tree.iter("object"):
            box = obj.find("bndbox")
            x0, y0, x1, y1 = (float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            coco.annotation(image_id, obj.findtext("name").strip(), bbox=[x0, y0, x1 - x0, y1 - y0])
    for path in images.values():
        coco.image(path, root)
    return coco.result()


def read_png_masks(root: Path) -> tuple[dict, dict]:
    """PNG label masks: a folder of masks (masks/, or any folder with "mask" in its name), one PNG per image,
    named like the image (img_7.png or img_7_mask.png), and classes.txt with one class name per line, starting
    with index 0 (background). Pixel values are class indices (0 = background, 255 = ignore), and each
    connected piece of a class becomes one object. If classes.txt names only ONE class besides background, the
    masks may instead number the objects (1, 2, 3 …, as Penn-Fudan does): every value is then its own object."""
    masks = {}
    for d in (d for d in root.rglob("*") if d.is_dir() and is_mask_dir(d.name)):
        for f in sorted(d.glob("*.png")):
            masks[f.stem.removesuffix("_mask")] = f
    classes_txt = next(iter(sorted(root.rglob("classes.txt"))), None)
    if classes_txt is None:
        raise SystemExit("❌ PNG masks need a classes.txt next to them: one class name per line, the first line "
                         "for pixel value 0 (background). Example for two classes:\n    background\n    horse\n"
                         "    sheep\nIf your masks number the objects (1, 2, 3 …) and all objects have one class, "
                         "write only background and that class.")
    names = dict(enumerate(ln.strip() for ln in classes_txt.read_text().splitlines() if ln.strip()))
    one_class = len(names) == 2  # background + one class: every non-zero value is an object of that class
    if one_class:
        print(f"ℹ️ classes.txt names one class ({names[1]}): every value in a mask is one {names[1]}.")
    coco = CocoBuilder()
    for path in _images_under(root, skip_dirs=is_mask_dir):
        image_id = coco.image(path, root)
        mask_path = masks.get(path.stem)
        if image_id is None or mask_path is None:
            continue
        with Image.open(mask_path) as m:
            if m.mode not in ("L", "P", "I", "I;16"):
                raise SystemExit(f"❌ {mask_path.name} is a {m.mode} color image. Masks must store one class "
                                 "index per pixel (a grayscale or palette PNG), not a color picture.")
            labels = np.array(m)
        for k in sorted(set(np.unique(labels).tolist()) - {0, 255}):
            if one_class:  # one object per value
                pieces, n, name = (labels == k).astype(np.uint8), 2, names[1]
            else:  # one object per connected piece of the class
                n, pieces = cv2.connectedComponents((labels == k).astype(np.uint8), connectivity=8)
                name = names.get(k, f"class_{k}")
            for piece in range(1, n):
                rle = mask_utils.encode(np.asfortranarray((pieces == piece).astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                coco.annotation(image_id, name, bbox=[float(v) for v in mask_utils.toBbox(rle)],
                                segmentation=rle, area=float(mask_utils.area(rle)))
    return coco.result()


READERS = {"folders": read_folders, "csv": read_csv, "coco": read_coco, "yolo": read_yolo, "voc": read_voc,
           "png-masks": read_png_masks}


def open_dataset(path) -> Path:
    """A local folder for `path`: a folder, a .zip (unzipped once) or a Google Drive link to a zip."""
    text = str(path).strip().strip("'\"")
    p = fetch_drive_link(text) if text.startswith(("http://", "https://")) else Path(text).expanduser()
    if str(p).startswith("/content/drive") and IN_COLAB and not Path("/content/drive/MyDrive").exists():
        from google.colab import drive

        drive.mount("/content/drive")
    if p.is_file() and p.suffix.lower() == ".zip":
        dest = (DATA_DIR / "cache" / "my_upload" / p.stem).resolve()
        if not dest.exists():
            with zipfile.ZipFile(p) as z:
                z.extractall(dest)
        print(f"ℹ️ unzipped {p.name} into {dest}")
        p = dest
    if not p.exists():
        parent = next(q for q in p.resolve().parents if q.exists())
        raise SystemExit(f"❌ DATASET_PATH: {p} does not exist (a typo, or the upload is not there).\n"
                         f"   The nearest folder that exists is {parent}, which holds:\n{_contents(parent)}\n"
                         "   Copy the path from the Files pane: right-click the folder → Copy path.")
    if not p.is_dir():
        raise SystemExit(f"❌ DATASET_PATH: {p} is a file. Point at the folder that holds your images and "
                         "labels, at a .zip, or at a Google Drive link to a zip.")
    while True:  # a zip often wraps everything in one more folder: step into it
        inside = _visible(p)
        if len(inside) == 1 and inside[0].is_dir() and any(d.is_dir() for d in _visible(inside[0])):
            p = inside[0]
            continue
        return p


def read_dataset(path, label_format: str, tasks: list[str]) -> tuple[dict, dict, str, Path]:
    """(COCO-style dict, {image id: path}, the format used, the dataset folder). Stops with a plain message if the format cannot
    carry the tasks you asked for, or if no image was found."""
    root = open_dataset(path)
    used = detect_format(root) if label_format == "auto" else label_format
    if used not in READERS:
        raise SystemExit(f"❌ LABEL_FORMAT = {label_format!r}: use \"auto\" or one of {', '.join(LABEL_FORMATS)}.")
    missing = set(tasks) - FORMAT_HAS[used]
    if missing:
        raise SystemExit(f"❌ TASKS asks for {' and '.join(sorted(missing))}, but {used} labels carry no "
                         f"{'boxes' if 'detection' in missing else 'masks'}. Remove it from TASKS, or export "
                         "your labels as COCO JSON (boxes and masks) or YOLO (boxes, or polygons for masks).")
    coco, paths = READERS[used](root)
    if not paths:
        raise SystemExit(f"❌ no images found under {root} (looked for {', '.join(sorted(IMAGE_SUFFIXES))}).")
    if "detection" in tasks and not any(a["bbox"] for a in coco["annotations"]):
        raise SystemExit(f"❌ TASKS includes detection, but no box was found in your {used} labels.")
    if "segmentation" in tasks and not any(a["segmentation"] for a in coco["annotations"]):
        raise SystemExit(f"❌ TASKS includes segmentation, but no mask was found in your {used} labels.")
    return coco, paths, used, root


def _shorten(value, keep: int = 8):
    """A label made readable: long number lists (outlines) keep their first `keep` values, long strings
    (RLE masks) their first 40 characters."""
    if isinstance(value, dict):
        return {k: _shorten(v, keep) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > keep and all(isinstance(v, (int, float)) for v in value):
            return [*value[:keep], f"… {len(value) - keep} more numbers"]
        return [_shorten(v, keep) for v in value]
    if isinstance(value, str) and len(value) > 40:
        return value[:40] + f"… ({len(value)} characters)"
    return value


def stored_label(fmt: str, root: Path, path: Path) -> str:
    """The labels of the image at `path` exactly as your files store them, in the format's own units."""
    rel = path.relative_to(root)
    if fmt == "folders":
        return f"{rel}: the folder the image sits in is its class."
    if fmt == "csv":
        table = sorted(root.rglob("*.csv"))[0]
        lines = table.read_text(encoding="utf-8-sig").splitlines()
        row = next((ln for ln in lines[1:] if path.name in ln), "(no row names this file)")
        return f"{table.relative_to(root)}:\n{lines[0]}\n{row}"
    if fmt == "coco":
        found = next(j for j in sorted(root.rglob("*.json"))
                     if {"images", "annotations", "categories"} <= set(json.loads(j.read_text())))
        raw = json.loads(found.read_text())
        image = next(im for im in raw["images"] if Path(im["file_name"]).name == path.name)
        anns = [a for a in raw["annotations"] if a["image_id"] == image["id"]]
        used = {a["category_id"] for a in anns}
        parts = [f"{found.relative_to(root)}, the image:", json.dumps(image, ensure_ascii=False)]
        parts += [f"its {len(anns)} annotation(s):"] + [json.dumps(_shorten(a), ensure_ascii=False) for a in anns]
        parts += ["their categories:"] + [json.dumps(c, ensure_ascii=False) for c in raw["categories"] if c["id"] in used]
        return "\n".join(parts)
    if fmt == "yolo":
        names = _yolo_names(root)
        label = next((t for t in sorted(root.rglob(f"{path.stem}.txt"))), None)
        if label is None:
            return f"no {path.stem}.txt: an image without a label file has no objects."
        lines = [ln if len(ln.split()) <= 9 else " ".join(ln.split()[:9]) + f" … {len(ln.split()) - 9} more numbers"
                 for ln in label.read_text().splitlines() if ln.strip()]
        return "\n".join([f"{label.relative_to(root)}:", *lines,
                          "class names: " + ", ".join(f"{k} = {n}" for k, n in sorted(names.items()))])
    if fmt == "voc":
        for x in sorted(root.rglob("*.xml")):
            tree = ET.parse(x).getroot()
            if tree.tag == "annotation" and Path((tree.findtext("filename") or x.stem).strip()).stem == path.stem:
                objects = [re.sub(r">\s+<", "><", ET.tostring(o, encoding="unicode").strip())
                           for o in tree.iter("object")]
                return "\n".join([f"{x.relative_to(root)}, its <object> elements:", *objects])
        return f"no .xml names {path.name}: an image without a label file has no objects."
    if fmt == "png-masks":
        mask = next((f for d in root.rglob("*") if d.is_dir() and d.name.lower() in MASK_DIRS
                     for f in d.glob(f"{path.stem}.png")), None)
        if mask is None:
            return f"no mask named {path.stem}.png: an image without a mask has no objects."
        names = next((dict(enumerate(ln.strip() for ln in t.read_text().splitlines() if ln.strip()))
                      for t in root.rglob("classes.txt")), {})
        with Image.open(mask) as m:
            values, counts = np.unique(np.array(m), return_counts=True)
        pixels = " · ".join(f"{v} = {names.get(int(v), 'ignore' if v == 255 else '?')}: {n:,} pixels"
                            for v, n in zip(values, counts, strict=True))
        return f"{mask.relative_to(root)}: one class index per pixel.\n{pixels}"
    return "(unknown format)"


# %% edition="own"
# @title Helpers: Park Sample, Class Colors and Skipped Cells { display-mode: "form" }
# With DATASET_PATH = None the notebook writes a small park sample in the format you chose and reads it back
# with the same reader your data would use, so you can see what your files should look like.


def write_sample(fmt: str, dest: Path, imgs: pd.DataFrame, anns: pd.DataFrame, polygons: bool = False) -> Path:
    """Write the park photos in `imgs` and their annotations in `anns` to `dest` in format `fmt`.

    polygons=True writes YOLO outlines (class x1 y1 x2 y2 …) instead of boxes, for segmentation."""
    shutil.rmtree(dest, ignore_errors=True)  # rebuilt from scratch: the same sample every run (N-013)
    names = sorted(anns["class"].unique())
    index = {c: i for i, c in enumerate(names)}
    folder = dest / ("images" if fmt != "folders" else "")
    for _, im in imgs.iterrows():
        target = dest / im["class"] if fmt == "folders" else folder
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(im["path"], target / Path(im["path"]).name)
    by_image = {k: g for k, g in anns.groupby("image_id")}
    if fmt == "csv":
        pd.DataFrame({"file": "images/" + imgs["path"].map(lambda p: Path(p).name),
                      "label": imgs["class"]}).to_csv(dest / "labels.csv", index=False)
    elif fmt == "coco":
        keep = anns[anns["image_id"].isin(imgs["id"])]
        coco = {"images": [{"id": int(r["id"]), "file_name": Path(r["path"]).name, "width": int(r["width"]),
                            "height": int(r["height"])} for _, r in imgs.iterrows()],
                "annotations": [{"id": int(a["id"]), "image_id": int(a["image_id"]),
                                 "category_id": index[a["class"]] + 1, "bbox": list(a["bbox"]),
                                 "area": float(a["area"]), "segmentation": a["segmentation"],
                                 "iscrowd": int(a["iscrowd"])} for _, a in keep.iterrows()],
                "categories": [{"id": i + 1, "name": c} for c, i in index.items()]}
        (folder / "annotations.json").write_text(json.dumps(coco))
    elif fmt in ("yolo", "voc", "png-masks"):
        labels = dest / ("masks" if fmt == "png-masks" else "labels")
        labels.mkdir(parents=True, exist_ok=True)
        for _, im in imgs.iterrows():
            w, h, stem = im["width"], im["height"], Path(im["path"]).stem
            own = by_image.get(im["id"], anns.iloc[:0])
            if fmt == "yolo" and polygons:  # one line per outline: class x1 y1 x2 y2 …, fractions of the photo
                lines = [f"{index[a['class']]} " + " ".join(
                    f"{v / (w if i % 2 == 0 else h):.6f}" for i, v in enumerate(outline))
                    for _, a in own.iterrows() if isinstance(a["segmentation"], list)
                    for outline in a["segmentation"]]  # an object in two parts becomes two lines
                (labels / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            elif fmt == "yolo":  # one line per object: class cx cy w h, as fractions of the photo
                lines = [f"{index[a['class']]} {(a['bbox'][0] + a['bbox'][2] / 2) / w:.6f} "
                         f"{(a['bbox'][1] + a['bbox'][3] / 2) / h:.6f} {a['bbox'][2] / w:.6f} {a['bbox'][3] / h:.6f}"
                         for _, a in own.iterrows()]
                (labels / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            elif fmt == "voc":
                objects = "".join(
                    f"<object><name>{a['class']}</name><bndbox><xmin>{a['bbox'][0]:.1f}</xmin><ymin>"
                    f"{a['bbox'][1]:.1f}</ymin><xmax>{a['bbox'][0] + a['bbox'][2]:.1f}</xmax><ymax>"
                    f"{a['bbox'][1] + a['bbox'][3]:.1f}</ymax></bndbox></object>" for _, a in own.iterrows())
                (labels / f"{stem}.xml").write_text(
                    f"<annotation><filename>{Path(im['path']).name}</filename><size><width>{w}</width>"
                    f"<height>{h}</height></size>{objects}</annotation>")
            else:  # one class index per pixel, 0 = background; class names in classes.txt
                canvas = np.zeros((h, w), dtype=np.uint8)
                for _, a in own.iterrows():
                    canvas[decode(to_rle(a["segmentation"], h, w)) > 0] = index[a["class"]] + 1
                Image.fromarray(canvas).save(labels / f"{stem}.png")
        if fmt == "yolo":
            (dest / "data.yaml").write_text("names: [" + ", ".join(names) + "]\n")
        if fmt == "png-masks":
            (dest / "classes.txt").write_text("\n".join(["background", *names]) + "\n")
    return dest


# The park's six colors first (cat, dog, bird, horse, sheep, zebra), then 16 more: 22 distinct class colors.
CLASS_PALETTE = [
    "#E07A5F", "#3D405B", "#F2CC8F", "#81B29A", "#9C89B8", "#2A9D8F",
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#1F77B4", "#D62728", "#8C564B", "#17BECF", "#BCBD22", "#6A3D9A", "#B15928",
]  # fmt: skip


def class_colors(classes: list[str]) -> dict[str, str]:
    """One color per class: a park class keeps its park color, any other class takes the next unused one.

    e.g. class_colors(["rose", "cat"]) → {"rose": "#3D405B", "cat": "#E07A5F"}
    """
    park = dict(zip(PARK_CLASSES, CLASS_PALETTE, strict=False))
    taken = {c: park[c] for c in classes if c in park}
    free = [col for col in CLASS_PALETTE if col not in taken.values()]
    others = [c for c in classes if c not in taken and c != "(no objects)"]
    colors = {c: taken.get(c) for c in classes}
    colors.update({c: free[k % len(free)] for k, c in enumerate(others)})  # repeats only after 22 classes
    if "(no objects)" in classes:
        colors["(no objects)"] = GRAY
    return colors


# Cells whose second line says `# needs: boxes` or `# needs: masks` run only when your labels have them.
NEEDS = {"boxes": "detection", "masks": "segmentation"}
_skips_announced = set()


def announce_skip(need: str) -> None:
    """Say in every skipped cell that it was skipped; the first time per kind, also why."""
    if need not in _skips_announced:
        _skips_announced.add(need)
        print(f"⏭️ Skipped: this part needs {need}, and TASKS does not include {NEEDS[need]!r}.")
    else:
        print(f"⏭️ Skipped: needs {need}.")


def _skip_without_labels(lines: list[str]) -> list[str]:
    """IPython input hook: a cell that needs boxes or masks TASKS does not include runs announce_skip instead.
    IPython may call a hook more than once per cell, so it keeps no state of its own."""
    found = re.search(r"^# needs: (boxes|masks)\s*$", "".join(lines[:3]), re.M)
    if found is None or NEEDS[found[1]] in TASKS:
        return lines
    return [f"announce_skip({found[1]!r})\n"]


_ip = get_ipython()
_ip.input_transformers_cleanup[:] = [t for t in _ip.input_transformers_cleanup
                                     if getattr(t, "__name__", "") != "_skip_without_labels"]
_ip.input_transformers_cleanup.append(_skip_without_labels)

# %% edition="own"
# @title Load Your Dataset (or the Park Sample) { display-mode: "form" }
if "segmentation" in TASKS and "detection" not in TASKS:
    TASKS.append("detection")  # every mask also gives its box
if DATASET_PATH is None:  # the park sample, written in your format and read back the way your files would be
    _fmt = LABEL_FORMAT if LABEL_FORMAT != "auto" else ("coco" if "detection" in TASKS else "folders")
    _play_images, _play_annotations = load_play_dataset(per_class=4 if SMOKE else 20)  # fewer in smoke (N-014)
    _sample = write_sample(_fmt, DATA_DIR / "cache" / "my_sample" / _fmt, _play_images, _play_annotations,
                           polygons=_fmt == "yolo" and "segmentation" in TASKS)
    coco_raw, _paths, label_format_used, dataset_root = read_dataset(_sample, LABEL_FORMAT, TASKS)
    dataset_source = f"the park sample, written as {_fmt} (DATASET_PATH is None)"
else:
    coco_raw, _paths, label_format_used, dataset_root = read_dataset(DATASET_PATH, LABEL_FORMAT, TASKS)
    dataset_source = str(dataset_root)

# Large datasets are sampled down to MY_MAX_IMAGES, keeping the class shares, before anything is measured.
if len(_paths) > MY_MAX_IMAGES:
    _names = {c["id"]: c["name"] for c in coco_raw["categories"]}
    _labels = {}
    for _a in coco_raw["annotations"]:
        _labels.setdefault(_a["image_id"], []).append(_names[_a["category_id"]])
    _class_of = pd.Series({i: Counter(_labels[i]).most_common(1)[0][0] if i in _labels else "(no objects)"
                           for i in _paths})
    _keep = pd.concat([g.sample(max(1, round(len(g) * MY_MAX_IMAGES / len(_paths))), random_state=SEED)
                       for _, g in _class_of.groupby(_class_of)]).index
    print(f"⚠️ {len(_paths)} images found: the notebook runs on a stratified sample of {len(_keep)}.")
    _paths = {i: _paths[i] for i in sorted(_keep)}

images, annotations = build_tables(coco_raw, _paths)
# Every later cell expects each image to have a class (as in the park), so images without objects are set aside.
no_object_images = images[images["class"] == "(no objects)"].reset_index(drop=True)
images = images[images["class"] != "(no objects)"].reset_index(drop=True)
images_by_id = images.set_index("id")  # the same table, looked up by image id
RESULTS["n_images"] = int(len(images))

# Classes are decided once, here: the classes that label at least one image, sorted by number of images. Every
# class gets a color, also one that only appears next to a more frequent class: it is drawn on its boxes, but has
# no panel of its own in the per-class figures.
_counts = Counter(images["class"])
_object_counts = Counter(annotations["class"])
CLASSES = sorted(set(_counts) - {"(no objects)"}, key=lambda c: (-_counts[c], -_object_counts[c], c))
_object_only = sorted(set(_object_counts) - set(CLASSES) - {"(no objects)"}, key=lambda c: (-_object_counts[c], c))
CLASS_COLORS = class_colors(CLASSES + _object_only)
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=[CLASS_COLORS[c] for c in CLASSES])
RESULTS["class_colors"] = dict(CLASS_COLORS)

_tasks_run = ["classification"] + [t for t in ("detection", "segmentation") if t in TASKS]
_objects = f" · {int(annotations['bbox'].notna().sum())} objects (boxes)" if "detection" in TASKS else ""
print(f"✅ source: {dataset_source} · format: {label_format_used} · {len(images)} images{_objects}")
print(f"{len(CLASSES)} classes: " + " · ".join(f"{c} {_counts[c]}" for c in CLASSES))
if len(no_object_images):
    print(f"ℹ️ {len(no_object_images)} images have no objects (no box or mask): they are set aside in the table "
          "no_object_images, and the rest of the notebook runs on the images with objects.")
if _object_only:
    print(f"ℹ️ {len(_object_only)} classes appear only next to a more frequent class in their images "
          f"({', '.join(_object_only[:6])}{' …' if len(_object_only) > 6 else ''}): their boxes are drawn, but they "
          "have no images of their own and no panel in the per-class figures.")
print(f"tasks that will run: {', '.join(_tasks_run)}")

# %% edition="own"
# @title 🔎 See How One Image Is Labeled { display-mode: "form" }
# One image, drawn with its labels, and its labels exactly as your files store them (in the format's own units).
_small = images[images["n_objects"].between(1, 3)] if "detection" in TASKS else images
_one = (_small if len(_small) else images).iloc[0]
_path = Path(_one["path"])
_anns = annotations[annotations["image_id"] == _one["id"]]
print(f"{_path.name} · class: {_one['class']} · {int(_one['width'])} × {int(_one['height'])} px\n")
print(f"How your files ({label_format_used}) store its labels:")
print(textwrap.indent(stored_label(label_format_used, dataset_root, _path), "    "))
if "detection" in TASKS:
    print("\nThe same boxes in the notebook's `annotations` table (x, y of the top-left corner, width, height, "
          "in pixels):")
    for _, _a in _anns.iterrows():
        print(f"    {_a['class']}: [{', '.join(f'{v:.0f}' for v in _a['bbox'])}]")

fig, ax = plt.subplots(figsize=(6, 4.5))
with Image.open(_path) as _im:
    ax.imshow(_im.convert("RGB"))
for _, _a in _anns.iterrows():
    _color = CLASS_COLORS[_a["class"]]
    if "segmentation" in TASKS and _a["segmentation"] is not None:
        _m = decode(to_rle(_a["segmentation"], int(_one["height"]), int(_one["width"])))
        ax.imshow(np.ma.masked_where(_m == 0, _m), cmap=mpl.colors.ListedColormap([_color]), alpha=0.45)
    if "detection" in TASKS and _a["bbox"] is not None:
        _x, _y, _w, _h = _a["bbox"]
        ax.add_patch(plt.Rectangle((_x, _y), _w, _h, fill=False, edgecolor=_color, linewidth=2))
        ax.text(_x, _y - 4, _a["class"], color=_color, fontsize=9, fontweight="bold")
ax.set_title(f"{_path.name} · {_one['class']}", fontsize=10)
frame(ax, CLASS_COLORS[_one["class"]])
plt.tight_layout()
plt.show()
