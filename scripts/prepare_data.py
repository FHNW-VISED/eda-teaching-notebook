"""Build the two small committed data manifests the notebook loads (issue #1).

    uv run python -I scripts/prepare_data.py

Outputs (committed, each well under 5 MB, N-019):
  data/coco_animals_subset.json  COCO-format annotation subset of val2017, six animal classes,
                                 images with exactly one animal class, cat/dog capped at 130.
                                 Licence: COCO annotations CC BY 4.0. Images are never committed;
                                 the notebook fetches them from each image's coco_url at runtime.
  data/cct20_slice.csv           Caltech Camera Traps slice: 5 classes x 8 locations, up to 4 whole
                                 sequences per pair. Licence: CDLA-Permissive. Images are fetched from
                                 LILA's public bucket at runtime.

Downloads go to data/cache/ (gitignored) and resume if interrupted. Only the standard library is used.
"""

from __future__ import annotations

import csv
import io
import json
import random
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
SEED = 42

# --- COCO -------------------------------------------------------------------------------------------
COCO_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_ZIP = CACHE / "annotations_trainval2017.zip"
COCO_MEMBER = "annotations/instances_val2017.json"
COCO_OUT = DATA / "coco_animals_subset.json"
COCO_CLASSES = ["cat", "dog", "bird", "horse", "sheep", "zebra"]
COCO_CAP = {"cat": 130, "dog": 130}
COCO_MAX_BYTES = 2 * 1024 * 1024

# --- CCT20 ------------------------------------------------------------------------------------------
CCT_URL = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz"
CCT_TAR = CACHE / "eccv_18_annotations.tar.gz"
CCT_OUT = DATA / "cct20_slice.csv"
CCT_CLASSES = ["opossum", "raccoon", "coyote", "bobcat", "rabbit"]
CCT_LOCATIONS = [46, 43, 38, 33, 100, 115, 120, 88]
CCT_SEQS_PER_PAIR = 4
CCT_IMAGE_BASES = [
    "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/",
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/",
]
CCT_COLUMNS = ["file_name", "url", "label", "location", "seq_id", "frame_num", "datetime", "bbox"]


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Download `url` to `dest`, resuming a half-finished `.part` file (N-019)."""
    if dest.exists():
        print(f"  cached: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as err:
        if err.code != 416:  # 416 = range not satisfiable: the .part file is already complete
            raise
        part.rename(dest)
        return dest
    with resp:
        if have and resp.status != 206:  # server ignored the Range header: start over
            have = 0
        total = have + int(resp.headers.get("Content-Length") or 0)
        mode = "ab" if have else "wb"
        print(f"  downloading {dest.name} ({total / 1e6:.0f} MB, resuming at {have / 1e6:.0f} MB)")
        with open(part, mode) as fh:
            done = have
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
                print(f"\r  {done / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    part.rename(dest)
    return dest


# ====================================================================================================
# COCO animals subset
# ====================================================================================================
def build_coco() -> dict:
    print("COCO")
    download(COCO_URL, COCO_ZIP)
    with zipfile.ZipFile(COCO_ZIP) as zf, zf.open(COCO_MEMBER) as fh:
        coco = json.load(fh)

    cats = {c["name"]: c for c in coco["categories"] if c["name"] in COCO_CLASSES}
    cat_ids = {c["id"]: name for name, c in cats.items()}
    all_cats = {c["id"]: c["name"] for c in coco["categories"]}

    per_image: dict[int, list[dict]] = defaultdict(list)
    other: dict[int, list[str]] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in cat_ids:
            per_image[ann["image_id"]].append(ann)
        else:
            other[ann["image_id"]].append(all_cats[ann["category_id"]])

    # Images with exactly one animal class (any number of instances).
    by_class: dict[str, list[int]] = defaultdict(list)
    for image_id, anns in per_image.items():
        classes = {cat_ids[a["category_id"]] for a in anns}
        if len(classes) == 1:
            by_class[classes.pop()].append(image_id)

    rng = random.Random(SEED)
    keep: set[int] = set()
    for name in COCO_CLASSES:
        ids = sorted(by_class[name])
        cap = COCO_CAP.get(name)
        if cap is not None and len(ids) > cap:
            ids = sorted(rng.sample(ids, cap))
        keep.update(ids)

    image_fields = [
        "id", "file_name", "width", "height", "license", "flickr_url", "coco_url", "date_captured",
    ]
    images = []
    for img in sorted(coco["images"], key=lambda i: i["id"]):
        if img["id"] not in keep:
            continue
        row = {k: img[k] for k in image_fields if k in img}
        row["other_categories"] = sorted(set(other.get(img["id"], [])))  # e.g. people in animal photos
        images.append(row)

    def round_seg(seg, nd):
        if isinstance(seg, list):  # polygons
            return [[round(v, nd) for v in poly] for poly in seg]
        return seg  # crowd RLE: leave as is

    ann_fields = ["id", "image_id", "category_id", "segmentation", "bbox", "area", "iscrowd"]
    annotations = [
        {k: a[k] for k in ann_fields}
        for a in sorted(coco["annotations"], key=lambda a: a["id"])
        if a["image_id"] in keep and a["category_id"] in cat_ids
    ]

    out = {
        "info": {
            "description": "COCO 2017 val animal subset for the Wildwood Farm Park teaching notebook",
            "source": "COCO 2017 instances_val2017.json (annotations CC BY 4.0)",
            "url": "https://cocodataset.org",
            "classes": COCO_CLASSES,
            "selection": "images containing exactly one of the six classes; cat and dog capped at 130 "
            f"images each (seed {SEED})",
            "note": "Images are not redistributed: each image keeps its license and flickr_url and is "
            "downloaded from coco_url at runtime.",
        },
        "licenses": coco["licenses"],
        "images": images,
        "annotations": annotations,
        "categories": sorted(cats.values(), key=lambda c: c["id"]),
    }

    # Keep the file <= 2 MB: round polygon coordinates only as far as needed.
    for nd in (None, 1, 0):
        if nd is not None:
            for a in out["annotations"]:
                a["segmentation"] = round_seg(a["segmentation"], nd)
                a["bbox"] = [round(v, nd) for v in a["bbox"]]
                a["area"] = round(a["area"], nd)
        text = json.dumps(out, separators=(",", ":"))
        if len(text.encode()) <= COCO_MAX_BYTES:
            break
    else:
        raise SystemExit(f"{COCO_OUT} would be larger than {COCO_MAX_BYTES} bytes even after rounding")
    COCO_OUT.write_text(text + "\n")

    img_class = {}
    for a in annotations:
        img_class[a["image_id"]] = cat_ids[a["category_id"]]
    img_per_class = Counter(img_class.values())
    inst_per_class = Counter(cat_ids[a["category_id"]] for a in annotations)
    multi = sum(1 for n in Counter(a["image_id"] for a in annotations).values() if n >= 2)
    print(f"  wrote {COCO_OUT.relative_to(ROOT)} ({len(text) / 1e6:.2f} MB, rounding={nd})")
    print(f"  images: {len(images)}  annotations: {len(annotations)}  images with >= 2 animals: {multi}")
    print("  images per class:    " + ", ".join(f"{c} {img_per_class[c]}" for c in COCO_CLASSES))
    print("  instances per class: " + ", ".join(f"{c} {inst_per_class[c]}" for c in COCO_CLASSES))
    return out


# ====================================================================================================
# CCT20 slice
# ====================================================================================================
def head_ok(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def build_cct() -> list[dict]:
    print("CCT20")
    download(CCT_URL, CCT_TAR)
    images: dict[str, dict] = {}
    anns_by_image: dict[str, list[dict]] = defaultdict(list)
    categories: dict[int, str] = {}
    with tarfile.open(CCT_TAR) as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".json"):
                continue
            data = json.load(io.TextIOWrapper(tar.extractfile(member), encoding="utf-8"))
            categories.update({c["id"]: c["name"] for c in data["categories"]})
            for img in data["images"]:
                images.setdefault(img["id"], img)
            for ann in data["annotations"]:
                anns_by_image[ann["image_id"]].append(ann)

    wanted = set(CCT_CLASSES)
    locations = set(CCT_LOCATIONS)

    # Group every image at a wanted location by its sequence; keep whole, single-label sequences.
    seqs: dict[str, list[dict]] = defaultdict(list)
    for img in images.values():
        if int(img["location"]) in locations:
            seqs[img["seq_id"]].append(img)

    candidates: dict[tuple[str, int], list[str]] = defaultdict(list)
    for seq_id, frames in seqs.items():
        if len(frames) != frames[0].get("seq_num_frames", len(frames)):
            continue  # a frame of this sequence is missing from the annotations: not whole
        labels = set()
        for img in frames:
            names = {categories[a["category_id"]] for a in anns_by_image.get(img["id"], [])}
            labels |= names or {"<unlabelled>"}
        if len(labels) != 1:
            continue
        label = labels.pop()
        if label in wanted:
            candidates[(label, int(frames[0]["location"]))].append(seq_id)

    rng = random.Random(SEED)
    chosen: list[tuple[str, int, str]] = []
    for label in CCT_CLASSES:
        for loc in CCT_LOCATIONS:
            ids = sorted(candidates.get((label, loc), []))
            if len(ids) > CCT_SEQS_PER_PAIR:
                ids = sorted(rng.sample(ids, CCT_SEQS_PER_PAIR))
            chosen.extend((label, loc, s) for s in ids)

    rows = []
    for label, loc, seq_id in chosen:
        for img in sorted(seqs[seq_id], key=lambda i: (i.get("frame_num", 0), i["file_name"])):
            anns = anns_by_image.get(img["id"], [])
            boxes = [[round(v, 2) for v in a["bbox"]] for a in anns if "bbox" in a]
            rows.append(
                {
                    "file_name": img["file_name"],
                    "label": label,
                    "location": loc,
                    "seq_id": seq_id,
                    "frame_num": img.get("frame_num", ""),
                    "datetime": img.get("date_captured", ""),
                    "bbox": json.dumps(boxes, separators=(",", ":")) if boxes else "",
                }
            )
    if not rows:
        raise SystemExit("no CCT20 sequences selected")

    base = CCT_IMAGE_BASES[0]
    if not head_ok(base + rows[0]["file_name"]):
        base = CCT_IMAGE_BASES[1]
        print(f"  primary image host unreachable, using fallback {base}")
        if not head_ok(base + rows[0]["file_name"]):
            print("  warning: fallback host did not answer the HEAD request either", file=sys.stderr)
    for r in rows:
        r["url"] = base + r["file_name"]

    with open(CCT_OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CCT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    per_class = Counter(r["label"] for r in rows)
    per_loc = Counter(r["location"] for r in rows)
    seq_per_class = Counter(label for label, _, _ in chosen)
    print(f"  wrote {CCT_OUT.relative_to(ROOT)}: {len(rows)} images, {len(chosen)} sequences, "
          f"{len(per_loc)} locations, {sum(1 for r in rows if r['bbox'])} images with boxes")
    print("  images per class:    " + ", ".join(f"{c} {per_class[c]}" for c in CCT_CLASSES))
    print("  sequences per class: " + ", ".join(f"{c} {seq_per_class[c]}" for c in CCT_CLASSES))
    print("  images per location: " + ", ".join(f"{loc} {per_loc[loc]}" for loc in CCT_LOCATIONS))
    return rows


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    build_coco()
    build_cct()
