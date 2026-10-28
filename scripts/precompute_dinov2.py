"""Precompute DINOv2-small CLS embeddings for every Play Dataset photo, as the notebook's compute path does
(CLS token, L2-normalised), so the classroom CPU run only loads them (PRD 2026-10-28, Q8–9).

Needs torch + transformers, which are not project dependencies (the notebook imports them only to compute):
    uv run --with torch --with transformers python scripts/precompute_dinov2.py . data/coco_animals_dinov2.npz
Checked 2026-10-28: within-class median similarities equal Susanne's A100 run to 3 decimals.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-small"
BATCH_SIZE = 16


def main(root: Path, out: Path) -> None:
    coco = json.loads((root / "data/coco_animals_subset.json").read_text())
    cache = root / "data/cache/coco"
    rows = sorted(coco["images"], key=lambda im: im["id"])
    pairs = [(im["id"], cache / im["file_name"]) for im in rows if (cache / im["file_name"]).exists()]
    print(f"{len(rows)} in the manifest, {len(pairs)} downloaded")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()
    vectors, start = [], time.time()
    for first in range(0, len(pairs), BATCH_SIZE):
        batch = []
        for _, path in pairs[first : first + BATCH_SIZE]:
            with Image.open(path) as im:
                batch.append(im.convert("RGB"))
        with torch.inference_mode():
            features = model(**processor(images=batch, return_tensors="pt")).last_hidden_state[:, 0]
            vectors.append(torch.nn.functional.normalize(features, dim=1).numpy())
    embeddings = np.concatenate(vectors).astype("float32")
    ids = np.array([image_id for image_id, _ in pairs], dtype=np.int64)
    np.savez_compressed(out, ids=ids, embeddings=embeddings, model=np.array(MODEL_NAME))
    print(f"{embeddings.shape} in {time.time() - start:.1f} s → {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
