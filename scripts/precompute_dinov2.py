"""Precompute DINOv2-small CLS embeddings for every Play Dataset photo, as the notebook's compute path does
(CLS token, L2-normalised), so the classroom CPU run only loads them (PRD 2026-10-28, Q8–9).

Needs torch + transformers, which are not project dependencies (the notebook imports them only when computing):
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

root = Path(sys.argv[1]); out = Path(sys.argv[2])
coco = json.loads((root / "data/coco_animals_subset.json").read_text())
rows = sorted(coco["images"], key=lambda im: im["id"])
cache = root / "data/cache/coco"
pairs = [(im["id"], cache / im["file_name"]) for im in rows if (cache / im["file_name"]).exists()]
print(len(rows), "in manifest,", len(pairs), "cached")
name = "facebook/dinov2-small"
proc = AutoImageProcessor.from_pretrained(name); model = AutoModel.from_pretrained(name).eval()
vecs, t0 = [], time.time()
for s in range(0, len(pairs), 16):
    batch = []
    for _, p in pairs[s:s + 16]:
        with Image.open(p) as im: batch.append(im.convert("RGB"))
    with torch.inference_mode():
        f = model(**proc(images=batch, return_tensors="pt")).last_hidden_state[:, 0]
        vecs.append(torch.nn.functional.normalize(f, dim=1).numpy())
emb = np.concatenate(vecs).astype("float32")
np.savez_compressed(out, ids=np.array([i for i, _ in pairs], dtype=np.int64), embeddings=emb, model=np.array(name))
print("shape", emb.shape, f"{time.time()-t0:.1f}s", out, out.stat().st_size // 1024, "KB")
