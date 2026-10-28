# Work list: Susanne's copy (2026-10-28)

The one list of what is done to her copy, in order. Details per item live in the two audits next to this file:
`2026-10-28-susanne-copy/cleanup-audit.md` (C-, N-, T-items and the Edition map E) and
`2026-10-28-susanne-copy/run-audit.md` (R-items: what breaks when it runs, timings). Cell numbers are indices into
`2026-10-28-susanne-copy/her-copy.ipynb`; every item also names the cell title, so it can be found in `sections/`.

## Rules for whoever applies this

- Her copy is the base. Her visible text changes **only** where an item below says so.
- What she removed stays removed (SN-010). Only what a removal breaks is fixed.
- One commit per group below; `just build` and the smoke run of the Teaching Edition must pass after each group.
- Never edit `reviews/2026-10-28-susanne-copy/` (her copy, the audits) or `versions/`.

## Decisions (Simone, 2026-10-28)

All `fix` items in both audits are applied. The `propose` items were answered as follows:

| # | Item(s) | Answer |
|---|---|---|
| P1 | C-12 one `callout()` helper for her boxes (inner HTML verbatim; one Sample Answer style) | yes |
| P2 | C-13 one `browse()` helper | **no** — N-13's renames fix the bug; her widgets stay as built |
| P3 | C-07 pHash once (1074 → 1068 candidate pairs printed) | yes |
| P4 | C-08 MD5 lines into the 🟢 duplicates task | yes |
| P5 | C-16 helpers take their table as an argument | yes |
| P6 | C-20 vectorised `find_copies` | yes |
| P7 | C-23 Own-Data: drop old checks 1–3, keep split suggestion + report card | yes |
| P8 | N-21 `n_animals` → `n_objects` (her two sentences that name the column follow) | yes |
| P9 | N-22 `CLASSES` / `CLASS_COLORS` stay capitals ("decided once, at load") | keep |
| P10 | C-06 pHash-only match label | "recompressed" |
| P11 | T-19 heading levels per SN-011, MCQ gets a markdown heading, "Your Project Dataset" | yes |
| P12 | T-20 "(Optional)" | add to all 🔴 headings |
| P13 | T-21, T-22, T-23 (131 → plain markdown; 101 Pitfall stays a blockquote), T-24, T-25, T-26 | yes |
| P14 | T-16 CV diagram sentence matches the figure (gray = training, yellow = validation) | yes |
| P15 | T-28 one train/test colour pair; T-30 trail-camera colours not the park's | yes |
| P16 | T-13 "photos" → "images" in printed outputs | yes |
| P17 | T-12 one dataset name | **no** — only "play dataset" → "Play Dataset" (194) |
| P18 | T-29 one hash threshold | **no** |
| — | N-23 toy-horse constant names | no |
| — | T-17 💡 on every summary, T-18 title patterns | yes (C-12 does T-17) |

Earlier decisions (PRD Q1–Q11): CPU target; `COMPUTE_EMBEDDINGS = False`, DINOv2 only on a GPU or when set;
park DINOv2 vectors precomputed as `data/coco_animals_dinov2.npz`; the `%pip` line keeps only what Colab lacks
(`imagehash`; umap-learn, bokeh, transformers and torch are preinstalled, checked against Colab's pip freeze 2026-10-28).

### Run-audit items (R-01 … R-10) and where the two audits disagree

- R-01 vs C-03/C-24: `flag_odd_files` is **deleted** (C-03). Chapter 10's `build_my_sample` is replaced by the
  per-format park sample of the Own-Data loader, so `clean_images` is not rebuilt.
- R-02, R-03: apply (`THUMB_FOR_HASH = (256, 256)`; `load_thumb(..., (192, 192))` / `(256, 256)`), in the cells that
  move to the Own-Data Edition.
- R-05 = T-05, R-06 = T-01…T-07, R-07 = T-14: apply once (the cleanup audit's wording).
- R-08: the gate is `COMPUTE_EMBEDDINGS = False`, computing only on a GPU or when set; the precomputed file is
  `data/coco_animals_dinov2.npz` (ids + embeddings, matched by image id); `load_embeddings` is rewritten to read it
  (C-02's pattern) and print what happened; cells 96/97 fall back to "Color histograms" when the pretrained key is
  missing. No `%pip` for umap-learn/bokeh/transformers (preinstalled in Colab, CPU and GPU, checked 2026-10-28).
- R-09, R-10: inform the Own-Data loader; no change now.
- Runner: plain `nbclient` stalled three times on her notebook. If `just smoke` stalls, switch `tests/test_smoke.py`
  to a `jupyter_client` runner that waits for each execute reply (the audit's `build/run_audit2.py` works).

## Order

1. **Breakage** — C-24, N-01, N-13, C-15, and every R-item marked high.
2. **Dead and duplicated code** — C-01 … C-11, C-17 … C-21, C-14 (imports once; helpers cell after the data load).
3. **Names** — N-02 … N-20, N-21 (P8).
4. **Boxes** — C-12 (P1), T-17, T-18.
5. **Text** — T-01 … T-11, T-14 … T-16, T-19 … T-28 as decided.
6. **Embeddings gate** — C-22, precomputed park vectors.
7. **Editions** — E-map tags, Own-Data loader (section 01), own text variants, closing cell (PRD Q5), C-23.
8. **Tests** — content tests on her rules, Own-Data smoke per format, CPU timing report.

## Log

| Group | Commit | Notes |
|---|---|---|
| 1 · Breakage | Her copy, group 1 | C-24 (flag_odd_files + EXTREME_SIDE_PX deleted, `build_my_sample` samples `pool = images`), R-02 (`THUMB_FOR_HASH = (256, 256)` in Check 3), R-03 (two int sizes → tuples), N-01, N-13 (prefixes as in the audit; also `pages` → `size_pages`, `examples` → `fill_examples`, `bursts` → `burst_list`, 149's `output` → `overlap_output`), C-15 (`math` imported in setup). Smoke (project venv): runs to cell 94 and stops on `import torch`, which her copy needs and the venv does not have; that is fixed by group 6, not here. Checked instead with a jupyter_client run that continues past errors: the only errors are 94 and its cascade 95–98; on the audit venv (torch installed) 0 errors. |
| 2 · Dead and duplicated code | Her copy, group 2 | C-01 (answer_box deleted; `callout()` comes in group 4), C-02, C-03 (group 1), C-04, C-05, C-06 (P10: "recompressed"), C-07, C-08, C-09, C-10, C-11, C-14, C-16, C-17, C-18, C-19, C-20, C-21. Deviations: **N-16 and N-17 applied here**, because C-14 deletes the `_Image` and `patches` imports they rename. **C-19** became two helpers, `jpeg_bytes` (the file, used by the planted copy in 81) and `jpeg_copy` (the reopened image, 78 and 86): 81 writes the JPEG bytes to the card, so a reopened image would have had to be encoded a second time. **C-14**: the shared helpers cell is "Helpers: Images and Masks" (load_thumb, frame, to_rle, decode, encode); the colour constants sit in setup next to `INK` (C-18's wording); box cells keep their own `HTML` import until group 4 replaces them. **C-07**: `MAX_PAIR_BITS = 20` added to the helper cell here (C-07 uses it); `PAIR_THRESHOLD` follows in group 3. **C-16**: `class_shares(table, ids)` takes the id-indexed table, so it is called with `images_by_id` (C-09). **C-20**: Check 3 keeps its own `ia`, `ib`, `same class` columns on top of `find_copies`; `THUMB_FOR_HASH` (group 1) is gone again with it. Smoke: as group 1 (project venv stops at the torch cell 94); audit venv 0 errors, text outputs identical to group 1 except the moved MD5 lines. |
