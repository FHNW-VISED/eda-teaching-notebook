# Data attribution and licences

The notebook itself carries no licence text (rule N-045); the credits live here and in `README.md`.

## Play Dataset: COCO 2017 animals (`data/coco_animals_subset.json`)

- **Source:** [COCO 2017](https://cocodataset.org/), val2017 split, images with exactly one animal class among
  cat, dog, bird, horse, sheep, zebra. Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014.
- **Annotations:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), © COCO Consortium. We redistribute
  only a subset of the annotation file (`data/coco_animals_subset.json`, ≤ 2 MB), keeping each image's
  `license` id and `flickr_url`.
- **Images:** every image keeps its original Flickr licence (about 34 % are NoDerivatives). We do **not**
  redistribute images; the notebook downloads them from `images.cocodataset.org` at runtime. The COCO
  licence ids are listed in the subset file under `licenses`.
- **Planted copies and defects** (flipped, rotated, resized, recompressed, grayscale, wrong labels, ugly masks)
  are generated inside the notebook at runtime and never redistributed.
- Known-issue reference: Singh et al., *Benchmarking Object Detectors with COCO: A New Path Forward* (COCO-ReM),
  ECCV 2024.

## Woodland trail cameras: Caltech Camera Traps CCT20 slice (`data/cct20_slice.csv`)

- **Source:** [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps), CCT20 subset.
  Beery, Van Horn, Perona, *Recognition in Terra Incognita*, ECCV 2018.
- **Licence:** [Community Data License Agreement (Permissive, v1.0)](https://cdla.io/permissive-1-0/).
- We commit only a CSV of file names, labels, locations, sequence ids and timestamps. Images are fetched from
  LILA's public HTTPS bucket at runtime and resized to 512 px.
- Related: Marco Willi's classroom subset `marco-willi/camera-trap-cct20` on Hugging Face (FHNW).

## Quotes

- Andrej Karpathy, [*A Recipe for Training Neural Networks*](https://karpathy.github.io/2019/04/25/recipe/)
  (2019): 2–3 sentence verbatim quotes in sections 2–5, linked.
