# Know Your Data: Exploring and Preparing Image Datasets at the Wildwood Farm Park

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FHNW-VISED/eda-teaching-notebook/blob/main/eda-teaching-notebook.ipynb)

Open the notebook in Google Colab with the badge above and run the first cell.

**Working locally instead.** You need Python 3.12 and git. Clone the repository, install the packages from
`requirements.txt` into a fresh virtual environment, and open the notebook in Jupyter:

```bash
git clone https://github.com/FHNW-VISED/eda-teaching-notebook.git
cd eda-teaching-notebook
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab eda-teaching-notebook.ipynb    # or eda-own-dataset-notebook.ipynb for your own data
```

`requirements.txt` pins the exact versions the notebook is tested with (the same as `uv.lock`), and it
includes Jupyter. The images (~115 MB) download into `data/` on the first run.

**Data and credits.** The Play Dataset is a slice of COCO 2017 (annotations CC BY 4.0; images keep their
Flickr licences and are downloaded at runtime) and the trail-camera part is a slice of Caltech Camera Traps
(CDLA-Permissive). Full attribution: [`data/ATTRIBUTION.md`](data/ATTRIBUTION.md).

---

**For maintainers.** `eda-teaching-notebook.ipynb` is built from `sections/` and is never edited directly.
`SPEC.md` describes what the notebook does, `RULES.md` the rules it follows, and `AGENTS.md` how to
work on it. `just check` is the gate. `scripts/night.sh` runs the open `night` issues unattended.
Every version of the notebook and its SPEC is kept in [`versions/`](versions/README.md); the newest folder is
always what is on `main`. Make the next one with `just version "what changed"` before merging.
