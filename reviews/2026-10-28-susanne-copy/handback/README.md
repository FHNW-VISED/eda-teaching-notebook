# Hand-back to Susanne (2026-10-28)

`know-your-images-changes.pdf`: the biggest structural changes made on top of her copy, her version left, the new one
right, a red number on the exact place. The rest of the changes are listed in `../visible-changes.md`.

How it was made (re-run after changes): `run_full.py` executes an Edition on CPU with a `jupyter_client` runner
(plain nbclient stalls on this notebook with the inline backend) → `render.py` renders it the way Colab shows it
(form cells as their title, `\r` progress lines collapsed) → `shots.py spec.json` takes annotated screenshots
(Playwright; the Chromium path at the top of the file is machine-specific) → `make_pdf.py`.
