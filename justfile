# One entry point for humans, agents and CI. `just check` is the acceptance gate.

# Assemble sections/*.py into eda-teaching-notebook.ipynb
build:
    uv run python scripts/build.py

# Rule checks that a script can decide + ruff on the section sources
lint:
    uv run python scripts/lint_notebook.py
    uv run ruff check --quiet sections scripts tests
    uv run pytest -q -p no:warnings tests/test_version.py tests/test_port.py

# Execute the whole notebook headless on tiny data (NB_SMOKE=1)
smoke: build
    uv run pytest -q -x -p no:warnings tests/test_smoke.py tests/test_content.py tests/test_cuts.py tests/test_my_dataset.py

# Lint + smoke: what the night run and CI require before a commit counts
check: lint smoke

# Every function/class defined anywhere in the notebook, with its file:line — read this instead of all sections
api:
    @grep -nE "^(def|class) |^    def " sections/*.py || true

# Open the notebook in Jupyter with sections paired (edit in the browser, sync back with `just sync`)
open: build
    uv run jupyter lab eda-teaching-notebook.ipynb

# After editing eda-teaching-notebook.ipynb by hand in Jupyter/Colab: split it back into sections (review the diff!)
sync:
    uv run python scripts/split.py

# Bring a hand-edited copy (e.g. a Colab download) back into sections/: aligns by content, restores callouts,
# writes reviews/port-<date>.md. Run it on a branch from the commit the copy was made from.
port notebook:
    uv run python scripts/port.py "{{notebook}}"

# Committed notebook must equal a fresh build (catches hand edits that skipped `just sync`)
drift: build
    git diff --exit-code -- eda-teaching-notebook.ipynb

# Compare runtime deps with what Google Colab preinstalls (gpu runtime by default)
colab-pins runtime="gpu":
    uv run python scripts/colab_pins.py {{runtime}}

# Snapshot SPEC.md + the built notebook as the next version in versions/ (do this before merging to main)
version label: build
    uv run python scripts/version.py new "{{label}}"

# The newest version in versions/ must equal SPEC.md and the built notebook (true on main, always)
version-check: build
    uv run python scripts/version.py check
