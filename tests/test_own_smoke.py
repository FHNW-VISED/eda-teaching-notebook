"""Smoke runs of the Own-Data Edition: one per label format, plus one broken label file (PRD 2026-10-28, Q10).

Each case sets the ✏️ settings cell by replacing its source lines, then runs the whole built notebook with
NB_SMOKE=1 through the runner of tests/test_smoke.py. With DATASET_PATH = None the notebook writes a small
park sample in the chosen format and reads it back with the reader a student's files would use.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import nbformat
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_smoke import ROOT, TOTAL_BUDGET_S, run_smoke  # noqa: E402

NOTEBOOK = ROOT / "eda-own-dataset-notebook.ipynb"
SAMPLES = ROOT / "data" / "cache" / "my_sample"
LOAD_CELL = "# @title Load Your Dataset (or the Park Sample)"

CASES = {  # id: (LABEL_FORMAT, TASKS)
    "folders": ("folders", ["classification"]),
    "csv": ("csv", ["classification"]),
    "coco": ("coco", ["classification", "detection", "segmentation"]),
    "yolo-boxes": ("yolo", ["classification", "detection"]),
    "yolo-polygons": ("yolo", ["classification", "detection", "segmentation"]),
    "voc": ("voc", ["classification", "detection"]),
    "png-masks": ("png-masks", ["classification", "detection", "segmentation"]),
}


def own_notebook(dataset_path: str | None, label_format: str, tasks: list[str]) -> nbformat.NotebookNode:
    """The built Own-Data Edition with its ✏️ settings cell set to these values."""
    assert NOTEBOOK.exists(), "run `just build` first"
    nb = nbformat.read(NOTEBOOK, as_version=4)
    settings = [c for c in nb.cells
                if c.cell_type == "code" and re.search(r"^DATASET_PATH = ", c.source, re.M)]
    assert len(settings) == 1, "expected exactly one ✏️ cell that sets DATASET_PATH"
    values = {"DATASET_PATH": repr(dataset_path), "TASKS": repr(tasks), "LABEL_FORMAT": repr(label_format)}
    for name, value in values.items():
        source = settings[0].source
        settings[0].source, n = re.subn(rf"^{name} = .*$", f"{name} = {value}", source, flags=re.M)
        assert n == 1, f"the ✏️ cell has no line '{name} = …'"
    return nb


@pytest.mark.parametrize("case", list(CASES))
def test_own_edition_runs_on_the_park_sample(case):
    label_format, tasks = CASES[case]
    nb = own_notebook(None, label_format, tasks)
    failure, _, elapsed = run_smoke(nb, ROOT / "build" / f"own-smoke-{case}.ipynb")
    if failure:
        pytest.fail(f"[{case}] {failure}", pytrace=False)
    assert elapsed < TOTAL_BUDGET_S, f"[{case}] smoke run took {elapsed:.0f}s > budget {TOTAL_BUDGET_S}s"
    summary = next(c for c in nb.cells if c.source.startswith(LOAD_CELL))
    text = "".join(o.get("text", "") for o in summary.outputs if o.output_type == "stream")
    assert f"format: {label_format}" in text, f"[{case}] the load cell did not read it as {label_format}"


def _broken_yolo(dest: Path) -> Path:
    """A copy of the YOLO park sample (or, if no case wrote it yet, two tiny images) with one label line
    of 4 numbers, the mistake a student makes when a column goes missing."""
    shutil.rmtree(dest, ignore_errors=True)
    if (SAMPLES / "yolo" / "labels").exists():
        shutil.copytree(SAMPLES / "yolo", dest)
    else:
        from PIL import Image

        (dest / "images").mkdir(parents=True)
        (dest / "labels").mkdir()
        for k in range(2):
            Image.new("RGB", (64, 48), (40 * k, 120, 80)).save(dest / "images" / f"img_{k}.jpg")
            (dest / "labels" / f"img_{k}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
        (dest / "data.yaml").write_text("names: [sheep]\n")
    label = sorted((dest / "labels").glob("*.txt"))[0]
    label.write_text(label.read_text() + "0 0.5 0.5 0.2\n")
    return dest


def test_own_edition_stops_on_a_broken_label_file(tmp_path):
    broken = _broken_yolo(tmp_path / "broken_yolo")
    nb = own_notebook(str(broken), "auto", ["classification", "detection"])
    failure, failing, _ = run_smoke(nb, ROOT / "build" / "own-smoke-broken.ipynb")
    assert failure is not None, "the notebook ran through a YOLO line with 4 numbers"
    cell = nb.cells[failing]
    assert cell.source.startswith(LOAD_CELL), f"stopped in the wrong cell: {cell.source.splitlines()[0]}"
    (error,) = [o for o in cell.outputs if o.output_type == "error"]
    assert error.ename == "SystemExit", f"expected SystemExit, got {error.ename}: {error.evalue}"
    message = error.evalue
    assert message.startswith("❌") and "line" in message and "4 numbers" in message, message
    trace = re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(error.traceback))
    assert "Traceback" not in trace and "File " not in trace, f"a traceback reached the student:\n{trace}"
