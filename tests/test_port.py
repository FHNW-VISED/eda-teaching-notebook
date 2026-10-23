"""scripts/port.py brings a hand-edited copy (Colab download: no metadata, outputs) back into sections."""

from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import port  # noqa: E402

import build  # noqa: E402


def _built_notebook() -> nbformat.NotebookNode:
    cells = []
    for path in build.section_files():
        import jupytext

        for k, cell in enumerate(jupytext.read(path, fmt="py:percent").cells):
            cells.append(build.convert_callout(cell, f"{path.name} {k}"))
    return nbformat.v4.new_notebook(cells=cells)


def test_port_round_trip(tmp_path: Path) -> None:
    sections = tmp_path / "sections"
    shutil.copytree(ROOT / "sections", sections)
    nb = _built_notebook()
    her = copy.deepcopy(nb)

    # Colab-like: metadata gone, outputs on code cells
    for cell in her.cells:
        cell.metadata = {}
        if cell.cell_type == "code" and "callout" not in cell.source:
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="ran\n")]

    plain = next(i for i, c in enumerate(her.cells) if c.cell_type == "markdown" and "## 2." in c.source)
    her.cells[plain].source = her.cells[plain].source.replace("Become one with the data", "Meet the data")
    box = next(i for i, c in enumerate(her.cells) if 'class="callout callout-recipe"' in c.source)
    her.cells[box].source = her.cells[box].source.replace("This step is critical.", "This step is ESSENTIAL.")
    start = next(i for i, c in enumerate(her.cells) if c.cell_type == "markdown" and "## 3." in c.source)
    gone = next(i for i in range(start + 1, len(her.cells)) if her.cells[i].cell_type == "markdown")
    gone_text = her.cells[gone].source
    del her.cells[gone]
    her.cells.insert(plain + 1, nbformat.v4.new_markdown_cell("A sentence Susanne added."))

    her_path = tmp_path / "her.ipynb"
    nbformat.write(her, her_path)
    before = {p.name: p.read_text() for p in sections.glob("*.py")}
    assert port.main([str(her_path), "--sections", str(sections), "--report", str(tmp_path / "r.md")]) == 0

    s02 = (sections / "02_become_one.py").read_text()
    assert "## 2. Meet the data" in s02
    assert "A sentence Susanne added." in s02
    assert "This step is ESSENTIAL." in s02 and "<!-- callout: recipe | Recipe · Step 1" in s02
    assert gone_text.strip().splitlines()[-1] not in (sections / "03_image_inspection.py").read_text()
    for name in ("00_setup.py", "05_errors_cleaning.py", "11_wrap_up.py"):
        assert (sections / name).read_text() == before[name], f"{name} changed but was not edited"
    rep = (tmp_path / "r.md").read_text()
    assert "Round-trip check:** ✅" in rep, rep[:600]
    assert "callout-edited" in rep and "inserted" in rep and "deleted" in rep
