"""Two Editions from one ordered list of sections (PRD 2026-10-28, Q4).

A cell without `edition` metadata belongs to both notebooks; `edition="teaching"` or `edition="own"` keeps it in
one. `just build` writes both, in the same order as the sections.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("build", ROOT / "scripts" / "build.py")
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)

SECTION = """# %% [markdown]
# # Shared title

# %% edition="teaching"
x = "park"

# %% edition="own"
x = "yours"

# %%
print(x)
"""


def _sources(path: Path) -> list[str]:
    return [c.source for c in nbformat.read(path, as_version=4).cells]


def test_build_writes_both_editions_in_order(tmp_path):
    sections = tmp_path / "sections"
    sections.mkdir()
    (sections / "00_a.py").write_text(SECTION)
    outs = build.build_all(sections=sections, out_dir=tmp_path)
    assert set(outs) == set(build.EDITIONS)
    assert _sources(outs["teaching"]) == ["# Shared title", 'x = "park"', "print(x)"]
    assert _sources(outs["own"]) == ["# Shared title", 'x = "yours"', "print(x)"]


def test_unknown_edition_stops_the_build(tmp_path):
    sections = tmp_path / "sections"
    sections.mkdir()
    (sections / "00_a.py").write_text('# %% edition="both"\nx = 1\n')
    with pytest.raises(SystemExit, match="unknown edition"):
        build.build_all(sections=sections, out_dir=tmp_path)


def test_repo_build_has_both_notebooks():
    for name in build.EDITIONS.values():
        assert (ROOT / name).exists(), f"run `just build`: {name} is missing"
