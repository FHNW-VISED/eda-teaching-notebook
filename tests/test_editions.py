"""Two Editions from one ordered list of sections (PRD 2026-10-28, Q4).

A cell without `edition` metadata belongs to both notebooks; `edition="teaching"` or `edition="own"` keeps it
in
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


def test_own_wording_swaps_apply_once_and_only_to_own(tmp_path):
    sections = tmp_path / "sections"
    sections.mkdir()
    (sections / "00_a.py").write_text(
        '# %% [markdown]\n# Count every animal.\n\n# %% edition="teaching"\nx = 1\n'
    )
    (sections / "own_wording.tsv").write_text("section\told\tnew\n00_a.py\tevery animal\tevery object\n")
    outs = build.build_all(sections=sections, out_dir=tmp_path)
    assert _sources(outs["teaching"])[0] == "Count every animal."
    assert _sources(outs["own"])[0] == "Count every object."


def test_own_wording_swap_that_no_longer_matches_stops_the_build(tmp_path):
    sections = tmp_path / "sections"
    sections.mkdir()
    (sections / "00_a.py").write_text("# %% [markdown]\n# Count every sheep.\n")
    (sections / "own_wording.tsv").write_text("section\told\tnew\n00_a.py\tevery animal\tevery object\n")
    with pytest.raises(SystemExit, match="own_wording"):
        build.build_all(sections=sections, out_dir=tmp_path)


def test_teaching_edition_has_no_skip_markers(tmp_path):
    """`# needs: boxes|masks` lines drive the Own-Data skip hook; the Teaching Edition never shows them."""
    sections = tmp_path / "sections"
    sections.mkdir()
    (sections / "00_a.py").write_text("# %%\n# @title Boxes\n# needs: boxes\nprint(1)\n")
    outs = build.build_all(sections=sections, out_dir=tmp_path)
    assert _sources(outs["teaching"]) == ["# @title Boxes\nprint(1)"]
    assert _sources(outs["own"]) == ["# @title Boxes\n# needs: boxes\nprint(1)"]
