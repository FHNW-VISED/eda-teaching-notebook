"""Susanne's rules, checked on both built Editions (PRD 2026-10-28 Q10; her overlay SN-008, SN-010…SN-016).

These replace the v05 content tests, which described a notebook she rewrote. Run `just build` first.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parent.parent
TEACHING = ROOT / "eda-teaching-notebook.ipynb"
OWN = ROOT / "eda-own-dataset-notebook.ipynb"
EDITIONS = {"teaching": TEACHING, "own": OWN}

TASK_RE = re.compile(r"^(#{2,4}) ([🟢🟡🔴]) (.+)$")
SMALL_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "in",
    "on",
    "to",
    "for",
    "vs",
    "vs.",
    "per",
    "by",
    "with",
    "from",
    "at",
    "as",
    "is",
    "are",
    "be",
}
BOX_KINDS = {"answer", "theory", "recipe", "deeper"}  # SN-015
# Tasks she left without a Sample Answer on purpose (her copy, 2026-10-28).
NO_ANSWER_BY_DESIGN = (
    "Images: What Do Images Look Like Across Classes?",  # answered with "Document Your Observations"
    "Splitting for Object Detection and Segmentation",  # the multiple-choice question checks itself
    "Exploring Bounding Boxes: What Do You Notice?",
)
# CAPITALS name settings decided once (SN-008, N-058). CLASSES and
# CLASS_COLORS are decided once at load, in setup or by the Own-Data loader (Simone, 2026-10-28, P9).
CAPS_EXCEPTIONS: set[str] = set()  # none: results is lowercase too (Simone, 2026-10-29)
SET_AT_LOAD = {"CLASSES", "CLASS_COLORS"}


def cells(edition: str) -> list:
    path = EDITIONS[edition]
    assert path.exists(), "run `just build` first"
    return nbformat.read(path, as_version=4).cells


def task_headings(edition: str) -> list[tuple[int, str, str, str]]:
    found = []
    for i, cell in enumerate(cells(edition)):
        if cell.cell_type == "markdown":
            for line in cell.source.splitlines():
                m = TASK_RE.match(line.strip())
                if m:
                    found.append((i, m[1], m[2], m[3]))
    return found


def is_sample_answer(cell) -> bool:
    return cell.cell_type == "code" and "💡 Sample Answer" in cell.source.lstrip().split("\n", 1)[0]


@pytest.mark.parametrize("edition", EDITIONS)
def test_task_headings_are_title_case_and_mark_optional(edition):
    for _, _, dot, title in task_headings(edition):
        lower = [
            w
            for w in re.findall(r"[A-Za-z][\w'-]*", title)
            if w.lower() not in SMALL_WORDS and w[0].islower() and not any(ch.isupper() for ch in w[1:])
        ]
        assert not lower, f"SN-012/SN-016: '{title}' is not Title Case ({', '.join(lower)})"
        if dot == "🔴":
            assert "(Optional)" in title, f"P12: 🔴 '{title}' needs '(Optional)'"


@pytest.mark.parametrize("edition", EDITIONS)
def test_no_question_or_section_numbers(edition):
    for i, cell in enumerate(cells(edition)):
        assert not re.search(r"\*\*Q\d+\b|\bQ\d+\.", cell.source), f"cell {i}: a Q-number (SN-012)"
        assert not re.search(r"\b[Ss]ection \d", cell.source), f"cell {i}: a section number (T-14)"


@pytest.mark.parametrize("edition", EDITIONS)
def test_boxes_are_her_four_kinds(edition):
    for i, cell in enumerate(cells(edition)):
        for kind in re.findall(r'\bcallout\(\s*"(\w+)"', cell.source):
            assert kind in BOX_KINDS, f"cell {i}: box kind '{kind}' is not one of hers (SN-015)"


def test_teaching_tasks_have_sample_answers():
    heads = task_headings("teaching")
    nb = cells("teaching")
    bounds = [i for i, *_ in heads] + [len(nb)]
    for (start, _, _, title), end in zip(heads, bounds[1:], strict=True):
        if title.startswith(NO_ANSWER_BY_DESIGN):
            continue
        assert any(is_sample_answer(c) for c in nb[start:end]), f"Task '{title}' has no 💡 Sample Answer"


def test_own_edition_has_no_sample_answers_and_no_park_hunts():
    for i, cell in enumerate(cells("own")):
        assert not is_sample_answer(cell), f"own cell {i} is a Sample Answer (PRD Q2)"
        for planted in ("mask_error_key", "duplicate_key", "missing_box_key", "cct_rows"):
            assert planted not in cell.source, f"own cell {i} uses the park-only '{planted}'"


@pytest.mark.parametrize("edition", EDITIONS)
def test_capitals_only_for_settings(edition):
    assigned = Counter()
    for i, cell in enumerate(cells(edition)):
        if cell.cell_type != "code":
            continue
        source = "\n".join(
            line for line in cell.source.splitlines() if not line.lstrip().startswith(("%", "!"))
        )
        for node in ast.parse(source).body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for name in (n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)):
                if not re.fullmatch(r"[A-Z][A-Z0-9_]+", name) or name in CAPS_EXCEPTIONS:
                    continue
                assigned[name] += 1
                computed = isinstance(node.value, ast.Call) and not (
                    isinstance(node.value.func, ast.Name)
                    and node.value.func.id in {"Path", "dict", "set", "tuple"}
                )
                assert not computed or name in SET_AT_LOAD, (
                    f"cell {i}: {name} is computed; use lowercase (SN-008)"
                )
    limit = {name: 2 if edition == "own" else 1 for name in SET_AT_LOAD}
    again = {n: k for n, k in assigned.items() if k > limit.get(n, 1)}
    assert not again, f"settings assigned more than once (N-058): {again}"
