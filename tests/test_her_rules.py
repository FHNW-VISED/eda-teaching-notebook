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


def code_trees(edition: str):
    """(index, cell, ast) of every code cell, with magic and shell lines blanked so the code parses."""
    for i, cell in enumerate(cells(edition)):
        if cell.cell_type != "code":
            continue
        source = "\n".join(
            "" if line.lstrip().startswith(("%", "!")) else line for line in cell.source.splitlines()
        )
        yield i, cell, ast.parse(source)


CAPITAL_RE = re.compile(r"[A-Z][A-Z0-9_]*")
MUTATORS = {"append", "extend", "insert", "update", "setdefault", "pop", "popitem", "remove", "clear", "add",
            "discard", "sort", "reverse"}  # fmt: skip


def _capital(node) -> str | None:
    return node.id if isinstance(node, ast.Name) and CAPITAL_RE.fullmatch(node.id) else None


@pytest.mark.parametrize("edition", EDITIONS)
def test_capitals_never_changed_in_place(edition):
    """N-058 (fix list row 23): a setting is never changed in place: no X.append/.update/.extend…,
    no X[k] = …, no X.attr = …, no X |= … (or any augmented assignment)."""
    found = []
    for i, _, tree in code_trees(edition):
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in MUTATORS and _capital(node.func.value)):
                found.append(f"cell {i}: {_capital(node.func.value)}.{node.func.attr}(…)")
            if isinstance(node, ast.Assign):
                targets = node.targets
            else:
                targets = [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
            for t in targets:
                if isinstance(t, (ast.Subscript, ast.Attribute)) and _capital(t.value):
                    found.append(f"cell {i}: {_capital(t.value)}[…] or .attr assigned")
                if isinstance(node, ast.AugAssign) and _capital(t):
                    found.append(f"cell {i}: {_capital(t)} changed by an augmented assignment")
    assert not found, "settings changed in place (N-058): " + "; ".join(found)


def learner_settings(edition: str) -> dict[str, str]:
    """{name: section} of every setting a learner changes: a `# @param` line or a ✏️ YOUR TURN cell."""
    settings = {}
    for _, cell, tree in code_trees(edition):
        section = cell.metadata["section"]
        for line in cell.source.splitlines():
            m = re.match(r"^\s*([A-Z][A-Z0-9_]*)\s*=.*#\s*@param\b", line)
            if m:
                settings[m[1]] = section
        if "✏️ YOUR TURN" in cell.source:
            for node in tree.body:
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    settings.update({t.id: section for t in targets if _capital(t)})
    return settings


@pytest.mark.parametrize("edition", EDITIONS)
def test_learner_settings_are_read_only_in_their_own_section(edition):
    """N-058 (fix list row 24): a later section that needs the same kind of value gets its own setting, so a
    change in one section never silently changes another (the case that started it: HASH_THRESHOLD)."""
    settings = learner_settings(edition)
    assert settings.get("HASH_THRESHOLD") == "04_errors_cleaning", "HASH_THRESHOLD should be set in section 4"
    found = []
    for i, cell, tree in code_trees(edition):
        read = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        for name in sorted(read & set(settings)):
            if settings[name] != cell.metadata["section"]:
                found.append(f"cell {i} ({cell.metadata['section']}) reads {name} from {settings[name]}")
    assert not found, "; ".join(found)


def test_own_edition_has_no_coding_exercises():
    """Her request for the Own-Data Edition, 28 Oct: "without the exercises" (fix list rows 1, 24)."""
    for i, cell in enumerate(cells("own")):
        assert "✍️ YOUR CODE" not in cell.source, f"own cell {i} is a ✍️ coding exercise"
        assert "Coding Exercise" not in cell.source, f"own cell {i} still says 'Coding Exercise'"
        assert "_ref_" not in cell.source, f"own cell {i} holds a reference solution of an exercise"


@pytest.mark.parametrize("edition", EDITIONS)
def test_no_capitals_inside_functions_or_as_parameters(edition):
    """N-058 / PEP 8 (fix list row 23): a CAPITAL name is a module-level setting, so a function never assigns
    one (locals are lowercase) and never takes one as a parameter (not even scikit-learn's X)."""
    found = []
    for i, _, tree in code_trees(edition):
        kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        for fn in (n for n in ast.walk(tree) if isinstance(n, kinds)):
            args = fn.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
                if arg is not None and CAPITAL_RE.fullmatch(arg.arg):
                    found.append(f"cell {i}: parameter {arg.arg}")
            body = fn.body if isinstance(fn.body, list) else [fn.body]
            for node in (n for stmt in body for n in ast.walk(stmt)):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and _capital(node):
                    found.append(f"cell {i}: {node.id} assigned inside {getattr(fn, 'name', 'a lambda')}()")
    assert not found, "; ".join(found)


def _shows_box(cell) -> bool:
    """True if the cell's last line shows one box drawn from literals only: callout("kind", "title", body)
    or HTML(body) with literal strings. Boxes with a figure (fig=) or values from the kernel can only be drawn
    at run time."""
    try:
        last = ast.parse(cell.source).body[-1]
    except (SyntaxError, IndexError):
        return False
    if not (isinstance(last, ast.Expr) and isinstance(last.value, ast.Call)
            and isinstance(last.value.func, ast.Name) and last.value.func.id in {"callout", "HTML"}):
        return False
    if last.value.keywords:
        return False
    try:
        args = [ast.literal_eval(a) for a in last.value.args]
    except ValueError:
        return False
    return any(isinstance(a, str) and "<" in a for a in args)


@pytest.mark.parametrize("edition", EDITIONS)
def test_boxes_show_before_anything_runs(edition):
    """N-034 (fix list rows 17, 24): every cell that shows a box has its HTML saved as output in the built
    notebook, so Colab shows the box (Learning Objectives, How to Use, every callout) before the cell runs."""
    missing = [f"cell {i} ({c.source.lstrip().splitlines()[0][:60]})" for i, c in enumerate(cells(edition))
               if c.cell_type == "code" and _shows_box(c) and not c.get("outputs")]
    assert not missing, "boxes without a saved output: " + "; ".join(missing)
