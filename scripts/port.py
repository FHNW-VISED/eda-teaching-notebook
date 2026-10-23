"""Bring a hand-edited copy of the notebook (e.g. Susanne's Colab download) back into sections/*.py.

split.py trusts the `section` cell metadata and writes callout boxes back as raw HTML code cells. A copy
edited in Colab may have lost that metadata, and its callouts are HTML. This script instead:

1. builds the current sections in memory (the base the copy was edited from: run it on a branch from the
   commit she received, e.g. main at v04),
2. aligns her cells with the base cells by content (difflib), so no metadata is needed,
3. turns every callout back into `<!-- callout: KIND | TITLE -->` markdown: unchanged boxes get their
   original source back exactly; edited boxes are converted from HTML (flagged ⚠️ in the report),
4. writes each cell into the section of the base cell it replaces (new cells join the section above),
5. writes a report of every change, section by section, and re-checks the result against her copy.

Usage: uv run python scripts/port.py HER.ipynb [--report reviews/port-YYYY-MM-DD.md] [--dry-run]
Then: review `git diff sections/` and the report, run `just check`.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import difflib
import html as htmllib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import jupytext
import nbformat
from markdownify import markdownify

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import CALLOUT_KINDS, build_cell  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CALLOUT_SRC_RE = re.compile(r'HTML\(r"""(?P<html>.*)"""\)\s*$', re.S)
CALLOUT_CLASS_RE = re.compile(r'class="callout callout-(?P<kind>[a-z-]+)"')


@dataclass
class Cell:
    section: str | None  # section stem (base cells) or None (her cells)
    kind: str  # "markdown" | "code" | "callout"
    source: str  # section-file source: markdown/code text, or the full marker markdown for a callout
    norm: str  # what alignment compares
    html: str = ""  # callout HTML as built / as found in her copy
    orig: object = None  # base only: the jupytext cell as read (reused, so unchanged files stay identical)


def _clean(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip("\n")


def _visible_text(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", htmllib.unescape(text)).strip()


# ---- base: the current sections, built in memory -----------------------------------------------


def base_cells(sections_dir: Path) -> list[Cell]:
    """Every cell as the build renders it (to compare), with the section source it came from (to keep)."""
    cells = []
    for number, path in enumerate(sorted(sections_dir.glob("[0-9][0-9]_*.py"))):
        nb = jupytext.read(path, fmt="py:percent")
        for k, src_cell in enumerate(nb.cells):
            built = build_cell(copy.deepcopy(src_cell), k, number, path)
            text = _clean(src_cell.source)
            if built.metadata.get("callout"):
                html = _callout_html_of(built.source)
                cells.append(Cell(path.stem, "callout", text, "CALLOUT " + html, html, src_cell))
            else:
                norm = f"{built.cell_type} {_clean(built.source)}"
                cells.append(Cell(path.stem, src_cell.cell_type, text, norm, "", src_cell))
    return cells


def _callout_html_of(code: str) -> str:
    m = CALLOUT_SRC_RE.search(code)
    return m.group("html").strip() if m else ""


# ---- her copy -----------------------------------------------------------------------------------


def her_cells(path: Path) -> list[Cell]:
    nb = nbformat.read(path, as_version=4)
    cells = []
    for cell in nb.cells:
        text = _clean(cell.source)
        if cell.cell_type == "code" and CALLOUT_CLASS_RE.search(text) and CALLOUT_SRC_RE.search(text):
            html = _callout_html_of(text)
            cells.append(Cell(None, "callout", "", "CALLOUT " + html, html))
        elif cell.cell_type in ("markdown", "code"):
            if cell.cell_type == "markdown" and cell.get("attachments"):
                print(f"⚠️ a markdown cell has pasted attachments (images); they are dropped: {text[:60]!r}")
            cells.append(Cell(None, cell.cell_type, text, f"{cell.cell_type} {text}"))
    return cells


def callout_to_markdown(html: str) -> str:
    """Rebuild `<!-- callout: KIND | TITLE -->` + markdown from a callout box's HTML (edited by hand)."""
    m = CALLOUT_CLASS_RE.search(html)
    kind = m.group("kind") if m else "theory"
    emoji = CALLOUT_KINDS.get(kind, ("",))[0]
    inner = re.sub(r"^\s*<div[^>]*>", "", html.strip(), count=1)
    inner = re.sub(r"</div>\s*$", "", inner)
    title = None
    if kind == "answer":
        s = re.search(r"<summary[^>]*>(?P<t>.*?)</summary>", inner, re.S)
        if s:
            title = s.group("t")
            inner = inner.replace(s.group(0), "", 1)
        inner = re.sub(r"^\s*<details[^>]*>", "", inner.strip())
        inner = re.sub(r"</details>\s*$", "", inner)
    else:
        h = re.match(r"\s*<div[^>]*>(?P<t>.*?)</div>", inner, re.S)
        if h:
            title = h.group("t")
            inner = inner[h.end() :]
    if title is not None:
        title = markdownify(title, strip=["p"]).strip()
        if emoji and title.startswith(emoji):
            title = title[len(emoji) :].strip()
    body = markdownify(inner, heading_style="ATX", bullets="-", escape_underscores=False, escape_misc=False)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    marker = f"<!-- callout: {kind} | {title} -->" if title else f"<!-- callout: {kind} -->"
    return f"{marker}\n{body}"


# ---- alignment ----------------------------------------------------------------------------------


@dataclass
class Change:
    op: str  # "modified" | "inserted" | "deleted" | "callout-edited"
    section: str
    before: str
    after: str


def _new_cell(cell_type: str, src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(src) if cell_type == "markdown" else nbformat.v4.new_code_cell(src)


def port(her: list[Cell], base: list[Cell]) -> tuple[dict[str, list], list[Change]]:
    """New cells per section (unchanged cells are the original objects), and the list of changes."""
    sections: dict[str, list] = {c.section: [] for c in base}
    changes: list[Change] = []
    sm = difflib.SequenceMatcher(None, [c.norm for c in base], [c.norm for c in her], autojunk=False)
    last_section = base[0].section

    def emit(section: str, cell: Cell, base_cell: Cell | None) -> None:
        if cell.kind == "callout":
            if base_cell is not None and base_cell.kind == "callout" and base_cell.html == cell.html:
                sections[section].append(base_cell.orig)
                return
            src = callout_to_markdown(cell.html)
            changes.append(Change("callout-edited", section, base_cell.source if base_cell else "", src))
            sections[section].append(_new_cell("markdown", src))
        else:
            sections[section].append(_new_cell(cell.kind, cell.source))

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for b in base[i1:i2]:
                sections[b.section].append(b.orig)
                last_section = b.section
            continue
        olds, news = base[i1:i2], her[j1:j2]
        for k in range(max(len(olds), len(news))):
            b = olds[k] if k < len(olds) else None
            h = news[k] if k < len(news) else None
            section = b.section if b else last_section
            if b and h:
                emit(section, h, b)
                if h.kind != "callout":
                    changes.append(Change("modified", section, b.source, h.source))
            elif h:
                emit(section, h, None)
                if h.kind != "callout":
                    changes.append(Change("inserted", section, "", h.source))
            else:
                changes.append(Change("deleted", section, b.source, ""))
            last_section = section
    return {s: cells for s, cells in sections.items()}, changes


def write_sections(new: dict[str, list], changed: set[str], sections_dir: Path) -> None:
    """Rewrite only the sections that changed; the others stay byte-identical."""
    for stem, cells in new.items():
        if stem not in changed:
            continue
        nb = nbformat.v4.new_notebook(cells=cells)
        nb.metadata["jupytext"] = {"notebook_metadata_filter": "-all"}
        jupytext.write(nb, sections_dir / f"{stem}.py", fmt="py:percent")


def self_check(her: list[Cell], sections_dir: Path) -> list[str]:
    """Rebuild from the ported sections and compare with her copy; callouts are compared by visible text."""
    rebuilt = base_cells(sections_dir)
    problems = []
    if len(rebuilt) != len(her):
        problems.append(f"cell count differs: rebuilt {len(rebuilt)} vs her copy {len(her)}")
    for n, (r, h) in enumerate(zip(rebuilt, her, strict=False)):
        if h.kind == "callout":
            ratio = difflib.SequenceMatcher(None, _visible_text(r.html), _visible_text(h.html)).ratio()
            if r.kind != "callout" or ratio < 0.97:
                problems.append(f"cell {n} ({r.section}): callout text matches only {ratio:.0%}")
        elif r.norm != h.norm:
            problems.append(f"cell {n} ({r.section}): {h.kind} cell differs after the round trip")
    return problems


def report(changes: list[Change], problems: list[str], her_path: Path) -> str:
    out = [f"# Port of `{her_path.name}` — {datetime.date.today()}", ""]
    ops = ("modified", "inserted", "deleted", "callout-edited")
    counts = {op: sum(c.op == op for c in changes) for op in ops}
    out += [", ".join(f"{v} {k}" for k, v in counts.items()), ""]
    verdict = "✅ the ported sections rebuild her copy." if not problems else "⚠️ see below"
    out += ["**Round-trip check:** " + verdict, ""]
    out += [f"- {p}" for p in problems]
    for section in dict.fromkeys(c.section for c in changes):
        out += ["", f"## {section}", ""]
        for c in (c for c in changes if c.section == section):
            flag = " ⚠️ converted from HTML, check the markdown" if c.op == "callout-edited" else ""
            out += [f"**{c.op}**{flag}", "", "```diff"]
            diff = difflib.unified_diff(c.before.splitlines(), c.after.splitlines(), lineterm="", n=1)
            out += list(diff)[2:] or ["(no text change)"]
            out += ["```", ""]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--sections", type=Path, default=ROOT / "sections")
    ap.add_argument("--report", type=Path, default=ROOT / "reviews" / f"port-{datetime.date.today()}.md")
    ap.add_argument("--dry-run", action="store_true", help="report only, do not touch sections/")
    args = ap.parse_args(argv)
    base, her = base_cells(args.sections), her_cells(args.notebook)
    new, changes = port(her, base)
    problems: list[str] = []
    if not args.dry_run:
        write_sections(new, {c.section for c in changes}, args.sections)
        problems = self_check(her, args.sections)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report(changes, problems, args.notebook))
    print(f"{len(changes)} changes → {args.report}" + (" (dry run)" if args.dry_run else ""))
    for p in problems:
        print("⚠️", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
