"""Assemble sections/*.py (Jupytext percent format) into the single student notebook.

Agents and humans edit only the small section files; the .ipynb is a build artifact.
Usage: uv run python scripts/build.py            -> writes eda-teaching-notebook.ipynb (outputs stripped)

Callouts (SPEC L-001, RULES N-034): a markdown cell whose first line is
`<!-- callout: KIND | TITLE -->` becomes a collapsed code cell that redraws a coloured HTML box, with the
same HTML saved as its output so the box shows in Colab before anything runs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import jupytext
import markdown
import nbformat

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "sections"
NAME = "eda-teaching-notebook"
OUT = ROOT / f"{NAME}.ipynb"

# ---- callouts (SPEC L-001) ---------------------------------------------------------------------
# kind id -> (emoji, default title, background, text and left-bar colour)
CALLOUT_KINDS: dict[str, tuple[str, str, str, str]] = {
    "theory": ("📘", "Theory", "#E3F2FD", "#0D47A1"),
    "your-turn": ("✏️", "Your turn", "#FFF4D6", "#7A4A00"),
    "questions": ("❓", "Questions", "#E0F7F4", "#00564D"),
    "across-tasks": ("🔀", "Across tasks", "#EDE7F6", "#311B92"),
    "pitfall": ("⚠️", "Pitfall", "#FDECEA", "#8B1A10"),
    "recipe": ("🧾", "Recipe", "#E8F5E9", "#1B5E20"),
    "objective": ("🎯", "Objective", "#B9E3EA", "#08505C"),
    "takeaway": ("✅", "Takeaway", "#E2F0C4", "#3A5A0B"),
    "go-deeper": ("🔍", "Go deeper", "#F3F4F6", "#4B5563"),
    "answer": ("💡", "Worked answer", "#F6EFE4", "#5C4027"),
    "banner": ("", "", "#0B6E79", "#FFFFFF"),
    # Susanne's set (RULES SN-001…004, walkthrough 2026-10-23): one Task kind for questions and code work,
    # in the notebook's own teal so it cannot be mistaken for a 🟢🟡🔴 level dot (N-044).
    "task": ("✏️", "Task", "#DDEFF1", "#0B4F57"),
}
# Old kinds rendered as the new ones, so sections need no edit: questions and your-turn are Tasks (SN-002),
# the Recipe keeps its icon but takes the theory blue (SN-004). `objective` is not a box any more (SN-003):
# see OBJECTIVE_AS_MARKDOWN.
KIND_ALIASES = {"questions": "task", "your-turn": "task"}
RECIPE_IN_THEORY_BLUE = True
OBJECTIVE_AS_MARKDOWN = True
TASK_BAR = "#0B6E79"

# Body text size of every box. Susanne saw box text smaller than the markdown around it (S9, N-054).
# Colab's markdown size could not be confirmed from documentation: check by eye in Colab and adjust here.
BODY_PX = 16

# Chapters (S12): sections sharing one task get its icon before their heading. A proposal: Susanne picks.
CHAPTER_ICONS = {
    "classification": ("🏷️", ("01", "02", "03", "04", "05", "06")),
    "object detection": ("📦", ("07",)),
    "segmentation": ("🎭", ("08",)),
}
NUMBER_SECTIONS = True  # S13 / N-056: the build writes "N." from the file order; typed numbers are replaced
BANNER_BAR = "#06464D"  # darker teal left bar on the white-on-teal banner
# A marker line is exactly `<!-- callout: KIND -->` or `<!-- callout: KIND | TITLE -->`.
MARKER_RE = re.compile(r"^<!--\s*callout:\s*(?P<kind>[a-z][a-z-]*)\s*(?:\|\s*(?P<title>.*?)\s*)?-->\s*$")
MARKER_HINT = "<!-- callout"  # anything containing this must be a valid marker on the cell's first line

FONT = "'Google Sans',Roboto,'Helvetica Neue',Arial,sans-serif"
MONO = "'Roboto Mono',SFMono-Regular,Menlo,Consolas,monospace"
MD_EXTENSIONS = ["fenced_code", "tables", "sane_lists", "md_in_html"]
# Inline styles for tags inside the box (outputs allow inline CSS; markdown cells do not).
TAG_STYLES = {
    "p": "margin:6px 0 0 0;",
    "ul": "margin:6px 0 0 0;padding-left:22px;",
    "ol": "margin:6px 0 0 0;padding-left:22px;",
    "li": "margin:2px 0;",
    "a": "color:inherit;text-decoration:underline;",
    "code": f"font-family:{MONO};font-size:0.92em;background:rgba(0,0,0,0.07);padding:1px 4px;"
    "border-radius:3px;",
    "pre": f"font-family:{MONO};font-size:0.92em;background:rgba(0,0,0,0.07);padding:8px 10px;"
    "border-radius:4px;margin:6px 0 0 0;white-space:pre-wrap;",
    "blockquote": "margin:6px 0 0 0;padding-left:10px;border-left:3px solid currentColor;",
    "details": "margin:6px 0 0 0;",
    "summary": "cursor:pointer;font-weight:bold;",
    "table": "border-collapse:collapse;margin:6px 0 0 0;",
    "th": "border:1px solid rgba(0,0,0,0.2);padding:3px 8px;white-space:normal;text-align:left;",
    "td": "border:1px solid rgba(0,0,0,0.2);padding:3px 8px;white-space:normal;",
    "h1": "font-size:1.2em;margin:8px 0 0 0;",
    "h2": "font-size:1.15em;margin:8px 0 0 0;",
    "h3": "font-size:1.1em;margin:8px 0 0 0;",
    "h4": "font-size:1em;margin:8px 0 0 0;",
}
TAG_RE = re.compile(r"<(" + "|".join(TAG_STYLES) + r")(\s[^>]*)?>")


def parse_marker(first_line: str) -> tuple[str, str | None] | None:
    """(kind, title or None) if the line is a callout marker, else None. Does not check the kind."""
    m = MARKER_RE.match(first_line.strip())
    if not m:
        return None
    return m.group("kind"), (m.group("title") or None)


def _style_tags(body: str) -> str:
    def add(m: re.Match) -> str:
        tag, attrs = m.group(1), m.group(2) or ""
        if tag == "a" and not re.search(r"\btarget\s*=", attrs):
            attrs += ' target="_blank" rel="noopener"'  # links in an output frame open in a new tab
        if re.search(r"\bstyle\s*=", attrs):
            return f"<{tag}{attrs}>"
        return f'<{tag}{attrs} style="{TAG_STYLES[tag]}">'

    body = body.replace("<pre><code", '<pre><code style="background:none;padding:0;"')
    return TAG_RE.sub(add, body)


def _md_to_html(text: str) -> str:
    # md_in_html renders markdown inside <details> only with markdown="1"; add it so authors need not.
    text = re.sub(r"<details(?![^>]*\bmarkdown\s*=)", '<details markdown="1"', text)
    return markdown.markdown(text, extensions=MD_EXTENSIONS)


def callout_html(kind: str, title: str | None, body_md: str) -> str:
    """The self-contained HTML box for one callout (inline CSS only)."""
    if kind in KIND_ALIASES:
        kind = KIND_ALIASES[kind]
        if title:
            title = re.sub(r"^(Your turn|Questions?)\b", "Task", title)
    emoji, default_title, bg, fg = CALLOUT_KINDS[kind]
    if kind == "recipe" and RECIPE_IN_THEORY_BLUE:
        bg, fg = CALLOUT_KINDS["theory"][2:]
    title = title if title is not None else default_title
    title_html = re.sub(r"^<p>(.*)</p>$", r"\1", _md_to_html(title).strip(), flags=re.S) if title else ""
    title_html = _style_tags(title_html)
    body = _style_tags(_md_to_html(body_md).strip())
    if kind == "answer":  # N-024: a worked answer is a toggle, closed until the learner opens it
        summary = _style_tags(f"<summary>{emoji} {title_html}</summary>")
        body = f'<details style="margin:0;">{summary}\n{body}\n</details>'
        title_html = ""
    bar = BANNER_BAR if kind == "banner" else TASK_BAR if kind == "task" else fg
    box = (
        f"background:{bg};color:{fg};border-left:6px solid {bar};border-radius:6px;padding:12px 14px;"
        f"line-height:1.5;font-family:{FONT};font-size:{BODY_PX}px;width:100%;box-sizing:border-box;"
        "white-space:normal;overflow-wrap:break-word;margin:2px 0;"
    )
    if kind == "banner":
        head = f'<div style="font-size:22px;font-weight:bold;line-height:1.3;">{title_html}</div>'
    elif title_html:
        head = f'<div style="font-weight:bold;">{(emoji + " ") if emoji else ""}{title_html}</div>'
    else:
        head = ""
    return f'<div class="callout callout-{kind}" style="{box}">\n{head}\n{body}\n</div>'


def callout_cell(kind: str, title: str | None, body_md: str) -> nbformat.NotebookNode:
    html = callout_html(kind, title, body_md)
    safe = html.replace('"""', "&quot;&quot;&quot;")  # cannot close the raw string below
    cell = nbformat.v4.new_code_cell(
        f'# @title\nfrom IPython.display import HTML; HTML(r"""{safe}""")',
        outputs=[
            nbformat.v4.new_output(
                "display_data",
                data={"text/html": safe, "text/plain": f"<callout: {kind}>"},
                metadata={},
            )
        ],
    )
    cell.execution_count = None
    cell.metadata["cellView"] = "form"
    cell.metadata["callout"] = KIND_ALIASES.get(kind, kind)
    return cell


def convert_callout(cell: nbformat.NotebookNode, where: str) -> nbformat.NotebookNode:
    """Return the callout code cell for a marked markdown cell, or the cell unchanged."""
    if cell.cell_type != "markdown" or MARKER_HINT not in cell.source:
        return cell
    first, _, body = cell.source.lstrip("\n").partition("\n")
    parsed = parse_marker(first)
    if parsed is None or cell.source.count(MARKER_HINT) > 1:
        sys.exit(f"{where}: malformed callout marker, expected '<!-- callout: KIND | TITLE -->' as line 1")
    kind, title = parsed
    if kind == "objective" and OBJECTIVE_AS_MARKDOWN:  # SN-003: plain markdown under the heading
        return nbformat.v4.new_markdown_cell(f"**Objective:** {body.strip()}")
    if kind not in CALLOUT_KINDS and kind not in KIND_ALIASES:
        sys.exit(f"{where}: unknown callout kind '{kind}' (known: {', '.join(CALLOUT_KINDS)})")
    return callout_cell(kind, title, body)


# ---- assembly ----------------------------------------------------------------------------------


def section_files() -> list[Path]:
    files = sorted(p for p in SECTIONS.glob("[0-9][0-9]_*.py"))
    if not files:
        sys.exit("no sections/NN_*.py files found")
    return files


SECTION_HEADING_RE = re.compile(r"^## (?:[^\w\s]\S* )?(?:\d+\.\s+)?(?P<title>.+)$", re.M)  # icon, number


def section_heading(cell: nbformat.NotebookNode, number: int, stem: str) -> nbformat.NotebookNode:
    """The section's `## Title` with its number from the file order and its chapter icon (S12, S13)."""
    if not NUMBER_SECTIONS or cell.cell_type != "markdown":
        return cell
    m = SECTION_HEADING_RE.search(cell.source)
    if m is None:
        return cell
    icon = next((i for i, prefixes in CHAPTER_ICONS.values() if stem[:2] in prefixes), "")
    heading = f"## {icon + ' ' if icon else ''}{number}. {m.group('title')}"
    cell.source = cell.source[: m.start()] + heading + cell.source[m.end() :]
    return cell


def build_cell(cell: nbformat.NotebookNode, k: int, number: int, path: Path) -> nbformat.NotebookNode:
    """Cell k of section file `path` (the number-th section) as it appears in the notebook.
    Mutates `cell`; scripts/port.py passes a copy."""
    cell = convert_callout(cell, f"sections/{path.name} cell {k}")
    if k <= 1:  # section 0 opens with the notebook title (#), then its ## heading
        cell = section_heading(cell, number, path.stem)
    cell.id = f"{path.stem[:50]}-{k}"  # deterministic ids: same sections -> byte-identical .ipynb
    cell.metadata.pop("lines_to_next_cell", None)
    cell.metadata["section"] = path.stem
    # Colab collapses a code cell whose first line is "# @title ..." into a form.
    if cell.cell_type == "code" and cell.source.lstrip().startswith("# @title"):
        cell.metadata["cellView"] = "form"
    return cell


def build() -> Path:
    cells = []
    for number, path in enumerate(section_files()):
        nb = jupytext.read(path, fmt="py:percent")
        for k, cell in enumerate(nb.cells):
            cells.append(build_cell(cell, k, number, path))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    nbformat.validate(nb)
    nbformat.write(nb, OUT)
    return OUT


if __name__ == "__main__":
    out = build()
    print(f"built {out.name}: {len(section_files())} sections")
