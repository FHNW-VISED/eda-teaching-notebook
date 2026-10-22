"""Cut-pass tests (issue #18, decisions of 2026-10-22): pin the decisions so a later pass cannot undo them.

Each test names the decision-record row it protects (reviews/2026-10-22-cut-decisions.md). They test what a
student SEES (headings, the 🔴 Optional parts, ✏️ cells, question text, printed output, figures) and never how
a cell is written. Question numbers are renumbered later, so questions are found by their wording, never by
number. The notebook is read the way tests/test_content.py reads it: build/smoke-executed.ipynb (run
`just smoke`), cell by cell, with each cell's section in metadata["section"].
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parent.parent
EXECUTED = ROOT / "build" / "smoke-executed.ipynb"
FACILITATION = ROOT / "FACILITATION.md"

# A question marker in a markdown cell (**Q7.**) or rendered inside a callout (<strong>Q7.</strong>).
Q_RE = re.compile(r"(?:\*\*|<strong>)Q(\d+)\.?(?:\*\*|</strong>)")
HEAD_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
TAG_RE = re.compile(r"<[^>]+>")


# ---- reading the notebook ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nb():
    if not EXECUTED.exists():
        pytest.skip("no executed notebook: run `just smoke`")
    return nbformat.read(EXECUTED, as_version=4)


def section(nb, n: int) -> list:
    return [c for c in nb.cells if c.metadata.get("section", "").startswith(f"{n:02d}_")]


def is_callout(c) -> bool:
    return c.cell_type == "code" and bool(c.metadata.get("callout"))


def is_prose(c) -> bool:
    """A cell the student reads as text: markdown or a callout box."""
    return c.cell_type == "markdown" or is_callout(c)


def plain(c) -> str:
    """The text of a cell without HTML tags (callout sources are HTML)."""
    return html.unescape(TAG_RE.sub("", c.source))


def prose(cells) -> str:
    return "\n".join(plain(c) for c in cells if is_prose(c))


def outputs_text(c) -> str:
    """Everything the cell printed or displayed as text."""
    parts = []
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            parts.append(o.get("text", ""))
        else:
            data = o.get("data", {})
            parts += [str(data.get("text/plain", "")), str(data.get("text/html", ""))]
    return "\n".join(parts)


def image_outputs(c) -> list[tuple[int, int]]:
    """(width, height) in pixels of every figure the cell displayed. Under the smoke run's Agg backend a
    figure shows up as the text '<Figure size WxH with N Axes>' (no PNG); plt.show() leaves no output."""
    sizes = []
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            continue
        m = re.search(r"<Figure size (\d+)x(\d+)", str(o.get("data", {}).get("text/plain", "")))
        if m:
            sizes.append((int(m.group(1)), int(m.group(2))))
    return sizes


def headings(cells) -> list[tuple[int, int, str]]:
    """(cell index, level, text) of every markdown heading (code fences skipped)."""
    found = []
    for i, c in enumerate(cells):
        if c.cell_type != "markdown" or is_callout(c):
            continue
        fence = False
        for line in c.source.splitlines():
            if line.lstrip().startswith("```"):
                fence = not fence
            elif not fence and (m := HEAD_RE.match(line)):
                found.append((i, len(m.group(1)), m.group(2)))
    return found


def find_heading(cells, pattern: str) -> int | None:
    for i, _, text in headings(cells):
        if re.search(pattern, text, re.I):
            return i
    return None


def optional_flags(cells) -> list[bool]:
    """True for every cell inside a `### 🔴 Optional …` part: from that heading to the next heading of the
    same or a higher level."""
    flags, state, level = [], False, 0
    heads = {}
    for i, lvl, text in headings(cells):
        heads.setdefault(i, []).append((lvl, text))
    for i, _ in enumerate(cells):
        for lvl, text in heads.get(i, []):
            if state and lvl <= level:
                state = False
            if "🔴" in text and "optional" in text.lower():
                state, level = True, lvl
        flags.append(state)
    return flags


def question_paragraphs(cells) -> list[tuple[int, str]]:
    """(cell index, text) for every question: the paragraph with the marker, plus the bullet or numbered
    list right after it (a question may be asked as "Two guesses:" followed by bullets)."""
    found = []
    for i, c in enumerate(cells):
        if not is_prose(c):
            continue
        paras = re.split(r"\n\s*\n", c.source)
        for k, para in enumerate(paras):
            if not Q_RE.search(para):
                continue
            text = para
            for nxt in paras[k + 1 :]:
                if not re.match(r"\s*(?:[-*]|\d+\.)\s", nxt) or Q_RE.search(nxt):
                    break
                text += "\n" + nxt
            found.append((i, html.unescape(TAG_RE.sub("", text))))
    return found


def questions_matching(cells, pattern: str) -> list[tuple[int, str]]:
    return [(i, p) for i, p in question_paragraphs(cells) if re.search(pattern, p, re.I | re.S)]


def cells_matching(cells, pattern: str, *, kinds=("prose", "code")) -> list[int]:
    """Indices of cells whose visible text (prose) or source (code) matches the pattern."""
    hits = []
    for i, c in enumerate(cells):
        text = plain(c) if is_prose(c) else c.source
        is_text = is_prose(c)
        if ((is_text and "prose" in kinds) or (not is_text and "code" in kinds)) and re.search(
            pattern, text, re.I
        ):
            hits.append(i)
    return hits


def your_turn_cells(cells) -> list[int]:
    return [i for i, c in enumerate(cells) if c.cell_type == "code" and c.source.lstrip().startswith("# ✏️")]


def facilitation_row(prefix: str) -> str:
    for line in FACILITATION.read_text().splitlines():
        if line.startswith(f"| {prefix} "):
            return line
    pytest.fail(f"no row '{prefix} …' in FACILITATION.md")


# ---- the whole notebook -----------------------------------------------------------------------------


def test_question_count_per_section_after_all_merges_and_cuts(nb):
    """Merges and cuts together: Q10 into Q11, Q16, Q20, Q21, Q30, Q33 cut, Q31 into Q32 → 35 become 28;
    then (2026-10-22, C1/C3) the metrics section's question and the detective beat's Q5 → 26.
    Counted by section, never by number (the numbers are renumbered continuously)."""
    expected = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 3, 7: 4, 8: 4, 9: 1, 10: 2, 11: 0}
    got = {n: len(question_paragraphs(section(nb, n))) for n in expected}
    assert got == expected, f"questions per section: {got}"
    assert sum(got.values()) == 26


# ---- section 1, 2 (row 16) --------------------------------------------------------------------------


def test_row16_one_class_grid_section_2_reuses_the_one_from_section_1(nb):
    assert not any("show_all_classes" in c.source for c in nb.cells), "the second class grid is gone"
    s2 = section(nb, 2)
    q = questions_matching(s2, r"park camera")
    assert q, "the question about the fixed park camera's view stays in section 2"
    q_idx = q[0][0]
    next_turn = next(
        i for i in range(q_idx, len(s2)) if is_callout(s2[i]) and s2[i].metadata["callout"] == "your-turn"
    )
    assert not any(image_outputs(c) for c in s2[q_idx:next_turn]), (
        "no second grid between the question and 'Your turn'"
    )


def test_row16_section_1_grid_before_the_first_question_shows_six_photos_per_class(nb):
    """Section 1 draws its grid with plt.show(), which leaves no output under the smoke run, so the promise is
    read from the sentence that introduces the grid."""
    s1 = section(nb, 1)
    assert not re.search(r"\bthree (example )?photos", prose(s1), re.I), (
        "the grid is still introduced as three"
    )
    grid = cells_matching(s1, r"\b(six|6) (example )?photos", kinds=("prose",))
    assert grid, "section 1's grid is raised to six photos per class"
    assert grid[0] < question_paragraphs(s1)[0][0], "and it comes before the first question"


# ---- section 3 (rows 11, 14; keep 12, 31) -----------------------------------------------------------


def test_c3_detective_beat_and_its_question_are_cut(nb):
    s3 = section(nb, 3)
    assert find_heading(s3, r"detective beat") is None, "the detective beat is cut (C3, 2026-10-22)"
    assert not questions_matching(s3, r"longest.side value do most photos share"), "its question is cut"
    assert not any("detective" in plain(c) for c in nb.cells if is_prose(c)), "nothing points at it any more"


def test_row14_brightness_violin_is_a_red_optional_part(nb):
    s3 = section(nb, 3)
    flags = optional_flags(s3)
    violin = cells_matching(s3, r"violin")
    assert violin, "the brightness violin is not cut, only made optional"
    assert all(flags[i] for i in violin), "every cell about the violin sits under a '### 🔴 Optional' heading"


def test_keep_row31_colour_half_stays_in_the_lesson(nb):
    s3 = section(nb, 3)
    flags = optional_flags(s3)
    colour = find_heading(s3, r"colour modes and colours")
    assert colour is not None and not flags[colour], "the colour half stays in the lesson"


# ---- section 4 and 8 (row 15; keep row 23) ----------------------------------------------------------


def test_row15_what_a_label_costs_moved_from_section_4_to_the_start_of_section_7(nb):
    s4, s8 = section(nb, 4), section(nb, 7)
    assert not re.search(r"what a label costs", prose(s4), re.I), (
        "section 4 no longer has the clicks table's heading"
    )
    assert not any("clicks per photo" in outputs_text(c) for c in s4), "no clicks table left in section 4"
    assert any("clicks per photo" in outputs_text(c) for c in s8), "the clicks table shows in section 8"
    costs = find_heading(s8, r"what a label costs")
    first_part = find_heading(s8, r"boxes per class vs photos per class")
    assert costs is not None and first_part is not None and costs < first_part, "it opens section 8"


def test_keep_row23_q8_stays_a_separate_question_in_section_4(nb):
    s4 = section(nb, 4)
    q = questions_matching(s4, r"more than one animal")
    assert q and not optional_flags(s4)[q[0][0]]


# ---- section 5 (rows 3, 18, 22; keep row 25) --------------------------------------------------------


def test_row3_least_typical_photos_are_a_red_optional_part_next_to_embeddings(nb):
    s5 = section(nb, 5)
    flags = optional_flags(s5)
    hits = cells_matching(s5, r"least typical")
    assert hits, "least-typical photos are moved, not cut"
    assert all(flags[i] for i in hits), "every cell about least-typical photos is under a 🔴 Optional heading"
    emb = questions_matching(s5, r"outliers")
    assert emb and flags[emb[0][0]], "the embeddings question is in the optional region too"


def test_row18_hunt_prints_the_answer_key_without_the_extra_figure(nb):
    s5 = section(nb, 5)
    key = [c for c in s5 if "answer key:" in outputs_text(c)]
    assert key, "the printed answer key stays"
    assert not any(image_outputs(c) for c in key), "the cell that prints the key no longer draws a figure"


def test_row22_exact_copies_are_the_first_line_of_the_near_duplicate_question(nb):
    s5 = section(nb, 5)
    assert not questions_matching(s5, r"byte-identical pairs do you expect"), (
        "no separate exact-duplicates question"
    )
    near = questions_matching(s5, r"threshold|flagged|near-duplicate")
    assert near, "the near-duplicate question stays"
    assert re.search(r"exact|identical|byte", near[0][1], re.I), "and it now asks about exact copies first"


def test_keep_row25_embeddings_question_stays_red_with_a_note_for_students(nb):
    s5 = section(nb, 5)
    flags = optional_flags(s5)
    q = questions_matching(s5, r"outliers")
    assert q, "the embeddings question is not cut"
    idx, para = q[0]
    assert "🔴" in para and flags[idx], "it keeps its 🔴 level and its place under '### 🔴 Optional'"
    region = [c for c, f in zip(s5, flags, strict=True) if f]
    assert re.search(r"\bscale\b", prose(region), re.I), (
        "the optional part has a short note that teams use embeddings at scale (Simone's wording)"
    )


# ---- section 6 (row 5; keep row 28) -----------------------------------------------------------------


def test_row5_cross_validation_question_and_table_gone_callout_and_fold_diagram_stay(nb):
    s6 = section(nb, 6)
    assert not questions_matching(s6, r"leave-one-out"), "the leave-one-out question is cut"
    assert not any("lowest class share" in outputs_text(c) for c in s6), (
        "the cross-validation table is not shown"
    )
    callouts = [plain(c) for c in s6 if is_callout(c) and c.metadata["callout"] == "theory"]
    assert any("leave-one-group-out" in t for t in callouts), "the 📘 callout on the four schemes stays"
    diagram = [c for c in s6 if "Toy table of 12 photos" in c.source]
    assert diagram and image_outputs(diagram[0]), "the fold diagram stays"


def test_keep_row28_ruler_question_stays_red_optional_in_section_6(nb):
    s6 = section(nb, 6)
    q = questions_matching(s6, r"guessing scores")
    assert q, "the ruler question stays"
    idx, para = q[0]
    assert "🔴" in para and optional_flags(s6)[idx]


# ---- C1 (2026-10-22): one lazy-model demo, the alert metric in section 4, no metrics section ----------


def test_c1_metrics_section_is_gone_and_the_alert_metric_is_chosen_in_section_4(nb):
    assert not any(
        t for c in nb.cells for _, _, t in headings([c]) if re.search(r"metrics for the park", t, re.I)
    )
    s4 = section(nb, 4)
    q8 = questions_matching(s4, r"lazy rules")
    alert = next((i for i, c in enumerate(s4) if re.search(r"^ALERT_METRIC\s*=", c.source, re.M)), None)
    assert q8 and alert is not None and alert > q8[0][0], (
        "✍️ ALERT_METRIC sits in section 4, after the lazy rules"
    )
    assert re.search(r"precision", prose(s4), re.I), "precision is defined where it is offered as a choice"


def test_c1_one_always_the_big_class_demo(nb):
    assert not any("prepared predictions" in plain(c).lower() for c in nb.cells if is_prose(c))
    assert not any(re.search(r"background.{0,3}everywhere", plain(c), re.I) for c in nb.cells if is_prose(c))
    assert not re.search(r"always answers .sheep", prose(section(nb, 1)), re.I), (
        "section 1 does not pre-empt Q8"
    )
    assert not any(
        "Pet-food shop" in outputs_text(c) or "pet-food shop" in plain(c).lower()
        for c in nb.cells
        if is_prose(c) or c.cell_type == "code"
    )


def test_c1_scoring_metrics_are_a_go_deeper_box_in_section_7(nb):
    s7 = section(nb, 7)
    assert any("def box_iou_dice" in c.source for c in s7), "box_iou_dice now lives in section 7 (boxes)"
    deeper = [plain(c) for c in s7 if is_callout(c) and c.metadata["callout"] == "go-deeper"]
    assert any(all(w in t for w in ("mAP", "Dice", "IoU")) for t in deeper), (
        "mAP, IoU and Dice in a go-deeper"
    )


# ---- section 7, boxes (rows 10, 19; keep row 30) -----------------------------------------------------------


def test_row10_crowding_is_a_red_optional_part_and_the_iou_drawing_stays(nb):
    s8 = section(nb, 7)
    flags = optional_flags(s8)
    q = questions_matching(s8, r"overlapping pairs")
    assert q and flags[q[0][0]] and "🔴" in q[0][1], (
        "the crowding question is 🔴 and under a 🔴 Optional heading"
    )
    table = [i for i, c in enumerate(s8) if "pair of boxes with IoU" in outputs_text(c)]
    assert table and all(flags[i] for i in table), "the crowding table and gallery are in the optional part"
    drawing = [c for c in s8 if "IoU = hatched area" in c.source and image_outputs(c)]
    assert drawing, "the IoU drawing stays (section 9 uses it)"


def test_row19_box_size_table_stays_but_the_stacked_share_bars_are_dropped(nb):
    s8 = section(nb, 7)
    assert any("small / medium / large" in outputs_text(c) for c in s8), "the table of shares stays"
    assert not any("Share of small (solid)" in c.source for c in s8), "the stacked share bars are dropped"


def test_keep_row30_box_centre_heatmap_and_missing_box_hunt_stay_in_the_lesson(nb):
    s8 = section(nb, 7)
    flags = optional_flags(s8)
    for pattern in (r"box-centre heatmap", r"the box that is not there"):
        h = find_heading(s8, pattern)
        assert h is not None and not flags[h], f"'{pattern}' stays in the lesson"
    q = questions_matching(s8, r"headcount")
    assert q and not flags[q[0][0]], "the missing-box question stays in the lesson"


# ---- section 9 (rows 2/32, 21, 26) ------------------------------------------------------------------


def test_row2_more_about_masks_is_one_red_optional_part_with_heatmaps_and_thin_parts(nb):
    s9 = section(nb, 8)
    flags = optional_flags(s9)
    more = find_heading(s9, r"more about masks")
    assert more is not None and flags[more], "a '### 🔴 Optional · More about masks' part exists"
    heat = cells_matching(s9, r"heatmap per class|mean mask per class")
    assert heat and all(flags[i] for i in heat), "the per-class mean-mask heatmaps are in it"
    thin = questions_matching(s9, r"thin parts")
    assert thin and flags[thin[0][0]] and "🔴" in thin[0][1], "the thin-parts question is in it, still 🔴"
    fill = questions_matching(s9, r"fraction of its box|fill")
    assert fill and not flags[fill[0][0]] and "🔴" not in fill[0][1], (
        "the fill-ratio question stays in the lesson"
    )


def test_row21_section_9_has_exactly_one_pencil_knob_iou_min(nb):
    s9 = section(nb, 8)
    turns = your_turn_cells(s9)
    assert len(turns) == 1, f"section 9 has {len(turns)} ✏️ cells, expected exactly one"
    knobs = set(re.findall(r"^([A-Z][A-Z_]*)\s*=", s9[turns[0]].source, re.M))
    assert knobs == {"IOU_MIN"}, f"the only ✏️ value is IOU_MIN, found {sorted(knobs)}"


def test_row26_edge_alignment_question_is_removed(nb):
    s9 = section(nb, 8)
    text = prose(s9) + "\n" + "\n".join(c.source for c in s9 if c.cell_type == "code" and not is_callout(c))
    assert not re.search(r"edge alignment|canny|outline follow", text, re.I), "no edge-alignment part remains"
    heads = [t for _, _, t in headings(s9) if "🔴" in t]
    assert all(re.search(r"more about masks", t, re.I) for t in heads), (
        f"only one 🔴 part in section 9: {heads}"
    )


# ---- section 10 (rows 6, 7, 27) ---------------------------------------------------------------------

_KIND_HDR = re.compile(r"^# \d+\. (.+?) — tool", re.M)


def _hunt_cell(s10) -> str:
    return next(c.source for c in s10 if len(_KIND_HDR.findall(c.source)) >= 10)


def test_row6_challenge_plan_is_inside_the_hunt_cell_not_a_separate_question(nb):
    s10 = section(nb, 9)
    assert not questions_matching(s10, r"plan the hunt|kind → tool"), "no separate written-plan question"
    assert len(question_paragraphs(s10)) == 1, "one question in the Challenge: the hunt"
    kinds = _KIND_HDR.findall(_hunt_cell(s10))
    assert len(kinds) == 10, f"one header per kind inside the hunt cell, found {kinds}"


def test_row7_copy_and_grayscale_checks_are_prewritten_the_four_disagreement_checks_are_not(nb):
    src = _hunt_cell(section(nb, 9))
    parts = list(_KIND_HDR.finditer(src))
    bodies = {}
    for k, m in enumerate(parts):
        end = parts[k + 1].start() if k + 1 < len(parts) else len(src)
        body = src[m.end() : end].partition("\n")[2]  # without the rest of the header line
        bodies[m.group(1).lower()] = [
            ln for ln in body.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
        ]

    def code_of(*names):
        return [ln for key, lines in bodies.items() if any(n in key for n in names) for ln in lines]

    copies_start = src.index(next(m for m in parts if "exact" in m.group(1).lower()).group(0))
    overlap_start = src.index(next(m for m in parts if "overlap" in m.group(1).lower()).group(0))
    copy_code = [
        ln
        for ln in src[copies_start:overlap_start].splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    assert copy_code, "the exact/flipped/rotated/recompressed copy checks are pre-written"
    assert code_of("grayscale"), "the grayscale check is pre-written"
    for kind in ("overlap", "wrong label", "missing box", "ugly mask"):
        assert not code_of(kind), (
            f"'{kind}' is one of the four 'two sources disagree' checks: students write it"
        )


def test_row27_challenge_ruler_part_is_removed(nb):
    s10 = section(nb, 9)
    assert not re.search(r"\bruler\b", prose(s10), re.I), (
        "no ruler in the Challenge (the ruler stays in section 6)"
    )
    red = [t for _, _, t in headings(s10) if "🔴" in t]
    assert len(red) == 1 and re.search(r"hunt", red[0], re.I), "the only 🔴 part is the hunt (C2)"


def test_c2_challenge_is_a_demo_and_the_hunt_is_optional(nb):
    s9 = section(nb, 9)
    flags = optional_flags(s9)
    hunt = questions_matching(s9, r"the hunt")
    assert hunt and flags[hunt[0][0]] and "🔴" in hunt[0][1], "the hunt question is 🔴 optional"
    key = find_heading(s9, r"answer key")
    assert key is not None and not flags[key], "the answer key stays in class, for the demo"
    ref = cells_matching(s9, r"_reference_findings\(\)", kinds=("code",))
    assert ref and not flags[ref[0]], "the reference solution runs in class"


# ---- section 11 (row 1) -----------------------------------------------------------------------------


def test_row1_section_11_in_class_part_then_an_after_the_lab_block(nb):
    s11 = section(nb, 10)
    lab = find_heading(s11, r"after the lab")
    assert lab is not None, "section 11 has an 'After the lab' heading"
    tree = cells_matching(s11, r"├──", kinds=("prose",))
    turn = next(i for i in your_turn_cells(s11) if "MY_DATASET_DIR" in s11[i].source)
    card = cells_matching(s11, r"report card", kinds=("code",))
    assert tree and card, "the folder tree and the report card stay"
    assert max(tree[0], turn, card[0]) < lab, "tree, ✏️ MY_DATASET_DIR and the report card come before it"
    after = {
        "the dataset question": questions_matching(s11, r"what you expect for your dataset"),
        "the 'how to read' notes": [
            (i, "") for i in cells_matching(s11, r"how to read the ratio", kinds=("prose",))
        ],
        "the guiding questions": [(find_heading(s11, r"^the guiding questions"), "")],
        "the guiding-questions answers": questions_matching(s11, r"guiding question"),
    }
    for what, hits in after.items():
        assert hits and hits[0][0] is not None and hits[0][0] > lab, f"{what} come after 'After the lab'"


# ---- the lesson sections still in class (keep row 29) -----------------------------------------------


def test_keep_row29_section_1_has_no_optional_or_after_the_lab_part(nb):
    s1 = section(nb, 1)
    assert not any(flag for flag in optional_flags(s1)), "section 1 stays in class"


# ---- FACILITATION.md (rows 4, 24, 1) ----------------------------------------------------------------


def test_c2_facilitation_runs_the_challenge_as_a_demo():
    row = facilitation_row("9 ·")
    assert re.search(r"demo", row, re.I) and re.search(r"optional", row, re.I), (
        "the Challenge row says demo in class, hunt optional"
    )


def test_row1_facilitation_names_the_after_the_lab_block():
    assert re.search(r"after the lab", facilitation_row("10 ·"), re.I)


# ---- review 2026-10-22 (R1, R7) and C4 ------------------------------------------------------------------


def test_r1_every_worked_answer_is_a_closed_toggle(nb):
    plain_answers = [
        c
        for c in nb.cells
        if c.cell_type == "markdown"
        and re.search(r"worked answer", c.source, re.I)
        and not re.search(r"open the worked answer", c.source, re.I)
    ]
    assert not plain_answers, "no worked answer left in an open markdown cell"
    boxes = [c for c in nb.cells if is_callout(c) and c.metadata["callout"] == "answer"]
    assert len(boxes) >= 26 and all(
        "<details" in c.source and not re.search(r"<details[^>]*\bopen\b", c.source) for c in boxes
    )


def test_c4_dedupe_before_split_is_said_once(nb):
    pitfalls = [
        plain(c)
        for c in nb.cells
        if is_callout(c)
        and c.metadata["callout"] == "pitfall"
        and re.search(r"dedup|split after cleaning", plain(c), re.I)
    ]
    assert len(pitfalls) == 1, f"one pitfall on deduplicating before splitting, found {len(pitfalls)}"
    assert not any(re.search(r"why cleaning decides", plain(c), re.I) for c in nb.cells if is_prose(c))


def test_r7_trail_camera_bursts_draw_the_animals_box(nb):
    src = "\n".join(c.source for c in section(nb, 6))
    assert "def draw_cct_boxes" in src and src.count("draw_cct_boxes(ax, row, thumb)") >= 2
