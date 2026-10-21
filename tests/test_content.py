"""Content assertions: does the notebook TEACH what SPEC.md promises? (Simone's G-28, 2026-10-19)

Written with Simone during grilling. ⛔ The night agent never edits this file
(scripts/night.sh reverts it). Two kinds of check:
  1. every section leaves its analysis behind and the counts add up;
  2. every planted problem is found by the notebook's own reference solution (so hunts are solvable).
Sections write named results into RESULTS; the smoke test dumps them to build/smoke-results.json
(SPEC §9). Runs after tests/test_smoke.py. **Progressive gate:** a test runs only once every section
it needs is marked built (status other than "todo") in SPEC.md's sections table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parent.parent
EXECUTED = ROOT / "build" / "smoke-executed.ipynb"
RESULTS_FILE = ROOT / "build" / "smoke-results.json"
CLASSES = {"cat", "dog", "bird", "horse", "sheep", "zebra"}
KARPATHY = "karpathy.github.io/2019/04/25/recipe"
CALLOUT_KINDS = (  # SPEC L-001: every Callout kind is used at least once ("go-deeper" is optional)
    "theory",
    "your-turn",
    "questions",
    "across-tasks",
    "pitfall",
    "recipe",
    "objective",
    "takeaway",
    "banner",
)


SPEC = ROOT / "SPEC.md"


def _statuses() -> dict[str, str]:
    rows = {}
    for line in SPEC.read_text().splitlines():
        m = re.match(r"^\|\s*\d+\s*\|\s*sections/(\d\d)_[\w-]+\.py\s*\|.*\|\s*([\w-]+)\s*\|\s*$", line)
        if m:
            rows[m.group(1)] = m.group(2).lower()
    return rows


def requires(*sections: str):
    """Skip until every listed section (e.g. "04") is built; "all" = every section in SPEC."""
    st = _statuses()
    wanted = list(st) if sections == ("all",) else list(sections)
    todo = [n for n in wanted if st.get(n, "todo") == "todo"]
    return pytest.mark.skipif(bool(todo), reason=f"sections not built yet: {todo}")


@pytest.fixture(scope="module")
def nb():
    if not EXECUTED.exists():
        pytest.skip("no executed notebook — run `just smoke`")
    return nbformat.read(EXECUTED, as_version=4)


@pytest.fixture(scope="module")
def R():
    if not RESULTS_FILE.exists():
        pytest.fail("build/smoke-results.json missing: 00_setup.py must create RESULTS = {} (SPEC §9)")
    return json.loads(RESULTS_FILE.read_text())


def source_of(nb, prefix: str) -> str:
    return "\n".join(c.source for c in nb.cells if c.metadata.get("section", "").startswith(prefix))


def need(R, key):
    assert key in R, f"RESULTS['{key}'] was never set — the section that owns it is missing or incomplete"
    return R[key]


# ---- whole-notebook checks -------------------------------------------------------------------


@requires("00")
def test_outcomes_and_story_stated(nb):
    src = source_of(nb, "00_")
    assert "Wildwood" in src, "00_setup must introduce the Wildwood Farm Park (SPEC §4)"
    assert src.count("🟢") and src.count("🟡") and src.count("🔴"), "level legend missing from the banner"


@requires("all")
def test_every_callout_kind_is_used(nb):
    # build.py turns each callout marker cell into a code cell tagged with metadata["callout"] (N-034)
    used = {c.metadata.get("callout") for c in nb.cells if c.cell_type == "code"}
    missing = [k for k in CALLOUT_KINDS if k not in used]
    assert not missing, f"Callout kinds never used: {missing} (SPEC L-001)"


@requires("02", "03", "04", "05")
def test_karpathy_recipe_quoted_in_sections_2_to_5(nb):
    missing = [p for p in ("02_", "03_", "04_", "05_") if KARPATHY not in source_of(nb, p)]
    assert not missing, f"no linked Karpathy Recipe Callout in sections {missing} (SPEC L-004)"


@requires("00")
def test_one_colour_per_class(R):
    assert set(need(R, "class_colors")) == CLASSES


# ---- section checks: analysis exists and counts add up ----------------------------------------


@requires("00", "03")
def test_s03_every_image_inspected(R):
    assert need(R, "image_stats_n") == need(R, "n_images")


@requires("00", "04")
def test_s04_class_counts_add_up(R):
    counts = need(R, "images_per_class")
    assert set(counts) == CLASSES
    assert sum(counts.values()) == need(R, "n_images")


@requires("00", "05")
def test_s05_guided_hunt_solvable_and_cleaning_reported(R):
    hunt = need(R, "hunt5")
    assert hunt["planted"] >= 4 and hunt["found"] == hunt["planted"], f"section 5 hunt: {hunt}"
    assert 0 < need(R, "n_clean") <= need(R, "n_images") + hunt["planted"]


@requires("06")
def test_s06_leakage_shown_without_a_model(R):
    assert need(R, "sibling_rate_random") > 0, "random split of bursts must leave siblings"
    assert need(R, "sibling_rate_by_camera") == 0, "a split by camera must leave no siblings across the split"
    assert {"kfold", "stratified_kfold", "loo", "logo"} <= set(need(R, "cv_schemes")), "G-27: 4 CV schemes"


@requires("07")
def test_s07_alert_metric(R):
    assert need(R, "alert_metric") == "recall"


@requires("08")
def test_s08_box_counts_add_up_and_hunt_solvable(R):
    boxes = need(R, "boxes_per_class")
    assert set(boxes) == CLASSES
    assert sum(boxes.values()) == need(R, "n_boxes")
    hunt = need(R, "hunt8")
    assert hunt["planted"] >= 1 and hunt["found"] == hunt["planted"], f"section 8 hunt: {hunt}"


@requires("09")
def test_s09_pixel_shares_and_ugly_masks(R):
    px = need(R, "pixels_per_class")
    assert "background" in px and CLASSES <= set(px)
    assert abs(sum(px.values()) - 1.0) < 1e-3, "pixel shares must sum to 1 (including background)"
    hunt = need(R, "hunt9")
    assert hunt["planted"] == 5 and hunt["found"] == 5, f"section 9 hunt (5 ugly-mask kinds): {hunt}"


@requires("10")
def test_s10_challenge_reference_scores_100_percent(R):
    ch = need(R, "challenge_reference")
    assert ch["planted"] >= 10 and ch["found"] == ch["planted"], f"challenge with reference solution: {ch}"
    assert need(R, "challenge_types") >= 8


@requires("11")
def test_s11_swap_in_cell_runs_on_folder_per_class(R):
    assert need(R, "swap_in_ok") is True
