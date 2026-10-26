# How to apply Susanne's edited copy (walkthrough 2026-10-23)

For Simone, and for the agent he asks to do it. The work list is `reviews/2026-10-23-walkthrough.md`.
**Order, never changed: her edits first, then the drafted items, then the rest. Where they clash, her text wins**
and Simone sees the before/after.

## What is already prepared

- **This branch (`susanne/2026-10-23`)**: `sections/` are exactly v04, the version she edited, plus `just port FILE`
  (`scripts/port.py`). It aligns her cells with a fresh build by content (Colab may drop cell metadata), restores
  every unchanged box from its source, converts edited boxes from HTML back to `<!-- callout: … -->` markdown
  (flagged ⚠️ in the report), writes `reviews/port-<date>.md`, and checks that the ported sections rebuild her copy.
  Do **not** use `just sync` on her copy: it writes boxes back as raw HTML code cells.
- **`drafts/susanne-prep`**: items built before her copy, one commit each, merged **after** her edits:
  - S4–S7, S9, S12, S13: `task` kind (questions + your-turn), Recipe in theory blue, objectives as plain markdown,
    box text size `BODY_PX`, section numbers from the file order, chapter icons (🏷️ 📦 🎭, a proposal for her).
    Done in `scripts/build.py` through aliases, so section text did not change.
  - S14: photo browsers in section 1 (class picker + ◀ ▶; 🔀 sets of six per class).
  - S19, S20: SPEC audience; hashing text (random pair vs closest neighbour); a lost word in Q3's answer.
  - `tests/test_content.py` callout kinds follow her set: protected file, **drop that commit if Simone says no**.

## Steps

```
cd ~/Work/notebooks/eda-teaching-notebook
git fetch origin && git switch susanne/2026-10-23 && git pull --ff-only
git switch -c susanne/apply-$(date +%F)
cp ~/Downloads/SUSANNE_FILE.ipynb build/susanne-copy.ipynb        # build/ is gitignored: her file stays local
uv run python scripts/port.py build/susanne-copy.ipynb --dry-run  # 1. read reviews/port-<date>.md
just port build/susanne-copy.ipynb                                # 2. port, round-trip check must be ✅
git add -A && git commit -m "Susanne's edits, ported as she made them"
git merge drafts/susanne-prep                                     # 3. conflicts: keep HER line in sections/
just build && git add -A && git commit --no-edit                  #    the .ipynb is rebuilt, never merged by hand
```

4. Apply the remaining **wait** items of the work list (S1–S3, S8, S10, S11, S15) where her copy has not settled
   them, rewrite SPEC L-001 to her box set, and list every clash for Simone.
5. `just check`, then `just version "Susanne's pass"`, then a PR. Simone runs it once in Colab and checks by eye:
   box text size vs markdown (`BODY_PX`), the photo browsers' buttons, the chapter icons in the table of contents.

## Checks

- Her copy must come from **v04** (main `9fa8c5f`). If the dry run lists changes she cannot have made, stop: she
  edited another version, and the port needs that version's sections as its base.
- The lint may flag her edits (e.g. a sentence between a heading and its objective). Her edit stays; the rule or
  the placement is adjusted afterwards, with Simone.
- Timing, section budgets and what is in or out are **not** touched unless she changed them (Simone, 2026-10-23).
