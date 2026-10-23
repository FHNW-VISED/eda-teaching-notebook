# Walkthrough with Susanne — 2026-10-23 (work list)

What came out of Simone and Susanne's walkthrough of v04. Susanne is editing her own copy of the notebook
for reader flow. **Nothing here is applied yet.** The order, once her copy is in:

1. Diff her `.ipynb` against the build (Jupytext text, cell by cell) and port her edits into `sections/*.py` **as she made them**.
2. Apply the items below that her edits have not already settled.
3. Where an item and her edit disagree, **her edit wins**; Simone sees the before/after.

⛔ **Timing:** no change to the section budget, the SPEC times or what is in or out. Susanne decides that.

RULES.md now includes her notebook rules (SN-001…SN-005) because SPEC.md says `**With:** Susanne Suter`.
L-001 (the callout kinds) and the level legend in section 0 conflict with SN-001…SN-003; they are brought
in line in step 2.

Status: **wait** = after her copy · **done** · **issue** = tracked as a GitHub issue.

## Order and setup

| # | Item | Proposed change | Status |
|---|---|---|---|
| S1 | The setup's ✍️ demo (`double`) comes right after the introduction and reads as a change of topic. | Placement follows her edit; add a bridging sentence if she kept the order. | wait |
| S2 | The ✍️ stub + reference-fallback mechanism is not introduced. | One short demo that says what it is and what the learner will see; no count of how often it comes (N-053). | wait |
| S3 | The takeaway under the pen icon mixes the mechanism with a dataset question (one row per photo or per animal). | Split: the mechanism goes in S2's intro, the dataset question becomes a Task in §1. | wait |

## Callouts

| # | Item | Proposed change | Status |
|---|---|---|---|
| S4 | Too many callout colours. | Keep 💡 answer toggle, 📘 theory/info (blue), 🔍 go-deeper (grey), and one Task kind; the rest become plain markdown or merge (SN-001). | wait |
| S5 | Questions and ✏️ Your turn look like two different things. | Merge into one **Task** style; legend "Tasks are marked beginner, intermediate, advanced" (SN-002). Meeting decision. | wait |
| S6 | 🧾 Recipe has its own green. | Recipe uses the theory blue (SN-004). | wait |
| S7 | 🎯 Objective boxes. | Objective as plain markdown under the section heading (SN-003, N-003). | wait |
| S8 | ✅ Takeaway green can be read as the 🟢 level. | Undecided: follow her copy. Either way no colour like a level dot (N-044). | wait |
| S9 | Callout text renders smaller than markdown text. | Callout HTML uses the markdown body size (N-054). | wait |
| S10 | A box that only says what the next three cells do. | Plain markdown, or drop it if her copy did. | wait |

## Navigation

| # | Item | Proposed change | Status |
|---|---|---|---|
| S11 | Tasks are hard to find in a long notebook. | Level dot in each Task heading, so the table of contents shows them (N-055). | wait |
| S12 | No sense of the three main parts. | Chapters: classification (§0–6), object detection (from "Same data, boxes"), segmentation (from "Same data, masks"), each with an icon in its headings; §9–11 sit after the three chapters. Simone proposes the emojis to Susanne. | wait |
| S13 | Section numbers go stale when sections move. | Build-time numbering from the section order (N-056); check that Colab's table of contents shows it. | wait |

## Interactivity

| # | Item | Proposed change | Status |
|---|---|---|---|
| S14 | The single-photo view and the grid always show the same first images. | Class selector and next/previous (N-057), smoke-safe. | wait |

## Pedagogy and audience

| # | Item | Proposed change | Status |
|---|---|---|---|
| S15 | Learning outcomes. | Bloom's-taxonomy verbs; take her renamed block from her copy (SN-005). | wait |
| S16 | Level calibration. | 🟢 readable off a plot/output · 🟡 needs reasoning beyond what is shown · 🔴 rare optional extension (N-005). Most Tasks stay 🟡. | wait |
| S17 | Across-tasks boxes and the click-count comparison work well. | Keep. | — |
| S19 | Audience is a CS class that knows ML and has had 7 weeks of classical image processing; this is their intro to deep-learning CV. | Update SPEC §2. | wait |

## Correctness

| # | Item | Proposed change | Status |
|---|---|---|---|
| S20 | §5 says unrelated photos sit "about 30/32 bits apart" (lines 126, 275), but the histogram shows nearest-neighbour distances (median ≈ 20) and an unrelated pair sits 12 apart. | Separate "a random pair ≈ 32" from "the closest of ~600 photos ≈ 20"; one line on why unrelated photos can be close (coarse 32×32 layout, only 64 bits). | wait |

## Later

| # | Item | Status |
|---|---|---|
| S21 | In-notebook labelling exercise (one click, a box, a mask polygon) to feel annotation cost. Next iteration or the AMLD workshop. | issue #20 (`future`) |
| S22 | Possible reuse in the continued-education course. | — |
