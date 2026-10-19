# Facilitation guide — Know Your Data: Exploring and Preparing Image Datasets at the Wildwood Farm Park

For whoever presents this notebook. It is generated from `SPEC.md` (N-033); regenerate it when the SPEC changes.

**Session:** <venue / course> · <total minutes> · <blocks and breaks>
**Before the session:** <setup check sent? data cached? room/projector?>
**Core principle:** predict → run → reflect. Ask learners to say or write what they expect before each interactive cell.

## Section-by-section

| Section | You present (what to say) | They do | Pause / interaction | Min |
|---|---|---|---|---|
| 0 · Setup & the park | Read the banner's three outcomes aloud; they come back as takeaways at the end. Tell the park story: four cameras, three jobs (recognise, count, outline), and the one case that matters, a dog in the sheep paddock. Say that nothing is trained today; we check whether the data is fit for the job first. Point at the `use()` demo: unfinished functions never break "Run all". | Run the Setup cell, the `use()` demo and the data cell. Fill in `double(x)` and rerun to see ✅. Skim `images.head()`. | Start the data cell *before* the story: the ~115 MB download takes a minute or two. Show of hands after the table: "who saw an `n_animals` above 1?" That seeds section 4's "which label then?" | 5 |
| 1 · Image datasets | Walk the two-column table of sources and labelling tools in one minute: "look before you label". Point out that our Play Dataset came from COCO and the trail cameras from LILA, and that reading the licence is part of finding a dataset. Then the 📘 checklist from the lecture: read the bullets aloud, they already know them. Frame the code cell as "the checklist applied to our data: numbers where the tables can answer, 👀 where only eyes can". | Predict Q1 (which bullets pass, which worries them for the dog alert), then run the checklist cell and compare with the worked answer. | Pause before running: show of hands "who expects the class balance to pass?" After the output, point at every ⚠️ and 👀 line: each names the section that handles it, which is the roadmap for the rest of the lab. | 5 |

## Where people get stuck
<!-- Known failure points, with the one-line fix to give them. -->

## If you are short on time
<!-- Which sections or 🔴 parts to skip, in order. -->
