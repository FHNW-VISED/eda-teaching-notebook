# Cut pass progress (issue #18, 2026-10-22)

Status: todo · done · partly · skipped

## Phases
- [x] 1 Tests first — done. tests/test_cuts.py (34 tests): 8 keep-as-is pass, 26 fail as expected (all apply rows +
  FACILITATION rows 4/24/1 + question count 35→28). Word-budget warnings in lint; test_cuts in `just smoke`.
  Words before: 0:584 1:852 2:618 3:1555 4:1168 5:1726 6:1558 7:1323 8:1393 9:1791 10:916 11:1341 12:329 (≈15,150 vs 7,050).
- [x] 2 Cuts + text diet (A–G) — done. Gate: smoke + content + cuts pass; lint N-006 only until phase 3 (by design).
  Orchestrator sent A, C, E back: worked answers had gone into <details> (SPEC L-rule line 248: shown openly);
  A also restored facts lost by rewording. Q1 worked answer restored by orchestrator. A read-only deletion audit
  found dropped facts in 04–11 and hidden answers in 06/07/11; owners restored them. 5 fragile cut tests fixed
  (test bugs, not notebook bugs).
- [x] 3 Numbering, FACILITATION, SPEC — done. 28 questions; map Q11→10 … Q35→28 (see final report). FACILITATION
  rows 1–11 + short-on-time list; SPEC bullets §2–11 + §6b extensions line. Ruff E501 re-wraps (Haiku) and three
  shortened titles (orchestrator). `just check` passes: 48 tests.
- [x] 4 Time keeper — done. reports/review-cuts-2.md: 144 lab / 183 careful (planner 114/146). SPEC + FACILITATION
  Time columns written (Σ 144).
- [x] 5 Finish — done. `just check` passes (48 tests); versions/v03-2026-10-22-cut-pass-2026-10-22;
  `just version-check` passes; reports/cuts-2026-10-22.md written.

## Words after phase 2 (lint, visible)
0: ≤400 · 1: ≤500 · 2: 545/450 · 3: 1294/700 · 4: 781/600 · 5: 1107/800 · 6: 981/800 (lint undercounts ≈220: Q18 🔴
part has no closing heading) · 7: 820/550 · 8: 986/400 · 9: 996/650 · 10: 659/650 · 11: 1101/300 (≈330 in class,
rest is the After-the-lab block, which the spec's exclusions still count) · 12: ≤250

## Decision rows
| # | Owner | Status | Note |
|---|---|---|---|
| 16 §2 one grid | A+G | done | §1 grid 6/class; show_all_classes gone |
| 11 §3 toy figure | A | done | |
| 14 §3 violin → 🔴 | A | done | 🔴 part before detective beat |
| 15 §4 clicks table → §8 | B | done | opens §8 |
| 10 §8 Q24 → 🔴 | B | done | IoU drawing + OVERLAP_IOU stay in lesson |
| 19 §8 Q23 bars | B | done | |
| 30 keep §8 heatmap + hunt | B | done (untouched) | |
| 3 §5 Q13 → 🔴 | C | done | one 🔴 part with Q13 + Q14 |
| 18 §5 answer-key figure | C | done | printed key stays |
| 22 §5 Q10 into Q11 | C | done | |
| 25 keep Q14 🔴 + note | C | done | "at scale" note |
| 5 §6 Q16 + CV table | D | done | cv_schemes in collapsed cell; "Read the picture" line carries 0 % |
| 4 §7 Q20 quiz | D | done | FACILITATION in phase 3 |
| 8 §7 Q21, IoU → go-deeper | D | partly | figure output still renders on the page |
| 9 moot | D | skipped | covered by #8 |
| 17 §7 shop cell | D | done | "about 82 % cat, 83 % dog, paddock recall on dog 50 %" (recomputed, seed 42) |
| 20 keep predictions.head() | D | done (untouched) | |
| 13 keep toy 2×2 | D | done (untouched) | |
| 2/32 §9 More about masks → 🔴, keep Q27 | E | done | 🔴 part placed last in §9 |
| 21 §9 one ✏️ IOU_MIN | E | done | other four hard-coded at top of Q29 cell, same values |
| 26 §9 Q30 remove | E | done | |
| 6 §10 Q31 into Q32 | F | done | |
| 7 §10 pre-write checks | F | done | kinds 1–5, 7 pre-written; 6, 8, 9, 10 stubs; reference cell moved above (N-046) |
| 24 §10 8-min box | phase 3 | done | FACILITATION |
| 27 §10 Q33 remove | F | done | |
| 1 §11 After the lab | F | done | check cells stay in class (report card needs them) |
| eW text diet | all | partly | moved only; overshoot listed above |
| 33 SPEC Time column | phase 4 | done | Σ 144 lab |
