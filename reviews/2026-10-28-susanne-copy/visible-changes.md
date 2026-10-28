# Visible changes to Susanne's copy (work-list groups 1–6)

Every change a student can see: markdown, box HTML, headings, cell titles shown in Colab's form view, printed
strings, figure titles, axis labels and legends. One line each: `cell title · before → after · item ID`.
Cell titles are the ones in `sections/` after the change (her copy's cell number in brackets where it helps).
Code that only a helper cell holds, and renamed variables a student never reads, are not listed.

## Group 1 · Breakage

- Read your folder (or build the sample), and show what was found [201] · the cell stopped with
  `NameError: name 'clean_images' is not defined`, and the five chapter-10 cells after it stopped too → the park
  sample is built from all Play Dataset images and the chapter runs to its end · C-24 / R-01
- Read your folder (or build the sample), and show what was found [201] · figure "A first look: are these your
  images and your classes?": every panel said "cannot be opened" → the panels show the images · R-03
- Check 3: exact, recompressed, flipped and rotated copies [210] · the cell would stop with
  `NameError: THUMB_FOR_HASH` and then `TypeError: 'int' object is not iterable` → it runs and draws the copy
  pairs · R-02, R-03
- Every browser (her cells 28, 72, 97, 120, 122, 129, 138, 145, 149, 152, 155, 159, 168, 176) · after a later
  browser had run, the buttons of an earlier one raised `AttributeError` or drew into another cell's output →
  each browser drives only itself (no text changes) · N-13
- 🔎 Explore Color Modes, Grayscale Images, and RGB Distributions [46] · no visible change with six classes (the
  grid now grows a row per three classes) · C-15
