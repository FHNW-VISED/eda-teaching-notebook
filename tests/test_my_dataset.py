"""Section 10's dataset reader on the folder layouts students really hand it (Simone, 2026-10-22: "have you
tested with another dataset? … make the setup as clear as possible … and test it out").

The reader cell is executed on its own, outside the notebook, so every layout is checked in milliseconds:
the right layout must be read, and every wrong one must stop with a message that says what to change.
"""

from __future__ import annotations

import zipfile
from collections import Counter
from pathlib import Path

import jupytext
import pandas as pd
import pytest
from PIL import Image

SECTION = Path(__file__).resolve().parent.parent / "sections" / "10_your_dataset.py"


@pytest.fixture(scope="module")
def reader(tmp_path_factory):
    """The functions of the 'Dataset reader' cell, with the few notebook globals it needs."""
    cell = next(c.source for c in jupytext.read(SECTION, fmt="py:percent").cells
                if c.source.startswith("# @title Dataset reader"))
    ns = {"Path": Path, "pd": pd, "Counter": Counter, "IN_COLAB": False,
          "DATA_DIR": tmp_path_factory.mktemp("data")}
    exec(cell, ns)  # noqa: S102 (our own notebook source)
    return ns


def img(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (90, 120, 60)).save(path)


def classes(root: Path, names=("ants", "bees"), n=3) -> Path:
    for c in names:
        for k in range(n):
            img(root / c / f"{c}_{k}.jpg")
    return root


def stops(reader, path, *fragments):
    with pytest.raises(SystemExit) as err:
        reader["find_dataset"](path)
    msg = str(err.value)
    assert msg.startswith("❌"), msg
    for f in fragments:
        assert f in msg, f"'{f}' missing from: {msg}"


def test_the_right_folder_is_read(reader, tmp_path):
    root = classes(tmp_path / "flowers")
    assert reader["find_dataset"](str(root)) == root
    files = reader["list_folder_per_class"](root)
    assert sorted(files["class"].unique()) == ["ants", "bees"] and len(files) == 6


def test_a_pasted_path_with_quotes_and_spaces_still_works(reader, tmp_path):
    root = classes(tmp_path / "flowers")
    assert reader["find_dataset"](f'  "{root}" ') == root


def test_a_lone_wrapper_folder_is_stepped_into(reader, tmp_path):
    root = classes(tmp_path / "upload" / "flowers")
    assert reader["find_dataset"](tmp_path / "upload") == root


def test_a_zip_is_unzipped_and_read(reader, tmp_path):
    classes(tmp_path / "src" / "flowers")
    zpath = tmp_path / "flowers.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for f in (tmp_path / "src").rglob("*.jpg"):
            z.write(f, f.relative_to(tmp_path / "src"))
        z.writestr("__MACOSX/flowers/._ants_0.jpg", b"mac junk")  # what a Mac zip adds
    root = reader["find_dataset"](zpath)
    assert sorted(d.name for d in root.iterdir()) == ["ants", "bees"]
    assert len(reader["list_folder_per_class"](root)) == 6


def test_a_typo_names_the_nearest_existing_folder(reader, tmp_path):
    classes(tmp_path / "flowers")
    stops(reader, tmp_path / "flower" / "x", "does not exist", str(tmp_path), "flowers/")


def test_a_class_folder_says_go_one_level_up(reader, tmp_path):
    root = classes(tmp_path / "flowers")
    stops(reader, root / "ants", "no class folders", "one level up", str(root))


def test_split_folders_say_point_at_train(reader, tmp_path):
    classes(tmp_path / "data" / "train")
    classes(tmp_path / "data" / "val")
    stops(reader, tmp_path / "data", "split folders", str(tmp_path / "data" / "train"))


def test_a_file_that_is_not_a_zip(reader, tmp_path):
    img(tmp_path / "one.jpg")
    stops(reader, tmp_path / "one.jpg", "is a file")


def test_a_folder_without_images(reader, tmp_path):
    (tmp_path / "docs" / "a").mkdir(parents=True)
    (tmp_path / "docs" / "a" / "notes.txt").write_text("x")
    stops(reader, tmp_path / "docs", "no image files", "a/")


def test_notes_list_what_was_skipped(reader, tmp_path):
    root = classes(tmp_path / "flowers", names=("ants",))
    (root / "bees").mkdir()
    (root / "ants" / "labels.csv").write_text("x")
    (root / "ants" / "scan.HEIC").write_bytes(b"x")
    img(root / "stray.jpg")
    img(root / "ants" / "deeper" / "IMG_9.JPG")  # deeper and upper-case: still counted
    files = reader["list_folder_per_class"](root)
    assert len(files) == 4, "images deeper inside a class folder count, upper-case suffixes too"
    notes = "\n".join(reader["folder_notes"](root, files))
    for expected in ("1 images sit directly", ".csv", ".heic", "with no images: bees", "only one class"):
        assert expected in notes, f"'{expected}' missing from: {notes}"
