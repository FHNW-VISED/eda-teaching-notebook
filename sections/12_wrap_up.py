# %% [markdown]
# ## 12. Wrap-up & next week

# %% [markdown]
# <!-- callout: objective | Objective -->
# Look back at the three things you set out to do, see the park's numbers for each, and know where next
# week's training starts.

# %% [markdown]
# Nothing was trained today, and yet we now know a lot about whether the park's data is fit for its three
# jobs. The next cell collects the numbers the earlier sections measured, grouped by the three outcomes from
# the title banner.

# %%
# @title The park's numbers from today, one block per learning outcome { display-mode: "form" }
_photos = RESULTS["images_per_class"]
_boxes = RESULTS["boxes_per_class"]
_pixels = RESULTS["pixels_per_class"]
_big, _small = max(_photos, key=_photos.get), min(_photos, key=_photos.get)
recap = {
    "1 · Inspect and judge the fit": [
        f"{RESULTS['n_images']} photos of {len(CLASSES)} classes, every one measured "
        f"({RESULTS['image_stats_n']} rows of image stats)",
        f"largest class {_big} ({_photos[_big]}), smallest {_small} ({_photos[_small]}): "
        f"{_photos[_big] / _photos[_small]:.1f} : 1",
    ],
    "2 · Find and fix errors": [
        f"removed {RESULTS['n_removed']}, kept {RESULTS['n_clean']} "
        "(copies, before any split)",
    ],
    "3 · Split, metric, and the three tasks": [
        f"test photos with a burst sibling in train: random split {RESULTS['sibling_rate_random']:.0%}, "
        f"split by camera {RESULTS['sibling_rate_by_camera']:.0%}",
        f"prepared predictions: accuracy {RESULTS['accuracy']:.0%}, "
        f"recall on dog {RESULTS['recall_dog']:.0%}",
        f"bird: {_photos['bird'] / sum(_photos.values()):.0%} of photos, "
        f"{_boxes['bird'] / sum(_boxes.values()):.0%} of boxes, {_pixels['bird']:.0%} of pixels",
    ],
}
for outcome, lines in recap.items():
    print(outcome)
    for line in lines:
        print("   ", line)

# %% [markdown]
# 💡 Every line is a decision you can now defend: what to fix, how to split, what to measure, and in which
# unit (photo, box or pixel) to count. In smoke mode the numbers are tiny; the full run gives the park's real
# ones.

# %% [markdown]
# <!-- callout: takeaway | Takeaways -->
# You can now do the three things from the banner:
#
# - **1 · Inspect.** **When you see** a new image dataset → **do** look at many photos before computing
#   anything, then measure sizes, modes, colours and class counts. An outlier points at a bug; a photo the
#   park's camera would never take points at a poor fit.
# - **2 · Fix errors.** **When you see** files that may be the same photo → **do** hash the files for exact
#   copies and the pixels (with flipped and rotated variants) for near copies, open every file once, and let
#   a human judge the least typical labels and the ugliest masks. Write down "removed N, kept M".
# - **3 · Split and measure.** **When you see** photos tied together by a camera or a burst → **do**
#   deduplicate, then split by that group. Pick the metric from the expensive mistake: recall on dog for the
#   paddock alert, counting error for the headcount, IoU per class for the outlines. Always say what you
#   count: photos, boxes or pixels.

# %% [markdown]
# ### Next week: training
#
# Next week the cleaned table and the split go into a model. Karpathy's recipe continues the same way: first
# an end-to-end skeleton with a dumb baseline, then a model that overfits, then regularisation. Two things
# from today come back straight away.

# %% [markdown]
# **Data augmentation.** The flipped, rotated and recompressed copies you hunted in section 5 are what
# augmentation makes *on purpose*: a flipped sheep is still a sheep, so training sees it twice. Augment only
# the training set, and only after the split, or the copies leak into the test set again.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · Data augmentation -->
# - [Data augmentation in Python (Neptune.ai)](https://neptune.ai/blog/data-augmentation-in-python):
#   an overview of the techniques and libraries.
# - [YOLO data augmentation (Ultralytics)](https://docs.ultralytics.com/guides/yolo-data-augmentation/):
#   what each augmentation does for detection, where boxes must move with the pixels.

# %% [markdown]
# **Tracking experiments (MLOps).** Training means many runs with different settings, say five learning
# rates for the dog-in-the-paddock model. A tracking tool records each run's settings, curves and scores so
# you can compare them later.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · Experiment-tracking tools -->
# - [Weights & Biases](https://wandb.ai/): hosted dashboards for runs, curves and scores.
# - [TensorBoard](https://www.tensorflow.org/tensorboard): training curves, runs on your own machine.
# - [Neptune.ai](https://neptune.ai/): hosted tracker for run settings and metadata.
# - [MLflow](https://mlflow.org/): open source, self-hosted tracking.

# %% [markdown]
# **Bring along:** your answers to the guiding questions of section 11 and your split
# (`data/cache/my_split.csv`, or the one you choose by hand).
