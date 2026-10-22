# %% [markdown]
# ## 12. Wrap-up & next week
#
# **Objective:** look back at the three things you set out to do, see the park's numbers for each, and know
# where next week's training starts.
#
# Nothing was trained today, and yet we now know a lot about whether the park's data is fit for its three
# jobs. The next cell collects the numbers the earlier sections measured, grouped by the three outcomes from
# the title banner.

# %%
# The park's numbers from today, one block per learning outcome.
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
        "(copies and broken files, before any split)",
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
#
# ### Takeaways
#
# <table width="100%"><tr><td bgcolor="#0B6E79"><font color="#FFFFFF">
# <b>You can now:</b>
# <ol>
# <li><b>Inspect an image dataset qualitatively and quantitatively, and say whether it fits a task.</b><br>
# Look at many photos before computing anything; then measure sizes, modes, colours and class counts. The
# outliers point at bugs, and a photo the park's camera would never take points at a poor fit.</li>
# <li><b>Find and fix common dataset errors before they leak into training.</b><br>
# Hash the files for exact copies, hash the pixels (and their flipped and rotated variants) for near copies,
# open every file once, and let a human judge the least typical labels and the ugliest masks. Write down
# "removed N, kept M".</li>
# <li><b>Choose a split and a metric that match the use case, and explain how imbalance, object size and
# leakage look different for classification, detection and segmentation.</b><br>
# Deduplicate, then split by whatever ties photos together (a camera, a burst). Pick the metric from the
# expensive mistake: recall on dog for the paddock alert, counting error for the headcount, IoU per class
# for the outlines. Always say what you count: photos, boxes or pixels.</li>
# </ol>
# </font></td></tr></table>
#
# ### Next week: training
#
# Next week the cleaned table and the split go into a model. Karpathy's recipe continues the same way:
# first an end-to-end skeleton with a dumb baseline, then a model that overfits, then regularisation.
# Two things from today come back straight away.
#
# **Data augmentation.** The flipped, rotated and recompressed copies you hunted in section 5 are what
# augmentation makes *on purpose*: new training views of the same photo. Augment only the training set,
# and only after the split, or the copies leak into the test set again.
# - Neptune.ai, [Data augmentation in Python](https://neptune.ai/blog/data-augmentation-in-python):
#   an overview of the techniques and libraries.
# - Ultralytics, [YOLO data augmentation](https://docs.ultralytics.com/guides/yolo-data-augmentation/):
#   what each augmentation does for detection, where boxes must move with the pixels.
#
# **Tracking experiments (MLOps).** Training means many runs with different settings; a tracking tool
# records each run's settings, curves and scores so you can compare them later.
# - [Weights & Biases](https://wandb.ai/)
# - [TensorBoard](https://www.tensorflow.org/tensorboard)
# - [Neptune.ai](https://neptune.ai/)
# - [MLflow](https://mlflow.org/)
#
# **Bring along:** your answers to the guiding questions of section 11 and your split
# (`data/cache/my_split.csv`, or the one you choose by hand).
