# %% [markdown]
# ## 7. Metrics for the park
#
# Which number should the dog alert be judged on?

# %% [markdown]
# <!-- callout: objective | Objective -->
# Pick the metric that matches the use case, and see why accuracy is the wrong one for the dog alert.

# %% [markdown]
# "Always sheep" scored 98 % on a paddock day and never saw a dog. This section repeats the comparison on
# **prepared predictions**, shaped like a decent classifier's output: right most of the time, wrong mostly on
# the rare class. Nothing is trained here (that is next week); a seed makes everyone look at the same table.

# %% [markdown]
# <!-- callout: theory | Theory · Accuracy, recall, precision and the confusion matrix -->
# A **confusion matrix** counts photos by (true class, predicted class): rows are what the photo shows,
# columns are what the classifier said, the diagonal is the correct ones. Three numbers come out of it:
#
# - **Accuracy** = correct photos / all photos. One number, dominated by the big classes.
# - **Recall** of a class = correct / its **row** total: "of all the real dogs, how many did we catch?" A dog
#   called sheep lowers it.
# - **Precision** of a class = correct / its **column** total: "of all the photos we called dog, how many
#   were dogs?" A sheep called dog lowers it.
#
# No metric is right in itself: the use case says which mistake is expensive.

# %% [markdown]
# Two contestants on a paddock week of 1000 frames, 20 with a dog: **always sheep** and a **decent
# classifier**. The next cell draws their 2 × 2 confusion matrices.

# %%
# @title Toy: a paddock week of 1000 frames, 'always sheep' vs a decent classifier { display-mode: "form" }
_toy_labels = ["sheep", "dog"]
_toy = {  # rows: true sheep, true dog · columns: said sheep, said dog
    "always sheep": np.array([[980, 0], [20, 0]]),
    "a decent classifier": np.array([[970, 10], [6, 14]]),
}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, (name, m) in zip(axes, _toy.items(), strict=True):
    _acc = np.trace(m) / m.sum()
    _rec = m[1, 1] / m[1].sum()
    _prec = f"{m[1, 1] / m[:, 1].sum():.0%}" if m[:, 1].sum() else "n/a (never says dog)"
    for i in range(2):
        for j in range(2):
            ax.add_patch(mpl.patches.Rectangle((j, i), 1, 1, facecolor=CLASS_COLORS[_toy_labels[i]],
                                               alpha=0.65 if i == j else 0.25, edgecolor="white"))
            ax.text(j + 0.5, i + 0.5, str(m[i, j]), ha="center", va="center", fontsize=14, color=INK)
    ax.add_patch(mpl.patches.Rectangle((0, 1), 2, 1, fill=False, edgecolor=INK, lw=3))
    ax.add_patch(mpl.patches.Rectangle((1, 0), 1, 2, fill=False, edgecolor=INK, lw=2, linestyle="--"))
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_xticks([0.5, 1.5], ["said sheep", "said dog"])
    ax.set_yticks([0.5, 1.5], ["true sheep", "true dog"])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("solid row = recall on dog · dashed column = precision on dog", fontsize=8)
    ax.set_title(f"{name}\naccuracy {_acc:.1%} · recall {_rec:.0%} · precision {_prec}",
                 fontsize=9, color=INK)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# 💡 The two contestants differ by 0.4 points of accuracy. One catches 14 of 20 dogs, the other catches none.

# %% [markdown]
# ### Prepared predictions
#
# The next cell writes `data/cache/park_predictions.csv` (one row per photo: the true class and a predicted
# class) and reads it back as `predictions`. Any classifier's test run gives you a table like it.
#
# The prepared classifier is wrong on 1 photo in 10 for every class except dog, where it is wrong on 1 in 2,
# mostly calling the dog a sheep.

# %%
# @title Prepared predictions { display-mode: "form" }
# Fake classifier output for clean_images, fixed by SEED (no model): each class gets a share of wrong
# predictions, dog a large one, and each wrong prediction goes to a plausible look-alike class.
PRED_PATH = DATA_DIR / "cache" / "park_predictions.csv"
ERROR_SHARE = {"cat": 0.10, "dog": 0.50, "bird": 0.10, "horse": 0.10, "sheep": 0.10, "zebra": 0.10}
CONFUSED_WITH = {  # when the prediction is wrong, which class it tends to say
    "cat": ["dog"],
    "dog": ["sheep", "sheep", "sheep", "cat"],
    "bird": ["cat"],
    "horse": ["sheep"],
    "sheep": ["horse"],
    "zebra": ["horse"],
}


def generate_predictions(table: pd.DataFrame, seed: int) -> pd.DataFrame:
    """A DataFrame (id, file, true, pred) with ERROR_SHARE of each class predicted as a look-alike class."""
    rng = np.random.default_rng(seed)
    true = table["class"].to_numpy()
    pred = true.copy()
    for c in CLASSES:
        idx = np.flatnonzero(true == c)
        n_wrong = int(round(len(idx) * ERROR_SHARE[c]))
        wrong = rng.choice(idx, n_wrong, replace=False)
        pred[wrong] = rng.choice(CONFUSED_WITH[c], n_wrong)
    return pd.DataFrame({"id": table["id"].to_numpy(), "file": table["file"].to_numpy(),
                         "true": true, "pred": pred})


PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
generate_predictions(clean_images, SEED).to_csv(PRED_PATH, index=False)
predictions = pd.read_csv(PRED_PATH)
print(f"{len(predictions)} predictions written to {PRED_PATH} and read back")
print("photos per true class: " + " · ".join(f"{c} {int((predictions['true'] == c).sum())}" for c in CLASSES))
predictions.head()

# %% [markdown]
# <!-- callout: questions -->
# Which number would the park manager quote in a report, and which one should the dog alert be judged on?

# %% [markdown]
# **Q17.** 🟡 From the error rates above and the photos per class printed, guess the accuracy of
# `predictions` and the recall on dog. Close, or far apart? Write both guesses down, then run.

# %%
# @title Q17 Confusion matrix, then accuracy, recall and precision per class { display-mode: "form" }
_true = predictions["true"].to_numpy()
_pred = predictions["pred"].to_numpy()
confusion = pd.DataFrame(0, index=CLASSES, columns=CLASSES, dtype=int)
for t, p in zip(_true, _pred, strict=True):
    confusion.loc[t, p] += 1
confusion.index.name, confusion.columns.name = "true", "predicted"

_diag = np.diag(confusion.to_numpy())
accuracy = float(_diag.sum() / len(predictions))
recall = pd.Series(_diag / confusion.sum(axis=1).replace(0, np.nan).to_numpy(), index=CLASSES)
precision = pd.Series(_diag / confusion.sum(axis=0).replace(0, np.nan).to_numpy(), index=CLASSES)
metrics = pd.DataFrame({"photos": confusion.sum(axis=1), "recall": recall.round(2),
                        "precision": precision.round(2)})
RESULTS["accuracy"] = accuracy
RESULTS["recall_dog"] = float(recall["dog"])
print(f"accuracy: {accuracy:.0%}   ·   recall on dog: {recall['dog']:.0%}   ·   precision on dog: "
      f"{precision['dog']:.0%}")
display(metrics.T)

# The confusion matrix, each row tinted in the true class's colour (darker = a larger share of that row).
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
_rgba = np.zeros((len(CLASSES), len(CLASSES), 4))
_row_share = confusion.to_numpy() / confusion.sum(axis=1).replace(0, 1).to_numpy()[:, None]
for i, c in enumerate(CLASSES):
    _rgba[i, :, :3] = mpl.colors.to_rgb(CLASS_COLORS[c])
    _rgba[i, :, 3] = 0.1 + 0.9 * _row_share[i]
ax.imshow(_rgba)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax.text(j, i, str(confusion.iat[i, j]), ha="center", va="center", fontsize=9,
                color="white" if _row_share[i, j] > 0.5 else INK)
_dog = CLASSES.index("dog")
ax.add_patch(mpl.patches.Rectangle((-0.5, _dog - 0.5), len(CLASSES), 1, fill=False, edgecolor=INK, lw=2.5))
ax.add_patch(mpl.patches.Rectangle((_dog - 0.5, -0.5), 1, len(CLASSES), fill=False, edgecolor=INK, lw=2,
                                   linestyle="--"))
ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
ax.set_yticks(range(len(CLASSES)), CLASSES)
for lab in ax.get_yticklabels():
    lab.set_color(CLASS_COLORS[lab.get_text()])
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title("Confusion matrix (rows = true class)\nsolid: dog row = recall · dashed: dog column = precision",
             fontsize=9)
ax.grid(False)

ax = axes[1]
ax.bar(CLASSES, recall.to_numpy(), color=[CLASS_COLORS[c] for c in CLASSES])
ax.axhline(accuracy, color=INK, linestyle="--", linewidth=1, label=f"accuracy {accuracy:.0%}")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
ax.set_ylabel("recall per class")
ax.set_title("Recall per class vs overall accuracy")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# <!-- callout: answer | Worked answer (Q17) -->
# Far apart: accuracy around 80 %, recall on dog around 50 %. Accuracy is a
# photo-weighted average, so it sits near the tall bars while the dog bar sits far below; most missed dogs
# were called *sheep*, exactly the frame that should have raised the alert. The manager is happy while every
# second dog walks by. Compare the gap, not the digits.

# %% [markdown]
# ### The mistakes behind the numbers
#
# - **Recall reads the dog row:** the real dogs the alert missed.
# - **Precision reads the dog column:** photos called dog that are not dogs (false alarms).
#
# The next cell shows up to six of each. The missed dogs are ordinary dogs that a classifier biased towards
# the big classes labelled as something else: recall on dog is nothing but the first gallery.

# %%
# @title Dogs the alert missed (recall) and false alarms (precision) { display-mode: "form" }
N_SHOWN = 6
_missed = predictions[(predictions["true"] == "dog") & (predictions["pred"] != "dog")].head(N_SHOWN)
_missed_rows = clean_images.set_index("id").loc[_missed["id"]].reset_index()
_missed_rows["pred"] = _missed["pred"].to_numpy()
show_rows(_missed_rows, f"Recall: {len(_missed)} of the {int((_true == 'dog').sum())} dogs, missed",
          caption=lambda row: f"true dog · said {row['pred']}")

_alarms = predictions[(predictions["true"] != "dog") & (predictions["pred"] == "dog")].head(N_SHOWN)
_alarm_rows = clean_images.set_index("id").loc[_alarms["id"]].reset_index()
_alarm_rows["pred"] = _alarms["pred"].to_numpy()
_n_alarms = int(((_true != "dog") & (_pred == "dog")).sum())
show_rows(_alarm_rows, f"Precision: {len(_alarms)} of the {_n_alarms} photos called dog that are not dogs",
          caption=lambda row: f"true {row['class']} · said dog")

# %% [markdown]
# ### The same classes, a different shop
#
# Move the same six classes to a **pet-food shop**, whose webcam looks at the customer's pet and recommends
# food. Horses and zebras never walk in, and a false "cat" is cheap; what costs the shop is *insisting*,
# calling a dog a cat with confidence. Its question is "when we say cat, is it a cat?", which is
# **precision**: on these predictions about 82 % on cat and 83 % on dog, while the paddock's recall on dog
# is 50 %.
#
# Same matrix, different slice: the paddock reads the dog *row*, the shop the cat and dog *columns*. The
# metric belongs to the use case, not to the data.
#
# <details><summary>More on the shop</summary>
#
# A false "cat" is cheap because the customer taps "no, a dog". A classifier can be good enough for one
# question and useless for the other, on the very same test set.
#
# </details>

# %% [markdown]
# ### ✍️ Pick the alert metric

# %% [markdown]
# <!-- callout: your-turn | Your turn · `ALERT_METRIC` -->
# The paddock alert fires when a dog is in front of the sheep camera. Missing a dog is worse than a false
# alarm (the park story, section 0).
#
# - **Do:** replace the `...` with `"accuracy"`, `"precision"` or `"recall"`: the metric, on the dog class,
#   the alert should be judged on.
# - **Then:** the check cell says whether it agrees and continues with the reference either way.

# %%
# ✍️ YOUR CODE
ALERT_METRIC = ...  # one of "accuracy", "precision", "recall": the number the paddock alert is judged on

# %%
# @title Check ALERT_METRIC against the reference choice { display-mode: "form" }
# Check ALERT_METRIC against the reference choice; the notebook continues with the reference either way.
_REFERENCE_ALERT_METRIC = "recall"
_WHY = {
    "accuracy": "accuracy is dominated by the big classes: 'always sheep' scores 98 % on a paddock day and "
                "catches no dog (section 4).",
    "precision": "precision on dog counts false alarms. A false alarm costs the farmer a look at the screen; "
                 "a missed dog costs sheep.",
}
_dog_scores = {"accuracy": accuracy, "precision": float(precision["dog"]), "recall": float(recall["dog"])}
if ALERT_METRIC is ...:
    print("✍️  ALERT_METRIC: not chosen yet, using the reference choice.")
elif ALERT_METRIC not in _dog_scores:
    raise SystemExit(f"ALERT_METRIC must be one of {sorted(_dog_scores)}, not {ALERT_METRIC!r}")
elif ALERT_METRIC == _REFERENCE_ALERT_METRIC:
    print(f"✅ ALERT_METRIC = {ALERT_METRIC!r} matches the reference choice.")
else:
    print(f"❌ ALERT_METRIC = {ALERT_METRIC!r}: {_WHY[ALERT_METRIC]} Using the reference.")
alert_metric = _REFERENCE_ALERT_METRIC
RESULTS["alert_metric"] = alert_metric
print(f"The paddock alert is judged on {alert_metric} on dog: {_dog_scores[alert_metric]:.0%} on the "
      f"prepared predictions (accuracy would have said {accuracy:.0%}).")

# %% [markdown]
# <!-- callout: answer | Worked answer -->
# `"recall"`.
#
# - The alert exists to catch the rare event, and recall on dog is the share of real dogs it catches.
# - Precision on dog would be the shop's choice; accuracy is nobody's, because it rewards ignoring the dog.
# - Next week, when a real model is trained, this is the number to put on the validation curve.

# %% [markdown]
# ### Across tasks: counting and outlining
#
# The park's two other jobs have their own numbers, and the lesson is the same: the use case names the
# expensive mistake, and the metric follows.

# %% [markdown]
# <!-- callout: across-tasks | Across tasks · Counting error, IoU and Dice -->
# - **Detection, the daily headcount:** **counting error** = |predicted count − true count| for a photo or a
#   day. It is the number the farmer reads, and it hides compensating mistakes: one missed sheep plus one bush
#   called sheep is a perfect count.
# - **Segmentation, how much of the enclosure each animal uses:** **IoU** = overlap area / union area of the
#   predicted and the true mask, per class, then averaged. **Dice** = 2 × overlap / (area A + area B) is the
#   same overlap counted more kindly (Dice ≥ IoU always). Birds cover few pixels (section 9), so only the
#   average over classes, not over pixels, notices them.
#
# In all three tasks the overall number (accuracy, total count, pixel accuracy) hides the rare class; the
# per-class number (recall on dog, per-class count error, per-class IoU) shows it.

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · mAP, the research metric for boxes -->
# <details><summary>How mAP scores every box</summary>
#
# **mAP** (mean average precision) scores every box instead of the headcount:
#
# 1. A box counts as correct when it overlaps a true box enough (IoU ≥ 0.5).
# 2. Precision and recall are traced as the confidence threshold moves; the area under that curve is the
#    class's *average precision*.
# 3. Averaged over classes, that is mAP. Tiny far-away birds (section 8) hurt it most.
#
# </details>

# %% [markdown]
# <!-- callout: go-deeper | Go deeper · IoU and Dice on a shifted box -->
# IoU and Dice score how well a predicted shape sits on the true one. On boxes, the next cell slides a
# predicted box away from the true one: the two numbers rank the same predictions in the same order, only
# their scale differs (Dice is always the kinder one), so compare numbers across papers with care. (The
# counting error is the number the farmer reads; it can be perfect while individual boxes are wrong.)

# %%
# @title IoU vs Dice of a shifted box (defines box_iou_dice, used in 8 and 9) { display-mode: "form" }
def box_iou_dice(a: tuple, b: tuple) -> tuple[float, float]:
    """IoU and Dice of two boxes given as (x, y, w, h)."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a, area_b = a[2] * a[3], b[2] * b[3]
    return inter / (area_a + area_b - inter), 2 * inter / (area_a + area_b)


_true_box = (20, 20, 60, 40)
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, shift in zip(axes, [0, 15, 30], strict=True):
    _pred_box = (_true_box[0] + shift, _true_box[1] + shift // 2, _true_box[2], _true_box[3])
    iou, dice = box_iou_dice(_true_box, _pred_box)
    ax.add_patch(mpl.patches.Rectangle(_true_box[:2], *_true_box[2:], fill=True, alpha=0.35,
                                       color=CLASS_COLORS["sheep"], label="true"))
    ax.add_patch(mpl.patches.Rectangle(_pred_box[:2], *_pred_box[2:], fill=False, lw=2,
                                       edgecolor=CLASS_COLORS["dog"], label="predicted"))
    ax.set_xlim(0, 120)
    ax.set_ylim(90, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(f"shift {shift} px · IoU {iou:.2f} · Dice {dice:.2f}", fontsize=9)
axes[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("The same overlap, two scores: Dice is always the kinder one", color=INK)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **When you see** one rare class the system exists for (the dog in the paddock) → **do** judge it on that
#   class's **recall** (its row), never on accuracy.
# - **When you see** a cost for false alarms (the shop's wrong food) → **do** judge on **precision** (the
#   column).
# - **When you see** counting or outlining tasks → **do** read per-class counting error and per-class IoU, not
#   the overall total.
# - **When you see** a high overall number → **do** open the per-class row before you believe it.
