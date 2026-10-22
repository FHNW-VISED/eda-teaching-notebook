# %% [markdown]
# ## 7. Metrics for the park
#
# **Objective:** pick the metric that matches the use case, and see why accuracy is the wrong one for the
# dog alert.
#
# Section 4 ended with the laziest classifier: "always sheep" scores 98 % on a paddock day and never sees
# a dog. That was a toy. This section does the same comparison on something closer to real life: a table of
# **prepared predictions**, one per photo of `clean_images`, that looks like the output of a decent
# classifier. Nothing is trained here (that is next week); the predictions are generated with a seed so that
# everyone looks at the same table, and they are built to behave like many real models on imbalanced data:
# right most of the time, and wrong mostly on the rare class.
#
# <table width="100%"><tr><td bgcolor="#E3F2FD"><font color="#0D47A1">
# <b>📘 Theory · Accuracy, recall, precision and the confusion matrix</b><br>
# A <b>confusion matrix</b> counts photos by (true class, predicted class): rows are what the photo shows,
# columns are what the classifier said, and the diagonal is the correct ones. Three numbers come out of it:
# <ul>
# <li><b>Accuracy</b> = diagonal / everything. One number for the whole table, dominated by the big
# classes.</li>
# <li><b>Recall</b> of a class = correct / <i>row</i> total: "of all the real dogs, how many did we catch?"
# A missed dog (a dog called sheep) lowers it.</li>
# <li><b>Precision</b> of a class = correct / <i>column</i> total: "of all the photos we called dog, how
# many were dogs?" A false alarm (a sheep called dog) lowers it.</li>
# </ul>
# No metric is right in itself. The use case says which mistake is expensive, and that picks the metric.
# </font></td></tr></table>
#
# ### Prepared predictions
#
# The collapsed cell writes `data/cache/park_predictions.csv`, one row per photo with the true class and a
# predicted class, and reads it back into `predictions`. Open the file if you like: it is the kind of table
# you get from any classifier's test run.

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
predictions.head()

# %% [markdown]
# <table width="100%"><tr><td bgcolor="#E0F7F4"><font color="#00564D">
# <b>❓ Questions</b><br>
# Read the first rows above: some predictions are wrong. Before computing anything, think about which
# number the park's manager would quote and which one the sheep farmer would.
# </font></td></tr></table>
#
# **Q19.** 🟡 Guess the accuracy of `predictions` (over all photos) and the recall on dog. Will the two be
# close, or far apart? Write both guesses down, then run.

# %%
# Q19  Confusion matrix of the prepared predictions, then accuracy, recall and precision per class.
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
fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.15, 1]})
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
ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
ax.set_yticks(range(len(CLASSES)), CLASSES)
for lab in ax.get_yticklabels():
    lab.set_color(CLASS_COLORS[lab.get_text()])
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title("Confusion matrix (rows = true class)")
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
# 💡 One dashed line, six bars: accuracy sits near the tall bars because it is a photo-weighted average,
# and the dog bar sits far below it. The dog row of the matrix says where the dogs went: most of the missed
# ones were called *sheep*, which on a paddock camera is exactly the frame that should have raised the alert.
#
# **Worked answer (Q19).** Accuracy is high (around 80 %) and recall on dog is around 50 %: the two are
# far apart. Accuracy is the same number the park's
# manager would be happy with, and it is reached while every second dog walks through the paddock unnoticed.
# The exact values depend on the table (dedup removed some photos in section 5), so compare the gap, not
# the digits.
#
# Instead of staring at the table, look at the mistakes that cost the most. These are the dogs the alert
# would have missed.

# %%
# The photos the alert missed: true dog, predicted something else (framed in the dog colour).
N_MISSED = 6
_missed = predictions[(predictions["true"] == "dog") & (predictions["pred"] != "dog")].head(N_MISSED)
_missed_rows = clean_images.set_index("id").loc[_missed["id"]].reset_index()
_missed_rows["pred"] = _missed["pred"].to_numpy()
show_rows(_missed_rows, f"{len(_missed)} of the {int((_true == 'dog').sum())} dogs: missed",
          caption=lambda row: f"true dog · said {row['pred']}")

# %% [markdown]
# 💡 Nothing about these photos is special: they are ordinary dogs that a classifier biased towards the big
# classes labelled as something else. Accuracy never looked at this row; recall on dog is nothing but this
# row.
#
# ### The same classes, a different shop
#
# Now move the same six classes to a **pet-food shop**. Its webcam looks at the customer's pet and the
# screen recommends food for it. Horses and zebras never walk in, so their rows of the matrix are empty and
# their recall does not matter. A false "cat" is cheap: the customer smiles, taps "no, a dog", and the screen
# corrects itself. What costs the shop is *insisting*: calling a dog a cat and recommending cat food with
# confidence, so the number to watch is **precision on cat and on dog**, "when we say cat, is it a cat?".
# Same classes, same predictions, same confusion matrix, different column: the paddock reads the dog *row*
# (recall), the shop reads the cat and dog *columns* (precision). The metric is a property of the use case,
# not of the data.

# %%
# The shop's view of the same predictions: precision on cat and dog (the columns), horse and zebra ignored.
_shop = metrics.loc[["cat", "dog"], ["precision", "recall"]]
print("Pet-food shop, same predictions:")
display(_shop.T)
print(f"The paddock quotes recall on dog ({recall['dog']:.0%}); the shop quotes precision on cat "
      f"({precision['cat']:.0%}) and dog ({precision['dog']:.0%}).")

# %% [markdown]
# 💡 The dog column is precise (few non-dogs are called dog) even though the dog row is poor: this classifier
# would be fine for the shop and useless for the paddock, on the very same test set.
#
# ### ✍️ Pick the alert metric
#
# <table width="100%"><tr><td bgcolor="#FFF4D6"><font color="#7A4A00">
# <b>✏️ Your turn · <code>ALERT_METRIC</code></b><br>
# The paddock alert fires when a dog is in front of the sheep camera. Missing a dog is worse than a false
# alarm (the park story, section 0). Replace the <code>...</code> with one of <code>"accuracy"</code>,
# <code>"precision"</code> or <code>"recall"</code>: the metric, computed on the dog class, that the alert
# should be judged on. The check cell tells you whether it agrees and continues with the reference choice
# either way, like the <code>use()</code> cells.
# </font></td></tr></table>

# %%
# ✍️ YOUR CODE
ALERT_METRIC = ...  # one of "accuracy", "precision", "recall": the number the paddock alert is judged on

# %%
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
# **Worked answer.** `"recall"`. The alert exists to catch the rare event, and recall on dog is the share
# of real dogs it catches. Precision on dog would be the shop's choice, and accuracy nobody's: it is the
# number that rewards ignoring the dog. Next week, when a real model is trained, this is the number to put
# on the validation curve, not accuracy.
#
# ### Quick check
#
# **Q20.** 🟢 The shop's webcam now also greets *rabbits* (a new, rare class). A customer with a rabbit is
# shown dog food once a week; a customer with a dog is never shown rabbit food. Which metric did the shop
# just lose, and which is still fine?
#
# <table width="100%"><tr><td bgcolor="#FFF4D6"><font color="#7A4A00">
# <b>✏️ Your turn · <code>QUIZ_ANSWER</code></b><br>
# Set <code>QUIZ_ANSWER</code> to one letter:<br>
# <b>A</b> recall on rabbit dropped, precision on dog dropped<br>
# <b>B</b> recall on rabbit dropped, precision on dog is fine<br>
# <b>C</b> precision on rabbit dropped, recall on dog dropped<br>
# <b>D</b> accuracy dropped, nothing else changed
# </font></td></tr></table>

# %%
# ✏️ YOUR TURN
QUIZ_ANSWER = None  # "A", "B", "C" or "D"

_QUIZ_KEY = "A"
_QUIZ_WHY = {
    "A": "a rabbit called dog is a miss in the rabbit row (recall on rabbit) and a wrong entry in the dog "
         "column (precision on dog). One mistake, two metrics.",
    "B": "a rabbit called dog also lands in the dog column: everything called dog that is not a dog lowers "
         "precision on dog.",
    "C": "no rabbit is predicted for a dog, so the rabbit column (precision on rabbit) and the dog row "
         "(recall on dog) are untouched.",
    "D": "accuracy barely moves (rabbits are rare), and two per-class metrics did change.",
}
if QUIZ_ANSWER is None:
    print("✏️  QUIZ_ANSWER not set yet; the answer is shown below.")
elif str(QUIZ_ANSWER).upper() not in _QUIZ_WHY:
    raise SystemExit(f"QUIZ_ANSWER must be one of A, B, C, D, not {QUIZ_ANSWER!r}")
elif str(QUIZ_ANSWER).upper() == _QUIZ_KEY:
    print(f"✅ {_QUIZ_KEY}: {_QUIZ_WHY[_QUIZ_KEY]}")
else:
    print(f"❌ {str(QUIZ_ANSWER).upper()}: {_QUIZ_WHY[str(QUIZ_ANSWER).upper()]}")

# %% [markdown]
# **Worked answer (Q20).** A. One mistake, "rabbit called dog", sits in the rabbit *row* and the dog
# *column* at once, so it lowers recall on rabbit and precision on dog. Precision on rabbit and recall on
# dog only move when a dog is called rabbit, which never happens. Accuracy hardly notices a weekly rabbit.
#
# ### Across tasks: counting and outlining
#
# The park's two other jobs have their own numbers, and the same lesson applies: the use case names the
# expensive mistake, and the metric follows.
#
# <table width="100%"><tr><td bgcolor="#EDE7F6"><font color="#311B92">
# <b>🔀 Across tasks · Counting error, mAP, IoU and Dice</b><br>
# <ul>
# <li><b>Detection (the daily headcount):</b> the park wants <b>counting error</b>, |predicted − true|
# animals per photo or per day, summed over the day. It is the number the farmer reads, and it hides
# compensating mistakes (one missed sheep plus one bush called sheep is a perfect count). The research
# metric, <b>mAP</b> (mean average precision), scores every box: a box counts as correct when it overlaps a
# true box enough (IoU ≥ 0.5), precision and recall are traced as the confidence threshold moves, the area
# under that curve is averaged over classes. Tiny far-away birds (section 8) hurt mAP most.</li>
# <li><b>Segmentation (how much of the enclosure each animal uses):</b> <b>IoU</b> (intersection over
# union) of the predicted and the true mask, per class, then the mean: the park's "share of the enclosure"
# is a pixel count, so a mask metric is the honest one. <b>Dice</b> = 2·|A∩B| / (|A|+|B|) is the same
# overlap counted more kindly (Dice ≥ IoU always). Birds are ~6 % of the pixels (section 9), so the mean
# over classes, not over pixels, is the one that notices them.</li>
# </ul>
# In all three tasks the overall number (accuracy, total count, pixel accuracy) is the one that hides the
# rare class, and the per-class number (recall on dog, per-class count error, per-class IoU) is the one
# that shows it.
# </font></td></tr></table>
#
# **Q21.** 🟡 A detector looks at a paddock photo with 5 sheep. It finds 4 of them and also calls a bush a
# sheep. What is the counting error for that photo, and how many mistakes would mAP see? Decide, then run.

# %%
# Q21  Counting error vs box-level mistakes on one photo, and IoU vs Dice of a shifted box.
true_count, found, false_boxes = 5, 4, 1
counting_error = abs((found + false_boxes) - true_count)
_box_mistakes = true_count - found + false_boxes
print(f"headcount: {found + false_boxes} predicted vs {true_count} true → counting error {counting_error}; "
      f"box-level mistakes: {true_count - found} missed + {false_boxes} false = {_box_mistakes}")


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
# 💡 The headcount is off by zero while the detector made two mistakes: a counting error is what the farmer
# wants to read, and mAP is what tells you the detector is not yet trustworthy. On the boxes, IoU and Dice
# rank the same predictions in the same order; only their scale differs, so compare numbers across papers
# with care.
#
# **Worked answer (Q21).** Counting error 0 (5 predicted, 5 true), box-level mistakes 2 (one missed sheep,
# one false box). The first number is the park's, the second is the one you fix the model with; a headcount
# that is right for the wrong reasons will be wrong on the next photo. For the sheep-paddock alert the same
# split holds: one summary number for the manager, the per-class number for the engineer.
