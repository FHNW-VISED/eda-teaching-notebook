# %% [markdown]
# ## 4. Labels & class distribution

# %% [markdown]
# <!-- callout: objective | Objective -->
# Count the photos of each class, see why a photo with several animals is a labelling problem, and
# understand why the dog alert cares about imbalance more than about accuracy.

# %% [markdown]
# This section looks at the **labels**, the one word per photo. Two questions decide whether they are usable:
# how many photos of each class, and is "one word per photo" the right shape for this data? Karpathy looks
# for this at the same point:

# %% [markdown]
# <!-- callout: recipe | Recipe · Step 1, continued: imbalances and biases -->
# *"Luckily, your brain is pretty good at this. One time I discovered that the data contained duplicate
# examples. Another time I found corrupted images / labels. I look for data imbalances and biases."*
#
# Andrej Karpathy, [A Recipe for Training Neural
# Networks](https://karpathy.github.io/2019/04/25/recipe/) (2019).

# %% [markdown]
# Duplicates and corrupted files are section 5's job. Here we take the last sentence: imbalances and biases.
#
# ### Images per class

# %% [markdown]
# <!-- callout: questions | Questions -->
# Predict before you run. Section 1's checklist printed the per-class counts once; now you see them as a bar
# chart, with each class's share of the dataset.

# %% [markdown]
# **Q7.** 🟢 Which class do you expect to have the most photos, which the fewest, and roughly how many times
# larger is the biggest class than the smallest? Write down your guess, then run the cell.

# %%
# @title Q7 Photos per class as bars, with each class's share written on the bar { display-mode: "form" }
images_per_class = {c: int((images["class"] == c).sum()) for c in CLASSES}
RESULTS["images_per_class"] = images_per_class

_total = sum(images_per_class.values())
_largest = max(images_per_class, key=images_per_class.get)
_smallest = min(images_per_class, key=images_per_class.get)
_ratio = images_per_class[_largest] / max(1, images_per_class[_smallest])
print(f"{_total} photos · largest class {_largest} ({images_per_class[_largest]}) · smallest {_smallest} "
      f"({images_per_class[_smallest]}) · ratio {_ratio:.1f}×")

fig, ax = plt.subplots(figsize=(9, 3.8))
_order = sorted(CLASSES, key=images_per_class.get, reverse=True)
ax.bar(_order, [images_per_class[c] for c in _order], color=[CLASS_COLORS[c] for c in _order])
for i, c in enumerate(_order):
    ax.text(i, images_per_class[c], f"{100 * images_per_class[c] / _total:.0f} %", ha="center", va="bottom",
            fontsize=9)
ax.set_ylabel("photos")
ax.set_title("Images per class (one label per photo)")
ax.set_ylim(0, max(images_per_class.values()) * 1.15)
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# 💡 **Worked answer (Q7).** On the full Play Dataset cat and dog are the largest classes (about 130 photos
# each) and sheep the smallest (about 50): roughly 2–3×. People photograph pets, not sheep; that is what
# Flickr uploads look like, not what the park's cameras will see. (The smoke sample is flat.)

# %% [markdown]
# Look at the ratio, then ask whether the smallest class is the one you care about. Dog is common in this
# dataset but will be rare in the paddock.
#
# ### Several animals in one photo: which label then?
#
# `images` has one `class` per photo, yet `n_animals` is often above 1. That is fine while all the animals
# are the same class. The Play Dataset keeps only COCO photos with **exactly one animal class**: a decision
# someone took *for* you. Check what it hides.

# %% [markdown]
# **Q8.** 🟡 What share of the photos do you think hold more than one animal, and which class do you expect to
# be the most "crowded"? Then run the cell: it counts animals per photo and shows the most crowded photos.

# %%
# @title Q8 Animals per photo, stacked by class, and the most crowded photos { display-mode: "form" }
_max_n = int(images["n_animals"].max())
_several = images[images["n_animals"] > 1]
print(f"{len(_several)} of {len(images)} photos ({100 * len(_several) / len(images):.0f} %) hold more than "
      f"one animal; the most crowded photo has {_max_n}.")
print("mean animals per photo: " + " · ".join(
    f"{c} {images.loc[images['class'] == c, 'n_animals'].mean():.1f}" for c in CLASSES))

fig, ax = plt.subplots(figsize=(9, 3.8))
_bins = np.arange(0.5, min(_max_n, 10) + 1.5, 1)
_per_class = [images.loc[images["class"] == c, "n_animals"].clip(upper=10) for c in CLASSES]
ax.hist(_per_class, bins=_bins, stacked=True, color=[CLASS_COLORS[c] for c in CLASSES], label=CLASSES)
ax.set_xticks(range(1, min(_max_n, 10) + 1))
ax.set_xticklabels([str(i) if i < 10 else "10+" for i in range(1, min(_max_n, 10) + 1)])
ax.set_xlabel("animals in the photo")
ax.set_ylabel("photos")
ax.set_title("Animals per photo")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
display(fig)
plt.close(fig)

N_CROWDED = 3 if SMOKE else 6  # how many crowded photos to show (N-014)
_crowded = images.sort_values(["n_animals", "id"], ascending=[False, True]).head(N_CROWDED)
show_rows(_crowded, "the most crowded photos", lambda r: f"{r['n_animals']} × {r['class']}", cols=N_CROWDED)

# %% [markdown]
# 💡 **Worked answer (Q8).** On the full Play Dataset about a third of the photos hold more than one animal;
# sheep and birds are the crowded classes (flocks), cats and dogs are mostly alone. One label per photo works
# only because they share a species, and it cannot say *how many*: thirty sheep and a lamb are both "sheep".

# %% [markdown]
# "Which label then?" The park picks one of three before labelling a single frame:
#
# - **one label, the main animal** (COCO's choice): cheap, but the labeller decides what "main" means;
# - **one label per animal class present**: "sheep and dog" is exactly the paddock alert, but it needs a
#   different loss and metric;
# - **one box per animal** (detection, section 8): answers "which" and "how many", at a higher cost.

# %% [markdown]
# The Play Dataset went further and dropped every photo with two classes. For the paddock camera, where the
# case that matters is a dog **among** sheep, that filter deletes exactly the frames the alert is for.

# %% [markdown]
# ### Why the dog alert cares about imbalance
#
# Imbalance is harmless in a bar chart and dangerous in a metric: the paddock camera sees sheep almost all
# the time and a dog very rarely.

# %% [markdown]
# <!-- callout: your-turn | Your turn · A paddock day -->
# `DOG_SHARE` is the fraction of the paddock camera's frames that show a dog (the rest show sheep). Keep
# `0.02` for Q9. Afterwards: edit the value (try `0.2`, then `0.5`), run this cell, then re-run the Q9 cell.

# %%
# ✏️ YOUR TURN
DOG_SHARE = 0.02

if not isinstance(DOG_SHARE, (int, float)) or not 0 <= DOG_SHARE <= 1:
    raise SystemExit(f"DOG_SHARE must be a number between 0 and 1, not {DOG_SHARE!r}")

# %% [markdown]
# **Q9.** 🟡 No model is trained here; this is a thought experiment about the *dataset*. Two lazy rules that
# ignore the photo, one question each. **Accuracy** is the share of photos answered correctly; **recall on
# dog** is, of the real dogs, how many the rule gets right. Guess both numbers for each rule, then run:
#
# - **Rule A, on the Play Dataset:** always answer *the biggest class* (read it off the Q7 bars). What are
#   its accuracy and its recall on dog?
# - **Rule B, on a paddock week:** 1000 frames, 20 of them with a dog (`DOG_SHARE = 0.02`), and the rule
#   always answers "sheep". What are its accuracy and its recall on dog?

# %%
# @title Q9 A lazy model stamps "sheep" on every frame: accuracy and dog recall { display-mode: "form" }
# Rule A on the whole Play Dataset: always the biggest class.
_majority = max(images_per_class, key=images_per_class.get)
_true = images["class"].to_numpy()
acc_play = float((_true == _majority).mean())
recall_dog_play = float(_majority == "dog") if (_true == "dog").any() else float("nan")
print(f"Rule A · Play Dataset, 'always {_majority}': accuracy {100 * acc_play:.0f} %, recall on dog "
      f"{100 * recall_dog_play:.0f} %")

# Rule B on a paddock day: PADDOCK_FRAMES frames, a share DOG_SHARE of them show a dog, the rest sheep.
PADDOCK_FRAMES = 200 if SMOKE else 1000
_n_dog = int(round(DOG_SHARE * PADDOCK_FRAMES))
acc_day = 1 - _n_dog / PADDOCK_FRAMES  # "always sheep" is right on every sheep frame
recall_dog_day = 0.0 if _n_dog else float("nan")  # and wrong on every dog frame
print(f"Rule B · paddock day, {_n_dog} dog frames in {PADDOCK_FRAMES}, 'always sheep': accuracy "
      f"{100 * acc_day:.0f} %, recall on dog {100 * recall_dog_day:.0f} %")

# The same day in miniature: 12 real photos, a share DOG_SHARE of them dogs, the rest sheep.
_N_STRIP = 12
_strip_dogs = min(int(round(DOG_SHARE * _N_STRIP)) or int(DOG_SHARE > 0), images_per_class["dog"])
_rng = random.Random(SEED)
_dog_idx = _rng.sample(sorted(images.index[images["class"] == "dog"]), _strip_dogs)
_sheep_idx = _rng.sample(sorted(images.index[images["class"] == "sheep"]),
                         min(_N_STRIP - _strip_dogs, images_per_class["sheep"]))
_strip_idx = _dog_idx + _sheep_idx
_rng.shuffle(_strip_idx)
_strip = images.loc[_strip_idx]

_cols = 6
_rows = -(-len(_strip) // _cols)
fig, axes = plt.subplots(_rows, _cols, figsize=(2.3 * _cols, 2.5 * _rows), squeeze=False)
for ax in axes.ravel():
    ax.axis("off")
for ax, (_, row) in zip(axes.ravel(), _strip.iterrows(), strict=False):
    ax.axis("on")
    ax.imshow(load_thumb(row["path"]))
    frame(ax, CLASS_COLORS[row["class"]])
    _right = row["class"] == "sheep"
    ax.text(0.04, 0.96, "sheep " + ("✔" if _right else "✘"), transform=ax.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold", color="white",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": CLASS_COLORS["sheep"], "edgecolor": "none"})
    ax.set_title(f"true: {row['class']}", fontsize=8, color="#2E7D32" if _right else "#C62828")
_strip_right = int((_strip["class"] == "sheep").sum())
fig.suptitle(f"A lazy model stamps 'sheep' on {len(_strip)} paddock frames: {_strip_right} right, "
             f"{len(_strip) - _strip_right} wrong", color=INK)
fig.tight_layout()
display(fig)
plt.close(fig)
print(f"This strip: accuracy {100 * _strip_right / len(_strip):.0f} %, recall on dog 0 %. "
      "The dog-in-the-paddock alert never fires.")

# %% [markdown]
# <!-- callout: pitfall | Pitfall · Accuracy hides a rare class -->
# **Worked answer (Q9).**
#
# - **Rule A, Play Dataset:** accuracy is the biggest class's share (about 20 % on the full set, a sixth in
#   the balanced smoke sample); recall on dog is 0 %, unless dog *is* the biggest class (then 100 % on dog,
#   0 % on everything else).
# - **Rule B, paddock week:** "always sheep" is right on 980 of 1000 frames: **98 %** accuracy and **0 %**
#   recall on dog, all 20 dogs missed.
#
# **Accuracy** is the share of all photos labelled correctly, so the big classes dominate it: the rarer the
# dog, the better "always sheep" looks. For a rare class, watch its **recall**. A balanced-looking Play
# Dataset says nothing about the imbalance of the *deployment*. Section 7 turns this into the park's metric;
# class weights and oversampling are next week's topic.

# %% [markdown]
# ### Across tasks: what a label costs
#
# The park's other two jobs need richer labels, and richer labels cost more. The cell below counts them in
# the crudest unit there is: clicks a labeller had to make.

# %% [markdown]
# <!-- callout: across-tasks | Across tasks · One label, one box, every pixel -->
# - **Classification:** one label per image. One click. It cannot say "how many" or "where".
# - **Detection:** one box per animal. Two clicks per box and a decision for every animal (is that blur a
#   bird?). Imbalance is now counted in *boxes* (section 8).
# - **Segmentation:** every pixel gets a class. One click per vertex of a polygon, many vertices per animal.
#   Imbalance is counted in *pixels*, and birds end up with a few per cent of them (section 9).
#
# The table below counts exactly these clicks on the Play Dataset. Each task costs several times more than
# the one before, which is why real datasets have many more image labels than boxes, and many more boxes
# than masks.

# %%
# @title Clicks per task: 1 per image label, 2 per box, 1 per mask vertex { display-mode: "form" }
_polygons = annotations["segmentation"].apply(lambda s: s if isinstance(s, list) else [])
_vertices = _polygons.apply(lambda polys: sum(len(p) // 2 for p in polys))
_n_crowd = int((annotations["iscrowd"] == 1).sum())
annotation_cost = pd.DataFrame(
    {
        "unit": ["image labels", "boxes", "mask vertices"],
        "count": [len(images), len(annotations), int(_vertices.sum())],
        "clicks": [len(images), 2 * len(annotations), int(_vertices.sum())],
    }
).set_index("unit")
annotation_cost["clicks per photo"] = (annotation_cost["clicks"] / len(images)).round(1)
_label_clicks = annotation_cost.loc["image labels", "clicks"]
annotation_cost["× classification"] = (annotation_cost["clicks"] / _label_clicks).round(1)
if _n_crowd:
    print(f"({_n_crowd} crowd regions are stored as pixel masks, not polygons, and count 0 vertices here.)")
display(annotation_cost)

# %% [markdown]
# 💡 The same photos, the same animals, and the click count climbs with every task. When the park asks for
# "a quick model that also counts and outlines", this table is the answer to "how quick".

# %% [markdown]
# <!-- callout: takeaway | Takeaway -->
# - **When you see** classes of very different size → **do** read the ratio between the largest and the
#   smallest, and ask whether the smallest is the class your use case depends on.
# - **When you see** photos with several animals → **do** decide between one label, one label per class or
#   one box per animal before labelling a single frame.
# - **When you see** a high accuracy and a rare class that matters → **do** compute that class's recall;
#   accuracy will happily ignore it.
# - **When you see** the Play Dataset's balance → **do** remember the deployment has its own imbalance.
#
# Section 5 goes after the labels and photos that are *wrong*, not just rare.
