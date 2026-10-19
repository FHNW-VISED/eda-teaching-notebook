# %% [markdown]
# ## 4. Labels & class distribution
#
# **Objective:** count the photos of each class, see why a photo with several animals is a labelling problem,
# and understand why the dog alert cares about imbalance more than about accuracy.
#
# Sections 2 and 3 looked at the photos. This section looks at the **labels**: the one word per photo that a
# classifier will learn to say. Two questions decide whether the labels are usable at all: how many photos of
# each class do we have, and is "one word per photo" even the right shape for this data? Karpathy lists the
# things he looks for at this point:
#
# <table width="100%"><tr><td bgcolor="#E8F5E9"><font color="#1B5E20">
# <b>🧾 Recipe · Step 1, continued: imbalances and biases</b><br>
# <i>"Luckily, your brain is pretty good at this. One time I discovered that the data contained duplicate
# examples. Another time I found corrupted images / labels. I look for data imbalances and biases."</i><br>
# Andrej Karpathy, <a href="https://karpathy.github.io/2019/04/25/recipe/">A Recipe for Training Neural
# Networks</a> (2019).
# </font></td></tr></table>
#
# Duplicates and corrupted files are section 5's job. Here we take the last sentence: imbalances and biases
# in the labels.
#
# ### Images per class
#
# <table width="100%"><tr><td bgcolor="#E0F7F4"><font color="#00564D">
# <b>❓ Questions</b><br>
# Predict before you run. Section 1's checklist already printed the per-class counts once; this time you will
# see them as a bar chart, next to each class's share of the dataset.
# </font></td></tr></table>
#
# **Q7.** 🟢 Which class do you expect to have the most photos, which the fewest, and roughly how many times
# larger is the biggest class than the smallest? Write down your guess, then run the cell.

# %%
# Q7  Photos per class as bars (CLASS_COLORS), with each class's share of the dataset written on the bar.
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
# 💡 The bars are not flat: the pet classes come first because people photograph their pets, and sheep come
# last because few people photograph sheep. Nobody designed this distribution; it is what Flickr uploads look
# like, and it has nothing to do with what the park's cameras will see (sheep all day, a dog once a month).
#
# **Worked answer (Q7).** On the full Play Dataset cat and dog are the largest classes (capped at about 130
# photos each) and sheep is the smallest (about 50), so the biggest class is roughly 2–3× the smallest. In the
# small smoke sample the bars are flat, because that sample takes the same number of photos per class.
# Whatever the sample, the habit is the same: read the *ratio* between the largest and the smallest class, and
# ask whether the smallest class is the one your use case cares about. Here it is not sheep that matters, it
# is dog, and dog is well represented in the photos while being the *rare* event in the paddock.
#
# ### Several animals in one photo: which label then?
#
# `images` has one `class` per photo, and section 0 already showed that `n_animals` is often above 1. That is
# fine as long as all the animals in a photo are the same class, and it is a real problem the moment they are
# not. The Play Dataset was built by keeping only COCO photos with **exactly one animal class**, which is a
# decision someone took *for* you. Check what the decision hides.
#
# **Q8.** 🟡 What share of the photos do you think hold more than one animal, and which class do you expect to
# be the most "crowded"? Then run the cell: it counts animals per photo and shows the most crowded photos.

# %%
# Q8  Animals per photo, stacked by class, then a gallery of the photos with the most animals.
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
# 💡 "One label per photo" works here only because every animal in a photo is the same species, and even then
# the label says nothing about *how many*: a flock of thirty sheep and a single lamb are both "sheep". Look
# at the crowded photos: some also hold people, fences or a second species that COCO does not label as an
# animal class we kept, and the label is silent about all of it.
#
# **Worked answer (Q8).** On the full Play Dataset about a third of the photos hold more than one animal, and
# sheep and birds are the crowded classes (flocks), while cats and dogs are mostly alone. "Which label then?"
# has three honest answers, and the park has to pick one before labelling a single frame:
# - **one label, the main animal**: cheap, but the labeller decides what "main" means, and the model learns
#   that fuzzy rule;
# - **one label per animal class present** (multi-label): "sheep and dog" is exactly the paddock alert, but
#   it needs a different loss and metric than plain classification;
# - **one box per animal** (detection, section 8): answers "which" and "how many" at once, at a higher
#   annotation cost (the 🔀 Callout below).
#
# COCO's builders took the first answer *for the photos they kept*, and the Play Dataset went further by
# dropping every photo with two classes. For the paddock camera, where the one case that matters is a dog
# **among** sheep, that filter would delete exactly the frames the alert is for.
#
# ### Why the dog alert cares about imbalance
#
# Imbalance is harmless in a bar chart and dangerous in a metric. The park's paddock camera sees sheep almost
# all the time and a dog very rarely, and the usual score, accuracy, is blind to that.
#
# <table width="100%"><tr><td bgcolor="#FDECEA"><font color="#8B1A10">
# <b>⚠️ Pitfall · Accuracy hides a rare class</b><br>
# Accuracy is the share of all photos labelled correctly, so it is dominated by the big classes. A model that
# <b>never</b> predicts a rare class loses only that class's share of the points. If dogs are 2 % of the
# paddock frames, "always say sheep" scores 98 % accuracy and misses every dog, which is the one event the
# alert exists for. The number to watch for a rare class is its <b>recall</b>: of the real dogs, how many did
# we catch? Section 7 turns this into the park's metric.
# </font></td></tr></table>
#
# No model is needed to feel this. The next cell builds the laziest classifier there is, "always predict the
# biggest class", and scores it twice: on the Play Dataset and on a paddock day whose dog share you choose.
#
# <table width="100%"><tr><td bgcolor="#FFF4D6"><font color="#7A4A00">
# <b>✏️ Your turn · A paddock day</b><br>
# <code>DOG_SHARE</code> is the fraction of the paddock camera's frames that show a dog (the rest show sheep).
# Try <code>0.02</code>, then <code>0.2</code>, then <code>0.5</code>, and watch what happens to the accuracy
# of "always sheep" and to its recall on dog.
# </font></td></tr></table>
#
# %%
# ✏️ YOUR TURN
DOG_SHARE = 0.02

if not isinstance(DOG_SHARE, (int, float)) or not 0 <= DOG_SHARE <= 1:
    raise SystemExit(f"DOG_SHARE must be a number between 0 and 1, not {DOG_SHARE!r}")

# %% [markdown]
# **Q9.** 🟡 Before you run: what accuracy does "always predict the biggest class" get on the Play Dataset,
# and what is its recall on dog? And on a paddock day with the `DOG_SHARE` you set? Write both down, then run.

# %%
# Q9  "Always the biggest class" on the Play Dataset: every photo gets the same label.
_majority = max(images_per_class, key=images_per_class.get)
_pred = np.full(len(images), _majority)
_true = images["class"].to_numpy()
acc_play = float((_pred == _true).mean())
recall_dog_play = float((_pred[_true == "dog"] == "dog").mean()) if (_true == "dog").any() else float("nan")
print(f"Play Dataset · 'always {_majority}': accuracy {100 * acc_play:.0f} %, recall on dog "
      f"{100 * recall_dog_play:.0f} %")

# A paddock day: PADDOCK_FRAMES frames, a share DOG_SHARE of them show a dog, the rest sheep.
PADDOCK_FRAMES = 200 if SMOKE else 1000
_n_dog = int(round(DOG_SHARE * PADDOCK_FRAMES))
_day = np.array(["dog"] * _n_dog + ["sheep"] * (PADDOCK_FRAMES - _n_dog))
_always_sheep = np.full(PADDOCK_FRAMES, "sheep")
acc_day = float((_always_sheep == _day).mean())
recall_dog_day = float((_always_sheep[_day == "dog"] == "dog").mean()) if _n_dog else float("nan")
print(f"Paddock day   · {_n_dog} dog frames in {PADDOCK_FRAMES}: 'always sheep' accuracy "
      f"{100 * acc_day:.0f} %, recall on dog {100 * recall_dog_day:.0f} %")

fig, ax = plt.subplots(figsize=(9, 3.4))
_shares = np.linspace(0, 0.5, 51)
_acc_curve = (1 - _shares) * 100
ax.plot(_shares * 100, _acc_curve, color=CLASS_COLORS["sheep"], lw=2, label="accuracy of 'always sheep'")
ax.plot(_shares * 100, np.zeros_like(_shares), color=CLASS_COLORS["dog"], lw=2, label="its recall on dog")
ax.axvline(DOG_SHARE * 100, color=INK, lw=0.8, ls=":")
ax.text(DOG_SHARE * 100, 50, f" DOG_SHARE = {DOG_SHARE:g}", color=INK, fontsize=9)
ax.set_xlabel("share of paddock frames that show a dog (%)")
ax.set_ylabel("score (%)")
ax.set_ylim(-5, 105)
ax.set_title("The laziest classifier on a paddock day")
ax.legend(frameon=False, fontsize=9, loc="center right")
fig.tight_layout()
display(fig)
plt.close(fig)

# %% [markdown]
# 💡 Accuracy of the lazy classifier *is* the majority share: the rarer the dog, the better "always sheep"
# looks, and its recall on dog stays at zero the whole way. A metric that rewards ignoring the event you built
# the system for is the wrong metric.
#
# **Worked answer (Q9).** On the Play Dataset "always the biggest class" scores the biggest class's share
# (about 20 % on the full set, a sixth in the balanced smoke sample) and a recall on dog of 0 % unless dog
# *is* the biggest class, in which case it scores 100 % on dog and 0 % on everything else. On a paddock day
# with `DOG_SHARE = 0.02` it reaches 98 % accuracy with 0 % recall on dog. Two lessons: a balanced-looking
# Play Dataset says nothing about the imbalance of the *deployment* (the paddock), and whatever the imbalance,
# the score has to be computed on the class that matters. Training-time fixes for imbalance (class weights,
# oversampling, a focal loss) are next week's topic; choosing the metric is section 7's.
#
# ### Across tasks: what a label costs
#
# All of the above assumed "one word per photo". The park's other two jobs need richer labels, and richer
# labels cost more to make. The next cell counts what the Play Dataset's annotations contain, in the crudest
# unit there is: how many clicks a labeller had to make.
#
# <table width="100%"><tr><td bgcolor="#EDE7F6"><font color="#311B92">
# <b>🔀 Across tasks · One label, one box, every pixel</b><br>
# <ul>
# <li><b>Classification:</b> one label per image. One click (or one keystroke) per photo; seconds each. The
# cheapest label, and the one that cannot say "how many" or "where".</li>
# <li><b>Detection:</b> one box per animal. Two clicks per box, and a decision for every animal (is that blur
# a bird?); tens of seconds per photo, more when the photo is crowded. Imbalance is now counted in
# <i>boxes</i>, and the ranking of the classes changes (section 8).</li>
# <li><b>Segmentation:</b> every pixel gets a class. A polygon of many vertices per animal; a minute or more
# per animal, and the hardest to get right at thin parts. Imbalance is counted in <i>pixels</i>, and birds end
# up with a few per cent of them (section 9).</li>
# </ul>
# The same photo therefore costs roughly 1 : 10 : 100 to label for the three tasks, which is why real datasets
# have many more image labels than boxes, and many more boxes than masks.
# </font></td></tr></table>

# %%
# Clicks per task on the Play Dataset: 1 per image label, 2 per box, one per polygon vertex for masks.
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
# 💡 The same photos, the same animals, and the click count grows by an order of magnitude per task. When the
# park asks for "a quick model that also counts and outlines", this table is the answer to "how quick".
#
# **Takeaway.** Count the labels before you trust them: the ratio between the biggest and the smallest class,
# the share of photos with several animals, and the share of the class your use case depends on. Then
# remember that the deployment has its own imbalance, and that accuracy will happily ignore the rare class the
# park built the camera for. Section 5 goes after the labels and photos that are *wrong*, not just rare.
