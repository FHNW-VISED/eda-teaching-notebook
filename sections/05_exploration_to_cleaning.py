# %% [markdown]
# ## From Data Exploration to Cleaning
#
# Before training a model, potential data quality issues should be reviewed to decide which images to keep, correct, or remove. Unreadable images and confirmed duplicates are candidates for removal, while near-duplicates and unusual images require closer inspection.
#

# %% [markdown]
# ### 🔴 Data Quality: Which Issues Should Be Addressed Before Training? (Optional)
#
# **Reflect and justify:**
# - Which images would you remove from this dataset, and why?
# - Why might automatically removing unusual images be problematic?
# - Why should duplicates be reviewed before splitting the dataset?
#
# > ⚠️ **Pitfall: Data Leakage**
# >
# > If duplicate or near-duplicate images appear in both training and test sets, performance estimates may be overly optimistic. Review duplicates **before splitting** and ensure related images do not end up in different splits.

# %% edition="teaching"
# @title 💡 Sample Answer { display-mode: "form" }
callout("answer", "💡 Sample Answer", r"""
<ul>
  <li><strong>Cleaning decisions:</strong> Remove unreadable images and
  confirmed duplicates, keeping one image per duplicate group.
  Review potential near-duplicates and suspicious labels before deciding.
  Unusual images are not necessarily incorrect.</li>

  <li><strong>Removing unusual images:</strong> Automatically excluding
  unusual images can reduce dataset diversity and introduce selection bias.
  These images may represent valid but less common conditions that the
  model should learn to handle.</li>

  <li><strong>Data leakage:</strong> Duplicate or near-duplicate images
  appearing in both training and test sets can produce overly optimistic
  performance estimates. Duplicate groups should therefore be reviewed
  before splitting and kept within the same split if retained.</li>
</ul>

<p><strong>Key takeaway:</strong> Dataset cleaning requires justified
decisions, not automatic removal of every unusual or similar image.</p>
""")


# %% [markdown] edition="own"
# ✍️ **Your answer** (double-click to edit):
#
# *In the park:* the plan was to remove unreadable images and confirmed duplicates, keeping one image per group, to review near-duplicates and suspicious labels, and to keep unusual images.

# %%
# @title 📘 Image Dataset Quality Checklist { display-mode: "form" }
callout("theory", "📘 Image Dataset Quality Checklist", r"""
<p style="margin:6px 0;">Before using an image dataset for computer vision, check:</p>

<div style="font-weight:bold;margin:10px 0 4px 0;">Qualitative Inspection</div>
<ul style="list-style:none;margin:0;padding-left:4px;">
<li style="margin:5px 0;">☐ <strong>Use-case fit:</strong> Do the images represent the conditions in which the model will be used?</li>
<li style="margin:5px 0;">☐ <strong>Visual variation:</strong> Do the images cover relevant differences in backgrounds, viewpoints, lighting, and object sizes?</li>
<li style="margin:5px 0;">☐ <strong>Recognizability:</strong> Are the relevant objects visible and distinguishable?</li>
<li style="margin:5px 0;">☐ <strong>Annotation availability:</strong> Are the labels, bounding boxes, or masks needed for the task available?</li>
<li style="margin:5px 0;">☐ <strong>Annotation quality:</strong> Are annotations accurate, complete, and consistent across images and objects?</li>
<li style="margin:5px 0;">☐ <strong>Label relevance:</strong> Are class definitions clear, and do assigned labels reflect the relevant image content?</li>
</ul>

<div style="font-weight:bold;margin:12px 0 4px 0;">Quantitative Inspection</div>
<ul style="list-style:none;margin:0;padding-left:4px;">
<li style="margin:5px 0;">☐ <strong>Dataset size:</strong> Are there enough images and annotations for the intended task?</li>
<li style="margin:5px 0;">☐ <strong>Class distribution:</strong> Are all classes sufficiently represented, or are some underrepresented?</li>
<li style="margin:5px 0;">☐ <strong>Image properties:</strong> Are image dimensions, resolutions, and aspect ratios suitable for the task?</li>
<li style="margin:5px 0;">☐ <strong>Image integrity:</strong> Can all image files be opened and processed correctly?</li>
<li style="margin:5px 0;">☐ <strong>Outliers:</strong> Are unusual images identified using different visual features and inspected before deciding whether to keep or remove them?</li>
<li style="margin:5px 0;">☐ <strong>Duplicates and data leakage:</strong> Are exact and potential near-duplicates reviewed, and are related images kept within the same data split?</li>
</ul>
""")


