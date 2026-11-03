# FireNet-CNN: reproduction, comparison and a thermal-teacher severity extension

This repository reproduces **"Real-Time Detection of Forest Fires Using FireNet-CNN and Explainable AI Techniques"** (G. M. I. Alam, N. Tasnia, T. Biswas, M. J. Hossen, S. A. Tanim, M. S. U. Miah, *IEEE Access*, vol. 13, pp. 51150-51181, 2025, doi:10.1109/ACCESS.2025.3552352) and extends it.

The work has three stages:

| Stage | What it does | Headline result |
|---|---|---|
| **Phase 1** | Rebuild FireNet-CNN and train it on a smaller 4-class dataset (4,800 images) mapped to fire / no fire | 94.88% accuracy (5-fold mean) |
| **Phase 2** | Retrain on the paper's own Dataset A and Dataset B plus Stable Diffusion synthetic images | 97.18% (A) and 98.90% (B) accuracy |
| **Research gap and extension** | Add a Low / Moderate / High **severity** stage that learns from thermal-derived labels and needs only RGB at test time | 61.6% accuracy on a preliminary 125-image split (preliminary, one burn) |

> **Status:** Phase 1 and Phase 2 are complete runs with saved results (Phase 2 strict protocol on Dataset B is the one exception, see below). The severity extension is a **preliminary** result from one prescribed burn, one random image-level split and a single run. It shows the task is learnable, not that it generalizes.

---

## 1. The paper in brief

- **Task:** real-time forest fire detection as binary classification (fire vs no fire), aimed at resource-limited platforms such as drones and embedded sensors.
- **Model:** FireNet-CNN, five blocks of 3x3 convolution, ReLU, 2x2 max-pooling and batch normalization (32, 64, 128, 256, 512 filters), then Dense 512, Dropout 0.5, Dense 256, Dropout 0.5 and one sigmoid output. **2,753,217 parameters, about 10.6 MB.**
- **Data:** two fire / non-fire image datasets (A and B) expanded with Stable Diffusion synthetic images, evaluated with 5-fold cross-validation.
- **Explainability:** Grad-CAM and saliency maps.
- **Reported result:** up to 99.05% accuracy, 99.41% precision and 98.28% recall (Dataset B), compared with VGG16, VGG19 and InceptionV3.

### Published results used as the reference (percent)

| Dataset | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| A | **FireNet-CNN** | 97.96 | 98.22 | 98.41 | 98.31 |
| A | VGG16 | 95.30 | 95.51 | 95.79 | 95.65 |
| A | VGG19 | 95.24 | 95.80 | 95.64 | 95.72 |
| A | InceptionV3 | 97.53 | 98.13 | 97.58 | 97.86 |
| B | **FireNet-CNN** | 99.05 | 99.41 | 98.28 | 98.84 |
| B | VGG16 | 97.51 | 96.92 | 97.04 | 96.98 |
| B | VGG19 | 97.93 | 97.28 | 97.57 | 97.42 |
| B | InceptionV3 | 98.88 | 98.96 | 98.54 | 98.75 |

The paper also reports AUC 0.9980 (A) and 0.9992 (B), 10.58 MB model size and 0.95 s inference time.

---

## 2. Repository layout

```
Trial_with_new_data/
  00_setup_and_data.ipynb        writes the shared firenet module and caches the data
  02_train_firenet_cv.ipynb      Phase 2: 5-fold CV, paper and strict protocols
  05_results_vs_paper.ipynb      compares our runs with the published tables
                                 (notebooks 00-05 hold the reproduction)
  inspect_dataset.py             pairs corrected-FOV RGB with Celsius TIFFs, checks counts and shapes
  thermal_summary.py             per-image thermal statistics -> thermal_summary.csv
  check_severity_distribution.py summarizes thermal feature distributions of fire images
  flame3_quick_experiment.py     builds labels, trains the RGB-only severity model, Grad-CAM
  flame3_severity.py             grouped 5-fold pipeline (implemented, not run)
  phase2_label_analysis.py       label sensitivity analysis (section 5.6)
  flame_severity/flame_severity.py   alternative module, not used for reported results
  quick_severity_results/        metrics, confusion matrix, training history, Grad-CAM image
Phase 1 notebooks
  FireNet_CNN_Reproduction.ipynb Phase 1 reproduction on the 4-class dataset
  gradcam_saliency.ipynb         Phase 1 Grad-CAM, saliency maps and quick ablation
```

## 3. Environment

```bash
git clone https://github.com/DivyaS09122006/FireNetCNN_XAI.git
cd FireNetCNN_XAI/Trial_with_new_data
pip install tensorflow pandas numpy scikit-learn opencv-python tifffile matplotlib scipy seaborn
```

- **Phase 1** ran on TensorFlow 2.20 with Keras 3.13 in Google Colab.
- **Phase 2** ran on Colab with `TF_USE_LEGACY_KERAS=1` and the `tf_keras` package (TensorFlow 2.21), because the shared `firenet` module is written in Keras 2 style.
- A GPU is required for training. Without one, a single epoch of the larger models becomes impractically slow.
- Seed 42 is used throughout (Python, NumPy, TensorFlow, and every `StratifiedKFold` / `train_test_split`).

---

## 4. Phase 1 - reproduction on a smaller 4-class dataset

**Goal:** rebuild FireNet-CNN from scratch and see how close a smaller, different dataset gets to the paper.

### Data

- 4,800 images from a single source, **1,200 per class**: `fire`, `smokefire` (smoke with flames), `smoke` (smoke only), `nofire`.
- Mapped to binary: **fire = fire + smokefire**, **no fire = smoke + nofire** (2,400 vs 2,400).
- The original train / val / test folders were merged into one pool, then re-split with **5-fold stratified cross-validation** (3,840 train / 960 validation per fold).
- Images resized to 150x150x3 and scaled to 0-1. The metadata shows a single source and no video IDs, and still-image filenames, so random folds were used.
- **No synthetic images** were used.

### Model and training

- `build_firenet()` gives exactly **2,753,217 parameters**, matching the paper. This is the architecture check.
- Adam (lr 1e-4), binary cross-entropy, batch size 32, 50 epochs.
- Augmentation on training folds only: horizontal flip, rotation (about 20 degrees), shift and zoom, using a `tf.data` pipeline. Shear and brightness changes from the paper are not used.
- Results per fold are saved to Drive as `fold{k}.pkl`, with saved models.

### Results (5-fold mean +/- std)

| Metric | Phase 1 | Paper (Dataset A) | Difference |
|---|---|---|---|
| Accuracy | 94.88 +/- 1.06 | 97.96 | -3.08 |
| Precision | 96.79 +/- 0.82 | 98.22 | -1.43 |
| Recall | 92.83 +/- 1.94 | 98.41 | -5.58 |
| F1 | 94.76 +/- 1.12 | 98.31 | -3.55 |
| Jaccard | 90.06 +/- 2.02 | 96.68 | -6.62 |
| MCC | 89.84 +/- 2.05 | 96.45 | -6.61 |
| AUC | 0.9909 +/- 0.0027 | 0.9980 | -0.0071 |

Training took about 1,590 s per fold.

### Where the model fails

Error rate per original class (each image is validated once across the five folds):

| Class | Images | Errors | Error rate |
|---|---|---|---|
| fire | 1,200 | 38 | 3.17% |
| nofire | 1,200 | 41 | 3.42% |
| smoke | 1,200 | 33 | 2.75% |
| **smokefire** | 1,200 | **134** | **11.17%** |

`smokefire` has three to four times the error rate of any other class. The hard cases are smoke with flames versus smoke alone, which explains the low recall and most of the gap to the paper.

### Grad-CAM, saliency and ablation (`gradcam_saliency.ipynb`)

- Grad-CAM and saliency maps were generated for the fold-1 model on 8 test images (two per class), using the last convolutional layer (5x5x512). Fire and no-fire images were all correct; **all 3 misclassifications were smoke or smokefire images.**
- **Quick ablation** (a single 80/20 split of the `train` folder only, 3,202 images, 10 epochs, so not comparable to the main protocol):

| Variant | Train acc | Val acc | Precision | Recall | F1 |
|---|---|---|---|---|---|
| No convolution | 83.91 | 88.77 | 94.62 | 82.24 | 88.00 |
| LeakyReLU | 94.06 | 95.16 | 97.39 | 92.83 | 95.06 |
| Sigmoid activation | 92.82 | 93.76 | 95.47 | 91.90 | 93.65 |
| No dropout | 98.40 | 94.70 | 97.36 | 91.90 | 94.55 |
| No batch norm | 94.53 | 95.32 | 95.33 | 95.33 | 95.33 |
| SGD optimizer | 74.46 | 50.08 | 50.08 | 100.00 | 66.74 |
| RMSProp optimizer | 95.43 | 94.23 | 96.71 | 91.59 | 94.08 |
| MSE loss | 91.60 | 50.23 | 50.16 | 100.00 | 66.81 |

Removing the convolutional layers hurts most, as in the paper. The SGD and MSE runs collapsed to predicting "fire" for every image, probably because 10 epochs on a small split were not enough for them to converge. This table has no full-model reference row.

### Not completed

The VGG16, VGG19 and InceptionV3 baselines were set up but **not completed**: the Colab GPU quota ran out and the first run was interrupted. The baseline numbers anywhere in this repository are the **paper's**, not ours.

---

## 5. Phase 2 - the paper's datasets plus synthetic images

**Goal:** follow the paper's data setup more closely and see how much of the Phase 1 gap closes.

### Data

| | Dataset A | Dataset B |
|---|---|---|
| Content | forest fire / fire-free forest images | 3,400 frames from 25 videos, several seasons, day and night |
| Synthetic images added | Stable Diffusion, 1,020 fire and 1,181 non-fire | same |
| Size after augmentation (paper) | 5,148 | 5,709 |

Our Dataset B folds total 4,567 + 1,142 = 5,709 images, matching the paper. A and B are trained **separately**, never merged.

### Two evaluation protocols

- **`paper`:** synthetic images are pooled with the real ones *before* the 5-fold split, as the paper describes.
- **`strict`:** folds are built from **real images only**, and all synthetic images are added to the training side only. This is the leakage-free number.

### Setup

Same architecture (assertion on 2,753,217 parameters), Adam 1e-4, binary cross-entropy, batch 32, **50 epochs**, 5-fold cross-validation, seed 42. Run notebook `02_train_firenet_cv.ipynb`; results are checkpointed after each completed run, so a disconnected session resumes where it stopped.

### Results (5-fold means, percent)

| Metric | A, paper split | A, strict split | Paper A | B, paper split | Paper B |
|---|---|---|---|---|---|
| Accuracy | 97.18 | 97.25 | 97.96 | 98.90 | 99.05 |
| Precision | 97.21 | 97.45 | 98.22 | 98.86 | 99.41 |
| Recall | 97.85 | 97.73 | 98.41 | 98.16 | 98.28 |
| F1 | 97.52 | 97.58 | 98.31 | 98.51 | 98.84 |
| Jaccard | 95.18 | 95.29 | 96.68 | 97.07 | 97.71 |
| MCC | 94.27 | 94.41 | 96.45 | 97.64 | 98.19 |
| AUC | 0.9962 | 0.9965 | 0.9980 | 0.9989 | 0.9992 |

On Dataset B, per-fold accuracy ranged from 98.51% to 99.39%. Training took about 830 s per fold on A and about 1,650 s per fold on B.

### Reading the results

- Accuracy is within **0.78 points** of the paper on Dataset A and **0.15 points** on Dataset B, but still **slightly below it on every metric**.
- Recall rose from 92.83% (Phase 1) to about 98% on both datasets. For fire detection this matters most, because a missed fire is the costliest error.
- On Dataset A the strict run (97.25%) matches the paper-protocol run (97.18%), so pooling synthetic images before the split did **not** inflate the validation score there.

### Not completed or not tested

- **`strict` on Dataset B did not finish** (interrupted at epoch 17 of fold 1). No result is reported.
- The Phase 2 improvement **cannot be credited to the synthetic images alone**, because the data also changed. A `no_synth` run would isolate it and has not been done.
- Dataset B frames come from videos and folds are random, so near-identical frames can sit in both training and validation. Neither the paper nor this reproduction controls for that.
- Single seed per run. No VGG/Inception reproduction on the new data.

### All stages side by side (percent)

| Metric | Phase 1 | Phase 2, A | Phase 2, B | Paper A | Paper B |
|---|---|---|---|---|---|
| Accuracy | 94.88 | 97.18 | 98.90 | 97.96 | 99.05 |
| Precision | 96.79 | 97.21 | 98.86 | 98.22 | 99.41 |
| Recall | 92.83 | 97.85 | 98.16 | 98.41 | 98.28 |
| F1 | 94.76 | 97.52 | 98.51 | 98.31 | 98.84 |
| AUC | 0.9909 | 0.9962 | 0.9989 | 0.9980 | 0.9992 |

---

## 6. Research gap and the severity extension

### 6.1 The gap

FireNet-CNN says **whether** a fire is present, not **how severe** it is. Its single sigmoid output cannot separate a small smouldering patch from a large, intense fire, so two very different alerts look identical. That matters when several drones report at once and responders must decide which alert to act on first.

The authors also acknowledge two weaknesses of their own: reliance on synthetic data and class imbalance.

### 6.2 Evidence from thermal data

Using the FLAME 3 CV subset (622 Fire and 116 No Fire image quartets from one prescribed burn at Sycan Marsh, Oregon, flown 25-27 October 2023), per-image temperature statistics from `thermal_summary.py` show how much a binary label discards:

| Measure (per image) | Fire (622) | No Fire (116) |
|---|---|---|
| Maximum temperature, median | 514 C | 24 C |
| Maximum temperature, range | 108 to 609 C | up to 47 C |
| Images with maximum >= 400 C | 482 (77.5%) | 0 |
| Share of pixels >= 100 C, 10th / 50th / 90th percentile | 0.19% / 1.07% / 4.64% | 0% in every image |

The two classes separate cleanly, but **inside the Fire class the hot area varies about 25-fold** between the 10th and 90th percentile images. A binary detector maps all 622 images to the same output. These are frame-level statistics, so camera distance and altitude also contribute.

### 6.3 Proposed improvement: thermal teacher, RGB student

A severity stage is added on top of detection. The paired radiometric thermal image (a FLAME 3 Celsius TIFF) is used **only during training** to create an automatic Low / Moderate / High label. A CNN then learns to predict that label from the **RGB image alone**, so a deployed drone needs no thermal camera.

Severity score:

```
score = 0.40 * norm(hot-area ratio at >= 100 C)
      + 0.30 * norm(thermal mean, C)
      + 0.30 * norm(thermal 95th percentile, C)
```

Each term is min-max scaled between its 5th and 95th percentile on the **training images only** and clipped to [0, 1]. Low, Moderate and High are the terciles of the score (cutoffs 0.1165 and 0.2714 on the quick experiment's training split). Fitting both the scaling and the cutoffs on training data only keeps validation labels from influencing them.

| Aspect | Original paper | This extension |
|---|---|---|
| Task | binary fire / no fire | three-level severity among fire images |
| Output | one fire probability | Low, Moderate or High |
| Supervision | human fire / no-fire labels | automatic labels from radiometric thermal images |
| Data | Datasets A and B plus synthetic images | FLAME 3 CV subset: paired RGB and thermal TIFF |
| Input at test time | RGB | RGB only |
| Explainability | Grad-CAM and saliency for detection | Grad-CAM on the severity classifier |

### 6.4 What is implemented and what was run

| Item | Status |
|---|---|
| RGB-only MobileNetV2 severity classifier (3 classes), `flame3_quick_experiment.py` | **Run**, source of all severity results below |
| Label sensitivity analysis (threshold, weights, area-only), `phase2_label_analysis.py` | **Run**, from saved CSVs |
| FireNet-backbone classifier and regressor with GroupKFold, `flame3_severity.py` | Implemented, **not run** (no session IDs available) |
| Area-only ablation as trained models, RGB colour-ratio baseline, optional VGG16 (`--run_vgg16`), two-stage helper `predict_two_stage(...)` | Implemented in `flame3_severity.py`, **not run** |
| Evaluation of the full two-stage pipeline (detect, then grade) | Not evaluated |

Two details to be aware of. First, `flame3_severity.py` computes the mean and 95th-percentile terms over **hot pixels only**, whereas the quick experiment uses **whole-image** statistics; every result below uses the whole-image version. Second, the grouped pipeline stops without a real `groups.csv` (section 7), so it has not been run.

### 6.5 Results of the quick experiment (preliminary)

**Setup.** The 622 Fire pairs only, RGB resized to 150x150. Random image-level 80/20 split (seed 42): 497 training and 125 validation images.

| Split | Images | Low | Moderate | High |
|---|---|---|---|---|
| Training | 497 | 166 | 165 | 166 |
| Validation | 125 | 34 | 45 | 46 |
| All fire images | 622 | 200 | 210 | 212 |

**Model.** A frozen ImageNet MobileNetV2 (150x150 input) feeds global average pooling, Dense 128 with ReLU, Dropout 0.4 and a 3-way softmax. Adam 1e-4, sparse cross-entropy, random flip, rotation (factor 0.05) and 10% zoom, early stopping on validation loss (patience 2, at most 12 epochs). Training ran to the 12-epoch cap and the reported model is the epoch-11 checkpoint. Grad-CAM is computed on the last convolutional activation (`out_relu`).

**Results on the 125 validation images.**

| Measure | Value |
|---|---|
| Accuracy | **61.6%** (95% Wilson interval 52.8% to 69.7%) |
| Macro-F1 | 0.616 |
| Majority-class baseline | 27.2% |
| Uniform chance | 33.3% |
| Accuracy within one severity level | 95.2% |
| High recall | 52.2% |

| Class | Precision | Recall | F1 | Images |
|---|---|---|---|---|
| Low | 0.704 | 0.559 | 0.623 | 34 |
| Moderate | 0.523 | 0.756 | 0.618 | 45 |
| High | 0.727 | 0.522 | 0.608 | 46 |
| Macro average | 0.651 | 0.612 | 0.616 | 125 |

Confusion matrix (rows: thermal-derived label, columns: RGB-only prediction):

| True \ Predicted | Low | Moderate | High |
|---|---|---|---|
| Low | 19 | 13 | 2 |
| Moderate | 4 | 34 | 7 |
| High | 4 | 18 | 24 |

Errors fall mostly between neighbouring levels; only 6 of 125 images (4.8%) are two levels off. The most important error is **under-calling severity**: 22 of the 46 High images were graded lower (18 as Moderate, 4 as Low), which is the wrong direction for a safety tool.

### 6.6 Are the labels stable?

Labels were recomputed under alternative definitions, with scaling and cutoffs fitted on the same training images, and compared with the labels used for training (recomputation of the original labels matches the saved `severity_labels.csv` on all 622 rows):

| Alternative definition | Agreement with used labels | Quadratic-weighted kappa |
|---|---|---|
| Equal weights (1/3 each) | 99.2% | 0.994 |
| Area threshold 150 C instead of 100 C | 96.9% | 0.977 |
| Temperature only (mean and 95th percentile) | 93.2% | 0.949 |
| Area threshold 200 C instead of 100 C | 89.7% | 0.922 |
| Area only (share of pixels >= 100 C) | 89.2% | 0.918 |

The labels change little with the threshold or the weights. The flip side is redundancy: the three terms are strongly correlated (Spearman 0.87-0.91), so the score mostly measures **how much of the frame is hot**.

---

## 7. Running the severity code

### Data layout

The scripts search recursively for corrected-FOV RGB images and same-stem TIFFs, and infer Fire / No Fire from directory names. **Confirm the generated `paired_manifest.csv` before training.**

### Critical dataset limitation

IEEE DataPort describes the FLAME 3 CV subset as 622 Fire and 116 No Fire image quartets from one prescribed burn. Each quartet has raw RGB, raw thermal JPG, corrected-FOV RGB and radiometric Celsius TIFF. It does **not** document video or session IDs.

`flame3_severity.py` therefore requires a genuine `groups.csv` mapping `frame_id,video_id`:

```csv
frame_id,video_id
00001,session_01
00002,session_01
...
```

Use real acquisition metadata. **Do not invent group IDs from frame names.** The program stops if the file is missing or has fewer than five groups. Without reliable session IDs you cannot claim video-grouped validation on this subset; use the larger multi-burn FLAME 3 collection only if you can obtain reliable burn or session metadata.

### Commands

Grouped pipeline (needs `groups.csv`):

```bash
python flame3_severity.py \
  --data_root "/path/to/Flame 3 Computer Vision Sets" \
  --groups_csv groups.csv \
  --out runs_flame3 \
  --threshold_c 100 \
  --epochs 20 --batch_size 16 --folds 5
```

Add `--run_vgg16` for the optional fine-tuned VGG16 baseline.

Inspection, thermal statistics and the quick experiment (run from `Trial_with_new_data/`):

```bash
python inspect_dataset.py
python thermal_summary.py
python check_severity_distribution.py
python flame3_quick_experiment.py
python phase2_label_analysis.py
```

<!-- TODO: confirm the command-line arguments of these five scripts and add them above. -->

`flame3_severity.py` saves the thermal feature CSV, out-of-fold predictions, metrics and fold models. `flame3_quick_experiment.py` writes metrics, the confusion matrix, training history and the Grad-CAM example to `quick_severity_results/`.

---

## 8. Interpretation and limitations

**Phase 1 and Phase 2**

- Phase 1 used a different, smaller dataset with a harder fire / smoke distinction, so the gap to the paper reflects the data as well as the model.
- Phase 2 matches the paper's setup but is still slightly below it on every metric; the strict check was completed for Dataset A only.
- Dataset B is built from video frames with random folds, so validation scores may be optimistic.
- Results are from a single seed per run.

**Severity extension**

- **Validation design.** One burn and a random image-level split: neighbouring frames are near-duplicates, so 61.6% is probably optimistic and is **not** an unseen-video or unseen-fire result.
- **Model selection.** The same 125 images chose the best epoch and produced the reported score. A separate test set is needed.
- **Label meaning.** Severity is a **relative thermal proxy**, not ground-truth fire intensity or operational fire severity. Terciles make the classes balanced by construction, and 100 C is an operational hot-pixel threshold, not a universal fire boundary.
- **Thermal area** is the fraction of the aligned image above threshold; it is not a physical fire-area measurement.
- Camera distance, altitude, view angle, occlusion and prescribed-burn conditions all affect the estimate, and 150x150 RGB cannot show distance or altitude.
- Generalization from a prescribed burn to wildfires is unproven.
- The RGB colour-ratio baseline is crude and must not be presented as a strong segmentation method.
- Explainability is one Grad-CAM example, with no quantitative check that the model attends to flames.

## 9. Roadmap

1. Validate across burns (the FLAME 3 paper lists Willamette Valley and Shoetank burns) with leave-one-burn-out testing.
2. Run the FireNet-backbone pipeline (`flame3_severity.py`) once reliable group IDs exist, for a like-for-like comparison with the original architecture.
3. Repeat with several seeds, add the RGB colour-ratio baseline and hold out a test set separate from the model-selection set.
4. Reduce High-to-Moderate errors, for example with the regression variant already implemented or an ordinal loss.
5. Evaluate the full two-stage pipeline on Fire and No Fire images together.
6. Finish Phase 2 `strict` on Dataset B, add a `no_synth` run, and rerun the VGG16, VGG19 and InceptionV3 baselines on the new data.

## 10. References

1. G. M. I. Alam, N. Tasnia, T. Biswas, M. J. Hossen, S. A. Tanim and M. S. U. Miah, "Real-Time Detection of Forest Fires Using FireNet-CNN and Explainable AI Techniques," *IEEE Access*, vol. 13, pp. 51150-51181, 2025. doi:10.1109/ACCESS.2025.3552352.
2. B. Hopkins et al., "FLAME 3 Dataset: Unleashing the Power of Radiometric Thermal UAV Imagery for Wildfire Management," arXiv:2412.02831.
3. M. Marinaccio and F. Afghah, "Seeing Heat with Color: RGB-Only Wildfire Temperature Inference from SAM-Guided Multimodal Distillation using Radiometric Ground Truth," arXiv:2505.01638, 2025.
4. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," *Int. J. Computer Vision*, 2020. arXiv:1610.02391.
5. M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR, 2018.
6. FLAME 3 CV subset description, IEEE DataPort.
