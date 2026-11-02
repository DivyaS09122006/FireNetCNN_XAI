"""
flame_severity.py
=================
Extension of the FireNet-CNN codebase for thermal-derived fire severity
estimation using the FLAME 3 dataset (arXiv:2412.02831, Hopkins et al. 2024).

FLAME 3 provides "image quartets":
  - corrected_fov_rgb.jpg  (640x512)  — model INPUT at inference
  - thermal.tiff           (640x512)  — per-pixel float32 °C, training-time teacher only

Research claim:
  "A model trained with thermal-derived severity labels can estimate
   visual fire severity from RGB alone." Labels are thermal-derived proxies;
   not ground-truth fire intensity.

Run order: 06 → 07 → 08 → 09
"""

import gc
import json
import os
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tifffile  # pip install tifffile
from scipy.stats import spearmanr
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import applications, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────── constants
IMG          = 150            # spatial input to FireNet (px)
SHAPE        = (IMG, IMG, 3)
EXT_IMG      = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
EXT_TIFF     = {".tiff", ".tif", ".TIFF", ".TIF"}
SEVERITY_NAMES = ["Low", "Moderate", "High"]

# Temperature thresholds (Hopkins 2024 §III-A1, citing wildland fire literature)
FIRE_MASK_THR = 80.0     # °C  — pixels ≥ this are "fire pixels" for area ratio
PHYS_LOW_MAX  = 200.0    # °C  — max_temp < 200 → Low
PHYS_MOD_MAX  = 400.0    # °C  — 200 ≤ max_temp < 400 → Moderate
                          #        max_temp ≥ 400 → High
TIFF_SATURATE = 500.0    # °C  — FLIR saturates at ~500°C per Kaggle card

GRADCAM_LAYER = "act5"   # last ReLU in FireNet backbone (same as firenet.py)


# ─────────────────────────────────────────── data discovery

def discover_quartets(flame3_root, fire_subdir="Fire", nofire_subdir="No_Fire",
                      verbose=True):
    """
    Walk a FLAME 3 directory and pair each .tiff with its corrected-FOV RGB.

    FLAME 3 Sycan Marsh Kaggle subset structure (inferred from Kaggle card):
    Each quartet shares the same stem; files differ by suffix or name token.
    Common conventions tried in order:
        <stem>_T.TIFF  +  <stem>_corrected.JPG
        <stem>.TIFF    +  <stem>_corrected.JPG
        <stem>.TIFF    +  <stem>.JPG
        <stem>_thermal.TIFF  +  <stem>_rgb.JPG

    Returns a list of dicts: {fire_label, rgb_path, tiff_path}
    """
    root = Path(flame3_root)
    records = []

    for label, subdir in [(1, fire_subdir), (0, nofire_subdir)]:
        folder = root / subdir
        if not folder.exists():
            if verbose:
                print(f"  [WARN] {folder} not found — skipping")
            continue

        tiffs = sorted(
            p for p in folder.rglob("*") if p.suffix in EXT_TIFF
        )
        if verbose:
            print(f"  [{subdir}] {len(tiffs)} TIFF files")

        for tiff_path in tiffs:
            rgb_path = _find_rgb_partner(tiff_path)
            if rgb_path is None:
                if verbose:
                    print(f"    [WARN] No RGB partner for {tiff_path.name}")
                continue
            records.append({
                "fire_label": label,
                "rgb_path":   str(rgb_path),
                "tiff_path":  str(tiff_path),
                "frame_id":   tiff_path.stem,
            })

    if verbose:
        fire_n   = sum(r["fire_label"] == 1 for r in records)
        nofire_n = sum(r["fire_label"] == 0 for r in records)
        print(f"  Quartets found: {len(records)} "
              f"(Fire={fire_n}, No-Fire={nofire_n})")
    return records


def _find_rgb_partner(tiff_path):
    """Try several naming conventions to find the corrected FOV RGB for a TIFF."""
    p = Path(tiff_path)
    stem = p.stem
    parent = p.parent

    # Build candidate list (most-specific first)
    stem_base = (stem
                 .replace("_T", "").replace("_thermal", "")
                 .replace("_tiff", "").replace("_TIFF", ""))

    candidates = []
    for b in [stem, stem_base]:
        candidates += [
            parent / f"{b}_corrected.JPG",
            parent / f"{b}_corrected.jpg",
            parent / f"corrected_{b}.JPG",
            parent / f"corrected_{b}.jpg",
            parent / f"{b}_rgb.JPG",
            parent / f"{b}_rgb.jpg",
            parent / f"{b}.JPG",
            parent / f"{b}.jpg",
        ]

    for c in candidates:
        if c.exists():
            return c

    # Last resort: any sibling image that shares the base stem
    for f in parent.iterdir():
        if f.suffix in EXT_IMG and stem_base.lower() in f.stem.lower():
            return f

    return None


# ─────────────────────────────────────────── frame loading

def load_rgb(rgb_path, img_size=IMG):
    """Load and resize one corrected-FOV RGB image → uint8 (H,W,3)."""
    img = cv2.imread(str(rgb_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {rgb_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)


def load_thermal_tiff(tiff_path):
    """
    Load a FLAME 3 thermal TIFF → float32 (H, W) in °C.
    Clips to [0, TIFF_SATURATE] to remove artefacts above sensor ceiling.
    """
    arr = tifffile.imread(str(tiff_path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]           # single band; take first channel
    arr = np.clip(arr, 0.0, TIFF_SATURATE)
    return arr


# ─────────────────────────────────────────── severity features

def compute_severity_features(thermal):
    """
    Compute severity features from a 2-D thermal array (°C).

    Returns dict:
        fire_area_ratio   — fraction of pixels ≥ FIRE_MASK_THR
        thermal_mean      — mean °C inside fire region (0 if no fire pixels)
        thermal_p95       — 95th-percentile °C inside fire region
        max_temp          — global maximum °C in frame
        fire_pixel_count  — raw count of fire pixels
    """
    H, W = thermal.shape
    fire_mask = thermal >= FIRE_MASK_THR
    n_fire    = int(fire_mask.sum())

    if n_fire > 0:
        fire_temps   = thermal[fire_mask]
        thermal_mean = float(fire_temps.mean())
        thermal_p95  = float(np.percentile(fire_temps, 95))
    else:
        thermal_mean = 0.0
        thermal_p95  = 0.0

    return {
        "fire_area_ratio":  n_fire / (H * W),
        "thermal_mean":     thermal_mean,
        "thermal_p95":      thermal_p95,
        "max_temp":         float(thermal.max()),
        "fire_pixel_count": n_fire,
    }


def physical_severity_class(max_temp):
    """
    Bin by physical thresholds (Hopkins 2024, citing wildland fire literature).
    Low < 200°C | Moderate 200–400°C | High ≥ 400°C
    """
    if max_temp < PHYS_LOW_MAX:
        return 0
    if max_temp < PHYS_MOD_MAX:
        return 1
    return 2


# ─────────────────────────────────────────── label dataframe

def build_labels_df(flame3_root, verbose=True):
    """
    Full pipeline: discover → load thermal → compute features → return DataFrame.

    Columns: frame_id, fire_label, rgb_path, tiff_path, frame_idx,
             fire_area_ratio, thermal_mean, thermal_p95, max_temp,
             fire_pixel_count, severity_class_phys
    """
    records = discover_quartets(flame3_root, verbose=verbose)
    if not records:
        raise RuntimeError(
            f"No quartets found under {flame3_root}. "
            "Check fire_subdir / nofire_subdir names."
        )

    rows = []
    for i, rec in enumerate(records):
        try:
            thermal = load_thermal_tiff(rec["tiff_path"])
            feats   = compute_severity_features(thermal)
        except Exception as e:
            if verbose:
                print(f"  [WARN] {rec['tiff_path']}: {e}")
            continue

        rows.append({
            **rec,
            "frame_idx": i,
            **feats,
            "severity_class_phys": physical_severity_class(feats["max_temp"]),
        })

    df = pd.DataFrame(rows).sort_values("tiff_path").reset_index(drop=True)
    df["frame_idx"] = np.arange(len(df))  # re-number after sort

    if verbose:
        fire_df = df[df.fire_label == 1]
        print(f"\nPhysical threshold distribution (Fire frames only):")
        counts = fire_df["severity_class_phys"].value_counts().sort_index()
        for c, n in counts.items():
            print(f"  {SEVERITY_NAMES[c]:10s}: {n} frames "
                  f"({100*n/len(fire_df):.1f}%)")
    return df


def assign_temporal_chunks(df, n_chunks=5):
    """
    Assign temporal chunk IDs for GroupKFold (contiguous time blocks).
    Only Fire frames receive a chunk_id 0..n_chunks-1.
    No-Fire frames get chunk_id = -1 (excluded from severity CV).
    """
    df = df.copy()
    df["chunk_id"] = -1
    fire_idx = df.index[df.fire_label == 1].tolist()

    chunk_size = len(fire_idx) // n_chunks
    for k in range(n_chunks):
        lo = k * chunk_size
        hi = (k + 1) * chunk_size if k < n_chunks - 1 else len(fire_idx)
        df.loc[fire_idx[lo:hi], "chunk_id"] = k

    print("Temporal chunks (Fire frames):")
    for k, n in df[df.fire_label == 1]["chunk_id"].value_counts().sort_index().items():
        print(f"  Chunk {k}: {n} frames")
    return df


def add_tercile_labels(df, label_col="severity_score", n_classes=3):
    """
    Compute the full-dataset tercile bins for a given score column.
    NOTE: call this ONLY after the train/val split — pass the training subset
    to compute cuts. Here we expose it as a utility; run_severity_cv handles
    per-fold normalisation internally.
    """
    cuts = [df[label_col].quantile(k / n_classes) for k in range(1, n_classes)]
    def _bin(s):
        for i, cut in enumerate(cuts):
            if s <= cut:
                return i
        return n_classes - 1
    return df[label_col].apply(_bin), cuts


# ─────────────────────────────────────────── severity score (composite)

def _normalise(series, mn, mx, eps=1e-8):
    return ((series - mn) / (mx - mn + eps)).clip(0, 1)


def add_severity_scores(df, train_mask):
    """
    Adds three columns derived from training-set statistics only:
        severity_score           — composite (0.4·area + 0.3·mean_T + 0.3·p95_T)
        severity_class_tercile   — tercile bin of severity_score
        severity_class_area_only — tercile bin of area_ratio only (ablation)
    Also adds two ablation weight variants:
        severity_score_w2        — (1.0·area, 0.0·mean, 0.0·p95) area-only composite
        severity_score_w3        — (0.0·area, 0.5·mean, 0.5·p95) thermal-only composite

    train_mask: boolean Series/array aligned with df.index
    """
    df = df.copy()
    fire_train = df[train_mask & (df.fire_label == 1)]

    # normalization stats from training fire frames
    ar_mn, ar_mx = fire_train.fire_area_ratio.min(), fire_train.fire_area_ratio.max()
    tm_mn, tm_mx = fire_train.thermal_mean.min(),    fire_train.thermal_mean.max()
    tp_mn, tp_mx = fire_train.thermal_p95.min(),     fire_train.thermal_p95.max()

    ar_n = _normalise(df.fire_area_ratio, ar_mn, ar_mx)
    tm_n = _normalise(df.thermal_mean,    tm_mn, tm_mx)
    tp_n = _normalise(df.thermal_p95,     tp_mn, tp_mx)

    # Weight set 1 (primary): 0.4 / 0.3 / 0.3
    df["severity_score"]    = 0.4 * ar_n + 0.3 * tm_n + 0.3 * tp_n
    # Weight set 2 (ablation): area only
    df["severity_score_w2"] = 1.0 * ar_n
    # Weight set 3 (ablation): thermal only
    df["severity_score_w3"] = 0.5 * tm_n + 0.5 * tp_n

    # Tercile cuts — computed on training set
    def _tercile_class(series, score_col):
        train_scores = df.loc[fire_train.index, score_col]
        cuts = [train_scores.quantile(1/3), train_scores.quantile(2/3)]
        def _b(s):
            if s <= cuts[0]: return 0
            if s <= cuts[1]: return 1
            return 2
        return series[score_col].apply(_b)

    df["severity_class_tercile"]   = _tercile_class(df, "severity_score")
    df["severity_class_area_only"] = _tercile_class(df, "severity_score_w2")
    df["severity_class_therm_only"]= _tercile_class(df, "severity_score_w3")

    return df


# ─────────────────────────────────────────── RGB-area baseline

def rgb_area_ratio(rgb_img):
    """
    Compute fire-pixel fraction from an RGB image using HSV thresholding.
    Mirrors firenet.py's segment_fire logic.
    Returns a float in [0, 1].
    """
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([20, 255, 255]))
    return mask.sum() / (255 * mask.size)


def rgb_area_baseline_classes(df, tercile_cuts):
    """
    Rule-based baseline: bin RGB fire-area-ratio into severity classes
    using the same tercile cuts derived from thermal labels.
    tercile_cuts: list of two floats [cut1, cut2] in area-ratio space.
    Returns array of predicted classes.
    """
    preds = []
    for _, row in df.iterrows():
        img = load_rgb(row.rgb_path)
        ar  = rgb_area_ratio(img)
        if ar <= tercile_cuts[0]:
            preds.append(0)
        elif ar <= tercile_cuts[1]:
            preds.append(1)
        else:
            preds.append(2)
    return np.array(preds)


# ─────────────────────────────────────────── models

def _act_block(x, act="relu", name=None):
    return (layers.LeakyReLU(name=name)(x) if act == "leaky_relu"
            else layers.Activation(act, name=name)(x))


def _firenet_backbone(input_tensor=None):
    """
    Shared FireNet-CNN feature extractor.
    5 × [Conv3×3 → ReLU → MaxPool2×2 → BN] with 32/64/128/256/512 filters,
    then Flatten → Dense(512) → Dropout(0.5) → Dense(256) → Dropout(0.5).
    Input: 150×150×3.  Output feature dim: 256.
    """
    inp = input_tensor or layers.Input(SHAPE)
    x = inp
    for i, f in enumerate([32, 64, 128, 256, 512], 1):
        x = layers.Conv2D(f, 3, padding="valid", name=f"conv{i}")(x)
        x = _act_block(x, "relu", name=f"act{i}")
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
    x = layers.Flatten()(x)
    for u in (512, 256):
        x = layers.Dense(u)(x)
        x = _act_block(x, "relu")
        x = layers.Dropout(0.5)(x)
    return inp, x


def build_firenet_severity_clf(n_classes=3):
    """3-class severity classifier (Low / Moderate / High)."""
    inp, x = _firenet_backbone()
    out = layers.Dense(n_classes, activation="softmax",
                       name="severity_out")(x)
    return tf.keras.Model(inp, out, name="FireNet_Severity_Clf")


def build_firenet_severity_reg():
    """Regression variant — predicts continuous severity_score ∈ [0,1]."""
    inp, x = _firenet_backbone()
    out = layers.Dense(1, activation="sigmoid",
                       name="severity_reg")(x)
    return tf.keras.Model(inp, out, name="FireNet_Severity_Reg")


def build_pretrained_severity(name="vgg16", n_classes=3):
    """VGG16 or InceptionV3 fine-tuned for severity classification."""
    base_fn = {"vgg16": applications.VGG16,
               "inceptionv3": applications.InceptionV3}[name]
    base = base_fn(weights="imagenet", include_top=False, input_shape=SHAPE)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(base.input, out, name=f"{name}_severity")


# ─────────────────────────────────────────── cross-validation

def _load_X(paths, img_size=IMG):
    """Batch-load RGB images into float32 array [0,1]."""
    imgs = []
    for p in paths:
        try:
            imgs.append(load_rgb(p, img_size))
        except Exception as e:
            print(f"  [WARN] Could not load {p}: {e}")
            imgs.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
    return np.stack(imgs).astype(np.float32) / 255.0


def run_severity_cv(
    df_fire,          # DataFrame of Fire frames (fire_label==1) with chunk_id
    model_fn,         # callable → uncompiled Keras model
    label_col,        # column to use as y (e.g. 'severity_class_tercile')
    is_regression=False,
    n_folds=5,
    batch_size=32,
    epochs=50,
    lr=1e-4,
    seed=42,
    out_dir="runs/severity",
    tag="firenet_clf",
    verbose=True,
):
    """
    GroupKFold cross-validation for severity models.
    Groups = temporal chunk IDs to prevent adjacent-frame leakage.
    Returns DataFrame of per-fold metrics.
    """
    out = Path(out_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nRunning CV: {tag}\n{'='*60}")

    X      = _load_X(df_fire.rgb_path.tolist())
    y      = df_fire[label_col].values
    groups = df_fire.chunk_id.values

    n_classes = len(np.unique(y)) if not is_regression else 1

    # Class weights for classifier (handle imbalance)
    cw_dict = None
    if not is_regression:
        cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
        cw_dict = dict(enumerate(cw))
        print(f"Class weights: {cw_dict}")

    aug_train = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        horizontal_flip=True, zoom_range=0.2,
        brightness_range=(0.8, 1.2), fill_mode="nearest",
    )
    aug_val = ImageDataGenerator()   # no augmentation

    gkf  = GroupKFold(n_splits=n_folds)
    rows = []
    cms  = []

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups), 1):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed + fold)

        m = model_fn()
        if is_regression:
            m.compile(optimizers.Adam(lr), loss="mse", metrics=["mae"])
        else:
            m.compile(optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True,
                monitor="val_loss", verbose=0,
            )
        ]

        t0 = time.time()
        m.fit(
            aug_train.flow(X[tr], y[tr], batch_size=batch_size,
                           shuffle=True, seed=seed),
            validation_data=aug_val.flow(X[va], y[va],
                                         batch_size=batch_size,
                                         shuffle=False),
            epochs=epochs,
            class_weight=cw_dict,
            callbacks=callbacks,
            verbose=2 if verbose else 0,
        )
        train_time = time.time() - t0

        preds = m.predict(X[va], verbose=0)

        row = {"fold": fold, "n_train": len(tr), "n_val": len(va),
               "train_time_s": round(train_time, 1)}

        if is_regression:
            pred_scores = preds.ravel()
            rho, pval   = spearmanr(y[va], pred_scores)
            mae         = float(np.abs(y[va] - pred_scores).mean())
            row.update({"spearman_rho": round(rho, 4),
                        "spearman_pval": round(pval, 4),
                        "mae": round(mae, 4)})
        else:
            pred_cls   = np.argmax(preds, axis=1)
            macro_f1   = f1_score(y[va], pred_cls, average="macro",
                                  zero_division=0)
            row["macro_f1"] = round(macro_f1, 4)
            for c in range(n_classes):
                row[f"prec_cls{c}"] = round(
                    precision_score(y[va], pred_cls, labels=[c],
                                    average="macro", zero_division=0), 4)
                row[f"rec_cls{c}"]  = round(
                    recall_score(y[va], pred_cls, labels=[c],
                                 average="macro", zero_division=0), 4)
            cm = confusion_matrix(y[va], pred_cls,
                                  labels=list(range(n_classes)))
            cms.append(cm)
            if verbose:
                print(classification_report(y[va], pred_cls,
                      target_names=SEVERITY_NAMES, zero_division=0))

        rows.append(row)
        print(f"  Fold {fold}: {row}")

        if fold == 1:
            m.save(out / "model_fold1.keras")
        del m; gc.collect()

    df_res = pd.DataFrame(rows)
    df_res.to_csv(out / "fold_results.csv", index=False)

    mean_row = df_res.mean(numeric_only=True).round(4)
    print(f"\n── Mean across {n_folds} folds ──")
    print(mean_row.to_string())
    mean_row.to_json(out / "summary.json")

    if cms:
        _plot_confusion(cms, out, tag, n_classes)

    return df_res


# ─────────────────────────────────────────── baselines

def majority_class_baseline(y_true):
    """Predict the majority class for every sample."""
    counts  = np.bincount(y_true)
    majority = int(np.argmax(counts))
    y_pred   = np.full_like(y_true, majority)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"Majority-class baseline  macro-F1={mf1:.4f}  "
          f"(predicts class {majority} = {SEVERITY_NAMES[majority]} always)")
    return y_pred, mf1


def rgb_area_rule_baseline(df_fire, label_col, out_dir="runs/severity"):
    """
    Rule-based baseline: HSV fire-pixel area → severity class.
    Finds tercile cuts on training labels and applies them to the HSV areas.
    Returns (y_pred, macro_f1).
    """
    print("\nComputing RGB-area rule baseline ...")
    rgb_areas = []
    for rgb_path in df_fire.rgb_path:
        img = load_rgb(rgb_path)
        rgb_areas.append(rgb_area_ratio(img))
    rgb_areas = np.array(rgb_areas)

    y_true = df_fire[label_col].values
    cuts   = [np.quantile(rgb_areas, 1/3), np.quantile(rgb_areas, 2/3)]

    y_pred = np.where(rgb_areas <= cuts[0], 0,
             np.where(rgb_areas <= cuts[1], 1, 2))

    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"RGB-area rule baseline   macro-F1={mf1:.4f}")
    print(classification_report(y_true, y_pred,
          target_names=SEVERITY_NAMES, zero_division=0))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "rgb_area_baseline_preds.npy", y_pred)
    return y_pred, mf1


# ─────────────────────────────────────────── XAI

def severity_gradcam(model, x_norm, class_idx=None, layer=GRADCAM_LAYER):
    """
    Grad-CAM for severity classifier.
    x_norm: float32 array (H, W, 3) in [0,1].
    class_idx: target class (0=Low,1=Moderate,2=High); None → argmax.
    Returns (cam_norm, pred_probs) where cam_norm is upsampled to H×W in [0,1].
    """
    gm = tf.keras.Model(
        model.inputs,
        [model.get_layer(layer).output, model.output],
    )
    x4 = tf.constant(x_norm[None])          # (1,H,W,3)
    with tf.GradientTape() as tape:
        conv_out, pred = gm(x4)
        if class_idx is None:
            class_idx = int(tf.argmax(pred[0]))
        score = pred[:, class_idx]

    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))[0].numpy()

    cam = cam / (cam.max() + 1e-8)
    h, w = x_norm.shape[:2]
    cam = cv2.resize(cam, (w, h))
    return cam, pred[0].numpy()


def plot_severity_gradcam(model, rgb_paths, true_classes, out_path,
                          n=6, layer=GRADCAM_LAYER, seed=42):
    """
    Save a grid of Grad-CAM overlays for n randomly-sampled fire frames.
    Shows Low / Moderate / High side-by-side for visual inspection.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rgb_paths), size=min(n, len(rgb_paths)),
                         replace=False)

    fig, axes = plt.subplots(len(indices), 3,
                             figsize=(12, 4 * len(indices)))
    axes = np.atleast_2d(axes)

    cmap_jet = plt.cm.jet

    for row_i, idx in enumerate(indices):
        rgb = load_rgb(rgb_paths[idx])
        x   = rgb.astype(np.float32) / 255.0

        for col_i, cls_idx in enumerate([0, 1, 2]):
            cam, probs = severity_gradcam(model, x, class_idx=cls_idx,
                                          layer=layer)
            heat = (cmap_jet(cam)[:, :, :3] * 255).astype(np.uint8)
            overlay = cv2.addWeighted(rgb, 0.55, heat, 0.45, 0)

            axes[row_i, col_i].imshow(overlay)
            axes[row_i, col_i].set_title(
                f"True:{SEVERITY_NAMES[true_classes[idx]]}  "
                f"Grad-CAM for {SEVERITY_NAMES[cls_idx]}\n"
                f"p={probs[cls_idx]:.2f}",
                fontsize=8,
            )
            axes[row_i, col_i].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM grid → {out_path}")


# ─────────────────────────────────────────── plots

def _plot_confusion(cms, out_dir, tag, n_classes):
    n  = len(cms)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.5 * (n + 1), 3.5))
    axes = np.atleast_1d(axes)
    mean_cm = np.mean(cms, axis=0).round().astype(int)

    for ax, cm, title in zip(
            axes,
            cms + [mean_cm],
            [f"Fold {i+1}" for i in range(n)] + ["Mean"],
    ):
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center", fontsize=9)
        ax.set_xticks(range(n_classes), SEVERITY_NAMES[:n_classes])
        ax.set_yticks(range(n_classes), SEVERITY_NAMES[:n_classes])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion — {title}", fontsize=8)

    plt.tight_layout()
    path = out_dir / f"{tag}_confusion.png"
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Saved confusion matrix → {path}")


def plot_score_distribution(df_fire, score_col="severity_score",
                            class_col="severity_class_tercile",
                            out_path="runs/severity/score_dist.png"):
    """Histogram of severity scores, coloured by class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: raw histogram coloured by class
    for cls, name, colour in zip([0, 1, 2], SEVERITY_NAMES,
                                 ["royalblue", "darkorange", "crimson"]):
        subset = df_fire[df_fire[class_col] == cls][score_col]
        axes[0].hist(subset, bins=20, alpha=0.7, color=colour, label=name)
    axes[0].set_xlabel(score_col)
    axes[0].set_ylabel("Frame count")
    axes[0].set_title("Severity score distribution")
    axes[0].legend()

    # Right: class counts bar chart
    counts = df_fire[class_col].value_counts().sort_index()
    axes[1].bar(SEVERITY_NAMES, counts.values,
                color=["royalblue", "darkorange", "crimson"])
    axes[1].set_ylabel("Frame count")
    axes[1].set_title(f"Class distribution ({class_col})")
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 1, str(v), ha="center", fontsize=10)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Saved score distribution → {out_path}")


def plot_ablation_table(results_dict, out_path="runs/severity/ablation.png"):
    """
    results_dict: {label: macro_f1_mean}
    Horizontal bar chart comparing baselines vs models vs ablations.
    """
    labels = list(results_dict.keys())
    values = list(results_dict.values())
    colours = ["#5c85d6" if "FireNet" in l else
               "#e07b39" if "VGG" in l or "Inception" in l else
               "#888888"
               for l in labels]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.55)))
    bars = ax.barh(labels, values, color=colours, height=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.axvline(1/3, color="gray", linestyle="--", lw=1,
               label="Random baseline (macro-F1 = 0.33)")
    ax.set_xlabel("Macro-F1")
    ax.set_title("Severity Model Comparison & Ablation")
    ax.legend(fontsize=8)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Saved ablation chart → {out_path}")


# ─────────────────────────────────────────── CLI entry point
# (not the primary usage — notebooks call functions directly)
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="flame_severity.py — quick label-CSV builder from CLI"
    )
    parser.add_argument("--flame3_root", required=True,
                        help="Root folder of FLAME 3 dataset (contains Fire/ and No_Fire/)")
    parser.add_argument("--out", default="flame_severity_labels.csv")
    parser.add_argument("--n_chunks", type=int, default=5)
    args = parser.parse_args()

    df = build_labels_df(args.flame3_root)
    df = assign_temporal_chunks(df, n_chunks=args.n_chunks)
    df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}  ({len(df)} rows)")
