"""
Preliminary FLAME 3 RGB-only fire severity experiment.

IMPORTANT:
- Uses only Fire samples for severity classification.
- Thermal TIFFs create weak/proxy labels; RGB images are model inputs.
- Uses a random image-level split because the supplied CV subset has no
  recording/session metadata. This is exploratory and NOT a leakage-safe
  estimate of generalization to unseen videos or fires.
- Fit feature normalization and severity cutoffs on training data only.
"""

from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import cv2
import tifffile
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, ConfusionMatrixDisplay
)

SEED = 42
IMG_SIZE = (150, 150)
BATCH_SIZE = 16
EPOCHS = 12
ROOT = Path("flame3_severity dataset/FLAME 3 CV Dataset (Sycan Marsh)")
OUT = Path("quick_severity_results")
OUT.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def thermal_features(tif_path):
    arr = tifffile.imread(str(tif_path)).astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.all():
        arr = np.where(finite, arr, np.nan)
    return {
        "hot_area_100": float(np.nanmean(arr >= 100.0)),
        "thermal_mean": float(np.nanmean(arr)),
        "thermal_p95": float(np.nanpercentile(arr, 95)),
    }

def build_manifest():
    rows = []
    for category in ("Fire",):
        rgb_dir = ROOT / category / "RGB" / "Corrected FOV"
        thermal_dir = ROOT / category / "Thermal" / "Celsius TIFF"
        if not rgb_dir.exists() or not thermal_dir.exists():
            raise FileNotFoundError(
                f"Expected folders not found:\n{rgb_dir}\n{thermal_dir}"
            )
        rgb_files = {p.stem: p for p in rgb_dir.glob("*.JPG")}
        thermal_files = {p.stem: p for p in thermal_dir.glob("*.TIFF")}
        common = sorted(set(rgb_files) & set(thermal_files))
        if not common:
            raise RuntimeError("No same-stem RGB/TIFF pairs found.")
        for stem in common:
            row = {
                "frame_id": stem,
                "rgb_path": str(rgb_files[stem]),
                "thermal_path": str(thermal_files[stem]),
                "class": category,
            }
            row.update(thermal_features(thermal_files[stem]))
            rows.append(row)
        print(f"{category}: RGB={len(rgb_files)}, thermal={len(thermal_files)}, paired={len(common)}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "paired_fire_manifest.csv", index=False)
    return df

def make_score(train_df, all_df):
    # Robust min-max scaling fitted ONLY on training samples.
    feature_cols = ["hot_area_100", "thermal_mean", "thermal_p95"]
    weights = [0.40, 0.30, 0.30]
    scaled = {}
    for col in feature_cols:
        lo = float(train_df[col].quantile(0.05))
        hi = float(train_df[col].quantile(0.95))
        if hi <= lo:
            hi = lo + 1e-8
        scaled[col] = ((all_df[col] - lo) / (hi - lo)).clip(0, 1)
    score = sum(w * scaled[c] for w, c in zip(weights, feature_cols))
    return score

def load_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0

def make_model():
    inputs = tf.keras.Input(shape=(150, 150, 3))

    # Current images are scaled to [0, 1].
    # MobileNetV2 expects values in [-1, 1].
    x = tf.keras.layers.Rescaling(
        scale=2.0, offset=-1.0
    )(inputs)

    # Data augmentation
    x = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ])(x)

    # Pretrained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(150, 150, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers
    base_model.trainable = False

    x = base_model(x, training=False)

    # Classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(
        3, activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs, outputs, name="MobileNetV2_RGB_Severity"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
def main():
    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    print("\nBuilding paired manifest and thermal features...")
    df = build_manifest()
    if len(df) < 30:
        raise RuntimeError("Too few paired Fire samples for this experiment.")

    # Split first. All score scaling and tercile cutoffs are fitted on train only.
    train_idx, val_idx = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=SEED
    )
    train_raw = df.iloc[train_idx].copy()
    all_df = df.copy()
    all_df["severity_score"] = make_score(train_raw, all_df)
    train_scores = all_df.iloc[train_idx]["severity_score"]
    q1, q2 = train_scores.quantile([1/3, 2/3]).tolist()
    all_df["severity_class"] = pd.cut(
        all_df["severity_score"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=[0, 1, 2],
        include_lowest=True
    ).astype(int)
    all_df["severity_name"] = all_df["severity_class"].map(
        {0: "Low", 1: "Moderate", 2: "High"}
    )
    all_df["split"] = "train"
    all_df.loc[val_idx, "split"] = "validation"
    all_df.to_csv(OUT / "severity_labels.csv", index=False)

    train_df = all_df.iloc[train_idx].copy()
    val_df = all_df.iloc[val_idx].copy()
    print("\nTraining class counts:")
    print(train_df["severity_name"].value_counts().sort_index())
    print("\nValidation class counts (using train cutoffs):")
    print(val_df["severity_name"].value_counts().sort_index())
    print(f"\nTraining-only score cutoffs: Low/Moderate <= {q1:.4f}; Moderate/High <= {q2:.4f}")

    print("\nLoading RGB images...")
    X_train = np.stack([load_rgb(p) for p in train_df["rgb_path"]])
    y_train = train_df["severity_class"].to_numpy(dtype=np.int32)
    X_val = np.stack([load_rgb(p) for p in val_df["rgb_path"]])
    y_val = val_df["severity_class"].to_numpy(dtype=np.int32)
    print("X_train:", X_train.shape, "X_val:", X_val.shape)

    model = make_model()
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(OUT / "best_severity_model.keras"),
            monitor="val_loss", save_best_only=True
        ),
    ]
    print("\nTraining RGB-only MobileNetV2...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(OUT / "final_severity_model.keras")

    probs = model.predict(X_val, verbose=0)
    preds = np.argmax(probs, axis=1)
    # Majority-class baseline
    majority_class = np.bincount(y_train).argmax()
    baseline_preds = np.full_like(y_val, majority_class)

    print("\nMAJORITY CLASS BASELINE")
    print("Accuracy:", accuracy_score(y_val, baseline_preds))
    print(
    "Macro-F1:",
    f1_score(y_val, baseline_preds, average="macro")
    )
    names = ["Low", "Moderate", "High"]
    report = classification_report(
        y_val, preds, labels=[0, 1, 2], target_names=names,
        output_dict=True, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_val, preds)),
        "macro_f1": float(f1_score(y_val, preds, average="macro")),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_val, preds, labels=[0,1,2]).tolist(),
        "evaluation_note": (
            "Preliminary random image-level split. The subset is from one prescribed "
            "burn and has no verified video/session IDs; this is not an unseen-video "
            "or unseen-fire generalization result."
        ),
        "severity_definition": (
            "Relative thermal proxy score: 0.40 normalized fraction of pixels >=100C, "
            "0.30 normalized thermal mean, 0.30 normalized thermal p95. Normalization "
            "and tercile thresholds fitted on training subset only."
        ),
        "train_score_tercile_cutoffs": [float(q1), float(q2)],
    }
    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nRESULTS")
    print("Accuracy:", metrics["accuracy"])
    print("Macro-F1:", metrics["macro_f1"])
    print(classification_report(y_val, preds, target_names=names, zero_division=0))

    cm = confusion_matrix(y_val, preds, labels=[0,1,2])
    disp = ConfusionMatrixDisplay(cm, display_labels=names)
    disp.plot(values_format="d")
    plt.title("RGB-only MobileNetV2: preliminary severity classification")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=180)
    plt.close()

    # Save training curves.
    pd.DataFrame(history.history).to_csv(OUT / "training_history.csv", index=False)

    # Grad-CAM for MobileNetV2
    try:
        base_model = model.get_layer("mobilenetv2_1.00_224")

        # Extract the final convolutional feature maps
        feature_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=base_model.get_layer("out_relu").output
        )

        # Select an image predicted as High, otherwise first image
        chosen = (
            int(np.where(preds == 2)[0][0])
            if np.any(preds == 2)
            else 0
        )

        sample = X_val[chosen:chosen + 1]
        print("Original RGB image:", val_df.iloc[chosen]["rgb_path"])

        with tf.GradientTape() as tape:

            # Apply MobileNetV2 preprocessing
            scaled_sample = sample * 2.0 - 1.0

            # Extract feature maps
            conv_out = feature_model(
                scaled_sample,
                training=False
            )

            tape.watch(conv_out)

            # Pass features through the classification head
            x = model.get_layer("global_average_pooling2d")(conv_out)
            x = model.get_layer("dense")(x)
            x = model.get_layer("dropout")(x, training=False)
            pred_out = model.get_layer("dense_1")(x)

            class_idx = int(np.argmax(pred_out[0]))
            loss = pred_out[:, class_idx]

        # Calculate gradients
        grads = tape.gradient(loss, conv_out)

        weights = tf.reduce_mean(grads, axis=(1, 2))

        cam = tf.reduce_sum(
            conv_out[0] * weights[0],
            axis=-1
        )

        cam = tf.maximum(cam, 0)
        cam = cam / (tf.reduce_max(cam) + 1e-8)

        cam = cv2.resize(cam.numpy(), IMG_SIZE)

        original = (sample[0] * 255).astype(np.uint8)

        heat = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET
        )

        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

        overlay = np.uint8(
            0.55 * original + 0.45 * heat
        )

        cv2.imwrite(
            str(OUT / "gradcam_example.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )

        print("Saved Grad-CAM example.")
        print("Predicted class:", ["Low", "Moderate", "High"][class_idx])

    except Exception as exc:
        print("Grad-CAM generation skipped:", repr(exc))
if __name__ == "__main__":
    main()
