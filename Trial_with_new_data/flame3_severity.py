"""
FLAME 3 RGB-only severity estimation with thermal-derived training labels.
TensorFlow/Keras implementation; no manual annotations.

IMPORTANT:
- Use FLAME 3 CVSubset (not original FLAME 1).
- The CVSubset contains corrected-FOV RGB + radiometric Celsius TIFF quartets.
- This subset is from ONE prescribed burn. The published subset description does
  not provide video/session IDs. For valid grouped evaluation, supply groups.csv
  with columns: frame_id,video_id. Do NOT treat individual images as independent
  groups. If no valid groups are available, this script stops before training.
- Severity is a thermal-derived VISUAL SEVERITY ESTIMATE, not true fire intensity.

Expected root layout (folder names can be nested):
  DATA_ROOT/
    .../Fire/RGB/Corrected FOV/00001.JPG
    .../Fire/Thermal/Celsius TIFF/00001.TIFF
    .../No Fire/RGB/Corrected FOV/...
    .../No Fire/Thermal/Celsius TIFF/...

Install:
  pip install tensorflow pandas numpy scikit-learn opencv-python tifffile matplotlib scipy

Run:
  python flame3_severity.py --data_root "/path/to/FLAME3 CVSubset" \
      --groups_csv groups.csv --out runs --epochs 20

groups.csv must map every image stem to a real recording/session/video group:
  frame_id,video_id
  00001,video_01
  00002,video_01
  ...
"""

import os, re, json, argparse, random, warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import cv2
import tifffile
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (f1_score, precision_recall_fscore_support,
                             confusion_matrix, mean_absolute_error,
                             classification_report)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, Model

SEED = 42
IMG_SIZE = 150
CLASS_NAMES = ["Low", "Moderate", "High"]

def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)

def find_rgb_files(root):
    files = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            s = str(p).lower().replace("\\", "/")
            if "corrected fov" in s or ("rgb" in s and "corrected" in s):
                if "fire" in s or "no fire" in s or "nofire" in s:
                    files.append(p)
    return sorted(set(files))

def class_from_path(p):
    parts = [x.lower().replace("_", " ").replace("-", " ") for x in p.parts]
    if any(x == "no fire" or x == "nofire" for x in parts): return 0
    if any(x == "fire" for x in parts): return 1
    return None

def find_matching_tiff(rgb_path, root):
    stem = rgb_path.stem.lower()
    rgb_class = class_from_path(rgb_path)
    candidates = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in (".tif", ".tiff") and p.stem.lower() == stem:
            if class_from_path(p) == rgb_class:
                candidates.append(p)
    # Prefer the TIFF in the thermal/Celsius TIFF folder.
    candidates.sort(key=lambda p: ("celsius" not in str(p).lower(),
                                   "thermal" not in str(p).lower(), len(str(p))))
    return candidates[0] if candidates else None

def read_temp_tiff(path):
    a = np.asarray(tifffile.imread(str(path))).astype(np.float32)
    # Some TIFFs may have a singleton channel.
    if a.ndim == 3:
        a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D radiometric TIFF, got {a.shape}: {path}")
    # Do not silently convert raw digital numbers to Celsius.
    return a

def make_manifest(root):
    rows, problems = [], []
    rgb_files = find_rgb_files(root)
    if not rgb_files:
        raise RuntimeError("No corrected-FOV RGB files found. Check --data_root and folder names.")
    for rp in rgb_files:
        label = class_from_path(rp)
        if label is None:
            problems.append((str(rp), "could_not_infer_fire_class")); continue
        tp = find_matching_tiff(rp, root)
        if tp is None:
            problems.append((str(rp), "missing_matching_thermal_tiff")); continue
        try:
            im = cv2.imread(str(rp))
            temp = read_temp_tiff(tp)
            if im is None: raise ValueError("RGB decode failed")
            if im.shape[:2] != temp.shape:
                problems.append((str(rp), f"RGB/TIFF dimension mismatch {im.shape[:2]} vs {temp.shape}"))
                continue
            rows.append(dict(frame_id=("fire_" if label == 1 else "nofire_") + rp.stem,
                             rgb_path=str(rp), thermal_path=str(tp),
                             binary_label=label, height=im.shape[0], width=im.shape[1]))
        except Exception as e:
            problems.append((str(rp), str(e)))
    df = pd.DataFrame(rows)
    if len(df) == 0: raise RuntimeError("No valid RGB-TIFF pairs found.")
    df.to_csv(Path(root)/"paired_manifest.csv", index=False)
    pd.DataFrame(problems, columns=["path","issue"]).to_csv(Path(root)/"data_quality_issues.csv", index=False)
    print(f"Paired RGB/TIFF: {len(df)} | fire={int(df.binary_label.sum())} | no-fire={int((1-df.binary_label).sum())}")
    print(f"Skipped/problem files: {len(problems)}")
    return df

def attach_groups(df, groups_csv):
    if not groups_csv:
        raise RuntimeError(
            "STOP: grouped evaluation requires real video_id/session_id values. "
            "The FLAME 3 CVSubset description lists paired still-image quartets from "
            "one prescribed burn, not video/session IDs. Supply --groups_csv with "
            "frame_id,video_id. Do not use random frame splits."
        )
    g = pd.read_csv(groups_csv, dtype={"frame_id":str, "video_id":str})
    if not {"frame_id","video_id"}.issubset(g.columns):
        raise ValueError("groups_csv must contain frame_id and video_id columns.")
    df["frame_id"] = df.frame_id.astype(str)
    g["frame_id"] = g.frame_id.astype(str)
    df = df.merge(g[["frame_id","video_id"]], on="frame_id", how="left", validate="one_to_one")
    if df.video_id.isna().any():
        missing = df.loc[df.video_id.isna(),"frame_id"].tolist()[:10]
        raise RuntimeError(f"Missing video_id mappings for frame IDs (first 10): {missing}")
    if df.video_id.nunique() < 5:
        raise RuntimeError(f"Need at least 5 distinct video/session groups for 5-fold GroupKFold; found {df.video_id.nunique()}.")
    return df

def thermal_features(temp, threshold_c=100.0):
    # Operational hot-pixel mask. Threshold is a declared heuristic, not a
    # universal physical definition of flame.
    hot = np.isfinite(temp) & (temp >= threshold_c)
    vals = temp[hot]
    if vals.size == 0:
        return dict(fire_area_ratio=0.0, thermal_mean_c=np.nan,
                    thermal_p95_c=np.nan, thermal_max_c=np.nan,
                    hot_pixel_count=0, thermal_mask_empty=1)
    return dict(fire_area_ratio=float(hot.mean()),
                thermal_mean_c=float(np.mean(vals)),
                thermal_p95_c=float(np.percentile(vals,95)),
                thermal_max_c=float(np.max(vals)),
                hot_pixel_count=int(vals.size), thermal_mask_empty=0)

def add_teacher_features(df, threshold_c):
    records = []
    for r in df.itertuples():
        t = read_temp_tiff(r.thermal_path)
        f = thermal_features(t, threshold_c)
        records.append(f)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)

def robust_norm(x, lo, hi):
    x = np.asarray(x, np.float32)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x)
    return np.clip((x-lo)/(hi-lo), 0, 1)

def score_from_train_stats(df, train_idx):
    # Fit all normalization bounds using TRAIN indices only.
    features = ["fire_area_ratio", "thermal_mean_c", "thermal_p95_c"]
    stats = {}
    for f in features:
        vals = df.iloc[train_idx][f].replace([np.inf,-np.inf], np.nan).dropna().values
        if len(vals) == 0: raise RuntimeError(f"No valid values for {f} in training fold.")
        stats[f] = [float(np.percentile(vals,5)), float(np.percentile(vals,95))]
    def norm_col(f):
        vals = df[f].fillna(df.iloc[train_idx][f].median()).values
        return robust_norm(vals, *stats[f])
    # 40% thermal extent, 30% mean temperature, 30% upper-tail temperature.
    score = (0.40*norm_col("fire_area_ratio") +
             0.30*norm_col("thermal_mean_c") +
             0.30*norm_col("thermal_p95_c"))
    return score.astype(np.float32), stats

def tercile_labels(score, train_idx):
    q1, q2 = np.quantile(score[train_idx], [1/3, 2/3])
    # Duplicate cut points can occur with small/quantized datasets.
    if q1 >= q2:
        raise RuntimeError("Training severity scores do not have distinct tercile cutoffs.")
    y = np.digitize(score, [q1,q2]).astype(np.int32)
    return y, float(q1), float(q2)

def load_rgb(path):
    im = cv2.imread(path)
    if im is None: raise ValueError(f"Cannot read {path}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA)
    return im.astype(np.float32)/255.0

def build_firenet(output="softmax"):
    # Same five convolution blocks as the binary FireNet backbone.
    inp = layers.Input((IMG_SIZE,IMG_SIZE,3))
    x = inp
    for i, f in enumerate([32,64,128,256,512],1):
        x = layers.Conv2D(f,3,padding="valid",name=f"conv{i}")(x)
        x = layers.ReLU(name=f"act{i}")(x)
        x = layers.MaxPooling2D(2,name=f"pool{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
    x = layers.Flatten()(x)
    for j,u in enumerate([512,256],1):
        x = layers.Dense(u,activation="relu",name=f"dense{j}")(x)
        x = layers.Dropout(0.5)(x)
    if output == "softmax":
        out = layers.Dense(3,activation="softmax",name="severity")(x)
    else:
        out = layers.Dense(1,activation="sigmoid",name="severity_score")(x)
    return Model(inp,out,name=f"FireNet_Severity_{output}")

def make_dataset(paths, targets, batch_size=16, training=False):
    def gen():
        for p,y in zip(paths,targets):
            try:
                x=load_rgb(p)
                if training and random.random()<0.5: x=np.fliplr(x).copy()
                yield x,y
            except Exception as e:
                warnings.warn(str(e))
    sig=(tf.TensorSpec((IMG_SIZE,IMG_SIZE,3),tf.float32),
         tf.TensorSpec((),tf.int32 if np.asarray(targets).dtype.kind in "iu" else tf.float32))
    ds=tf.data.Dataset.from_generator(gen,output_signature=sig)
    if training: ds=ds.shuffle(min(len(paths),1024),seed=SEED)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def rgb_fire_ratio_baseline(paths):
    # Simple RGB-only red/orange pixel ratio. This is a deliberately crude baseline.
    ratios=[]
    for p in paths:
        im=cv2.imread(p)
        im=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        m1=cv2.inRange(im,np.array([0,80,80]),np.array([25,255,255]))
        m2=cv2.inRange(im,np.array([160,80,80]),np.array([180,255,255]))
        ratios.append(float(((m1>0)|(m2>0)).mean()))
    return np.asarray(ratios,np.float32)

def save_confusion(y_true,y_pred,path):
    cm=confusion_matrix(y_true,y_pred,labels=[0,1,2])
    fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(cm,cmap="Blues")
    ax.set_xticks(range(3),CLASS_NAMES); ax.set_yticks(range(3),CLASS_NAMES)
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def save_severity_gradcam(model, image_paths, out_dir, max_images=6, class_index=2):
    """Grad-CAM for the High class; qualitative diagnostic, not proof of causal attention."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    grad_model = tf.keras.Model(model.inputs,
                                [model.get_layer("act5").output, model.output])
    for n, path in enumerate(image_paths[:max_images]):
        x = load_rgb(path)
        xt = tf.convert_to_tensor(x[None], dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_out, pred = grad_model(xt)
            target = pred[:, class_index]
        grads = tape.gradient(target, conv_out)
        weights = tf.reduce_mean(grads, axis=(1,2), keepdims=True)
        cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))[0].numpy()
        cam = cv2.resize(cam, (IMG_SIZE,IMG_SIZE))
        cam = (cam-cam.min())/(cam.max()-cam.min()+1e-8)
        heat = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)/255.0
        overlay = np.clip(0.55*x + 0.45*heat,0,1)
        fig,ax=plt.subplots(1,2,figsize=(7,3))
        ax[0].imshow(x); ax[0].set_title("RGB input")
        ax[1].imshow(overlay); ax[1].set_title(f"Grad-CAM: High prob={float(pred[0,class_index]):.3f}")
        for a in ax: a.axis("off")
        fig.tight_layout(); fig.savefig(out_dir/f"gradcam_{n+1}.png",dpi=150); plt.close(fig)

def build_vgg16_severity():
    base=tf.keras.applications.VGG16(weights="imagenet",include_top=False,
                                      input_shape=(IMG_SIZE,IMG_SIZE,3))
    base.trainable=True
    x=layers.GlobalAveragePooling2D()(base.output)
    x=layers.Dense(256,activation="relu")(x)
    x=layers.Dropout(0.5)(x)
    out=layers.Dense(3,activation="softmax")(x)
    return Model(base.input,out,name="VGG16_severity")

def predict_two_stage(binary_model_path, severity_model_path, image_path, threshold=0.5):
    """At inference, only the RGB image is passed to either model."""
    binary=tf.keras.models.load_model(binary_model_path)
    severity=tf.keras.models.load_model(severity_model_path)
    x=load_rgb(image_path)[None]
    p_fire=float(binary.predict(x,verbose=0).ravel()[0])
    if p_fire < threshold:
        return {"prediction":"No Fire","fire_probability":p_fire}
    probs=severity.predict(x,verbose=0).ravel()
    return {"prediction":CLASS_NAMES[int(np.argmax(probs))],
            "fire_probability":p_fire,
            "severity_probabilities":dict(zip(CLASS_NAMES,[float(v) for v in probs]))}

def train_grouped(df, out, epochs=20, batch_size=16, folds=5, run_vgg16=False):
    # Severity model is trained only on fire-positive images.
    fire=df[df.binary_label==1].reset_index(drop=True)
    if fire.video_id.nunique()<folds:
        raise RuntimeError("Not enough distinct fire video groups for requested GroupKFold.")
    paths=fire.rgb_path.tolist()
    groups=fire.video_id.values
    gkf=GroupKFold(n_splits=folds)
    oof_true=[]; oof_pred=[]; oof_reg=[]; oof_score=[]; all_rows=[]
    model_dir=Path(out)/"models"; model_dir.mkdir(parents=True,exist_ok=True)

    # RGB-only rule baseline uses training-fold terciles, never validation labels.
    for fold,(tr,va) in enumerate(gkf.split(paths,groups=groups),1):
        fold_df=fire.copy()
        score, norm_stats=score_from_train_stats(fold_df,tr)
        y_cls,q1,q2=tercile_labels(score,tr)
        Xtr=[paths[i] for i in tr]; Xva=[paths[i] for i in va]
        ytr=y_cls[tr]; yva=y_cls[va]
        regtr=score[tr]; regva=score[va]

        # Class weights handle imbalance.
        cw=compute_class_weight(class_weight="balanced",classes=np.unique(ytr),y=ytr)
        class_weight={int(c):float(w) for c,w in zip(np.unique(ytr),cw)}
        clf=build_firenet("softmax")
        clf.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                    loss="sparse_categorical_crossentropy",metrics=["accuracy"])
        clf.fit(make_dataset(Xtr,ytr,batch_size,True),validation_data=make_dataset(Xva,yva,batch_size),
                epochs=epochs,verbose=2,class_weight=class_weight)
        p=clf.predict(make_dataset(Xva,np.zeros(len(Xva),np.int32),batch_size),verbose=0)
        pred=p.argmax(1)
        clf.save(model_dir/f"severity_classifier_fold{fold}.keras")
        if fold == 1:
            save_severity_gradcam(clf, Xva, Path(out)/"gradcam", max_images=6)

        # Ablation: labels generated from thermal area only (no temperature terms).
        area_vals=fold_df["fire_area_ratio"].fillna(0).values
        alo=float(np.percentile(area_vals[tr],5)); ahi=float(np.percentile(area_vals[tr],95))
        area_score=robust_norm(area_vals,alo,ahi)
        area_y, aq1, aq2=tercile_labels(area_score,tr)
        area_weights=compute_class_weight(class_weight="balanced",
                                          classes=np.unique(area_y[tr]),y=area_y[tr])
        area_cw={int(c):float(w) for c,w in zip(np.unique(area_y[tr]),area_weights)}
        area_model=build_firenet("softmax")
        area_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                           loss="sparse_categorical_crossentropy",metrics=["accuracy"])
        area_model.fit(make_dataset(Xtr,area_y[tr],batch_size,True),
                       validation_data=make_dataset(Xva,area_y[va],batch_size),
                       epochs=epochs,verbose=0,class_weight=area_cw)
        area_pred=area_model.predict(make_dataset(Xva,np.zeros(len(Xva),np.int32),batch_size),verbose=0).argmax(1)
        pd.DataFrame({"frame_id":fire.iloc[va].frame_id.values,
                      "area_only_true":area_y[va],"area_only_pred":area_pred}).to_csv(
                          Path(out)/f"area_only_ablation_fold{fold}.csv",index=False)
        area_model.save(model_dir/f"area_only_fold{fold}.keras")
        del area_model

        if run_vgg16:
            vgg=build_vgg16_severity()
            vgg.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                        loss="sparse_categorical_crossentropy",metrics=["accuracy"])
            vgg.fit(make_dataset(Xtr,ytr,batch_size,True),
                    validation_data=make_dataset(Xva,yva,batch_size),
                    epochs=epochs,verbose=2,class_weight=class_weight)
            vp=vgg.predict(make_dataset(Xva,np.zeros(len(Xva),np.int32),batch_size),verbose=0).argmax(1)
            pd.DataFrame({"frame_id":fire.iloc[va].frame_id.values,
                          "true":yva,"vgg16_pred":vp}).to_csv(
                              Path(out)/f"vgg16_fold{fold}.csv",index=False)
            vgg.save(model_dir/f"vgg16_fold{fold}.keras")
            del vgg

        reg=build_firenet("regression")
        reg.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss="mae")
        reg.fit(make_dataset(Xtr,regtr,batch_size,True),validation_data=make_dataset(Xva,regva,batch_size),
                epochs=epochs,verbose=2)
        rp=reg.predict(make_dataset(Xva,np.zeros(len(Xva),np.float32),batch_size),verbose=0).ravel()
        # Convert regression output to Low/Moderate/High using training score terciles.
        # q1/q2 are score-space thresholds, but score is fold-specific normalized.
        rpred=np.digitize(rp,[q1,q2]).astype(int)
        reg.save(model_dir/f"severity_regressor_fold{fold}.keras")

        rgb_ratio=rgb_fire_ratio_baseline(Xva)
        bq1,bq2=np.quantile(rgb_fire_ratio_baseline(Xtr),[1/3,2/3])
        base_pred=np.digitize(rgb_ratio,[bq1,bq2]).astype(int)

        oof_true.extend(yva); oof_pred.extend(pred); oof_reg.extend(rp); oof_score.extend(regva)
        for j,idx in enumerate(va):
            all_rows.append(dict(frame_id=fire.iloc[idx].frame_id,video_id=fire.iloc[idx].video_id,
                                 fold=fold,teacher_score=float(score[idx]),severity_class=int(y_cls[idx]),
                                 predicted_class=int(pred[j]),regression_score=float(rp[j]),
                                 rgb_ratio=float(rgb_ratio[j]),rgb_rule_class=int(base_pred[j])))
        print(f"Fold {fold}: classifier macro-F1={f1_score(yva,pred,average='macro'):.4f}; "
              f"RGB-rule macro-F1={f1_score(yva,base_pred,average='macro'):.4f}; "
              f"reg MAE={mean_absolute_error(regva,rp):.4f}; Spearman={spearmanr(regva,rp).statistic:.4f}")

    y=np.asarray(oof_true); pred=np.asarray(oof_pred); rp=np.asarray(oof_reg); truth_score=np.asarray(oof_score)
    rows=all_rows
    oof_df=pd.DataFrame(rows)
    oof_df.to_csv(Path(out)/"oof_predictions.csv",index=False)
    # Consolidated automatic label file; each row's score/class was produced
    # with normalization and tercile cutoffs fitted only on that row's train fold.
    label_cols=["frame_id","video_id","teacher_score","severity_class"]
    labels=oof_df[label_cols].rename(columns={"teacher_score":"severity_score"})
    thermal_cols=["frame_id","fire_area_ratio","thermal_mean_c","thermal_p95_c",
                  "thermal_max_c","hot_pixel_count","thermal_mask_empty"]
    labels=labels.merge(fire[thermal_cols],on="frame_id",how="left",validate="one_to_one")
    labels.to_csv(Path(out)/"severity_labels_oof.csv",index=False)
    metrics={
      "classifier_macro_f1":float(f1_score(y,pred,average="macro")),
      "classifier_per_class":classification_report(y,pred,target_names=CLASS_NAMES,output_dict=True,zero_division=0),
      "regression_mae":float(mean_absolute_error(truth_score,rp)),
      "regression_spearman":float(spearmanr(truth_score,rp).statistic),
      "rgb_rule_macro_f1":float(f1_score([r["severity_class"] for r in rows],
                                         [r["rgb_rule_class"] for r in rows],average="macro")),
      "confusion_matrix":confusion_matrix(y,pred,labels=[0,1,2]).tolist()
    }
    with open(Path(out)/"metrics.json","w") as f: json.dump(metrics,f,indent=2)
    save_confusion(y,pred,Path(out)/"confusion_matrix.png")
    print(json.dumps(metrics,indent=2))
    return metrics

def export_all_teacher_labels(df, out, threshold_c):
    # CSV uses raw thermal features. Fold-specific normalized score and class
    # thresholds are stored in OOF predictions; do not compute global terciles.
    cols=["frame_id","video_id","rgb_path","thermal_path","binary_label",
          "fire_area_ratio","thermal_mean_c","thermal_p95_c","thermal_max_c",
          "hot_pixel_count","thermal_mask_empty"]
    df[cols].to_csv(Path(out)/"thermal_teacher_features.csv",index=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root",required=True)
    ap.add_argument("--groups_csv",required=True)
    ap.add_argument("--out",default="runs_flame3")
    ap.add_argument("--threshold_c",type=float,default=100.0,
                    help="Operational hot-pixel threshold in Celsius; report sensitivity to this choice.")
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--batch_size",type=int,default=16)
    ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--run_vgg16",action="store_true",
                    help="Also train VGG16 per fold; slower and optional.")
    a=ap.parse_args(); seed_everything(a.seed)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    print("PHASE 0: indexing corrected-FOV RGB / radiometric TIFF pairs")
    df=make_manifest(a.data_root)
    print("PHASE 0 DATA AUDIT")
    print(df.groupby("binary_label").size().rename(index={0:"no_fire",1:"fire"}))
    print("Unique thermal dimensions:",df[["height","width"]].drop_duplicates().values.tolist())
    df=attach_groups(df,a.groups_csv)
    print("Unique recording groups:",df.video_id.nunique())
    print("PHASE 1: thermal teacher features")
    df=add_teacher_features(df,a.threshold_c)
    df.to_csv(out/"paired_teacher_dataset.csv",index=False)
    export_all_teacher_labels(df,out,a.threshold_c)
    print("Thermal mask empty rate (fire subset):",
          float(df[df.binary_label==1].thermal_mask_empty.mean()))
    print("PHASE 2/3: grouped training/evaluation")
    train_grouped(df,out,a.epochs,a.batch_size,a.folds,a.run_vgg16)
    print("Finished. Outputs in",out.resolve())

if __name__=="__main__":
    main()
