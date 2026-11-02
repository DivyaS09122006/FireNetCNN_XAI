# FLAME 3 thermal-teacher / RGB-only severity extension

## What this implements
- Scans FLAME 3 CVSubset corrected-FOV RGB images and matching Celsius TIFFs.
- Checks that RGB and TIFF dimensions match.
- Extracts thermal hot-pixel area ratio, mean, 95th percentile and max temperature.
- Builds a fold-specific teacher score:
  `score = 0.40 * norm(fire_area_ratio) + 0.30 * norm(thermal_mean_C) + 0.30 * norm(thermal_p95_C)`
- Normalization bounds (5th/95th percentiles) and Low/Moderate/High tercile cutoffs are fitted on each training fold only.
- Trains two RGB-only FireNet variants: 3-class softmax and continuous-score regression.
- Evaluates with GroupKFold, macro-F1, regression MAE/Spearman, RGB-only colour-ratio baseline and confusion matrix.
- Includes an area-only thermal-label ablation, Grad-CAM examples for the High class, and optional fine-tuned VGG16 baseline (`--run_vgg16`).
- Includes a two-stage RGB-only inference helper `predict_two_stage(...)`.
- Saves thermal feature CSV, OOF predictions, metrics and fold models.

## Critical dataset limitation
IEEE DataPort describes the FLAME 3 CV subset as 622 Fire and 116 No Fire image quartets from one prescribed burn. Each quartet has raw RGB, raw thermal JPG, corrected-FOV RGB and radiometric Celsius TIFF. It does **not** document video/session IDs for this CV subset. Therefore, this code requires a genuine `groups.csv` mapping `frame_id,video_id`. Do not invent group IDs from frame names. If you cannot obtain recording/session IDs, you cannot claim video-grouped validation on this subset. Use the larger multi-burn FLAME 3 collection only if you can obtain reliable burn/session group metadata.

## Data layout
The script searches recursively for corrected-FOV RGB images and same-stem TIFFs. It infers Fire/No Fire from directory names. Confirm the generated `paired_manifest.csv` before training.

## Setup
```bash
pip install tensorflow pandas numpy scikit-learn opencv-python tifffile matplotlib scipy
```

## Run
```bash
python flame3_severity.py \
  --data_root "/path/to/Flame 3 Computer Vision Sets" \
  --groups_csv groups.csv \
  --out runs_flame3 \
  --threshold_c 100 \
  --epochs 20 --batch_size 16 --folds 5
```

`groups.csv` must contain:
```csv
frame_id,video_id
00001,session_01
00002,session_01
...
```
Use actual acquisition metadata. The program intentionally stops if it is missing or has fewer than five groups.

## Interpretation / limitations
- 100°C is an operational hot-pixel threshold, not a universal fire boundary. Perform threshold sensitivity analysis (e.g. 80, 100, 150°C).
- Thermal area is calculated as the fraction of the aligned image above threshold; it is not a physical fire-area measurement.
- The score is a thermal-derived proxy. It is not ground-truth fire intensity or operational fire severity.
- Camera distance, altitude, view angle, occlusion and prescribed-burn conditions affect the estimate.
- The CV subset is one prescribed burn; generalization to wildfires is unproven.
- RGB colour-ratio baseline is crude and must not be presented as a strong segmentation method.
- VGG16/Inception baseline and Grad-CAM are not included in this fast starter script; add only after the grouped protocol and labels are verified.
