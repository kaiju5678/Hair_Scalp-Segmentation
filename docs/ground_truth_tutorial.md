# Ground-Truth Tutorial

This guide explains how the hair ground truth (GT) for the 198 scalp images was built. The approach does not label from blank images. It corrects existing masks, using the following steps:

```
initial U²-Net masks
   │
   ├─ normal images (149) ──► rule-based pre-label ──► manual correction in CVAT ──┐
   │                                                                                ├─► GT v1 (198 images)
   └─ high-noise images (49) ──► prediction of a first U-Net ──► manual correction ┘
                                                                                     │
                       5-fold CV predictions ──► review low-IoU images ──► GT v2 (filled hair the annotator missed)
```

All commands run from the repository root. Path arguments are relative to `work/`, and the scripts expect the images and initial masks in `original/<camera>/<name>.jpg|png` (see the README).

---

## 1. Labeling rules

Agree on these rules before labeling. Every correction follows them.

- Two classes: **hair** and **scalp** (background).
- **White and translucent hair count as hair.** The initial masks miss them most often.
- Short, newly grown hair counts as hair.
- Out-of-focus hair counts as hair if its path can still be traced. If it is too blurred to draw, leave it out. The GT keeps only clear strands.
- Masks follow the hair edge **tightly**. Do not add a margin around blurred edges.
- Dandruff or flakes stuck to a hair are **scalp**. Paint only the hair itself.

## 2. Split normal and high-noise images

```bash
python scripts/split_noise.py
```

The script thresholds each initial mask at 64 and measures `speckle`: the share of the image covered by connected components smaller than 150 px. Hair forms long strands, whereas noise and flaky skin make U²-Net predict many small specks. An image with `speckle > 0.02` goes to the high-noise group.

Output: `work/groups/normal.txt`, `work/groups/noisy.txt`, and `work/groups/noise_report.csv`. Check the report by eye. An image with a large noise patch can fall below the cut-off, and you can move it to `noisy.txt` by hand.

## 3. Make pre-labels for normal images

```bash
python scripts/make_prelabel.py --list groups/normal.txt
```

For each image, the script runs these steps:

1. Threshold the initial mask at 64.
2. Close with a 5 px kernel and fill holes smaller than 400 px. White hair is often detected only at its two edges, and this step fills the middle.
3. Remove components smaller than 100 px (specks on the scalp).
4. Apply a black-hat filter (15 px kernel) to find dark strands that U²-Net missed. The script keeps only thin, elongated pieces (length ≥ 40 px, length/width ≥ 5) that do not touch the existing mask.

Black-hat filtering cannot find white hair. The annotator adds white hair in CVAT.

Check `work/overlay/<camera>/*.jpg` (image | initial mask | pre-label). In the overlay, red means the hair came from the initial mask, yellow means black-hat added it, and cyan means the speck was removed.

## 4. Correct the masks in CVAT

### 4.1 Prepare the upload files

```bash
python scripts/make_cvat_import.py --list groups/normal.txt
```

The script writes, per camera:
- `work/cvat_upload/<camera>/images/*.jpg`: the images used to create the task
- `work/cvat_upload/<camera>_prelabel.zip`: the pre-labels in **Segmentation mask 1.1** format. All the hair in one image is a single object, so it opens as one editable mask.

### 4.2 Create a task and import the pre-labels

1. In [CVAT](https://www.cvat.ai), create a task for each camera with one label, **`hair`**, and upload the images from `cvat_upload/<camera>/images/`.
2. Open the task, then select **Actions → Upload annotations** → format **Segmentation mask 1.1** → choose `<camera>_prelabel.zip`.
   **Clear "Convert masks to polygons"**. Otherwise the masks turn into polygons that are hard to edit.

### 4.3 Edit

- To enter edit mode, **Shift + double-click** the mask (or right-click → ⋮ → Edit).
- Use **Brush** to add and **Eraser** to remove. **Polygon-plus / Polygon-minus** add or remove an area point by point, which is often easier than dragging.
- Press **N** (or ✓) to finish editing, then **Ctrl+S** to save. Save on **every** image.
- Press **F** (or the **>** button) to go to the next image.
- To make hidden hair easier to see, lower the mask opacity under **Appearance → Opacity**.
- For white or translucent hair, **AI Tools → Segment Anything** can give a starting shape. Always check its edges.
- Do not open the same job in several tabs or on several machines at the same time.
- If saving fails with `could not save annotations` / `no available server`, the CVAT server is temporarily down. Keep the tab open, do not refresh, stop editing, and press Ctrl+S again every 5–10 minutes.

Review every camera a second time after the first pass.

### 4.4 Export and back up

- Export annotations as **Segmentation mask 1.1** and save the file as `work/cvat_export/<camera>_cvat.zip`.
- Also export a full backup (dataset with images) and keep it outside CVAT.

## 5. Check the export and write the GT

```bash
python scripts/check_cvat_export.py <camera> cvat_export/<camera>_cvat.zip
```

The script converts the export into binary masks in `work/gt/<camera>/<name>.png` (0 = scalp, 255 = hair) and writes overlays to `work/gt_overlay/<camera>/` (image | pre-label | corrected). In the overlays, green means added, red means removed, and yellow means unchanged. It also writes a `_report.csv` with per-image statistics.

The script prints missing and extra images, then one line per image sorted by how much changed (`chg`). `other` counts pixels with an unexpected label color and should be 0. **Look at images with `chg 0.0%` at the top of the list.** An image identical to the pre-label usually means it was skipped or not saved. Open it again in CVAT. If it really needs no change, confirm that by eye.

## 6. High-noise images: model-assisted labeling

For high-noise images, the pre-label would need too many corrections. Instead, the starting mask comes from a model trained on the corrected normal images.

```bash
python scripts/make_split.py                         # split the labeled images by capture spot
python scripts/train.py --name first_full --aug full --epochs 120
python scripts/evaluate.py runs/first_full --list groups/noisy.txt --out pred/noisy_first
python scripts/make_cvat_import.py --list groups/noisy.txt --mask-dir pred/noisy_first \
       --out cvat_upload_noisy --suffix modelpred
```

Create new CVAT tasks (e.g. `<camera>_noisy`) from `cvat_upload_noisy/<camera>/images/`. Upload `cvat_upload_noisy/<camera>_modelpred.zip` in the same way, with "Convert masks to polygons" cleared. Correct the masks, and export them under a **different name** so the normal export is not overwritten:

```bash
python scripts/check_cvat_export.py <camera> cvat_export/<camera>_noisy_cvat.zip \
       --ref pred/noisy_first/<camera> --report _report_noisy.csv
```

> **Check these images as carefully as the normal images, even if the model looks good.** Any model error you keep becomes GT, and the model is then partly scored against its own output. In this project, the model that produced the starting masks scored 0.972 on high-noise test images, while two other seeds of the same recipe scored 0.926 and 0.930. Look especially at blurred out-of-focus strands, thin hair tips, and noise specks that the model may have marked as hair.

At this point `work/gt/` contains the GT for all images (**GT v1**).

## 7. Optional: GT v2 from cross-validation review

After cross-validation, every image has a prediction from a model that never saw it. Where the model finds real hair that the annotator missed, you can add that hair to the GT.

```bash
python scripts/make_cv_split.py
bash scripts/run_cv.sh
python scripts/collect_gt_review.py
```

`work/gt_review/<camera>/<rank>_iou<x>_<name>.jpg` shows image | GT | model, sorted from the lowest IoU. In the model panel, green means correct, **red means predicted by the model but not in the GT**, and blue means missed. Review the low-IoU images. If most red areas are real hair (blurred strands, gaps in a strand, white hair), copy those rows of `work/gt_review/review.csv` (columns `rank,camera,name,fold`) to a new CSV. This project used IoU < 0.80, which selected 32 images. Then run:

```bash
python scripts/make_gt_v2.py --list gt_review/<your_selection>.csv
```

For each selected image, the script takes the red pixels from the out-of-fold model. It drops a 2 px rim (where the model draws hair slightly wider than the GT), keeps pieces of at least 50 px, and grows them back so they join the existing strand. It backs up `work/gt/` to `work/gt_v1/` and writes `work/gt_v2/`. It does not change `work/gt/`.

Review `work/gt_v2_overlay/` (image | GT v1 | GT v2, where yellow means original and purple means added). In this project, the model wrote on the mi.moa watermark in three images. Those images are listed in `WATERMARK_IMAGES` in `make_gt_v2.py`, so nothing is added inside the watermark box. The list is per image, because other images have real hair crossing the watermark.

> GT v2 is built from model output, so scores measured on it are slightly optimistic. Report it separately, and compare augmentation settings on GT v1, because the additions came from the full-augmentation model.

## 8. Summary of outputs

| Folder (in `work/`) | Content |
|---|---|
| `groups/` | normal / noisy lists and the noise report |
| `prelabel/`, `overlay/` | pre-labels and their check images |
| `cvat_upload/`, `cvat_upload_noisy/` | images and zips for CVAT |
| `cvat_export/` | exports downloaded from CVAT |
| `gt/`, `gt_overlay/` | GT v1 masks (0/255) and check images |
| `gt_review/`, `gt_v2/`, `gt_v2_overlay/` | CV review images, GT v2, and its check images |
