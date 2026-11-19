# Hair Segmentation for Scalp Microscope Images

Ground-truth construction and U-Net hair segmentation for scalp microscope (trichoscopy) images, with a comparison of data augmentation settings.

The dataset came with hair masks from the U²-Net stage of [ScalpVision](https://github.com/winston1214/ScalpVision). Those masks miss white and translucent hair, and on noisy images they turn sensor noise and flaky skin into false hair. This project:

1. refines those masks into ground truth for 198 images from four cameras (rule-based pre-labels → manual correction in CVAT → model-assisted labeling for high-noise images), and
2. trains a **U-Net with an ImageNet-pretrained ResNet34 encoder** on that ground truth, comparing no augmentation, geometric augmentation, and geometric + color/noise augmentation.

This work was done during the Global Lab Internship at Daegu Catholic University (ANCHOR AX Innovation Talent Project).

## Results

Group-aware 5-fold cross-validation (every image is scored by a model that never saw it; images from the same capture spot stay in the same fold), full augmentation, IoU:

| Method | All (197) | Normal (148) | High-noise (49) |
|---|---|---|---|
| Original U²-Net masks (threshold 64) | 0.678 | 0.730 | 0.551 |
| **Ours (U-Net + ResNet34)** | **0.858** | **0.836** | **0.935** |

On normal images the model is better than U²-Net on 137 of 148 images.

Fixed 53-image test set, compared with the full ScalpVision pipeline (U²-Net + SAM ViT-H):

| Method | All (53) | Normal (32) | High-noise (21) |
|---|---|---|---|
| U²-Net | 0.643 | 0.723 | 0.553 |
| ScalpVision | 0.746 | 0.744 | 0.748 |
| **Ours** | **0.867** | **0.824** | **0.937** |

Augmentation (CV IoU, all / high-noise): none 0.840 / 0.900, geom 0.854 / 0.926, full 0.858 / 0.935.

> **Read the high-noise numbers with care.** Ground truth for high-noise images was corrected from masks predicted by an earlier version of our model, so those scores favor our model. Normal-image ground truth was corrected from U²-Net-based pre-labels, so those scores favor U²-Net. The normal-image CV result (0.836 vs 0.730) is therefore the fairest single comparison.

## Repository layout

```
model/      predict.py (stand-alone inference), config.json (training settings)
scripts/    ground-truth tools, training, evaluation, cross-validation, ScalpVision comparison
demo/       side-by-side demo: U²-Net vs our model on your own images
docs/       ground_truth_tutorial.md: how the ground truth was built, step by step
```

The dataset images, the ground-truth masks, and the model weights are **not** stored in this repository.

## Quick start: predict hair masks

0. (Recommended) Create a virtual environment so these packages stay separate from your other projects:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scriptsctivate
   # macOS / Linux
   source .venv/bin/activate
   ```

1. Install the packages (Python 3.10+; install the PyTorch build that matches your GPU from [pytorch.org](https://pytorch.org) first if you want CUDA):

   ```bash
   pip install torch segmentation-models-pytorch opencv-python numpy
   ```

2. Download the weights **`hair_unet_r34.pt`** (98 MB) from the Google Drive folder below and put the file in `model/`:

   **[Google Drive: hair_unet_r34.pt](https://drive.google.com/drive/folders/1KvK0v7hwC6VB-hQR_bGfhpi8M380DOM6?usp=sharing)**

3. Run:

   ```bash
   python model/predict.py path/to/image_or_folder output_folder
   ```

   For each image this writes `<name>.png` (0 = scalp, 255 = hair, same size as the input) and `<name>_overlay.jpg` (image | predicted hair in yellow).
   Options: `--thr 0.5` probability threshold, `--cpu` to force CPU, `--no-tta` to skip flip test-time augmentation (faster, slightly less accurate).

The model was trained on 640×480 scalp microscope images. It also follows hair in close-up clinical photographs, but it does not work on distant photographs of the head.

## Training and evaluation

Inside the same virtual environment, install the full environment (tested with Python 3.12, PyTorch 2.14 + CUDA 13.0 on a 4 GB laptop GPU):

```bash
pip install -r requirements.txt
```

The scripts expect this layout next to `scripts/`:

```
original/<camera>/<name>.jpg   images (640x480)
original/<camera>/<name>.png   initial U²-Net masks (0-255)
work/                          everything the scripts create (gt/, prelabel/, runs/, splits/, ...)
```

Path arguments such as `runs/x` or `splits/split.csv` are relative to `work/`. Cameras used here: `ccd`, `mimoa`, `captures`, `usb-ts`.

| Step | Command |
|---|---|
| Split normal / high-noise | `python scripts/split_noise.py` |
| Build the ground truth | see [docs/ground_truth_tutorial.md](docs/ground_truth_tutorial.md) |
| Train / val / test split by capture spot | `python scripts/make_split.py` |
| Train one model | `python scripts/train.py --name unet_r34 --aug full --epochs 120` |
| Evaluate on the test split | `python scripts/evaluate.py runs/unet_r34 --split test` |
| Augmentation comparison (3 settings × 3 seeds) | `PREFIX=abl198 bash scripts/run_aug_ablation.sh` → `python scripts/summarize_ablation.py abl198` |
| 5-fold cross-validation | `python scripts/make_cv_split.py` → `bash scripts/run_cv.sh` → `python scripts/summarize_cv.py` |
| Compare with ScalpVision | `python scripts/compare_scalpvision.py --name test53 original/ccd/x.jpg ...` (needs the setup below) |

Training settings: BCE + Dice loss, AdamW (lr 5×10⁻⁴, weight decay 10⁻⁴), one-cycle schedule, 120 epochs, mixed precision, best validation IoU checkpoint, flip TTA at inference. About 9 s per epoch on an RTX 3050 Laptop GPU.

### ScalpVision comparison and demo

`scripts/compare_scalpvision.py` and `demo/run_demo.py` reuse code and weights from ScalpVision:

```bash
git clone https://github.com/winston1214/ScalpVision external/ScalpVision
pip install scikit-image segment-anything matplotlib
```

Put `u2net_tar_min.pth` (ScalpVision's U²-Net weights, [Hugging Face](https://huggingface.co/winston1214/scalpvision_hair_segmentation)) and, for the comparison only, `sam_vit_h_4b8939.pth` ([SAM ViT-H](https://github.com/facebookresearch/segment-anything)) in `weight/`. For the demo, place images in `demo/images/` and run `python demo/run_demo.py`.

## Data

The 198 scalp images and the ground-truth masks belong to the host laboratory and are not distributed here. Please contact the laboratory at Daegu Catholic University for access.

## Acknowledgment

Thanks to Prof. Tae-Young Byun and Prof. Doohee Lee for their supervision and guidance, and to the authors of ScalpVision for releasing their code and weights.

## References

- Y. Kim et al., "Scalp Diagnostic System With Label-Free Segmentation and Training-Free Image Translation," MICCAI 2025. [Code](https://github.com/winston1214/ScalpVision)
- X. Qin et al., "U²-Net: Going deeper with nested U-structure for salient object detection," Pattern Recognition, 2020.
- O. Ronneberger et al., "U-Net: Convolutional networks for biomedical image segmentation," MICCAI 2015.
- P. Iakubovskii, [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch), 2019.
- A. Buslaev et al., "Albumentations: Fast and flexible image augmentations," Information, 2020.
