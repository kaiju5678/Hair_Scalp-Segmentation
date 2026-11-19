"""Demo: compare the old model (U2-Net) with our model (U-Net + ResNet34) on scalp images.

Run from the repository root:
    python demo/run_demo.py
Needs external/ScalpVision (git clone) and weight/u2net_tar_min.pth for the U2-Net side,
and model/hair_unet_r34.pt for our side (see README.md).

Reads every .jpg/.png in demo/images/ and writes demo/results/<name>_compare.jpg:
    original | old model: U2-Net (red) | new model: ours (green)
plus demo/results/all_compare.jpg with every image stacked,
and demo/mask_results/<name>_mask_compare.png:
    U2-Net mask | our mask | difference (red = only U2-Net, green = only ours, white = both)

Put your own images in demo/images/ (no images are included in this repository).
Results on images without ground truth can only be compared by eye.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent
IMAGES = DEMO / "images"
RESULTS = DEMO / "results"
MASK_RESULTS = DEMO / "mask_results"
U2NET_CODE = ROOT / "external" / "ScalpVision" / "segmentation"   # U2-Net code from the ScalpVision repo
U2NET_WEIGHTS = ROOT / "weight" / "u2net_tar_min.pth"              # weights that made the dataset's original masks
OURS_DIR = ROOT / "model"                                          # our model: predict.py + hair_unet_r34.pt
U2NET_THR = 64      # same threshold used for the original masks in the report
OURS_THR = 0.5

sys.path.insert(0, str(U2NET_CODE))
sys.path.insert(0, str(OURS_DIR))
from data_loader import RescaleT, SalObjDataset, ToTensorLab  # noqa: E402
from model import U2NET  # noqa: E402
from torchvision import transforms  # noqa: E402
import predict as ours  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402


def load_models(dev):
    u2 = U2NET(3, 1)
    u2.load_state_dict(torch.load(U2NET_WEIGHTS, map_location=dev))
    u2.to(dev).eval()
    ck = torch.load(OURS_DIR / "hair_unet_r34.pt", map_location=dev, weights_only=False)
    net = getattr(smp, ck["arch"])(ck["encoder"], encoder_weights=None, classes=1)
    net.load_state_dict(ck["state"])
    net.to(dev).eval()
    return u2, net


@torch.no_grad()
def u2net_mask(u2, path, dev):
    """Same steps as ScalpVision's u2net_test.py: resize to 320, predict, normalize, resize back, threshold 64."""
    ds = SalObjDataset(img_name_list=[str(path)], lbl_name_list=[],
                       transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
    x = ds[0]["image"][None].float().to(dev)
    d = u2(x)[0][0, 0]
    d = (d - d.min()) / (d.max() - d.min())
    h, w = cv2.imread(str(path)).shape[:2]
    from PIL import Image
    im = Image.fromarray(d.cpu().numpy() * 255).convert("RGB").resize((w, h), resample=Image.BILINEAR)
    return np.array(im)[:, :, 0] >= U2NET_THR


def tint(bgr, m, color):
    o = bgr.copy()
    o[m] = (0.45 * o[m] + 0.55 * np.array(color)).astype(np.uint8)
    return o


def label(img, text, color=(255, 255, 255)):
    bar = np.full((44, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    RESULTS.mkdir(exist_ok=True)
    MASK_RESULTS.mkdir(exist_ok=True)
    files = sorted(p for p in IMAGES.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        print(f"no images in {IMAGES}")
        return
    u2, net = load_models(dev)
    rows = []
    print(f"{'image':<48}{'old U2-Net':>14}{'new (ours)':>14}")
    for p in files:
        bgr = cv2.imread(str(p))
        t0 = time.time(); old = u2net_mask(u2, p, dev); t_old = time.time() - t0
        t0 = time.time(); new = ours.predict(net, bgr, dev, tta=True) > OURS_THR; t_new = time.time() - t0
        gap = np.full((bgr.shape[0] + 44, 10, 3), 255, np.uint8)
        row = np.hstack([label(bgr, "original"), gap,
                         label(tint(bgr, old, (40, 40, 230)), f"old: U2-Net  hair {old.mean() * 100:.1f}%", (120, 140, 255)), gap,
                         label(tint(bgr, new, (60, 200, 60)), f"new: ours  hair {new.mean() * 100:.1f}%", (120, 230, 120))])
        cv2.imwrite(str(RESULTS / f"{p.stem}_compare.jpg"), row, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append(row)
        # masks only: old | new | difference (red = only U2-Net, green = only ours, white = both)
        diff = np.zeros_like(bgr)
        diff[old & new] = (255, 255, 255)
        diff[old & ~new] = (40, 40, 230)
        diff[new & ~old] = (60, 200, 60)
        mrow = np.hstack([label(np.repeat(old[:, :, None] * 255, 3, 2).astype(np.uint8), "old: U2-Net mask"), gap,
                          label(np.repeat(new[:, :, None] * 255, 3, 2).astype(np.uint8), "new: our mask"), gap,
                          label(diff, "red = only old  green = only new")])
        cv2.imwrite(str(MASK_RESULTS / f"{p.stem}_mask_compare.png"), mrow)
        print(f"{p.name[:47]:<48}{old.mean() * 100:>9.1f}% {t_old:4.1f}s{new.mean() * 100:>8.1f}% {t_new:4.1f}s")
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 10, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    cv2.imwrite(str(RESULTS / "all_compare.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"saved {len(files)} comparisons + all_compare.jpg to {RESULTS}")
    print(f"saved {len(files)} mask comparisons to {MASK_RESULTS}")


if __name__ == "__main__":
    main()
