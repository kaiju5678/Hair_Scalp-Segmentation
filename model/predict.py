"""Predict hair masks with U-Net + ResNet34 (hair_unet_r34.pt)

Install: pip install torch segmentation-models-pytorch opencv-python numpy
Usage:   python predict.py <image .jpg or folder> <output folder> [--thr 0.5] [--cpu] [--no-tta]
Output:  <out>/<name>.png  mask, 0 = scalp, 255 = hair (same size as the input image)
         <out>/<name>_overlay.jpg  image | predicted hair (yellow)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


@torch.no_grad()
def predict(model, bgr, dev, tta=True):
    h, w = bgr.shape[:2]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32  # U-Net needs sizes divisible by 32
    img = cv2.copyMakeBorder(bgr, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    x = torch.from_numpy(((img[:, :, ::-1].astype(np.float32) - MEAN) / STD).transpose(2, 0, 1).copy())[None].to(dev)
    flips = [[], [3], [2], [2, 3]] if tta else [[]]
    acc = 0
    for f in flips:
        xi = x.flip(f) if f else x
        p = torch.sigmoid(model(xi).float())
        acc = acc + (p.flip(f) if f else p)
    return (acc / len(flips))[0, 0, :h, :w].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--weights", default=str(Path(__file__).with_name("hair_unet_r34.pt")))
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-tta", action="store_true")
    a = ap.parse_args()
    dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    ck = torch.load(a.weights, map_location=dev, weights_only=False)
    model = getattr(smp, ck["arch"])(ck["encoder"], encoder_weights=None, classes=1)
    model.load_state_dict(ck["state"])
    model.to(dev).eval()
    src = Path(a.src)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if src.is_dir() else [src]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in files:
        bgr = cv2.imread(str(p))
        m = predict(model, bgr, dev, not a.no_tta) > a.thr
        cv2.imwrite(str(out / f"{p.stem}.png"), m.astype(np.uint8) * 255)
        o = bgr.copy()
        o[m] = (0.45 * o[m] + 0.55 * np.array((0, 220, 220))).astype(np.uint8)
        cv2.imwrite(str(out / f"{p.stem}_overlay.jpg"), np.hstack([bgr, o]))
        print(p.name, f"hair {m.mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
