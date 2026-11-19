"""Split images into normal / noisy groups by how speckled the initial mask is

- speckle    = area fraction of small specks (connected components < MIN_AREA px) in the mask thresholded at >= 64
               hair forms long connected strands, whereas scalp noise breaks into many small specks
- color_noise = mean of |a,b channels (Lab) - 5x5 median blur| of the original image, used to tell the cause
               high -> sensor color noise; low but high speckle -> flaky skin / reflections

Usage: python scripts/split_noise.py [--speckle-thr 0.02] [--color-thr 1.944]
Output: groups/noise_report.csv, groups/normal.txt, groups/noisy.txt  (image files are not moved or changed)
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "original"  # original images + initial U2-Net masks (captures/ccd/mimoa/usb-ts)
WORK = ROOT / "work"      # generated files (gt, prelabel, runs, ...); path arguments are relative to work/
FOLDERS = ["captures", "ccd", "mimoa", "usb-ts"]
MASK_THR = 64
MIN_AREA = 150


def color_noise(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return sum(np.abs(lab[..., c].astype(np.float32) - cv2.medianBlur(lab[..., c], 5)).mean() for c in (1, 2))


def speckle(mask_gray):
    b = (mask_gray >= MASK_THR).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return areas[areas < MIN_AREA].sum() / b.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speckle-thr", type=float, default=0.02, help="noisy if speckle > this value")
    ap.add_argument("--color-thr", type=float, default=1.944, help="color_noise > this value means the cause is sensor noise")
    args = ap.parse_args()

    rows = []
    for folder in FOLDERS:
        for p in sorted((DATA / folder).glob("*.jpg")):
            img = cv2.imread(str(p))
            mask = cv2.imread(str(p.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
            cn, sp = color_noise(img), speckle(mask)
            noisy = sp > args.speckle_thr
            cause = ("sensor_noise" if cn > args.color_thr else "skin_texture") if noisy else ""
            rows.append({"image": f"{folder}/{p.name}", "folder": folder, "speckle": sp,
                         "color_noise": cn, "group": "noisy" if noisy else "normal", "cause": cause})
    rows.sort(key=lambda r: -r["speckle"])

    out = WORK / "groups"
    out.mkdir(exist_ok=True)
    with open(out / "noise_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "folder", "speckle", "color_noise", "group", "cause"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "speckle": f"{r['speckle']:.4f}", "color_noise": f"{r['color_noise']:.3f}"})
    for g in ("normal", "noisy"):
        (out / f"{g}.txt").write_text("\n".join(r["image"] for r in rows if r["group"] == g) + "\n", encoding="utf-8")

    noisy = [r for r in rows if r["group"] == "noisy"]
    print(f"normal={len(rows) - len(noisy)} noisy={len(noisy)}")
    for key in ("folder", "cause"):
        counts = {}
        for r in noisy:
            counts[r[key]] = counts.get(r[key], 0) + 1
        print(f"  noisy by {key}: {counts}")


if __name__ == "__main__":
    main()
