"""Check a CVAT export zip (Segmentation mask 1.1), convert it to binary masks, and draw overlays against the pre-label

Usage: python scripts/check_cvat_export.py ccd cvat_export/ccd_cvat.zip
Output:
  gt/<folder>/<name>.png            0 = scalp, 255 = hair
  gt_overlay/<folder>/<name>.jpg    image | pre-label | corrected (green = added, red = removed, yellow = unchanged)
  gt_overlay/<folder>/_report.csv   per-image statistics
"""
import argparse
import csv
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "original"  # original images + initial U2-Net masks (captures/ccd/mimoa/usb-ts)
WORK = ROOT / "work"      # generated files (gt, prelabel, runs, ...); path arguments are relative to work/


def read_labelmap(z):
    colors = {}
    for line in z.read("labelmap.txt").decode().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, rgb = line.split(":")[:2]
        colors[name] = tuple(int(v) for v in rgb.split(","))
    return colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("zip")
    ap.add_argument("--ref", help="folder of starting masks to compare with (default prelabel/<folder>), e.g. pred/noisy_full_s0/<folder>")
    ap.add_argument("--report", default="_report.csv", help="report file name in gt_overlay/<folder>/")
    a = ap.parse_args()

    gt_dir = WORK / "gt" / a.folder
    ov_dir = WORK / "gt_overlay" / a.folder
    ref_dir = WORK / a.ref if a.ref else WORK / "prelabel" / a.folder
    gt_dir.mkdir(parents=True, exist_ok=True)
    ov_dir.mkdir(parents=True, exist_ok=True)

    z = zipfile.ZipFile(WORK / a.zip)
    hair = read_labelmap(z)["hair"]
    names = sorted(n for n in z.namelist() if n.startswith("SegmentationClass/") and n.endswith(".png"))
    expected = {p.stem for p in ref_dir.glob("*.png")}
    got = {Path(n).stem for n in names}
    print(f"masks {len(got)} / expected {len(expected)}")
    print("missing:", sorted(expected - got))
    print("extra:", sorted(got - expected))

    rows = []
    for n in names:
        stem = Path(n).stem
        bgr = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
        rgb = bgr[:, :, ::-1]
        gt = (rgb == hair).all(axis=2)
        other = ~gt & (rgb != 0).any(axis=2)
        pre = cv2.imread(str(ref_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 0
        img = cv2.imread(str(DATA / a.folder / f"{stem}.jpg"))
        assert img is not None and img.shape[:2] == gt.shape == pre.shape, stem
        cv2.imwrite(str(gt_dir / f"{stem}.png"), gt.astype(np.uint8) * 255)

        added, removed = gt & ~pre, pre & ~gt
        union = (gt | pre).sum()
        def tint(mask, color, base):
            base[mask] = (0.35 * base[mask] + 0.65 * np.array(color)).astype(np.uint8)
        left = img.copy()
        mid = img.copy(); tint(pre, (0, 255, 255), mid)
        right = img.copy(); tint(gt & pre, (0, 255, 255), right); tint(added, (0, 255, 0), right); tint(removed, (0, 0, 255), right)
        cv2.imwrite(str(ov_dir / f"{stem}.jpg"), np.hstack([left, mid, right]), [cv2.IMWRITE_JPEG_QUALITY, 92])

        rows.append(dict(name=stem, shape=f"{gt.shape[1]}x{gt.shape[0]}", hair_pct=round(gt.mean() * 100, 2),
                         pre_pct=round(pre.mean() * 100, 2), added_px=int(added.sum()), removed_px=int(removed.sum()),
                         changed_pct=round((added.sum() + removed.sum()) / gt.size * 100, 2),
                         iou_vs_pre=round((gt & pre).sum() / union, 4) if union else 1.0, other_px=int(other.sum())))

    with open(ov_dir / a.report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for r in sorted(rows, key=lambda r: r["changed_pct"]):
        print(f"{r['name']:<34}{r['shape']:>9} hair {r['hair_pct']:>6} pre {r['pre_pct']:>6} +{r['added_px']:>6} -{r['removed_px']:>6} chg {r['changed_pct']:>5}% iou {r['iou_vs_pre']:.3f} other {r['other_px']}")


if __name__ == "__main__":
    main()
