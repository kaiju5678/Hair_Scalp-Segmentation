"""Build GT v2: add the parts the model predicts beyond the GT (red in gt_review) to the existing GT, after filtering
(the supervisor allowed using the model output as GT where the model is better, 2026-11-16)

Uses the model of the CV fold in which the image is a test image (never saw it) -> cv5_full_f<fold>/best.pt
Filtering the red pixels (red = pred & ~gt):
  1. core = red pixels farther than --band px from the GT (drops thin rims where the model draws hair wider than the GT)
  2. keep only core pieces with area >= --min-area px
  3. grow the kept core back by --band px, limited to red, so the added piece connects to the existing strand

Does not touch work/gt/: backs it up to work/gt_v1/ (first run only), then writes work/gt_v2/ (all 198 images + the selected ones filled)
Output: gt_v2/<camera>/<name>.png, gt_v2_overlay/<rank>_<camera>_<name>.jpg (image | GT v1 | GT v2: yellow = original, purple = added)
        gt_v2_overlay/_added.csv

Usage: .venv/Scripts/python.exe scripts/make_gt_v2.py --list gt_review/_iou_below_0.80/review_below_0.80.csv
"""
import argparse
import csv
import shutil

import cv2
import numpy as np

from evaluate import DATA, WORK, load_model, predict_prob

# mi.moa watermark box (x0, y0, x1, y1): nothing is added inside it, only for images where the model drew on the text (checked by eye 2026-11-16)
# Not applied to every image, because mimoa images ranked 5, 7, 12, 16, 26 have real hair crossing the watermark box
WATERMARK = {"mimoa": (570, 440, 630, 470)}
WATERMARK_IMAGES = {"h_2023_12_04_20_43_10_ms", "h_2023_12_04_20_54_03_ms", "h_2023_08_11_10_24_22_t"}  # ranks 15, 17, 22


def fill(gt, pred, band, min_area):
    red = pred & ~gt
    dist = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 5)
    core = red & (dist > band)
    n, lab, st, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(core)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            keep |= lab == i
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 3, 2 * band + 3))
    add = red & (cv2.dilate(keep.astype(np.uint8), k) > 0)
    return add, int(red.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="csv with columns rank,camera,name,fold (relative to work/)")
    ap.add_argument("--band", type=int, default=2)
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()

    v1, v2, ov = WORK / "gt_v1", WORK / "gt_v2", WORK / "gt_v2_overlay"
    if not v1.exists():
        shutil.copytree(WORK / "gt", v1)
        print("backup gt -> gt_v1")
    if v2.exists():
        shutil.rmtree(v2)
    shutil.copytree(v1, v2)
    if ov.exists():
        shutil.rmtree(ov)
    ov.mkdir()

    rows = list(csv.DictReader(open(WORK / a.list, encoding="utf-8-sig")))
    models, out = {}, []
    for r in rows:
        cam, name, fold = r["camera"], r["name"], int(r["fold"])
        if fold not in models:
            models[fold] = load_model(f"runs/cv5_full_f{fold}", "best")
        bgr = cv2.imread(str(DATA / cam / f"{name}.jpg"))
        gt = cv2.imread(str(v1 / cam / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 127
        pred = predict_prob(models[fold], bgr) > a.thr
        add, red_px = fill(gt, pred, a.band, a.min_area)
        if cam in WATERMARK and name in WATERMARK_IMAGES:
            x0, y0, x1, y1 = WATERMARK[cam]
            if add[y0:y1, x0:x1].any():
                print(f"    watermark: drop {int(add[y0:y1, x0:x1].sum())} px")
            add[y0:y1, x0:x1] = False
        new = gt | add
        cv2.imwrite(str(v2 / cam / f"{name}.png"), new.astype(np.uint8) * 255)

        def paint(masks):
            o = bgr.copy()
            for m, c in masks:
                o[m] = (0.45 * o[m] + 0.55 * np.array(c)).astype(np.uint8)
            return o
        vis = np.hstack([bgr, paint([(gt, (0, 220, 220))]),
                         paint([(gt, (0, 220, 220)), (add, (255, 0, 255))])])
        cv2.imwrite(str(ov / f"{int(r['rank']):03d}_{cam}_{name}.jpg"), vis)
        n_add = cv2.connectedComponents(add.astype(np.uint8), connectivity=8)[0] - 1
        out.append({"rank": r["rank"], "camera": cam, "name": name, "fold": fold,
                    "gt_px": int(gt.sum()), "red_px": red_px, "added_px": int(add.sum()),
                    "added_pct_of_gt": round(100 * add.sum() / max(gt.sum(), 1), 1),
                    "added_pct_of_red": round(100 * add.sum() / max(red_px, 1), 1),
                    "added_pieces": n_add})
        print(f"{r['rank']:>3} {cam:8s} {name:40s} +{out[-1]['added_pct_of_gt']}% of GT "
              f"({out[-1]['added_pct_of_red']}% of red, {n_add} pieces)")

    with open(ov / "_added.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(out[0]))
        w.writeheader()
        w.writerows(out)
    tot_gt = sum(o["gt_px"] for o in out); tot_add = sum(o["added_px"] for o in out)
    tot_red = sum(o["red_px"] for o in out)
    print(f"\n{len(out)} images: added {tot_add} px = {100*tot_add/tot_gt:.1f}% of GT, "
          f"{100*tot_add/tot_red:.1f}% of red  (band={a.band}, min_area={a.min_area})")


if __name__ == "__main__":
    main()
