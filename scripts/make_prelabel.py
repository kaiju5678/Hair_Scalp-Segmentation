"""Make pre-labels (0/255 masks) from the initial masks + black-hat filtering, with overlay images for checking

Steps per image:
  1. threshold the original mask (values 0-255) at --thr
  2. closing + fill small holes inside hair (white/translucent hair is often detected only at its two edges)
  3. remove components smaller than --min-area (rough specks on the scalp)
  4. black-hat to find missed dark hair; keep only thin, elongated components (--bh-min-len, --bh-min-aspect)
     that do not touch the mask from step 3 (so existing hair edges are not widened)

Usage: python scripts/make_prelabel.py [--list groups/normal.txt] [--thr 64] ...
Output:
  prelabel/<folder>/<name>.png   mask 1 channel: 0 = scalp, 255 = hair
  overlay/<folder>/<name>.jpg    [image | original mask >= thr | pre-label]
                                 red = from the original mask, yellow = added by black-hat, cyan = removed (small specks)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "original"  # original images + initial U2-Net masks (captures/ccd/mimoa/usb-ts)
WORK = ROOT / "work"      # generated files (gt, prelabel, runs, ...); path arguments are relative to work/

RED, YELLOW, CYAN = (0, 0, 255), (0, 255, 255), (255, 255, 0)


def keep_components(b, pred):
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = pred(st[1:])
    return keep[lab].astype(np.uint8)


def keep_thin_lines(b, min_len, min_aspect):
    """Keep only thin, elongated components that look like hair (using minAreaRect); drop skin-texture blobs"""
    out = np.zeros_like(b)
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        long, short = max(w, h), max(min(w, h), 1.0)
        if long >= min_len and long / short >= min_aspect:
            cv2.drawContours(out, [c], -1, 1, thickness=cv2.FILLED)
    return out & b


def fill_small_holes(b, max_hole):
    holes = keep_components(1 - b, lambda st: st[:, cv2.CC_STAT_AREA] <= max_hole)
    return b | holes


def prelabel(img, mask, a):
    b = (mask >= a.thr).astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.close, a.close)))
    b = fill_small_holes(b, a.max_hole)
    base = keep_components(b, lambda st: st[:, cv2.CC_STAT_AREA] >= a.min_area)
    removed = b & (1 - base)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.bh_kernel, a.bh_kernel))
    cand = (cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k) >= a.bh_thr).astype(np.uint8)
    cand = keep_thin_lines(cand, a.bh_min_len, a.bh_min_aspect)
    near = cv2.dilate(base, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(cand, connectivity=8)
    touching = np.unique(lab[(near > 0) & (cand > 0)])
    added = (cand > 0) & ~np.isin(lab, touching)

    return base, added.astype(np.uint8), removed


def tint(img, b, color, alpha=0.55):
    out = img.copy()
    out[b > 0] = ((1 - alpha) * out[b > 0] + alpha * np.array(color)).astype(np.uint8)
    return out


def label(im, text):
    cv2.putText(im, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(im, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="groups/normal.txt")
    ap.add_argument("--thr", type=int, default=64)
    ap.add_argument("--close", type=int, default=5, help="closing kernel size (px)")
    ap.add_argument("--max-hole", type=int, default=400, help="fill holes smaller than this (px)")
    ap.add_argument("--min-area", type=int, default=100, help="remove components smaller than this (px)")
    ap.add_argument("--bh-kernel", type=int, default=15)
    ap.add_argument("--bh-thr", type=int, default=20)
    ap.add_argument("--bh-min-len", type=int, default=40, help="minimum length (bbox diagonal) of strands added by black-hat")
    ap.add_argument("--bh-min-aspect", type=float, default=5.0, help="minimum length/width ratio of strands added by black-hat")
    a = ap.parse_args()

    names = (WORK / a.list).read_text(encoding="utf-8").split()
    for i, name in enumerate(names, 1):
        src = DATA / name
        img = cv2.imread(str(src))
        mask = cv2.imread(str(src.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
        base, added, removed = prelabel(img, mask, a)
        final = base | added

        out_mask = WORK / "prelabel" / Path(name).with_suffix(".png")
        out_mask.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_mask), final * 255)

        orig = tint(img, mask >= a.thr, RED)
        pre = tint(tint(tint(img, base, RED), added, YELLOW), removed, CYAN)
        sheet = np.hstack([label(img.copy(), "image"), label(orig, f"original >= {a.thr}"), label(pre, "pre-label")])
        out_ov = WORK / "overlay" / Path(name).with_suffix(".jpg")
        out_ov.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_ov), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[{i}/{len(names)}] {name}  hair={final.mean():.3f}  +bh={added.sum()}px  -small={removed.sum()}px")


if __name__ == "__main__":
    main()
