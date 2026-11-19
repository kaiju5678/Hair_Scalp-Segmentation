"""Combine 5-fold cross-validation results (runs/cv5_<aug>_f<k>/eval_test.csv) into one table covering every image

Usage (.venv): python scripts/summarize_cv.py  -> prints the table + writes work/runs/cv5_summary.txt
Pooled IoU = sum pixels over all images first, then compute (same as evaluate.py); valid because each image is in test for one fold only
u2net_orig / prelabel are not trained, so they do not depend on the fold (computed from the same rows)
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
K = 5
AUGS = ["full", "none", "geom"]
SUBSETS = ["all", "normal", "noisy", "ccd", "mimoa", "captures", "usb-ts"]


def pooled(rows, m):
    tp = sum(int(r[f"{m}_tp"]) for r in rows)
    fp = sum(int(r[f"{m}_fp"]) for r in rows)
    fn = sum(int(r[f"{m}_fn"]) for r in rows)
    return tp / max(tp + fp + fn, 1), tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def sub(rows, s):
    return [r for r in rows if s == "all" or r["group"] == s or r["camera"] == s]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="e.g. _v2 -> reads eval_test_v2.csv, writes cv5_summary_v2.txt")
    tag = ap.parse_args().tag
    res = {}
    for aug in AUGS:
        paths = [WORK / "runs" / f"cv5_{aug}_f{k}" / f"eval_test{tag}.csv" for k in range(K)]
        if all(p.exists() for p in paths):
            res[aug] = [dict(r, fold=k) for k, p in enumerate(paths) for r in csv.DictReader(open(p, encoding="utf-8"))]
    if not res:
        print("no augmentation setting has finished all 5 folds yet")
        return
    base = next(iter(res.values()))
    L = [f"5-fold cross-validation (group-aware) — runs/cv5_<aug>_f<k>, seed 0, {len(base)} images "
         f"(every image scored by a model that never saw it)", "pooled-pixel IoU" + (f"  [GT {tag}]" if tag else ""), ""]
    L.append(f"{'method':12s} " + " ".join(f"{s + '(' + str(len(sub(base, s))) + ')':>13s}" for s in SUBSETS))
    lines = [(f"model_{a}", rows, "model") for a, rows in res.items()]
    lines += [("u2net_orig", base, "u2net_orig")]
    for name, rows, m in lines:
        L.append(f"{name:12s} " + " ".join(f"{pooled(sub(rows, s), m)[0]:13.3f}" for s in SUBSETS))
    pl = [r for r in base if r["prelabel_iou"] != ""]
    L.append(f"{'prelabel':12s} {pooled(pl, 'prelabel')[0]:13.3f}  (normal images with a pre-label only, n={len(pl)})")
    L += ["", "Precision / Recall (all)"]
    for name, rows, m in lines:
        _, p, r = pooled(rows, m)
        L.append(f"{name:12s} Prec {p:.3f}  Rec {r:.3f}")
    L += ["", "Per-fold IoU (all): fold = " + " ".join(str(k) for k in range(K)) + "   [n images]"]
    for name, rows, m in lines:
        per = [pooled([r for r in rows if r["fold"] == k], m)[0] for k in range(K)]
        L.append(f"{name:12s} " + " ".join(f"{v:.3f}" for v in per) + f"   mean {np.mean(per):.3f} +- {np.std(per):.3f}")
    L.append(f"{'n':12s} " + " ".join(f"{sum(r['fold'] == k for r in base):5d}" for k in range(K)))
    L += ["", "Images where the model beats U²-Net (per-image IoU)"]
    for a, rows in res.items():
        for g in ("all", "normal", "noisy"):
            rs = sub(rows, g)
            win = sum(float(r["model_iou"]) > float(r["u2net_orig_iou"]) for r in rs)
            L.append(f"model_{a:6s} {g:6s} {win}/{len(rs)}")
    L += ["", "Note: noisy-image GT was corrected from the first-round model prediction (abl_full_s0) -> noisy scores lean toward the model",
          "      normal-image GT was corrected from the U²-Net pre-label -> normal scores lean toward U²-Net"]
    txt = "\n".join(L)
    (WORK / "runs" / f"cv5_summary{tag}.txt").write_text(txt, encoding="utf-8")
    import sys; sys.stdout.reconfigure(encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
