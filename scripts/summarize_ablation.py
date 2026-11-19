"""Collect ablation results (none/geom/full x seed) from eval_test*_summary.txt and log.csv into a mean +- std table

Usage (.venv): python scripts/summarize_ablation.py abl198            (reads eval_test_summary.txt)
               python scripts/summarize_ablation.py abl --tag _198    (149-image models scored on the 198-image test set)
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "original"  # original images + initial U2-Net masks (captures/ccd/mimoa/usb-ts)
WORK = ROOT / "work"      # generated files (gt, prelabel, runs, ...); path arguments are relative to work/
AUGS = ["none", "geom", "full"]
SUBSETS = ["all", "normal", "noisy", "ccd", "mimoa", "captures", "usb-ts"]


def parse(path):
    """-> {(subset, method): (n, iou, dice, prec, rec)}"""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([\w-]+)\((\d+)\)\s+(\w+)\s+([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)$", line)
        if m:
            out[(m[1], m[3])] = (int(m[2]), *map(float, m.groups()[3:]))
    return out


def ms(v):
    return f"{np.mean(v):.3f}+-{np.std(v):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    seeds = a.seeds.split(",")
    res, logs = {}, {}
    for aug in AUGS:
        for s in seeds:
            run = WORK / "runs" / f"{a.prefix}_{aug}_s{s}"
            res[(aug, s)] = parse(run / f"eval_test{a.tag}_summary.txt")
            rows = list(csv.DictReader(open(run / "log.csv")))
            best = max(rows, key=lambda r: float(r["val_iou"]))
            logs[(aug, s)] = (int(best["epoch"]), float(rows[-1]["train_loss"]), float(rows[-1]["val_loss"]),
                              float(best["val_iou"]))
    r0 = res[(AUGS[0], seeds[0])]
    n = {sub: r0[(sub, "model")][0] for sub in SUBSETS if (sub, "model") in r0}
    L = [f"runs/{a.prefix}_<aug>_s<seed>  eval_test{a.tag}  seeds {a.seeds}  (pooled-pixel IoU, mean+-std over seeds)", ""]
    L.append(f"{'aug':5s} " + " ".join(f"{s + f'({n[s]})':>15s}" for s in SUBSETS[:3]) + " | "
             + " ".join(f"{s + f'({n[s]})':>13s}" for s in SUBSETS[3:]))
    for aug in AUGS:
        cells = [ms([res[(aug, s)][(sub, "model")][1] for s in seeds]) for sub in SUBSETS[:3]]
        cams = [f"{np.mean([res[(aug, s)][(sub, 'model')][1] for s in seeds]):.3f}" for sub in SUBSETS[3:]]
        L.append(f"{aug:5s} " + " ".join(f"{c:>15s}" for c in cells) + " | " + " ".join(f"{c:>13s}" for c in cams))
    L += ["", "Precision / Recall (all test images, mean over seeds)"]
    for aug in AUGS:
        p = [res[(aug, s)][("all", "model")][3] for s in seeds]
        r = [res[(aug, s)][("all", "model")][4] for s in seeds]
        d = [res[(aug, s)][("all", "model")][2] for s in seeds]
        L.append(f"{aug:5s} Dice {ms(d)}  Prec {ms(p)}  Rec {ms(r)}")
    L += ["", "Per seed: best epoch / final train loss / final val loss / best val IoU"]
    for aug in AUGS:
        L.append(f"{aug:5s} " + "   ".join(f"s{s}: {e:3d} / {tl:.3f} / {vl:.3f} / {vi:.3f}"
                                            for s in seeds for e, tl, vl, vi in [logs[(aug, s)]]))
    L += ["", "Per seed IoU on noisy test images: " + "  ".join(
        f"{aug} " + "/".join(f"{res[(aug, s)][('noisy', 'model')][1]:.3f}" for s in seeds) for aug in AUGS)]
    L += ["", "Baselines (same test images)"]
    for meth in ("u2net_orig", "prelabel"):
        L.append(f"{meth:10s} " + "  ".join(f"{sub}({r0[(sub, meth)][0]}) {r0[(sub, meth)][1]:.3f}"
                                            for sub in SUBSETS if (sub, meth) in r0))
    txt = "\n".join(L)
    print(txt)
    return txt


if __name__ == "__main__":
    main()
