"""Evaluate a model on the test split against the original masks (U²-Net, threshold 64) and the pre-labels,
or let the model predict any list of images (e.g. the noisy group)

Usage (.venv):
  .venv/Scripts/python.exe scripts/evaluate.py runs/unet_r34 --split test
  .venv/Scripts/python.exe scripts/evaluate.py runs/unet_r34 --list groups/noisy.txt --out pred/noisy
Output (--split): runs/<name>/eval_<split>.csv (per image), eval_<split>_summary.txt, vis_<split>/*.jpg
         vis image: image | ground truth | model (green = correct, red = false positive, blue = missed)
Output (--list): <out>/<camera>/<name>.png (0/255) and <out>_overlay/<camera>/<name>.jpg
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "original"  # original images + initial U2-Net masks (captures/ccd/mimoa/usb-ts)
WORK = ROOT / "work"      # generated files (gt, prelabel, runs, ...); path arguments are relative to work/
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def load_model(run, which):
    ck = torch.load(WORK / run / f"{which}.pt", map_location="cuda", weights_only=False)
    m = getattr(smp, ck["arch"])(ck["encoder"], encoder_weights=None, classes=1)
    m.load_state_dict(ck["state"])
    print(f"{run}/{which}.pt epoch {ck['epoch']} val IoU {ck['val_iou']:.4f}")
    return m.cuda().eval()


@torch.no_grad()
def predict_prob(model, bgr, tta=True):
    rgb = bgr[:, :, ::-1].astype(np.float32)
    x = torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())[None].cuda()
    flips = [[], [3], [2], [2, 3]] if tta else [[]]
    acc = 0
    for f in flips:
        xi = x.flip(f) if f else x
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(xi).float())
        acc = acc + (p.flip(f) if f else p)
    return (acc / len(flips))[0, 0].cpu().numpy()


def stats(pred, gt):
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    return tp, fp, fn


def scores(tp, fp, fn):
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def vis(bgr, gt, pred):
    def paint(m_list):
        o = bgr.copy()
        for m, c in m_list:
            o[m] = (0.45 * o[m] + 0.55 * np.array(c)).astype(np.uint8)
        return o
    gt_v = paint([(gt, (0, 220, 220))])
    pr_v = paint([(pred & gt, (0, 200, 0)), (pred & ~gt, (0, 0, 255)), (~pred & gt, (255, 80, 0))])
    return np.hstack([bgr, gt_v, pr_v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--which", default="best")
    ap.add_argument("--split")
    ap.add_argument("--list")
    ap.add_argument("--out")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for result files, e.g. _198, so earlier results are not overwritten")
    ap.add_argument("--split-file", default="splits/split.csv", help="relative to work/, e.g. splits/cv5/fold0.csv")
    ap.add_argument("--gt-dir", default="gt", help="ground truth folder relative to work/, e.g. gt_v2 (use together with --tag)")
    a = ap.parse_args()
    model = load_model(a.run, a.which)
    tta = not a.no_tta

    if a.list:
        out = WORK / a.out
        ov = WORK / (a.out + "_overlay")
        for line in open(WORK / a.list, encoding="utf-8"):
            rel = line.strip()
            if not rel:
                continue
            cam, fn = rel.split("/")
            stem = Path(fn).stem
            bgr = cv2.imread(str(DATA / cam / f"{stem}.jpg"))
            pred = predict_prob(model, bgr, tta) > a.thr
            (out / cam).mkdir(parents=True, exist_ok=True)
            (ov / cam).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / cam / f"{stem}.png"), pred.astype(np.uint8) * 255)
            o = bgr.copy()
            o[pred] = (0.45 * o[pred] + 0.55 * np.array((0, 220, 220))).astype(np.uint8)
            cv2.imwrite(str(ov / cam / f"{stem}.jpg"), np.hstack([bgr, o]))
        print("saved to", out)
        return

    rows = list(csv.DictReader(open(WORK / a.split_file, encoding="utf-8")))
    items = [(r["camera"], r["name"]) for r in rows if r["split"] == a.split]
    noisy = {Path(l.strip()).with_suffix("").as_posix()
             for l in open(WORK / "groups" / "noisy.txt", encoding="utf-8") if l.strip()}
    grp = lambda cam, name: "noisy" if f"{cam}/{name}" in noisy else "normal"
    run = WORK / a.run
    vdir = run / f"vis_{a.split}{a.tag}"
    vdir.mkdir(exist_ok=True)
    methods = ["model", "u2net_orig", "prelabel"]  # noisy images have no pre-label -> prelabel is scored only where it exists
    cats = ["all", "normal", "noisy", "ccd", "mimoa", "captures", "usb-ts"]
    agg = {(m, c): [0, 0, 0] for m in methods for c in cats}
    cnt = {(m, c): 0 for m in methods for c in cats}
    per = []
    for cam, name in items:
        bgr = cv2.imread(str(DATA / cam / f"{name}.jpg"))
        gt = cv2.imread(str(WORK / a.gt_dir / cam / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 127
        preds = {
            "model": predict_prob(model, bgr, tta) > a.thr,
            "u2net_orig": cv2.imread(str(DATA / cam / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 64,
        }
        pl = WORK / "prelabel" / cam / f"{name}.png"
        if grp(cam, name) == "normal" and pl.exists():
            preds["prelabel"] = cv2.imread(str(pl), cv2.IMREAD_GRAYSCALE) > 0
        row = {"camera": cam, "name": name, "group": grp(cam, name)}
        for m in methods:
            if m not in preds:
                row[f"{m}_iou"] = row[f"{m}_tp"] = row[f"{m}_fp"] = row[f"{m}_fn"] = ""
                continue
            s = stats(preds[m], gt)
            for c in ("all", grp(cam, name), cam):
                agg[(m, c)] = [u + v for u, v in zip(agg[(m, c)], s)]
                cnt[(m, c)] += 1
            row[f"{m}_iou"] = round(scores(*s)["iou"], 4)
            row[f"{m}_tp"], row[f"{m}_fp"], row[f"{m}_fn"] = s
        per.append(row)
        cv2.imwrite(str(vdir / f"{cam}__{name}.jpg"), vis(bgr, gt, preds["model"]))

    with open(run / f"eval_{a.split}{a.tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(per[0]))
        w.writeheader()
        w.writerows(per)
    lines = [f"{a.run} ({a.which}.pt) split={a.split} n={len(items)} thr={a.thr} tta={tta}",
             "pooled over all pixels of the split (micro); (n) = images counted; prelabel exists only for normal-group images",
             f"{'subset':13s} {'method':11s}   IoU   Dice  Prec   Rec"]
    for c in cats:
        for m in methods:
            if not cnt[(m, c)]:
                continue
            s = scores(*agg[(m, c)])
            lines.append(f"{c + f'({cnt[(m, c)]})':13s} {m:11s} {s['iou']:.3f} {s['dice']:.3f} {s['precision']:.3f} {s['recall']:.3f}")
    for g in ("all", "normal", "noisy"):
        for m in methods:
            v = [r[f"{m}_iou"] for r in per if r[f"{m}_iou"] != "" and g in ("all", r["group"])]
            if v:
                lines.append(f"mean per-image IoU {g:6s} {m:11s} {np.mean(v):.3f}")
    txt = "\n".join(lines)
    (run / f"eval_{a.split}{a.tag}_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
