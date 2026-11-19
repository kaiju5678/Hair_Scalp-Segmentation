"""Collect the vis images from 5-fold CV (each image predicted by a model that never saw it), sorted from the lowest IoU,
so that a reviewer can find images where the model is better than the ground truth.

Output: work/gt_review/<camera>/<rank>_iou<xxx>_<name>.jpg  (image | ground truth | model)
        work/gt_review/review.csv  (fill in the choice column: gt / model / empty)
"""
import argparse
import csv
import shutil
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent / "work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="full")
    ap.add_argument("--out", default="gt_review")
    a = ap.parse_args()
    out = WORK / a.out
    if (out / "review.csv").exists():
        raise SystemExit(f"{out/'review.csv'} already exists; not overwriting (it may contain your choices)")
    rows = []
    for k in range(5):
        run = WORK / "runs" / f"cv5_{a.aug}_f{k}"
        for r in csv.DictReader(open(run / "eval_test.csv", encoding="utf-8")):
            r["fold"] = k
            r["src"] = run / "vis_test" / f"{r['camera']}__{r['name']}.jpg"
            rows.append(r)
    rows.sort(key=lambda r: float(r["model_iou"]))
    recs = []
    for i, r in enumerate(rows, 1):
        d = out / r["camera"]
        d.mkdir(parents=True, exist_ok=True)
        fn = f"{i:03d}_iou{float(r['model_iou']):.3f}_{r['name']}.jpg"
        shutil.copy2(r["src"], d / fn)
        recs.append({"rank": i, "camera": r["camera"], "name": r["name"], "group": r["group"],
                     "fold": r["fold"], "model_iou": r["model_iou"],
                     "model_fp_px": r["model_fp"], "model_fn_px": r["model_fn"],
                     "file": f"{r['camera']}/{fn}", "choice": "", "note": ""})
    with open(out / "review.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    print(f"{len(recs)} images -> {out}")


if __name__ == "__main__":
    main()
