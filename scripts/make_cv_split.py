"""Split the 198 images into 5 folds for cross-validation, keeping images of the same capture spot (group) in the same fold

Usage: python scripts/make_cv_split.py   -> work/splits/cv5/fold<k>.csv  (k = 0..4)
- uses the group column of splits/split.csv (the same groups as the normal train/val/test split)
- assigns groups camera by camera, largest group first, to the fold with the fewest images of that camera
  (ties -> fold with fewer noisy images -> fold with fewer images overall -> lower fold number)
- fold<k>.csv: test = fold k, val = fold (k+1) % 5, train = the other 3 folds
- images in KEEP_OUT_OF_TEST (blurred hair not labeled) always stay in train and are never scored -> 197 images scored
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
K = 5
KEEP_OUT_OF_TEST = {"h_2023_12_05_19_42_09_ms"}


def main():
    rows = list(csv.DictReader(open(WORK / "splits" / "split.csv", encoding="utf-8")))
    noisy = {Path(l.strip()).with_suffix("").as_posix()
             for l in open(WORK / "groups" / "noisy.txt", encoding="utf-8") if l.strip()}
    members = defaultdict(list)
    for r in rows:
        members[r["group"]].append(r)
    fold_of = {}
    n_all, n_noisy = [0] * K, [0] * K
    for cam in sorted({r["camera"] for r in rows}):
        n_cam = [0] * K
        gs = sorted([g for g in members if members[g][0]["camera"] == cam], key=lambda g: (-len(members[g]), g))
        for g in gs:
            k = min(range(K), key=lambda i: (n_cam[i], n_noisy[i], n_all[i], i))
            fold_of[g] = k
            size = len(members[g])
            n_cam[k] += size
            n_all[k] += size
            n_noisy[k] += sum(f"{r['camera']}/{r['name']}" in noisy for r in members[g])
    out = WORK / "splits" / "cv5"
    out.mkdir(parents=True, exist_ok=True)
    for k in range(K):
        with open(out / f"fold{k}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["camera", "name", "group", "split"])
            for r in rows:
                fk = fold_of[r["group"]]
                s = "test" if fk == k else "val" if fk == (k + 1) % K else "train"
                if s == "test" and r["name"] in KEEP_OUT_OF_TEST:
                    s = "train"
                w.writerow([r["camera"], r["name"], r["group"], s])
    print("fold  images  noisy  " + "  ".join(sorted({r['camera'] for r in rows})))
    for k in range(K):
        cams = defaultdict(int)
        for r in rows:
            if fold_of[r["group"]] == k:
                cams[r["camera"]] += 1
        print(f"{k:4d}  {n_all[k]:6d}  {n_noisy[k]:5d}  " + "  ".join(f"{cams[c]:3d}" for c in sorted(cams)))
    print("saved to", out)


if __name__ == "__main__":
    main()
