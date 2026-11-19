"""Split the labeled images into train/val/test, keeping images of the same capture spot in the same split

Usage: python scripts/make_split.py            -> splits/split.csv
The 149 original images keep their split from splits/split_149.csv (if present); nothing is re-randomized
New images (e.g. the 49 noisy images) in the same group as an original image -> follow that image's split
Groups with only new images -> largest first, into test until new images reach 20% of test, then val up to 15%, rest train
Groups per camera:
  usb-ts   : M number (e.g. M6405) = same person or session, all modes and timestamps
  ccd      : same timestamp (different modes of the same spot)
  mimoa    : images taken at most 2 minutes apart form one group
  captures : images taken at most 60 seconds apart form one group
"""
import csv
import random
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "original"  # original images + initial U2-Net masks (captures/ccd/mimoa/usb-ts)
WORK = ROOT / "work"      # generated files (gt, prelabel, runs, ...); path arguments are relative to work/
CAMS = ["ccd", "mimoa", "captures", "usb-ts"]
RATIO = {"test": 0.2, "val": 0.15}
SEED = 13
OLD_SPLIT = WORK / "splits" / "split_149.csv"  # split of the earlier 149-image experiment
KEEP_OUT_OF_TEST = {"h_2023_12_05_19_42_09_ms"}  # blurred hair not labeled; not used for scoring


def chain_groups(names, t, gap):
    names = sorted(names, key=t)
    out, gid, prev = {}, 0, None
    for n in names:
        if prev is not None and t(n) - t(prev) > gap:
            gid += 1
        out[n] = gid
        prev = n
    return out


def groups(cam, names):
    if cam == "usb-ts":
        return {n: re.search(r"M\d+", n).group() for n in names}
    if cam == "ccd":
        return {n: n.split("_")[0] for n in names}
    if cam == "mimoa":
        t = lambda n: datetime.strptime(n[2:21], "%Y_%m_%d_%H_%M_%S").timestamp()
        return {n: f"m{g}" for n, g in chain_groups(names, t, 120).items()}
    t = lambda n: int(n.split("_")[1]) / 1000
    return {n: f"c{g}" for n, g in chain_groups(names, t, 60).items()}


def inherit(rows):
    """Keep the original images in their split_149.csv split, then assign new images by group"""
    old = {(r["camera"], r["name"]): r["split"]
           for r in csv.DictReader(open(OLD_SPLIT, encoding="utf-8"))}
    members = {}
    for r in rows:
        members.setdefault(r[2], []).append(r)
    assign, fresh = {}, []
    for g, rs in members.items():
        olds = {old[(r[0], r[1])] for r in rs if (r[0], r[1]) in old}
        assert len(olds) <= 1, f"group {g} has original images in more than one split {olds}"
        if olds:
            assign[g] = olds.pop()
        else:
            fresh.append(g)
    new = [r for r in rows if (r[0], r[1]) not in old]
    got = {s: sum(assign.get(r[2]) == s for r in new) for s in ("test", "val")}
    target = {s: round(len(new) * q) for s, q in RATIO.items()}
    for g in sorted(fresh, key=lambda g: (-len(members[g]), g)):
        assign[g] = "train"
        for s in ("test", "val"):
            if got[s] < target[s]:
                assign[g] = s
                got[s] += len(members[g])
                break
    for r in rows:
        r[3] = assign[r[2]]
    print(f"old images kept: {sum((r[0], r[1]) in old for r in rows)}, new images: {len(new)} "
          f"-> {dict((s, sum(r[3] == s for r in new)) for s in ('train', 'val', 'test'))}")


def main():
    rng = random.Random(SEED)
    rows = []
    for cam in CAMS:
        names = sorted(p.stem for p in (WORK / "gt" / cam).glob("*.png"))
        g = groups(cam, names)
        members = {}
        for n in names:
            members.setdefault(g[n], []).append(n)
        keys = sorted(members)
        rng.shuffle(keys)
        target = {s: round(len(names) * r) for s, r in RATIO.items()}
        got = {"test": 0, "val": 0}
        assign = {}
        for k in keys:
            size = len(members[k])
            for s in ("test", "val"):
                bad = s == "test" and any(m in KEEP_OUT_OF_TEST for m in members[k])
                if not bad and got[s] < target[s] and got[s] + size <= target[s] + 1:
                    assign[k] = s
                    got[s] += size
                    break
            else:
                assign[k] = "train"
        for n in names:
            rows.append([cam, n, f"{cam}:{g[n]}", assign[g[n]]])
    if OLD_SPLIT.exists():
        inherit(rows)
    out = WORK / "splits" / "split.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["camera", "name", "group", "split"])
        w.writerows(rows)
    for cam in CAMS + [None]:
        sub = [r for r in rows if cam is None or r[0] == cam]
        c = {s: sum(r[3] == s for r in sub) for s in ("train", "val", "test")}
        print(f"{cam or 'total':9s} groups {len({r[2] for r in sub}):3d}  {c}")


if __name__ == "__main__":
    main()
