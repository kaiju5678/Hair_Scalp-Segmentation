"""Train hair segmentation (U-Net + ImageNet-pretrained encoder) on gt/ using splits/split.csv

Usage (Python from .venv):
  .venv/Scripts/python.exe scripts/train.py --name unet_r34
  --aug full  = geom + color/noise/blur/JPEG (default)
  --aug geom  = scale/rotate/flip/random crop only
  --aug none  = full 640x480 images, no transforms (use --bs 4 --repeat 2 so steps and pixels per step match the crop settings)
Output: runs/<name>/best.pt (best val IoU), last.pt, log.csv, config.json
"""
import argparse
import csv
import json
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "original"  # original images + initial U2-Net masks (captures/ccd/mimoa/usb-ts)
WORK = ROOT / "work"      # generated files (gt, prelabel, runs, ...); path arguments are relative to work/
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def read_split(split, split_file="splits/split.csv"):
    rows = list(csv.DictReader(open(WORK / split_file, encoding="utf-8")))
    return [(r["camera"], r["name"]) for r in rows if r["split"] in split]


def load(cam, name):
    img = cv2.imread(str(DATA / cam / f"{name}.jpg"))[:, :, ::-1]
    mask = cv2.imread(str(WORK / "gt" / cam / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    return np.ascontiguousarray(img), (mask > 127).astype(np.uint8)


def to_tensor(img):
    return torch.from_numpy(((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1))


def train_aug(crop, kind, seed):
    if kind == "none":
        return None
    geom = [
        A.Affine(scale=(0.8, 1.25), rotate=(-30, 30), p=0.7),
        A.PadIfNeeded(crop, crop, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomCrop(crop, crop),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
    ]
    if kind == "geom":
        return A.Compose(geom, seed=seed)
    return A.Compose(geom + [
        A.RandomBrightnessContrast(0.25, 0.25, p=0.7),
        A.HueSaturationValue(10, 25, 20, p=0.5),
        A.OneOf([A.GaussNoise(std_range=(0.02, 0.08)), A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.1, 0.6))], p=0.5),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=5)], p=0.2),
        A.ImageCompression(quality_range=(60, 100), p=0.3),
    ], seed=seed)


class DS(Dataset):
    def __init__(self, items, aug=None, repeat=1):
        self.data = [load(*it) for it in items]
        self.aug, self.repeat = aug, repeat

    def __len__(self):
        return len(self.data) * self.repeat

    def __getitem__(self, i):
        img, mask = self.data[i % len(self.data)]
        if self.aug:
            r = self.aug(image=img, mask=mask)
            img, mask = r["image"], r["mask"]
        return to_tensor(img), torch.from_numpy(mask[None].astype(np.float32))


def worker_init(_):
    # Compose keeps its own random state; on Windows every worker gets the same copy -> seed each worker separately
    info = torch.utils.data.get_worker_info()
    if info.dataset.aug is not None:
        info.dataset.aug.set_random_seed(torch.initial_seed() % 2**32)


def iou_per_image(logits, y):
    p = logits > 0
    y = y > 0.5
    inter = (p & y).flatten(1).sum(1).float()
    union = (p | y).flatten(1).sum(1).float()
    return (inter / union.clamp(min=1)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--repeat", type=int, default=4, help="how many passes over the train images per epoch")
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--train-splits", default="train")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--aug", choices=["full", "geom", "none"], default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-file", default="splits/split.csv", help="relative to work/, e.g. splits/cv5/fold0.csv")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = WORK / "runs" / a.name
    out.mkdir(parents=True, exist_ok=False)
    json.dump(vars(a), open(out / "config.json", "w"), indent=1)

    dev = "cuda"
    tr = DS(read_split(a.train_splits.split(","), a.split_file), train_aug(a.crop, a.aug, a.seed), a.repeat)
    va = DS(read_split(["val"], a.split_file))
    tl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True, pin_memory=True, worker_init_fn=worker_init)
    vl = DataLoader(va, 2, num_workers=2, persistent_workers=True)
    print(f"train {len(tr.data)} imgs x{a.repeat}, val {len(va.data)}")

    model = getattr(smp, a.arch)(a.encoder, encoder_weights="imagenet", classes=1).to(dev)
    dice = smp.losses.DiceLoss("binary")
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()

    best = -1
    log = open(out / "log.csv", "w", newline="")
    lw = csv.writer(log)
    lw.writerow(["epoch", "train_loss", "val_loss", "val_iou", "sec"])
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        tot = 0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                lo = model(x)
            lo = lo.float()
            loss = bce(lo, y) + dice(lo, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        model.eval()
        ious, vloss = [], 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(dev), y.to(dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    lo = model(x)
                lo = lo.float()
                vloss += (bce(lo, y) + dice(lo, y)).item()
                ious += iou_per_image(lo, y)
        viou = float(np.mean(ious))
        sec = time.time() - t0
        lw.writerow([ep, round(tot / len(tl), 4), round(vloss / len(vl), 4), round(viou, 4), round(sec, 1)])
        log.flush()
        flag = ""
        if viou > best:
            best = viou
            torch.save({"arch": a.arch, "encoder": a.encoder, "aug": a.aug, "seed": a.seed, "state": model.state_dict(), "epoch": ep, "val_iou": viou}, out / "best.pt")
            flag = " *"
        print(f"ep {ep:3d} loss {tot / len(tl):.4f} val {vloss / len(vl):.4f} iou {viou:.4f} {sec:.0f}s{flag}", flush=True)
    torch.save({"arch": a.arch, "encoder": a.encoder, "aug": a.aug, "seed": a.seed, "state": model.state_dict(), "epoch": a.epochs, "val_iou": viou}, out / "last.pt")
    print("best val IoU", round(best, 4))


if __name__ == "__main__":
    main()
