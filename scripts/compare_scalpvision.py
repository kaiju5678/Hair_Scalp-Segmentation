"""Compare ScalpVision masks (GitHub winston1214/ScalpVision) with our U-Net on a small set of images

ScalpVision pipeline as described in its README (segmentation/):
  1. u2net_test.py      U²-Net (weight/u2net_tar_min.pth) -> M̂ (prob map, normPRED, bilinear resize back to full size)
  2. sam_guide.py       cluster() picks point prompts from the skeleton of M̂
  3. sam_predict.py     SAM ViT-H (weight/sam_vit_h_4b8939.pth): points on hair = label 0, 10 random points = label 1,
                        invert the mask, drop pieces < 400 px, save as .jpg
  4. make_final_mask.py M = bitwise_and(M̂, SAM), then drop pieces < 400 px
Functions/classes are imported from the repo directly (U2NET, RescaleT, ToTensorLab, SalObjDataset, cluster). The scripts with
hard-coded paths (sam_predict, make_final_mask) are re-implemented step by step with the same logic. Differences from the original:
  - use the real image size instead of the hard-coded 640x480 (Kaggle images are not 640x480)
  - fix the random seed to 0 so that runs are reproducible (the original samples new points every time)

Usage (.venv):
  python scripts/compare_scalpvision.py --name old  original/ccd/x.jpg ...  (images from original/ are scored against work/gt/ automatically)
  python scripts/compare_scalpvision.py --name new  new_dataset/a/b.jpg ...
Output: work/compare_scalpvision/<name>/
  masks/<method>/<name>.png  (0/255)   method = u2net_thr64, scalpvision, ours
  compare_<name>.jpg  top row: image | ScalpVision | ours (| GT)  bottom row: differences
  metrics.csv
"""
import argparse
import ast
import csv
import io as _io
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "original"
WORK = ROOT / "work"
SV = ROOT / "external" / "ScalpVision" / "segmentation"
MODEL_DIR = ROOT / "model"
sys.path.insert(0, str(SV))
sys.path.insert(0, str(MODEL_DIR))

from data_loader import RescaleT, SalObjDataset, ToTensorLab  # noqa: E402  (ScalpVision)
from model import U2NET  # noqa: E402  (ScalpVision)
from torchvision import transforms  # noqa: E402
import predict as ours_predict  # noqa: E402  (model/predict.py)


def load_cluster():
    """Load the functions of sam_guide.py without running its top-level code (which hard-codes paths and writes json)"""
    src = (SV / "sam_guide.py").read_text(encoding="utf-8")
    src = src.replace("640>pts[0]>0 and 480>pts[1]>0", "IMG_W>pts[0]>0 and IMG_H>pts[1]>0")
    tree = ast.parse(src)
    tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    ns = {}
    exec(compile(tree, "sam_guide.py", "exec"), ns)
    return ns


def u2net_prob(net, path, dev):
    """u2net_test.py: RescaleT(320) + ToTensorLab(0) -> d1 -> normPRED -> PIL bilinear resize back -> uint8"""
    from PIL import Image
    ds = SalObjDataset(img_name_list=[str(path)], lbl_name_list=[],
                       transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
    x = ds[0]["image"][None].float().to(dev)
    with torch.no_grad():
        d1 = net(x)[0][:, 0]
    d = (d1 - d1.min()) / (d1.max() - d1.min())
    h, w = cv2.imread(str(path)).shape[:2]
    im = Image.fromarray(d.squeeze().cpu().numpy() * 255).convert("RGB")
    return np.array(im.resize((w, h), resample=Image.BILINEAR))  # HxWx3 uint8, same as the .png saved by the original


def keep_big(binary, min_area=400):
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary, None, None, None, 8, cv2.CV_32S)
    out = np.zeros(lab.shape, np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 255
    return out


def jpg_roundtrip(img):
    """The original saves the SAM output as .jpg and reads it back in make_final_mask -> reproduce the JPEG effect too"""
    return cv2.imdecode(cv2.imencode(".jpg", img)[1], cv2.IMREAD_COLOR)


def scalpvision(path, net, predictor, ns, tmp, dev, rng):
    bgr = cv2.imread(str(path))
    h, w = bgr.shape[:2]
    m_hat = u2net_prob(net, path, dev)
    seg_png = tmp / f"{path.stem}.png"
    cv2.imwrite(str(seg_png), cv2.cvtColor(m_hat, cv2.COLOR_RGB2BGR))  # u2net_test.save_output
    ns["IMG_W"], ns["IMG_H"] = w, h
    pts, _ = ns["cluster"](str(seg_png), seg_png.name)  # sam_guide.py
    seg_img = cv2.imread(str(seg_png))
    if len(pts) == 0:  # sam_predict.py: no points -> use M̂ instead of the SAM output
        sam_img = jpg_roundtrip(seg_img)
        n_pts = 0
    else:
        tmp_pts = np.array(pts)
        tmp_pts = tmp_pts[tmp_pts.min(axis=1) > 0]
        rand_idx = rng.choice(len(tmp_pts), len(tmp_pts) // 2, replace=False)
        predictor.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        neg = []
        while len(neg) < 10:
            xy = [int(rng.integers(w)), int(rng.integers(h))]
            if xy not in tmp_pts:
                neg.append(xy)
        inp = tmp_pts[rand_idx]
        final_point = np.append(inp, np.array(neg)).reshape(-1, 2)
        label = np.array([0] * len(inp) + [1] * len(neg))
        masks, scores, _ = predictor.predict(point_coords=final_point, point_labels=label, multimask_output=True)
        sam_mask = masks[np.argmax(scores)].reshape(h, w)
        binary_map = np.where(sam_mask > 0, 0, 255).astype(np.uint8)
        sam_img = jpg_roundtrip(cv2.cvtColor(keep_big(binary_map), cv2.COLOR_GRAY2BGR))
        n_pts = len(inp)
    added = cv2.bitwise_and(seg_img, sam_img)  # make_final_mask.py
    final = keep_big(cv2.cvtColor(added, cv2.COLOR_BGR2GRAY))
    return final > 0, cv2.cvtColor(m_hat, cv2.COLOR_RGB2GRAY), n_pts


def scores(p, g):
    tp = (p & g).sum(); fp = (p & ~g).sum(); fn = (~p & g).sum()
    return {"iou": tp / max(tp + fp + fn, 1), "dice": 2 * tp / max(2 * tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def tint(bgr, m, color):
    o = bgr.copy()
    o[m] = (0.45 * o[m] + 0.55 * np.array(color)).astype(np.uint8)
    return o


def label(img, text):
    o = img.copy()
    s = max(0.5, img.shape[1] / 900)
    cv2.rectangle(o, (0, 0), (o.shape[1], int(30 * s)), (0, 0, 0), -1)
    cv2.putText(o, text, (8, int(21 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (255, 255, 255), max(1, int(s)), cv2.LINE_AA)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--name", required=True)
    ap.add_argument("--sam-device", default="cuda")
    a = ap.parse_args()
    out = WORK / "compare_scalpvision" / a.name
    for m in ("u2net_thr64", "scalpvision", "ours"):
        (out / "masks" / m).mkdir(parents=True, exist_ok=True)
    tmp = out / "_u2net_prob"
    tmp.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    net = U2NET(3, 1)
    net.load_state_dict(torch.load(ROOT / "weight" / "u2net_tar_min.pth", map_location=dev))
    net.to(dev).eval()
    ck = torch.load(MODEL_DIR / "hair_unet_r34.pt", map_location=dev, weights_only=False)
    import segmentation_models_pytorch as smp
    ours = getattr(smp, ck["arch"])(ck["encoder"], encoder_weights=None, classes=1)
    ours.load_state_dict(ck["state"])
    ours.to(dev).eval()

    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=str(ROOT / "weight" / "sam_vit_h_4b8939.pth"))
    try:
        sam.to(a.sam_device)
        sam_dev = a.sam_device
    except RuntimeError:
        sam.to("cpu")
        sam_dev = "cpu"
    predictor = SamPredictor(sam)
    ns = load_cluster()
    rng = np.random.default_rng(0)
    print("SAM on", sam_dev)

    rows = []
    for p in map(Path, a.images):
        p = p if p.is_absolute() else ROOT / p
        bgr = cv2.imread(str(p))
        try:
            sv, m_hat, n_pts = scalpvision(p, net, predictor, ns, tmp, dev, rng)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            sam.to("cpu"); predictor = SamPredictor(sam); sam_dev = "cpu"
            print("CUDA OOM -> SAM on cpu")
            sv, m_hat, n_pts = scalpvision(p, net, predictor, ns, tmp, dev, rng)
        u2 = m_hat > 64
        ou = ours_predict.predict(ours, bgr, dev) > 0.5
        for m, v in (("u2net_thr64", u2), ("scalpvision", sv), ("ours", ou)):
            cv2.imwrite(str(out / "masks" / m / f"{p.stem}.png"), v.astype(np.uint8) * 255)

        rel = p.relative_to(DATA) if DATA in p.parents else None
        gt_path = WORK / "gt" / rel.with_suffix(".png") if rel else None
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 127 if gt_path and gt_path.exists() else None
        row = {"image": p.relative_to(ROOT).as_posix(), "size": f"{bgr.shape[1]}x{bgr.shape[0]}", "sam_points": n_pts}
        for m, v in (("u2net_thr64", u2), ("scalpvision", sv), ("ours", ou)):
            row[f"{m}_hair_pct"] = round(v.mean() * 100, 2)
            if gt is not None:
                for k, s in scores(v, gt).items():
                    row[f"{m}_{k}"] = round(s, 4)
        if gt is not None:
            row["gt_hair_pct"] = round(gt.mean() * 100, 2)
        row["agree_iou_sv_vs_ours"] = round(scores(sv, ou)["iou"], 4)
        rows.append(row)

        top = [label(bgr, "image"), label(tint(bgr, sv, (0, 220, 220)), "ScalpVision (GitHub)"),
               label(tint(bgr, ou, (0, 220, 220)), "Ours (U-Net)")]
        if gt is not None:
            top.append(label(tint(bgr, gt, (0, 220, 220)), "Ground truth"))
            bot = [label(tint(tint(tint(bgr, sv & gt, (0, 200, 0)), sv & ~gt, (0, 0, 255)), ~sv & gt, (255, 80, 0)),
                         f"ScalpVision vs GT  IoU {row['scalpvision_iou']:.3f}"),
                   label(tint(tint(tint(bgr, ou & gt, (0, 200, 0)), ou & ~gt, (0, 0, 255)), ~ou & gt, (255, 80, 0)),
                         f"Ours vs GT  IoU {row['ours_iou']:.3f}"),
                   label(tint(bgr, u2, (0, 220, 220)), f"U2-Net only (thr 64)  IoU {row['u2net_thr64_iou']:.3f}"),
                   np.full_like(bgr, 40)]
        else:
            bot = [label(tint(bgr, u2, (0, 220, 220)), "U2-Net only (thr 64)"),
                   label(tint(tint(tint(bgr, sv & ou, (0, 200, 0)), sv & ~ou, (255, 0, 255)), ~sv & ou, (0, 140, 255)),
                         "green both / magenta SV only / orange ours only"),
                   label(np.full_like(bgr, 30), f"agreement IoU {row['agree_iou_sv_vs_ours']:.3f}")]
        cv2.imwrite(str(out / f"compare_{p.stem}.jpg"), np.vstack([np.hstack(top), np.hstack(bot)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(p.name, {k: v for k, v in row.items() if k.endswith("iou") or k.endswith("pct")})

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
