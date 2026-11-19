"""Prepare files for CVAT: .jpg images + a zip of pre-labels in "Segmentation mask 1.1" format, one per camera

Usage: python scripts/make_cvat_import.py [--list groups/normal.txt]
       python scripts/make_cvat_import.py --list groups/noisy.txt --mask-dir pred/noisy_v1 --out cvat_upload_noisy --suffix modelpred
Output:
  cvat_upload/<folder>/images/*.jpg          images for creating the task (copied; originals are not touched)
  cvat_upload/<folder>_prelabel.zip          use with Actions -> Upload annotations

Zip layout:
  labelmap.txt
  ImageSets/Segmentation/default.txt         image names (without extension)
  SegmentationClass/<name>.png               colors from labelmap (hair = HAIR_RGB)
  SegmentationObject/<name>.png              INSTANCE_RGB; all hair in an image is one object -> one editable mask in CVAT
"""
import argparse
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "original"  # original images + initial U2-Net masks (captures/ccd/mimoa/usb-ts)
WORK = ROOT / "work"      # generated files (gt, prelabel, runs, ...); path arguments are relative to work/
HAIR_RGB = (250, 50, 83)
INSTANCE_RGB = (128, 0, 0)  # SegmentationObject must use a VOC palette color (index 1), not the label color
LABELMAP = (
    "# label:color_rgb:parts:actions\n"
    "background:0,0,0::\n"
    f"hair:{HAIR_RGB[0]},{HAIR_RGB[1]},{HAIR_RGB[2]}::\n"
)


def color_png(mask01, rgb):
    out = np.zeros((*mask01.shape, 3), np.uint8)
    out[mask01 > 0] = rgb[::-1]  # OpenCV writes BGR
    return cv2.imencode(".png", out)[1].tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="groups/normal.txt")
    ap.add_argument("--mask-dir", default="prelabel", help="mask folder <folder>/<name>.png where > 0 = hair")
    ap.add_argument("--out", default="cvat_upload")
    ap.add_argument("--suffix", default="prelabel")
    a = ap.parse_args()

    names = (WORK / a.list).read_text(encoding="utf-8").split()
    by_folder = {}
    for n in names:
        by_folder.setdefault(n.split("/")[0], []).append(n)

    out_root = WORK / a.out
    for folder, items in sorted(by_folder.items()):
        img_dir = out_root / folder / "images"
        if img_dir.exists():
            shutil.rmtree(img_dir)
        img_dir.mkdir(parents=True)
        zpath = out_root / f"{folder}_{a.suffix}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("labelmap.txt", LABELMAP)
            stems = []
            for n in sorted(items):
                src = DATA / n
                stem = src.stem
                shutil.copy2(src, img_dir / src.name)
                m = cv2.imread(str(WORK / a.mask_dir / folder / f"{stem}.png"), cv2.IMREAD_GRAYSCALE) > 0
                z.writestr(f"SegmentationClass/{stem}.png", color_png(m, HAIR_RGB))
                z.writestr(f"SegmentationObject/{stem}.png", color_png(m, INSTANCE_RGB))
                stems.append(stem)
            z.writestr("ImageSets/Segmentation/default.txt", "\n".join(stems) + "\n")
        print(f"{folder}: {len(items)} images -> {zpath.relative_to(WORK)}")


if __name__ == "__main__":
    main()
