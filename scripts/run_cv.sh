#!/usr/bin/env bash
# 5-fold cross-validation on 198 images: every image is scored by a model that never saw it (except 1 image that always stays in train)
# Same hyperparameters as run_aug_ablation.sh, seed 0, folds from scripts/make_cv_split.py
# Order: full on all 5 folds first (main result), then none and geom (can be stopped; re-running skips finished runs)
# Usage: bash scripts/run_cv.sh > work/runs/cv5.log 2>&1
# Summary: .venv/Scripts/python.exe scripts/summarize_cv.py
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}   # override with: PY=python bash scripts/...
EPOCHS=120
for aug in full none geom; do
  for k in 0 1 2 3 4; do
    name="cv5_${aug}_f${k}"
    split="splits/cv5/fold${k}.csv"
    if [ -f "work/runs/$name/eval_test_summary.txt" ]; then echo "skip $name"; continue; fi
    rm -rf "work/runs/$name"
    if [ "$aug" = none ]; then extra="--bs 4 --repeat 2"; else extra="--bs 8 --repeat 4"; fi
    echo "=== $name $(date +%H:%M)"
    $PY scripts/train.py --name "$name" --aug "$aug" --seed 0 --epochs $EPOCHS --split-file "$split" $extra 2>&1 | grep -E "^train |best val|Error|Traceback|memory"
    $PY scripts/evaluate.py "runs/$name" --split test --split-file "$split" 2>&1 | grep -E "^all|^mean per-image IoU all|Error|Traceback"
  done
done
echo "=== done $(date +%H:%M)"
