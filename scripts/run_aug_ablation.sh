#!/usr/bin/env bash
# Compare training without augmentation / geometric only / full augmentation (as suggested by the supervisor, 2026-11-13)
# All settings: U-Net + ResNet34, 120 epochs, 47 steps per epoch, same val/test split, 3 seeds
# Usage: PREFIX=abl198 bash scripts/run_aug_ablation.sh > work/runs/ablation198.log
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python.exe}   # override with: PY=python bash scripts/...
PREFIX=${PREFIX:-abl}   # 198-image run: PREFIX=abl198
EPOCHS=120
for seed in 0 1 2; do
  for aug in none geom full; do
    name="${PREFIX}_${aug}_s${seed}"
    if [ -f "work/runs/$name/eval_test_summary.txt" ]; then echo "skip $name"; continue; fi
    rm -rf "work/runs/$name"
    if [ "$aug" = none ]; then extra="--bs 4 --repeat 2"; else extra="--bs 8 --repeat 4"; fi
    echo "=== $name $(date +%H:%M)"
    $PY scripts/train.py --name "$name" --aug "$aug" --seed "$seed" --epochs $EPOCHS $extra 2>&1 | grep -E "best val|Error|Traceback|memory"
    $PY scripts/evaluate.py "runs/$name" --split test 2>&1 | grep -E "^all|^mean per-image IoU model|Error|Traceback"
  done
done
echo "=== done $(date +%H:%M)"
