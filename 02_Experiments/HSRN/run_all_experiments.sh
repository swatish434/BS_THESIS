#!/bin/bash
# ===========================================================
# Sequential HSRN Experiment Runner
# Runs all 6 experiments one-by-one so GPU memory is not
# exhausted by running multiple models in parallel.
# ===========================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs Results

run_exp() {
    local name="$1"
    local logfile="$2"
    shift 2
    echo ""
    echo "========================================"
    echo " Starting: $name"
    echo " Log:      $logfile"
    echo "========================================"
    python3 train_hsrn_experiments.py "$@" 2>&1 | tee "$logfile"
    echo ""
    echo " ✓ Finished: $name"
}

# ── RGB experiments ───────────────────────────────────────
run_exp "Exp 01: RGB — No Augmentation" \
    logs/exp01_rgb_none.log \
    --data_type rgb --augment none --epochs 100

run_exp "Exp 02: RGB — Copy-Paste Augmentation" \
    logs/exp02_rgb_copypaste.log \
    --data_type rgb --augment copypaste --epochs 100

run_exp "Exp 03: RGB — CutMix Augmentation" \
    logs/exp03_rgb_cutmix.log \
    --data_type rgb --augment cutmix --epochs 100

# ── HSI experiments ───────────────────────────────────────
run_exp "Exp 04: HSI — No Augmentation" \
    logs/exp04_hsi_none.log \
    --data_type hsi --augment none --epochs 100 --batch_size 4

run_exp "Exp 05: HSI — Copy-Paste Augmentation" \
    logs/exp05_hsi_copypaste.log \
    --data_type hsi --augment copypaste --epochs 100 --batch_size 4

run_exp "Exp 06: HSI — CutMix Augmentation" \
    logs/exp06_hsi_cutmix.log \
    --data_type hsi --augment cutmix --epochs 100 --batch_size 4

echo ""
echo "All 6 experiments completed!"
echo "Results saved to: $SCRIPT_DIR/Results/"
ls -lh Results/*.pth 2>/dev/null || true
