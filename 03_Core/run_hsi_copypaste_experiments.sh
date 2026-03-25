#!/bin/bash
# Run HSI CopyPaste Experiments Sequentially

# 1. HSI AttUNet with CopyPaste
echo "Starting HSI AttUNet CopyPaste Training..."
python3 train_hsi_overlap.py --model attunet --augment copypaste --epochs 30

# 2. HSI ResUNet with CopyPaste
echo "Starting HSI ResUNet CopyPaste Training..."
python3 train_hsi_overlap.py --model resunet --augment copypaste --epochs 30

echo "All HSI CopyPaste Experiments Completed."
