#!/bin/bash
set -e

echo "=== Running Visualizations for RGB Models ==="
python3 visualize_features.py --data_type rgb --augment none --num_samples 5
python3 visualize_features.py --data_type rgb --augment copypaste --num_samples 5
python3 visualize_features.py --data_type rgb --augment cutmix --num_samples 5

echo "=== Running Visualizations for HSI Models ==="
python3 visualize_features.py --data_type hsi --augment none --num_samples 5
python3 visualize_features.py --data_type hsi --augment copypaste --num_samples 5
python3 visualize_features.py --data_type hsi --augment cutmix --num_samples 5

echo "All visualizations generated in Results/visualizations/"
