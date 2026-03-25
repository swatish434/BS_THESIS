#!/bin/bash
# Automation Script: Auto-Evaluate After Training Completes
# Usage: ./auto_eval_after_training.sh <training_process_id>

TRAIN_PID=$1

if [ -z "$TRAIN_PID" ]; then
    echo "Error: Please provide the training process ID"
    echo "Usage: ./auto_eval_after_training.sh <PID>"
    exit 1
fi

echo "=========================================="
echo "Auto-Evaluation Script Started"
echo "Monitoring training process: $TRAIN_PID"
echo "=========================================="

# Wait for training to complete
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 30
    echo "[$(date +%H:%M:%S)] Training still running (PID: $TRAIN_PID)..."
done

echo ""
echo "=========================================="
echo "Training Complete! Starting Evaluation..."
echo "=========================================="
echo ""

# Navigate to PCBVision directory
cd "$(dirname "$0")"

# Step 1: Run Evaluation
echo "[STEP 1/2] Running evaluate_models.py..."
python3 evaluate_models.py
EVAL_EXIT=$?

if [ $EVAL_EXIT -ne 0 ]; then
    echo "⚠️  Evaluation failed with exit code $EVAL_EXIT"
else
    echo "✅ Evaluation completed successfully"
fi

echo ""

# Step 2: Run Visualization
echo "[STEP 2/2] Running visualize_results.py..."
python3 visualize_results.py
VIZ_EXIT=$?

if [ $VIZ_EXIT -ne 0 ]; then
    echo "⚠️  Visualization failed with exit code $VIZ_EXIT"
else
    echo "✅ Visualization completed successfully"
fi

echo ""
echo "=========================================="
echo "All Post-Training Steps Complete!"
echo "=========================================="
echo ""
echo "Check results in:"
echo "  - Metrics: Evaluation/benchmark_results/evaluation_metrics.png"
echo "  - Visualizations: Evaluation/benchmark_results/viz_*.png"
echo "  - Feature Maps: Evaluation/benchmark_results/feature_map_*.png"
echo ""

# Optional: Play sound or send notification (Linux)
if command -v notify-send &> /dev/null; then
    notify-send "PCBVision Training" "Training, Evaluation, and Visualization Complete! ✅"
fi

# Optional: Beep
echo -e "\a"
