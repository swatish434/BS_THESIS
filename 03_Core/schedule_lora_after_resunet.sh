#!/bin/bash
# Automated Training Scheduler: ResUNet → LoRA
# Waits for ResUNet to finish, then starts LoRA training automatically

RESUNET_PID=2371198
LORA_SCRIPT="scripts/train_sd_lora.py"
LOG_FILE="scheduled_lora_training.log"

echo "========================================"
echo "Automated Training Scheduler"
echo "========================================"
echo "Date: $(date)"
echo "Waiting for ResUNet training to complete..."
echo "ResUNet PID: $RESUNET_PID"
echo ""

# Monitor ResUNet process
while kill -0 $RESUNET_PID 2>/dev/null; do
    # Get GPU memory usage
    GPU_MEM=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep $RESUNET_PID | awk '{print $2}')
    
    echo "[$(date +%H:%M:%S)] ResUNet still running (GPU: ${GPU_MEM:-Unknown})"
    sleep 60  # Check every minute
done

echo ""
echo "========================================"
echo "ResUNet training completed!"
echo "Completion time: $(date)"
echo "========================================"
echo ""

# Wait a bit for GPU memory to clear
echo "Waiting 30 seconds for GPU memory to clear..."
sleep 30

# Show current GPU status
echo ""
echo "Current GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
echo ""

# Start LoRA training
echo "========================================"
echo "Starting LoRA Training"
echo "========================================"
echo "Start time: $(date)"
echo "Command: python3 $LORA_SCRIPT"
echo "Log file: $LOG_FILE"
echo ""

# Run LoRA training with logging
python3 $LORA_SCRIPT 2>&1 | tee $LOG_FILE

# Completion
echo ""
echo "========================================"
echo "LoRA Training Completed"
echo "========================================"
echo "End time: $(date)"
echo "Log saved to: $LOG_FILE"
echo ""

# Show results
echo "LoRA checkpoints:"
ls -lh models/sd_lora_pcb/checkpoint-* 2>/dev/null || echo "No checkpoints found"
ls -lh models/sd_lora_pcb/final/ 2>/dev/null || echo "Final model not found"
