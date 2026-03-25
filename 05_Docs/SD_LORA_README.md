# SD1.5 LoRA Training for PCB Augmentation

Quick start guide for training Stable Diffusion 1.5 Inpainting LoRA on PCB patches.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM)
- ~10GB disk space for model weights

## Setup

### 1. Install Dependencies

```bash
./install_sdlora_deps.sh
```

Or manually:
```bash
pip install -r requirements_sdlora.txt
```

### 2. Verify Installation

```bash
python -c "import diffusers, peft, xformers; print('All imports successful!')"
```

## Training

### Quick Start (Recommended Settings)

```bash
python scripts/train_sd_lora.py \
  --data_dir data/cutmix_patches \
  --output_dir models/sd_lora_pcb \
  --resolution 256 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --max_train_steps 5000 \
  --lora_rank 16 \
  --mixed_precision fp16 \
  --enable_xformers \
  --gradient_checkpointing
```

**Expected time**: 2-4 hours on RTX 3080 / A100

### Monitor Training

```bash
tensorboard --logdir models/sd_lora_pcb/logs
```

## Configuration

### LoRA Parameters

- `--lora_rank`: LoRA rank (default: 16)
  - Lower = faster, less expressive
  - Higher = slower, more expressive
  - Range: 8-32

- `--lora_alpha`: LoRA scaling (default: 32)
  - Typically 2× rank

### Training Parameters

- `--train_batch_size`: Batch size (default: 1)
  - Keep at 1 for 8GB GPU
  
- `--gradient_accumulation_steps`: Effective batch size (default: 16)
  - Effective batch = batch_size × accumulation_steps
  - Increase for more stable training

- `--learning_rate`: Learning rate (default: 1e-4)
  - Lower = more stable, slower convergence
  - Higher = faster, risk of instability

- `--max_train_steps`: Total training steps (default: 5000)
  - ~2 hours for 5000 steps
  - Monitor loss and stop if converged

### Memory Optimization

- `--mixed_precision fp16`: Use FP16 (saves ~50% memory)
- `--enable_xformers`: Memory-efficient attention
- `--gradient_checkpointing`: Trade compute for memory

## Output

Checkpoints saved to `models/sd_lora_pcb/`:
- `checkpoint-500/`: Every 500 steps
- `checkpoint-1000/`: ...
- `final/`: Final trained LoRA weights

## Next Steps

After training completes:

1. **Generate Augmentation Bank**:
   ```bash
   python scripts/generate_aug_bank.py \
     --lora_path models/sd_lora_pcb/final
   ```

2. **Integrate with Training**:
   ```bash
   python RGB_Experiments/train_rgb.py \
     --augment-mode sdlora \
     --aug-bank-path data/aug_bank
   ```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `--resolution` to 128
- Ensure `--gradient_checkpointing` is enabled
- Close other GPU processes

### Slow Training

- Verify xFormers is working: check logs for "xFormers enabled"
- Use `--mixed_precision fp16`
- Reduce `--checkpointing_steps` frequency

### Poor Quality Generations

- Train longer (increase `--max_train_steps`)
- Increase `--lora_rank` to 32
- Check training loss convergence

## Dataset Statistics

Current extracted patches (from `data/cutmix_patches/`):
- Capacitor: 500 patches
- IC: 246 patches
- Connector: 500 patches
- **Total**: 1,246 training samples

All patches are 256×256 RGB with inpainting masks.
