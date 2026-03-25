#!/bin/bash
# Install dependencies for SD1.5 LoRA training

echo "Installing SD LoRA Training Dependencies..."
echo "==========================================="

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: CUDA not detected. xFormers may not work properly."
fi

# Install from requirements
pip install -r requirements_sdlora.txt

echo ""
echo "Installation complete!"
echo ""
echo "Test installation with:"
echo "  python -c 'import diffusers, peft, xformers; print(\"All imports successful!\")'"
echo ""
echo "To start training:"
echo "  python scripts/train_sd_lora.py"
