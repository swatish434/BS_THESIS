# Model Summary Quick Reference

## Overview
Added `print_model_summary()` function to `models/DeepLabv3_plus.py` for analyzing model architecture and parameters.

## Usage

### Option 1: Run the model file directly
```bash
cd /home/bs_thesis/Documents/BS_THESIS/PCBVision
python3 models/DeepLabv3_plus.py
```

This will display summaries for both RGB and HSI versions.

### Option 2: Use in your own scripts
```python
from models.DeepLabv3_plus import DeepLabv3_plus, print_model_summary

# Create model
model = DeepLabv3_plus(nInputChannels=214, n_classes=4)

# Print summary
print_model_summary(model, input_size=(214, 256, 256))
```

## Sample Output

**RGB DeepLabv3+ (3 channels, 4 classes)**:
- Total Parameters: **59,339,940**
- Model Size: **226.36 MB**
- Trainable: 100%

**HSI DeepLabv3+ (214 channels, 4 classes)**:
- Total Parameters: **60,001,636**  
- Model Size: **228.89 MB**
- Trainable: 100%

## Key Observations

1. **HSI only adds 661,696 more parameters** despite 71× more input channels (214 vs 3)
   - Why? The first conv layer expands from 3→64 vs 214→64
   - Extra params: (214-3) × 64 × kernel = 211 × 64 × 7 × 7 ≈ 661K

2. **ResNet backbone dominates**: 72% of all parameters (42-43M out of 60M)

3. **ASPP modules**: Each ASPP branch has ~4.7M parameters (4 branches total)

## Enhanced Details (Optional)

For more detailed layer-by-layer breakdown, install `torchinfo`:
```bash
pip install torchinfo
```

Then re-run the summary - it will automatically use the advanced version showing:
- Input/Output shapes for each layer
- Receptive field information
- Memory usage estimates
