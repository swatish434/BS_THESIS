
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_plots():
    output_dir = "reports/figures"
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # Plot 1: Overall Mean IoU Comparison
    # ==========================================
    models = ['RGB UNet', 'RGB DeepLab\n(Baseline)', 'RGB DeepLab\n(Copy-Paste)', 'RGB DeepLab\n(SD-LoRA)', 'HSI AttUNet', 'Sim2Real']
    miou = [67.43, 51.54, 75.11, 75.04, 49.78, 32.79]
    colors = ['gray', 'gray', 'orange', 'green', 'blue', 'red']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, miou, color=colors)
    
    # Add values on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', ha='center', va='bottom', fontweight='bold')

    plt.title('Mean IoU Comparison Across Methods', fontsize=14)
    plt.ylabel('Mean IoU (%)', fontsize=12)
    plt.ylim(0, 90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(output_dir, "method_comparison.png"), dpi=300, bbox_inches='tight')
    print("Saved method_comparison.png")
    plt.close()

    # ==========================================
    # Plot 2: Per-Class IoU Improvement
    # ==========================================
    # Focus on DeepLab variants to show augmentation impact
    labels = ['IC', 'Capacitor', 'Connector']
    
    baseline = [38.58, 26.70, 48.77]
    copypaste = [67.48, 55.45, 79.76]
    sdlora = [64.75, 63.28, 73.46]
    
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    rects1 = plt.bar(x - width, baseline, width, label='Baseline (No Aug)', color='gray')
    rects2 = plt.bar(x, copypaste, width, label='Copy-Paste', color='orange')
    rects3 = plt.bar(x + width, sdlora, width, label='SD-LoRA (Generative)', color='green')

    plt.ylabel('IoU (%)', fontsize=12)
    plt.title('Impact of Augmentation on Minority Classes (DeepLabv3+)', fontsize=14)
    plt.xticks(x, labels, fontsize=11)
    plt.legend()
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add improvements for SD-LoRA over Baseline
    for i in range(len(labels)):
        imp = sdlora[i] - baseline[i]
        plt.text(x[i] + width, sdlora[i] + 1, f'+{imp:.1f}%', ha='center', va='bottom', color='green', fontweight='bold', fontsize=9)

    plt.savefig(os.path.join(output_dir, "per_class_iou.png"), dpi=300, bbox_inches='tight')
    print("Saved per_class_iou.png")
    plt.close()

if __name__ == "__main__":
    generate_plots()
