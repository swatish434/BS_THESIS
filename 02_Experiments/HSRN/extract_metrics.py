import json
import os
import glob

results_dir = '/home/bs_thesis/Documents/BS_THESIS/PCBVision/02_Experiments/HSRN/Results'
logs = glob.glob(os.path.join(results_dir, '*log.json'))
print(f"Found {len(logs)} logs.")

print('| Experiment | Modality | Augmentation | Best Val mIoU | Best Val PA |')
print('| --- | --- | --- | --- | --- |')
for l in sorted(logs):
    with open(l, 'r') as f:
        try:
            d = json.load(f)
            hist = d.get('history', [])
            if not hist: continue
            best_epoch = max(hist, key=lambda x: x.get('val_miou', 0))
            name = os.path.basename(l)
            modality = 'HSI' if 'hsi' in name else 'RGB'
            aug = 'None'
            if 'copypaste' in name: aug = 'CopyPaste'
            if 'cutmix' in name: aug = 'CutMix'
            print(f"| {name.replace('_log.json', '')} | {modality} | {aug} | {best_epoch.get('val_miou',0):.2f}% | {best_epoch.get('val_pa',0):.2f}% |")
        except Exception as e:
            print(f"Error reading {l}: {e}")
