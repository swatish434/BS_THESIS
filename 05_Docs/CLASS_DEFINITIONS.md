# PCB Segmentation Class Definitions - Official Reference

## Standard Class Nomenclature (Use This for Presentation)

| Class ID | Official Name | Description | Color Code |
|:---:|:---|:---|:---:|
| **0** | **Background** | PCB substrate (black board) | Black |
| **1** | **Component** | Passive components (capacitors, resistors, SMDs) | Red |
| **2** | **IC** | Integrated Circuits (chips, processors) | Green |
| **3** | **Connector** | Port connectors, headers, pins | Blue |

---

## Key Points for Presentation

### Class 1: "Component" (Not "Capacitor")
**What it includes**:
- ✅ Capacitors (cylindrical, ceramic)
- ✅ Resistors (rectangular SMDs)
- ✅ Other passive surface-mount devices
- ✅ Small electronic components

**Why not "Capacitor"?**
- This class is a **generalized category** for all small passive components
- Using "Capacitor" is too specific and inaccurate
- "Component" better represents the actual training labels

---

## Standardized Across All Files

The following files have been updated to use consistent nomenclature:

✅ `README.md` - Dataset description  
✅ `Evaluation/evaluate_hsi_comparison.py` - Evaluation script  
✅ `utils/visualize_spectral_signatures.py` - Visualization tools  
✅ `evaluate_models.py` - Main evaluation pipeline  

---

## For Thesis/Presentation

**When describing your work, use**:
> "The model segments PCB images into 4 classes: Background, Component (passive elements like capacitors and resistors), Integrated Circuits, and Connectors."

**Don't say**:
> ~~"The model detects Capacitors, ICs, and Connectors"~~ ❌

**Correct terminology avoids**:
- Confusion about what "Capacitor" means
- Implying the model can distinguish capacitor types
- Overpromising segmentation granularity

---

## Visual Reference

```
Sample PCB Component Breakdown:
┌─────────────────────────────────┐
│  Background (Black PCB)         │ ← Class 0
│                                 │
│  🔴 Component (Cap/Res)         │ ← Class 1
│  🟢 IC (Intel Chip)             │ ← Class 2  
│  🔵 Connector (USB Port)        │ ← Class 3
└─────────────────────────────────┘
```

---

## Updated: January 23, 2026
All references to "Capacitor" have been replaced with "Component" for accuracy and consistency.
