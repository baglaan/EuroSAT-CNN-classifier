# EuroSAT Multispectral Land-Use Classification

A custom 4-layer convolutional neural network for classifying Sentinel-2 satellite imagery across 10 land-use categories. EuroSAT Data was taken from Kaggle.
## Test results

| Metric          | Value                          |
| --------------- | ------------------------------ |
| **Accuracy**    | **97.93%**                     |
| **Macro-F1**    | **0.9782**                     |
| Parameters      | ~1.6 M                         |
| Training epochs | ~36 (early stopping triggered) |


---

## Multispectral vs RGB

 **Identical architecture** on RGB-only (bands B2/B3/B4) vs the full 13 bands:

| Input | Channels | Test Macro-F1 |
|---|---|---|
| RGB only | 3 | ~0.968 |
| Full multispectral | 13 | **0.9782** |

The non-visible bands provide a small but consistent improvement (+0.010 macro-F1), with **every class equal or better**. The single largest gain is on **River (+0.025)** - narrow water features that are visually ambiguous in RGB but unmistakable in the infrared bands, where water's strong NIR absorption is a distinctive signature.

In contrast, **SeaLake shows no gain** (0.997 in RGB; 0.997 in 13-band): large water bodies are already unambiguous visually. The multispectral benefit thus concentrates exactly where the visual signal is weak but the spectral signal is strong - a physically grounded result rather than a uniform improvement.

---

## Dataset

[EuroSAT](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset/data) — Sentinel-2 satellite imagery covering 10 European land-use classes. The all-bands variant used here contains 13 spectral bands (visible + red-edge + NIR + SWIR), each image 64×64 pixels at 10 m ground resolution.

**Classes:** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.

Predefined CSV splits (`train.csv` / `validation.csv` / `test.csv`) and `label_map.json` from the dataset distribution are used, ensuring comparability with benchmarks and a proper held-out test set.

### Sentinel-2 bands

| Band | Wavelength | Use |
|---|---|---|
| B1 | 443 nm | Coastal aerosol |
| B2 | 490 nm | Blue (visible) |
| B3 | 560 nm | Green (visible) |
| B4 | 665 nm | Red (visible) |
| B5–B7 | 705–783 nm | Vegetation red-edge |
| B8 | 842 nm | NIR — vegetation/biomass |
| B8A | 865 nm | Narrow NIR |
| B9 | 945 nm | Water vapor |
| B10 | 1375 nm | SWIR cirrus |
| B11–B12 | 1610–2190 nm | SWIR moisture / burnt areas |

---

## Architecture

Four convolutional blocks followed by a classifier head. The input layer is modified to accept 13 channels instead of the conventional 3:

```
Input (13 × 64 × 64)
│
├─ Conv2d(13 → 64,  3×3) → BatchNorm → ReLU → MaxPool(2×2)   →  64 × 32 × 32
├─ Conv2d(64 → 128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)   → 128 × 16 × 16
├─ Conv2d(128 → 256, 3×3) → BatchNorm → ReLU → MaxPool(2×2)  → 256 ×  8 ×  8
├─ Conv2d(256 → 512, 3×3) → BatchNorm → ReLU → MaxPool(2×2)  → 512 ×  4 ×  4
│
├─ Flatten → Dropout(0.5) → Linear(8192 → 10)
└─ Logits (10 classes)
```

Design notes:
- `bias=False` on all convs since BatchNorm's β term handles the shift.
- No softmax in the model — `CrossEntropyLoss` applies log-softmax internally.

---

## Training setup

|                  | Value                                          |
| ---------------- | ---------------------------------------------- |
| Loss             | CrossEntropyLoss                               |
| Optimizer        | Adam (lr=1e-3)                                 |
| Scheduler        | StepLR (step_size=20, γ=0.1)                   |
| Batch size       | 32                                             |
| Normalization    | Per-band z-score, stats from training set only |
| Selection metric | Validation macro-F1                            |
| Stopping         | triggered by early stopping                    |
| Hardware         | Kaggle T4 GPU                                  |

Per-band normalization is computed once on the training set and applied identically to all three splits (train, val, test). The same statistics are saved alongside the model so inference on new images uses the exact normalization the model was trained with.

---

## Results

### Per-class test results (13-band model)

| Class                | Precision  | Recall     | F1         |
| -------------------- | ---------- | ---------- | ---------- |
| AnnualCrop           | 0.9575     | 0.9767     | 0.9670     |
| Forest               | 0.9967     | 1.0000     | **0.9983** |
| HerbaceousVegetation | 0.9544     | 0.9767     | 0.9654     |
| Highway              | 0.9757     | 0.9640     | 0.9698     |
| Industrial           | 0.9842     | 0.9960     | 0.9901     |
| Pasture              | 0.9695     | 0.9550     | 0.9622     |
| PermanentCrop        | 0.9752     | 0.9440     | 0.9593     |
| Residential          | 0.9933     | 0.9867     | 0.9900     |
| River                | 0.9802     | 0.9880     | 0.9841     |
| SeaLake              | **1.0000** | 0.9916     | 0.9958     |
| **Macro avg**        | **0.9787** | **0.9779** | **0.9782** |



### Per-class delta:

| Class | 13-band F1 | RGB F1 | Gain from multispectral |
|---|---|---|---|
| **River** | 0.9802 | 0.9553 | **+0.0249** |
| HerbaceousVegetation | 0.9718 | 0.9570 | +0.0149 |
| Forest | 0.9983 | 0.9835 | +0.0148 |
| AnnualCrop | 0.9768 | 0.9652 | +0.0116 |
| Pasture | 0.9648 | 0.9543 | +0.0105 |
| PermanentCrop | 0.9597 | 0.9499 | +0.0098 |
| Residential | 0.9950 | 0.9867 | +0.0083 |
| Industrial | 0.9919 | 0.9880 | +0.0039 |
| Highway | 0.9780 | 0.9761 | +0.0019 |
| SeaLake | 0.9972 | 0.9972 | 0.0000 |

The pattern: **classes whose visual appearance is ambiguous but spectral signature is distinctive benefit most.** Narrow rivers in RGB can resemble roads or field boundaries; in the NIR/SWIR bands, water's distinctive absorption makes them unmistakable. 

## Band13 Confusion Matrix:
<img width="671" height="541" alt="image" src="https://github.com/user-attachments/assets/26fbf940-4ae1-4c2b-9b2d-3c0fef9318ee" />


## RGB Confusion Matrix:
<img width="666" height="543" alt="image" src="https://github.com/user-attachments/assets/00d2544b-2925-4f1d-ab1c-f92e4c7fe861" />


## Implementation notes

**Handling 13-channel input.** Modified the first `Conv2d` to accept 13 input channels.  The model uses a 3×3 stride 1 first conv to preserve resolution. Standard color-jitter augmentation is inappropriate for multispectral data (it only operates on RGB channels) and was skipped.

**Why macro-F1 over accuracy or loss for model selection.** Accuracy works on EuroSAT (classes are roughly balanced), but macro-F1 weights every class equally and is the metric reported in the writeup;  

---

## Reproducibility

**Environment:** Kaggle Notebooks with GPU T4. A single training run takes roughly 1–2 hours.

**Adjust paths if running outside Kaggle.**


```

**Key dependencies:**
torch
torchvision
rasterio        # for reading 13-band .tif files
scikit-learn    
pandas
tqdm
matplotlib
```


