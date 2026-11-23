# Robust Recognition of Handwritten Arabic Characters in Low-Quality Images Using Preprocessing and HOG Features

**ARTI 403 – Image Processing | Group 2**
Imam Abdulrahman Bin Faisal University – College of Computer Science and Information Technology
**Instructor:** Dr. Manal

## Overview

This project studies how image preprocessing affects the recognition of isolated handwritten Arabic characters in low-quality images. A classical pipeline (preprocessing → HOG + zoning features → SVM) is compared with a CNN baseline, and both are tested on clean images and on images degraded with Gaussian noise, blur and rotation.

## Dataset

**AHCD** (Arabic Handwritten Characters Dataset): 16,800 grayscale images (32×32) of the 28 Arabic letters, written by 60 participants.
[Click here to access the AHCD dataset](https://github.com/mloey/Arabic-Handwritten-Characters-Dataset)

Split used: 11,423 training, 2,016 validation (stratified 15% of the original training set, one blank image removed) and 3,360 test images.

## Methodology

- **Preprocessing:** centering and resizing, 3×3 cross-shaped median filter, Otsu thresholding, and thinning.
- **Feature extraction:** HOG (9 orientations, 4×4 cells) and 4×4 zoning (1,780 features).
- **Classification:** RBF-kernel SVM (C = 10, γ = 0.001) and a compact 3-block CNN.
- **Evaluation:** HOG settings comparison, SVM tuning on the validation set, an ablation study of each preprocessing step, and robustness tests under Gaussian noise (σ = 0.05, 0.1), mean blur (3×3, 5×5) and rotation (±5°, ±10°).

## Key Results (test accuracy, %)

| Condition | SVM raw | SVM preprocessed | CNN raw | CNN preprocessed |
|---|---|---|---|---|
| Clean | 74.9 | 86.8 | 97.5 | 92.6 |
| Noise σ = 0.1 | 3.6 | 84.6 | 3.9 | 91.7 |
| Blur 5×5 | 34.8 | 37.3 | 23.8 | 41.1 |
| Rotation ±10° | 71.6 | 83.1 | 96.5 | 90.8 |

- Preprocessing made both models robust to noise, while the models trained on raw images collapsed.
- The ablation study showed that centering and Otsu thresholding are the most important steps, and that the SVM performs best without thinning (91.4% on clean images).

## How to Run

1. Open the notebook in Google Colab and select a GPU runtime (needed for the CNN).
2. Run all cells. The dataset is downloaded automatically into `data/`.
3. A full run takes about 1.5–2.5 hours. All results are saved to `results.csv`, and all figures to `figures/`.

Requirements: Python 3, NumPy, pandas, SciPy, scikit-image, scikit-learn, Matplotlib and PyTorch (all preinstalled in Colab).

## Repository Contents

- `Arabic_Handwritten_Character_Recognition.ipynb` – complete implementation, with outputs.
- `results.csv` – all validation and test results.
- `figures/` – all figures used in the paper.

## Team Members

- Joud Fayez Althagafi
- Layan Altamimi
- Lina Bashamakh
- Maryam Abu Qurain
- Renad Ghazi AlYami
- Retal Alharbi
