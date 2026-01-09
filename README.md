# CPF
CPF: Cross-Polarization Fusion for SAR-Based Flood Mapping is an attention-driven technique designed to fuse Sentinel-1 VV and VH polarizations for accurate flood segmentation.

The code is implemented in PyTorch and tested on Ubuntu 20.04.6 (Python 3.10+, PyTorch ≥ 1.11) with NVIDIA RTX A4000 (16GB VRAM).

---

## Contents
1. [Introduction](#introduction)
2. [Key Highlights](#key-highlights)
3. [Dependencies](#dependencies)
4. [Train](#train)
5. [Test](#test)
6. [Results](#results)
7. [Acknowledgements](#acknowledgements)

---

## Introduction

This repository provides the official implementation of **CPF — Cross-Polarization Fusion**, a modular fusion strategy for flood mapping using dual-polarized Sentinel-1 SAR imagery.

Unlike conventional SAR-based flood mapping approaches that rely on a single polarization (VV or VH) or naive channel concatenation, CPF explicitly models the complementary scattering behavior of VV and VH through a learnable attention-based fusion mechanism.

CPF introduces:
- Independent feature extraction for VV and VH channels
- Bidirectional cross-polarization attention
- Adaptive feature recalibration before segmentation

The fusion module is designed to be **architecture-agnostic** and is evaluated with both:
- A U-Net segmentation backbone
- A convolutional autoencoder (without skip connections)

CPF consistently improves flood delineation accuracy, particularly in vegetated, urban, and mixed land–water environments where single-polarization methods fail.

---

![SAR_VV & SAR_VH Fusion](./Figures/FIGURE1.PNG)
![CPF Architecture](./Figures/FIGURE2.PNG)

---

## Key Highlights

* **Cross-Polarization Attention Fusion**  
  Explicitly models interactions between VV and VH features instead of simple stacking or averaging.

* **Plug-and-Play Design**  
  CPF can be integrated into existing segmentation architectures without increasing depth or parameter count significantly.

* **Polarization-Aware Learning**  
  Preserves polarization-specific scattering characteristics while enhancing complementary responses.

* **Backbone-Agnostic Evaluation**  
  Validated using both U-Net (with skip connections) and Autoencoder architectures.

* **Improved Flood Boundary Delineation**  
  Produces cleaner water boundaries and reduces false positives in vegetation-covered and urban regions.

* **Robust Cross-Event Generalization**  
  Trained and tested across different flood events to ensure temporal and geographic robustness.

---

## Dependencies

* Python 3.10+
* PyTorch ≥ 1.11.0
* CUDA 11.x / 12.x
* numpy
* matplotlib
* scikit-image
* imageio
* tqdm
* opencv-python (optional, for visualization)

---

## Train

### Prepare Training Data

1. Download the **DeepFlood Dataset**, which includes co-registered Sentinel-1 SAR VV/VH images and flood masks:  
   https://figshare.com/articles/dataset/DEEPFLOOD_DATASET_High-Resolution_Dataset_for_Accurate_Flood_Mappingand_Segmentation/28328339

2. Use **SAR_VV** and **SAR_VH** as dual-channel inputs.

3. Create dataset splits:
   - 80% training  
   - 10% validation  
   - 10% testing  

4. Generate CSV index files for train/val/test splits.

5. Specify dataset paths in `config.py`.

---

### Begin Training

Navigate to the project root and run:

```bash
# Example: Train CPF with U-Net backbone
python train_cpf.py
or
Copy code
# Example: Train CPF with Autoencoder backbone
python train_cpf_ae.py
```
---

## Test
Quick Start

Update paths for test CSV and model weights in config.py.

Run:

```bash
Copy code
# Test CPF with UNet 
python eval_cpf.py
or
# Example: Test CPF with Autoencoder 
python eval_cpf_ae.py
```
---

## Results

Quantitative Performance
CPF consistently outperforms single-polarization baselines (VV-only, VH-only) across IoU and F1-score metrics on the DeepFlood dataset.

### U-Net Results
| Input | IoU (%) | F1-score (%) |
|------|--------|--------------|
| VV only | 66.2 | 79.7 |
| VH only | 62.5 | 76.9 |
| **CPF (VV, VH)** | **69.8** | **82.2** |

### Autoencoder Results
| Input | IoU (%) | F1-score (%) |
|------|--------|--------------|
| VV only | 60.4 | 75.3 |
| VH only | 57.1 | 72.7 |
| **CPF (VV, VH)** | **63.2** | **77.5** |

✔ Consistent gains  
✔ Better performance in vegetated and mixed land–water regions  
✔ Improved generalization to unseen flood events

Qualitative Visualization

Visual Results

![Result1](./Figures/FIGURE3.PNG)
![Result2](./Figures/FIGURE4.PNG)


Acknowledgements
This work builds upon open-source segmentation frameworks in PyTorch and benefits from the DeepFlood dataset.

The authors acknowledge support from:

NASA Award 80NSSC23M0051

NSF Award 2401942
