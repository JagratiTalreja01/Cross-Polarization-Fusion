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

![CPF Architecture](./Figures/FIGURE1.pdf)

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
python train_unet.py
or

bash
Copy code
# Example: Train CPF with Autoencoder backbone
python train_autoencoder.py
```
Test
Quick Start
Ensure trained model checkpoints are available.

Update paths for test CSV and model weights in config.py.

Run:

```bash
Copy code
# Test CPF model
python test.py
```
Results
Quantitative Performance
CPF consistently outperforms single-polarization baselines (VV-only, VH-only) across IoU and F1-score metrics on the DeepFlood dataset.

Qualitative Visualization
Each result panel includes:

SAR VV

SAR VH

CPF (VV + VH)

Ground-truth flood mask

Prediction probability map

Overlay of prediction on:

VV

VH

CPF fusion output

Visual Results




Acknowledgements
This work builds upon open-source segmentation frameworks in PyTorch and benefits from the DeepFlood dataset.

The authors acknowledge support from:

NASA Award 80NSSC23M0051

NSF Award 2401942
