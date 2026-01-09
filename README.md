# Cross-Polarization Fusion (CPF) for SAR-Based Flood Mapping

Official implementation of the paper:

**Cross-Polarization Fusion of VV and VH SAR Observations for Improved Flood Mapping**  
Jagrati Talreja, Tewodros Syum Gebre, Leila Hashemi-Beni  
*IEEE IGARSS 2026 (under review)*

---

## 🌊 Motivation

Synthetic Aperture Radar (SAR) is the gold standard for flood mapping—clouds, rain, darkness? SAR doesn’t care.

But here’s the catch:
- **VV polarization** → great for open water and smooth surfaces  
- **VH polarization** → better for vegetation and partially inundated areas  

Most methods pick *one* and call it a day.

**CPF doesn’t.**  
It explicitly *models the interaction* between VV and VH using attention-based fusion, instead of naive channel stacking.

---

## 🧠 Key Idea: Cross-Polarization Fusion (CPF)

CPF is an **attention-driven fusion module** designed to:
- Preserve polarization-specific scattering behavior
- Adaptively emphasize informative regions and channels
- Improve flood boundary delineation in heterogeneous environments

### What makes CPF different?
✔ Separate feature extraction for VV and VH  
✔ Bidirectional cross-polarization attention  
✔ Plug-and-play (no architecture redesign required)

---

## 🏗 Architecture Overview

**Pipeline**
1. VV and VH processed independently with lightweight convolutional stems
2. Features fused using **CPF module**
3. Fused representation passed to a segmentation backbone

**Backbones supported**
- U-Net (with skip connections)
- Convolutional Autoencoder (no skips)

> CPF is inserted at the **input stage** so improvements come from fusion—not deeper networks.

---

## 📊 Experimental Results

Evaluated under identical training conditions using:
- VV-only
- VH-only
- CPF (VV + VH)

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

---

## 🖼 Qualitative Results

Each result panel includes:
- SAR VV
- SAR VH
- CPF (VV, VH)
- Ground truth flood mask
- Prediction probability map
- Overlay of prediction on:
  - VV
  - VH
  - CPF (VV, VH)

This makes failure cases *painfully obvious*—which is exactly what you want.

---

## 🗂 Dataset

We use the **DeepFlood dataset**, derived from:
- Hurricane Matthew (2016)
- Hurricane Florence (2018)

**Sensor**
- Sentinel-1 SAR  
- Dual polarization: VV & VH  
- Resolution: 10 m  

**Evaluation strategy**
- Train: Hurricane Florence
- Test: Hurricane Matthew (cross-event generalization)

---

## ⚙️ Training Details

- Task: Binary flood segmentation
- Loss: Binary Cross-Entropy
- Optimizer: Same across all experiments
- Augmentation: Random flips and rotations
- Metrics: IoU, F1-score, Overall Accuracy

> No tricks. No cherry-picking. Fair comparisons only.

---

## 📁 Repository Structure

