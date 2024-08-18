# RFAConv-CBM-ViT: Enhanced Vision Transformer for Metal Surface Defect Detection

This repo contains the official **PyTorch** code for RFAConv-CBM-ViT .

## Introduction

<p align="center">
    <img src="figures/Fig1.jpg" width= "600">
</p>

In the metal manufacturing process, surface defect detection is crucial for maintaining product quality and production efficiency. Traditional models struggle with the diverse and subtle nature of defects, especially under conditions of class imbalance. Our proposed RFAConv-CBM-ViT model leverages advanced attention mechanisms to improve the focus on critical features and reduce the impact of outliers, leading to superior performance across multiple datasets.

### Key Features:
**Receptive-Field Attention Convolution (RFAConv)**: Enhances feature extraction by expanding the receptive field and applying a spatial attention mechanism.
**Context Broadcasting Median (CBM)**: Improves model robustness and training efficiency by using median pooling in attention maps, reducing the influence of noise and outliers.
**High Accuracy**: Demonstrated top-1 classification accuracy of 97.71% on the aluminum surface defect dataset, 99.25% on the X-SSD hot-rolled steel strip surface defect dataset, and 99.27% on the nut surface defect dataset.
