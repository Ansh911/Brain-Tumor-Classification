# Brain Tumor Classification using CBAM-Enhanced ResNet50

## Overview

This project presents a deep learning framework for brain tumor classification from MRI scans using transfer learning and the Convolutional Block Attention Module (CBAM).

## Dataset

* ~7200 MRI images
* Classes:

  * Glioma
  * Meningioma
  * Pituitary
  * No Tumor

## Models Evaluated

* MobileNetV2
* ResNet50
* ResNet50 + CBAM

## Results

| Model           | Test Accuracy |
| --------------- | ------------- |
| MobileNetV2     | 84.19%        |
| ResNet50        | 84.62%        |
| ResNet50 + CBAM | 92.06%        |

CBAM improved classification accuracy by **7.44%** over the baseline ResNet50 model.

## Tech Stack

* Python
* PyTorch
* NumPy
* Matplotlib
* Scikit-Learn

## Key Features

* Transfer Learning
* Attention Mechanisms (CBAM)
* MRI Image Classification
* Confusion Matrix Analysis
* Performance Comparison Across Architectures

## Repository Structure

```text
.
├── brain_tumor_cbam_resnet50.py
├── notebook.ipynb
├── results/
└── README.md
```

## Author

Ansh Hatwal
