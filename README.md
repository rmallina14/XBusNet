# XBusNet: Text-Guided Breast Ultrasound Segmentation
This repository contains the code used in the paper ["XBusNet: Text-Guided Breast Ultrasound Segmentation via Multimodal Vision–Language Learning"](https://doi.org/10.3390/diagnostics15222849).

#### XBusNet is a multimodal medical segmentation system that combines:

- Global prompts derived from lesion size and approximate location
- Local prompts describing shape, margin, and BI-RADS properties
- A dual-branch architecture integrating CLIP-based global semantics with a U-Net/ResNet-based local decoder
- A prompt-driven modulation mechanism (Semantic Feature Adjustment, SFA)
- This enables segmentation aligned with clinical interpretation, particularly improving small and low-contrast lesions.

## Architecture Diagram
This is the Architecture of our model

<img width="558" height="317" alt="image" src="https://github.com/user-attachments/assets/54b8cf2c-e1ef-401f-882f-2ed82f5d76fb" />




## Quick Start

Clone the repository:
```bash
git clone https://github.com/AAR-UNLV/XBusNet.git
cd XBusNet
```
## Third Party Dependencies

XBusNet relies on the following external pretrained weights:

#### 1. ResNet50 Backbone (Local Feature Extractor)
Download the ImageNet-pretrained ResNet50 weights directly from PyTorch:
```
https://download.pytorch.org/models/resnet50-0676ba61.pth
```
#### 2. CLIP Text Encoder (Global Feature Extractor)
XBusNet uses the OpenAI CLIP ViT-B/16 model.  
The model is automatically downloaded when installed via the CLIP package:
```
pip install git+https://github.com/openai/CLIP.git
```

## Citation
```
Mallina, R.; Shareef, B. XBusNet: Text-Guided Breast Ultrasound Segmentation via Multimodal Vision–Language Learning.
Diagnostics 2025, 15, 2849. https://doi.org/10.3390/diagnostics15222849
```


