# Studio — FiratSuper

## GPU (Colab)
- **Provider**: Google Colab
- **GPU**: T4 (15 GB VRAM typical)
- **Runtime**: Python 3.10 + PyTorch (Colab preinstalled)

## Base Models
- **SD 1.5**: `MyDrive/FiratSuper/models/v1-5-pruned-emaonly.safetensors`

## Training Backend
- **kohya sd-scripts** v0.10.1 (Colab notebook)

## LoRAs
- **lapetitemilf** (Quick, rank 16, 5 epochs): `MyDrive/FiratSuper/loras/lapetitemilf_lora.safetensors` (~13.6 MB)
  - Trigger: `ohwx woman`
  - Start weight: 0.6-0.8

