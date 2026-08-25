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
- **lapetitemilf (Quick, identity fail)**: `MyDrive/FiratSuper/loras/lapetitemilf_lora.safetensors` (~13.6 MB)
  - Trigger: `ohwx woman`
  - 5 epochs, rank 16, UNet only — preview did not match the subject
- **lapetitemilf (Standard, pending)**: `MyDrive/FiratSuper/loras/lapetitemilf_standard.safetensors`
  - 10 epochs, rank 32, UNet + text encoder

