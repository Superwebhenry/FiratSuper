# Studio — FiratSuper

## GPU (Colab)
- **Provider**: Google Colab
- **GPU**: T4 (15 GB VRAM typical)
- **Runtime**: Python 3.10 + PyTorch (Colab preinstalled)

## Base Models
- **SD 1.5**: `MyDrive/FiratSuper/models/v1-5-pruned-emaonly.safetensors`
- **Realistic Vision V5.1** (photoreal SD 1.5): `MyDrive/FiratSuper/models/Realistic_Vision_V5.1_fp16-no-ema.safetensors`
- **VAE**: `MyDrive/FiratSuper/models/vae-ft-mse-840000-ema-pruned.safetensors`

## Training Backend
- **kohya sd-scripts** v0.10.1 (Colab notebook)

## LoRAs
- **lapetitemilf (Quick, identity fail)**: `MyDrive/FiratSuper/loras/lapetitemilf_lora.safetensors` (~13.6 MB)
  - Trigger: `ohwx woman`
  - 5 epochs, rank 16, UNet only — preview did not match the subject
- **lapetitemilf (Standard, identity closer)**: `MyDrive/FiratSuper/loras/lapetitemilf_standard.safetensors`
  - 10 epochs, rank 32, UNet + text encoder, vanilla SD 1.5
  - ON preview closer to the subject; still plastic / airbrushed
- **lapetitemilf (Thorough, pending)**: `MyDrive/FiratSuper/loras/lapetitemilf_thorough.safetensors`
  - 15 epochs, rank 32, UNet + text encoder, Realistic Vision V5.1

