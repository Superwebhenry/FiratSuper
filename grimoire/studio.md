# Studio — FiratSuper

## GPU (Colab)
- **Provider**: Google Colab
- **GPU**: Colab Pro — T4 is enough; **A100 or L4** is faster. Do not use TPU.
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
- **lapetitemilf (Thorough, face better)**: `MyDrive/FiratSuper/loras/lapetitemilf_thorough.safetensors`
  - 15 epochs, rank 32, UNet + text encoder, Realistic Vision V5.1
  - Face preview improved. Cell 10 swimsuit pose sheet: 5 poses, **not the subject**.
- **lapetitemilf (Body)**: `MyDrive/FiratSuper/loras/lapetitemilf_body.safetensors`
  - Same Thorough recipe + 14 full-body photos. Figure closer. Face generic. Not for portraits.
- **lapetitemilf (Face, hits sometimes)**: `MyDrive/FiratSuper/loras/lapetitemilf_face.safetensors`
  - Same Thorough recipe + ~15 face keepers. Trigger: `ohwx woman`
  - Weight: 0.8-1.0. CLIP skip 2. Base: Realistic Vision V5.1
  - Cell 9c: **2 of 5 similar** on serious close-up looking at camera. Not locked.
  - Winning prompt: `ohwx woman, long wavy highlighted blonde hair, brown eyes, adult woman, portrait, close up face, looking at camera, serious, detailed face, photorealistic, raw photo`
  - Do not overwrite. Next lock file would be `lapetitemilf_face2` after NEW unique close-ups.

