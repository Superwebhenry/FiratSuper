# Training Config: lapetitemilf

- **Backend**: kohya sd-scripts
- **Base model**: Realistic Vision V5.1 (`Realistic_Vision_V5.1_fp16-no-ema.safetensors`)
- **VAE**: `vae-ft-mse-840000-ema-pruned.safetensors` (preview)
- **Preset**: Thorough (after Standard identity was close but plastic)
- **Key parameters**:
  - epochs: 15
  - unet learning_rate: 5e-5
  - text_encoder_lr: 5e-5
  - batch: 1
  - rank (network_dim): 32
  - alpha: 32
  - resolution: 512
  - repeats: 10 (folder `10_ohwx_woman`)
  - optimizer: AdamW
  - clip_skip: 2
  - noise_offset: 0.1
  - train text encoder: **yes**
- **Estimated VRAM**: ~8–12 GB (T4 OK with gradient checkpointing)
- **Estimated time**: ~45–60 min on Colab T4 (+ download of RV ~2 GB)
- **Config file**: Colab notebook cell 2 + cell 7
- **Output name**: `lapetitemilf_face` (body and Thorough files stay)

## History
- Quick (done): 5 epochs, rank 16, UNet-only, vanilla SD 1.5. Identity fail.
- Standard (done): 10 epochs, rank 32, TE on, vanilla SD 1.5. Identity closer; look is plastic.
- Thorough (done): same rank, 15 epochs, lower LR, Realistic Vision V5.1. Face closer; swimsuit poses not the subject.
- Body (done): same Thorough recipe + 14 imported full-body photos. Figure closer. Face generic.
- Face (done): same Thorough recipe + ~15 face keepers. Cell 9c: 2 of 5 similar. Do not overwrite.
- Face2 (only after NEW unique NEUTRAL close-ups): `RUN_NAME = "face2"`.
