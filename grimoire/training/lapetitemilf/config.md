# Training Config: lapetitemilf

- **Backend**: kohya sd-scripts
- **Base model**: SD 1.5 (`v1-5-pruned-emaonly.safetensors`)
- **Preset**: Standard (after Quick identity failure)
- **Key parameters**:
  - epochs: 10
  - unet learning_rate: 1e-4
  - text_encoder_lr: 5e-5
  - batch: 1
  - rank (network_dim): 32
  - alpha: 32
  - resolution: 512
  - repeats: 10 (folder `10_ohwx_woman`)
  - optimizer: AdamW
  - train text encoder: **yes** (required for character trigger -> face)
- **Estimated VRAM**: ~8–12 GB (T4 OK with gradient checkpointing)
- **Estimated time**: ~30–40 min on Colab T4
- **Config file**: Colab notebook cell 2 + cell 7
- **Output name**: `lapetitemilf_standard` (does not overwrite Quick)

## History
- Quick (done): 5 epochs, rank 16, UNet-only. Preview was generic SD 1.5 — identity fail.
- If Standard preview is close but weak → Thorough (15 epochs, lr 5e-5).
