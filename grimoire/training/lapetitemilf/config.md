# Training Config: lapetitemilf

## Flux (current, Colab A100)

- **Backend**: Ostris ai-toolkit
- **Base model**: `black-forest-labs/FLUX.1-dev` (gated Hugging Face)
- **Notebook**: `notebooks/Flux_LoRA_Training_Colab.ipynb`
- **Recipe file**: `configs/train_lora_flux_a100.yaml`
- **Dataset**: `ADD_FLUX_PHOTOS/` (31) + `ADD_FLUX_CHEST/` (7 keepers) = 38 pairs
- **Trigger**: `ohwx woman`
- **Key parameters** (Ostris 24GB Flux recipe, A100 ~40 GB):
  - steps: 2000
  - batch: 1
  - rank (linear): 16
  - alpha: 16
  - lr: 1e-4
  - optimizer: adamw8bit
  - dtype: bf16
  - quantize: 8bit
  - noise_scheduler: flowmatch
  - train text encoder: **no** (Flux)
  - resolutions: 512 / 768 / 1024
  - gradient_checkpointing: yes
- **VRAM**: A100 40 GB (T4/L4 refused by the notebook)
- **Time**: first run downloads ~24 GB of Flux; then roughly 1-2 hours for 2000 steps
- **Output name**: `lapetitemilf_flux_v2` only. Never overwrite `lapetitemilf_flux` (v1) or `lapetitemilf_face`.

## SD 1.5 (archived, do not retrain)

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
- **Output name**: `lapetitemilf_face` (keep). Together file only after new waist-up photos.

## History
- Quick (done): 5 epochs, rank 16, UNet-only, vanilla SD 1.5. Identity fail.
- Standard (done): 10 epochs, rank 32, TE on, vanilla SD 1.5. Identity closer; look is plastic.
- Thorough (done): same rank, 15 epochs, lower LR, Realistic Vision V5.1. Face closer; swimsuit poses not the subject.
- Body (done): same Thorough recipe + 14 imported full-body photos. Figure closer. Face generic.
- Face (done): same Thorough recipe + ~15 face keepers. User: faces ~60% similar. Do not overwrite.
- Together (blocked): needs 8+ NEW waist-up `both_*` photos. `RUN_NAME = "together"`.
