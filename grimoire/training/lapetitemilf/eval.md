# Evaluation: lapetitemilf

## Quick (done)

- **Checkpoint**: `lapetitemilf_lora.safetensors` (rank 16, 5 epochs, UNet only)
- Identity **failed**. Generic SD 1.5 woman.

## Standard (done)

- **Checkpoint**: `lapetitemilf_standard.safetensors` (rank 32, 10 epochs, UNet + text encoder)
- **Preview**: OFF vs ON, weight 1.0, portrait prompt
- **OFF**: generic SD 1.5 face (smooth, airbrushed). Does not look like the subject. Expected.
- **ON**: closer to the subject (hair/eye color, face structure). Identity is working.
- **Gap**: still not photographic. Vanilla SD 1.5 produces plastic skin even when identity is present.

## Decision

Do **not** raise LoRA weight on the Standard file to fake realism.

Retrain **Thorough** on **Realistic Vision V5.1** (photoreal SD 1.5):

- 15 epochs, rank/alpha 32, lr 5e-5, text encoder on
- `--clip_skip=2`, `--noise_offset=0.1`
- MSE VAE at preview
- Captions append `photorealistic, raw photo, natural skin`
- New file: `loras/lapetitemilf_thorough.safetensors` (does not overwrite Standard)

## Thorough (done)

- **Checkpoint**: `lapetitemilf_thorough.safetensors` (rank 32, 15 epochs, Realistic Vision V5.1)
- Face preview looked better than Standard (user: "looks better").
- Body was **not** tested in cell 9 (portrait close-up only).

## Body (user: face did not match on the full-body shot)

Expected on SD 1.5: at 512x768 the face is a tiny patch, so identity drops even when a portrait LoRA is good.

Cell 10 now generates a **swimsuit pose sheet** (no retraining):
- waist_up (face should still match)
- stand_front, stand_side, sitting, walking

If waist_up looks like her and full-body does not, use After Detailer in Forge for full-body, or add head-to-toe swimsuit photos and retrain. Do not raise LoRA weight to fake a body/face that the crop cannot hold.
