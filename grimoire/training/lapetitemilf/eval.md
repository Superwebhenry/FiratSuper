# Evaluation: lapetitemilf (Quick)

- **Checkpoint**: Quick final (`lapetitemilf_lora.safetensors`, rank 16, 5 epochs, UNet only)
- **Preview file**: `MyDrive/FiratSuper/output/lapetitemilf/preview.png`
- **Prompt**: `ohwx woman, swimsuit, fashion photography, studio lighting, high quality`
- **Weight**: 0.7 (`fuse_lora`)

## Result

Identity **failed**. The preview is a generic SD 1.5 fashion woman (studio, swimsuit, smooth face). It does not match the training photos.

This matches "trigger ignored / identity absent" in the training failure-modes guide, not overfit.

## Causes (stacked)

1. **Text encoder was not trained** (`--network_train_unet_only`). For a character LoRA the trigger `ohwx woman` must be learned by CLIP. UNet-only usually keeps a generic "woman".
2. **Quick preset is too short** for a face: 5 epochs, rank 16, ~1250 steps.
3. Preview used weight 0.7 and `fuse_lora`, which can hide a weak signal. Raising weight will not create identity that was never trained.

## Decision

Do **not** keep iterating on the Quick file. Retrain with **Standard**:

- 10 epochs, rank/alpha 32
- train UNet **and** text encoder (`text_encoder_lr=5e-5`)
- preview OFF vs ON at weight 1.0, same seed, close-up portrait prompt

Quick file is kept on Drive as `lapetitemilf_lora.safetensors` / `lapetitemilf_quick.safetensors`.
