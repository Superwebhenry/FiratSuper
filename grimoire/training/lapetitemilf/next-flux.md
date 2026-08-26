# Next: leave SD 1.5. Train Flux.

SD 1.5 + Realistic Vision is **not** the right tool for high-end lingerie/bikini of a real person.

User (2026-08-26) full-body cell 13: missing hands, arm fused into leg, cross-eyed, not similar. Those are **base-model** failures. Retraining the same SD 1.5 recipe on the same (or similar) photos will not fix anatomy.

Do **not** delete existing LoRA files. The waist-up face LoRA still has some identity value. Do **not** run another Thorough/face/body SD 1.5 train.

## What to use instead

**Flux.1 [dev] character LoRA** (or Flux.2 LoRA when the trainer is available). This is the current photoreal standard: hands, eyes, and full-body hold together far better than SD 1.5.

Paid trainers (user said they will pay):

- fal.ai Flux LoRA trainer
- Replicate `ostris/flux-dev-lora-trainer`

Colab Pro **A100** can also train Flux with Ostris ai-toolkit. T4 is too small for a comfortable Flux train.

Midjourney looks pretty but identity lock is weaker than a real character LoRA. Not the first choice if the face must be her.

## Photos required (Gate 1 for Flux)

The old 25 social JPGs (~90 KB) are too small. Flux needs sharp originals.

Drop **20-30 original camera files** (not WhatsApp, not Instagram beauty-filter) into a new Drive folder `MyDrive/FiratSuper/ADD_FLUX_PHOTOS/`:

- 8+ face close-ups, looking at camera, sharp, one person
- 8+ waist-up, face large, different outfits
- 8+ **full body**, face still visible, standing and sitting
- Mix of bikini and lingerie, indoor and outdoor
- No extra people, no blurry night shots, no video

Long side **1024px or more**. Keep the trigger `ohwx woman`.

## After photos are in Drive

Build a Flux Colab (ai-toolkit) **or** zip the set and train on fal.ai. New file only: `lapetitemilf_flux`. Do not overwrite `lapetitemilf_face`.
