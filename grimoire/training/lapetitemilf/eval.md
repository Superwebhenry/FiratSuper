# Evaluation: lapetitemilf

## Quick / Standard / Thorough (done)

- Quick: identity fail (UNet-only).
- Standard: face closer on vanilla SD 1.5, still plastic.
- Thorough on Realistic Vision: **portrait face improved**. User agreed.

## Body / swimsuit pose sheet (FAIL)

- Cell 10 produced 5 swimsuit poses (`preview_swim_SHEET.png`) — five different frames, not a display duplicate.
- User (2026-08-25): **not close at all** (face and body). Confirmed after seeing all 5.
- Cause: the 25 training images are small social JPGs; captions are almost all hair + top. There are not enough head-to-toe swimsuit shots for the LoRA to learn the figure.
- More epochs on the same 25 photos will not fix this. Raising LoRA weight will not either.

## Body LoRA (`lapetitemilf_body`) — closer, not locked

- User: **not far, but still not close**.
- Body/figure improved vs Thorough swimsuit sheet. Face still generic, especially on full-body frames.
- Cause mix: SD 1.5 full-body faces are tiny; eval prompts were only `ohwx woman + swimsuit` so Realistic Vision pulled a generic bikini woman; original photos are still small social JPGs.
- Next: **prompt lock, no retrain** (hair/eyes/adult woman in cell 9-10). Judge face on cell 9 and `1_waist_up`.
- If waist_up is still not her: add sharp camera close-ups, not more epochs.

## Face close-ups uploaded (2026-08-25 evening)

- Inbox now has a new batch mixed with the old body photos.
- **Keep ~15**: 9 face close-ups (looking at camera / smile / side glance) plus full-body shots that still show the face.
- **Skip**: mp4, 3 blurry night Santa frames (one with extra people), 2 re-uploads of photos already in the body set.
- Instagram screenshot is usable as a face crop but is smoothed; one of many, not the only face source.

## Face LoRA (`lapetitemilf_face`) — hits sometimes, not locked

- User: close-up is a bit similar, still not her. Body LoRA cell 9 face is unrelated (expected).
- Face sheet (cell 9b): **`2_front_neutral` similar**, other four so-so.
- Hit-prompt sheet (cell 9c, 2026-08-26): **2 of 5 similar**. Same serious close-up prompt, 5 seeds.
- Diagnosis: identity is in the weights. Seed lottery is still too high (usable ~40%, not a lock).
- Failure mode: insufficient unique sharp NEUTRAL close-ups looking at camera. Smiles / head-tilt / beauty-filter frames pull a generic woman.
- User (2026-08-26): faces look OK, **about 60% similar**. Ready for face AND body together.

### Winning portrait prompt

```
ohwx woman, long wavy highlighted blonde hair, brown eyes, adult woman, portrait, close up face, looking at camera, serious, detailed face, photorealistic, raw photo, natural skin texture, natural lighting, high quality
```

- Seed that already hit: **707**. Keep it when generating.
- Sweet spot: LoRA weight 0.8-1.0, CLIP skip 2, Realistic Vision V5.1.

## Face + body together

Cell 11 stack **failed**. Loading face LoRA + body LoRA in one generate fried frames 2-3 (RGB noise / glitch). Frame 1 (face LoRA only, waist-up) was the only usable image.

Do **not** stack two character LoRAs in one txt2img (Colab or Forge).

Together test = **face LoRA only on waist-up**. Full-body faces stay soft; use After Detailer in Forge.

Cell 11 (clean, one LoRA): user marked **frames 1-3 similar** (`1_face_waist`, `2_face_waist_b`, `3_face_waist_side`). Frame 4 full-body stand is not. Frame 5 body-LoRA waist-up is not.

Winning together recipe: **face LoRA only, waist-up, looking at camera, weight 0.9**. Seeds that hit: **707, 2025, 1301**.

Do **not** use `lapetitemilf_body` for identity. Do **not** stack two LoRAs.

Cell 11b (waist-up keepers): user **fine**. This is the production recipe.

Cell 13 full-body (2026-08-26): user **not even close**. Missing hands, arm fused into leg, cross-eyed, identity miss. SD 1.5 anatomy ceiling. Do not retrain this stack. Next: Flux + new sharp photos (`next-flux.md`).

### Winning waist-up prompt

```
ohwx woman, long wavy highlighted blonde hair, brown eyes, adult woman, waist up, swimsuit, looking at camera, detailed face, photorealistic, raw photo, natural lighting, high quality
```

### Photos for a later `together` run (Gate 1 currently FAIL)

Need **8-12 new waist-up** shots, not more close-ups and not more distant full-body:

- Head sharp and large (about 1/4 of the frame)
- Shoulders + chest + waist visible
- Looking at camera, original camera files
- Caption: `ohwx woman, long wavy highlighted blonde hair, brown eyes, adult woman, waist up, [outfit], looking at camera`

Then `IMPORT_AS = "both"`, `RUN_NAME = "together"` (writes `lapetitemilf_together`, does not overwrite face/body).

## Flux LoRA (`lapetitemilf_flux`) — first generate (2026-08-27)

- User: pictures are **not bad at all**. Identity usable on Flux.
- One nude waist-up had thin **vertical lines under both breasts**. Flux anatomy lottery, not a dataset scar.
- Bad advice (reverted): adding `no scars, no surgical marks` to the **positive** prompt. Flux has no real negative. User then ran seeds through 400 and **every** nude had the lines.
- Fix: delete those words. Nude prompt = identity + `waist-up standing nude, soft even indoor lighting, smooth skin`. LoRA weight **0.75**, guidance **2.5**, seeds from **501**. Do **not** retrain. Do **not** add porn photos.
- User then ran more frontals: still **a bit** of under-breast line, but identity is good. Stay **frontal**. Do not switch to side views as the fix.
- Next: cell 11 img2img refine on the best frontal (`STRENGTH` 0.22-0.32). Same prompt, no scar words. Do **not** retrain.

## Flux v2 chest add-on (2026-08-31)

- User uploaded 8 photos to `ADD_FLUX_CHEST/`.
- Audit: **7 KEEP**, **1 DROP** (`1788183845578.jpg`, EXIF 2026-08-27 — likely a Flux generate). Details: `flux-chest.md`.
- Train **v2 only**: `lapetitemilf_flux_v2`. Keep the original 31. Do not overwrite v1.
- Judge frontal nudes on cell 10 after v2. Stay frontal. Still no scar words.



