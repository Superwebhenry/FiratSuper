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

## Face + body together (next, no retrain yet)

SD 1.5 full-body images have a tiny face. The face LoRA cannot lock identity on a 40px head. Judge togetherness on **waist-up** (head about 1/4 of the frame).

Do **not** train a combined LoRA on the same photos. The face run already mixed that folder. More epochs will not join face and body.

### Today (cell 11)

1. Face LoRA only, waist-up, seed 707.
2. Face LoRA 0.85 + body LoRA 0.55, same prompt/seed.
3. One more stacked waist-up seed, plus two full-body frames (expect soft faces).

If frame 2 keeps her face and the body gets closer, stacking is the Forge workflow: face 0.8-0.9 + body 0.4-0.6 + After Detailer on full body.

### Photos for a later `together` run (Gate 1 currently FAIL)

Need **8-12 new waist-up** shots, not more close-ups and not more distant full-body:

- Head sharp and large (about 1/4 of the frame)
- Shoulders + chest + waist visible
- Looking at camera, original camera files
- Caption: `ohwx woman, long wavy highlighted blonde hair, brown eyes, adult woman, waist up, [outfit], looking at camera`

Then `IMPORT_AS = "both"`, `RUN_NAME = "together"` (writes `lapetitemilf_together`, does not overwrite face/body).

