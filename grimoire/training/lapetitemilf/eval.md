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
- Do **not** retrain on the same photos. Do **not** raise LoRA weight to fake a lock. Do **not** overwrite `lapetitemilf_face`.

### Winning portrait prompt

```
ohwx woman, long wavy highlighted blonde hair, brown eyes, adult woman, portrait, close up face, looking at camera, serious, detailed face, photorealistic, raw photo, natural skin texture, natural lighting, high quality
```

- Seed that already hit: **707**. Keep it when generating.
- Sweet spot: LoRA weight 0.8-1.0, CLIP skip 2, Realistic Vision V5.1.

### Next (no retrain)

1. Cell **9d**: 10 more portraits of that prompt. Keep the similar files.
2. If they want **4/5 instead of 2/5**: unique sharp NEUTRAL face close-ups looking at camera (original camera files, not WhatsApp/Instagram smiles). Then `RUN_NAME = "face2"`.
3. Full-body face on SD 1.5 stays soft. Forge: After Detailer for full-body faces.
