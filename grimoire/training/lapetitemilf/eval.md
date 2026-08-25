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
