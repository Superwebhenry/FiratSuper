# Evaluation: lapetitemilf

## Quick / Standard / Thorough (done)

- Quick: identity fail (UNet-only).
- Standard: face closer on vanilla SD 1.5, still plastic.
- Thorough on Realistic Vision: **portrait face improved**. User agreed.

## Body / swimsuit pose sheet (FAIL)

- Cell 10 produced 5 swimsuit poses (`preview_swim_SHEET.png`).
- User: **not close at all** (face and body).
- Cause: the 25 training images are small social JPGs; captions are almost all hair + top. There are not enough head-to-toe swimsuit shots for the LoRA to learn the figure.
- More epochs on the same 25 photos will not fix this.

## Decision

Stop generating. Collect **10-15 new photos**:
- same person
- swimsuit
- feet in frame (full body)
- mix of front / 3/4 / side / sitting / standing
- original camera files if possible, not tiny crops

Drop them in Drive folder `ADD_BODY_PHOTOS`, run import cell 6b, then retrain Thorough to a new file (`lapetitemilf_body`). Keep the current Thorough file for portraits.
