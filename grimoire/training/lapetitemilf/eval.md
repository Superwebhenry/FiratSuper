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

## Decision

14 new photos are in `ADD_BODY_PHOTOS`. Import with cell 6b, train Thorough recipe as **`lapetitemilf_body`**. Keep Thorough for portraits.

Skip the mp4. Two green-screen shots stay, with "green screen background" in the caption.
