# Dataset Report: lapetitemilf

- **Images**: 25 JPG original + 14 new body photos in `ADD_BODY_PHOTOS` (mp4 skipped)
- **Captioned**: original 25/25; new 14 have per-image captions (not a generic swimsuit line)
- **Average file size**: original ~90 KB; new body set includes camera files up to ~4 MB
- **Duplicates found**: Two near-pairs in the new set (carriage bikini crop; green-screen lingerie pair) — keep, poses differ
- **Quality issues**: 2 WhatsApp compressions, 2 Photoroom edits, 2 green-screen shots (captioned), 1 face-cropped mirror selfie, 1 face hidden by phone. Still usable for **body**.

## Diversity Assessment
- Original 25: weak body coverage (hair + top)
- New 14: mix of bikini (beach/outdoor) and lingerie (indoor), standing / sitting / kneeling, front and side
- Best identity+body frame: green-screen standing front (`20250924_221459.jpg`) — must keep "green screen" in the caption
- Consistent identity (same person) — correct for character LoRA

## Content-Style Balance
- Captions separate outfit/pose from trailing photoreal tags
- Trigger word `ohwx woman` present in all captions
- Lingerie is **not** labeled as swimsuit

## Gate 1 Decision: **GO for body run** (done). **FAIL for together run** until 8-12 new waist-up photos exist.

Face close-ups (~15) and full-body (~14) are already trained as separate files. The missing slice is **waist-up**: head about 1/4 of the frame, shoulders-to-waist visible, looking at camera, original camera files. Retraining the mixed folder will not join face and body.
