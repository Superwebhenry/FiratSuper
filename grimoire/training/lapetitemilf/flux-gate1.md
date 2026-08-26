# Flux Gate 1: ADD_FLUX_PHOTOS audit (2026-08-26)

Folder: `MyDrive/FiratSuper/ADD_FLUX_PHOTOS/` (`1oLtTmwg2kt-Jn6zuci06ipRQoK6AOFVZ`)

## Summary

- Images: 28 JPEG, no video, no captions
- Long side: all except `face (10).jpg` are 1556px+; most numbered files are 3480px tall
- Exact MD5 duplicates: none
- Near-duplicates: 4 pairs (drop one from each)
- Extra people: none
- Identity: same adult woman across the set (blonde/bronde hair, brown eyes)

## Quality

Better than the old ~90 KB social JPGs. File sizes 259 KB to 3.0 MB.

Most numbered `178774*.jpg` files:

- height locked to 3480px
- camera Make/Model stripped
- DateTime stamped 2026-08-26 (re-export today)
- heavy JPEG quantization (`qt0=7`) at only 260-800 KB for a 3480px image

Those are phone-share / gallery re-exports, not camera originals. Still usable for Flux. Prefer originals if they still exist on the phone.

Real camera / high-quality keepers:

- `face (3).jpg` — Samsung SM-G996B EXIF, 2208x2944, 1.45 MB
- `face (10)(1).jpg` — 3376x2532, 3.0 MB
- `face (6).jpg`, `face (8).jpg`, `1787743839143.jpg` — stronger compression tables

Soft / beauty-smoothed (keep, but not as identity anchors): `face (9).jpg`, `face (8).jpg`, `1787744953823.jpg`.

## Drop (near-duplicates)

Keep the sharper file from each pair:

| Drop | Keep instead |
|------|----------------|
| `1787744239247.jpg` | `1787744211698.jpg` (same black gold-chain set, standing) |
| `1787744266618.jpg` | `1787744315770.jpg` (same night balcony) |
| `1787744388113.jpg` | `1787744409089.jpg` (same indoor black lace, hand behind head) |
| `1787745291211.jpg` | `1787745372054.jpg` (same kneeling pink on bed) |

Optional drop: `face (10).jpg` (958x1280, 254 KB). `face (10)(1).jpg` is the high-res version of a similar lying-down selfie.

After drops: **24 keepers** (or 23 if `face (10).jpg` goes). Target was 20-30.

## Mix among keepers

| Bucket | Count | Notes |
|--------|-------|--------|
| Face close-up, looking at camera | 6-7 | `3839143`, `face (6)`, `face (8)`, `face (9)`, `face (10)(1)`, plus chest-up `4538557` / `face (3)` |
| Waist-up, face large | 8+ | floral red, blue lace pair, black lace, neon lime, burgundy, car |
| Full body, face visible | 9 | standing indoor, one night balcony, one beach, one kneeling, one on bed |

Outfit mix: bikini + lingerie, indoor + one beach + one night balcony. One person only.

## Still missing

1. **Sitting full-body with face visible** (chair/sofa/edge of bed, head to toe). Kneeling and the car waist-up do not cover this.
2. A few more **tight face close-ups looking at camera** (target 8+; we are on the line).
3. **Daylight outdoor** beyond the one beach shot (sunglasses hide the eyes there).
4. Captions (`.txt` next to each image, trigger `ohwx woman`). Not started.

## Gate 1 Decision: **HOLD** (almost GO)

Photos are good enough to build a Flux set after dropping the 4 near-dups. Do **not** start training until:

- the 4 near-dups are out
- 3-4 sitting full-body shots are in (face still visible)
- captions exist

Do not overwrite `lapetitemilf_face`. New file only: `lapetitemilf_flux`.
