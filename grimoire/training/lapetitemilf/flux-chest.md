# Flux chest inbox audit (2026-08-31)

Folder: `MyDrive/FiratSuper/ADD_FLUX_CHEST/` (`1iEmUvagFQVJ2TArN_7ee4Af4TUti1hZw`)

Purpose: extra original photos so Flux v2 can learn the chest. Keep the v1 31. Do not train on Flux outputs.

## Decision: **GO as add-on** (7 keepers + 7 captions)

8 JPEGs uploaded. All 3480px tall social re-exports (same signature as earlier keepers). No hash dups.

| file | EXIF | decision |
|------|------|----------|
| `1788183598798.jpg` | 2025-08-22 | KEEP — kneeling on bed, zebra pillow |
| `1788183618588.jpg` | none | KEEP — glasses, unzipped jeans, gold bar pendant |
| `1788183636665.jpg` | 2026-05-08 | KEEP — bathtub, bird tattoo |
| `1788183657894.jpg` | 2026-06-15 | KEEP — window, topless, white wrap at hips |
| `1788183672326.jpg` | none | KEEP — standing, white lace underwear |
| `1788183698082.jpg` | none | KEEP — nude, pulling down underwear, orange curtains |
| `1788183845578.jpg` | **2026-08-27 16:43** | **DROP** — dated the Flux generate day; likely a generated image. Leave on Drive. Do not caption. Gate 1 skips images with no `.txt`. |
| `1788183872719.jpg` | none | KEEP — hot tub, wet hair, gold bar pendant |

Captions: `grimoire/training/lapetitemilf/captions/` and matching `.txt` on Drive.
Trigger: `ohwx woman`. Outfit / pose / setting named. Do not mention scars.

v2 train set = 31 (PHOTOS) + 7 (CHEST) = **38 pairs**.
New file only: `lapetitemilf_flux_v2`. Do not overwrite `lapetitemilf_flux` or `lapetitemilf_face`.
