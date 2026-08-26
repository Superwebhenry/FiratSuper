# Flux Gate 1: ADD_FLUX_PHOTOS audit (2026-08-26, pass 3)

Folder: `MyDrive/FiratSuper/ADD_FLUX_PHOTOS/` (`1oLtTmwg2kt-Jn6zuci06ipRQoK6AOFVZ`)

## Pass 3 (latest upload)

Pass-2 junk is mostly gone (`lapetitemilf (*)` and Photoroom PNG). User added 8 files at 13:13 UTC.

### Keep one

`1787749977324.jpg` (or `1787749926873.jpg` — same shot, keep one). Tight face, hair pulled back. Useful hair variety. Drop the other.

### Drop the rest of the new batch

These are the old social / Photoroom images again, just re-exported at 3480px:

| File | Why |
|------|-----|
| `1787749724307.jpg` | Same as old `lapetitemilf (5)` |
| `1787749746592.jpg` | Same as old `lapetitemilf (14)` (ahash dist 0) |
| `1787749775444.jpg` | Same as old `lapetitemilf (25)` |
| `1787749805307.jpg` | Same as old `lapetitemilf (20)` |
| `1787749887781.jpg` | Photoroom sand shot again |
| `1787749926873.jpg` | Near-dup of `1787749977324` (keep only one) |
| extra `1787746090924.jpg` | Exact re-upload of the existing car selfie (two copies in the folder) |
| `IMG-20230924-WA0044.jpg` | WhatsApp, still in the folder from pass 2 |

## Mix

Sitting camera originals from pass 2 still carry the set. Photo mix remains a GO after the drops above. About 30 keepers.

## Pass 3 note: bigger file is not a sharper original

The 3480px re-uploads of `lapetitemilf (5)/(14)/(20)/(25)` match those 1024px social crops (PSNR ~34-37 dB). Extra detail vs a Lanczos upscale of the old file is **0.97x-1.09x**. So the pixels are larger; the real face detail is not.

Optional extra: keep `1787749977324.jpg` (hair pulled back) and maybe `1787749724307.jpg` (dramatic lighting). Drop Photoroom, WhatsApp, the duplicate car selfie, and the other social re-exports.

## Gate 1 Decision: **HOLD for captions only**

Do not start Flux training. Do not overwrite `lapetitemilf_face`. New file only: `lapetitemilf_flux`.
