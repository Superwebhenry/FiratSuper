# Flux Gate 1: ADD_FLUX_PHOTOS audit (2026-08-26)

Folder: `MyDrive/FiratSuper/ADD_FLUX_PHOTOS/` (`1oLtTmwg2kt-Jn6zuci06ipRQoK6AOFVZ`)

## Decision: **GO**

- 31 keeper images
- 31 matching `.txt` captions on Drive and in `grimoire/training/lapetitemilf/captions/`
- Trigger: `ohwx woman`
- Mix: face close-ups, waist-up, standing full body, sitting full body
- No WhatsApp, no Photoroom, no extra people, no video

Trainer chosen (2026-08-26): **Colab Pro A100** + Ostris ai-toolkit.
Notebook: `notebooks/Flux_LoRA_Training_Colab.ipynb`
v1 file (done, protected): `lapetitemilf_flux`. Do not overwrite `lapetitemilf_face`.

## v2 add-on (2026-08-31)

See `flux-chest.md`. Gate 1 now copies `ADD_FLUX_PHOTOS` + `ADD_FLUX_CHEST`.
Skip images with no `.txt`. Expected pairs: **38**. New file: `lapetitemilf_flux_v2`.
