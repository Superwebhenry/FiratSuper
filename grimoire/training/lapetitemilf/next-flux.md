# Next: generate with locked Flux v2

Training is done. User (2026-08-31): v2 is a hit. Do **not** retrain.

Locked file: `MyDrive/FiratSuper/loras/lapetitemilf_flux_v2.safetensors`

## Make pictures (Colab A100)

Open (reopen the GitHub link if you need the locked notebook):

https://colab.research.google.com/github/Superwebhenry/FiratSuper/blob/cursor/sd-lora-colab-34fe/notebooks/Flux_LoRA_Training_Colab.ipynb

1. A100 GPU. One Google account. Allow ALL.
2. Cells **1, 2, 3**. New runtime: also **4**. Then **10**.
3. Do **not** run cells 5-9 (training is locked).
4. Cell 10: `MODE` = `identity` / `lingerie` / `nude` / `all`. Change `SEED` for new frames.
5. Nude: LoRA **0.75**, guidance **2.5**, stay frontal. No "no scars". Optional cell 11 refine.
6. Copy keepers to `MyDrive/FiratSuper/keepers/`.

If the tab is still open from training, just run cell 10.

## Later (optional)

ComfyUI: Flux.1 [dev] checkpoint + `lapetitemilf_flux_v2.safetensors`, trigger `ohwx woman`. Do not load SD 1.5 LoRAs on Flux.

Do not feed generated images back into a dataset.
