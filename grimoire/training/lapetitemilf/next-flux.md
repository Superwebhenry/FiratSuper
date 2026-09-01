# Next: generate with locked Flux v2

Training is done. User likes v2. Do **not** retrain.

Locked file: `MyDrive/FiratSuper/loras/lapetitemilf_flux_v2.safetensors`

## Make pictures (Colab A100)

Each of cells **13-22** is one series: same place, far camera, 20-shot gradual undress. Run one cell at a time.

User may send replacement series ideas; swap that cell only.

1. A100. Cells 1, 2, 3. New runtime: also 4. Skip 5-9.
2. Then **one** of 13-22 (~15-30 min, keep the tab open).
3. If it dies: set `SHOT_START` in that cell and rerun.
4. Cell 10 = identity/lingerie/nude. Cell 12 = mixed outdoor nudes.
5. Nude: LoRA 0.75, guidance 2.5. No "no scars".
6. Copy keepers to `keepers/`. Do not train on gens.
