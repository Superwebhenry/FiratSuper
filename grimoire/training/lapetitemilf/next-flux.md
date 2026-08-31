# Next: train Flux v2 on Colab A100

SD 1.5 is done. Flux v1 (`lapetitemilf_flux`) is trained and **protected**.

v2 adds 7 original chest photos so frontal nudes can lock the chest. Do not train on Flux outputs.

## Notebook

Open (must reopen the GitHub link so Colab is not a stale cache):

https://colab.research.google.com/github/Superwebhenry/FiratSuper/blob/cursor/sd-lora-colab-34fe/notebooks/Flux_LoRA_Training_Colab.ipynb

Runtime: **A100 GPU only**. Do not pick T4. Do not pick TPU.

Generator: `scripts/generate_flux_notebook.py`

Needs a Hugging Face READ token and license accept for `black-forest-labs/FLUX.1-dev`.

## Photos (Gate 1 GO for v2)

- 31 keepers in `ADD_FLUX_PHOTOS/` (`1oLtTmwg2kt-Jn6zuci06ipRQoK6AOFVZ`)
- 7 keepers in `ADD_FLUX_CHEST/` (`1iEmUvagFQVJ2TArN_7ee4Af4TUti1hZw`)
- 1 chest JPEG left uncaptioned on purpose (dropped gen)
- Captions: `grimoire/training/lapetitemilf/captions/`
- Trigger: `ohwx woman`
- Total: **38** pairs

## Output

New file only: `MyDrive/FiratSuper/loras/lapetitemilf_flux_v2.safetensors`

Do **not** overwrite `lapetitemilf_flux` or `lapetitemilf_face`.

## Run order

1. A100, one Google account, Allow ALL
2. Cells 1-8 (train). Cell 7 is the 5-step dry run.
3. Cells 9-11 (copy LoRA, generate, refine)

If the runtime dies after the LoRA is on Drive: rerun 1, 2, 3, 10, 11. Skip 4-9.
