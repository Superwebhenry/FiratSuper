# Next: train Flux on Colab A100

SD 1.5 + Realistic Vision is **not** the right tool for high-end lingerie/bikini of a real person.

User (2026-08-26) full-body cell 13: missing hands, arm fused into leg, cross-eyed, not similar. Those are **base-model** failures. Retraining the same SD 1.5 recipe will not fix anatomy.

Do **not** delete existing LoRA files. Do **not** run another Thorough/face/body SD 1.5 train.

## Trainer choice (2026-08-26): Colab Pro A100

User asked what is best for NSFW generation, then chose **Colab A100**.

- **Colab Pro A100 + Ostris ai-toolkit**: no platform safety checker. Generate nudes in the last notebook cell. Adult subject only.
- fal.ai / Replicate web trainers: convenience, but their hosted generate UIs typically run a safety checker. Not the path for NSFW.

Do **not** add porn to the dataset. Flux already knows anatomy. Lingerie / full-body identity photos are enough. NSFW is an inference prompt, not a training target.

## Notebook

Open (must reopen the GitHub link to see new cells):

https://colab.research.google.com/github/Superwebhenry/FiratSuper/blob/cursor/sd-lora-colab-34fe/notebooks/Flux_LoRA_Training_Colab.ipynb

Runtime: **A100 GPU only**. Do not pick T4. Do not pick TPU.

Generator: `scripts/generate_flux_notebook.py`
Recipe: `configs/train_lora_flux_a100.yaml` (Ostris 24GB Flux YAML, A100 ~40GB)

Needs a Hugging Face READ token and license accept for `black-forest-labs/FLUX.1-dev`.

## Photos (Gate 1 GO)

31 keepers + 31 captions in `MyDrive/FiratSuper/ADD_FLUX_PHOTOS/` (`1oLtTmwg2kt-Jn6zuci06ipRQoK6AOFVZ`).
Copies: `grimoire/training/lapetitemilf/captions/`.
Trigger: `ohwx woman`.

## Output

New file only: `MyDrive/FiratSuper/loras/lapetitemilf_flux.safetensors`

Do **not** overwrite `lapetitemilf_face` or any old SD LoRA.
