# Status: lapetitemilf

- **Phase**: Flux LoRA trained. First generate pass: user likes identity. Nude scar-lines were made worse by writing "no scars" in the prompt.
- **Eval (2026-08-27)**: seeds through 400, every nude still had vertical breast lines after the "no scars" prompt. Flux treats those words as content. Strip them. Nude: LoRA 0.75, guidance 2.5, new seeds from 501. Do not retrain.
- **Do not**: delete existing LoRAs, retrain SD 1.5, overwrite `lapetitemilf_face`, add porn to the dataset, put "no scars" in a Flux prompt.
- **Notebook**: https://colab.research.google.com/github/Superwebhenry/FiratSuper/blob/cursor/sd-lora-colab-34fe/notebooks/Flux_LoRA_Training_Colab.ipynb
