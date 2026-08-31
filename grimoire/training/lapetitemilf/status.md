# Status: lapetitemilf

- **Phase**: Flux v1 trained. Chest add-on audited. **Flux v2 ready to train** (38 pairs).
- **v1 file (protected)**: `MyDrive/FiratSuper/loras/lapetitemilf_flux.safetensors`
- **v2 file (new only)**: `MyDrive/FiratSuper/loras/lapetitemilf_flux_v2.safetensors`
- **Chest inbox**: `ADD_FLUX_CHEST/` — 7 keepers + captions; 1 dropped (likely Flux gen, EXIF 2026-08-27). See `flux-chest.md`.
- **Do not**: delete existing LoRAs, retrain SD 1.5, overwrite `lapetitemilf_face` or `lapetitemilf_flux`, add porn, put "no scars" in a Flux prompt, train on Flux outputs.
- **Notebook**: https://colab.research.google.com/github/Superwebhenry/FiratSuper/blob/cursor/sd-lora-colab-34fe/notebooks/Flux_LoRA_Training_Colab.ipynb
- Reopen that GitHub link after each push (Colab caches old notebooks). Runtime: A100. Cells 1-8 train. Then 9-11 generate/refine.
