# Status: lapetitemilf

- **Phase**: Face ~60% similar. Next is face AND body together. No retrain yet.
- **Face LoRA**: `loras/lapetitemilf_face.safetensors` (keep)
- **Body LoRA**: `loras/lapetitemilf_body.safetensors` (keep)
- **Next**: reopen Colab, cell 9 then **cell 11** (face-only waist-up vs face+body stack). Do not run cell 7.
- **Later together run**: 8-12 new waist-up photos (`both_*`), then `RUN_NAME = "together"`.
- **Notebook**: https://colab.research.google.com/github/Superwebhenry/FiratSuper/blob/cursor/sd-lora-colab-34fe/notebooks/SD_LoRA_Training_Colab.ipynb
