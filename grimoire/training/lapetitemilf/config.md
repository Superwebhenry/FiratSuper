# Training Config: lapetitemilf

- **Backend**: kohya sd-scripts
- **Base model**: SD 1.5 (`v1-5-pruned-emaonly.safetensors`)
- **Preset**: Quick (first run — test if concept works)
- **Key parameters**:
  - epochs: 5
  - learning_rate: 1e-4
  - batch: 1
  - rank (network_dim): 16
  - alpha: 16
  - resolution: 512
  - repeats: 10 (folder `10_ohwx_woman`)
  - optimizer: AdamW8bit
- **Estimated VRAM**: ~8–10 GB (T4 OK)
- **Estimated time**: ~15–25 min on Colab T4
- **Config file**: Colab notebook cell 2 + cell 7

## After Quick run
If preview looks promising → rerun with **Standard** preset (10 epochs, rank 32).
