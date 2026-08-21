#!/usr/bin/env python3
"""Generate the FiratSuper Colab notebook."""
import json
from pathlib import Path

import nbformat
from nbformat.validator import normalize

cells = []


def md(source: str) -> None:
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines:
        src.append(lines[-1])
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(source: str) -> None:
    lines = source.split("\n")
    src = [line + "\n" for line in lines[:-1]]
    if lines:
        src.append(lines[-1])
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "source": src,
            "outputs": [],
            "execution_count": None,
        }
    )


md(
    """# FiratSuper — אימון LoRA ל-Stable Diffusion (Google Colab)

נוטבוק זה מאמן LoRA עם **kohya sd-scripts**, שומר הכל ב-**Google Drive**, ומייצא `.safetensors`.

**דאטאסט מחובר:** תיקיית Drive `Lapetitemilf Model / dataset` — 25 תמונות JPG.

## לפני שמתחילים
1. **Runtime → Change runtime type → GPU** (T4 ל-SD 1.5)
2. הרץ את התאים **לפי הסדר**
3. אשר חיבור ל-Google Drive

## מבנה תיקיות ב-Drive
```
MyDrive/FiratSuper/
├── datasets/lapetitemilf/10_ohwx_woman/   ← עותק אימון + captions
├── output/lapetitemilf/                     ← checkpoints
├── models/                                  ← מודל בסיס
└── loras/                                   ← LoRA סופי
```"""
)

code(
    """# @title 1) בדיקת GPU
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "לא נמצא GPU! עבור ל-Runtime → Change runtime type → T4 GPU ואז הרץ שוב."
    )

gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {gpu}")
print(f"VRAM: {vram:.1f} GB")"""
)

code(
    """# @title 2) חיבור Google Drive + הגדרות פרויקט
from google.colab import drive
import os

drive.mount("/content/drive")

# === ערוך כאן ===
PROJECT_NAME = "lapetitemilf"
TRIGGER_WORD = "ohwx woman"       # מילת טריגר בפרומפט אחרי האימון
SOURCE_FOLDER_ID = "1FOwDPkzqjmOo0LPuNKmgJtK4YuWU9Pmi"
SOURCE_FOLDER_URL = "https://drive.google.com/drive/folders/1FOwDPkzqjmOo0LPuNKmgJtK4YuWU9Pmi"
REPEATS = 10                      # כמה פעמים כל תמונה נספרת ב-epoch
MODEL_TYPE = "sd15"               # "sd15" או "sdxl"
MAX_TRAIN_EPOCHS = 10
NETWORK_DIM = 32
NETWORK_ALPHA = 16
LEARNING_RATE = 1e-4
RESOLUTION = 512 if MODEL_TYPE == "sd15" else 1024
AUTO_CAPTION = True
CAPTION_STYLE = "blip"            # "blip" (משפט) או "tags"
STYLE_TAGS = "fashion photo, swimsuit, lingerie, high quality"
DATASET_PREPARED = True           # True = תמונות+captions כבר ב-Drive (Cursor הכין)
# =================
KEEP_TOKENS = len(TRIGGER_WORD.split())

ROOT = "/content/drive/MyDrive/FiratSuper"
DATASET_DIR = (
    f"{ROOT}/datasets/{PROJECT_NAME}/{REPEATS}_{TRIGGER_WORD.replace(' ', '_')}"
)
OUTPUT_DIR = f"{ROOT}/output/{PROJECT_NAME}"
MODELS_DIR = f"{ROOT}/models"
LORAS_DIR = f"{ROOT}/loras"
LOGS_DIR = f"{ROOT}/logs/{PROJECT_NAME}"

for path in [DATASET_DIR, OUTPUT_DIR, MODELS_DIR, LORAS_DIR, LOGS_DIR]:
    os.makedirs(path, exist_ok=True)

if MODEL_TYPE == "sd15":
    BASE_MODEL_FILE = MODELS_DIR + "/v1-5-pruned-emaonly.safetensors"
    BASE_MODEL_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    BASE_MODEL_NAME = "v1-5-pruned-emaonly.safetensors"
elif MODEL_TYPE == "sdxl":
    BASE_MODEL_FILE = MODELS_DIR + "/sd_xl_base_1.0.safetensors"
    BASE_MODEL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
    BASE_MODEL_NAME = "sd_xl_base_1.0.safetensors"
else:
    raise ValueError("MODEL_TYPE must be sd15 or sdxl")

print("Project:", PROJECT_NAME)
print("Trigger:", TRIGGER_WORD)
print("Dataset:", DATASET_DIR)
print("Output:", OUTPUT_DIR)
print("Base model file:", BASE_MODEL_FILE)"""
)

code(
    """# @title 3) התקנת sd-scripts (בלי לדרוס את Torch של Colab)
import os
import subprocess
import sys
import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Runtime -> Change runtime type -> T4 GPU")

SD_SCRIPTS = "/content/sd-scripts"
if not os.path.exists(os.path.join(SD_SCRIPTS, "train_network.py")):
    if os.path.exists(SD_SCRIPTS):
        subprocess.run(["rm", "-rf", SD_SCRIPTS], check=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "v0.10.1",
            "https://github.com/kohya-ss/sd-scripts.git",
            SD_SCRIPTS,
        ],
        check=True,
    )

os.chdir(SD_SCRIPTS)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "accelerate",
        "transformers",
        "diffusers",
        "safetensors",
        "bitsandbytes",
        "einops",
        "ftfy",
        "opencv-python",
        "toml",
        "voluptuous",
        "imagesize",
        "huggingface_hub",
    ],
    check=True,
)
print("sd-scripts ready:", SD_SCRIPTS)
print("train_network.py exists:", os.path.exists(os.path.join(SD_SCRIPTS, "train_network.py")))"""
)

code(
    """# @title 4) הורדת מודל SD 1.5 (קובץ safetensors יחיד)
from huggingface_hub import hf_hub_download
import os
import shutil

if os.path.exists(BASE_MODEL_FILE) and os.path.getsize(BASE_MODEL_FILE) > 1000000000:
    print("Base model already on Drive:", BASE_MODEL_FILE)
    print("Size GB:", round(os.path.getsize(BASE_MODEL_FILE) / 1024 / 1024 / 1024, 2))
else:
    print("Downloading base model to Drive. This takes several minutes...")
    downloaded = hf_hub_download(
        repo_id=BASE_MODEL_REPO,
        filename=BASE_MODEL_NAME,
        local_dir=MODELS_DIR,
    )
    if os.path.abspath(downloaded) != os.path.abspath(BASE_MODEL_FILE):
        shutil.copy2(downloaded, BASE_MODEL_FILE)
    print("Saved:", BASE_MODEL_FILE)
    print("Size GB:", round(os.path.getsize(BASE_MODEL_FILE) / 1024 / 1024 / 1024, 2))"""
)

md(
    """## 5) בדיקת דאטאסט

התמונות וה-captions כבר מוכנים ב-Drive. התאים הבאים רק מוודאים שהכל במקום."""
)

code(
    """# @title 5) בדיקת תמונות
import os

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
files = os.listdir(DATASET_DIR)
images = [f for f in files if f.lower().endswith(IMAGE_EXT)]
print("Dataset folder:", DATASET_DIR)
print("Images found:", len(images))
if len(images) == 0:
    raise RuntimeError("No images in dataset folder. Check Drive mount.")
print("OK")"""
)

code(
    """# @title 6) בדיקת captions
import os

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
images = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(IMAGE_EXT)]
missing = []
for name in images:
    stem = os.path.splitext(name)[0]
    txt = os.path.join(DATASET_DIR, stem + ".txt")
    if not os.path.exists(txt):
        missing.append(name)

print("Images:", len(images))
print("Missing captions:", len(missing))
if missing:
    raise RuntimeError("Missing caption files: " + ", ".join(missing[:5]))
print("Captions ready")"""
)

code(
    """# @title 7) אימון LoRA
import os
import subprocess
import sys
import torch

print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO")
print("Model file exists:", os.path.exists(BASE_MODEL_FILE), BASE_MODEL_FILE)
print("Dataset exists:", os.path.exists(DATASET_DIR), DATASET_DIR)
print("train_network.py:", os.path.exists("/content/sd-scripts/train_network.py"))
if not os.path.exists(BASE_MODEL_FILE):
    raise RuntimeError("Base model file missing. Re-run cell 4.")
if not torch.cuda.is_available():
    raise RuntimeError("No GPU. Runtime -> Change runtime type -> T4 GPU")

parent_dataset = os.path.dirname(DATASET_DIR)
os.chdir("/content/sd-scripts")
cmd = [
    sys.executable,
    "-m",
    "accelerate.commands.launch",
    "--num_cpu_threads_per_process=1",
    "--mixed_precision=fp16",
    "--num_processes=1",
    "train_network.py",
    "--pretrained_model_name_or_path=" + BASE_MODEL_FILE,
    "--train_data_dir=" + parent_dataset,
    "--output_dir=" + OUTPUT_DIR,
    "--output_name=" + PROJECT_NAME + "_lora",
    "--save_model_as=safetensors",
    "--save_precision=fp16",
    "--save_every_n_epochs=1",
    "--max_train_epochs=" + str(MAX_TRAIN_EPOCHS),
    "--train_batch_size=1",
    "--resolution=" + str(RESOLUTION),
    "--caption_extension=.txt",
    "--shuffle_caption",
    "--keep_tokens=" + str(KEEP_TOKENS),
    "--enable_bucket",
    "--gradient_checkpointing",
    "--learning_rate=" + str(LEARNING_RATE),
    "--lr_scheduler=cosine",
    "--lr_warmup_steps=0",
    "--optimizer_type=AdamW8bit",
    "--mixed_precision=fp16",
    "--seed=42",
    "--max_data_loader_n_workers=2",
    "--cache_latents",
    "--network_module=networks.lora",
    "--network_dim=" + str(NETWORK_DIM),
    "--network_alpha=" + str(NETWORK_ALPHA),
    "--network_train_unet_only",
    "--logging_dir=" + LOGS_DIR,
]
print("Starting training...")
print(" ".join(cmd))
result = subprocess.run(cmd)
if result.returncode != 0:
    raise RuntimeError("Training failed with exit code " + str(result.returncode) + ". Scroll up for the real error.")
print("Training finished.")"""
)

code(
    """# @title 8) ייצוא LoRA סופי
import glob
import os
import shutil

pattern = os.path.join(OUTPUT_DIR, "*.safetensors")
files = glob.glob(pattern)
files.sort(key=os.path.getmtime)
if len(files) == 0:
    raise RuntimeError("No LoRA found in " + OUTPUT_DIR)

latest = files[-1]
final_name = PROJECT_NAME + "_lora.safetensors"
final_path = os.path.join(LORAS_DIR, final_name)
shutil.copy2(latest, final_path)
size_mb = round(os.path.getsize(final_path) / 1024 / 1024, 2)
print("LoRA exported to:", final_path)
print("Size MB:", size_mb)"""
)

code(
    """# @title 9) בדיקה מהירה (אופציונלי)
import os
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from IPython.display import display

if MODEL_TYPE != "sd15":
    print("Preview cell is for SD 1.5 only.")
else:
    pipe = StableDiffusionPipeline.from_single_file(
        BASE_MODEL_FILE,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    lora_file = os.path.join(LORAS_DIR, PROJECT_NAME + "_lora.safetensors")
    pipe.load_lora_weights(lora_file)
    prompt = TRIGGER_WORD + ", swimsuit, fashion photography, studio lighting, high quality"
    image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
    preview_path = os.path.join(OUTPUT_DIR, "preview.png")
    image.save(preview_path)
    display(image)
    print("Preview saved:", preview_path)"""
)

md(
    """## סיום

ה-LoRA שלך נמצא ב:
`MyDrive/FiratSuper/loras/lapetitemilf_lora.safetensors`

### שימוש ב-Automatic1111 / ComfyUI / Forge
1. העתק את קובץ `.safetensors` לתיקיית `models/Lora/`
2. בפרומפט: `ohwx woman, swimsuit, ...` (או lingerie)
3. משקל LoRA מומלץ להתחלה: `0.6–0.9`

### טיפים
- אם התוצאה "נשכחת" את הנושא — הוסף עוד תמונות או העלה `MAX_TRAIN_EPOCHS`
- אם overfit (עותק מדויק מדי) — הורד epochs או הוסף גיוון בתמונות
- שמור את ה-Drive מסודר — checkpoints גדולים תופסים מקום"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

notebook = nbformat.from_dict(nb)
normalize(notebook)
for i, cell in enumerate(notebook.cells):
    if "id" not in cell:
        cell["id"] = f"cell-{i:04d}"

out = Path("/workspace/notebooks/SD_LoRA_Training_Colab.ipynb")
nbformat.write(notebook, out)
json.loads(out.read_text(encoding="utf-8"), strict=True)
print(f"Wrote {out} ({len(cells)} cells)")
