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
    BASE_MODEL_REPO = "runwayml/stable-diffusion-v1-5"
    BASE_MODEL_DIR = f"{MODELS_DIR}/stable-diffusion-v1-5"
elif MODEL_TYPE == "sdxl":
    BASE_MODEL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
    BASE_MODEL_DIR = f"{MODELS_DIR}/stable-diffusion-xl-base-1.0"
else:
    raise ValueError('MODEL_TYPE חייב להיות "sd15" או "sdxl"')

print("Project:", PROJECT_NAME)
print("Trigger:", TRIGGER_WORD)
print("Source folder ID:", SOURCE_FOLDER_ID)
print("Dataset:", DATASET_DIR)
print("Output:", OUTPUT_DIR)
print("Base model:", BASE_MODEL_REPO)"""
)

code(
    """# @title 3) התקנת sd-scripts ותלויות
import os
import subprocess
import sys

SD_SCRIPTS = "/content/sd-scripts"

if not os.path.exists(SD_SCRIPTS):
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/kohya-ss/sd-scripts.git",
            SD_SCRIPTS,
        ],
        check=True,
    )

os.chdir(SD_SCRIPTS)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "pip"], check=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "torch",
        "torchvision",
        "xformers",
        "bitsandbytes",
        "accelerate",
        "transformers",
        "diffusers",
        "safetensors",
        "opencv-python",
        "einops",
        "ftfy",
        "imagesize",
        "lion-pytorch",
        "prodigyopt",
        "altair",
        "huggingface_hub",
        "toml",
        "voluptuous",
        "invisible-watermark",
    ],
    check=True,
)

if os.path.exists("requirements.txt"):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        check=True,
    )

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "pyyaml",
    ],
    check=True,
)

# accelerate config (ידני — יותר אמין מ-accelerate config default)
import yaml

accel_config = {
    "compute_environment": "LOCAL_MACHINE",
    "distributed_type": "NO",
    "downcast_bf16": "no",
    "gpu_ids": "all",
    "machine_rank": 0,
    "main_training_function": "main",
    "mixed_precision": "fp16",
    "num_machines": 1,
    "num_processes": 1,
    "rdzv_backend": "static",
    "same_network": True,
    "tpu_env": [],
    "tpu_use_cluster": False,
    "tpu_use_sudo": False,
    "use_cpu": False,
}
with open("/content/accelerate_config.yaml", "w", encoding="utf-8") as f:
    yaml.dump(accel_config, f)

print("sd-scripts מוכן:", SD_SCRIPTS)"""
)

code(
    """# @title 4) הורדת מודל בסיס (פעם אחת, נשמר ב-Drive)
from huggingface_hub import snapshot_download
import os

marker = os.path.join(BASE_MODEL_DIR, "model_index.json")
if not os.path.exists(marker):
    print("מוריד מודל בסיס — זה ייקח כמה דקות בפעם הראשונה...")
    snapshot_download(
        repo_id=BASE_MODEL_REPO,
        local_dir=BASE_MODEL_DIR,
        local_dir_use_symlinks=False,
    )
    print("הורדה הושלמה.")
else:
    print("מודל בסיס כבר קיים ב-Drive:", BASE_MODEL_DIR)"""
)

md(
    """## 5) הכנת הדאטאסט

**הדאטאסט כבר מוכן ב-Drive** (25 תמונות + captions).
התא הבא ידלג אוטומטית אם `DATASET_PREPARED = True`."""
)

code(
    """# @title 5) העתקת תמונות (דילוג אם כבר מוכן)
import os
import shutil
import subprocess
import sys

if DATASET_PREPARED:
    image_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    count = len(
        [
            f
            for f in os.listdir(DATASET_DIR)
            if os.path.splitext(f)[1].lower() in image_ext
        ]
    )
    print(f"DATASET_PREPARED=True — מדלג. נמצאו {count} תמונות ב-{DATASET_DIR}")
else:
    IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def find_source_folder(folder_id):
        shortcut = f"/content/drive/.shortcut-targets-by-id/{folder_id}"
        if os.path.isdir(shortcut):
            return shortcut
        for root in ("/content/drive/MyDrive", "/content/drive/Shareddrives"):
            if not os.path.isdir(root):
                continue
            for dirpath, dirnames, _ in os.walk(root):
                if os.path.basename(dirpath) == "dataset" and "Lapetitemilf" in dirpath:
                    return dirpath
                if dirpath.count(os.sep) - root.count(os.sep) > 4:
                    dirnames.clear()
        return None

    src = find_source_folder(SOURCE_FOLDER_ID)
    if src is None:
        print("לא נמצא shortcut מקומי — מוריד את התיקייה עם gdown...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        tmp = "/content/source_dataset"
        os.makedirs(tmp, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gdown",
                "--folder",
                SOURCE_FOLDER_URL,
                "-O",
                tmp,
            ],
            check=True,
        )
        nested = [
            os.path.join(tmp, n)
            for n in os.listdir(tmp)
            if os.path.isdir(os.path.join(tmp, n))
        ]
        src = nested[0] if nested else tmp

    print("Source:", src)
    copied = 0
    for name in os.listdir(src):
        path = os.path.join(src, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMAGE_EXT:
            continue
        dest = os.path.join(DATASET_DIR, name)
        if not os.path.exists(dest):
            shutil.copy2(path, dest)
        copied += 1
    print(f"{copied} images ready in {DATASET_DIR}")"""
)

code(
    """# @title 6) יצירת captions (דילוג אם כבר מוכן)
import glob
import os

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(folder):
    files = []
    for path in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXT:
            files.append(path)
    return sorted(files)


images = list_images(DATASET_DIR)
if not images:
    raise RuntimeError(f"לא נמצאו תמונות ב-{DATASET_DIR}")

print(f"נמצאו {len(images)} תמונות")

if DATASET_PREPARED:
    missing = [p for p in images if not os.path.exists(os.path.splitext(p)[0] + ".txt")]
    if missing:
        print(f"אזהרה: חסרים {len(missing)} captions — ממשיך ליצור...")
        AUTO_CAPTION_RUN = True
    else:
        print("DATASET_PREPARED=True — כל ה-captions כבר קיימים. מדלג.")
        AUTO_CAPTION_RUN = False
else:
    AUTO_CAPTION_RUN = AUTO_CAPTION

if AUTO_CAPTION_RUN:
    if CAPTION_STYLE == "blip":
        from PIL import Image
        from transformers import BlipForConditionalGeneration, BlipProcessor
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)

        for img_path in images:
            caption_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(caption_path):
                continue
            image = Image.open(img_path).convert("RGB")
            inputs = processor(image, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=40)
            caption = processor.decode(out[0], skip_special_tokens=True)
            extra = f", {STYLE_TAGS}" if STYLE_TAGS else ""
            full_caption = f"{TRIGGER_WORD}, {caption}{extra}"
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(full_caption)
            print("Caption:", os.path.basename(caption_path))

    elif CAPTION_STYLE == "tags":
        tagger_script = "/content/sd-scripts/finetune/make_captions.py"
        if not os.path.exists(tagger_script):
            raise FileNotFoundError("Caption script not found in sd-scripts")
        import subprocess
        import sys

        subprocess.run(
            [
                sys.executable,
                tagger_script,
                "--caption_extension",
                ".txt",
                "--batch_size",
                "1",
                DATASET_DIR,
            ],
            check=True,
        )
        for img_path in images:
            caption_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(caption_path):
                with open(caption_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                extra = f", {STYLE_TAGS}" if STYLE_TAGS else ""
                if TRIGGER_WORD not in text:
                    text = f"{TRIGGER_WORD}, {text}{extra}"
                    with open(caption_path, "w", encoding="utf-8") as f:
                        f.write(text)
    else:
        raise ValueError('CAPTION_STYLE חייב להיות "blip" או "tags"')
else:
    missing = [p for p in images if not os.path.exists(os.path.splitext(p)[0] + ".txt")]
    if missing:
        raise RuntimeError(
            f"חסרים {len(missing)} קבצי caption (.txt). הפעל AUTO_CAPTION=True או צור ידנית."
        )

print("Captions ready.")"""
)

code(
    """# @title 7) אימון LoRA
import os
import subprocess
import textwrap

dataset_config_path = f"/content/{PROJECT_NAME}_dataset.toml"
grad_accum = 4 if MODEL_TYPE == "sdxl" else 1

dataset_toml = textwrap.dedent(
    f\"\"\"
    [general]
    shuffle_caption = true
    caption_extension = ".txt"
    keep_tokens = {KEEP_TOKENS}

    [[datasets]]
    resolution = {RESOLUTION}
    batch_size = 1
    enable_bucket = true
    min_bucket_reso = 256
    max_bucket_reso = 1024
    bucket_reso_steps = 64

      [[datasets.subsets]]
      image_dir = "{DATASET_DIR}"
    \"\"\"
).strip()

with open(dataset_config_path, "w", encoding="utf-8") as f:
    f.write(dataset_toml)

print("Dataset config:", dataset_config_path)
print(open(dataset_config_path).read())
print("Starting training...")

os.chdir("/content/sd-scripts")
cmd = [
    "accelerate",
    "launch",
    "--num_cpu_threads_per_process",
    "1",
    "--config_file",
    "/content/accelerate_config.yaml",
    "train_network.py",
    f"--pretrained_model_name_or_path={BASE_MODEL_DIR}",
    f"--dataset_config={dataset_config_path}",
    f"--output_dir={OUTPUT_DIR}",
    f"--output_name={PROJECT_NAME}_lora",
    "--save_model_as=safetensors",
    "--save_precision=fp16",
    "--save_every_n_epochs=1",
    f"--max_train_epochs={MAX_TRAIN_EPOCHS}",
    "--train_batch_size=1",
    "--gradient_checkpointing",
    f"--gradient_accumulation_steps={grad_accum}",
    f"--learning_rate={LEARNING_RATE}",
    "--lr_scheduler=cosine",
    "--lr_warmup_steps=0",
    "--optimizer_type=AdamW8bit",
    "--mixed_precision=fp16",
    "--seed=42",
    "--max_data_loader_n_workers=2",
    "--persistent_data_loader_workers",
    "--max_token_length=75",
    "--xformers",
    "--cache_latents",
    "--cache_latents_to_disk",
    "--network_module=networks.lora",
    f"--network_dim={NETWORK_DIM}",
    f"--network_alpha={NETWORK_ALPHA}",
    "--network_train_unet_only",
    f"--logging_dir={LOGS_DIR}",
    "--log_with=tensorboard",
]
if MODEL_TYPE == "sdxl":
    cmd.append("--sdxl")

result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print(result.stderr)
if result.returncode != 0:
    raise RuntimeError(f"Training failed (exit {result.returncode}). See output above.")
print("Training finished.")"""
)

code(
    """# @title 8) ייצוא LoRA סופי לתיקיית loras/
import glob
import os
import shutil

candidates = sorted(
    glob.glob(os.path.join(OUTPUT_DIR, "*.safetensors")),
    key=os.path.getmtime,
)
if not candidates:
    raise RuntimeError(f"לא נמצא LoRA ב-{OUTPUT_DIR}")

latest = candidates[-1]
final_name = f"{PROJECT_NAME}_lora.safetensors"
final_path = os.path.join(LORAS_DIR, final_name)
shutil.copy2(latest, final_path)

print("LoRA exported to:", final_path)
print("Size MB:", round(os.path.getsize(final_path) / 1024 / 1024, 2))"""
)

code(
    """# @title 9) בדיקה מהירה (אופציונלי)
import os
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from IPython.display import display

if MODEL_TYPE != "sd15":
    print("תא זה מוגדר ל-SD 1.5. ל-SDXL השתמש ב-AutoPipelineForText2Image.")
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    lora_file = os.path.join(LORAS_DIR, f"{PROJECT_NAME}_lora.safetensors")
    pipe.load_lora_weights(lora_file)

    prompt = (
        f"{TRIGGER_WORD}, swimsuit, lingerie, fashion photography, "
        "studio lighting, high quality, detailed"
    )
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
