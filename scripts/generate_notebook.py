#!/usr/bin/env python3
"""Generate the FiratSuper Colab notebook."""
import json
from pathlib import Path

cells = []


def md(source: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.split("\n")})


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

נוטבוק זה מגדיר אימון LoRA מלא עם **kohya sd-scripts**, שומר הכל ב-**Google Drive**, ומייצא קובץ `.safetensors` מוכן לשימוש.

## לפני שמתחילים
1. **Runtime → Change runtime type → GPU** (מומלץ T4; SDXL דורש GPU חזק יותר)
2. הרץ את התאים **לפי הסדר**
3. בהרשאת Drive — אשר גישה לחשבון Google שלך
4. העלה תמונות אימון לתיקייה שתיווצר ב-Drive

## מבנה תיקיות ב-Drive
```
MyDrive/FiratSuper/
├── datasets/<שם_פרויקט>/10_trigger/   ← תמונות + קבצי .txt
├── output/<שם_פרויקט>/                  ← checkpoints בזמן אימון
├── models/                              ← מודל בסיס (מורד אוטומטית)
└── loras/                               ← LoRA סופי
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
PROJECT_NAME = "my_lora"          # שם הפרויקט (באנגלית, בלי רווחים)
TRIGGER_WORD = "sks person"       # מילת טריגר שתופיע בכל caption
MODEL_TYPE = "sd15"               # "sd15" או "sdxl"
MAX_TRAIN_EPOCHS = 10
NETWORK_DIM = 32
NETWORK_ALPHA = 16
LEARNING_RATE = 1e-4
RESOLUTION = 512 if MODEL_TYPE == "sd15" else 1024
AUTO_CAPTION = True               # יצירת captions אוטומטית לפני אימון
CAPTION_STYLE = "blip"            # "blip" (משפט) או "tags" (wd14 tagger)
# =================

ROOT = "/content/drive/MyDrive/FiratSuper"
DATASET_DIR = f"{ROOT}/datasets/{PROJECT_NAME}/10_{TRIGGER_WORD.replace(' ', '_')}"
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
    ["accelerate", "config", "default", "--config_file", "/content/accelerate_config.yaml"],
    check=True,
)
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
    """## 5) העלאת תמונות אימון

**דרישות:**
- 10–30 תמונות איכותיות (יותר = בדרך כלל יותר טוב)
- רזולוציה מינימלית ~512px
- פורמats: `.jpg`, `.jpeg`, `.png`, `.webp`
- גיוון בזוויות, תאורה, רקע

**אפשרות א — העלאה ידנית:**  
גרור תמונות לתיקייה `DATASET_DIR` ב-Google Drive (הנתיב מודפס בתא 2).

**אפשרות ב — העלאה מהמחשב:** הרץ את התא הבא."""
)

code(
    """# @title 5) העלאת תמונות מהמחשב (אופציונלי)
from google.colab import files
import os

uploaded = files.upload()
image_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

for name, data in uploaded.items():
    ext = os.path.splitext(name)[1].lower()
    if ext in image_ext:
        dest = os.path.join(DATASET_DIR, name)
        with open(dest, "wb") as f:
            f.write(data)
        print("Saved:", dest)
    else:
        print("Skipped (not an image):", name)

count = len(
    [
        f
        for f in os.listdir(DATASET_DIR)
        if os.path.splitext(f)[1].lower() in image_ext
    ]
)
print("\\nImages in dataset:", count)"""
)

code(
    """# @title 6) יצירת captions אוטומטית
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
    raise RuntimeError(
        f"לא נמצאו תמונות ב-{DATASET_DIR}. העלה תמונות ל-Drive והרץ שוב."
    )

print(f"נמצאו {len(images)} תמונות")

if AUTO_CAPTION:
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
            full_caption = f"{TRIGGER_WORD}, {caption}"
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
                if TRIGGER_WORD not in text:
                    with open(caption_path, "w", encoding="utf-8") as f:
                        f.write(f"{TRIGGER_WORD}, {text}")
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

parent_dataset = os.path.dirname(DATASET_DIR)
config_path = f"/content/{PROJECT_NAME}_train.toml"
sdxl_flag = "true" if MODEL_TYPE == "sdxl" else "false"
grad_accum = 4 if MODEL_TYPE == "sdxl" else 1

config = textwrap.dedent(
    f\"\"\"
    [model_arguments]
    pretrained_model_name_or_path = "{BASE_MODEL_DIR}"
    v2 = false
    v_parameterization = false
    sdxl = {sdxl_flag}

    [dataset_arguments]
    train_data_dir = "{parent_dataset}"
    reg_data_dir = ""
    resolution = "{RESOLUTION},{RESOLUTION}"
    enable_bucket = true
    min_bucket_reso = 256
    max_bucket_reso = 2048
    bucket_reso_steps = 64
    bucket_no_upscale = false
    caption_extension = ".txt"
    shuffle_caption = true
    keep_tokens = 1

    [training_arguments]
    output_dir = "{OUTPUT_DIR}"
    output_name = "{PROJECT_NAME}_lora"
    save_model_as = "safetensors"
    save_precision = "fp16"
    save_every_n_epochs = 1
    max_train_epochs = {MAX_TRAIN_EPOCHS}
    train_batch_size = 1
    gradient_checkpointing = true
    gradient_accumulation_steps = {grad_accum}
    learning_rate = {LEARNING_RATE}
    lr_scheduler = "cosine"
    lr_warmup_steps = 0
    optimizer_type = "AdamW8bit"
    mixed_precision = "fp16"
    seed = 42
    max_data_loader_n_workers = 2
    persistent_data_loader_workers = true
    max_token_length = 75
    xformers = true
    cache_latents = true
    cache_latents_to_disk = true

    [network_arguments]
    network_module = "networks.lora"
    network_dim = {NETWORK_DIM}
    network_alpha = {NETWORK_ALPHA}
    network_train_unet_only = true
    network_train_text_encoder = false

    [sample_prompt_arguments]
    sample_every_n_epochs = 0

    [logging_arguments]
    log_with = "tensorboard"
    logging_dir = "{LOGS_DIR}"
    \"\"\"
).strip()

with open(config_path, "w", encoding="utf-8") as f:
    f.write(config)

print("Config written:", config_path)
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
    "--config_file",
    config_path,
]
subprocess.run(cmd, check=True)
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

    prompt = f"portrait photo of {TRIGGER_WORD}, high quality, detailed"
    image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
    preview_path = os.path.join(OUTPUT_DIR, "preview.png")
    image.save(preview_path)
    display(image)
    print("Preview saved:", preview_path)"""
)

md(
    """## סיום

ה-LoRA שלך נמצא ב:
`MyDrive/FiratSuper/loras/<PROJECT_NAME>_lora.safetensors`

### שימוש ב-Automatic1111 / ComfyUI / Forge
1. העתק את קובץ `.safetensors` לתיקיית `models/Lora/`
2. בפרומпт השתמש במילת הטריגר שהגדרת (`TRIGGER_WORD`)
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

out = Path("/workspace/notebooks/SD_LoRA_Training_Colab.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {out} ({len(cells)} cells)")
