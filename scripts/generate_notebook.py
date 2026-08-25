#!/usr/bin/env python3
"""Generate the FiratSuper Colab notebook (ASCII-only JSON for Colab)."""
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
    """# FiratSuper - Stable Diffusion LoRA training (Google Colab)

This notebook trains a character LoRA with **kohya sd-scripts**, stores files on **Google Drive**, and exports `.safetensors`.

**Dataset:** Drive folder `Lapetitemilf Model / dataset` - 25 JPG images.

## Before you start
1. **Runtime > Change runtime type > GPU** (T4 for SD 1.5)
2. Run cells **in order** (1 to 9)
3. Approve Google Drive access
4. **Cell 7:** `DRY_RUN = False` for full training (dry run already passed)

## Current preset: Standard (character identity)
- Quick run already finished. Preview looked like a generic SD 1.5 woman, not the subject.
- Cause: 5 epochs + UNet-only (text encoder was not trained), so `ohwx woman` did not bind to the face.
- This notebook trains **UNet + text encoder**, 10 epochs, rank 32 (~30-40 min on T4).
- Quick LoRA is kept as `loras/lapetitemilf_lora.safetensors` (not overwritten).

## Drive layout
```
MyDrive/FiratSuper/
|-- datasets/lapetitemilf/10_ohwx_woman/   # images + captions
|-- output/lapetitemilf/quick/             # Quick checkpoints (done)
|-- output/lapetitemilf/standard/          # Standard checkpoints
|-- models/                                # base model
`-- loras/                                 # final LoRA files
```"""
)

code(
    """# @title 1) GPU check
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU. Runtime > Change runtime type > T4 GPU, then rerun."
    )

gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU: {gpu}")
print(f"VRAM: {vram:.1f} GB")"""
)

code(
    """# @title 2) Mount Google Drive + project settings
from google.colab import drive
import os

drive.mount("/content/drive")

# === edit here ===
PROJECT_NAME = "lapetitemilf"
TRIGGER_WORD = "ohwx woman"       # trigger word in prompts after training
SOURCE_FOLDER_ID = "1FOwDPkzqjmOo0LPuNKmgJtK4YuWU9Pmi"
SOURCE_FOLDER_URL = "https://drive.google.com/drive/folders/1FOwDPkzqjmOo0LPuNKmgJtK4YuWU9Pmi"
REPEATS = 10                      # how many times each image counts per epoch
MODEL_TYPE = "sd15"               # "sd15" or "sdxl"
TRAINING_PRESET = "standard"      # "quick" | "standard" | "thorough"
TRAIN_TEXT_ENCODER = True         # required for character identity (trigger -> face)
DRY_RUN = False                   # True = 5 steps (Gate 4). False = full training
DATASET_PREPARED = True           # True = images+captions already on Drive
AUTO_CAPTION = False              # captions already exist
CAPTION_STYLE = "blip"
STYLE_TAGS = "fashion photo, swimsuit, lingerie, high quality"

PRESETS = {
    "quick": {"epochs": 5, "dim": 16, "alpha": 16, "lr": 1e-4, "save_every": 2},
    "standard": {"epochs": 10, "dim": 32, "alpha": 32, "lr": 1e-4, "save_every": 2},
    "thorough": {"epochs": 15, "dim": 32, "alpha": 32, "lr": 5e-5, "save_every": 3},
}
p = PRESETS.get(TRAINING_PRESET, PRESETS["quick"])
MAX_TRAIN_EPOCHS = p["epochs"]
NETWORK_DIM = p["dim"]
NETWORK_ALPHA = p["alpha"]
LEARNING_RATE = p["lr"]
SAVE_EVERY_N_EPOCHS = p["save_every"]
RESOLUTION = 512 if MODEL_TYPE == "sd15" else 1024
# =================
KEEP_TOKENS = len(TRIGGER_WORD.split())

ROOT = "/content/drive/MyDrive/FiratSuper"
DATASET_DIR = (
    f"{ROOT}/datasets/{PROJECT_NAME}/{REPEATS}_{TRIGGER_WORD.replace(' ', '_')}"
)
# Separate output folder so Standard does not overwrite Quick checkpoints
OUTPUT_DIR = f"{ROOT}/output/{PROJECT_NAME}/{TRAINING_PRESET}"
MODELS_DIR = f"{ROOT}/models"
LORAS_DIR = f"{ROOT}/loras"
LOGS_DIR = f"{ROOT}/logs/{PROJECT_NAME}/{TRAINING_PRESET}"
LORA_BASENAME = PROJECT_NAME + "_" + TRAINING_PRESET

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
print("Preset:", TRAINING_PRESET, p)
print("Train text encoder:", TRAIN_TEXT_ENCODER)
print("Dry run:", DRY_RUN)
print("Dataset:", DATASET_DIR)
print("Output:", OUTPUT_DIR)
print("LoRA export name:", LORA_BASENAME + ".safetensors")
print("Base model file:", BASE_MODEL_FILE)"""
)

code(
    """# @title 3) Install sd-scripts (do not overwrite Colab torch)
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
# Colab now ships transformers 5.x (no CLIPFeatureExtractor). Pin 4.x for kohya.
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "accelerate",
        "transformers>=4.46.0,<5.0.0",
        "diffusers==0.32.1",
        "safetensors",
        "einops",
        "ftfy",
        "opencv-python",
        "toml",
        "voluptuous",
        "imagesize",
        "huggingface_hub",
        "rich",
        "omegaconf",
        "lion-pytorch",
    ],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", SD_SCRIPTS, "--no-deps"],
    check=False,
)

# Fallback: kohya still imports CLIPFeatureExtractor (removed in transformers 5)
lpw = os.path.join(SD_SCRIPTS, "library", "lpw_stable_diffusion.py")
if os.path.exists(lpw):
    text = open(lpw, encoding="utf-8").read()
    old = "from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection"
    new = "from transformers import CLIPImageProcessor as CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection"
    if old in text:
        open(lpw, "w", encoding="utf-8").write(text.replace(old, new, 1))
        print("Patched CLIPFeatureExtractor -> CLIPImageProcessor")

import transformers
print("transformers:", transformers.__version__)
if transformers.__version__.startswith("5"):
    raise RuntimeError(
        "transformers 5.x is still loaded. Runtime > Restart session, then run cells 1-4 and 7 again."
    )
print("sd-scripts ready:", SD_SCRIPTS)
print("train_network.py exists:", os.path.exists(os.path.join(SD_SCRIPTS, "train_network.py")))"""
)

code(
    """# @title 4) Download SD 1.5 (single safetensors file)
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
    """## 5-6) Gate 1 - dataset check

25 images + 25 captions are already on Drive. The next cells verify they are present."""
)

code(
    """# @title 5) Check images
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
    """# @title 6) Check captions
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
    """# @title 7) Gate 3-4 + LoRA training (dry run or full)
import os
import subprocess
import sys
import torch

print("=== Gate 3: Environment ===")
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO")
if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print("VRAM GB:", round(vram_gb, 1))
print("Model file:", os.path.exists(BASE_MODEL_FILE), BASE_MODEL_FILE)
print("Dataset:", os.path.exists(DATASET_DIR), DATASET_DIR)
print("sd-scripts:", os.path.exists("/content/sd-scripts/train_network.py"))
if not os.path.exists(BASE_MODEL_FILE):
    raise RuntimeError("Base model missing - rerun cell 4.")
if not torch.cuda.is_available():
    raise RuntimeError("No GPU - Runtime -> Change runtime type -> T4 GPU")

parent_dataset = os.path.dirname(DATASET_DIR)
os.chdir("/content/sd-scripts")

# Direct python (no accelerate launch). Colab + accelerate.launch often exits 1 with empty logs.
def build_cmd(max_epochs, max_steps=None):
    c = [
        sys.executable,
        "-u",
        "train_network.py",
        "--pretrained_model_name_or_path=" + BASE_MODEL_FILE,
        "--train_data_dir=" + parent_dataset,
        "--output_dir=" + OUTPUT_DIR,
        "--output_name=" + LORA_BASENAME,
        "--save_model_as=safetensors",
        "--save_precision=fp16",
        "--save_every_n_epochs=" + str(SAVE_EVERY_N_EPOCHS),
        "--max_train_epochs=" + str(max_epochs),
        "--train_batch_size=1",
        "--resolution=" + str(RESOLUTION) + "," + str(RESOLUTION),
        "--caption_extension=.txt",
        "--shuffle_caption",
        "--keep_tokens=" + str(KEEP_TOKENS),
        "--enable_bucket",
        "--min_bucket_reso=256",
        "--max_bucket_reso=1024",
        "--bucket_reso_steps=64",
        "--gradient_checkpointing",
        "--learning_rate=" + str(LEARNING_RATE),
        "--lr_scheduler=cosine",
        "--lr_warmup_steps=0",
        "--optimizer_type=AdamW",
        "--mixed_precision=fp16",
        "--seed=42",
        "--max_data_loader_n_workers=0",
        "--cache_latents",
        "--sdpa",
        "--network_module=networks.lora",
        "--network_dim=" + str(NETWORK_DIM),
        "--network_alpha=" + str(NETWORK_ALPHA),
        "--unet_lr=" + str(LEARNING_RATE),
        "--max_grad_norm=1.0",
        "--logging_dir=" + LOGS_DIR,
        "--console_log_simple",
    ]
    # Character LoRA: train CLIP so the trigger word maps to the face.
    if TRAIN_TEXT_ENCODER:
        c.append("--text_encoder_lr=5e-5")
    else:
        c.append("--network_train_unet_only")
    if max_steps is not None:
        c.append("--max_train_steps=" + str(max_steps))
    return c

if DRY_RUN:
    print("=== Gate 4: Dry Run (5 steps) ===")
    cmd = build_cmd(max_epochs=1, max_steps=5)
else:
    print("=== Full training:", TRAINING_PRESET, "preset ===")
    print("Train text encoder:", TRAIN_TEXT_ENCODER)
    cmd = build_cmd(max_epochs=MAX_TRAIN_EPOCHS)

print("Command:")
print(" ".join(cmd))
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["ACCELERATE_DISABLE_RICH"] = "1"
result = subprocess.run(
    cmd,
    cwd="/content/sd-scripts",
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
log = result.stdout or "(no output from train_network.py)"
print(log)
if result.returncode != 0:
    tail = log[-4000:] if log else ""
    raise RuntimeError(
        "Training failed (exit " + str(result.returncode) + "). Last log lines: " + tail
    )
if DRY_RUN:
    print("Dry run PASSED. Set DRY_RUN = False in cell 2 and rerun this cell for full training.")
else:
    print("Training finished.")"""
)

code(
    """# @title 8) Export final LoRA
import glob
import os
import shutil

pattern = os.path.join(OUTPUT_DIR, LORA_BASENAME + ".safetensors")
files = glob.glob(pattern)
if len(files) == 0:
    pattern = os.path.join(OUTPUT_DIR, "*.safetensors")
    files = glob.glob(pattern)
files.sort(key=os.path.getmtime)
if len(files) == 0:
    raise RuntimeError("No LoRA found in " + OUTPUT_DIR)

latest = files[-1]
final_name = LORA_BASENAME + ".safetensors"
final_path = os.path.join(LORAS_DIR, final_name)

# Keep the Quick LoRA if it still exists under the old name
quick_path = os.path.join(LORAS_DIR, PROJECT_NAME + "_lora.safetensors")
quick_bak = os.path.join(LORAS_DIR, PROJECT_NAME + "_quick.safetensors")
if os.path.exists(quick_path) and not os.path.exists(quick_bak):
    shutil.copy2(quick_path, quick_bak)
    print("Kept Quick LoRA as:", quick_bak)

shutil.copy2(latest, final_path)
size_mb = round(os.path.getsize(final_path) / 1024 / 1024, 2)
print("Exported from:", latest)
print("LoRA exported to:", final_path)
print("Size MB:", size_mb)"""
)

code(
    """# @title 9) Identity check: base model vs LoRA (same seed)
import gc
import os
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from IPython.display import display
from safetensors.torch import safe_open

# Colab torchao is 0.10; peft requires >=0.16 and crashes.
# Patch the name INSIDE peft.tuners.lora.torchao (from-import copy, not import_utils).
def _disable_peft_torchao():
    def _no(*args, **kwargs):
        return False

    def _skip(*args, **kwargs):
        return None

    try:
        import peft.import_utils as iu
        iu.is_torchao_available = _no
    except Exception:
        pass
    try:
        import peft.tuners.lora.torchao as tao
        tao.is_torchao_available = _no
        tao.dispatch_torchao = _skip
    except Exception:
        pass

_disable_peft_torchao()

if MODEL_TYPE != "sd15":
    print("Preview cell is for SD 1.5 only.")
else:
    lora_file = os.path.join(LORAS_DIR, LORA_BASENAME + ".safetensors")
    if not os.path.exists(lora_file):
        raise RuntimeError("LoRA not found: " + lora_file + " - rerun cell 8.")

    with safe_open(lora_file, framework="pt") as st:
        keys = list(st.keys())
    n_te = sum(1 for k in keys if "lora_te" in k or "text_encoder" in k)
    n_unet = sum(1 for k in keys if "lora_unet" in k or k.startswith("unet"))
    print("LoRA file:", lora_file)
    print("LoRA keys:", len(keys), "| unet:", n_unet, "| text_encoder:", n_te)
    if TRAIN_TEXT_ENCODER and n_te == 0:
        print("WARNING: no text-encoder keys. Trigger word will not bind to the face.")

    if "pipe" in globals():
        try:
            del pipe
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    print("Loading base model from Drive...")
    pipe = StableDiffusionPipeline.from_single_file(
        BASE_MODEL_FILE,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    _disable_peft_torchao()

    prompt = (
        TRIGGER_WORD
        + ", portrait, close up face, looking at camera, fashion photography, high quality"
    )
    seed = 42
    print("Prompt:", prompt)
    print("Generating OFF (base model only)...")
    gen0 = torch.Generator(device="cuda").manual_seed(seed)
    image_off = pipe(
        prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=gen0,
    ).images[0]
    off_path = os.path.join(OUTPUT_DIR, "preview_off.png")
    image_off.save(off_path)
    print("BASE MODEL (no LoRA) - if this already looks like the subject, ignore LoRA:")
    display(image_off)

    print("Loading LoRA at weight 1.0 (no fuse)...")
    pipe.load_lora_weights(lora_file)
    gen1 = torch.Generator(device="cuda").manual_seed(seed)
    image_on = pipe(
        prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=gen1,
        cross_attention_kwargs={"scale": 1.0},
    ).images[0]
    on_path = os.path.join(OUTPUT_DIR, "preview_on.png")
    image_on.save(on_path)
    print("WITH LORA weight 1.0 - should look closer to the training photos:")
    display(image_on)
    print("Saved:", off_path)
    print("Saved:", on_path)
    print("If OFF and ON look the same, the LoRA did not apply.")
    print("If ON is different but still not the subject, train longer (thorough).")
    print("Ignore HF_TOKEN warning - public SD 1.5 does not need a token.")"""
)

md(
    """## Done

Standard LoRA export:
`MyDrive/FiratSuper/loras/lapetitemilf_standard.safetensors`

Quick LoRA (kept, identity was weak):
`MyDrive/FiratSuper/loras/lapetitemilf_lora.safetensors`

### Use in Automatic1111 / ComfyUI / Forge
1. Copy the Standard `.safetensors` file to `models/Lora/`
2. Prompt: `ohwx woman, portrait, close up face, ...`
3. Character LoRA weight: start at `0.8-1.0`

### How to read cell 9
- OFF = base SD 1.5 (generic woman)
- ON = same prompt + LoRA
- You want ON to look like the training photos, not like OFF

### Next if identity is still weak
- Set `TRAINING_PRESET = "thorough"` in cell 2 and rerun cell 7
- Add more face close-ups to the dataset"""
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
payload = json.loads(nbformat.writes(notebook))
# ASCII-only JSON: Colab fails to open notebooks with raw control chars or mixed UTF-8.
text = json.dumps(payload, ensure_ascii=True, indent=1) + "\n"
if any(ord(ch) > 127 for ch in text):
    raise RuntimeError("notebook JSON is not ASCII")
if "\\u" in text:
    raise RuntimeError("notebook JSON still contains unicode escapes")
json.loads(text, strict=True)
if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in text):
    raise RuntimeError("notebook contains raw control characters")
out.write_text(text, encoding="ascii")
print(f"Wrote {out} ({len(cells)} cells, {out.stat().st_size} bytes)")
