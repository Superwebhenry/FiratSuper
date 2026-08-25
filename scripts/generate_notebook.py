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
3. Approve Google Drive access. In the popup click Continue, then **Allow ALL permissions** (do not uncheck boxes).
4. **Cell 7:** `DRY_RUN = False` for full training (dry run already passed)

## If cell 2 fails: credential propagation
This is a Colab Drive-login popup issue, not the training code.
1. Left sidebar: folder icon, then **Mount Drive**, then rerun cell 2
2. Chrome, one Google account only (the account that owns the photos)
3. In the popup: Continue, then **Allow ALL permissions** (do not uncheck boxes)
4. If FUSE still fails, cell 2 tries Google login, then copies the dataset without mounting Drive

## Current preset: Thorough on Realistic Vision
- Standard run finished: OFF (base SD 1.5) is a generic face. ON is closer to the subject. Identity is working.
- Vanilla SD 1.5 still looks plastic / airbrushed. That is the base model, not a broken LoRA.
- This notebook retrains on **Realistic Vision V5.1** (photoreal SD 1.5), 15 epochs, text encoder on.
- Standard LoRA is kept as `loras/lapetitemilf_standard.safetensors` (not overwritten).
- New file: `loras/lapetitemilf_thorough.safetensors` (~45-60 min on T4).

## Drive layout
```
MyDrive/FiratSuper/
|-- datasets/lapetitemilf/10_ohwx_woman/   # images + captions
|-- output/lapetitemilf/quick/             # Quick checkpoints (done)
|-- output/lapetitemilf/standard/          # Standard checkpoints (done)
|-- output/lapetitemilf/thorough/          # Thorough checkpoints
|-- models/                                # base models
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
    """# @title 2) Connect Drive + project settings
from google.colab import auth, drive
import os

MOUNT = "/content/drive"
MYDRIVE = os.path.join(MOUNT, "MyDrive")
FIRATSUPER_DRIVE_ID = "18UE4fijDjq8ggmkDYjaUpRQXVDE0cqYt"
USE_DRIVE_API = False
DRIVE_SERVICE = None


def _drive_ok():
    return os.path.isdir(MYDRIVE)


def _mount_fuse(force=False):
    if _drive_ok() and not force:
        return True
    print("Drive popup: click Continue, then Allow ALL permissions. Do not uncheck boxes.")
    try:
        drive.mount(MOUNT, force_remount=force)
    except Exception as err:
        print("drive.mount failed:", err)
    return _drive_ok()


def _google_login():
    print("Google login popup (not the Drive FUSE popup)...")
    try:
        auth.authenticate_user()
        print("Google login OK")
        return True
    except Exception as err:
        print("Google login failed:", err)
        return False


def _api_service():
    from googleapiclient.discovery import build
    return build("drive", "v3")


def api_find_child(service, parent_id, name):
    q = "'" + parent_id + "' in parents and name = '" + name + "' and trashed = false"
    resp = service.files().list(
        q=q,
        fields="files(id, name, mimeType)",
        pageSize=10,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    return files[0] if files else None


def api_ensure_folder(service, parent_id, name):
    found = api_find_child(service, parent_id, name)
    if found:
        return found["id"]
    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    return service.files().create(
        body=meta, fields="id", supportsAllDrives=True
    ).execute()["id"]


def api_list_children(service, folder_id):
    items = []
    token = None
    while True:
        resp = service.files().list(
            q="'" + folder_id + "' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=token,
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        items.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            break
    return items


def api_download_file(service, file_id, dest):
    from googleapiclient.http import MediaIoBaseDownload
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with open(dest, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _status, done = downloader.next_chunk()


def api_download_folder(service, folder_id, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for item in api_list_children(service, folder_id):
        path = os.path.join(dest_dir, item["name"])
        if item["mimeType"] == "application/vnd.google-apps.folder":
            api_download_folder(service, item["id"], path)
        else:
            print("  copy", item["name"])
            api_download_file(service, item["id"], path)


def api_upload_file(service, local_path, parent_id, name):
    from googleapiclient.http import MediaFileUpload
    media = MediaFileUpload(local_path, resumable=True)
    found = api_find_child(service, parent_id, name)
    if found:
        service.files().update(
            fileId=found["id"], media_body=media, supportsAllDrives=True
        ).execute()
        return found["id"]
    body = {"name": name, "parents": [parent_id]}
    created = service.files().create(
        body=body, media_body=media, fields="id", supportsAllDrives=True
    ).execute()
    return created["id"]


def upload_project_file(local_path):
    if not USE_DRIVE_API:
        return
    if DRIVE_SERVICE is None:
        print("Skip Drive upload: no API client")
        return
    rel = os.path.relpath(local_path, ROOT)
    print("Uploading to Drive:", rel)
    parts = rel.split(os.sep)
    parent = FIRATSUPER_DRIVE_ID
    for folder in parts[:-1]:
        parent = api_ensure_folder(DRIVE_SERVICE, parent, folder)
    api_upload_file(DRIVE_SERVICE, local_path, parent, parts[-1])
    print("Uploaded:", rel)


def sync_dataset_via_api(local_root):
    print("FUSE mount failed. Copying dataset via Drive API to", local_root)
    service = _api_service()
    os.makedirs(local_root, exist_ok=True)
    datasets = api_find_child(service, FIRATSUPER_DRIVE_ID, "datasets")
    if not datasets:
        raise RuntimeError("Drive API: datasets folder not found in FiratSuper")
    project_ds = api_find_child(service, datasets["id"], "lapetitemilf")
    if not project_ds:
        raise RuntimeError("Drive API: datasets/lapetitemilf not found")
    api_download_folder(
        service,
        project_ds["id"],
        os.path.join(local_root, "datasets", "lapetitemilf"),
    )
    for sub in ("output", "models", "loras", "logs"):
        os.makedirs(os.path.join(local_root, sub), exist_ok=True)
    return service


if _drive_ok():
    print("Drive already mounted at", MYDRIVE)
elif _mount_fuse(force=False):
    print("Drive mounted at", MYDRIVE)
else:
    print("FUSE mount failed. Trying Google login, then mount again...")
    logged_in = _google_login()
    if logged_in and _mount_fuse(force=True):
        print("Drive mounted after Google login:", MYDRIVE)
    elif logged_in:
        USE_DRIVE_API = True
        DRIVE_SERVICE = sync_dataset_via_api("/content/FiratSuper")
        print("DRIVE MODE: API fallback (local /content/FiratSuper)")
        print("Base model will download from HuggingFace in cell 4.")
        print("LoRA and previews will upload back to Drive after training.")
    else:
        print("Could not connect to Drive.")
        print("1. Left sidebar: folder icon -> Mount Drive, then rerun this cell")
        print("2. Chrome, one Google account only (the account that owns the photos)")
        print("3. Allow ALL permissions. Do not close extra popups")
        print("4. Runtime > Disconnect and delete runtime, reconnect T4 GPU")
        raise RuntimeError("Drive is not connected. See the steps printed above.")

if (not USE_DRIVE_API) and (not _drive_ok()):
    raise RuntimeError("Drive folder is empty. Grant access and rerun this cell.")

# === edit here ===
PROJECT_NAME = "lapetitemilf"
TRIGGER_WORD = "ohwx woman"       # trigger word in prompts after training
SOURCE_FOLDER_ID = "1FOwDPkzqjmOo0LPuNKmgJtK4YuWU9Pmi"
SOURCE_FOLDER_URL = "https://drive.google.com/drive/folders/1FOwDPkzqjmOo0LPuNKmgJtK4YuWU9Pmi"
REPEATS = 10                      # how many times each image counts per epoch
MODEL_TYPE = "sd15"               # "sd15" or "sdxl"
BASE_CHECKPOINT = "realistic_vision"  # "sd15" or "realistic_vision"
TRAINING_PRESET = "thorough"      # "quick" | "standard" | "thorough"
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

if USE_DRIVE_API:
    ROOT = "/content/FiratSuper"
else:
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

if MODEL_TYPE == "sd15" and BASE_CHECKPOINT == "realistic_vision":
    BASE_MODEL_FILE = MODELS_DIR + "/Realistic_Vision_V5.1_fp16-no-ema.safetensors"
    BASE_MODEL_REPO = "SG161222/Realistic_Vision_V5.1_noVAE"
    BASE_MODEL_NAME = "Realistic_Vision_V5.1_fp16-no-ema.safetensors"
    VAE_FILE = MODELS_DIR + "/vae-ft-mse-840000-ema-pruned.safetensors"
    VAE_REPO = "stabilityai/sd-vae-ft-mse-original"
    VAE_NAME = "vae-ft-mse-840000-ema-pruned.safetensors"
    CLIP_SKIP = 2
elif MODEL_TYPE == "sd15":
    BASE_MODEL_FILE = MODELS_DIR + "/v1-5-pruned-emaonly.safetensors"
    BASE_MODEL_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    BASE_MODEL_NAME = "v1-5-pruned-emaonly.safetensors"
    VAE_FILE = ""
    VAE_REPO = ""
    VAE_NAME = ""
    CLIP_SKIP = 1
elif MODEL_TYPE == "sdxl":
    BASE_MODEL_FILE = MODELS_DIR + "/sd_xl_base_1.0.safetensors"
    BASE_MODEL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
    BASE_MODEL_NAME = "sd_xl_base_1.0.safetensors"
    VAE_FILE = ""
    VAE_REPO = ""
    VAE_NAME = ""
    CLIP_SKIP = 1
else:
    raise ValueError("MODEL_TYPE must be sd15 or sdxl")

print("Project:", PROJECT_NAME)
print("Trigger:", TRIGGER_WORD)
print("Preset:", TRAINING_PRESET, p)
print("Base checkpoint:", BASE_CHECKPOINT)
print("CLIP skip:", CLIP_SKIP)
print("Train text encoder:", TRAIN_TEXT_ENCODER)
print("Dry run:", DRY_RUN)
print("Drive mode:", "API fallback (local copy)" if USE_DRIVE_API else "FUSE mount")
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
    """# @title 4) Download base checkpoint (+ VAE if needed)
from huggingface_hub import hf_hub_download
import os
import shutil

def _download_weight(repo, name, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 100000000:
        print("Already on disk:", dest)
        print("Size GB:", round(os.path.getsize(dest) / 1024 / 1024 / 1024, 2))
        return
    print("Downloading", name, "...")
    downloaded = hf_hub_download(repo_id=repo, filename=name, local_dir=os.path.dirname(dest))
    if os.path.abspath(downloaded) != os.path.abspath(dest):
        shutil.copy2(downloaded, dest)
    print("Saved:", dest)
    print("Size GB:", round(os.path.getsize(dest) / 1024 / 1024 / 1024, 2))

_download_weight(BASE_MODEL_REPO, BASE_MODEL_NAME, BASE_MODEL_FILE)
if VAE_FILE:
    _download_weight(VAE_REPO, VAE_NAME, VAE_FILE)"""
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
    """# @title 6) Check captions + photo tags
import os

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
images = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(IMAGE_EXT)]
missing = []
PHOTO_TAGS = "photorealistic, raw photo, natural skin"
updated = 0
for name in images:
    stem = os.path.splitext(name)[0]
    txt = os.path.join(DATASET_DIR, stem + ".txt")
    if not os.path.exists(txt):
        missing.append(name)
        continue
    cap = open(txt, encoding="utf-8").read().strip()
    if "photorealistic" not in cap.lower():
        cap = cap + ", " + PHOTO_TAGS
        open(txt, "w", encoding="utf-8").write(cap)
        updated += 1

print("Images:", len(images))
print("Missing captions:", len(missing))
print("Captions tagged photorealistic:", updated)
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
        "--noise_offset=0.1",
    ]
    if CLIP_SKIP > 1:
        c.append("--clip_skip=" + str(CLIP_SKIP))
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
    print("Base checkpoint:", BASE_CHECKPOINT)
    print("CLIP skip:", CLIP_SKIP)
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
print("Size MB:", size_mb)
upload_project_file(final_path)"""
)

code(
    """# @title 9) Identity check: base model vs LoRA (same seed)
import gc
import os
import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionPipeline
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

    print("Loading base model...")
    vae = None
    if VAE_FILE and os.path.exists(VAE_FILE):
        print("Loading VAE:", VAE_FILE)
        vae = AutoencoderKL.from_single_file(VAE_FILE, torch_dtype=torch.float16)
    pipe_kw = {
        "torch_dtype": torch.float16,
        "safety_checker": None,
    }
    if vae is not None:
        pipe_kw["vae"] = vae
    pipe = StableDiffusionPipeline.from_single_file(BASE_MODEL_FILE, **pipe_kw).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    _disable_peft_torchao()

    prompt = (
        TRIGGER_WORD
        + ", portrait, close up face, looking at camera, photorealistic, "
        + "raw photo, natural skin texture, natural lighting, high quality"
    )
    negative = (
        "cgi, 3d render, cartoon, anime, painting, airbrushed, plastic skin, "
        "doll, deformed, extra fingers, blurry"
    )
    seed = 42
    gen_kw = {
        "prompt": prompt,
        "negative_prompt": negative,
        "num_inference_steps": 28,
        "guidance_scale": 6.0,
    }
    if CLIP_SKIP > 1:
        gen_kw["clip_skip"] = CLIP_SKIP
    print("Prompt:", prompt)
    print("Negative:", negative)
    print("CLIP skip:", CLIP_SKIP)
    print("Generating OFF (base model only)...")
    gen0 = torch.Generator(device="cuda").manual_seed(seed)
    image_off = pipe(generator=gen0, **gen_kw).images[0]
    off_path = os.path.join(OUTPUT_DIR, "preview_off.png")
    image_off.save(off_path)
    print("BASE MODEL (no LoRA) - generic face is expected. Identity comes from the LoRA:")
    display(image_off)

    print("Loading LoRA at weight 1.0 (no fuse)...")
    pipe.load_lora_weights(lora_file)
    gen1 = torch.Generator(device="cuda").manual_seed(seed)
    image_on = pipe(
        generator=gen1,
        cross_attention_kwargs={"scale": 1.0},
        **gen_kw,
    ).images[0]
    on_path = os.path.join(OUTPUT_DIR, "preview_on.png")
    image_on.save(on_path)
    print("WITH LORA weight 1.0 - should look like the training photos, more photographic:")
    display(image_on)
    print("Saved:", off_path)
    print("Saved:", on_path)
    upload_project_file(off_path)
    upload_project_file(on_path)
    print("If OFF and ON look the same, the LoRA did not apply.")
    print("If ON is closer but still plastic, add sharper face close-ups and retrain.")
    print("Ignore HF_TOKEN warning - public checkpoints do not need a token.")"""
)

md(
    """## Done

Thorough LoRA export (this run):
`MyDrive/FiratSuper/loras/lapetitemilf_thorough.safetensors`

Standard LoRA (kept, identity was close):
`MyDrive/FiratSuper/loras/lapetitemilf_standard.safetensors`

Quick LoRA (kept, identity was weak):
`MyDrive/FiratSuper/loras/lapetitemilf_lora.safetensors`

### Use in Automatic1111 / ComfyUI / Forge
1. Load **Realistic Vision V5.1** as the checkpoint (not vanilla SD 1.5)
2. Copy the Thorough `.safetensors` file to `models/Lora/`
3. Prompt: `ohwx woman, portrait, close up face, photorealistic, raw photo, ...`
4. Negative: `cgi, 3d render, cartoon, anime, airbrushed, plastic skin`
5. CLIP skip: 2. Character LoRA weight: start at `0.8-1.0`

### How to read cell 9
- OFF = Realistic Vision only (generic photoreal woman)
- ON = same prompt + LoRA
- You want ON to look like the training photos, not like OFF

### Next if identity is still weak
- Add 10-15 sharper face close-ups (the current set is small phone/social JPGs)
- Keep Thorough settings and rerun cell 7"""
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
