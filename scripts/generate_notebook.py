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
1. **Runtime > Change runtime type > GPU**. T4 is enough. A100 or L4 is faster (Colab Pro). Do not pick TPU.
2. Run cells **in order** (1 to 10)
3. Approve Google Drive access. In the popup click Continue, then **Allow ALL permissions** (do not uncheck boxes).
4. **Cell 7:** `DRY_RUN = False` for full training (dry run already passed)

## If cell 2 fails: credential propagation
This is a Colab Drive-login popup issue, not the training code.
1. Left sidebar: folder icon, then **Mount Drive**, then rerun cell 2
2. Chrome, one Google account only (the account that owns the photos)
3. In the popup: Continue, then **Allow ALL permissions** (do not uncheck boxes)
4. If FUSE still fails, cell 2 tries Google login, then copies the dataset without mounting Drive

## Current preset: face LoRA hits sometimes (2 of 5 on the hit prompt)
- Winning look: serious close-up, looking at camera (`2_front_neutral`).
- Identity can hit. It is not locked. Do not retrain on the same photos.
- Run **cell 9d** to generate 10 more portraits and keep the similar ones.
- Portraits: use `lapetitemilf_face` only. Body LoRA is not for faces.

## Drive layout
```
MyDrive/FiratSuper/
|-- ADD_BODY_PHOTOS/                       # drop 10-15 full-body swimsuit photos here
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
        "No GPU. Runtime > Change runtime type > GPU (T4, L4, or A100). Do not pick TPU, then rerun."
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
    inbox = api_find_child(service, FIRATSUPER_DRIVE_ID, "ADD_BODY_PHOTOS")
    inbox_dir = os.path.join(local_root, "ADD_BODY_PHOTOS")
    if inbox:
        api_download_folder(service, inbox["id"], inbox_dir)
    else:
        os.makedirs(inbox_dir, exist_ok=True)
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
        print("4. Runtime > Disconnect and delete runtime, reconnect GPU (T4, L4, or A100)")
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
RUN_NAME = "face"                 # lapetitemilf_face. Use "face2" only after NEW unique close-ups
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
# Separate output folder so a new run does not overwrite Thorough portraits
RUN_TAG = RUN_NAME if RUN_NAME else TRAINING_PRESET
OUTPUT_DIR = f"{ROOT}/output/{PROJECT_NAME}/{RUN_TAG}"
MODELS_DIR = f"{ROOT}/models"
LORAS_DIR = f"{ROOT}/loras"
LOGS_DIR = f"{ROOT}/logs/{PROJECT_NAME}/{TRAINING_PRESET}"
LORA_BASENAME = PROJECT_NAME + "_" + RUN_TAG

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
print("Base model file:", BASE_MODEL_FILE)
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")
n_body = 0
if os.path.isdir(DATASET_DIR):
    n_body = len(
        [
            f
            for f in os.listdir(DATASET_DIR)
            if f.startswith("body_") and f.lower().endswith(IMAGE_EXT)
        ]
    )
print("Full-body photos imported (body_*):", n_body)
n_face = 0
if os.path.isdir(DATASET_DIR):
    n_face = len(
        [
            f
            for f in os.listdir(DATASET_DIR)
            if f.startswith("face_") and f.lower().endswith(IMAGE_EXT)
        ]
    )
print("Face close-ups imported (face_*):", n_face)
print("This run writes:", LORA_BASENAME + ".safetensors")
if RUN_TAG == "face" and n_face < 8:
    print("Next: run cell 6b to import new close-ups from ADD_BODY_PHOTOS.")
    print("Inbox: https://drive.google.com/drive/folders/1YK-nUV4ihzqpDhxZICwM9YFngFbS34LP")
elif RUN_TAG == "body" and n_body < 10:
    print("Next: run cell 6b to copy full-body photos into the training folder.")
else:
    print("Imported photos look ready. Next after 6b: cell 7.")"""
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
    raise RuntimeError("CUDA is not available. Runtime -> Change runtime type -> GPU (not TPU)")

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
    """# @title 6b) Import inbox photos (skips ones already copied)
import os
import shutil

BODY_INBOX = os.path.join(ROOT, "ADD_BODY_PHOTOS")
os.makedirs(BODY_INBOX, exist_ok=True)
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")
incoming = [f for f in os.listdir(BODY_INBOX) if f.lower().endswith(IMAGE_EXT)]
skipped = [
    f
    for f in os.listdir(BODY_INBOX)
    if (not f.lower().endswith(IMAGE_EXT)) and (not f.lower().endswith(".txt"))
]
print("Inbox:", BODY_INBOX)
print("Photos in inbox:", len(incoming))
if skipped:
    print("Skipped (not images):", ", ".join(skipped))
print("Upload folder: https://drive.google.com/drive/folders/1YK-nUV4ihzqpDhxZICwM9YFngFbS34LP")
if len(incoming) == 0:
    print("Empty inbox.")
else:
    copied = 0
    skipped_exist = 0
    skipped_nocap = 0
    for name in sorted(incoming):
        src = os.path.join(BODY_INBOX, name)
        stem = os.path.splitext(name)[0].replace(" ", "_")
        ext = os.path.splitext(name)[1].lower()
        if ext == ".jpeg":
            ext = ".jpg"
        already = os.path.join(DATASET_DIR, "body_" + stem + ext)
        dst = os.path.join(DATASET_DIR, "face_" + stem + ext)
        if os.path.isfile(already) or os.path.isfile(dst):
            skipped_exist += 1
            continue
        inbox_txt = os.path.splitext(src)[0] + ".txt"
        if not os.path.isfile(inbox_txt):
            print(" skip (no caption, not used):", name)
            skipped_nocap += 1
            continue
        shutil.copy2(src, dst)
        text = open(inbox_txt, encoding="utf-8").read().strip()
        if not text.lower().startswith(TRIGGER_WORD.lower()):
            text = TRIGGER_WORD + ", " + text
        cap = os.path.splitext(dst)[0] + ".txt"
        open(cap, "w", encoding="utf-8").write(text)
        copied += 1
        print(" ", os.path.basename(dst))
        print("   ", text[:140])
    n_img = len([f for f in os.listdir(DATASET_DIR) if f.lower().endswith(IMAGE_EXT)])
    n_face = len(
        [
            f
            for f in os.listdir(DATASET_DIR)
            if f.startswith("face_") and f.lower().endswith(IMAGE_EXT)
        ]
    )
    print("Already in dataset, skipped:", skipped_exist)
    print("No caption, skipped:", skipped_nocap)
    print("Copied new face photos:", copied)
    print("Face photos in dataset:", n_face)
    print("Dataset images now:", n_img)
    if n_face < 8:
        print("Still under 8 face photos. Add more close-ups before cell 7.")
    else:
        print("Ready. Confirm RUN_NAME in cell 2, then run cell 7.")
        print("If lapetitemilf_face already exists, set RUN_NAME = 'face2'.")
        print("Do not overwrite lapetitemilf_face, body, or Thorough.")"""
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
    raise RuntimeError("No GPU - Runtime -> Change runtime type -> GPU (T4, L4, or A100)")

# Body identity already failed on the current 25 cropped photos. Do not retrain them.
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")
body_files = [
    f
    for f in os.listdir(DATASET_DIR)
    if f.startswith("body_") and f.lower().endswith(IMAGE_EXT)
]
face_files = [
    f
    for f in os.listdir(DATASET_DIR)
    if f.startswith("face_") and f.lower().endswith(IMAGE_EXT)
]
print("Imported full-body photos (body_*):", len(body_files))
print("Imported face photos (face_*):", len(face_files))
print("This run would write:", LORA_BASENAME + ".safetensors")
if not DRY_RUN:
    if RUN_TAG.startswith("face"):
        if len(face_files) < 8:
            raise RuntimeError(
                "Stopped. Need 8+ face close-ups. Run cell 6b after adding "
                "captioned photos to ADD_BODY_PHOTOS. Do not overwrite "
                "lapetitemilf_body.safetensors."
            )
        if RUN_TAG == "face":
            existing = os.path.join(LORAS_DIR, "lapetitemilf_face.safetensors")
            if os.path.isfile(existing):
                raise RuntimeError(
                    "Stopped. lapetitemilf_face already exists (2/5 similar). "
                    "Do not overwrite it. After NEW unique close-ups, set "
                    "RUN_NAME = 'face2'."
                )
    elif RUN_TAG == "body":
        if len(body_files) < 10:
            raise RuntimeError(
                "Stopped. Need 10+ body_* photos. Run cell 6b, then this cell."
            )
    else:
        raise RuntimeError(
            "Stopped. Set RUN_NAME to 'face', 'face2', or 'body' so Thorough "
            "and body files are not overwritten."
        )

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
        + ", long wavy highlighted blonde hair, brown eyes, adult woman, "
        + "portrait, close up face, looking at camera, photorealistic, "
        + "raw photo, natural skin texture, natural lighting, high quality"
    )
    negative = (
        "cgi, 3d render, cartoon, anime, painting, airbrushed, plastic skin, "
        "doll, deformed, extra fingers, blurry, black hair, child, teen, "
        "different person, extra people"
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
    print("This is ONE seed. Run cell 9b for 5 different face crops.")
    print("Ignore HF_TOKEN warning - public checkpoints do not need a token.")"""
)

code(
    """# @title 9b) Face close-up sheet (5 crops, no retraining)
import os
import torch
from IPython.display import display
from PIL import Image, ImageDraw

if "pipe" not in globals():
    raise RuntimeError("Run cell 9 first so the model is loaded.")

lora_file = os.path.join(LORAS_DIR, LORA_BASENAME + ".safetensors")
if not os.path.exists(lora_file):
    raise RuntimeError("LoRA not found: " + lora_file)

try:
    pipe.unload_lora_weights()
except Exception:
    pass
pipe.load_lora_weights(lora_file)
print("Face sheet LoRA:", os.path.basename(lora_file))
print("This cell does NOT retrain. 5 close-ups, different seeds.")

LOOK = "long wavy highlighted blonde hair, brown eyes, adult woman"
negative = (
    "cgi, 3d render, cartoon, anime, painting, airbrushed, plastic skin, "
    "doll, deformed, extra fingers, blurry, black hair, child, teen, "
    "different person, extra people, extra faces"
)
faces = [
    {
        "name": "1_front_smile",
        "extra": "portrait, close up face, looking at camera, slight smile, detailed face",
        "seed": 101,
    },
    {
        "name": "2_front_neutral",
        "extra": "portrait, close up face, looking at camera, serious, detailed face",
        "seed": 707,
    },
    {
        "name": "3_three_quarter",
        "extra": "portrait, close up face, three quarter view, looking at camera, detailed face",
        "seed": 2024,
    },
    {
        "name": "4_head_tilt",
        "extra": "portrait, close up face, looking at camera, head tilted, detailed face",
        "seed": 31415,
    },
    {
        "name": "5_wide_smile",
        "extra": "portrait, close up face, looking at camera, smiling, teeth, detailed face",
        "seed": 99991,
    },
]

images = []
paths = []
for face in faces:
    prompt = (
        TRIGGER_WORD
        + ", "
        + LOOK
        + ", "
        + face["extra"]
        + ", photorealistic, raw photo, natural skin texture, natural lighting, high quality"
    )
    gen_kw = {
        "prompt": prompt,
        "negative_prompt": negative,
        "num_inference_steps": 28,
        "guidance_scale": 6.0,
        "width": 512,
        "height": 512,
        "output_type": "pil",
    }
    if CLIP_SKIP > 1:
        gen_kw["clip_skip"] = CLIP_SKIP
    print("===", face["name"], "seed", face["seed"], "===")
    print("Prompt:", prompt)
    gen = torch.Generator(device="cpu").manual_seed(face["seed"])
    image = pipe(
        generator=gen,
        cross_attention_kwargs={"scale": 1.0},
        **gen_kw,
    ).images[0]
    path = os.path.join(OUTPUT_DIR, "preview_face_" + face["name"] + ".png")
    image.save(path)
    images.append(image)
    paths.append(path)
    print("Saved:", path)

thumb = 256
labeled = []
for face, image in zip(faces, images):
    im = image.copy()
    im.thumbnail((thumb, thumb - 28))
    canvas = Image.new("RGB", (thumb, thumb), (16, 16, 16))
    canvas.paste(im, ((thumb - im.width) // 2, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), face["name"], fill=(255, 255, 255))
    labeled.append(canvas)

sheet_w = thumb * 3
sheet_h = thumb * 2
sheet = Image.new("RGB", (sheet_w, sheet_h), (0, 0, 0))
for idx, tile in enumerate(labeled):
    x = (idx % 3) * thumb
    y = (idx // 3) * thumb
    sheet.paste(tile, (x, y))
sheet_path = os.path.join(OUTPUT_DIR, "preview_face_SHEET.png")
sheet.save(sheet_path)
print("CONTACT SHEET (all 5 faces):")
display(sheet)
print("Saved:", sheet_path)
upload_project_file(sheet_path)
for path in paths:
    upload_project_file(path)
print("Wrote", len(paths), "face files.")
print("Judge identity on this sheet, not on the body LoRA.")
print("If ONE crop is her and the others are not, run cell 9c with that same prompt.")"""
)

code(
    """# @title 9c) Repeat the hit face prompt (5 seeds, no retraining)
import os
import torch
from IPython.display import display
from PIL import Image, ImageDraw

if "pipe" not in globals():
    raise RuntimeError("Run cell 9 first so the model is loaded.")

lora_file = os.path.join(LORAS_DIR, LORA_BASENAME + ".safetensors")
if not os.path.exists(lora_file):
    raise RuntimeError("LoRA not found: " + lora_file)

try:
    pipe.unload_lora_weights()
except Exception:
    pass
pipe.load_lora_weights(lora_file)
print("This cell does NOT retrain.")
print("Same prompt as 2_front_neutral (the crop that looked similar).")

LOOK = "long wavy highlighted blonde hair, brown eyes, adult woman"
prompt = (
    TRIGGER_WORD
    + ", "
    + LOOK
    + ", portrait, close up face, looking at camera, serious, detailed face, "
    + "photorealistic, raw photo, natural skin texture, natural lighting, high quality"
)
negative = (
    "cgi, 3d render, cartoon, anime, painting, airbrushed, plastic skin, "
    "doll, deformed, extra fingers, blurry, black hair, child, teen, "
    "different person, extra people, extra faces"
)
seeds = [707, 1301, 4096, 7777, 24680]
print("Prompt:", prompt)
print("Seeds:", seeds)

images = []
paths = []
for seed in seeds:
    gen_kw = {
        "prompt": prompt,
        "negative_prompt": negative,
        "num_inference_steps": 28,
        "guidance_scale": 6.0,
        "width": 512,
        "height": 512,
        "output_type": "pil",
    }
    if CLIP_SKIP > 1:
        gen_kw["clip_skip"] = CLIP_SKIP
    print("=== seed", seed, "===")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        generator=gen,
        cross_attention_kwargs={"scale": 1.0},
        **gen_kw,
    ).images[0]
    path = os.path.join(OUTPUT_DIR, "preview_face_hit_" + str(seed) + ".png")
    image.save(path)
    images.append(image)
    paths.append(path)
    print("Saved:", path)

thumb = 256
labeled = []
for seed, image in zip(seeds, images):
    im = image.copy()
    im.thumbnail((thumb, thumb - 28))
    canvas = Image.new("RGB", (thumb, thumb), (16, 16, 16))
    canvas.paste(im, ((thumb - im.width) // 2, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), "seed " + str(seed), fill=(255, 255, 255))
    labeled.append(canvas)

sheet_w = thumb * 3
sheet_h = thumb * 2
sheet = Image.new("RGB", (sheet_w, sheet_h), (0, 0, 0))
for idx, tile in enumerate(labeled):
    x = (idx % 3) * thumb
    y = (idx // 3) * thumb
    sheet.paste(tile, (x, y))
sheet_path = os.path.join(OUTPUT_DIR, "preview_face_HIT_SHEET.png")
sheet.save(sheet_path)
print("CONTACT SHEET (same prompt, 5 seeds):")
display(sheet)
print("Saved:", sheet_path)
upload_project_file(sheet_path)
for path in paths:
    upload_project_file(path)
print("If 2 of 5 look similar, identity can hit but is not locked.")
print("Do not retrain on the same photos. Next: cell 9d (10 more portraits).")"""
)

code(
    """# @title 9d) Keeper hunt (10 portraits, same hit prompt, no retraining)
import os
import torch
from IPython.display import display
from PIL import Image, ImageDraw

if "pipe" not in globals():
    raise RuntimeError("Run cell 9 first so the model is loaded.")

lora_file = os.path.join(LORAS_DIR, LORA_BASENAME + ".safetensors")
if not os.path.exists(lora_file):
    raise RuntimeError("LoRA not found: " + lora_file)

try:
    pipe.unload_lora_weights()
except Exception:
    pass
pipe.load_lora_weights(lora_file)
print("This cell does NOT retrain.")
print("Same prompt as cell 9c. New seeds. Keep the similar files.")

LOOK = "long wavy highlighted blonde hair, brown eyes, adult woman"
prompt = (
    TRIGGER_WORD
    + ", "
    + LOOK
    + ", portrait, close up face, looking at camera, serious, detailed face, "
    + "photorealistic, raw photo, natural skin texture, natural lighting, high quality"
)
negative = (
    "cgi, 3d render, cartoon, anime, painting, airbrushed, plastic skin, "
    "doll, deformed, extra fingers, blurry, black hair, child, teen, "
    "different person, extra people, extra faces"
)
# 707 already hit. Other seeds are new (not the cell 9c set).
seeds = [707, 42, 314, 2025, 8192, 12345, 33333, 44444, 55555, 88888]
print("Prompt:", prompt)
print("Seeds:", seeds)
print("About 10 images. T4: a few minutes.")

images = []
paths = []
for seed in seeds:
    gen_kw = {
        "prompt": prompt,
        "negative_prompt": negative,
        "num_inference_steps": 28,
        "guidance_scale": 6.0,
        "width": 512,
        "height": 512,
        "output_type": "pil",
    }
    if CLIP_SKIP > 1:
        gen_kw["clip_skip"] = CLIP_SKIP
    print("=== seed", seed, "===")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        generator=gen,
        cross_attention_kwargs={"scale": 1.0},
        **gen_kw,
    ).images[0]
    path = os.path.join(OUTPUT_DIR, "preview_face_keeper_" + str(seed) + ".png")
    image.save(path)
    images.append(image)
    paths.append(path)
    print("Saved:", path)

thumb = 220
labeled = []
for seed, image in zip(seeds, images):
    im = image.copy()
    im.thumbnail((thumb, thumb - 28))
    canvas = Image.new("RGB", (thumb, thumb), (16, 16, 16))
    canvas.paste(im, ((thumb - im.width) // 2, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), "seed " + str(seed), fill=(255, 255, 255))
    labeled.append(canvas)

sheet_w = thumb * 5
sheet_h = thumb * 2
sheet = Image.new("RGB", (sheet_w, sheet_h), (0, 0, 0))
for idx, tile in enumerate(labeled):
    x = (idx % 5) * thumb
    y = (idx // 5) * thumb
    sheet.paste(tile, (x, y))
sheet_path = os.path.join(OUTPUT_DIR, "preview_face_KEEPER_SHEET.png")
sheet.save(sheet_path)
print("KEEPER SHEET (same prompt, 10 seeds):")
display(sheet)
print("Saved:", sheet_path)
upload_project_file(sheet_path)
for path in paths:
    upload_project_file(path)
print("Keep the similar files. Discard the generic ones.")
print("This is the current ceiling of this dataset: identity hits, not locked.")
print("To lock it: unique sharp NEUTRAL close-ups looking at camera, then face2.")"""
)

code(
    """# @title 10) Swimsuit pose sheet (run after cell 9, no retraining)
import os
import torch
from IPython.display import display
from PIL import Image, ImageDraw

if "pipe" not in globals():
    raise RuntimeError("Run cell 9 first so the model is loaded.")

lora_file = os.path.join(LORAS_DIR, LORA_BASENAME + ".safetensors")
if not os.path.exists(lora_file):
    raise RuntimeError("LoRA not found: " + lora_file)

try:
    pipe.unload_lora_weights()
except Exception:
    pass
pipe.load_lora_weights(lora_file)

print("This cell does NOT retrain. It only generates with a tighter identity prompt.")
LOOK = "long wavy highlighted blonde hair, brown eyes, adult woman"
negative = (
    "cgi, 3d render, cartoon, anime, painting, airbrushed, plastic skin, "
    "doll, deformed, extra fingers, extra legs, extra people, cropped head, "
    "black hair, child, teen, different person, extra faces"
)
# Far-apart seeds + different poses so the sheet cannot collapse to one frame.
poses = [
    {
        "name": "1_waist_up",
        "extra": "waist up, swimsuit, looking at camera, hands on hips, detailed face",
        "seed": 101,
        "width": 512,
        "height": 640,
    },
    {
        "name": "2_stand_front",
        "extra": "standing, full body, front view, arms at sides, swimsuit, looking at camera",
        "seed": 707,
        "width": 512,
        "height": 768,
    },
    {
        "name": "3_stand_side",
        "extra": "standing, full body, side view, swimsuit, looking over shoulder",
        "seed": 2024,
        "width": 512,
        "height": 768,
    },
    {
        "name": "4_sitting",
        "extra": "sitting on a chair, full body, swimsuit, legs visible, looking at camera",
        "seed": 31415,
        "width": 512,
        "height": 768,
    },
    {
        "name": "5_walking",
        "extra": "walking toward camera, full body, swimsuit, one foot forward",
        "seed": 99991,
        "width": 512,
        "height": 768,
    },
]

print("This cell must create 5 DIFFERENT images plus one contact sheet.")
print("Refresh the GitHub notebook if this cell still has the old generic prompt.")

images = []
paths = []
for pose in poses:
    prompt = (
        TRIGGER_WORD
        + ", "
        + LOOK
        + ", "
        + pose["extra"]
        + ", photorealistic, raw photo, natural lighting, high quality"
    )
    gen_kw = {
        "prompt": prompt,
        "negative_prompt": negative,
        "num_inference_steps": 28,
        "guidance_scale": 6.0,
        "width": pose["width"],
        "height": pose["height"],
        "output_type": "pil",
    }
    if CLIP_SKIP > 1:
        gen_kw["clip_skip"] = CLIP_SKIP
    print("===", pose["name"], "seed", pose["seed"], "===")
    print("Prompt:", prompt)
    gen = torch.Generator(device="cpu").manual_seed(pose["seed"])
    image = pipe(
        generator=gen,
        cross_attention_kwargs={"scale": 1.0},
        **gen_kw,
    ).images[0]
    path = os.path.join(OUTPUT_DIR, "preview_swim_" + pose["name"] + "_id.png")
    image.save(path)
    images.append(image)
    paths.append(path)
    print("Saved:", path)

# Fail loudly if two frames are almost the same pixels.
for i in range(len(images)):
    for j in range(i + 1, len(images)):
        a = images[i].resize((128, 128)).convert("RGB")
        b = images[j].resize((128, 128)).convert("RGB")
        pa = list(a.getdata())
        pb = list(b.getdata())
        diff = 0
        for k in range(len(pa)):
            diff += abs(pa[k][0] - pb[k][0]) + abs(pa[k][1] - pb[k][1]) + abs(pa[k][2] - pb[k][2])
        mean = diff / float(len(pa) * 3)
        if mean < 4.0:
            print("WARNING: ", poses[i]["name"], "looks almost identical to", poses[j]["name"])

# One contact sheet so Colab cannot hide the extra images.
thumb_w = 256
thumb_h = 384
labeled = []
for pose, image in zip(poses, images):
    im = image.copy()
    im.thumbnail((thumb_w, thumb_h - 28))
    canvas = Image.new("RGB", (thumb_w, thumb_h), (16, 16, 16))
    canvas.paste(im, ((thumb_w - im.width) // 2, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), pose["name"], fill=(255, 255, 255))
    labeled.append(canvas)

sheet_w = thumb_w * 3
sheet_h = thumb_h * 2
sheet = Image.new("RGB", (sheet_w, sheet_h), (0, 0, 0))
for idx, tile in enumerate(labeled):
    x = (idx % 3) * thumb_w
    y = (idx // 3) * thumb_h
    sheet.paste(tile, (x, y))
sheet_path = os.path.join(OUTPUT_DIR, "preview_swim_SHEET_id.png")
sheet.save(sheet_path)
print("CONTACT SHEET (all 5 poses):")
display(sheet)
print("Saved:", sheet_path)
upload_project_file(sheet_path)
for path in paths:
    upload_project_file(path)
print("Wrote", len(paths), "pose files (identity prompt, no retrain).")
print("Judge the FACE on 1_waist_up and on cell 9. Judge the BODY on 2-5.")
print("Full-body faces stay soft on SD 1.5. That is not a LoRA failure.")
print("If waist_up is still not her, we need sharper face close-ups, not more epochs.")"""
)

md(
    """## Done

Thorough LoRA export (portraits, keep this file):
`MyDrive/FiratSuper/loras/lapetitemilf_thorough.safetensors`

Body LoRA export (keep this file):
`MyDrive/FiratSuper/loras/lapetitemilf_body.safetensors`

Face LoRA export (this run, after cell 6b + cell 7):
`MyDrive/FiratSuper/loras/lapetitemilf_face.safetensors`

Standard LoRA (kept, identity was close):
`MyDrive/FiratSuper/loras/lapetitemilf_standard.safetensors`

Quick LoRA (kept, identity was weak):
`MyDrive/FiratSuper/loras/lapetitemilf_lora.safetensors`

### Use in Automatic1111 / ComfyUI / Forge
1. Load **Realistic Vision V5.1** as the checkpoint (not vanilla SD 1.5)
2. Copy `lapetitemilf_face.safetensors` to `models/Lora/` after this run (body file still exists)
3. Always keep these identity words after the trigger:
   `ohwx woman, long wavy highlighted blonde hair, brown eyes, adult woman`
4. Face (winning prompt): `..., portrait, close up face, looking at camera, serious, detailed face, photorealistic, raw photo, natural skin texture`
5. Waist-up: `..., waist up, swimsuit, looking at camera, photorealistic, raw photo`
6. Full body poses (512x768). Change only the pose words, keep the identity words:
   - `..., standing, full body, front view, swimsuit, photorealistic, raw photo`
   - `..., sitting, full body, swimsuit, photorealistic, raw photo`
   - `..., walking, full body, swimsuit, photorealistic, raw photo`
7. Negative: `cgi, 3d render, cartoon, anime, airbrushed, plastic skin, extra people, black hair, child, teen, different person`
8. CLIP skip: 2. LoRA weight `0.8-1.0`. Full-body face: enable After Detailer if you have it.

### How to read cell 9, 9b, 9c, 9d, and cell 10
- Cell 9 = one seed, OFF vs ON
- Cell 9b = 5 face close-ups. User: 2_front_neutral was similar, others so-so
- Cell 9c = same prompt, 5 seeds. User: **2 of 5 similar**
- Cell 9d = same prompt, 10 seeds. Keep the similar files
- Cell 10 = swimsuit body. Use face LoRA for portraits, not body

### Next
- Reopen the GitHub notebook, run cell 9 then **cell 9d** (no cell 7)
- Keep lapetitemilf_face. Do not overwrite it
- For a lock: unique sharp NEUTRAL close-ups looking at camera, then RUN_NAME = face2"""
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
