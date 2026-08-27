#!/usr/bin/env python3
"""Generate the FiratSuper Flux LoRA Colab notebook (ASCII-only JSON)."""
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
    """# FiratSuper Flux LoRA (Colab A100)

Train a **Flux.1 [dev] character LoRA** with Ostris ai-toolkit.

**Runtime:** Runtime > Change runtime type > **A100 GPU**. Do not pick T4. Do not pick TPU.
High RAM can stay off.

**Drive:** Chrome, one Google account only (`superweb.contact@gmail.com`). In the popup: Continue, then **Allow ALL** permissions.

**Hugging Face:** Accept the license for [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and paste a READ token.

**Output file only:** `MyDrive/FiratSuper/loras/lapetitemilf_flux.safetensors`
Do **not** overwrite `lapetitemilf_face` or any old SD 1.5 LoRA.

**Dataset:** `MyDrive/FiratSuper/ADD_FLUX_PHOTOS/` (31 images + 31 captions, Gate 1 GO).
Trigger: `ohwx woman`.

**NSFW:** this Colab has no platform safety checker. Generate after training in the last cell. Adult subject only. Do not add porn to the dataset.

**Cell 4:** Colab is Python 3.13. Ostris still pins scipy 1.12, which cannot install there. This notebook patches that. Cell 4 takes several minutes. Do not stop it.

## Cells
1. A100 GPU check
2. Drive + settings
3. Hugging Face login
4. Install ai-toolkit
5. Gate 1: copy photos + captions
6. Gate 2: write YAML
7. Gate 4: dry run (5 steps)
8. Full train (~2000 steps)
9. Copy LoRA to Drive and SHOW training samples
10. Generate and SHOW pictures (identity + lingerie + nude, no filter)

## Drive layout
```
MyDrive/FiratSuper/
|-- ADD_FLUX_PHOTOS/                 # 31 keepers + .txt captions
|-- loras/lapetitemilf_flux.safetensors   # NEW file this notebook writes
|-- loras/lapetitemilf_face.safetensors   # protected, do not touch
|-- output/lapetitemilf/flux_eval/        # generations from cell 10
`-- keepers/
```"""
)

code(
    """# @title 1) A100 GPU check
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU. Runtime > Change runtime type > A100 GPU. Do not pick TPU, then rerun."
    )

gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print("GPU:", gpu)
print("VRAM: %.1f GB" % vram)

name = gpu.upper()
if "TPU" in name:
    raise RuntimeError("TPU is not supported. Runtime > Change runtime type > A100 GPU.")
if ("A100" not in name) and ("H100" not in name):
    raise RuntimeError(
        "This notebook needs Colab Pro A100 (or H100). Got: %s (%.1f GB). "
        "Runtime > Change runtime type > A100 GPU. T4 and L4 are too small."
        % (gpu, vram)
    )
if vram < 35:
    raise RuntimeError(
        "A100-class GPU but VRAM is %.1f GB. Need ~40 GB. Change runtime type."
        % vram
    )
print("A100 check OK.")"""
)

code(
    """# @title 2) Connect Drive + project settings
from google.colab import auth, drive
import os
import shutil

MOUNT = "/content/drive"
MYDRIVE = os.path.join(MOUNT, "MyDrive")
FIRATSUPER_DRIVE_ID = "18UE4fijDjq8ggmkDYjaUpRQXVDE0cqYt"
FLUX_INBOX_ID = "1oLtTmwg2kt-Jn6zuci06ipRQoK6AOFVZ"
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
    if name in PROTECTED_LORAS:
        raise RuntimeError("Refusing to overwrite protected LoRA: " + name)
    media = MediaFileUpload(local_path, resumable=True)
    found = api_find_child(service, parent_id, name)
    if found:
        if found["name"] in PROTECTED_LORAS:
            raise RuntimeError("Refusing to overwrite protected LoRA: " + name)
        service.files().update(
            fileId=found["id"], media_body=media, supportsAllDrives=True
        ).execute()
        return found["id"]
    body = {"name": name, "parents": [parent_id]}
    created = service.files().create(
        body=body, media_body=media, fields="id", supportsAllDrives=True
    ).execute()
    return created["id"]


def upload_project_file(local_path, dest_rel=None):
    if dest_rel is None:
        dest_rel = os.path.relpath(local_path, ROOT)
    if USE_DRIVE_API:
        if DRIVE_SERVICE is None:
            print("Skip Drive upload: no API client")
            return
        print("Uploading to Drive:", dest_rel)
        parts = dest_rel.split("/")
        parent = FIRATSUPER_DRIVE_ID
        for folder in parts[:-1]:
            parent = api_ensure_folder(DRIVE_SERVICE, parent, folder)
        api_upload_file(DRIVE_SERVICE, local_path, parent, parts[-1])
        print("Uploaded:", dest_rel)
        return
    dest = os.path.join(ROOT, dest_rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.abspath(local_path) != os.path.abspath(dest):
        shutil.copy2(local_path, dest)
    print("Saved:", dest)


def sync_flux_via_api(local_root):
    print("FUSE mount failed. Copying ADD_FLUX_PHOTOS via Drive API to", local_root)
    service = _api_service()
    os.makedirs(local_root, exist_ok=True)
    inbox = api_find_child(service, FIRATSUPER_DRIVE_ID, "ADD_FLUX_PHOTOS")
    inbox_id = inbox["id"] if inbox else FLUX_INBOX_ID
    api_download_folder(
        service,
        inbox_id,
        os.path.join(local_root, "ADD_FLUX_PHOTOS"),
    )
    for sub in ("output", "loras", "logs", "keepers"):
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
        DRIVE_SERVICE = sync_flux_via_api("/content/FiratSuper")
        print("DRIVE MODE: API fallback (local /content/FiratSuper)")
        print("LoRA and previews will upload back to Drive after training.")
    else:
        print("Could not connect to Drive.")
        print("1. Left sidebar: folder icon -> Mount Drive, then rerun this cell")
        print("2. Chrome, one Google account only (the account that owns the photos)")
        print("3. Allow ALL permissions. Do not close extra popups")
        print("4. Runtime > Disconnect and delete runtime, reconnect A100 GPU")
        raise RuntimeError("Drive is not connected. See the steps printed above.")

if (not USE_DRIVE_API) and (not _drive_ok()):
    raise RuntimeError("Drive folder is empty. Grant access and rerun this cell.")

# === edit here ===
PROJECT_NAME = "lapetitemilf"
TRIGGER_WORD = "ohwx woman"
LORA_NAME = "lapetitemilf_flux"
EXPECTED_PAIRS = 31
TRAIN_STEPS = 2000
DRY_RUN_STEPS = 5
NETWORK_DIM = 16
NETWORK_ALPHA = 16
LEARNING_RATE = 1e-4
SAVE_EVERY = 500
SAMPLE_EVERY = 500
SUBJECT_IS_ADULT = True
# =================

PROTECTED_LORAS = {
    "lapetitemilf_face.safetensors",
    "lapetitemilf_body.safetensors",
    "lapetitemilf_thorough.safetensors",
    "lapetitemilf_standard.safetensors",
    "lapetitemilf_lora.safetensors",
    "lapetitemilf_together.safetensors",
}
OUTPUT_LORA_NAME = LORA_NAME + ".safetensors"
if OUTPUT_LORA_NAME in PROTECTED_LORAS:
    raise RuntimeError("LORA_NAME is protected. Use lapetitemilf_flux only.")
if not SUBJECT_IS_ADULT:
    raise RuntimeError("This notebook is for an adult subject only.")

if USE_DRIVE_API:
    ROOT = "/content/FiratSuper"
else:
    ROOT = "/content/drive/MyDrive/FiratSuper"

INBOX_DIR = os.path.join(ROOT, "ADD_FLUX_PHOTOS")
LORAS_DIR = os.path.join(ROOT, "loras")
KEEPERS_DIR = os.path.join(ROOT, "keepers")
EVAL_DIR = os.path.join(ROOT, "output", PROJECT_NAME, "flux_eval")
SAMPLES_DIR = os.path.join(ROOT, "output", PROJECT_NAME, "flux_samples")
DATASET_DIR = "/content/dataset"
TRAIN_OUTPUT_DIR = "/content/output"
CONFIG_PATH = "/content/lapetitemilf_flux.yaml"
os.makedirs(LORAS_DIR, exist_ok=True)
os.makedirs(KEEPERS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)

free_gb = shutil.disk_usage("/content").free / 1024**3
print("ROOT:", ROOT)
print("Inbox:", INBOX_DIR)
print("Local dataset:", DATASET_DIR)
print("LoRA out:", os.path.join(LORAS_DIR, OUTPUT_LORA_NAME))
print("Free disk: %.1f GB" % free_gb)
if free_gb < 40:
    raise RuntimeError("Need ~40 GB free for Flux.1-dev. Have %.1f GB." % free_gb)
print("Drive settings OK.")"""
)

code(
    """# @title 3) Hugging Face login (FLUX.1-dev is gated)
import getpass
import os

print("1. Open https://huggingface.co/black-forest-labs/FLUX.1-dev")
print("2. Accept the license while logged in")
print("3. Create a READ token: https://huggingface.co/settings/tokens")
print("4. Paste it below. Colab will hide it.")

token = None
try:
    from google.colab import userdata
    token = userdata.get("HF_TOKEN")
    if token:
        print("Using HF_TOKEN from Colab Secrets.")
except Exception:
    token = None

if not token:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token:
    token = getpass.getpass("HF READ token: ").strip()
if not token:
    raise RuntimeError("No Hugging Face token. Paste a READ token and rerun.")

os.environ["HF_TOKEN"] = token
os.environ["HUGGING_FACE_HUB_TOKEN"] = token
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import HfApi, login
login(token=token, add_to_git_credential=False)
api = HfApi(token=token)
try:
    api.model_info("black-forest-labs/FLUX.1-dev")
except Exception as err:
    raise RuntimeError(
        "Cannot read black-forest-labs/FLUX.1-dev. "
        "Accept the license on the model page, use a READ token from the same account. "
        "Detail: " + str(err)
    )
print("Hugging Face login OK. FLUX.1-dev is readable.")"""
)

code(
    r"""# @title 4) Install Ostris ai-toolkit (Gate 3)
import os
import sys
import subprocess
import shutil

INSTALL_MARK = "/content/ai-toolkit/.firat_install_ok"
INSTALL_VER = "2"
ROOT_TK = "/content/ai-toolkit"
SKIP_PKGS = ("torchcodec", "av==", "librosa==", "mutagen==", "gradio")


def run(cmd, cwd=None):
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    ret = subprocess.call(cmd, cwd=cwd, env=env)
    if ret != 0:
        raise RuntimeError("command failed (%d): %s" % (ret, " ".join(cmd)))


def pip_req(path):
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--prefer-binary",
            "--only-binary=scipy,numpy",
            "-r",
            path,
        ],
        cwd=ROOT_TK,
    )


def write_slim(src, dest):
    scipy_line = "scipy>=1.14.1\n" if sys.version_info >= (3, 13) else "scipy>=1.12.0\n"
    out = [scipy_line]
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                out.append(line if line.endswith("\n") else line + "\n")
                continue
            if any(stripped.startswith(s) for s in SKIP_PKGS):
                print("skip", stripped)
                continue
            out.append(line if line.endswith("\n") else line + "\n")
    with open(dest, "w", encoding="utf-8") as fh:
        fh.writelines(out)
    print("Wrote", dest)


print("Python", sys.version.replace("\n", " "))
print("This cell can take several minutes. Let it finish.")

if not os.path.isdir(os.path.join(ROOT_TK, ".git")):
    if os.path.isdir(ROOT_TK):
        shutil.rmtree(ROOT_TK)
    run(["git", "clone", "https://github.com/ostris/ai-toolkit", ROOT_TK])
run(["git", "submodule", "update", "--init", "--recursive"], cwd=ROOT_TK)

# Ostris pins scipy==1.12.0. No Python 3.13 wheel, pip tries to compile and dies.
req_overlay = os.path.join(ROOT_TK, "requirements_colab.txt")
if sys.version_info >= (3, 13):
    with open(req_overlay, "w", encoding="ascii") as fh:
        fh.write("-r requirements_base.txt\n")
        fh.write("scipy>=1.14.1\n")
    print("Python 3.13+: using scipy>=1.14.1 instead of scipy==1.12.0")
else:
    req_overlay = os.path.join(ROOT_TK, "requirements.txt")

already = False
if os.path.isfile(INSTALL_MARK):
    already = open(INSTALL_MARK, encoding="ascii").read().strip() == INSTALL_VER

if already:
    print("ai-toolkit packages already installed (mark %s). Skipping pip." % INSTALL_VER)
else:
    run([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel", "hf_transfer"])
    print("Installing Ostris requirements. If it fails, the red pip error is above this traceback.")
    try:
        pip_req(req_overlay)
    except RuntimeError:
        print("Full requirements failed. Retry without video extras (not needed for image LoRA).")
        slim = os.path.join(ROOT_TK, "requirements_colab_slim.txt")
        write_slim(os.path.join(ROOT_TK, "requirements_base.txt"), slim)
        pip_req(slim)
    with open(INSTALL_MARK, "w", encoding="ascii") as fh:
        fh.write(INSTALL_VER)
    print("Wrote", INSTALL_MARK)

sys.path.insert(0, ROOT_TK)
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("ai-toolkit clone:", ROOT_TK)
print("If Colab asks to Restart Runtime, restart, then rerun cells 1-4.")
print("Gate 3 install OK.")"""
)

code(
    r"""# @title 5) Gate 1: copy ADD_FLUX_PHOTOS (images + captions)
import os
import shutil

IMG_EXT = {".jpg", ".jpeg", ".png"}
SKIP_NAMES = {".drive_upload.json", ".ds_store", "thumbs.db"}

if USE_DRIVE_API and (not os.path.isdir(INBOX_DIR) or not os.listdir(INBOX_DIR)):
    print("Re-copy inbox via Drive API...")
    DRIVE_SERVICE = sync_flux_via_api(ROOT)

if not os.path.isdir(INBOX_DIR):
    raise RuntimeError("Missing inbox folder: " + INBOX_DIR)

if os.path.isdir(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)
os.makedirs(DATASET_DIR, exist_ok=True)

pairs = []
missing_txt = []
skipped = []
for name in sorted(os.listdir(INBOX_DIR)):
    if name.startswith("."):
        continue
    if name.lower() in SKIP_NAMES:
        skipped.append(name)
        continue
    path = os.path.join(INBOX_DIR, name)
    if os.path.isdir(path):
        skipped.append(name + "/")
        continue
    stem, ext = os.path.splitext(name)
    if ext.lower() not in IMG_EXT:
        if ext.lower() != ".txt":
            skipped.append(name)
        continue
    txt_name = stem + ".txt"
    txt_path = os.path.join(INBOX_DIR, txt_name)
    if not os.path.isfile(txt_path):
        missing_txt.append(name)
        continue
    shutil.copy2(path, os.path.join(DATASET_DIR, name))
    shutil.copy2(txt_path, os.path.join(DATASET_DIR, txt_name))
    with open(txt_path, "r", encoding="utf-8") as fh:
        caption = fh.read().strip()
    pairs.append((name, caption))

print("Copied pairs:", len(pairs))
if skipped:
    print("Skipped:", ", ".join(skipped[:20]))
if missing_txt:
    print("Images with no .txt:")
    for name in missing_txt:
        print("  ", name)
    raise RuntimeError("Gate 1 FAIL: %d images have no matching .txt" % len(missing_txt))

if len(pairs) < 20:
    raise RuntimeError("Gate 1 FAIL: need at least 20 image+txt pairs. Got %d" % len(pairs))
if len(pairs) != EXPECTED_PAIRS:
    print("NOTE: expected %d pairs, found %d. Continuing if captions look OK." % (EXPECTED_PAIRS, len(pairs)))

bad_trigger = [name for name, cap in pairs if not cap.lower().startswith("ohwx woman")]
if bad_trigger:
    print("Captions missing trigger at start:")
    for name in bad_trigger[:10]:
        print("  ", name)
    raise RuntimeError("Gate 1 FAIL: every caption must start with: ohwx woman")

print("--- sample caption ---")
print(pairs[0][0])
print(pairs[0][1][:400])
print("----------------------")
print("Gate 1 GO. Local train folder:", DATASET_DIR)"""
)

code(
    r"""# @title 6) Gate 2: write Ostris Flux YAML (A100 / 24GB recipe)
import os

def write_flux_yaml(path, steps, dry):
    sample_flag = "true" if dry else "false"
    skip_first = "true"
    save_every = 10000 if dry else SAVE_EVERY
    sample_every = 10000 if dry else SAMPLE_EVERY
    lines = [
        "job: extension",
        "config:",
        '  name: "%s"' % LORA_NAME,
        "  process:",
        "    - type: sd_trainer",
        '      training_folder: "%s"' % TRAIN_OUTPUT_DIR,
        "      device: cuda:0",
        '      trigger_word: "%s"' % TRIGGER_WORD,
        "      network:",
        "        type: lora",
        "        linear: %d" % NETWORK_DIM,
        "        linear_alpha: %d" % NETWORK_ALPHA,
        "      save:",
        "        dtype: float16",
        "        save_every: %d" % save_every,
        "        max_step_saves_to_keep: 4",
        "        push_to_hub: false",
        "      datasets:",
        '        - folder_path: "%s"' % DATASET_DIR,
        "          caption_ext: txt",
        "          caption_dropout_rate: 0.05",
        "          shuffle_tokens: false",
        "          cache_latents_to_disk: true",
        "          resolution: [512, 768, 1024]",
        "      train:",
        "        batch_size: 1",
        "        steps: %d" % steps,
        "        gradient_accumulation_steps: 1",
        "        train_unet: true",
        "        train_text_encoder: false",
        "        gradient_checkpointing: true",
        "        noise_scheduler: flowmatch",
        "        optimizer: adamw8bit",
        "        lr: %.0e" % LEARNING_RATE,
        "        skip_first_sample: %s" % skip_first,
        "        disable_sampling: %s" % sample_flag,
        "        ema_config:",
        "          use_ema: true",
        "          ema_decay: 0.99",
        "        dtype: bf16",
        "      model:",
        '        name_or_path: "black-forest-labs/FLUX.1-dev"',
        "        is_flux: true",
        "        quantize: true",
        "      sample:",
        "        sampler: flowmatch",
        "        sample_every: %d" % sample_every,
        "        sample_start_step: 0",
        "        width: 1024",
        "        height: 1024",
        "        prompts:",
        '          - "ohwx woman, close-up portrait of an adult woman with long highlighted blonde hair and brown eyes, looking at the camera, photorealistic raw photo, natural skin texture"',
        '          - "ohwx woman, an adult woman with long highlighted blonde hair and brown eyes, waist-up, looking at the camera, photorealistic fashion photo, natural skin texture"',
        '          - "ohwx woman, an adult woman with long highlighted blonde hair and brown eyes, full body standing, looking at the camera, photorealistic raw photo, natural skin texture"',
        '          - "ohwx woman, an adult woman with long highlighted blonde hair sitting indoors, looking at the camera, photorealistic raw photo, natural skin texture"',
        '        neg: ""',
        "        seed: 42",
        "        walk_seed: true",
        "        guidance_scale: 4",
        "        sample_steps: 20",
        "meta:",
        '  name: "[name]"',
        '  version: "1.0"',
        "",
    ]
    text = "\n".join(lines)
    if any(ord(ch) > 127 for ch in text):
        raise RuntimeError("YAML is not ASCII")
    with open(path, "w", encoding="ascii") as fh:
        fh.write(text)
    print("Wrote", path, "steps=%d dry=%s" % (steps, dry))


write_flux_yaml(CONFIG_PATH, TRAIN_STEPS, dry=False)
print("--- YAML head ---")
with open(CONFIG_PATH, "r", encoding="ascii") as fh:
    print("".join(fh.readlines()[:40]))
print("Gate 2 YAML OK. Rank", NETWORK_DIM, "steps", TRAIN_STEPS, "bf16 + 8bit quantize.")"""
)

code(
    r"""# @title 7) Gate 4: dry run (5 steps, no samples)
import os
import subprocess
import sys

write_flux_yaml(CONFIG_PATH, DRY_RUN_STEPS, dry=True)
print("Dry run: %d steps. This downloads FLUX.1-dev the first time (~24 GB)." % DRY_RUN_STEPS)
cmd = [sys.executable, "run.py", CONFIG_PATH]
print("+", " ".join(cmd))
subprocess.check_call(cmd, cwd="/content/ai-toolkit")
write_flux_yaml(CONFIG_PATH, TRAIN_STEPS, dry=False)
print("Gate 4 dry run OK. YAML restored to", TRAIN_STEPS, "steps.")
print("If that worked, run cell 8 for the full train. Do not close this tab.")"""
)

code(
    r"""# @title 8) Full Flux train (keep this tab open)
import os
import subprocess
import sys
import torch

write_flux_yaml(CONFIG_PATH, TRAIN_STEPS, dry=False)
print("Full train:", TRAIN_STEPS, "steps on", torch.cuda.get_device_name(0))
print("Keep this tab open. Colab will drop the GPU if the tab sleeps too long.")
cmd = [sys.executable, "run.py", CONFIG_PATH]
print("+", " ".join(cmd))
subprocess.check_call(cmd, cwd="/content/ai-toolkit")
print("Training process finished.")
print("This cell does not show pictures. Run cell 9 to copy the LoRA AND display samples.")
print("Then run cell 10 to generate identity / lingerie / nude and see them here.")"""
)

code(
    r"""# @title 9) Copy LoRA + SHOW training samples
import os
import glob
import shutil
from IPython.display import display, Image as IPyImage

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def iter_images(root):
    if not os.path.isdir(root):
        return
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in IMG_EXTS:
                yield os.path.join(dirpath, name)


def show_images(paths, limit=12):
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print("No pictures to show.")
        return
    show = paths[-limit:]
    print("Showing", len(show), "of", len(paths), "pictures")
    for path in show:
        print(path)
        display(IPyImage(filename=path, width=384))


if OUTPUT_LORA_NAME in PROTECTED_LORAS:
    raise RuntimeError("Refusing to write a protected LoRA name.")

run_dir = os.path.join(TRAIN_OUTPUT_DIR, LORA_NAME)
candidates = []
final_path = os.path.join(run_dir, OUTPUT_LORA_NAME)
if os.path.isfile(final_path):
    candidates.append(final_path)
candidates.extend(sorted(glob.glob(os.path.join(run_dir, "*.safetensors"))))
preferred = [p for p in candidates if os.path.basename(p) == OUTPUT_LORA_NAME]
if not preferred:
    preferred = [p for p in candidates if "step" not in os.path.basename(p).lower()]
if not preferred:
    preferred = candidates
if not preferred:
    raise RuntimeError("No .safetensors found in " + run_dir)

src = preferred[0]
print("Using:", src, "size_mb=%.1f" % (os.path.getsize(src) / 1024**2))

dest_rel = "loras/" + OUTPUT_LORA_NAME
dest_abs = os.path.join(ROOT, dest_rel)
if os.path.basename(dest_abs) in PROTECTED_LORAS:
    raise RuntimeError("Refusing to overwrite protected LoRA: " + dest_abs)

upload_project_file(src, dest_rel)
print("LoRA saved as", dest_rel)

sample_dir = os.path.join(run_dir, "samples")
local_imgs = list(iter_images(sample_dir))
if not local_imgs:
    local_imgs = list(iter_images(run_dir))
copied_src = []
seen = set()
for path in local_imgs:
    name = os.path.basename(path)
    if name in seen:
        continue
    seen.add(name)
    dest_rel_img = "output/%s/flux_samples/%s" % (PROJECT_NAME, name)
    upload_project_file(path, dest_rel_img)
    copied_src.append(path)

print("Copied", len(copied_src), "training samples to Drive:", SAMPLES_DIR)
print("--- training samples ---")
show_images(copied_src)
if not copied_src:
    print("Ostris did not write sample PNGs in", sample_dir)
    print("That is OK. Run cell 10 to generate pictures and see them in this notebook.")
print("Protected files were not touched.")
print("Next: cell 10 generates identity / lingerie / nude and DISPLAYS them here.")"""
)

code(
    r"""# @title 10) Generate (identity / lingerie / nude) -- no safety checker
import os
import gc
import sys
import subprocess
import importlib
import torch
from datetime import datetime
from PIL import Image
from IPython.display import display

print("Clearing GPU cache after training.")
print("If this cell OOMs: Runtime > Restart session, rerun cells 1-3, then this cell.")
gc.collect()
torch.cuda.empty_cache()


def _purge_old_torchao():
    # ai-toolkit trains with torchao==0.10.0. New peft refuses that and
    # cannot load a Flux LoRA. Inference does not need torchao.
    try:
        from importlib.metadata import version
        ver = version("torchao")
    except Exception:
        print("torchao not installed")
        return
    print("torchao", ver)
    nums = []
    for part in ver.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            nums.append(int(digits))
    while len(nums) < 3:
        nums.append(0)
    if tuple(nums[:3]) >= (0, 16, 0):
        return
    print("peft needs torchao>=0.16 to load Flux LoRA. Uninstalling", ver)
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
    for key in list(sys.modules):
        if key == "torchao" or key.startswith("torchao.") or key.startswith("peft.tuners.lora.torchao"):
            del sys.modules[key]
    if "peft.import_utils" in sys.modules:
        importlib.reload(sys.modules["peft.import_utils"])
    print("old torchao removed")


_purge_old_torchao()

# identity = face check
# lingerie = clothed NSFW
# nude = unclothed NSFW
# all = three of each
MODE = "all"          # "identity" | "lingerie" | "nude" | "all"
NUM_PER_PROMPT = 2
GUIDANCE = 3.5
STEPS = 28
WIDTH = 768
HEIGHT = 1024
SEED = 42
LORA_WEIGHT = 1.0
SKIN_LOCK = (
    "smooth natural skin, no scars, no surgical marks, no stretch marks, "
    "unblemished skin, natural anatomy"
)

if not SUBJECT_IS_ADULT:
    raise RuntimeError("Adult subject only.")

lora_path = os.path.join(LORAS_DIR, OUTPUT_LORA_NAME)
if USE_DRIVE_API and (not os.path.isfile(lora_path)):
    lora_path = os.path.join(TRAIN_OUTPUT_DIR, LORA_NAME, OUTPUT_LORA_NAME)
if not os.path.isfile(lora_path):
    local_final = os.path.join(TRAIN_OUTPUT_DIR, LORA_NAME, OUTPUT_LORA_NAME)
    if os.path.isfile(local_final):
        lora_path = local_final
if not os.path.isfile(lora_path):
    raise RuntimeError("LoRA not found. Run cell 8 and cell 9 first. Looked for " + lora_path)

from diffusers import FluxPipeline

print("Loading FLUX.1-dev + LoRA (no safety checker)...")
print("LoRA file:", lora_path)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    token=os.environ.get("HF_TOKEN"),
)
if getattr(pipe, "safety_checker", None) is not None:
    pipe.safety_checker = None
    print("Disabled pipeline.safety_checker")
if hasattr(pipe, "requires_safety_checker"):
    pipe.requires_safety_checker = False
if getattr(pipe, "watermarker", None) is not None:
    pipe.watermarker = None
    print("Disabled watermarker")
try:
    pipe.load_lora_weights(lora_path)
except ImportError as err:
    print("load_lora_weights ImportError:", err)
    _purge_old_torchao()
    pipe.load_lora_weights(lora_path)
try:
    pipe.fuse_lora(lora_scale=LORA_WEIGHT)
except Exception as err:
    print("fuse_lora skipped:", err)
try:
    pipe.to("cuda")
except torch.cuda.OutOfMemoryError:
    print("GPU full. Using CPU offload.")
    gc.collect()
    torch.cuda.empty_cache()
    pipe.enable_model_cpu_offload()
print("LoRA loaded.")

PROMPTS = {
    "identity": (
        "ohwx woman, close-up portrait of an adult woman with long highlighted "
        "blonde hair and brown eyes, looking at the camera, photorealistic raw photo, "
        "natural skin texture, sharp eyes, " + SKIN_LOCK
    ),
    "lingerie": (
        "ohwx woman, an adult woman with long highlighted blonde hair and brown eyes, "
        "full body standing, black lace lingerie, looking at the camera, indoor fashion photo, "
        "photorealistic, natural skin texture, " + SKIN_LOCK
    ),
    "nude": (
        "ohwx woman, an adult woman with long highlighted blonde hair and brown eyes, "
        "full body standing nude, looking at the camera, photorealistic raw photo, "
        "natural skin texture, realistic anatomy, " + SKIN_LOCK
    ),
}

if MODE == "all":
    selected = ["identity", "lingerie", "nude"]
else:
    if MODE not in PROMPTS:
        raise RuntimeError("MODE must be identity, lingerie, nude, or all")
    selected = [MODE]

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(EVAL_DIR, stamp)
os.makedirs(out_dir, exist_ok=True)
saved = []
idx = 0
for kind in selected:
    prompt = PROMPTS[kind]
    for i in range(NUM_PER_PROMPT):
        seed = SEED + idx
        print("gen", kind, "seed", seed)
        image = pipe(
            prompt=prompt,
            guidance_scale=GUIDANCE,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        fname = "%s_%02d_seed%d.png" % (kind, i + 1, seed)
        path = os.path.join(out_dir, fname)
        image.save(path)
        saved.append(path)
        print("saved", path)
        display(image)
        idx += 1

print("Saved", len(saved), "pictures in", out_dir)
if USE_DRIVE_API:
    for path in saved:
        rel = os.path.relpath(path, ROOT)
        upload_project_file(path, rel)
print("Pictures are also on Drive:", out_dir)
print("No safety checker ran. Copy keepers to MyDrive/FiratSuper/keepers/")
print("Use ohwx woman in every prompt. Do not stack old SD LoRAs.")"""
)

md(
    """## Done

LoRA file (new only):
`MyDrive/FiratSuper/loras/lapetitemilf_flux.safetensors`

Training samples:
`MyDrive/FiratSuper/output/lapetitemilf/flux_samples/`

Generations from cell 10:
`MyDrive/FiratSuper/output/lapetitemilf/flux_eval/`

Cell 8 does not print pictures. Cell 9 and cell 10 display them in the notebook.

Protected (not touched):
`loras/lapetitemilf_face.safetensors`

### Use later
1. Keep this Colab. Rerun cells 1, 2, 3, then cell 10.
2. In ComfyUI: Flux.1 [dev] checkpoint + this LoRA, trigger `ohwx woman`.
3. Do not load SD 1.5 LoRAs on Flux.
4. Adult content only.

### If training dies
- Tab slept / GPU dropped: reconnect A100, rerun 1-4, then 8. Checkpoints are in `/content/output/lapetitemilf_flux/` until the VM is deleted.
- Hugging Face 403: accept FLUX.1-dev license, new READ token.
- Drive popup: Allow ALL, one Google account."""
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
        "colab": {"provenance": [], "gpuType": "A100"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

notebook = nbformat.from_dict(nb)
normalize(notebook)
for i, cell in enumerate(notebook.cells):
    if "id" not in cell:
        cell["id"] = f"cell-{i:04d}"

out = Path("/workspace/notebooks/Flux_LoRA_Training_Colab.ipynb")
payload = json.loads(nbformat.writes(notebook))
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
