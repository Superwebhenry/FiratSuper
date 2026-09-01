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

**Training is done.** Use this notebook to **generate** with the locked v2 LoRA.

**Runtime:** Runtime > Change runtime type > **A100 GPU**. Do not pick T4. Do not pick TPU.
High RAM can stay off.

**Drive:** Chrome, one Google account only (`superweb.contact@gmail.com`). In the popup: Continue, then **Allow ALL** permissions.

**Hugging Face:** Accept the license for [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and paste a READ token.

**Locked LoRA (do not overwrite):** `MyDrive/FiratSuper/loras/lapetitemilf_flux_v2.safetensors`
Also locked: `lapetitemilf_flux` (v1) and `lapetitemilf_face`. Do not retrain. Do not run cells 5-9.

**Generate path:** cells 1, 2, 3, then 4 if new runtime, then **one series cell** (13-22). Skip 5-9. Cell 11 is optional refine. Cell 12 is mixed outdoor nudes.

**Trigger:** `ohwx woman`. Do not write "no scars" in prompts. Adult subject only.

## Cells
1. A100 GPU check
2. Drive + settings
3. Hugging Face login
4. Install (needed on a fresh runtime; skip if packages are already in this session)
5-9. Training (LOCKED -- do not run)
10. Generate and SHOW pictures (identity + lingerie + nude, no filter)
11. Refine a keeper (img2img, keep frontal)
12. 20 outdoor nude styles (forest, meadow, rocks, path, bench)
13. Far strip: pool, dark bikini (20)
14. Far strip: hotel, evening gown to black lingerie (20)
15. Far strip: outdoor shower, towel (20)
16. Far strip: beach, bikini (20)
17. Far strip: forest, white sundress (20)
18. Far strip: apartment, shirt to plaid lingerie (20)
19. Far strip: bathtub, white robe (20)
20. Far strip: hot tub, bikini (20)
21. Far strip: greenhouse, white dress (20)
22. Far strip: bedroom, black slip (20)

## Drive layout
```
MyDrive/FiratSuper/
|-- ADD_FLUX_PHOTOS/                      # 31 keepers + .txt captions
|-- ADD_FLUX_CHEST/                       # 7 chest keepers + .txt (skip unpaired)
|-- loras/lapetitemilf_flux_v2.safetensors # LOCKED production LoRA
|-- loras/lapetitemilf_flux.safetensors    # v1, locked
|-- loras/lapetitemilf_face.safetensors    # locked
|-- output/lapetitemilf/flux_eval_v2/      # generations from cell 10
`-- keepers/                              # copy keepers here
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
FLUX_CHEST_ID = "1iEmUvagFQVJ2TArN_7ee4Af4TUti1hZw"
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
    print("FUSE mount failed. Copying Flux folders via Drive API to", local_root)
    service = _api_service()
    os.makedirs(local_root, exist_ok=True)
    inbox = api_find_child(service, FIRATSUPER_DRIVE_ID, "ADD_FLUX_PHOTOS")
    inbox_id = inbox["id"] if inbox else FLUX_INBOX_ID
    api_download_folder(
        service,
        inbox_id,
        os.path.join(local_root, "ADD_FLUX_PHOTOS"),
    )
    chest = api_find_child(service, FIRATSUPER_DRIVE_ID, "ADD_FLUX_CHEST")
    chest_id = chest["id"] if chest else FLUX_CHEST_ID
    api_download_folder(
        service,
        chest_id,
        os.path.join(local_root, "ADD_FLUX_CHEST"),
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
LORA_NAME = "lapetitemilf_flux_v2"
EXPECTED_PAIRS = 38
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
    "lapetitemilf_flux.safetensors",
    "lapetitemilf_flux_v2.safetensors",
}
OUTPUT_LORA_NAME = LORA_NAME + ".safetensors"
if OUTPUT_LORA_NAME in PROTECTED_LORAS:
    print("LoRA is LOCKED:", OUTPUT_LORA_NAME)
    print("Do not run cells 5-9. Generate with cell 10.")
if not SUBJECT_IS_ADULT:
    raise RuntimeError("This notebook is for an adult subject only.")

if USE_DRIVE_API:
    ROOT = "/content/FiratSuper"
else:
    ROOT = "/content/drive/MyDrive/FiratSuper"

INBOX_DIR = os.path.join(ROOT, "ADD_FLUX_PHOTOS")
CHEST_DIR = os.path.join(ROOT, "ADD_FLUX_CHEST")
LORAS_DIR = os.path.join(ROOT, "loras")
KEEPERS_DIR = os.path.join(ROOT, "keepers")
EVAL_DIR = os.path.join(ROOT, "output", PROJECT_NAME, "flux_eval_v2")
SAMPLES_DIR = os.path.join(ROOT, "output", PROJECT_NAME, "flux_samples_v2")
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
print("Chest:", CHEST_DIR)
print("Local dataset:", DATASET_DIR)
print("LoRA out:", os.path.join(LORAS_DIR, OUTPUT_LORA_NAME))
print("Free disk: %.1f GB" % free_gb)
if free_gb < 40:
    raise RuntimeError("Need ~40 GB free for Flux.1-dev. Have %.1f GB." % free_gb)


def _purge_old_torchao():
    import sys
    import subprocess
    import importlib
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


def ensure_flux_pipe():
    global pipe
    import gc
    import os
    import torch
    need_load = True
    if "pipe" in globals() and pipe is not None:
        if type(pipe).__name__ == "FluxPipeline":
            need_load = False
            print("Using Flux txt2img already in memory.")
        else:
            print("In-memory pipe is", type(pipe).__name__, "- loading txt2img.")
    if not need_load:
        return pipe
    lora_path = os.path.join(LORAS_DIR, OUTPUT_LORA_NAME)
    if not os.path.isfile(lora_path):
        local_final = os.path.join(TRAIN_OUTPUT_DIR, LORA_NAME, OUTPUT_LORA_NAME)
        if os.path.isfile(local_final):
            lora_path = local_final
    if not os.path.isfile(lora_path):
        raise RuntimeError("LoRA not found: " + lora_path)
    gc.collect()
    torch.cuda.empty_cache()
    _purge_old_torchao()
    from diffusers import FluxPipeline
    print("Loading FLUX.1-dev + LoRA (no safety checker)...")
    print("LoRA file:", lora_path)
    loaded = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    if getattr(loaded, "safety_checker", None) is not None:
        loaded.safety_checker = None
    if hasattr(loaded, "requires_safety_checker"):
        loaded.requires_safety_checker = False
    if getattr(loaded, "watermarker", None) is not None:
        loaded.watermarker = None
    try:
        loaded.load_lora_weights(lora_path)
    except ImportError as err:
        print("load_lora_weights ImportError:", err)
        _purge_old_torchao()
        loaded.load_lora_weights(lora_path)
    try:
        loaded.unfuse_lora()
    except Exception:
        pass
    loaded.enable_model_cpu_offload()
    pipe = loaded
    print("LoRA loaded.")
    return pipe


def run_far_strip(slug, place, shots, seed_base, shot_start=0, shot_end=20):
    import os
    import torch
    from datetime import datetime
    from IPython.display import display
    if not SUBJECT_IS_ADULT:
        raise RuntimeError("Adult subject only.")
    if len(shots) != 20:
        raise RuntimeError("Need 20 shots in this series. Got %d" % len(shots))
    if shot_start < 0 or shot_end > 20 or shot_start >= shot_end:
        raise RuntimeError("SHOT_START/END must be inside 0..20 and START < END")
    ensure_flux_pipe()
    far = (
        "full body from a far camera, she is small in the frame, "
        "the location fills most of the shot"
    )
    ident = (
        "ohwx woman, an adult woman with long highlighted blonde hair and brown eyes, "
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(EVAL_DIR, "strip_" + slug + "_" + stamp)
    os.makedirs(out_dir, exist_ok=True)
    print("Series", slug)
    print("Shots", shot_start, "to", shot_end, "->", out_dir)
    print("Do not write scars or surgical in these prompts.")
    print("Keep this tab open.")
    saved = []
    for pidx in range(shot_start, shot_end):
        shot_slug, kind, action = shots[pidx]
        if kind == "nude":
            weight = 0.75
            guidance = 2.5
        else:
            weight = 1.0
            guidance = 3.5
        try:
            pipe.set_adapters(["default"], adapter_weights=[weight])
        except Exception as err:
            print("set_adapters:", err)
        seed = seed_base + pidx
        prompt = (
            ident + far + ", " + action + ". " + place +
            ", photorealistic raw photo, natural skin texture"
        )
        print("---", shot_slug, "seed", seed, kind)
        print(prompt)
        image = pipe(
            prompt=prompt,
            guidance_scale=guidance,
            height=1024,
            width=768,
            num_inference_steps=32,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        path = os.path.join(out_dir, "%s_seed%d.png" % (shot_slug, seed))
        image.save(path)
        saved.append(path)
        print("saved", path)
        display(image)
    print("Saved", len(saved), "pictures in", out_dir)
    if USE_DRIVE_API:
        for path in saved:
            upload_project_file(path, os.path.relpath(path, ROOT))
    print("If it stopped early, set SHOT_START to the next index and rerun this cell.")
    print("Copy keepers to MyDrive/FiratSuper/keepers/")
    print("Do not put these pictures back into the training folders.")


print("Drive settings OK. Far-strip helper ready for cells 13-22.")"""
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
    r"""# @title 5) Gate 1: copy ADD_FLUX_PHOTOS + ADD_FLUX_CHEST
import os
import shutil

if OUTPUT_LORA_NAME in PROTECTED_LORAS:
    raise RuntimeError(
        "Training is locked. " + OUTPUT_LORA_NAME + " is on Drive. Run cell 10 to generate."
    )

IMG_EXT = {".jpg", ".jpeg", ".png"}
SKIP_NAMES = {".drive_upload.json", ".ds_store", "thumbs.db"}
SOURCE_FOLDERS = [
    ("ADD_FLUX_PHOTOS", INBOX_DIR),
    ("ADD_FLUX_CHEST", CHEST_DIR),
]

need_sync = False
if USE_DRIVE_API:
    if (not os.path.isdir(INBOX_DIR)) or (not os.listdir(INBOX_DIR)):
        need_sync = True
    if (not os.path.isdir(CHEST_DIR)) or (not os.listdir(CHEST_DIR)):
        need_sync = True
if need_sync:
    print("Re-copy Flux folders via Drive API...")
    DRIVE_SERVICE = sync_flux_via_api(ROOT)

if not os.path.isdir(INBOX_DIR):
    raise RuntimeError("Missing inbox folder: " + INBOX_DIR)
if not os.path.isdir(CHEST_DIR):
    print("WARNING: missing chest folder:", CHEST_DIR)

if os.path.isdir(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)
os.makedirs(DATASET_DIR, exist_ok=True)

pairs = []
missing_txt = []
skipped = []
seen_stems = {}
for label, folder in SOURCE_FOLDERS:
    if not os.path.isdir(folder):
        print("WARNING: skip missing folder", label)
        continue
    print("Scanning", label)
    for name in sorted(os.listdir(folder)):
        if name.startswith("."):
            continue
        if name.lower() in SKIP_NAMES:
            skipped.append(label + "/" + name)
            continue
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            skipped.append(label + "/" + name + "/")
            continue
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMG_EXT:
            if ext.lower() != ".txt":
                skipped.append(label + "/" + name)
            continue
        txt_name = stem + ".txt"
        txt_path = os.path.join(folder, txt_name)
        if not os.path.isfile(txt_path):
            missing_txt.append(label + "/" + name)
            continue
        if stem in seen_stems:
            skipped.append(label + "/" + name + " (dup of " + seen_stems[stem] + ")")
            continue
        dest_img = os.path.join(DATASET_DIR, name)
        dest_txt = os.path.join(DATASET_DIR, txt_name)
        shutil.copy2(path, dest_img)
        shutil.copy2(txt_path, dest_txt)
        with open(txt_path, "r", encoding="utf-8") as fh:
            caption = fh.read().strip()
        seen_stems[stem] = label
        pairs.append((name, caption))

print("Copied pairs:", len(pairs))
if skipped:
    print("Skipped:", ", ".join(skipped[:20]))
if missing_txt:
    print("WARNING: skipping images with no matching .txt (not trained):")
    for name in missing_txt:
        print("  ", name)
    print("That is OK for a dropped gen. Do not caption Flux outputs.")

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

if OUTPUT_LORA_NAME in PROTECTED_LORAS:
    raise RuntimeError(
        "Training is locked. " + OUTPUT_LORA_NAME + " is on Drive. Run cell 10 to generate."
    )

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

if OUTPUT_LORA_NAME in PROTECTED_LORAS:
    raise RuntimeError(
        "Training is locked. " + OUTPUT_LORA_NAME + " is on Drive. Run cell 10 to generate."
    )

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

if OUTPUT_LORA_NAME in PROTECTED_LORAS:
    raise RuntimeError(
        "Training is locked. " + OUTPUT_LORA_NAME + " is on Drive. Run cell 10 to generate."
    )

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
    dest_rel_img = "output/%s/flux_samples_v2/%s" % (PROJECT_NAME, name)
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
#
# Flux has no real negative prompt. Do NOT write "no scars" or "surgical".
# Those words make Flux DRAW scars. First batch had one bad frame; after
# adding "no scars" every nude got the lines.
MODE = "nude"         # "identity" | "lingerie" | "nude" | "all"
NUM_PER_PROMPT = 4
GUIDANCE = 3.5
NUDE_GUIDANCE = 2.5
STEPS = 32
WIDTH = 768
HEIGHT = 1024
SEED = 501
LORA_WEIGHT = 1.0
NUDE_LORA_WEIGHT = 0.75

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
    pipe.unfuse_lora()
except Exception:
    pass
try:
    pipe.set_adapters(["default"], adapter_weights=[LORA_WEIGHT])
except Exception as err:
    print("set_adapters skipped:", err)
    try:
        pipe.fuse_lora(lora_scale=LORA_WEIGHT)
    except Exception as err2:
        print("fuse_lora skipped:", err2)
print("Using CPU offload so cell 11 can run on the same A100 (T5+Flux fill 40GB).")
pipe.enable_model_cpu_offload()
print("LoRA loaded.")

PROMPTS = {
    "identity": (
        "ohwx woman, close-up portrait of an adult woman with long highlighted "
        "blonde hair and brown eyes, looking at the camera, photorealistic raw photo, "
        "natural skin texture, sharp eyes, soft even lighting"
    ),
    "lingerie": (
        "ohwx woman, an adult woman with long highlighted blonde hair and brown eyes, "
        "full body standing, black lace lingerie, looking at the camera, indoor fashion photo, "
        "photorealistic, natural skin texture, soft even lighting"
    ),
    "nude": (
        "ohwx woman, an adult woman with long highlighted blonde hair and brown eyes, "
        "waist-up, facing the camera, square frontal view, standing nude, "
        "looking at the camera, soft even indoor lighting, photorealistic raw photo, smooth skin"
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
    weight = NUDE_LORA_WEIGHT if kind == "nude" else LORA_WEIGHT
    guidance = NUDE_GUIDANCE if kind == "nude" else GUIDANCE
    try:
        pipe.set_adapters(["default"], adapter_weights=[weight])
    except Exception as err:
        print("set_adapters:", err)
    for i in range(NUM_PER_PROMPT):
        seed = SEED + idx
        print("gen", kind, "seed", seed, "guidance", guidance, "lora", weight)
        print("prompt:", prompt)
        image = pipe(
            prompt=prompt,
            guidance_scale=guidance,
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
print("Use ohwx woman in every prompt. Do not stack old SD LoRAs.")
print("If a frontal nude is close but the chest has faint lines: run cell 11 on that file.")"""
)

code(
    r"""# @title 11) Refine a frontal keeper (img2img, do not retrain)
import os
import gc
import inspect
import torch
from datetime import datetime
from PIL import Image
from IPython.display import display

# Empty SOURCE = last picture from cell 10.
# Example: "/content/drive/MyDrive/FiratSuper/output/lapetitemilf/flux_eval/20260827_092544/nude_03_seed506.png"
SOURCE = ""
STRENGTH = 0.28
REFINE_SEED = 7
REFINE_STEPS = 28
REFINE_GUIDANCE = 2.5

if "pipe" not in globals():
    raise RuntimeError("Run cell 10 first so FLUX + LoRA stay in memory, then this cell.")

src_path = SOURCE.strip()
if not src_path:
    if "saved" in globals() and saved:
        src_path = saved[-1]
    else:
        raise RuntimeError("Set SOURCE to a PNG path, or run cell 10 first.")
if not os.path.isfile(src_path):
    raise RuntimeError("File not found: " + src_path)

base = Image.open(src_path).convert("RGB")
try:
    resample = Image.Resampling.LANCZOS
except AttributeError:
    resample = Image.LANCZOS
base = base.resize((WIDTH, HEIGHT), resample)
print("source:", src_path, base.size)

prompt = (
    "ohwx woman, an adult woman with long highlighted blonde hair and brown eyes, "
    "waist-up, facing the camera, square frontal view, standing nude, "
    "looking at the camera, soft even indoor lighting, photorealistic raw photo, smooth skin"
)
print("prompt:", prompt)
print("Do not put the words scars or surgical in this prompt.")

print("VRAM before offload: %.1f GB" % (torch.cuda.memory_allocated() / 1024**3))
print("Offloading the SAME Flux to CPU. Do not clone. from_pipe recasts and OOMs.")
try:
    pipe.enable_model_cpu_offload()
except Exception as err:
    print("enable_model_cpu_offload:", err)
    try:
        pipe.to("cpu")
    except Exception as err2:
        print("pipe.to cpu:", err2)
gc.collect()
torch.cuda.empty_cache()
print("VRAM after offload: %.1f GB" % (torch.cuda.memory_allocated() / 1024**3))

from diffusers import FluxImg2ImgPipeline

sig = inspect.signature(FluxImg2ImgPipeline.__init__)
comps = {}
for key, value in pipe.components.items():
    if key in sig.parameters:
        comps[key] = value
img2img = FluxImg2ImgPipeline(**comps)
print("Wrapped components. No from_pipe, no second download.")
try:
    img2img.enable_model_cpu_offload()
except Exception as err:
    print("img2img offload:", err)

try:
    img2img.set_adapters(["default"], adapter_weights=[NUDE_LORA_WEIGHT])
except Exception as err:
    print("set_adapters:", err)

pipe = img2img
print("VRAM after wrap: %.1f GB" % (torch.cuda.memory_allocated() / 1024**3))

image = img2img(
    prompt=prompt,
    image=base,
    strength=STRENGTH,
    guidance_scale=REFINE_GUIDANCE,
    num_inference_steps=REFINE_STEPS,
    generator=torch.Generator("cuda").manual_seed(REFINE_SEED),
).images[0]

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(EVAL_DIR, "refine_" + stamp)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "refine_s%.2f_seed%d.png" % (STRENGTH, REFINE_SEED))
image.save(out_path)
print("saved", out_path)
print("--- before ---")
display(base)
print("--- after ---")
display(image)
if USE_DRIVE_API:
    upload_project_file(out_path, os.path.relpath(out_path, ROOT))
print("If the face drifted, lower STRENGTH to 0.22 and rerun. If lines remain, try 0.32.")
print("Stay frontal. Copy keepers to MyDrive/FiratSuper/keepers/")"""
)

code(
    r"""# @title 12) 20 outdoor nude styles -- no safety checker
import os
import gc
import sys
import subprocess
import importlib
import torch
from datetime import datetime
from IPython.display import display

# 20 original outdoor looks for THIS character. Not copies of stock sites.
# START=0 END=20 runs all. For a short test: END=3
# If the runtime dies, rerun with START set to the next index.
START = 0
END = 20
SEED = 800
STEPS = 32
NUDE_GUIDANCE = 2.5
NUDE_LORA_WEIGHT = 0.75

ID = (
    "ohwx woman, an adult woman with long highlighted blonde hair and brown eyes, "
)

STYLES = [
    ("01_forest_stand", 768, 1024, ID + "full body standing nude in a green forest, looking at the camera, dappled sunlight through trees, photorealistic raw photo, natural skin texture"),
    ("02_meadow_stand", 768, 1024, ID + "full body standing nude in a lush green meadow, looking at the camera, tall grass around her legs, bright daylight, photorealistic raw photo, natural skin texture"),
    ("03_golden_field", 768, 1024, ID + "full body standing nude in a dry golden grass field, looking at the camera, warm late-afternoon sunlight, photorealistic raw photo, natural skin texture"),
    ("04_forest_sit", 768, 1024, ID + "sitting nude on the forest floor, looking at the camera, green plants around her, soft overcast forest light, photorealistic raw photo, natural skin texture"),
    ("05_meadow_kneel", 768, 1024, ID + "kneeling nude in green grass, looking at the camera, outdoor meadow, even daylight, photorealistic raw photo, natural skin texture"),
    ("06_tree_lean", 768, 1024, ID + "standing nude leaning back against a tree trunk, looking at the camera, forest around her, dappled sunlight, photorealistic raw photo, natural skin texture"),
    ("07_rock_sit", 768, 1024, ID + "sitting nude on a large sunlit rock, looking at the camera, rocky hillside behind her, bright daylight, photorealistic raw photo, natural skin texture"),
    ("08_grass_lie", 1024, 768, ID + "lying nude in tall green grass, looking at the camera, meadow around her, soft daylight, photorealistic raw photo, natural skin texture"),
    ("09_cliff_stand", 768, 1024, ID + "full body standing nude on a rocky outcrop, looking at the camera, distant hills behind her, bright daylight, photorealistic raw photo, natural skin texture"),
    ("10_forest_path", 768, 1024, ID + "full body standing nude on a dirt path in the woods, looking at the camera, trees on both sides, natural daylight, photorealistic raw photo, natural skin texture"),
    ("11_wood_bench", 768, 1024, ID + "sitting nude on a wooden outdoor bench, looking at the camera, trees and grass behind her, soft daylight, photorealistic raw photo, natural skin texture"),
    ("12_rural_fence", 768, 1024, ID + "standing nude beside a weathered wooden fence, looking at the camera, rural field behind her, warm sunlight, photorealistic raw photo, natural skin texture"),
    ("13_creek_stand", 768, 1024, ID + "standing nude at a shallow forest creek, looking at the camera, water around her feet, green woods, soft daylight, photorealistic raw photo, natural skin texture"),
    ("14_shoulder_look", 768, 1024, ID + "full body standing nude in a sunlit forest clearing, looking back over her shoulder at the camera, photorealistic raw photo, natural skin texture"),
    ("15_outdoor_close", 768, 1024, ID + "waist-up nude outdoor portrait, looking at the camera, blurred trees behind her, soft daylight, photorealistic raw photo, natural skin texture"),
    ("16_meadow_waist", 768, 1024, ID + "waist-up standing nude in a green meadow, looking at the camera, sunlight on her skin, photorealistic raw photo, natural skin texture"),
    ("17_wide_field", 1024, 768, ID + "full body standing nude in a wide grassy field, looking at the camera, big sky and landscape around her, bright daylight, photorealistic raw photo, natural skin texture"),
    ("18_fallen_log", 768, 1024, ID + "sitting nude on a fallen log in the woods, looking at the camera, moss and trees around her, overcast forest light, photorealistic raw photo, natural skin texture"),
    ("19_wildflowers", 768, 1024, ID + "full body standing nude among wildflowers in a meadow, looking at the camera, bright daylight, photorealistic raw photo, natural skin texture"),
    ("20_overcast_woods", 768, 1024, ID + "full body standing nude in a quiet overcast forest, looking at the camera, even soft light, photorealistic raw photo, natural skin texture"),
]

if not SUBJECT_IS_ADULT:
    raise RuntimeError("Adult subject only.")
if START < 0 or END > len(STYLES) or START >= END:
    raise RuntimeError("Set START/END inside 0..%d and START < END" % len(STYLES))

print("Do not write scars or surgical in these prompts.")
print("Keep this tab open. 20 pictures can take a while.")


def _purge_old_torchao():
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


def _load_txt2img():
    global pipe
    lora_path = os.path.join(LORAS_DIR, OUTPUT_LORA_NAME)
    if not os.path.isfile(lora_path):
        local_final = os.path.join(TRAIN_OUTPUT_DIR, LORA_NAME, OUTPUT_LORA_NAME)
        if os.path.isfile(local_final):
            lora_path = local_final
    if not os.path.isfile(lora_path):
        raise RuntimeError("LoRA not found: " + lora_path)
    gc.collect()
    torch.cuda.empty_cache()
    _purge_old_torchao()
    from diffusers import FluxPipeline
    print("Loading FLUX.1-dev + LoRA (no safety checker)...")
    print("LoRA file:", lora_path)
    loaded = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    if getattr(loaded, "safety_checker", None) is not None:
        loaded.safety_checker = None
    if hasattr(loaded, "requires_safety_checker"):
        loaded.requires_safety_checker = False
    if getattr(loaded, "watermarker", None) is not None:
        loaded.watermarker = None
    try:
        loaded.load_lora_weights(lora_path)
    except ImportError as err:
        print("load_lora_weights ImportError:", err)
        _purge_old_torchao()
        loaded.load_lora_weights(lora_path)
    try:
        loaded.unfuse_lora()
    except Exception:
        pass
    loaded.enable_model_cpu_offload()
    pipe = loaded
    print("LoRA loaded.")


need_load = True
if "pipe" in globals() and pipe is not None:
    if type(pipe).__name__ == "FluxPipeline":
        need_load = False
        print("Using Flux txt2img already in memory.")
    else:
        print("In-memory pipe is", type(pipe).__name__, "- loading txt2img.")
if need_load:
    _load_txt2img()

try:
    pipe.set_adapters(["default"], adapter_weights=[NUDE_LORA_WEIGHT])
except Exception as err:
    print("set_adapters:", err)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(EVAL_DIR, "outdoor_" + stamp)
os.makedirs(out_dir, exist_ok=True)
saved = []
batch = STYLES[START:END]
print("Running styles", START, "to", END, "->", out_dir)

for i, (slug, width, height, prompt) in enumerate(batch):
    idx = START + i
    seed = SEED + idx
    print("---", slug, "seed", seed, width, "x", height)
    print(prompt)
    image = pipe(
        prompt=prompt,
        guidance_scale=NUDE_GUIDANCE,
        height=height,
        width=width,
        num_inference_steps=STEPS,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    fname = "%s_seed%d.png" % (slug, seed)
    path = os.path.join(out_dir, fname)
    image.save(path)
    saved.append(path)
    print("saved", path)
    display(image)

print("Saved", len(saved), "pictures in", out_dir)
if USE_DRIVE_API:
    for path in saved:
        upload_project_file(path, os.path.relpath(path, ROOT))
print("Copy keepers to MyDrive/FiratSuper/keepers/")
print("Do not put these pictures back into the training folders.")"""
)

code(
    r"""# @title 13) Far strip: pool, dark bikini (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'private backyard pool, pale stone deck, a sun lounger, bright daylight'
SLUG = '13_pool_bikini'
SEED_BASE = 2000
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a dark two-piece bikini'),
    ('02_walk', 'clothed', 'walking, wearing a dark two-piece bikini'),
    ('03_sit', 'clothed', 'sitting, wearing a dark two-piece bikini'),
    ('04_shoulder', 'clothed', 'wearing a dark two-piece bikini, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'standing, wearing a dark two-piece bikini, untying the bikini top'),
    ('06_loosen', 'clothed', 'standing, wearing a dark two-piece bikini, untying the bikini top, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, topless, wearing only dark bikini bottoms'),
    ('08_mid_walk', 'clothed', 'walking, topless, wearing only dark bikini bottoms'),
    ('09_mid_sit', 'clothed', 'sitting, topless, wearing only dark bikini bottoms'),
    ('10_mid_shoulder', 'clothed', 'topless, wearing only dark bikini bottoms, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'topless, pulling down dark bikini bottoms'),
    ('12_almost_hips', 'clothed', 'topless, pulling down dark bikini bottoms, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'topless, pulling down dark bikini bottoms, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the dark bikini'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 14) Far strip: hotel, evening gown (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'large hotel suite in the evening, tall windows, a sofa, warm lamps, polished floor'
SLUG = '14_hotel_gown'
SEED_BASE = 2100
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a long black evening gown'),
    ('02_walk', 'clothed', 'walking, wearing a long black evening gown'),
    ('03_sit', 'clothed', 'sitting, wearing a long black evening gown'),
    ('04_shoulder', 'clothed', 'wearing a long black evening gown, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'standing, wearing a long black evening gown, unzipping the back'),
    ('06_loosen', 'clothed', 'standing, wearing a long black evening gown, unzipping the back, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, wearing a black bra and black panties, the gown off'),
    ('08_mid_walk', 'clothed', 'walking, wearing a black bra and black panties, the gown off'),
    ('09_mid_sit', 'clothed', 'sitting, wearing a black bra and black panties, the gown off'),
    ('10_mid_shoulder', 'clothed', 'wearing a black bra and black panties, the gown off, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'topless, pulling down black panties'),
    ('12_almost_hips', 'clothed', 'topless, pulling down black panties, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'topless, pulling down black panties, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the black evening gown'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 15) Far strip: outdoor shower, towel (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'outdoor shower, dark stacked-stone wall, white outdoor stairs with a metal rail, a shower head pouring water, green plants, bright daylight'
SLUG = '15_outdoor_shower'
SEED_BASE = 2200
SHOTS = [
    ('01_stand', 'clothed', 'standing, wrapped in a white towel, wet highlighted blonde hair, water falling over her'),
    ('02_walk', 'clothed', 'walking, wrapped in a white towel, wet highlighted blonde hair, water falling over her'),
    ('03_sit', 'clothed', 'sitting, wrapped in a white towel, wet highlighted blonde hair, water falling over her'),
    ('04_shoulder', 'clothed', 'wrapped in a white towel, wet highlighted blonde hair, water falling over her, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'standing under the water, loosening a white towel wrapped around her'),
    ('06_loosen', 'clothed', 'standing under the water, loosening a white towel wrapped around her, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, the white towel around her hips, topless, wet skin'),
    ('08_mid_walk', 'clothed', 'walking, the white towel around her hips, topless, wet skin'),
    ('09_mid_sit', 'clothed', 'sitting, the white towel around her hips, topless, wet skin'),
    ('10_mid_shoulder', 'clothed', 'the white towel around her hips, topless, wet skin, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'the white towel around her thighs, otherwise nude'),
    ('12_almost_hips', 'clothed', 'the white towel around her thighs, otherwise nude, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'the white towel around her thighs, otherwise nude, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the white towel'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 16) Far strip: beach, bikini (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'sandy beach at the ocean, bright daylight, blue water and sky behind her'
SLUG = '16_beach_bikini'
SEED_BASE = 2300
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a dark two-piece bikini'),
    ('02_walk', 'clothed', 'walking, wearing a dark two-piece bikini'),
    ('03_sit', 'clothed', 'sitting, wearing a dark two-piece bikini'),
    ('04_shoulder', 'clothed', 'wearing a dark two-piece bikini, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'standing on the sand, wearing a dark two-piece bikini, untying the bikini top'),
    ('06_loosen', 'clothed', 'standing on the sand, wearing a dark two-piece bikini, untying the bikini top, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, topless, wearing only dark bikini bottoms'),
    ('08_mid_walk', 'clothed', 'walking, topless, wearing only dark bikini bottoms'),
    ('09_mid_sit', 'clothed', 'sitting, topless, wearing only dark bikini bottoms'),
    ('10_mid_shoulder', 'clothed', 'topless, wearing only dark bikini bottoms, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'topless, pulling down dark bikini bottoms'),
    ('12_almost_hips', 'clothed', 'topless, pulling down dark bikini bottoms, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'topless, pulling down dark bikini bottoms, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the dark bikini'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 17) Far strip: forest, white sundress (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'green forest with tree trunks and leafy ground, dappled sunlight through the trees'
SLUG = '17_forest_dress'
SEED_BASE = 2400
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a short white sundress'),
    ('02_walk', 'clothed', 'walking, wearing a short white sundress'),
    ('03_sit', 'clothed', 'sitting, wearing a short white sundress'),
    ('04_shoulder', 'clothed', 'wearing a short white sundress, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'standing among the trees, slipping a short white sundress off one shoulder'),
    ('06_loosen', 'clothed', 'standing among the trees, slipping a short white sundress off one shoulder, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, wearing white panties, holding the sundress'),
    ('08_mid_walk', 'clothed', 'walking, wearing white panties, holding the sundress'),
    ('09_mid_sit', 'clothed', 'sitting, wearing white panties, holding the sundress'),
    ('10_mid_shoulder', 'clothed', 'wearing white panties, holding the sundress, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'pulling down white panties'),
    ('12_almost_hips', 'clothed', 'pulling down white panties, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'pulling down white panties, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the white sundress'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 18) Far strip: apartment, shirt to plaid (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'classic apartment with herringbone wood parquet, a tall dark antique wardrobe, a leather pouf, large white-framed windows, soft daylight'
SLUG = '18_plaid_apartment'
SEED_BASE = 2500
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing an oversized white shirt, barefoot'),
    ('02_walk', 'clothed', 'walking, wearing an oversized white shirt, barefoot'),
    ('03_sit', 'clothed', 'sitting, wearing an oversized white shirt, barefoot'),
    ('04_shoulder', 'clothed', 'wearing an oversized white shirt, barefoot, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'unbuttoning an oversized white shirt, a red and black plaid lingerie set underneath'),
    ('06_loosen', 'clothed', 'unbuttoning an oversized white shirt, a red and black plaid lingerie set underneath, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, wearing a red and black plaid lingerie set, barefoot, the shirt off'),
    ('08_mid_walk', 'clothed', 'walking, wearing a red and black plaid lingerie set, barefoot, the shirt off'),
    ('09_mid_sit', 'clothed', 'sitting, wearing a red and black plaid lingerie set, barefoot, the shirt off'),
    ('10_mid_shoulder', 'clothed', 'wearing a red and black plaid lingerie set, barefoot, the shirt off, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'topless, pulling down the plaid panties'),
    ('12_almost_hips', 'clothed', 'topless, pulling down the plaid panties, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'topless, pulling down the plaid panties, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the white shirt'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 19) Far strip: bathtub, white robe (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'indoor bathroom with a white bathtub, a gold-frame mirror, tile floor, even indoor light'
SLUG = '19_white_tub'
SEED_BASE = 2600
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a loose white robe'),
    ('02_walk', 'clothed', 'walking, wearing a loose white robe'),
    ('03_sit', 'clothed', 'sitting, wearing a loose white robe'),
    ('04_shoulder', 'clothed', 'wearing a loose white robe, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'standing by the tub, loosening a loose white robe'),
    ('06_loosen', 'clothed', 'standing by the tub, loosening a loose white robe, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, the white robe open, otherwise nude'),
    ('08_mid_walk', 'clothed', 'walking, the white robe open, otherwise nude'),
    ('09_mid_sit', 'clothed', 'sitting, the white robe open, otherwise nude'),
    ('10_mid_shoulder', 'clothed', 'the white robe open, otherwise nude, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'the white robe slipping off her hips'),
    ('12_almost_hips', 'clothed', 'the white robe slipping off her hips, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'the white robe slipping off her hips, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the white robe'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 20) Far strip: hot tub, bikini (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'indoor spa with a bubbling hot tub, stone tile, warm lamps, light steam'
SLUG = '20_hot_tub'
SEED_BASE = 2700
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a dark two-piece bikini, wet highlighted blonde hair'),
    ('02_walk', 'clothed', 'walking, wearing a dark two-piece bikini, wet highlighted blonde hair'),
    ('03_sit', 'clothed', 'sitting, wearing a dark two-piece bikini, wet highlighted blonde hair'),
    ('04_shoulder', 'clothed', 'wearing a dark two-piece bikini, wet highlighted blonde hair, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'in the hot tub, wearing a dark two-piece bikini, untying the bikini top'),
    ('06_loosen', 'clothed', 'in the hot tub, wearing a dark two-piece bikini, untying the bikini top, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, topless in the hot tub, wearing only dark bikini bottoms'),
    ('08_mid_walk', 'clothed', 'walking, topless in the hot tub, wearing only dark bikini bottoms'),
    ('09_mid_sit', 'clothed', 'sitting, topless in the hot tub, wearing only dark bikini bottoms'),
    ('10_mid_shoulder', 'clothed', 'topless in the hot tub, wearing only dark bikini bottoms, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'topless, pulling down dark bikini bottoms at the hot tub edge'),
    ('12_almost_hips', 'clothed', 'topless, pulling down dark bikini bottoms at the hot tub edge, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'topless, pulling down dark bikini bottoms at the hot tub edge, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the dark bikini'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 21) Far strip: greenhouse, white dress (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'glass greenhouse full of green plants, soft daylight through the glass roof'
SLUG = '21_greenhouse'
SEED_BASE = 2800
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a light white summer dress, barefoot'),
    ('02_walk', 'clothed', 'walking, wearing a light white summer dress, barefoot'),
    ('03_sit', 'clothed', 'sitting, wearing a light white summer dress, barefoot'),
    ('04_shoulder', 'clothed', 'wearing a light white summer dress, barefoot, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'standing among the plants, slipping a light white summer dress down'),
    ('06_loosen', 'clothed', 'standing among the plants, slipping a light white summer dress down, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, wearing white lingerie, the dress off'),
    ('08_mid_walk', 'clothed', 'walking, wearing white lingerie, the dress off'),
    ('09_mid_sit', 'clothed', 'sitting, wearing white lingerie, the dress off'),
    ('10_mid_shoulder', 'clothed', 'wearing white lingerie, the dress off, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'topless, pulling down white panties'),
    ('12_almost_hips', 'clothed', 'topless, pulling down white panties, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'topless, pulling down white panties, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the white dress'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 22) Far strip: bedroom, black slip (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'bedroom with a made bed, a dark headboard, warm lamps, curtains half open'
SLUG = '22_bedroom_slip'
SEED_BASE = 2900
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a short black slip dress, barefoot'),
    ('02_walk', 'clothed', 'walking, wearing a short black slip dress, barefoot'),
    ('03_sit', 'clothed', 'sitting, wearing a short black slip dress, barefoot'),
    ('04_shoulder', 'clothed', 'wearing a short black slip dress, barefoot, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'standing by the bed, slipping a short black slip dress off one strap'),
    ('06_loosen', 'clothed', 'standing by the bed, slipping a short black slip dress off one strap, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, wearing a black lingerie set, the slip off'),
    ('08_mid_walk', 'clothed', 'walking, wearing a black lingerie set, the slip off'),
    ('09_mid_sit', 'clothed', 'sitting, wearing a black lingerie set, the slip off'),
    ('10_mid_shoulder', 'clothed', 'wearing a black lingerie set, the slip off, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'topless, pulling down black panties'),
    ('12_almost_hips', 'clothed', 'topless, pulling down black panties, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'topless, pulling down black panties, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the black slip'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

md(
    """## Done

Locked production LoRA:
`MyDrive/FiratSuper/loras/lapetitemilf_flux_v2.safetensors`

Run ONE far-strip cell at a time (20 pictures, about 15-30 min). Keep the tab open.

Cells 13-22 write to:
`MyDrive/FiratSuper/output/lapetitemilf/flux_eval_v2/strip_*/`

Copy keepers to:
`MyDrive/FiratSuper/keepers/`

Do not write "no scars" in prompts. Flux will draw them.
Do not train on generated pictures. Do not retrain.

Also locked:
`loras/lapetitemilf_flux.safetensors` (v1)
`loras/lapetitemilf_face.safetensors`

### Make more pictures
1. A100. Cells 1, 2, 3. New runtime: also cell 4. Then ONE of cells 13-22.
2. Cell 10: identity / lingerie / nude. Cell 12: mixed outdoor nudes.
3. Each of 13-22 is one place, far camera, 20-shot strip. If it dies, set SHOT_START and rerun that cell.
4. Nude recipe: LoRA 0.75, guidance 2.5. No scar words.
5. Adult content only. Do not train on generated pictures.

### If the runtime dies
- LoRA is already on Drive. Rerun 1, 2, 3, then the series cell. New runtime: also 4. Skip 5-9.
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
