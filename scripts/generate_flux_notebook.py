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

**Training is done for her LoRA.** Generate with locked v2 (cells 13-40). Optional new path: train a **separate** male LoRA (cells 41-45), then 5-shot genital close-ups (46-48).

**Runtime:** Runtime > Change runtime type > **A100 GPU**. Do not pick T4. Do not pick TPU.
High RAM can stay off.

**Drive:** Chrome, one Google account only (`superweb.contact@gmail.com`). In the popup: Continue, then **Allow ALL** permissions.

**Hugging Face:** Accept the license for [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and paste a READ token.

**Locked LoRA (do not overwrite):** `MyDrive/FiratSuper/loras/lapetitemilf_flux_v2.safetensors`
Also locked: `lapetitemilf_flux` (v1) and `lapetitemilf_face`. Do not retrain. Do not run cells 5-9.

**Generate path (woman LoRA only):** cells 1, 2, 3, then 4 if new runtime, then **one series cell** (13-22 far strip, 23-27 explicit sets, 28-37 far strip, or 38-40 explicit couple). Skip 5-9. Cells 13-40 still load only `lapetitemilf_flux_v2`.

**Male LoRA path (new, does not touch v2):** cells **41-45** train `henry_penis_flux_v1` from `ADD_HENRY_BODY_PHOTOS` (26 real waist-down photos, trigger `hrmale`). Then cells **46-48** are three 5-shot extreme close-ups (genital / oral / facial) with **both** LoRAs and **short** prompts that lead with penis/semen so CLIP-77 cannot drop them.

**Trigger:** `ohwx woman` (her LoRA). Male trigger: `hrmale`. Do not write "no scars" in prompts. Adult subject only.
Do not train on generated pictures. Two-person sex shots often glitch on Flux; rerun with a new SEED_BASE if anatomy breaks.
**Chest look is prompt-only.** v2 LoRA stays locked. Do not retrain v2. Do not train on gens. Use ADD_FLUX_CHEST only if he later asks for v3.

## Cells
1. A100 GPU check
2. Drive + settings
3. Hugging Face login
4. Install (needed on a fresh runtime; skip if packages are already in this session)
5-9. Training (LOCKED -- do not run)
10. Generate and SHOW pictures (identity + lingerie + nude, no filter)
11. Refine a keeper (img2img, keep frontal)
12. 20 outdoor nude styles (forest, meadow, rocks, path, bench)
13. Far strip: fireplace room, sheer black top (20)
14. Far strip: forest creek, rocks (20)
15. Far strip: gold bikini, volcanic beach (20)
16. Far strip: paddock fence, horse (20)
17. Far strip: wildflower meadow (20)
18. Far strip: damask lounge, black lace (20)
19. Far strip: blinds window, red lingerie (20)
20. Far strip: beach, straw hat (20)
21. Far strip: shoreline poses, red bikini (20)
22. Far strip: sandstone cave, camo leggings (20)
23. Explicit set: bright apartment couple (20)
24. Explicit set: tile room athletic couple (20)
25. Explicit set: POV oral, bedroom (20)
26. Explicit set: webcam ring-light POV (20)
27. Explicit set: facial close-ups (20)
28. Far strip: white bedroom, bikini top (20)
29. Far strip: white bedroom, bikini + fishnets (20)
30. Far strip: tile studio, striped socks (20)
31. Far strip: white sofa, beanie + lip tee (20)
32. Far strip: grey sofa, metal stairs (20)
33. Far strip: beige carpet, thigh-high socks (20)
34. Far strip: window bedroom, white slip (20)
35. Far strip: window bedroom, black lingerie (20)
36. Far strip: white bedroom, black lace (20)
37. Far strip: white sectional, white tee (20)
38. Explicit set: white room penis visible (20)
39. Explicit set: grey sofa semen (20)
40. Explicit set: genital close and facial (20)
41. Male LoRA: copy 26 KEEP photos + write hrmale captions
42. Male LoRA: write Ostris YAML (new filename, not v2)
43. Male LoRA: dry run (5 steps)
44. Male LoRA: full train -> henry_penis_flux_v1
45. Male LoRA: copy to Drive/loras/ (will not overwrite v2)
46. Dual LoRA 5-shot: extreme genital close-up
47. Dual LoRA 5-shot: oral on penis close-up
48. Dual LoRA 5-shot: facial, semen + glans

## Drive layout
```
MyDrive/FiratSuper/
|-- ADD_FLUX_PHOTOS/                      # 31 keepers + .txt captions
|-- ADD_FLUX_CHEST/                       # 7 chest keepers + .txt (skip unpaired)
|-- ADD_HENRY_BODY_PHOTOS/                # real male body photos (cell 41 filters to 26)
|-- loras/lapetitemilf_flux_v2.safetensors # LOCKED production LoRA
|-- loras/henry_penis_flux_v1.safetensors  # male LoRA from cells 41-45
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
HENRY_BODY_ID = "1CmFmJVtOW-a39rRndSZ4PDJc8iJIX3sm"
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
    henry = api_find_child(service, FIRATSUPER_DRIVE_ID, "ADD_HENRY_BODY_PHOTOS")
    henry_id = henry["id"] if henry else HENRY_BODY_ID
    api_download_folder(
        service,
        henry_id,
        os.path.join(local_root, "ADD_HENRY_BODY_PHOTOS"),
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
HENRY_INBOX_DIR = os.path.join(ROOT, "ADD_HENRY_BODY_PHOTOS")
LORAS_DIR = os.path.join(ROOT, "loras")
KEEPERS_DIR = os.path.join(ROOT, "keepers")
EVAL_DIR = os.path.join(ROOT, "output", PROJECT_NAME, "flux_eval_v2")
SAMPLES_DIR = os.path.join(ROOT, "output", PROJECT_NAME, "flux_samples_v2")
DATASET_DIR = "/content/dataset"
TRAIN_OUTPUT_DIR = "/content/output"
CONFIG_PATH = "/content/lapetitemilf_flux.yaml"
HENRY_LORA_NAME = "henry_penis_flux_v1"
HENRY_TRIGGER = "hrmale"
HENRY_EXPECTED = 26
HENRY_TRAIN_STEPS = 2000
HENRY_DATASET_DIR = "/content/dataset_henry"
HENRY_CONFIG_PATH = "/content/henry_penis_flux.yaml"
HENRY_OUTPUT_LORA = HENRY_LORA_NAME + ".safetensors"
if HENRY_OUTPUT_LORA in PROTECTED_LORAS:
    raise RuntimeError("Male LoRA name is on the lock list. Pick a new filename.")
if HENRY_LORA_NAME == LORA_NAME or HENRY_OUTPUT_LORA == OUTPUT_LORA_NAME:
    raise RuntimeError("Male LoRA must not reuse the locked v2 filename.")
os.makedirs(LORAS_DIR, exist_ok=True)
os.makedirs(KEEPERS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(HENRY_DATASET_DIR, exist_ok=True)
os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)

free_gb = shutil.disk_usage("/content").free / 1024**3
print("ROOT:", ROOT)
print("Inbox:", INBOX_DIR)
print("Chest:", CHEST_DIR)
print("Henry inbox:", HENRY_INBOX_DIR)
print("Local dataset:", DATASET_DIR)
print("Henry dataset:", HENRY_DATASET_DIR)
print("LoRA out:", os.path.join(LORAS_DIR, OUTPUT_LORA_NAME), "(LOCKED)")
print("Male LoRA out:", os.path.join(LORAS_DIR, HENRY_OUTPUT_LORA))
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


def ensure_flux_dual_pipe():
    # v2 stays adapter 'default'. Male file loads as adapter 'hrmale'. Cells 13-40 keep default only.
    ensure_flux_pipe()
    male_path = os.path.join(LORAS_DIR, HENRY_OUTPUT_LORA)
    if not os.path.isfile(male_path):
        local_final = os.path.join(TRAIN_OUTPUT_DIR, HENRY_LORA_NAME, HENRY_OUTPUT_LORA)
        if os.path.isfile(local_final):
            male_path = local_final
    if not os.path.isfile(male_path):
        raise RuntimeError(
            "Male LoRA not found: " + male_path +
            " Run cells 41-45 first. Do not retrain lapetitemilf_flux_v2."
        )
    if getattr(pipe, "_henry_adapter_loaded", False):
        print("Male adapter already on this pipe:", male_path)
        return pipe
    print("Loading second LoRA (male, adapter hrmale):", male_path)
    try:
        pipe.load_lora_weights(male_path, adapter_name="hrmale")
    except TypeError as err:
        raise RuntimeError(
            "This diffusers build cannot stack two LoRAs (%s). "
            "Runtime > Restart session, rerun cells 1-4, then 46." % err
        )
    pipe._henry_adapter_loaded = True
    print("Stacked LoRAs: default=%s + hrmale=%s" % (OUTPUT_LORA_NAME, HENRY_OUTPUT_LORA))
    return pipe


def write_henry_yaml(path, steps, dry):
    sample_flag = "true" if dry else "false"
    skip_first = "true"
    save_every = 10000 if dry else SAVE_EVERY
    sample_every = 10000 if dry else SAMPLE_EVERY
    lines = [
        "job: extension",
        "config:",
        '  name: "%s"' % HENRY_LORA_NAME,
        "  process:",
        "    - type: sd_trainer",
        '      training_folder: "%s"' % TRAIN_OUTPUT_DIR,
        "      device: cuda:0",
        '      trigger_word: "%s"' % HENRY_TRIGGER,
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
        '        - folder_path: "%s"' % HENRY_DATASET_DIR,
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
        '          - "hrmale, erect penis close-up, visible glans, veined shaft, photorealistic raw photo"',
        '          - "hrmale, erect penis, glans and shaft veins, waist-down close-up, photorealistic photo"',
        '          - "hrmale, side view erect penis, glans, shaft, natural skin texture, photorealistic"',
        '          - "hrmale, looking down at an erect penis, glans and veins, photorealistic raw photo"',
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
    text = chr(10).join(lines)
    if any(ord(ch) > 127 for ch in text):
        raise RuntimeError("YAML is not ASCII")
    if LORA_NAME in text or OUTPUT_LORA_NAME in text:
        raise RuntimeError("Henry YAML must not name the locked v2 LoRA.")
    with open(path, "w", encoding="ascii") as fh:
        fh.write(text)
    print("Wrote", path, "steps=%d dry=%s name=%s" % (steps, dry, HENRY_LORA_NAME))


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
        "fair pale skin, natural soft teardrop breasts, medium circular pinkish-tan textured areolae, prominent nipples, "
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


def run_scene_set(
    slug,
    place,
    shots,
    seed_base,
    shot_start=0,
    shot_end=20,
    ident=None,
    adapter_names=None,
    adapter_weights=None,
    height=1024,
    width=768,
):
    import os
    import torch
    from datetime import datetime
    from IPython.display import display
    if not SUBJECT_IS_ADULT:
        raise RuntimeError("Adult subject only.")
    nshots = len(shots)
    if nshots < 1:
        raise RuntimeError("Need at least 1 shot in this series.")
    if shot_start < 0 or shot_end > nshots or shot_start >= shot_end:
        raise RuntimeError(
            "SHOT_START/END must be inside 0..%d and START < END" % nshots
        )
    use_male = adapter_names is not None and "hrmale" in adapter_names
    if use_male:
        ensure_flux_dual_pipe()
    else:
        ensure_flux_pipe()
    if ident is None:
        ident = (
            "ohwx woman, an adult woman with long highlighted blonde hair and brown eyes, "
            "fair pale skin, natural soft teardrop breasts, medium circular pinkish-tan textured areolae, prominent nipples, "
        )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(EVAL_DIR, "scene_" + slug + "_" + stamp)
    os.makedirs(out_dir, exist_ok=True)
    print("Scene set", slug)
    print("Shots", shot_start, "to", shot_end, "->", out_dir)
    if use_male:
        print("Adapters:", adapter_names, adapter_weights)
        print("Prompts lead with genital/facial words (CLIP 77).")
    else:
        print("Two-person shots often glitch on Flux. That is the base model, not a bad LoRA.")
    print("Do not write scars or surgical in these prompts.")
    print("Keep this tab open.")
    saved = []
    for pidx in range(shot_start, shot_end):
        shot_slug, kind, action = shots[pidx]
        if kind in ("nude", "sex"):
            weight = 0.7
            guidance = 2.5
        else:
            weight = 0.9
            guidance = 3.0
        if adapter_names is None:
            names = ["default"]
            weights = [weight]
        else:
            names = list(adapter_names)
            if adapter_weights is None:
                weights = [weight] * len(names)
            else:
                weights = list(adapter_weights)
        try:
            pipe.set_adapters(names, adapter_weights=weights)
        except Exception as err:
            print("set_adapters:", err)
        seed = seed_base + pidx
        if ident:
            prompt = ident + action + ". " + place + ", photorealistic raw photo, natural skin texture"
        else:
            prompt = action + ". " + place + ", photorealistic raw photo, natural skin texture"
        print("---", shot_slug, "seed", seed, kind, "lora", names, weights)
        print(prompt)
        image = pipe(
            prompt=prompt,
            guidance_scale=guidance,
            height=height,
            width=width,
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


print("Drive settings OK. Helpers ready for cells 13-40 and 41-48.")"""
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

if isinstance(token, dict):
    picked = token.get("value")
    if not picked:
        for item in token.values():
            if isinstance(item, str) and item:
                picked = item
                break
    token = picked
token = str(token).strip() if token else token
if not token:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if isinstance(token, dict):
        picked = token.get("value")
        if not picked:
            for item in token.values():
                if isinstance(item, str) and item:
                    picked = item
                    break
        token = picked
    token = str(token).strip() if token else token
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
        "blonde hair and brown eyes, fair pale skin, looking at the camera, photorealistic raw photo, "
        "natural skin texture, sharp eyes, soft even lighting"
    ),
    "lingerie": (
        "ohwx woman, an adult woman with long highlighted blonde hair and brown eyes, "
        "fair pale skin, "
        "full body standing, black lace lingerie, looking at the camera, indoor fashion photo, "
        "photorealistic, natural skin texture, soft even lighting"
    ),
    "nude": (
        "ohwx woman, an adult woman with long highlighted blonde hair and brown eyes, "
        "fair pale skin, natural soft teardrop breasts, medium circular pinkish-tan textured areolae, prominent nipples, "
        "waist-up, facing the camera, square frontal view, standing nude, "
        "looking at the camera, soft even indoor lighting, photorealistic raw photo, natural skin texture"
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
    "fair pale skin, natural soft teardrop breasts, medium circular pinkish-tan textured areolae, prominent nipples, "
    "waist-up, facing the camera, square frontal view, standing nude, "
    "looking at the camera, soft even indoor lighting, photorealistic raw photo, natural skin texture"
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
    "fair pale skin, natural soft teardrop breasts, medium circular pinkish-tan textured areolae, prominent nipples, "
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
    r"""# @title 13) Far strip: fireplace room, sheer black top (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'dim elegant room, beige fireplace mantel packed with books and warm orange light in the hearth, dark wood floor, two glowing pendant bulbs, paneled wall'
SLUG = '13_fireplace_room'
SEED_BASE = 3000
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a sheer black short-sleeve polka-dot top, black panties, and black high-heel sandals, leaning on the fireplace mantel'),
    ('02_walk', 'clothed', 'walking, wearing a sheer black short-sleeve polka-dot top, black panties, and black high-heel sandals, leaning on the fireplace mantel'),
    ('03_sit', 'clothed', 'sitting, wearing a sheer black short-sleeve polka-dot top, black panties, and black high-heel sandals, leaning on the fireplace mantel'),
    ('04_shoulder', 'clothed', 'wearing a sheer black short-sleeve polka-dot top, black panties, and black high-heel sandals, leaning on the fireplace mantel, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'standing at the mantel, wearing a sheer black polka-dot top and black panties, sliding the panties down'),
    ('06_loosen', 'clothed', 'standing at the mantel, wearing a sheer black polka-dot top and black panties, sliding the panties down, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, wearing only a sheer black polka-dot top and black high-heel sandals, bottomless'),
    ('08_mid_walk', 'clothed', 'walking, wearing only a sheer black polka-dot top and black high-heel sandals, bottomless'),
    ('09_mid_sit', 'clothed', 'sitting, wearing only a sheer black polka-dot top and black high-heel sandals, bottomless'),
    ('10_mid_shoulder', 'clothed', 'wearing only a sheer black polka-dot top and black high-heel sandals, bottomless, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'taking off the sheer black polka-dot top, otherwise nude, still in black high-heel sandals'),
    ('12_almost_hips', 'clothed', 'taking off the sheer black polka-dot top, otherwise nude, still in black high-heel sandals, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'taking off the sheer black polka-dot top, otherwise nude, still in black high-heel sandals, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the sheer black top'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 14) Far strip: forest creek, rocks (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'shallow rocky forest creek, large grey rocks, clear water, dense yellow-green trees, dappled sunlight'
SLUG = '14_forest_creek'
SEED_BASE = 3100
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a simple earth-tone sundress, sitting on a rock in the shallow creek'),
    ('02_walk', 'clothed', 'walking, wearing a simple earth-tone sundress, sitting on a rock in the shallow creek'),
    ('03_sit', 'clothed', 'sitting, wearing a simple earth-tone sundress, sitting on a rock in the shallow creek'),
    ('04_shoulder', 'clothed', 'wearing a simple earth-tone sundress, sitting on a rock in the shallow creek, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'standing in the shallow creek, slipping an earth-tone sundress off one shoulder'),
    ('06_loosen', 'clothed', 'standing in the shallow creek, slipping an earth-tone sundress off one shoulder, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, topless, a wrap of fabric at her hips, sitting on the rocks in the water'),
    ('08_mid_walk', 'clothed', 'walking, topless, a wrap of fabric at her hips, sitting on the rocks in the water'),
    ('09_mid_sit', 'clothed', 'sitting, topless, a wrap of fabric at her hips, sitting on the rocks in the water'),
    ('10_mid_shoulder', 'clothed', 'topless, a wrap of fabric at her hips, sitting on the rocks in the water, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'the fabric around her thighs, sitting on rocks in the creek, otherwise nude'),
    ('12_almost_hips', 'clothed', 'the fabric around her thighs, sitting on rocks in the creek, otherwise nude, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'the fabric around her thighs, sitting on rocks in the creek, otherwise nude, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the sundress'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 15) Far strip: gold bikini, volcanic beach (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'tropical beach, bright white sand, dark volcanic rocks, turquoise ocean, clear blue sky, hard sunlight'
SLUG = '15_gold_bikini_beach'
SEED_BASE = 3200
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a shiny metallic gold bandeau bikini and gold cuff bracelets, kneeling on the white sand'),
    ('02_walk', 'clothed', 'walking, wearing a shiny metallic gold bandeau bikini and gold cuff bracelets, kneeling on the white sand'),
    ('03_sit', 'clothed', 'sitting, wearing a shiny metallic gold bandeau bikini and gold cuff bracelets, kneeling on the white sand'),
    ('04_shoulder', 'clothed', 'wearing a shiny metallic gold bandeau bikini and gold cuff bracelets, kneeling on the white sand, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'kneeling on the sand in a shiny metallic gold bikini, untying the gold bikini top'),
    ('06_loosen', 'clothed', 'kneeling on the sand in a shiny metallic gold bikini, untying the gold bikini top, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, topless, wearing only shiny metallic gold bikini bottoms and gold bracelets, kneeling on the sand'),
    ('08_mid_walk', 'clothed', 'walking, topless, wearing only shiny metallic gold bikini bottoms and gold bracelets, kneeling on the sand'),
    ('09_mid_sit', 'clothed', 'sitting, topless, wearing only shiny metallic gold bikini bottoms and gold bracelets, kneeling on the sand'),
    ('10_mid_shoulder', 'clothed', 'topless, wearing only shiny metallic gold bikini bottoms and gold bracelets, kneeling on the sand, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'topless, pulling down shiny metallic gold bikini bottoms on the sand'),
    ('12_almost_hips', 'clothed', 'topless, pulling down shiny metallic gold bikini bottoms on the sand, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'topless, pulling down shiny metallic gold bikini bottoms on the sand, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the gold bikini'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 16) Far strip: paddock fence, horse (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'rural paddock, weathered wooden fence, green pasture, a brown horse behind the fence, warm golden-hour light'
SLUG = '16_paddock_horse'
SEED_BASE = 3300
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a rust-orange lace bralette and unbuttoned blue jeans, leaning on the wooden fence, a brown horse behind her'),
    ('02_walk', 'clothed', 'walking, wearing a rust-orange lace bralette and unbuttoned blue jeans, leaning on the wooden fence, a brown horse behind her'),
    ('03_sit', 'clothed', 'sitting, wearing a rust-orange lace bralette and unbuttoned blue jeans, leaning on the wooden fence, a brown horse behind her'),
    ('04_shoulder', 'clothed', 'wearing a rust-orange lace bralette and unbuttoned blue jeans, leaning on the wooden fence, a brown horse behind her, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'leaning on the fence in a rust-orange lace bralette, unzipping blue jeans'),
    ('06_loosen', 'clothed', 'leaning on the fence in a rust-orange lace bralette, unzipping blue jeans, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, wearing a rust-orange lace bralette, jeans pulled open at the hips'),
    ('08_mid_walk', 'clothed', 'walking, wearing a rust-orange lace bralette, jeans pulled open at the hips'),
    ('09_mid_sit', 'clothed', 'sitting, wearing a rust-orange lace bralette, jeans pulled open at the hips'),
    ('10_mid_shoulder', 'clothed', 'wearing a rust-orange lace bralette, jeans pulled open at the hips, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'topless, pushing the jeans down her thighs at the fence'),
    ('12_almost_hips', 'clothed', 'topless, pushing the jeans down her thighs at the fence, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'topless, pushing the jeans down her thighs at the fence, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the jeans'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 17) Far strip: wildflower meadow (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'tall wildflower meadow with small white daisies, dark green forest behind, soft daylight'
SLUG = '17_wildflower_meadow'
SEED_BASE = 3400
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a light linen sundress and a chunky brown wooden-bead necklace, standing in the wildflowers'),
    ('02_walk', 'clothed', 'walking, wearing a light linen sundress and a chunky brown wooden-bead necklace, standing in the wildflowers'),
    ('03_sit', 'clothed', 'sitting, wearing a light linen sundress and a chunky brown wooden-bead necklace, standing in the wildflowers'),
    ('04_shoulder', 'clothed', 'wearing a light linen sundress and a chunky brown wooden-bead necklace, standing in the wildflowers, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'standing in the wildflowers, slipping a light linen sundress off one shoulder, chunky brown wooden-bead necklace'),
    ('06_loosen', 'clothed', 'standing in the wildflowers, slipping a light linen sundress off one shoulder, chunky brown wooden-bead necklace, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, topless in the wildflowers, fabric at her hips, chunky brown wooden-bead necklace'),
    ('08_mid_walk', 'clothed', 'walking, topless in the wildflowers, fabric at her hips, chunky brown wooden-bead necklace'),
    ('09_mid_sit', 'clothed', 'sitting, topless in the wildflowers, fabric at her hips, chunky brown wooden-bead necklace'),
    ('10_mid_shoulder', 'clothed', 'topless in the wildflowers, fabric at her hips, chunky brown wooden-bead necklace, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'the sundress around her thighs in the wildflowers, chunky brown wooden-bead necklace, otherwise nude'),
    ('12_almost_hips', 'clothed', 'the sundress around her thighs in the wildflowers, chunky brown wooden-bead necklace, otherwise nude, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'the sundress around her thighs in the wildflowers, chunky brown wooden-bead necklace, otherwise nude, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the linen sundress'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 18) Far strip: damask lounge, black lace (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'upscale lounge, tan damask wallpaper, a beige floral sofa with lavender pillows, light carpet, warm indoor light'
SLUG = '18_damask_lounge'
SEED_BASE = 3500
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a black lace harness lingerie set, a garter belt, and sheer black thigh-high stockings, kneeling by the sofa'),
    ('02_walk', 'clothed', 'walking, wearing a black lace harness lingerie set, a garter belt, and sheer black thigh-high stockings, kneeling by the sofa'),
    ('03_sit', 'clothed', 'sitting, wearing a black lace harness lingerie set, a garter belt, and sheer black thigh-high stockings, kneeling by the sofa'),
    ('04_shoulder', 'clothed', 'wearing a black lace harness lingerie set, a garter belt, and sheer black thigh-high stockings, kneeling by the sofa, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'kneeling by the sofa in black lace lingerie and stockings, unhooking the harness bra'),
    ('06_loosen', 'clothed', 'kneeling by the sofa in black lace lingerie and stockings, unhooking the harness bra, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, topless, wearing a black garter belt, black panties, and sheer black thigh-high stockings'),
    ('08_mid_walk', 'clothed', 'walking, topless, wearing a black garter belt, black panties, and sheer black thigh-high stockings'),
    ('09_mid_sit', 'clothed', 'sitting, topless, wearing a black garter belt, black panties, and sheer black thigh-high stockings'),
    ('10_mid_shoulder', 'clothed', 'topless, wearing a black garter belt, black panties, and sheer black thigh-high stockings, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'topless in stockings, pulling down black panties by the sofa'),
    ('12_almost_hips', 'clothed', 'topless in stockings, pulling down black panties by the sofa, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'topless in stockings, pulling down black panties by the sofa, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the black lace lingerie'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 19) Far strip: blinds window, red lingerie (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'bright indoor room, floor-to-ceiling windows with white horizontal blinds, light tile floor, daylight through the slats'
SLUG = '19_blinds_red'
SEED_BASE = 3600
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a red lingerie set, black-rimmed glasses, a silver pendant necklace, and black platform heels'),
    ('02_walk', 'clothed', 'walking, wearing a red lingerie set, black-rimmed glasses, a silver pendant necklace, and black platform heels'),
    ('03_sit', 'clothed', 'sitting, wearing a red lingerie set, black-rimmed glasses, a silver pendant necklace, and black platform heels'),
    ('04_shoulder', 'clothed', 'wearing a red lingerie set, black-rimmed glasses, a silver pendant necklace, and black platform heels, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'standing by the blinds in a red lingerie set and black-rimmed glasses, untying the red panty bows'),
    ('06_loosen', 'clothed', 'standing by the blinds in a red lingerie set and black-rimmed glasses, untying the red panty bows, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, topless, wearing only red panties, black-rimmed glasses, and black platform heels'),
    ('08_mid_walk', 'clothed', 'walking, topless, wearing only red panties, black-rimmed glasses, and black platform heels'),
    ('09_mid_sit', 'clothed', 'sitting, topless, wearing only red panties, black-rimmed glasses, and black platform heels'),
    ('10_mid_shoulder', 'clothed', 'topless, wearing only red panties, black-rimmed glasses, and black platform heels, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'topless, pulling down red panties, black-rimmed glasses and black heels still on'),
    ('12_almost_hips', 'clothed', 'topless, pulling down red panties, black-rimmed glasses and black heels still on, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'topless, pulling down red panties, black-rimmed glasses and black heels still on, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the red lingerie'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 20) Far strip: beach, straw hat (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'tropical beach, white sand, turquoise ocean with small waves, a rocky headland, bright blue sky, midday sun'
SLUG = '20_beach_hat'
SEED_BASE = 3700
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a dark swimsuit and holding a wide-brim straw hat with a black ribbon'),
    ('02_walk', 'clothed', 'walking, wearing a dark swimsuit and holding a wide-brim straw hat with a black ribbon'),
    ('03_sit', 'clothed', 'sitting, wearing a dark swimsuit and holding a wide-brim straw hat with a black ribbon'),
    ('04_shoulder', 'clothed', 'wearing a dark swimsuit and holding a wide-brim straw hat with a black ribbon, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'standing on the sand, slipping off a dark swimsuit while holding a wide-brim straw hat'),
    ('06_loosen', 'clothed', 'standing on the sand, slipping off a dark swimsuit while holding a wide-brim straw hat, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, topless, wearing swimsuit bottoms, holding a wide-brim straw hat in front of her'),
    ('08_mid_walk', 'clothed', 'walking, topless, wearing swimsuit bottoms, holding a wide-brim straw hat in front of her'),
    ('09_mid_sit', 'clothed', 'sitting, topless, wearing swimsuit bottoms, holding a wide-brim straw hat in front of her'),
    ('10_mid_shoulder', 'clothed', 'topless, wearing swimsuit bottoms, holding a wide-brim straw hat in front of her, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'otherwise nude, holding a wide-brim straw hat against her body for cover'),
    ('12_almost_hips', 'clothed', 'otherwise nude, holding a wide-brim straw hat against her body for cover, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'otherwise nude, holding a wide-brim straw hat against her body for cover, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the straw hat'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 21) Far strip: shoreline poses, red bikini (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'sandy shoreline with small waves, wet sand, ocean and a rocky cliff in the distance, bright daylight'
SLUG = '21_shore_poses'
SEED_BASE = 3800
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a small red triangle bikini, standing at the waterline'),
    ('02_walk', 'clothed', 'walking, wearing a small red triangle bikini, standing at the waterline'),
    ('03_sit', 'clothed', 'sitting, wearing a small red triangle bikini, standing at the waterline'),
    ('04_shoulder', 'clothed', 'wearing a small red triangle bikini, standing at the waterline, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'standing in the shallow waves in a small red triangle bikini, untying the red bikini top'),
    ('06_loosen', 'clothed', 'standing in the shallow waves in a small red triangle bikini, untying the red bikini top, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, topless, wearing only red bikini bottoms, kneeling far back on the wet sand'),
    ('08_mid_walk', 'clothed', 'walking, topless, wearing only red bikini bottoms, kneeling far back on the wet sand'),
    ('09_mid_sit', 'clothed', 'sitting, topless, wearing only red bikini bottoms, kneeling far back on the wet sand'),
    ('10_mid_shoulder', 'clothed', 'topless, wearing only red bikini bottoms, kneeling far back on the wet sand, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'topless, pulling down red bikini bottoms at the waterline'),
    ('12_almost_hips', 'clothed', 'topless, pulling down red bikini bottoms at the waterline, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'topless, pulling down red bikini bottoms at the waterline, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the red bikini'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 22) Far strip: sandstone cave, camo leggings (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'shallow sandstone cave, textured light-brown rock walls, fine sand floor, soft daylight from the cave mouth'
SLUG = '22_sandstone_cave'
SEED_BASE = 3900
SHOTS = [
    ('01_stand', 'clothed', 'standing, wearing a simple dark tank top and grey-green camouflage leggings, barefoot on the sand'),
    ('02_walk', 'clothed', 'walking, wearing a simple dark tank top and grey-green camouflage leggings, barefoot on the sand'),
    ('03_sit', 'clothed', 'sitting, wearing a simple dark tank top and grey-green camouflage leggings, barefoot on the sand'),
    ('04_shoulder', 'clothed', 'wearing a simple dark tank top and grey-green camouflage leggings, barefoot on the sand, looking back over her shoulder at the camera'),
    ('05_start', 'clothed', 'standing in the cave, lifting a dark tank top, camouflage leggings on'),
    ('06_loosen', 'clothed', 'standing in the cave, lifting a dark tank top, camouflage leggings on, the garment loosening'),
    ('07_mid_stand', 'clothed', 'standing, topless, wearing grey-green camouflage leggings pulled slightly down on her hips, barefoot'),
    ('08_mid_walk', 'clothed', 'walking, topless, wearing grey-green camouflage leggings pulled slightly down on her hips, barefoot'),
    ('09_mid_sit', 'clothed', 'sitting, topless, wearing grey-green camouflage leggings pulled slightly down on her hips, barefoot'),
    ('10_mid_shoulder', 'clothed', 'topless, wearing grey-green camouflage leggings pulled slightly down on her hips, barefoot, looking back over her shoulder at the camera'),
    ('11_almost_pull', 'clothed', 'topless, pushing camouflage leggings down her thighs in the cave'),
    ('12_almost_hips', 'clothed', 'topless, pushing camouflage leggings down her thighs in the cave, pulled to her hips'),
    ('13_almost_thighs', 'clothed', 'topless, pushing camouflage leggings down her thighs in the cave, around her thighs'),
    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),
    ('15_hold', 'nude', 'nude, holding the camouflage leggings'),
    ('16_nude_stand', 'nude', 'standing nude'),
    ('17_nude_walk', 'nude', 'walking nude'),
    ('18_nude_sit', 'nude', 'sitting nude'),
    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),
    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),
]
run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)


code(
    r"""# @title 23) Explicit set: bright apartment couple (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'modern bright apartment, white walls, large windows, a white bed, a light rug, daylight'
SLUG = '23_apt_couple'
SEED_BASE = 4000
SHOTS = [
    ('01_stand_panties', 'clothed', 'standing next to an adult man whose face is out of frame, she wears light-blue lace panties, topless, looking at the camera'),
    ('02_stand_nude', 'nude', 'standing nude next to an adult man whose face is out of frame, close together, looking at the camera'),
    ('03_hold_him', 'sex', 'standing nude, holding his erect penis with both hands, looking at the camera, an adult man whose face is out of frame'),
    ('04_kiss_chest', 'sex', 'standing nude, kissing his chest, an adult man whose face is out of frame, daylight from large windows'),
    ('05_bed_back', 'sex', 'lying on her back nude on the white bed, an adult man whose face is out of frame between her legs, looking at the camera'),
    ('06_missionary', 'sex', 'missionary sex on the white bed, she looks at the camera, an adult man whose face is out of frame'),
    ('07_oral_bed', 'sex', 'on the white bed performing oral sex, side view, an adult man whose face is out of frame, her face visible'),
    ('08_cowgirl', 'sex', 'sitting on top of an adult man whose face is out of frame on the white bed, facing the camera, legs apart, nude'),
    ('09_cowgirl_lean', 'sex', 'sitting on top of an adult man whose face is out of frame on the white bed, leaning forward, looking at the camera'),
    ('10_from_behind', 'sex', 'on the white bed on all fours, an adult man whose face is out of frame behind her, she looks back at the camera'),
    ('11_standing_sex', 'sex', 'standing sex by the window, she looks at the camera, an adult man whose face is out of frame behind her'),
    ('12_pov_oral', 'sex', 'POV oral sex, camera from his point of view, she looks up at the camera, kneeling, nude'),
    ('13_oral_close', 'sex', 'close-up of her face performing oral sex, looking at the camera, an adult man whose face is out of frame'),
    ('14_kneel_ready', 'sex', 'kneeling nude on the rug, looking up, an adult man whose face is out of frame standing in front of her'),
    ('15_facial_start', 'sex', 'kneeling nude, semen landing on her face, looking at the camera, an adult man whose face is out of frame'),
    ('16_facial_smear', 'sex', 'close-up of her face with semen on her cheeks and lips, looking at the camera'),
    ('17_facial_smile', 'sex', 'close-up, she smiles at the camera with semen on her face and chin'),
    ('18_after_sit', 'nude', 'sitting nude on the white bed after sex, semen on her chest, looking at the camera'),
    ('19_wide_bed', 'sex', 'wide shot of her and an adult man whose face is out of frame on the white bed, she is nude, daylight'),
    ('20_face_close', 'sex', 'tight close-up of her face, highlighted blonde hair, brown eyes, semen on her lips, looking at the camera'),
]
run_scene_set(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 24) Explicit set: tile room athletic couple (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'indoor room with light tile floor, beige walls, a dark couch with red and black pillows, a potted plant, even indoor light'
SLUG = '24_tile_athletic'
SEED_BASE = 4100
SHOTS = [
    ('01_shorts_lookback', 'clothed', 'standing, wearing a yellow sports bra and blue-and-white athletic shorts, looking back over her shoulder, an adult man whose face is out of frame behind her'),
    ('02_lift_top', 'clothed', 'lifting off a yellow sports bra, still wearing blue-and-white athletic shorts, looking at the camera'),
    ('03_kneel_hold', 'sex', 'kneeling on the tile floor, topless, blue shorts on, holding his erect penis, looking down, an adult man whose face is out of frame'),
    ('04_kneel_rear', 'clothed', 'kneeling, topless, blue shorts on, an adult man whose face is out of frame hand on her hip, she looks back'),
    ('05_topless_smile', 'clothed', 'medium shot, topless, blue shorts on, slight smile, looking at the camera'),
    ('06_shorts_down_rear', 'sex', 'rear view, topless, blue shorts pulled down, looking back, an adult man whose face is out of frame in the foreground POV'),
    ('07_on_lap', 'sex', 'sitting on his lap, topless, shorts down, looking back over her shoulder, an adult man whose face is out of frame'),
    ('08_face_open', 'sex', 'close-up of her face and topless chest, mouth open, looking at the camera'),
    ('09_oral_tongue', 'sex', 'extreme close-up, she performs oral sex, tongue out, an adult man whose face is out of frame'),
    ('10_oral_eye_contact', 'sex', 'close-up oral sex, she looks into the camera, an adult man whose face is out of frame'),
    ('11_profile', 'sex', 'close-up profile of her face looking toward the camera, an adult man whose face is out of frame nearby'),
    ('12_nude_lookback', 'sex', 'full body rear, completely nude, shorts at her ankles, looking back, an adult man whose face is out of frame POV in the foreground'),
    ('13_kneel_nude', 'sex', 'kneeling nude on the tile, looking at the camera, an adult man whose face is out of frame in front of her'),
    ('14_facial_kneel', 'sex', 'kneeling nude, semen on her face and chest, looking at the camera, an adult man whose face is out of frame'),
    ('15_facial_more', 'sex', 'kneeling, more semen on her face, mouth open, looking at the camera'),
    ('16_facial_smile', 'sex', 'close-up, she smiles at the camera with semen covering her face, hands framing her chin'),
    ('17_shorts_off', 'nude', 'standing nude, blue shorts on the floor, looking at the camera, couch behind her'),
    ('18_couch_sit', 'sex', 'sitting nude on the dark couch, an adult man whose face is out of frame beside her, looking at the camera'),
    ('19_wide_room', 'sex', 'wide shot of the tile room, she kneels nude, an adult man whose face is out of frame standing, plant in the background'),
    ('20_face_end', 'sex', 'tight close-up of her face, highlighted blonde hair, brown eyes, semen on her lips, smiling'),
]
run_scene_set(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 25) Explicit set: POV oral, bedroom (20)
SHOT_START = 0
SHOT_END = 20
PLACE = "bedroom, a bed with rumpled sheets, indoor light, camera often from the man's point of view"
SLUG = '25_pov_oral'
SEED_BASE = 4200
SHOTS = [
    ('01_kneel_clothed', 'clothed', 'kneeling on a bed, wearing a black tank top, POV oral sex, looking up at the camera, an adult man whose face is out of frame'),
    ('02_side_oral', 'sex', 'side view oral sex on a bed, her face in profile, an adult man whose face is out of frame'),
    ('03_pov_eye', 'sex', 'POV from his point of view, she performs oral sex and looks into the camera'),
    ('04_hair_hold', 'sex', 'POV oral sex, his hand in her highlighted blonde hair, she looks up'),
    ('05_topless_kneel', 'sex', 'kneeling topless on the bed, performing oral sex, looking at the camera, an adult man whose face is out of frame'),
    ('06_both_hands', 'sex', 'she holds his penis with both hands near her mouth, looking at the camera, an adult man whose face is out of frame'),
    ('07_close_mouth', 'sex', 'close-up of her mouth on his penis, her brown eyes looking up, an adult man whose face is out of frame'),
    ('08_outdoor_pov', 'sex', 'outdoor POV oral sex on a lounge chair, she looks up at the camera, an adult man whose face is out of frame'),
    ('09_couch_lean', 'sex', 'leaning over a couch, performing oral sex, looking at the camera, an adult man whose face is out of frame'),
    ('10_smile_hold', 'sex', 'she smiles at the camera while holding his penis, kneeling, an adult man whose face is out of frame'),
    ('11_deep', 'sex', 'POV oral sex, closer to her face, she looks up, an adult man whose face is out of frame'),
    ('12_nude_kneel', 'sex', 'kneeling fully nude, performing oral sex, looking at the camera, an adult man whose face is out of frame'),
    ('13_profile_open', 'sex', 'profile close-up, her mouth open next to his penis, an adult man whose face is out of frame'),
    ('14_two_hands_stroke', 'sex', 'lying on her stomach on the bed, stroking him with both hands, smiling at the camera, an adult man whose face is out of frame'),
    ('15_facial_pov', 'sex', 'POV, semen on her lips, she looks up at the camera, an adult man whose face is out of frame'),
    ('16_cheek', 'sex', 'close-up, semen on her cheek and mouth, looking at the camera'),
    ('17_tongue', 'sex', 'close-up, tongue out, semen on her tongue, looking at the camera'),
    ('18_after_sit', 'nude', 'sitting on the bed after, wiping her mouth, looking at the camera'),
    ('19_wide_bed', 'sex', 'wider bedroom shot, she kneels nude on the bed, an adult man whose face is out of frame in front of her'),
    ('20_face_end', 'sex', 'tight close-up of her face, highlighted blonde hair, brown eyes, looking at the camera'),
]
run_scene_set(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 26) Explicit set: webcam ring-light POV (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'amateur bedroom, white bed, a black ring light on a stand, a large mirror, warm direct light'
SLUG = '26_webcam_ring'
SEED_BASE = 4300
SHOTS = [
    ('01_crop_smile', 'clothed', 'lying on a white bed, wearing a black long-sleeve crop top and a black thong, smiling at the camera, ring light behind her, an adult man whose face is out of frame POV'),
    ('02_two_hands', 'sex', 'on the white bed in a black crop top and black thong, stroking him with both hands, smiling, ring light, an adult man whose face is out of frame POV'),
    ('03_oral_webcam', 'sex', 'webcam POV oral sex on the white bed, ring light and a mirror in the background, she looks at the camera'),
    ('04_thong_look', 'sex', 'on her stomach, black thong, looking at the camera while holding him, an adult man whose face is out of frame POV'),
    ('05_top_off', 'sex', 'crop top off, black thong on, kneeling on the white bed, looking at the camera, ring light'),
    ('06_nude_stroke', 'sex', 'nude on the white bed, stroking him, smiling at the camera, ring light, an adult man whose face is out of frame POV'),
    ('07_oral_close', 'sex', 'close-up oral sex, ring light bokeh behind her, she looks up'),
    ('08_mirror', 'sex', 'shot that includes the mirror, she performs oral sex on the bed, ring light visible'),
    ('09_from_above', 'sex', 'high angle, she kneels on the white bed performing oral sex, looking up, an adult man whose face is out of frame'),
    ('10_side_bed', 'sex', 'side of the bed, she performs oral sex, amateur bedroom light, an adult man whose face is out of frame'),
    ('11_thong_off', 'nude', 'taking off the black thong on the white bed, looking at the camera, ring light'),
    ('12_all_fours', 'sex', 'on all fours on the white bed, looking at the camera, an adult man whose face is out of frame behind her'),
    ('13_cowgirl_cam', 'sex', 'sitting on an adult man whose face is out of frame on the white bed, facing the webcam, ring light'),
    ('14_facial_bed', 'sex', 'kneeling on the white bed, semen on her face, smiling, ring light, an adult man whose face is out of frame'),
    ('15_facial_close', 'sex', 'close-up, semen on her face, ring light glow, looking at the camera'),
    ('16_laugh', 'sex', 'she laughs at the camera with semen on her chin, white bed, ring light'),
    ('17_wipe', 'sex', 'wiping semen with her fingers, looking at the camera, white bed'),
    ('18_lie_after', 'nude', 'lying on the white bed nude after, looking at the camera, ring light still on'),
    ('19_wide_room', 'sex', 'wide amateur bedroom, white bed, ring light stand, mirror, she kneels nude, an adult man whose face is out of frame'),
    ('20_face_end', 'sex', 'tight close-up of her face, highlighted blonde hair, brown eyes, looking at the camera'),
]
run_scene_set(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 27) Explicit set: facial close-ups (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'simple indoor close-up, mostly her face, dark or plain background, bright directional light'
SLUG = '27_facial_close'
SEED_BASE = 4400
SHOTS = [
    ('01_clean_close', 'nude', 'tight close-up portrait, looking at the camera, no extra people'),
    ('02_mouth_open', 'sex', 'close-up, mouth open, looking up, an adult man whose face is out of frame just out of focus'),
    ('03_tongue', 'sex', 'close-up, tongue out, looking at the camera'),
    ('04_penis_near', 'sex', 'close-up of her face next to his erect penis, looking at the camera, an adult man whose face is out of frame'),
    ('05_first_drop', 'sex', 'close-up, first drop of semen on her lips, looking at the camera'),
    ('06_cheek', 'sex', 'close-up, semen on her cheek and jaw, looking at the camera'),
    ('07_forehead', 'sex', 'close-up, semen on her forehead and nose, looking at the camera'),
    ('08_mouth', 'sex', 'close-up, semen in and around her mouth, looking at the camera'),
    ('09_eyes_closed', 'sex', 'close-up, eyes closed, semen on her face, head tilted back'),
    ('10_smile', 'sex', 'close-up, she smiles with semen on her face, looking at the camera'),
    ('11_high_angle', 'sex', 'high angle looking down at her face, semen on her face, looking up'),
    ('12_both_hands', 'sex', 'close-up, her hands near her face, semen on fingers and lips'),
    ('13_heavy', 'sex', 'close-up, face heavily covered in semen, looking at the camera'),
    ('14_profile', 'sex', 'profile close-up, semen on her cheek and mouth, dark background'),
    ('15_tongue_out', 'sex', 'close-up, tongue out with semen, brown eyes looking at the camera'),
    ('16_after_drip', 'sex', 'close-up, semen dripping from her chin, looking at the camera'),
    ('17_hair', 'sex', 'close-up, highlighted blonde hair messy, semen on her lips, looking at the camera'),
    ('18_soft_smile', 'sex', 'close-up, softer light, semen on her face, slight smile'),
    ('19_wipe_lip', 'sex', 'close-up, wiping her lip with a thumb, semen still on her face'),
    ('20_final', 'sex', 'tight close-up of her face, highlighted blonde hair, brown eyes, semen on her lips, looking at the camera'),
]
run_scene_set(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)


def far_strip_cell(num, short_title, slug, seed_base, place, clothed, start, mid, almost, hold):
    code(
        "# @title %d) Far strip: %s (20)\n"
        "SHOT_START = 0\n"
        "SHOT_END = 20\n"
        "PLACE = %r\n"
        "SLUG = %r\n"
        "SEED_BASE = %d\n"
        "SHOTS = [\n"
        "    ('01_stand', 'clothed', 'standing, wearing %s'),\n"
        "    ('02_walk', 'clothed', 'walking, wearing %s'),\n"
        "    ('03_sit', 'clothed', 'sitting, wearing %s'),\n"
        "    ('04_shoulder', 'clothed', 'wearing %s, looking back over her shoulder at the camera'),\n"
        "    ('05_start', 'clothed', '%s'),\n"
        "    ('06_loosen', 'clothed', '%s, the garment loosening'),\n"
        "    ('07_mid_stand', 'clothed', 'standing, %s'),\n"
        "    ('08_mid_walk', 'clothed', 'walking, %s'),\n"
        "    ('09_mid_sit', 'clothed', 'sitting, %s'),\n"
        "    ('10_mid_shoulder', 'clothed', '%s, looking back over her shoulder at the camera'),\n"
        "    ('11_almost_pull', 'clothed', '%s'),\n"
        "    ('12_almost_hips', 'clothed', '%s, pulled to her hips'),\n"
        "    ('13_almost_thighs', 'clothed', '%s, around her thighs'),\n"
        "    ('14_step_out', 'clothed', 'stepping out of the last garment, otherwise nude'),\n"
        "    ('15_hold', 'nude', 'nude, holding %s'),\n"
        "    ('16_nude_stand', 'nude', 'standing nude'),\n"
        "    ('17_nude_walk', 'nude', 'walking nude'),\n"
        "    ('18_nude_sit', 'nude', 'sitting nude'),\n"
        "    ('19_nude_shoulder', 'nude', 'nude, looking back over her shoulder at the camera'),\n"
        "    ('20_nude_farther', 'nude', 'standing nude, camera even farther back'),\n"
        "]\n"
        "run_far_strip(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"
        % (
            num,
            short_title,
            place,
            slug,
            seed_base,
            clothed,
            clothed,
            clothed,
            clothed,
            start,
            start,
            mid,
            mid,
            mid,
            mid,
            almost,
            almost,
            almost,
            hold,
        )
    )


WHITE_BED = (
    "bright white bedroom, a white bed, a small white bedside table with a vase of yellow and blue flowers, bright indoor daylight"
)
WINDOW_BED = (
    "bright modern bedroom, a white tufted headboard, floor-to-ceiling windows, daylight, white bedding"
)

far_strip_cell(
    28,
    "white bedroom, bikini top",
    "28_white_bed_bikini",
    4500,
    WHITE_BED,
    "a blue-and-white patterned bikini top, matching bikini bottoms, and a thin silver pendant necklace",
    "standing by the white bed, untying the blue-and-white patterned bikini top, thin silver pendant necklace",
    "topless, wearing matching bikini bottoms and a thin silver pendant necklace",
    "topless, pulling down the bikini bottoms, thin silver pendant necklace",
    "the bikini",
)
far_strip_cell(
    29,
    "white bedroom, bikini + fishnets",
    "29_white_bed_fishnets",
    4600,
    WHITE_BED,
    "a blue-and-white patterned bikini top, matching bikini bottoms, a thin silver pendant necklace, and black fishnet stockings",
    "standing by the white bed, untying the blue-and-white patterned bikini top, black fishnet stockings",
    "topless, wearing matching bikini bottoms, a thin silver pendant necklace, and black fishnet stockings",
    "topless, pulling down the bikini bottoms over black fishnet stockings",
    "the bikini",
)
far_strip_cell(
    30,
    "tile studio, striped socks",
    "30_studio_socks",
    4700,
    "bright studio, light tile floor, a large plush blue rug, light wood paneled walls, even studio daylight",
    "a blue bikini top, a blue bikini bottom, and blue-and-white striped knee-high socks",
    "standing on the blue rug, untying the blue bikini top, blue-and-white striped knee-high socks",
    "topless, wearing a blue bikini bottom and blue-and-white striped knee-high socks",
    "topless, pulling down the blue bikini bottom, blue-and-white striped knee-high socks still on",
    "the bikini",
)
far_strip_cell(
    31,
    "white sofa, beanie + lip tee",
    "31_beanie_liptee",
    4800,
    "bright room, large gray floor tiles, a large white leather sofa, even daylight",
    "a red knit beanie, a white t-shirt with a big red lip graphic, and panties",
    "standing by the white leather sofa, lifting the white lip-graphic t-shirt, red knit beanie on",
    "topless, wearing a red knit beanie and panties",
    "topless, pulling down panties, red knit beanie still on",
    "the t-shirt",
)
far_strip_cell(
    32,
    "grey sofa, metal stairs",
    "32_grey_sofa_stairs",
    4900,
    "modern apartment, a large grey sofa, a metal stair railing, indoor leafy plants, dark grey tile floor, bright indoor light",
    "a simple black tank top and black panties",
    "standing by the grey sofa, lifting the black tank top",
    "topless, wearing black panties",
    "topless, pulling down black panties",
    "the black tank",
)
far_strip_cell(
    33,
    "beige carpet, thigh-high socks",
    "33_thighhigh_carpet",
    5000,
    "bedroom with tan beige carpet, rumpled light sheets, a dark leather tufted headboard, beige walls, indoor daylight",
    "a small white top, white thigh-high socks with pink stripes, and panties",
    "standing by the bed, slipping the small white top off one shoulder, white thigh-high socks with pink stripes",
    "topless, wearing white thigh-high socks with pink stripes and panties",
    "topless, pulling down panties, white thigh-high socks with pink stripes still on",
    "the white top",
)
far_strip_cell(
    34,
    "window bedroom, white slip",
    "34_window_slip",
    5100,
    WINDOW_BED,
    "a simple white slip",
    "standing by the windows, slipping a white slip off one shoulder",
    "topless, the white slip at her hips",
    "the white slip around her thighs, otherwise nude",
    "the white slip",
)
far_strip_cell(
    35,
    "window bedroom, black lingerie",
    "35_window_sofa",
    5200,
    "bright modern bedroom, a white tufted headboard, floor-to-ceiling windows, a white leather sofa, daylight, white bedding",
    "a black lingerie set",
    "standing by the windows, unhooking the black lingerie bra",
    "topless, wearing black panties",
    "topless, pulling down black panties",
    "the black lingerie",
)
far_strip_cell(
    36,
    "white bedroom, black lace",
    "36_flower_lace",
    5300,
    "bright white bedroom, a white bed, a prominent vase of yellow and blue flowers on the bedside table, bright indoor daylight",
    "a black lace lingerie set",
    "standing by the flower vase, unhooking the black lace bra",
    "topless, wearing black lace panties",
    "topless, pulling down black lace panties",
    "the black lace lingerie",
)
far_strip_cell(
    37,
    "white sectional, white tee",
    "37_sectional_tee",
    5400,
    "bright room with white walls, large gray floor tiles, a large white leather sectional sofa, even daylight",
    "a plain white t-shirt and panties",
    "standing by the white leather sectional, lifting the plain white t-shirt",
    "topless, wearing panties",
    "topless, pulling down panties",
    "the white t-shirt",
)


code(
    r"""# @title 38) Explicit set: white room penis visible (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'bright modern white room, white leather couch, large white bed with white linens, high-key daylight'
SLUG = '38_white_room_explicit'
SEED_BASE = 5500
SHOTS = [
    ('01_stand_hip', 'nude', 'standing nude by the white bed, his erect penis with visible glans, shaft, and veins beside her hip, she smiles at the camera, an adult man whose face is out of frame'),
    ('02_couch_legs', 'nude', 'sitting nude on the white leather couch, legs apart, his erect photorealistic penis with glans and shaft veins in the foreground, she looks at the camera and smiles, an adult man whose face is out of frame'),
    ('03_cowgirl', 'sex', 'cowgirl on the white bed, his erect penis with visible glans and veined shaft entering her, she smiles at the camera, an adult man whose face is out of frame'),
    ('04_cowgirl_lean', 'sex', 'cowgirl leaning back on the white bed, his erect shaft and glans visible with natural skin texture, she looks at the camera, an adult man whose face is out of frame'),
    ('05_missionary', 'sex', 'missionary on the white bed, her legs spread toward the camera, his erect penis with glans and veins visible at her vulva, she smiles, an adult man whose face is out of frame'),
    ('06_doggy', 'sex', 'doggy on the white bed, his erect penis with visible glans and shaft entering her from behind, she looks back and smiles, an adult man whose face is out of frame'),
    ('07_hold_penis', 'sex', 'standing nude, she holds his erect penis with both hands, photorealistic glans, shaft, and veins visible, smiling at the camera, an adult man whose face is out of frame'),
    ('08_kneel_oral', 'sex', 'kneeling oral, his erect penis with visible glans and veined shaft in her mouth, she looks up smiling, an adult man whose face is out of frame'),
    ('09_oral_close', 'sex', 'close oral, her tongue on the glans, shaft veins visible, natural skin texture, eye contact, slight smile, an adult man whose face is out of frame'),
    ('10_extreme_close', 'sex', 'extreme close-up of his erect penis next to her vulva, photorealistic glans, shaft, veins, and natural skin texture, an adult man whose face is out of frame'),
    ('11_legs_up', 'sex', 'on her back on the white bed, legs up, his erect penis with visible glans at her vulva, she smiles at the camera, an adult man whose face is out of frame'),
    ('12_side_sex', 'sex', 'side sex on the white bed, his erect penis with shaft and glans visible, she looks at the camera, an adult man whose face is out of frame'),
    ('13_sit_facing', 'sex', 'sitting on him facing the camera on the white bed, his erect penis with veins and glans visible, she smiles, an adult man whose face is out of frame'),
    ('14_pen_close', 'sex', 'close-up of penetration, realistic erect penis and vulva with natural skin texture, she looking at the camera in the background, an adult man whose face is out of frame'),
    ('15_facial_start', 'sex', 'kneeling, thick white semen landing on her smiling face, his erect penis with visible glans still in frame, an adult man whose face is out of frame'),
    ('16_semen_lips', 'sex', 'close-up, thick white semen on her lips and chin, she smiles at the camera, his erect penis still in frame, an adult man whose face is out of frame'),
    ('17_semen_chest', 'sex', 'thick white semen on her chest and face, she smiles, holding his erect penis with visible glans and veins, an adult man whose face is out of frame'),
    ('18_after_drip', 'sex', 'sitting on the white bed after, thick white semen dripping, his erect penis visible, she smiles, an adult man whose face is out of frame'),
    ('19_wide_bed', 'sex', 'wide shot of both on the white bed, she nude, his erect penis with glans and shaft visible, thick white semen on her, daylight, she smiles, an adult man whose face is out of frame'),
    ('20_face_glans', 'sex', 'tight close-up of her smiling face with thick white semen on her lips, his glans near her cheek, an adult man whose face is out of frame'),
]
run_scene_set(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 39) Explicit set: grey sofa semen (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'modern apartment, large grey fabric sofa, indoor plants, dark grey tile, even indoor light'
SLUG = '39_grey_sofa_semen'
SEED_BASE = 5600
SHOTS = [
    ('01_wide_doggy', 'sex', 'wide doggy on the grey sofa, his erect penis with visible glans and veined shaft behind her, she looks back and smiles, an adult man whose face is out of frame'),
    ('02_closer_doggy', 'sex', 'closer doggy on the grey sofa, his erect penis entering her, glans and shaft visible, she looks back at the camera with a smile, an adult man whose face is out of frame'),
    ('03_rear_close', 'sex', 'rear close-up, photorealistic erect penis and vulva with natural skin texture, thick white semen starting to drip, she looks back, an adult man whose face is out of frame'),
    ('04_all_fours', 'sex', 'on all fours on the grey sofa, his hands on her hips, his erect penis with glans and veins visible, she smiles over her shoulder, an adult man whose face is out of frame'),
    ('05_hold_face', 'sex', 'kneeling in front of the sofa, holding his erect penis near her smiling face, glans and veins visible, natural skin texture, an adult man whose face is out of frame'),
    ('06_oral_sofa', 'sex', 'oral on the grey sofa, his erect penis with visible glans in her mouth, she looks up smiling, an adult man whose face is out of frame'),
    ('07_oral_tongue', 'sex', 'close oral, tongue out next to the glans, shaft veins visible, eye contact, smile, an adult man whose face is out of frame'),
    ('08_lie_back', 'sex', 'lying back on the grey sofa, legs spread, his erect penis with glans at her vulva, she smiles at the camera, an adult man whose face is out of frame'),
    ('09_missionary', 'sex', 'missionary on the grey sofa, his erect penis with shaft and veins visible, she smiles, an adult man whose face is out of frame'),
    ('10_stand_behind', 'sex', 'standing sex from behind at the grey sofa, his erect penis with visible glans entering her, she looks back smiling, an adult man whose face is out of frame'),
    ('11_sit_him', 'sex', 'sitting on him on the grey sofa, his erect penis with glans and veins visible, she smiles at the camera, an adult man whose face is out of frame'),
    ('12_pen_close', 'sex', 'close penetration, photorealistic erect penis and vulva, visible glans, shaft veins, natural skin texture, an adult man whose face is out of frame'),
    ('13_semen_butt', 'sex', 'thick white semen splashed on her buttocks, his erect penis with glans still visible, she looks back, an adult man whose face is out of frame'),
    ('14_semen_vulva', 'sex', 'thick white semen dripping from her vulva, his erect penis in frame, she smiles back at the camera, an adult man whose face is out of frame'),
    ('15_facial_hold', 'sex', 'kneeling, thick white semen on her face, holding his erect penis with visible glans and veins, broad smile at the camera, an adult man whose face is out of frame'),
    ('16_semen_close', 'sex', 'close-up, thick white semen on lips, chin, and cheeks, she smiles, his erect penis near her mouth, an adult man whose face is out of frame'),
    ('17_semen_chest', 'sex', 'thick white semen on her chest and neck, she smiles, grey sofa behind her, his erect penis in frame, an adult man whose face is out of frame'),
    ('18_wide_sofa', 'sex', 'wide sofa shot, she nude, thick white semen on her, his erect penis with glans and shaft visible, plants in the background, she smiles, an adult man whose face is out of frame'),
    ('19_wipe_mouth', 'sex', 'she wipes thick white semen from her smiling mouth, his erect penis in frame, an adult man whose face is out of frame'),
    ('20_face_glans', 'sex', 'tight smiling facial close-up, thick white semen on her lips, his glans by her cheek, an adult man whose face is out of frame'),
]
run_scene_set(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)

code(
    r"""# @title 40) Explicit set: genital close and facial (20)
SHOT_START = 0
SHOT_END = 20
PLACE = 'bright white bedroom, white bed, white leather sofa, high-key light'
SLUG = '40_genital_facial'
SEED_BASE = 5700
SHOTS = [
    ('01_waist_hold', 'sex', 'waist-up, she holds his erect penis with both hands, smiling at the camera, photorealistic glans, shaft, and veins visible, an adult man whose face is out of frame'),
    ('02_kiss_shaft', 'sex', 'she kisses the shaft of his erect penis, looking at the camera, slight smile, glans and veins visible, natural skin texture, an adult man whose face is out of frame'),
    ('03_oral_up', 'sex', 'oral close, his erect penis with visible glans in her mouth, brown eyes looking up, smile around it, an adult man whose face is out of frame'),
    ('04_glans_lips', 'sex', 'extreme close-up of the glans and her lips, photorealistic skin, shaft veins, an adult man whose face is out of frame'),
    ('05_vulva_penis', 'sex', 'her vulva and his erect penis in one close frame, photorealistic glans, shaft, veins, natural skin texture, an adult man whose face is out of frame'),
    ('06_pen_smile', 'sex', 'penetration close-up, realistic erect penis and vulva with natural skin texture, her smile visible at the top of the frame, an adult man whose face is out of frame'),
    ('07_cowgirl_low', 'sex', 'cowgirl from a low angle, his erect penis with glans and veins visible entering her, she smiles down at the camera, an adult man whose face is out of frame'),
    ('08_missionary', 'sex', 'missionary close, legs spread, his erect penis with visible glans at her vulva, she smiles, an adult man whose face is out of frame'),
    ('09_doggy_close', 'sex', 'doggy close, his erect penis with shaft and glans visible, she looks back smiling, an adult man whose face is out of frame'),
    ('10_folded', 'sex', 'on her back folded, his erect penis near her face and vulva both in frame, glans visible, open-mouth smile, an adult man whose face is out of frame'),
    ('11_side_oral', 'sex', 'side close oral, his erect penis with veins and glans, her smile, eye contact, an adult man whose face is out of frame'),
    ('12_stroke', 'sex', 'she strokes him with both hands, smiling, photorealistic glans and shaft veins visible, an adult man whose face is out of frame'),
    ('13_semen_shoot', 'sex', 'thick white semen shooting toward her smiling face, his erect penis with visible glans in frame, an adult man whose face is out of frame'),
    ('14_semen_tongue', 'sex', 'thick white semen on her tongue, she smiles at the camera, his erect penis in frame, an adult man whose face is out of frame'),
    ('15_semen_hold', 'sex', 'thick white semen on her cheeks and lips, wide smile, holding his erect penis with visible glans and veins, an adult man whose face is out of frame'),
    ('16_semen_drip', 'sex', 'thick white semen dripping down her chin onto her chest, she smiles, his erect penis in frame, an adult man whose face is out of frame'),
    ('17_afterglow', 'sex', 'afterglow close, thick white semen on her vulva, his erect penis resting nearby with visible glans, she smiles, an adult man whose face is out of frame'),
    ('18_semen_rear', 'sex', 'thick white semen on her buttocks and vulva, she looks back smiling, his erect penis in frame, an adult man whose face is out of frame'),
    ('19_medium_bed', 'sex', 'medium shot on the white bed, thick white semen on her face and chest, his erect penis with glans and shaft visible, she smiles, an adult man whose face is out of frame'),
    ('20_final_face', 'sex', 'final tight face, thick white semen on her lips, happy smile, his glans at the edge of the frame, an adult man whose face is out of frame'),
]
run_scene_set(SLUG, PLACE, SHOTS, SEED_BASE, SHOT_START, SHOT_END)"""
)


md(
    """## Male penis LoRA (cells 41-45) then 5-shot dual generate (46-48)

v2 stays locked. Do not run cells 5-9. These cells write `henry_penis_flux_v1.safetensors` only.

Dataset: `ADD_HENRY_BODY_PHOTOS` (26 real waist-down photos). Trigger: `hrmale`.
Never train on Flux outputs. After 41-45, run **one** of 46 / 47 / 48 (5 pictures each).
Cells 46-48 load **both** LoRAs and use short prompts that start with penis/semen/mouth."""
)

code(
    r"""# @title 41) Male LoRA: copy 26 KEEP photos + hrmale captions
import os
import shutil

if HENRY_OUTPUT_LORA in PROTECTED_LORAS:
    raise RuntimeError("Male LoRA name is protected. Do not train.")
if OUTPUT_LORA_NAME in (HENRY_OUTPUT_LORA,):
    raise RuntimeError("Refusing to reuse the locked v2 filename.")

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
SKIP_NAMES = {".drive_upload.json", ".ds_store", "thumbs.db"}
HENRY_EXCLUDE = {
    "IMG-20231126-WA0035.jpg",
    "20260509_072024(1).jpg",
    "20251227_185216_1.jpg",
    "20260509_121218(1).jpg",
    "20260807_122635(1).jpg",
    "20260711_083049(2).jpg",
    "20260711_082212(2).jpg",
    "20250906_135503(1).jpg",
}
HENRY_KEEP = {
    "20250906_135503.jpg": "hrmale, erect penis, visible glans, veined shaft, looking down POV, tiled floor, waist-down close-up, photorealistic photo",
    "20250906_140426.jpg": "hrmale, erect penis, visible glans, veined shaft, looking down at the lap, waist-down close-up, photorealistic photo",
    "20251227_185216.jpg": "hrmale, erect penis held at the base, visible glans, veined shaft, lying on a bed, waist-down close-up, photorealistic photo",
    "InShot_20260212_193432944.jpg": "hrmale, erect penis side view, visible glans, veined shaft, waist-down close-up, photorealistic photo",
    "20260228_074019.jpg": "hrmale, erect penis, visible glans, veined shaft, standing waist-down close-up, photorealistic photo",
    "20260228_080247(1).jpg": "hrmale, erect penis side profile, visible glans, veined shaft, standing waist-down, photorealistic photo",
    "20260328_075339(2).jpg": "hrmale, erect penis side view, visible glans, veined shaft, outdoor daylight, waist-down close-up, photorealistic photo",
    "20260328_075339(3).jpg": "hrmale, erect penis side view, glans and shaft veins, outdoor balcony, waist-down close-up, photorealistic photo",
    "20260328_075339(4).jpg": "hrmale, erect penis side view, visible glans, veined shaft, balcony daylight, waist-down, photorealistic photo",
    "20260329_013617.jpg": "hrmale, erect penis side view, visible glans, veined shaft, waist-down close-up, photorealistic photo",
    "20260329_013617(1).jpg": "hrmale, erect penis, visible glans, veined shaft, front close-up, photorealistic photo",
    "20260509_064836.jpg": "hrmale, erect penis, visible glans, veined shaft, standing waist-down, photorealistic photo",
    "20260509_072024.jpg": "hrmale, erect penis, visible glans, veined shaft, lying down, waist-down close-up, photorealistic photo",
    "20260509_072024(2).jpg": "hrmale, erect penis looking down, visible glans, veined shaft, thighs in frame, photorealistic photo",
    "20260509_121218.jpg": "hrmale, erect penis looking down, visible glans, veined shaft, waist-down close-up, photorealistic photo",
    "20260711_082212.jpg": "hrmale, erect penis extreme close-up, glans, veined shaft, photorealistic photo",
    "20260711_083049.jpg": "hrmale, erect penis looking down, visible glans, veined shaft, waist-down close-up, photorealistic photo",
    "20260711_083049(1).jpg": "hrmale, erect penis looking down, glans and shaft veins, tiled floor, photorealistic photo",
    "20260711_113135.jpg": "hrmale, extreme close-up of the glans, shaft veins, photorealistic photo",
    "20260807_122635.jpg": "hrmale, erect penis side profile, visible glans, veined shaft, standing close-up, photorealistic photo",
    "20260807_122635(2).jpg": "hrmale, erect penis side view, visible glans, veined shaft, standing close-up, photorealistic photo",
    "20260815_222042.jpg": "hrmale, erect penis looking down POV, visible glans, veined shaft, waist-down close-up, photorealistic photo",
    "DJI_20250802_221407_72_null_video(1).jpg": "hrmale, erect penis side view, visible glans, veined shaft, waist-down close-up, photorealistic photo",
    "Dalia V2 Part 2.jpg": "hrmale, erect penis looking down, visible glans, veined shaft, close-up, photorealistic photo",
    "Dalia V2 Part 2(1).jpg": "hrmale, extreme close-up of the glans and shaft, photorealistic photo",
    "Dalia V2 Part 2(2).jpg": "hrmale, extreme close-up of the glans and veined shaft, photorealistic photo",
}

need_sync = False
if not os.path.isdir(HENRY_INBOX_DIR) or (not os.listdir(HENRY_INBOX_DIR)):
    need_sync = True
if need_sync:
    print("Copying ADD_HENRY_BODY_PHOTOS via Drive API...")
    service = DRIVE_SERVICE or _api_service()
    found = api_find_child(service, FIRATSUPER_DRIVE_ID, "ADD_HENRY_BODY_PHOTOS")
    folder_id = found["id"] if found else HENRY_BODY_ID
    api_download_folder(service, folder_id, HENRY_INBOX_DIR)

if not os.path.isdir(HENRY_INBOX_DIR):
    raise RuntimeError("Missing male photo folder: " + HENRY_INBOX_DIR)

if os.path.isdir(HENRY_DATASET_DIR):
    shutil.rmtree(HENRY_DATASET_DIR)
os.makedirs(HENRY_DATASET_DIR, exist_ok=True)

print("Woman dataset folder left untouched:", DATASET_DIR)
print("Locked LoRA left untouched:", OUTPUT_LORA_NAME)

pairs = []
skipped = []
for name in sorted(os.listdir(HENRY_INBOX_DIR)):
    if name.startswith("."):
        continue
    if name.lower() in SKIP_NAMES:
        skipped.append(name + " (skip file)")
        continue
    path = os.path.join(HENRY_INBOX_DIR, name)
    if os.path.isdir(path):
        skipped.append(name + "/")
        continue
    stem, ext = os.path.splitext(name)
    low = name.lower()
    if "flux_eval" in low or "_seed" in low or low.startswith("scene_") or low.startswith("strip_"):
        skipped.append(name + " (looks generated, never train)")
        continue
    if ext.lower() not in IMG_EXT:
        skipped.append(name + " (not image)")
        continue
    if name in HENRY_EXCLUDE:
        skipped.append(name + " (exclude)")
        continue
    if name not in HENRY_KEEP:
        skipped.append(name + " (not in 26 KEEP)")
        continue
    caption = HENRY_KEEP[name]
    if not caption.lower().startswith("hrmale"):
        raise RuntimeError("Caption must start with hrmale: " + name)
    if "ohwx" in caption.lower() or "lapetitemilf" in caption.lower():
        raise RuntimeError("Male captions must not use ohwx / LaPetiteMilf: " + name)
    dest_img = os.path.join(HENRY_DATASET_DIR, name)
    dest_txt = os.path.join(HENRY_DATASET_DIR, stem + ".txt")
    shutil.copy2(path, dest_img)
    with open(dest_txt, "w", encoding="ascii") as fh:
        fh.write(caption + "\n")
    pairs.append((name, caption))

print("Kept", len(pairs), "male photos")
if skipped:
    print("Skipped:")
    for row in skipped:
        print(" ", row)

missing = sorted(set(HENRY_KEEP) - set(p[0] for p in pairs))
if missing:
    raise RuntimeError("KEEP photos missing from Drive folder: " + ", ".join(missing))
if len(pairs) != HENRY_EXPECTED:
    raise RuntimeError("Need %d unique KEEP photos. Got %d" % (HENRY_EXPECTED, len(pairs)))

print("--- sample caption ---")
print(pairs[0][0])
print(pairs[0][1])
print("----------------------")
print("Male Gate 1 GO. Local folder:", HENRY_DATASET_DIR)
print("Does not overwrite", OUTPUT_LORA_NAME)"""
)

code(
    r"""# @title 42) Male LoRA: write Ostris YAML (henry_penis_flux_v1)
if HENRY_OUTPUT_LORA in PROTECTED_LORAS:
    raise RuntimeError("Male LoRA name is protected.")
write_henry_yaml(HENRY_CONFIG_PATH, HENRY_TRAIN_STEPS, dry=False)
print("--- YAML name/dataset ---")
with open(HENRY_CONFIG_PATH, "r", encoding="ascii") as fh:
    for i, line in enumerate(fh):
        if i >= 45:
            break
        print(line.rstrip("\n"))
print("YAML file is", HENRY_CONFIG_PATH)
print("Locked v2 YAML path left untouched:", CONFIG_PATH)
print("Output name:", HENRY_LORA_NAME, "trigger:", HENRY_TRIGGER)"""
)

code(
    r"""# @title 43) Male LoRA: dry run (5 steps)
import os
import subprocess
import sys

if HENRY_OUTPUT_LORA in PROTECTED_LORAS:
    raise RuntimeError("Male LoRA name is protected.")
if not os.path.isdir("/content/ai-toolkit"):
    raise RuntimeError("ai-toolkit missing. Run cell 4, then rerun this cell. Do not run cells 5-9.")
n_txt = len([n for n in os.listdir(HENRY_DATASET_DIR) if n.endswith(".txt")])
if n_txt != HENRY_EXPECTED:
    raise RuntimeError("Run cell 41 first. Need %d captions, got %d" % (HENRY_EXPECTED, n_txt))
write_henry_yaml(HENRY_CONFIG_PATH, DRY_RUN_STEPS, dry=True)
print("Male dry run: %d steps. Downloads FLUX.1-dev the first time." % DRY_RUN_STEPS)
print("This does not write", OUTPUT_LORA_NAME)
cmd = [sys.executable, "run.py", HENRY_CONFIG_PATH]
print("+", " ".join(cmd))
subprocess.check_call(cmd, cwd="/content/ai-toolkit")
write_henry_yaml(HENRY_CONFIG_PATH, HENRY_TRAIN_STEPS, dry=False)
print("Male dry run OK. YAML restored to", HENRY_TRAIN_STEPS, "steps.")
print("Next: cell 44 full train. Keep this tab open.")"""
)

code(
    r"""# @title 44) Male LoRA: full train henry_penis_flux_v1
import os
import subprocess
import sys
import torch

if HENRY_OUTPUT_LORA in PROTECTED_LORAS:
    raise RuntimeError("Male LoRA name is protected.")
if HENRY_LORA_NAME == LORA_NAME:
    raise RuntimeError("Refusing to train into the locked v2 name.")
if not os.path.isdir("/content/ai-toolkit"):
    raise RuntimeError("ai-toolkit missing. Run cell 4, then this cell. Do not run cells 5-9.")
write_henry_yaml(HENRY_CONFIG_PATH, HENRY_TRAIN_STEPS, dry=False)
print("Male full train:", HENRY_TRAIN_STEPS, "steps on", torch.cuda.get_device_name(0))
print("Writes /content/output/%s/  NOT  %s" % (HENRY_LORA_NAME, OUTPUT_LORA_NAME))
print("Keep this tab open.")
cmd = [sys.executable, "run.py", HENRY_CONFIG_PATH]
print("+", " ".join(cmd))
subprocess.check_call(cmd, cwd="/content/ai-toolkit")
print("Male training finished.")
print("Next: cell 45 copies henry_penis_flux_v1.safetensors to Drive/loras/")"""
)

code(
    r"""# @title 45) Male LoRA: copy henry_penis_flux_v1 to Drive (not v2)
import os
import glob
from IPython.display import display, Image as IPyImage

if HENRY_OUTPUT_LORA in PROTECTED_LORAS:
    raise RuntimeError("Refusing to write a protected LoRA name.")
if HENRY_OUTPUT_LORA == OUTPUT_LORA_NAME:
    raise RuntimeError("Refusing to overwrite locked v2.")

run_dir = os.path.join(TRAIN_OUTPUT_DIR, HENRY_LORA_NAME)
candidates = []
final_path = os.path.join(run_dir, HENRY_OUTPUT_LORA)
if os.path.isfile(final_path):
    candidates.append(final_path)
candidates.extend(sorted(glob.glob(os.path.join(run_dir, "*.safetensors"))))
preferred = [p for p in candidates if os.path.basename(p) == HENRY_OUTPUT_LORA]
if not preferred:
    preferred = [p for p in candidates if "step" not in os.path.basename(p).lower()]
if not preferred:
    preferred = candidates
if not preferred:
    raise RuntimeError("No .safetensors found in " + run_dir)

src = preferred[0]
base = os.path.basename(src)
if base in PROTECTED_LORAS or base == OUTPUT_LORA_NAME:
    raise RuntimeError("Refusing to copy a locked LoRA: " + base)
print("Using:", src, "size_mb=%.1f" % (os.path.getsize(src) / 1024**2))

dest_rel = "loras/" + HENRY_OUTPUT_LORA
upload_project_file(src, dest_rel)
print("Male LoRA saved as", dest_rel)
print("Locked file was not touched:", OUTPUT_LORA_NAME)

sample_dir = os.path.join(run_dir, "samples")
shown = 0
if os.path.isdir(sample_dir):
    for name in sorted(os.listdir(sample_dir)):
        path = os.path.join(sample_dir, name)
        if os.path.splitext(name)[1].lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        print(path)
        display(IPyImage(filename=path, width=384))
        shown += 1
        if shown >= 8:
            break
print("Next: cells 46-48 generate 5-shot genital/oral/facial with BOTH LoRAs.")
print("Do not put generated pictures into ADD_HENRY_BODY_PHOTOS.")"""
)

md(
    """## Dual LoRA 5-shot close-ups (cells 46-48)

Need `henry_penis_flux_v1.safetensors` from cell 45 (or already on Drive).
Each cell: **5** pictures, short prompts, penis/mouth/face fill the frame.
Loads locked v2 **and** the male LoRA. Does not change cells 13-40."""
)

code(
    r"""# @title 46) Dual LoRA 5-shot: extreme genital close-up
SHOT_START = 0
SHOT_END = 5
PLACE = "extreme close-up, frame filled by penis and vulva"
SLUG = "46_genital_close"
SEED_BASE = 6100
SHOTS = [
    ("01_glans_vulva", "sex", "hrmale erect penis glans and veined shaft against ohwx woman vulva, extreme close-up"),
    ("02_shaft_labia", "sex", "hrmale erect veined shaft along ohwx woman labia, glans visible, extreme close-up"),
    ("03_glans_entering", "sex", "hrmale erect penis glans entering ohwx woman vulva, shaft veins, extreme close-up"),
    ("04_pressed_close", "sex", "hrmale erect penis pressed to ohwx woman vulva, glans and shaft fill the frame"),
    ("05_side_join", "sex", "hrmale erect penis beside ohwx woman vulva, visible glans, veins, shaft, extreme close-up"),
]
run_scene_set(
    SLUG,
    PLACE,
    SHOTS,
    SEED_BASE,
    SHOT_START,
    SHOT_END,
    ident="",
    adapter_names=["default", "hrmale"],
    adapter_weights=[0.7, 0.85],
    height=1024,
    width=1024,
)"""
)

code(
    r"""# @title 47) Dual LoRA 5-shot: oral on penis close-up
SHOT_START = 0
SHOT_END = 5
PLACE = "extreme close-up, frame filled by penis and mouth"
SLUG = "47_oral_close"
SEED_BASE = 6200
SHOTS = [
    ("01_glans_lips", "sex", "hrmale erect penis glans on ohwx woman lips, shaft veins, extreme close-up"),
    ("02_in_mouth", "sex", "hrmale erect penis in ohwx woman mouth, visible glans and shaft, she looks up"),
    ("03_tongue_glans", "sex", "hrmale glans, ohwx woman tongue, veined shaft, extreme close-up"),
    ("04_lips_shaft", "sex", "hrmale veined shaft in ohwx woman lips, glans visible, extreme close-up"),
    ("05_open_mouth", "sex", "hrmale erect penis at ohwx woman open mouth, glans and veins, extreme close-up"),
]
run_scene_set(
    SLUG,
    PLACE,
    SHOTS,
    SEED_BASE,
    SHOT_START,
    SHOT_END,
    ident="",
    adapter_names=["default", "hrmale"],
    adapter_weights=[0.7, 0.85],
    height=1024,
    width=1024,
)"""
)

code(
    r"""# @title 48) Dual LoRA 5-shot: facial, semen plus glans
SHOT_START = 0
SHOT_END = 5
PLACE = "extreme close-up, frame filled by smiling face and glans"
SLUG = "48_facial_glans"
SEED_BASE = 6300
SHOTS = [
    ("01_semen_glans", "sex", "thick white semen on ohwx woman smiling face, hrmale glans visible, extreme close-up"),
    ("02_semen_lips", "sex", "thick white semen on ohwx woman lips, hrmale erect glans beside her mouth, extreme close-up"),
    ("03_semen_cheek", "sex", "thick white semen on ohwx woman smiling cheek, hrmale glans and shaft in frame"),
    ("04_semen_chin", "sex", "thick white semen on ohwx woman chin and mouth, hrmale glans visible, extreme close-up"),
    ("05_semen_tongue", "sex", "thick white semen on ohwx woman tongue, hrmale glans at her lips, smile"),
]
run_scene_set(
    SLUG,
    PLACE,
    SHOTS,
    SEED_BASE,
    SHOT_START,
    SHOT_END,
    ident="",
    adapter_names=["default", "hrmale"],
    adapter_weights=[0.7, 0.85],
    height=1024,
    width=1024,
)"""
)

md(
    """## Done

Locked production LoRA:
`MyDrive/FiratSuper/loras/lapetitemilf_flux_v2.safetensors`

Male LoRA (cells 41-45, does not overwrite v2):
`MyDrive/FiratSuper/loras/henry_penis_flux_v1.safetensors`

Run ONE series cell at a time. Keep the tab open.

Cells 13-22 (far strip) write to:
`MyDrive/FiratSuper/output/lapetitemilf/flux_eval_v2/strip_*/`

Cells 28-37 (far strip) write to:
`MyDrive/FiratSuper/output/lapetitemilf/flux_eval_v2/strip_*/`

Cells 23-27 (explicit couple / POV / facial) write to:
`MyDrive/FiratSuper/output/lapetitemilf/flux_eval_v2/scene_*/`

Cells 38-40 (explicit couple, visible penis, semen) write to:
`MyDrive/FiratSuper/output/lapetitemilf/flux_eval_v2/scene_*/`

Cells 46-48 (5-shot genital / oral / facial, both LoRAs, short prompts) write to:
`MyDrive/FiratSuper/output/lapetitemilf/flux_eval_v2/scene_*/`

Copy keepers to:
`MyDrive/FiratSuper/keepers/`

Do not write "no scars" in prompts. Flux will draw them.
Do not train on generated pictures. Do not retrain v2.
Two-person shots often glitch on Flux. Change SEED_BASE and rerun that cell.

Also locked:
`loras/lapetitemilf_flux.safetensors` (v1)
`loras/lapetitemilf_face.safetensors`

### Make more pictures
1. A100. Cells 1, 2, 3. New runtime: also cell 4. Then ONE of cells 13-40 (v2 only).
2. Male LoRA once: cells 41, 42, 43, 44, 45. Then ONE of cells 46-48 (both LoRAs, 5 shots).
3. Cells 13-22 and 28-37: far camera strip. Cells 23-27 and 38-40: explicit couple / POV / facial sets (20).
4. If it dies, set SHOT_START and rerun that cell.
5. Nude/sex recipe: woman LoRA about 0.7, male LoRA about 0.85, guidance 2.5. No scar words.
6. Adult content only. Do not train on generated pictures.

### If the runtime dies
- v2 LoRA is already on Drive. Rerun 1, 2, 3, then the series cell. New runtime: also 4. Skip 5-9.
- After male train, `henry_penis_flux_v1.safetensors` is on Drive. Skip 41-45 next time.
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
