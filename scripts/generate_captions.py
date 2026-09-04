#!/usr/bin/env python3
"""Generate BLIP captions for images in a local folder."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

TRIGGER = "ohwx woman"
STYLE = "fashion photo, swimsuit, lingerie, high quality"
INPUT_DIR = Path("/tmp/lora_dataset")
OUTPUT_DIR = Path("/tmp/lora_captions")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    images = sorted(
        p
        for p in INPUT_DIR.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        raise SystemExit(f"No images found in {INPUT_DIR}")

    manifest = {}
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=40)
        caption = processor.decode(out[0], skip_special_tokens=True)
        full = f"{TRIGGER}, {caption}, {STYLE}"
        txt_name = img_path.with_suffix(".txt").name
        (OUTPUT_DIR / txt_name).write_text(full, encoding="utf-8")
        manifest[img_path.name] = full
        print(f"{img_path.name}: {full}")

    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(images)} captions to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
