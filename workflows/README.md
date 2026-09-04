# RunComfy couple workflows

Drag one JSON onto the ComfyUI canvas. Do not launch a second machine.

## 1) couple_kontext_lora.json  (use this first)

No mask. Same LoRA path that already matched her face.

1. Download `couple_kontext_lora.json`
2. Drag it onto the RunComfy canvas
3. In Load Image, pick the couple photo
4. Paste the Hugging Face READ token into `hf_token`
5. Click Queue once

First run downloads FLUX.1 Kontext. Accept the license:
https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev

## 2) couple_fill_lora.json  (only if Kontext changes the man)

This is a flat Fill graph. The LoRA is wired into the Fill UNET at strength 1.0.
The packed "Flux.1 Inpaint" template cannot take a LoRA. Do not use that template.

1. Drag `couple_fill_lora.json` onto the canvas
2. Load the couple photo
3. Right click the photo -> Open in MaskEditor
4. Paint the whole woman (hair, hands, legs)
5. Queue once
