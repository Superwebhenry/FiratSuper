#!/usr/bin/env python3
"""Write ready-to-drag ComfyUI workflow JSON for RunComfy couple identity."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workflows"

LORA = "/workspace/ComfyUI/models/loras/lapetitemilf_flux_v2.safetensors"
IDENTITY = (
    "ohwx woman, an adult woman with long highlighted blonde hair and brown eyes, "
    "photorealistic"
)
KONTEXT_PROMPT = (
    IDENTITY
    + ", keep the man and the room unchanged, replace only the woman, same pose, same clothes lighting"
)
FILL_PROMPT = IDENTITY + ", same pose, full body, photorealistic"


def _dump(name: str, data: dict) -> None:
    path = OUT / name
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    print("wrote", path)


def kontext_workflow() -> dict:
    return {
        "id": "lapetitemilf-couple-kontext",
        "revision": 0,
        "last_node_id": 4,
        "last_link_id": 2,
        "version": 0.4,
        "groups": [],
        "config": {},
        "extra": {"frontendVersion": "1.49.6"},
        "nodes": [
            {
                "id": 1,
                "type": "MarkdownNote",
                "pos": [-420, 80],
                "size": [380, 280],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "properties": {},
                "widgets_values": [
                    "# Couple identity (Kontext)\n\n"
                    "1. Load the couple photo in Load Image.\n"
                    "2. Paste your Hugging Face READ token into hf_token.\n"
                    "3. Queue once. First run downloads FLUX.1 Kontext.\n\n"
                    "Accept the license: https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev\n"
                ],
                "color": "#432",
                "bgcolor": "#653",
            },
            {
                "id": 2,
                "type": "LoadImage",
                "pos": [0, 80],
                "size": [320, 320],
                "flags": {},
                "order": 1,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [1], "slot_index": 0},
                    {"name": "MASK", "type": "MASK", "links": None},
                ],
                "properties": {"Node name for S&R": "LoadImage"},
                "widgets_values": ["example.png", "image"],
            },
            {
                "id": 3,
                "type": "RCFluxKontext",
                "pos": [360, 80],
                "size": [400, 430],
                "flags": {},
                "order": 2,
                "mode": 0,
                "inputs": [
                    {
                        "name": "control_image",
                        "type": "IMAGE",
                        "link": 1,
                    },
                    {
                        "name": "prompt",
                        "type": "STRING",
                        "widget": {"name": "prompt"},
                        "link": None,
                    },
                    {
                        "name": "width",
                        "type": "INT",
                        "widget": {"name": "width"},
                        "link": None,
                    },
                    {
                        "name": "height",
                        "type": "INT",
                        "widget": {"name": "height"},
                        "link": None,
                    },
                    {
                        "name": "sample_steps",
                        "type": "INT",
                        "widget": {"name": "sample_steps"},
                        "link": None,
                    },
                    {
                        "name": "guidance_scale",
                        "type": "FLOAT",
                        "widget": {"name": "guidance_scale"},
                        "link": None,
                    },
                    {
                        "name": "seed",
                        "type": "INT",
                        "widget": {"name": "seed"},
                        "link": None,
                    },
                    {
                        "name": "offload_mode",
                        "type": "COMBO",
                        "widget": {"name": "offload_mode"},
                        "link": None,
                    },
                    {
                        "name": "lora_path",
                        "type": "STRING",
                        "widget": {"name": "lora_path"},
                        "link": None,
                    },
                    {
                        "name": "lora_scale",
                        "type": "FLOAT",
                        "widget": {"name": "lora_scale"},
                        "link": None,
                    },
                    {
                        "name": "hf_token",
                        "type": "STRING",
                        "widget": {"name": "hf_token"},
                        "link": None,
                    },
                    {
                        "name": "negative_prompt",
                        "type": "STRING",
                        "widget": {"name": "negative_prompt"},
                        "link": None,
                    },
                ],
                "outputs": [
                    {
                        "name": "image",
                        "type": "IMAGE",
                        "links": [2],
                    }
                ],
                "properties": {
                    "Node name for S&R": "RCFluxKontext",
                    "aux_id": "runcomfy-com/ai-toolkit-inference",
                },
                "widgets_values": [
                    KONTEXT_PROMPT,
                    1024,
                    1024,
                    25,
                    4.0,
                    42,
                    "fixed",
                    "model",
                    LORA,
                    1.0,
                    "",
                    "",
                ],
            },
            {
                "id": 4,
                "type": "SaveImage",
                "pos": [800, 80],
                "size": [320, 380],
                "flags": {},
                "order": 3,
                "mode": 0,
                "inputs": [
                    {"name": "images", "type": "IMAGE", "link": 2},
                    {
                        "name": "filename_prefix",
                        "type": "STRING",
                        "widget": {"name": "filename_prefix"},
                        "link": None,
                    },
                ],
                "outputs": [],
                "properties": {"Node name for S&R": "SaveImage"},
                "widgets_values": ["couple_kontext"],
            },
        ],
        "links": [
            [1, 2, 0, 3, 0, "IMAGE"],
            [2, 3, 0, 4, 0, "IMAGE"],
        ],
    }


def fill_workflow() -> dict:
    """Flat Fill graph so LoRA sits on the UNET, not on a locked subgraph."""
    return {
        "id": "lapetitemilf-couple-fill",
        "revision": 0,
        "last_node_id": 14,
        "last_link_id": 16,
        "version": 0.4,
        "groups": [],
        "config": {},
        "extra": {"frontendVersion": "1.49.6"},
        "nodes": [
            {
                "id": 1,
                "type": "MarkdownNote",
                "pos": [-480, 40],
                "size": [360, 260],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "properties": {},
                "widgets_values": [
                    "# Couple Fill + LoRA\n\n"
                    "Use this if Kontext changed the man too much.\n\n"
                    "1. Load the couple photo.\n"
                    "2. Right click the photo -> Open in MaskEditor.\n"
                    "3. Paint the WHOLE woman (hair, hands, legs).\n"
                    "4. Queue. LoRA is already wired into the Fill UNET at 1.0.\n"
                ],
                "color": "#432",
                "bgcolor": "#653",
            },
            {
                "id": 2,
                "type": "UNETLoader",
                "pos": [0, 40],
                "size": [320, 82],
                "flags": {},
                "order": 1,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "MODEL", "type": "MODEL", "links": [1], "slot_index": 0}
                ],
                "properties": {"Node name for S&R": "UNETLoader"},
                "widgets_values": ["flux1-fill-dev.safetensors", "default"],
            },
            {
                "id": 3,
                "type": "LoraLoaderModelOnly",
                "pos": [360, 40],
                "size": [340, 82],
                "flags": {},
                "order": 2,
                "mode": 0,
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": 1},
                ],
                "outputs": [
                    {"name": "MODEL", "type": "MODEL", "links": [2], "slot_index": 0}
                ],
                "properties": {"Node name for S&R": "LoraLoaderModelOnly"},
                "widgets_values": ["lapetitemilf_flux_v2.safetensors", 1.0],
            },
            {
                "id": 4,
                "type": "DifferentialDiffusion",
                "pos": [740, 40],
                "size": [240, 58],
                "flags": {},
                "order": 3,
                "mode": 0,
                "inputs": [{"name": "model", "type": "MODEL", "link": 2}],
                "outputs": [
                    {"name": "MODEL", "type": "MODEL", "links": [3], "slot_index": 0}
                ],
                "properties": {"Node name for S&R": "DifferentialDiffusion"},
                "widgets_values": [1],
            },
            {
                "id": 5,
                "type": "DualCLIPLoader",
                "pos": [0, 160],
                "size": [320, 130],
                "flags": {},
                "order": 4,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "CLIP", "type": "CLIP", "links": [4], "slot_index": 0}
                ],
                "properties": {"Node name for S&R": "DualCLIPLoader"},
                "widgets_values": [
                    "clip_l.safetensors",
                    "t5xxl_fp16.safetensors",
                    "flux",
                    "default",
                ],
            },
            {
                "id": 6,
                "type": "VAELoader",
                "pos": [0, 330],
                "size": [320, 58],
                "flags": {},
                "order": 5,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "VAE", "type": "VAE", "links": [5, 6], "slot_index": 0}
                ],
                "properties": {"Node name for S&R": "VAELoader"},
                "widgets_values": ["ae.safetensors"],
            },
            {
                "id": 7,
                "type": "LoadImage",
                "pos": [0, 430],
                "size": [320, 320],
                "flags": {},
                "order": 6,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [7], "slot_index": 0},
                    {"name": "MASK", "type": "MASK", "links": [8], "slot_index": 1},
                ],
                "properties": {"Node name for S&R": "LoadImage"},
                "widgets_values": ["example.png", "image"],
            },
            {
                "id": 8,
                "type": "GrowMask",
                "pos": [360, 430],
                "size": [240, 82],
                "flags": {},
                "order": 7,
                "mode": 0,
                "inputs": [{"name": "mask", "type": "MASK", "link": 8}],
                "outputs": [
                    {"name": "MASK", "type": "MASK", "links": [9], "slot_index": 0}
                ],
                "properties": {"Node name for S&R": "GrowMask"},
                "widgets_values": [16, True],
            },
            {
                "id": 9,
                "type": "CLIPTextEncode",
                "pos": [360, 160],
                "size": [400, 200],
                "flags": {},
                "order": 8,
                "mode": 0,
                "inputs": [{"name": "clip", "type": "CLIP", "link": 4}],
                "outputs": [
                    {
                        "name": "CONDITIONING",
                        "type": "CONDITIONING",
                        "links": [10, 11],
                        "slot_index": 0,
                    }
                ],
                "title": "Positive Prompt",
                "properties": {"Node name for S&R": "CLIPTextEncode"},
                "widgets_values": [FILL_PROMPT],
            },
            {
                "id": 10,
                "type": "FluxGuidance",
                "pos": [800, 160],
                "size": [240, 58],
                "flags": {},
                "order": 9,
                "mode": 0,
                "inputs": [
                    {"name": "conditioning", "type": "CONDITIONING", "link": 10}
                ],
                "outputs": [
                    {
                        "name": "CONDITIONING",
                        "type": "CONDITIONING",
                        "links": [12],
                        "slot_index": 0,
                    }
                ],
                "properties": {"Node name for S&R": "FluxGuidance"},
                "widgets_values": [30],
            },
            {
                "id": 11,
                "type": "ConditioningZeroOut",
                "pos": [800, 250],
                "size": [240, 30],
                "flags": {},
                "order": 10,
                "mode": 0,
                "inputs": [
                    {"name": "conditioning", "type": "CONDITIONING", "link": 11}
                ],
                "outputs": [
                    {
                        "name": "CONDITIONING",
                        "type": "CONDITIONING",
                        "links": [13],
                        "slot_index": 0,
                    }
                ],
                "properties": {"Node name for S&R": "ConditioningZeroOut"},
                "widgets_values": [],
            },
            {
                "id": 12,
                "type": "InpaintModelConditioning",
                "pos": [360, 560],
                "size": [320, 150],
                "flags": {},
                "order": 11,
                "mode": 0,
                "inputs": [
                    {"name": "positive", "type": "CONDITIONING", "link": 12},
                    {"name": "negative", "type": "CONDITIONING", "link": 13},
                    {"name": "vae", "type": "VAE", "link": 5},
                    {"name": "pixels", "type": "IMAGE", "link": 7},
                    {"name": "mask", "type": "MASK", "link": 9},
                ],
                "outputs": [
                    {
                        "name": "positive",
                        "type": "CONDITIONING",
                        "links": [14],
                        "slot_index": 0,
                    },
                    {
                        "name": "negative",
                        "type": "CONDITIONING",
                        "links": [15],
                        "slot_index": 1,
                    },
                    {
                        "name": "latent",
                        "type": "LATENT",
                        "links": [16],
                        "slot_index": 2,
                    },
                ],
                "properties": {"Node name for S&R": "InpaintModelConditioning"},
                "widgets_values": [True],
            },
            {
                "id": 13,
                "type": "KSampler",
                "pos": [1080, 40],
                "size": [280, 262],
                "flags": {},
                "order": 12,
                "mode": 0,
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": 3},
                    {"name": "positive", "type": "CONDITIONING", "link": 14},
                    {"name": "negative", "type": "CONDITIONING", "link": 15},
                    {"name": "latent_image", "type": "LATENT", "link": 16},
                ],
                "outputs": [
                    {"name": "LATENT", "type": "LATENT", "links": [17], "slot_index": 0}
                ],
                "properties": {"Node name for S&R": "KSampler"},
                "widgets_values": [42, "fixed", 20, 1, "euler", "normal", 1],
            },
            {
                "id": 14,
                "type": "VAEDecode",
                "pos": [1400, 40],
                "size": [210, 50],
                "flags": {},
                "order": 13,
                "mode": 0,
                "inputs": [
                    {"name": "samples", "type": "LATENT", "link": 17},
                    {"name": "vae", "type": "VAE", "link": 6},
                ],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [18], "slot_index": 0}
                ],
                "properties": {"Node name for S&R": "VAEDecode"},
                "widgets_values": [],
            },
            {
                "id": 15,
                "type": "SaveImage",
                "pos": [1400, 140],
                "size": [320, 380],
                "flags": {},
                "order": 14,
                "mode": 0,
                "inputs": [
                    {"name": "images", "type": "IMAGE", "link": 18},
                    {
                        "name": "filename_prefix",
                        "type": "STRING",
                        "widget": {"name": "filename_prefix"},
                        "link": None,
                    },
                ],
                "outputs": [],
                "properties": {"Node name for S&R": "SaveImage"},
                "widgets_values": ["couple_fill"],
            },
        ],
        "links": [
            [1, 2, 0, 3, 0, "MODEL"],
            [2, 3, 0, 4, 0, "MODEL"],
            [3, 4, 0, 13, 0, "MODEL"],
            [4, 5, 0, 9, 0, "CLIP"],
            [5, 6, 0, 12, 2, "VAE"],
            [6, 6, 0, 14, 1, "VAE"],
            [7, 7, 0, 12, 3, "IMAGE"],
            [8, 7, 1, 8, 0, "MASK"],
            [9, 8, 0, 12, 4, "MASK"],
            [10, 9, 0, 10, 0, "CONDITIONING"],
            [11, 9, 0, 11, 0, "CONDITIONING"],
            [12, 10, 0, 12, 0, "CONDITIONING"],
            [13, 11, 0, 12, 1, "CONDITIONING"],
            [14, 12, 0, 13, 1, "CONDITIONING"],
            [15, 12, 1, 13, 2, "CONDITIONING"],
            [16, 12, 2, 13, 3, "LATENT"],
            [17, 13, 0, 14, 0, "LATENT"],
            [18, 14, 0, 15, 0, "IMAGE"],
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fill = fill_workflow()
    fill["last_node_id"] = 15
    fill["last_link_id"] = 18
    _dump("couple_kontext_lora.json", kontext_workflow())
    _dump("couple_fill_lora.json", fill)


if __name__ == "__main__":
    main()
