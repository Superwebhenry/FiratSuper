#!/usr/bin/env python3
"""Create the FiratSuper folder layout on Google Drive."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ROOT = Path("/content/drive/MyDrive/FiratSuper")


def create_layout(root: Path, project_name: str) -> dict[str, Path]:
    folders = {
        "root": root,
        "datasets": root / "datasets" / project_name / "10_trigger",
        "output": root / "output" / project_name,
        "models": root / "models",
        "logs": root / "logs" / project_name,
        "loras": root / "loras",
    }

    for path in folders.values():
        path.mkdir(parents=True, exist_ok=True)

    readme = root / "README_DRIVE.txt"
    if not readme.exists():
        readme.write_text(
            "FiratSuper — Stable Diffusion LoRA training\n\n"
            "datasets/<project>/10_trigger/  → training images + .txt captions\n"
            "output/<project>/               → trained LoRA checkpoints\n"
            "models/                         → base SD checkpoints (cached)\n"
            "loras/                          → final exported LoRA files\n",
            encoding="utf-8",
        )

    return folders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-name", required=True, help="Project / LoRA name")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Drive root folder")
    args = parser.parse_args()

    folders = create_layout(args.root, args.project_name)
    print("Created FiratSuper layout:")
    for name, path in folders.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
