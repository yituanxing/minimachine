#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minimachine.image import ImageError, parse_module_image


def main() -> int:
    p = argparse.ArgumentParser(description="Probe a linked LLVM module as a MiniMachine global data image.")
    p.add_argument("input", type=Path)
    args = p.parse_args()

    text = args.input.read_text()
    try:
        image = parse_module_image(text)
    except ImageError as exc:
        print(f"IMAGE_PROBE_FAIL {exc}")
        return 1

    sections = {}
    for obj in image.objects:
        key = obj.section or "<default>"
        sections[key] = sections.get(key, 0) + obj.size

    print(
        "IMAGE_PROBE "
        f"objects={len(image.objects)} "
        f"bytes={image.byte_size} "
        f"relocations={image.relocation_count} "
        f"aliases={len(image.aliases)} "
        f"external_data={len(image.external_data)} "
        f"external_functions={len(image.external_functions)} "
        f"skipped_linker_metadata={len(image.skipped_linker_metadata)} "
        f"undef_bytes={image.undef_bytes} "
        f"sections={len(sections)}"
    )
    for name, size in sorted(sections.items(), key=lambda x: (-x[1], x[0]))[:20]:
        print(f"IMAGE_SECTION bytes={size} name={name}")
    for name in image.external_data[:40]:
        print(f"IMAGE_EXTERNAL_DATA {name}")
    for name in image.external_functions[:40]:
        print(f"IMAGE_EXTERNAL_FUNCTION {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
