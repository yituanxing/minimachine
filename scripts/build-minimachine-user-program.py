#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minimachine.image import parse_module_image
from src.minimachine.user_bundle import build_user_program_from_llvm
from src.minimachine.user_image import (
    BFLT_HEADER_SIZE,
    build_bflt_program,
    unpack_user_image,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build a namespaced whole-program MiniMachine userspace image."
    )
    p.add_argument("input", type=Path, help="whole-program LLVM .ll")
    p.add_argument("output", type=Path, help="output bFLT/MMP3 image")
    p.add_argument("--entry", default="main")
    p.add_argument("--namespace", default="user")
    p.add_argument(
        "--entry-args",
        choices=("none", "linux-main"),
        default="linux-main",
    )
    p.add_argument("--stack-size", type=lambda x: int(x, 0), default=0x80000)
    p.add_argument("--report", type=Path)
    args = p.parse_args()

    text = args.input.read_text()
    source_image = parse_module_image(text)
    program, surface = build_user_program_from_llvm(
        text,
        entry=args.entry,
        entry_args=args.entry_args,
        namespace=args.namespace,
    )

    image = build_bflt_program(
        program,
        stack_size=args.stack_size,
        compress_payload=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(image)

    data_start = int.from_bytes(image[12:16], "big")
    payload_body = int.from_bytes(
        image[BFLT_HEADER_SIZE + 8:BFLT_HEADER_SIZE + 12],
        "big",
    )
    decoded = unpack_user_image(image[BFLT_HEADER_SIZE:data_start])
    if decoded.entry != program.entry:
        raise RuntimeError("user image round-trip changed entry")
    if len(decoded.functions) != len(program.functions):
        raise RuntimeError("user image round-trip changed function count")

    report = {
        "namespace": args.namespace,
        "entry_original": args.entry,
        "entry_namespaced": program.entry,
        "entry_args": program.entry_args,
        "functions": len(program.functions),
        "runtime_helpers": len(program.runtime_helpers),
        "global_objects": len(program.image.objects if program.image else ()),
        "global_bytes": program.image.byte_size if program.image else 0,
        "relocations": program.image.relocation_count if program.image else 0,
        "external_functions": list(source_image.external_functions),
        "external_data": list(source_image.external_data),
        "bflt_bytes": len(image),
        "compressed_payload_bytes": payload_body,
        "payload_under_16m": payload_body <= 16 * 1024 * 1024,
        "runtime_surface_helpers": len(surface.helpers),
    }

    print(
        "USER_BUNDLE "
        f"namespace={args.namespace} entry={program.entry} "
        f"functions={report['functions']} "
        f"objects={report['global_objects']} "
        f"global_bytes={report['global_bytes']} "
        f"relocs={report['relocations']} "
        f"helpers={report['runtime_helpers']} "
        f"payload_bytes={payload_body} "
        f"bflt_bytes={len(image)} "
        f"under16m={int(report['payload_under_16m'])}",
        flush=True,
    )
    for name in source_image.external_functions:
        print(f"USER_EXTERNAL_FUNCTION {name}", flush=True)
    for name in source_image.external_data:
        print(f"USER_EXTERNAL_DATA {name}", flush=True)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
