#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minimachine.user_image import (
    BFLT_HEADER_SIZE,
    BFLT_MAGIC,
    BFLT_VERSION,
    UserImageError,
    unpack_user_image,
)
from src.minimachine.user_image_cache import save_user_image_cache


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a trusted parsed MiniMachine userspace image cache."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--slim-metadata",
        action="store_true",
        help="drop P3 instruction bodies for native append-cache replay",
    )
    parser.add_argument(
        "--slim-output",
        type=Path,
        help="also write a slim metadata cache from the same parsed image",
    )
    args = parser.parse_args()
    if args.slim_metadata and args.slim_output is not None:
        raise SystemExit("--slim-metadata and --slim-output are mutually exclusive")

    blob = args.input.read_bytes()
    if len(blob) < BFLT_HEADER_SIZE:
        raise SystemExit("truncated bFLT image")
    fields = struct.unpack(">4s15I", blob[:BFLT_HEADER_SIZE])
    if fields[0] != BFLT_MAGIC or fields[1] != BFLT_VERSION:
        raise SystemExit("unsupported bFLT image")
    entry = fields[2]
    data_start = fields[3]
    if entry != BFLT_HEADER_SIZE or not (entry <= data_start <= len(blob)):
        raise SystemExit("invalid bFLT payload bounds")
    payload = blob[entry:data_start]
    logical = 12 + int.from_bytes(payload[8:12], "big")
    payload = payload[:logical]
    digest = hashlib.sha256(payload).hexdigest()
    try:
        image = unpack_user_image(payload)
    except UserImageError as exc:
        raise SystemExit(f"invalid MiniMachine userspace payload: {exc}") from exc
    save_user_image_cache(
        image,
        args.output,
        payload_sha256=digest,
        slim_metadata=args.slim_metadata,
    )
    if args.slim_output is not None:
        save_user_image_cache(
            image,
            args.slim_output,
            payload_sha256=digest,
            slim_metadata=True,
        )
    print(
        "USER_IMAGE_CACHE_BUILT "
        f"input={args.input} output={args.output} "
        f"payload_sha256={digest} functions={len(image.functions)} "
        f"slim_metadata={int(args.slim_metadata)} "
        f"slim_output={args.slim_output if args.slim_output is not None else '-'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
