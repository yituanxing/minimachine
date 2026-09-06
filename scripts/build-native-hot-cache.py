#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.minimachine.checkpoint import image_fingerprint
from src.minimachine.native_hot_cache import save_native_hot_cache
from src.minimachine.program_cache import load_program_cache


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build a slim metadata cache for native MiniMachine hot replay."
    )
    p.add_argument("linked_llvm", type=Path)
    p.add_argument("program_cache", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()

    llvm_text = args.linked_llvm.read_text()
    image_sha256 = image_fingerprint(llvm_text)
    cache = load_program_cache(
        args.program_cache,
        image_sha256=image_sha256,
    )
    save_native_hot_cache(cache, args.output)
    print(
        "NATIVE_HOT_CACHE_BUILT "
        f"input={args.program_cache} output={args.output} "
        f"bytes={args.output.stat().st_size} "
        f"functions={cache.function_count} "
        f"blocks={len(cache.program.block_code)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
