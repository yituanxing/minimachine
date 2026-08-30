#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minimachine.abi import HEADER_SIZE, WORD, expand_function
from src.minimachine.layout import DataLayout
from src.minimachine.legalize import _register_metadata, legalize_function
from src.minimachine.llvm_text import parse_module
from src.minimachine.lower_p3 import lower_function
from src.minimachine.verify import verify_muir, verify_p3
from src.minimachine.vm import LinkedFunction


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Refresh one P3 function inside an existing ProgramCache while "
            "preserving its linked code addresses and frame ABI."
        )
    )
    p.add_argument("linked_ll", type=Path)
    p.add_argument("cache_in", type=Path)
    p.add_argument("cache_out", type=Path)
    p.add_argument("--function", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    llvm_text = args.linked_ll.read_text()
    functions = parse_module(llvm_text)
    source = next((fn for fn in functions if fn.name == args.function), None)
    if source is None:
        raise SystemExit(f"missing LLVM function: {args.function}")

    layout = DataLayout.from_module(llvm_text)
    current_muir, _ = legalize_function(
        source,
        layout,
        _register_metadata(llvm_text),
    )
    verify_muir(current_muir)
    expanded, _ = expand_function(current_muir)
    current_p3 = lower_function(expanded)
    verify_p3(current_p3)

    with gzip.open(args.cache_in, "rb") as handle:
        payload = pickle.load(handle)
    cache = payload.get("cache")
    if cache is None:
        raise SystemExit("cache payload is missing 'cache'")
    program = cache.program
    old = program.functions.get(args.function)
    if old is None:
        raise SystemExit(f"cache is missing function: {args.function}")

    old_labels = tuple(block.label for block in old.function.blocks)
    new_labels = tuple(block.label for block in current_p3.blocks)
    if old_labels != new_labels:
        raise SystemExit(
            "unsafe refresh: block labels changed "
            f"old={len(old_labels)} new={len(new_labels)}"
        )

    old_slots = set(old.slot_offsets)
    new_slots = set(current_p3.frame_slots)
    if old_slots != new_slots:
        missing = sorted(old_slots - new_slots)[:8]
        added = sorted(new_slots - old_slots)[:8]
        raise SystemExit(
            "unsafe refresh: frame slots changed "
            f"missing={missing} added={added}"
        )

    expected_frame = HEADER_SIZE + len(new_slots) * WORD
    if old.frame_size != expected_frame:
        raise SystemExit(
            "unsafe refresh: frame size changed "
            f"old={old.frame_size} current={expected_frame}"
        )

    for label in new_labels:
        code = program.block_code.get((args.function, label))
        if code is None:
            raise SystemExit(f"unsafe refresh: missing code address for {label}")
        if program.code_block.get(code) != (args.function, label):
            raise SystemExit(f"unsafe refresh: reverse code map mismatch for {label}")

    old_symbol_refs = sum(
        "minimachine_console_buffer" in repr(inst)
        for block in old.function.blocks
        for inst in block.instructions
    )
    new_symbol_refs = sum(
        "minimachine_console_buffer" in repr(inst)
        for block in current_p3.blocks
        for inst in block.instructions
    )

    program.functions[args.function] = LinkedFunction(
        current_p3,
        dict(old.slot_offsets),
        old.frame_size,
        {block.label: block for block in current_p3.blocks},
    )

    args.cache_out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.cache_out, "wb", compresslevel=3) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        "P3_CACHE_FUNCTION_REFRESH "
        f"function={args.function} "
        f"blocks={len(new_labels)} frame={old.frame_size} "
        f"old_symbol_refs={old_symbol_refs} "
        f"new_symbol_refs={new_symbol_refs} "
        f"cache_version={payload.get('version')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
