#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import struct

from src.minimachine import muir, p3
from src.minimachine.user_image import (
    BFLT_HEADER_SIZE,
    BFLT_MAGIC,
    BFLT_VERSION,
    UserImageError,
    UserProgramImage,
    unpack_user_image,
)


@dataclass(frozen=True)
class Census:
    path: Path
    functions: int
    runtime_helpers: tuple[str, ...]
    system_descriptors: tuple[str, ...]
    serialized_system_ops: tuple[str, ...]
    external_functions: tuple[str, ...]
    external_data: tuple[str, ...]
    missing_system_descriptors: tuple[str, ...]


def _system_symbol(op: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_]+", "_", op).strip("_")
    if not tag:
        raise ValueError(f"empty system operation after sanitizing: {op!r}")
    return "__mm_sys_" + tag


def _value_symbols(value) -> set[str]:
    if isinstance(value, muir.Symbol):
        return {value.name}
    if isinstance(value, muir.Reloc):
        return {value.symbol}
    if isinstance(value, muir.BlockAddr):
        return {value.function}
    return set()


def _address_symbols(address: muir.Address) -> set[str]:
    return _value_symbols(address.base)


def _operand_symbols(operand) -> set[str]:
    if isinstance(operand, p3.Mem):
        return _address_symbols(operand.address)
    return _value_symbols(operand)


def _target_symbols(target: muir.Target) -> set[str]:
    if target.symbol is not None:
        return {target.symbol}
    if target.address is not None:
        return _address_symbols(target.address)
    return set()


def referenced_symbols(program: UserProgramImage) -> set[str]:
    symbols: set[str] = set()
    for function in program.functions:
        for block in function.blocks:
            for inst in block.instructions:
                if isinstance(inst, p3.Mov):
                    symbols.update(_operand_symbols(inst.dst))
                    symbols.update(_operand_symbols(inst.src))
                elif isinstance(inst, p3.Sub):
                    symbols.update(_value_symbols(inst.a))
                    symbols.update(_value_symbols(inst.b))
                elif isinstance(inst, p3.Br):
                    symbols.update(_value_symbols(inst.a))
                    symbols.update(_value_symbols(inst.b))
                    symbols.update(_target_symbols(inst.true_target))
                    symbols.update(_target_symbols(inst.false_target))
                else:
                    raise TypeError(
                        f"unsupported P3 instruction in census: {type(inst).__name__}"
                    )
    return symbols


def load_bflt_program(path: Path) -> UserProgramImage:
    image = path.read_bytes()
    if len(image) < BFLT_HEADER_SIZE:
        raise UserImageError(f"truncated bFLT image: {path}")
    fields = struct.unpack(">4s15I", image[:BFLT_HEADER_SIZE])
    if fields[0] != BFLT_MAGIC:
        raise UserImageError(f"bad bFLT magic in {path}: {fields[0]!r}")
    if fields[1] != BFLT_VERSION:
        raise UserImageError(f"unsupported bFLT version in {path}: {fields[1]}")
    data_start = int(fields[3])
    if data_start < BFLT_HEADER_SIZE or data_start > len(image):
        raise UserImageError(
            f"invalid bFLT data_start in {path}: {data_start}/{len(image)}"
        )
    return unpack_user_image(image[BFLT_HEADER_SIZE:data_start])


def census_program(path: Path, program: UserProgramImage) -> Census:
    symbols = referenced_symbols(program)
    system_descriptors = tuple(
        sorted(symbol for symbol in symbols if symbol.startswith("__mm_sys_"))
    )
    serialized_system_ops = tuple(
        sorted(getattr(program, "runtime_system_ops", ()))
    )
    serialized_descriptors = {
        _system_symbol(op) for op in serialized_system_ops
    }
    image = program.image
    return Census(
        path=path,
        functions=len(program.functions),
        runtime_helpers=tuple(sorted(program.runtime_helpers)),
        system_descriptors=system_descriptors,
        serialized_system_ops=serialized_system_ops,
        external_functions=tuple(sorted(image.external_functions if image else ())),
        external_data=tuple(sorted(image.external_data if image else ())),
        missing_system_descriptors=tuple(
            sorted(set(system_descriptors) - serialized_descriptors)
        ),
    )


def print_census(census: Census) -> None:
    print(
        "MMCOMPAT "
        f"path={census.path} functions={census.functions} "
        f"helpers={len(census.runtime_helpers)} "
        f"system_refs={len(census.system_descriptors)} "
        f"serialized_system_ops={len(census.serialized_system_ops)} "
        f"external_functions={len(census.external_functions)} "
        f"external_data={len(census.external_data)} "
        f"missing_system_refs={len(census.missing_system_descriptors)}"
    )
    for descriptor in census.system_descriptors:
        state = (
            "missing"
            if descriptor in census.missing_system_descriptors
            else "serialized"
        )
        inferred = descriptor[len("__mm_sys_"):]
        print(
            "MMCOMPAT_SYSTEM "
            f"path={census.path} descriptor={descriptor} "
            f"inferred_op={inferred} state={state}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Census MiniMachine bFLT runtime surfaces before expensive real-software runs."
        )
    )
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail when a referenced __mm_sys_* descriptor is not serialized",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failed = False
    for path in args.images:
        program = load_bflt_program(path)
        census = census_program(path, program)
        print_census(census)
        failed = failed or bool(census.missing_system_descriptors)
    if args.strict and failed:
        print("MMCOMPAT_RESULT=FAIL missing-runtime-system-surface")
        return 1
    print("MMCOMPAT_RESULT=PASS" if not failed else "MMCOMPAT_RESULT=OBSERVED_GAPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
