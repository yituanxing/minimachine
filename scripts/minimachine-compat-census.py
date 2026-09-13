#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct
import sys
import zlib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minimachine import muir, p3
from src.minimachine.runtime import system_callback
from src.minimachine.system_surface import system_op_from_symbol
from src.minimachine.user_image import (
    BFLT_HEADER_SIZE,
    BFLT_MAGIC,
    BFLT_VERSION,
    PROGRAM_FORMAT,
    USER_PAYLOAD_HEADER_SIZE,
    USER_PROGRAM_PAYLOAD_VERSION,
    USER_PROGRAM_PAYLOAD_ZLIB_VERSION,
    UserImageError,
    UserProgramImage,
    unpack_user_image,
)


_SYSTEM_DESCRIPTOR_RE = re.compile(rb"__mm_sys_[A-Za-z0-9_]+")


@dataclass(frozen=True)
class Census:
    path: Path
    functions: int
    runtime_helpers: tuple[str, ...]
    system_descriptors: tuple[str, ...]
    serialized_system_ops: tuple[str, ...]
    resolvable_system_descriptors: tuple[str, ...]
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


def _bflt_program_json(path: Path) -> tuple[dict, bytes] | None:
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

    payload = image[BFLT_HEADER_SIZE:data_start]
    if len(payload) < USER_PAYLOAD_HEADER_SIZE:
        raise UserImageError(f"truncated MiniMachine payload in {path}")
    magic, version, size = struct.unpack(">4sII", payload[:USER_PAYLOAD_HEADER_SIZE])
    if magic != b"MMP3":
        raise UserImageError(f"bad MiniMachine payload magic in {path}: {magic!r}")
    end = USER_PAYLOAD_HEADER_SIZE + size
    if end > len(payload):
        raise UserImageError(f"truncated MiniMachine payload body in {path}")
    if version not in {USER_PROGRAM_PAYLOAD_VERSION, USER_PROGRAM_PAYLOAD_ZLIB_VERSION}:
        return None

    raw = payload[USER_PAYLOAD_HEADER_SIZE:end]
    if version == USER_PROGRAM_PAYLOAD_ZLIB_VERSION:
        try:
            raw = zlib.decompress(raw)
        except zlib.error as exc:
            raise UserImageError(f"invalid compressed program payload in {path}") from exc
    try:
        obj = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UserImageError(f"invalid program JSON in {path}") from exc
    if not isinstance(obj, dict) or obj.get("format") != PROGRAM_FORMAT:
        raise UserImageError(f"invalid MiniMachine program object in {path}")
    return obj, raw


def load_bflt_program(path: Path) -> UserProgramImage:
    image = path.read_bytes()
    fields = struct.unpack(">4s15I", image[:BFLT_HEADER_SIZE])
    return unpack_user_image(image[BFLT_HEADER_SIZE:int(fields[3])])


def _finish_census(
    *,
    path: Path,
    functions: int,
    runtime_helpers: tuple[str, ...],
    system_descriptors: tuple[str, ...],
    serialized_system_ops: tuple[str, ...],
    external_functions: tuple[str, ...],
    external_data: tuple[str, ...],
) -> Census:
    serialized_descriptors = {_system_symbol(op) for op in serialized_system_ops}
    resolvable_descriptors = {
        symbol
        for symbol in system_descriptors
        if (
            (op := system_op_from_symbol(symbol)) is not None
            and system_callback(op) is not None
        )
    }
    provided_descriptors = serialized_descriptors | resolvable_descriptors
    return Census(
        path=path,
        functions=functions,
        runtime_helpers=runtime_helpers,
        system_descriptors=system_descriptors,
        serialized_system_ops=serialized_system_ops,
        resolvable_system_descriptors=tuple(sorted(resolvable_descriptors)),
        external_functions=external_functions,
        external_data=external_data,
        missing_system_descriptors=tuple(
            sorted(set(system_descriptors) - provided_descriptors)
        ),
    )


def census_program(path: Path, program: UserProgramImage) -> Census:
    symbols = referenced_symbols(program)
    image = program.image
    return _finish_census(
        path=path,
        functions=len(program.functions),
        runtime_helpers=tuple(sorted(program.runtime_helpers)),
        system_descriptors=tuple(
            sorted(symbol for symbol in symbols if symbol.startswith("__mm_sys_"))
        ),
        serialized_system_ops=tuple(sorted(getattr(program, "runtime_system_ops", ()))),
        external_functions=tuple(sorted(image.external_functions if image else ())),
        external_data=tuple(sorted(image.external_data if image else ())),
    )


def census_bflt(path: Path) -> Census:
    packed = _bflt_program_json(path)
    if packed is None:
        return census_program(path, load_bflt_program(path))
    obj, raw = packed
    image = obj.get("image") or {}
    system_descriptors = tuple(
        sorted({match.group(0).decode("ascii") for match in _SYSTEM_DESCRIPTOR_RE.finditer(raw)})
    )
    return _finish_census(
        path=path,
        functions=len(obj.get("functions", ())),
        runtime_helpers=tuple(sorted(obj.get("runtime_helpers", ()))),
        system_descriptors=system_descriptors,
        serialized_system_ops=tuple(sorted(obj.get("runtime_system_ops", ()))),
        external_functions=tuple(sorted(image.get("external_functions", ()))),
        external_data=tuple(sorted(image.get("external_data", ()))),
    )


def print_census(census: Census) -> None:
    print(
        "MMCOMPAT "
        f"path={census.path} functions={census.functions} "
        f"helpers={len(census.runtime_helpers)} "
        f"system_refs={len(census.system_descriptors)} "
        f"serialized_system_ops={len(census.serialized_system_ops)} "
        f"runtime_resolvable_system_refs={len(census.resolvable_system_descriptors)} "
        f"external_functions={len(census.external_functions)} "
        f"external_data={len(census.external_data)} "
        f"missing_system_refs={len(census.missing_system_descriptors)}"
    )
    serialized = {_system_symbol(op) for op in census.serialized_system_ops}
    resolvable = set(census.resolvable_system_descriptors)
    for descriptor in census.system_descriptors:
        if descriptor in serialized:
            state = "serialized"
        elif descriptor in resolvable:
            state = "runtime-resolvable"
        else:
            state = "missing"
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
        help="fail when a referenced __mm_sys_* descriptor is neither serialized nor runtime-resolvable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failed = False
    for path in args.images:
        census = census_bflt(path)
        print_census(census)
        failed = failed or bool(census.missing_system_descriptors)
    if args.strict and failed:
        print("MMCOMPAT_RESULT=FAIL missing-runtime-system-surface")
        return 1
    print("MMCOMPAT_RESULT=PASS" if not failed else "MMCOMPAT_RESULT=OBSERVED_GAPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
