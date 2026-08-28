#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minimachine import muir, p3
from src.minimachine.abi import expand_function
from src.minimachine.image import (
    BlockExpr,
    ImageError,
    SymbolExpr,
    install_module_image,
    parse_module_image,
)
from src.minimachine.legalize import LegalizeError, legalize_module
from src.minimachine.lower_p3 import MachineLoweringError, lower_function
from src.minimachine.runtime import collect_runtime_surface, install_runtime
from src.minimachine.verify import VerifyError, verify_muir, verify_p3
from src.minimachine.vm import Program, VMError


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build a linked LLVM module into one MiniMachine Program as far as "
            "the frozen strict-P3/system boundary permits."
        )
    )
    p.add_argument("input", type=Path)
    p.add_argument("--max-blocked-functions", type=int)
    p.add_argument("--max-arch-escape-sites", type=int)
    p.add_argument("--require-runtime-closed", action="store_true")
    p.add_argument("--require-image-installed", action="store_true")
    p.add_argument(
        "--symbol-alias",
        action="append",
        default=[],
        metavar="NAME=TARGET",
        help="define a target linker symbol alias before image relocation",
    )
    return p.parse_args()


def _parse_symbol_aliases(items: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid symbol alias: {item}")
        name, target = item.split("=", 1)
        name = name.strip()
        target = target.strip()
        if not name or not target:
            raise ValueError(f"invalid symbol alias: {item}")
        aliases[name] = target
    return aliases


def _trap_symbol(reason: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_]+", "_", reason).strip("_")
    return "__mm_trap_" + (tag or "unknown")


def _arch_escapes(function: muir.Function) -> list[muir.ArchEscape]:
    return [
        inst
        for block in function.blocks
        for inst in block.instructions
        if isinstance(inst, muir.ArchEscape)
    ]


def _trap_reasons(function: muir.Function) -> set[str]:
    return {
        inst.reason
        for block in function.blocks
        for inst in block.instructions
        if isinstance(inst, muir.Trap)
    }


def _value_symbols(value: p3.Value):
    if isinstance(value, muir.Symbol):
        yield ("symbol", value.name)
    elif isinstance(value, muir.Reloc):
        yield ("symbol", value.symbol)
    elif isinstance(value, muir.BlockAddr):
        yield ("block", f"{value.function}:{value.label}")


def _operand_symbols(operand: p3.Operand):
    if isinstance(operand, p3.Mem):
        yield from _value_symbols(operand.address.base)
    else:
        yield from _value_symbols(operand)


def _target_symbols(target: muir.Target):
    if target.symbol is not None:
        yield ("symbol", target.symbol)
    if target.address is not None:
        yield from _value_symbols(target.address.base)


def _p3_references(function: p3.Function):
    for block in function.blocks:
        for inst in block.instructions:
            if isinstance(inst, p3.Mov):
                yield from _operand_symbols(inst.dst)
                yield from _operand_symbols(inst.src)
            elif isinstance(inst, p3.Sub):
                yield from _value_symbols(inst.a)
                yield from _value_symbols(inst.b)
            elif isinstance(inst, p3.Br):
                yield from _value_symbols(inst.a)
                yield from _value_symbols(inst.b)
                yield from _target_symbols(inst.true_target)
                yield from _target_symbols(inst.false_target)


def _image_unresolved(
    program: Program,
    image,
    symbol_aliases: dict[str, str],
) -> tuple[set[str], set[str]]:
    future_symbols = set(program.symbol_addresses)
    future_symbols.update(obj.name for obj in image.objects)
    future_symbols.update(alias.name for alias in image.aliases)
    future_symbols.update(symbol_aliases)

    missing_symbols: set[str] = set()
    missing_blocks: set[str] = set()

    for alias in image.aliases:
        if alias.target.symbol not in future_symbols:
            missing_symbols.add(alias.target.symbol)
    for target in symbol_aliases.values():
        if target not in future_symbols:
            missing_symbols.add(target)

    for obj in image.objects:
        for reloc in obj.relocations:
            target = reloc.target
            if isinstance(target, SymbolExpr):
                if target.symbol not in future_symbols:
                    missing_symbols.add(target.symbol)
            elif isinstance(target, BlockExpr):
                key = (target.function, target.label)
                if key not in program.block_code:
                    missing_blocks.add(f"{target.function}:{target.label}")

    return missing_symbols, missing_blocks


def _program_unresolved(
    program: Program,
    functions: list[p3.Function],
    image,
    symbol_aliases: dict[str, str],
) -> tuple[Counter[str], Counter[str]]:
    future_symbols = set(program.symbol_addresses)
    future_symbols.update(obj.name for obj in image.objects)
    future_symbols.update(alias.name for alias in image.aliases)
    future_symbols.update(symbol_aliases)

    missing_symbols: Counter[str] = Counter()
    missing_blocks: Counter[str] = Counter()

    for function in functions:
        for kind, name in _p3_references(function):
            if kind == "symbol":
                if name not in future_symbols:
                    missing_symbols[name] += 1
            else:
                fn, label = name.split(":", 1)
                if (fn, label) not in program.block_code:
                    missing_blocks[name] += 1

    return missing_symbols, missing_blocks


def _register_traps(program: Program, reasons: set[str]) -> None:
    def callback(reason: str):
        def trap(_vm, _args):
            raise VMError(f"MiniMachine trap reached: {reason}")
        return trap

    for reason in sorted(reasons):
        symbol = _trap_symbol(reason)
        if symbol in program.symbol_addresses:
            continue
        program.register_service(symbol, callback(reason))


def main() -> int:
    args = parse_args()
    text = args.input.read_text()

    try:
        symbol_aliases = _parse_symbol_aliases(args.symbol_alias)
        functions, _legal_stats = legalize_module(text)
        image = parse_module_image(text)

        strict_source: list[muir.Function] = []
        p3_functions: list[p3.Function] = []
        blocked_functions: list[str] = []
        arch_escape_sites = 0
        trap_reasons: set[str] = set()
        p3_instruction_count = 0

        for function in functions:
            verify_muir(function)
            trap_reasons.update(_trap_reasons(function))
            expanded, _abi_stats = expand_function(function)
            escapes = _arch_escapes(expanded)
            if escapes:
                blocked_functions.append(function.name)
                arch_escape_sites += len(escapes)
                continue

            lowered = lower_function(expanded)
            verify_p3(lowered)
            strict_source.append(function)
            p3_functions.append(lowered)
            p3_instruction_count += sum(
                len(block.instructions) for block in lowered.blocks
            )

        program = Program(p3_functions)
        surface = collect_runtime_surface(strict_source)
        install_runtime(program, surface)
        _register_traps(program, trap_reasons)

        p3_missing_symbols, p3_missing_blocks = _program_unresolved(
            program, p3_functions, image, symbol_aliases
        )
        image_missing_symbols, image_missing_blocks = _image_unresolved(
            program, image, symbol_aliases
        )

        image_installed = False
        if not image_missing_symbols and not image_missing_blocks:
            install_module_image(
                program,
                image,
                symbol_aliases=symbol_aliases,
            )
            image_installed = True

    except (
        ImageError,
        LegalizeError,
        MachineLoweringError,
        VerifyError,
        VMError,
        ValueError,
    ) as exc:
        print(f"EXEC_PROBE_FAIL {exc}")
        return 1

    missing_helpers = tuple(getattr(program, "runtime_missing_helpers", ()))
    missing_systems = tuple(getattr(program, "runtime_missing_systems", ()))

    print(
        "EXEC_PROBE "
        f"functions={len(functions)} "
        f"p3_functions={len(p3_functions)} "
        f"blocked_functions={len(blocked_functions)} "
        f"arch_escape_sites={arch_escape_sites} "
        f"p3_instructions={p3_instruction_count} "
        f"runtime_helpers={len(surface.helpers)} "
        f"runtime_systems={len(surface.system_ops)} "
        f"missing_runtime_helpers={len(missing_helpers)} "
        f"missing_runtime_systems={len(missing_systems)} "
        f"image_objects={len(image.objects)} "
        f"image_bytes={image.byte_size} "
        f"image_relocations={image.relocation_count} "
        f"external_data={len(image.external_data)} "
        f"external_functions={len(image.external_functions)} "
        f"linker_aliases={len(symbol_aliases)} "
        f"image_installed={int(image_installed)} "
        f"p3_unresolved_symbols={len(p3_missing_symbols)} "
        f"p3_unresolved_blocks={len(p3_missing_blocks)} "
        f"image_unresolved_symbols={len(image_missing_symbols)} "
        f"image_unresolved_blocks={len(image_missing_blocks)}"
    )

    for name in blocked_functions[:40]:
        print(f"EXEC_BLOCKED_FUNCTION {name}")
    for name, count in p3_missing_symbols.most_common(60):
        print(f"EXEC_P3_UNRESOLVED count={count} symbol={name}")
    for name, count in p3_missing_blocks.most_common(20):
        print(f"EXEC_P3_UNRESOLVED_BLOCK count={count} target={name}")
    for name in sorted(image_missing_symbols)[:60]:
        print(f"EXEC_IMAGE_UNRESOLVED symbol={name}")
    for name in sorted(image_missing_blocks)[:20]:
        print(f"EXEC_IMAGE_UNRESOLVED_BLOCK target={name}")
    for name in missing_helpers[:40]:
        print(f"EXEC_RUNTIME_MISSING_HELPER {name}")
    for name in missing_systems[:40]:
        print(f"EXEC_RUNTIME_MISSING_SYSTEM {name}")

    failed = False
    if (
        args.max_blocked_functions is not None
        and len(blocked_functions) > args.max_blocked_functions
    ):
        print(
            "EXEC_GATE_FAIL "
            f"blocked_functions={len(blocked_functions)} "
            f"max={args.max_blocked_functions}"
        )
        failed = True
    if (
        args.max_arch_escape_sites is not None
        and arch_escape_sites > args.max_arch_escape_sites
    ):
        print(
            "EXEC_GATE_FAIL "
            f"arch_escape_sites={arch_escape_sites} "
            f"max={args.max_arch_escape_sites}"
        )
        failed = True
    if args.require_runtime_closed and (missing_helpers or missing_systems):
        print(
            "EXEC_GATE_FAIL "
            f"missing_runtime_helpers={len(missing_helpers)} "
            f"missing_runtime_systems={len(missing_systems)}"
        )
        failed = True
    if args.require_image_installed and not image_installed:
        print(
            "EXEC_GATE_FAIL "
            f"image_unresolved_symbols={len(image_missing_symbols)} "
            f"image_unresolved_blocks={len(image_missing_blocks)}"
        )
        failed = True

    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
