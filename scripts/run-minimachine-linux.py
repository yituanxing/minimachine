#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.image import install_module_image, parse_module_image
from src.minimachine.legalize import legalize_module
from src.minimachine.linker import LinkerContract
from src.minimachine.lower_p3 import lower_function
from src.minimachine.runtime import collect_runtime_surface, install_runtime
from src.minimachine.verify import verify_muir, verify_p3
from src.minimachine.vm import Program, VMError


def parse_args():
    p = argparse.ArgumentParser(
        description="Execute a linked MiniMachine Linux LLVM image in the P3 VM."
    )
    p.add_argument("input", type=Path)
    p.add_argument(
        "--linker-contract",
        type=Path,
        required=True,
        help="MiniMachine linker contract used to install the Linux data image",
    )
    p.add_argument("--entry", default="start_kernel")
    p.add_argument("--max-steps", type=int, default=2_000_000)
    return p.parse_args()


def trap_symbol(reason: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_]+", "_", reason).strip("_")
    return "__mm_trap_" + (tag or "unknown")


def arch_escapes(function: muir.Function):
    return [
        inst
        for block in function.blocks
        for inst in block.instructions
        if isinstance(inst, muir.ArchEscape)
    ]


def trap_reasons(function: muir.Function):
    return {
        inst.reason
        for block in function.blocks
        for inst in block.instructions
        if isinstance(inst, muir.Trap)
    }


def register_traps(program: Program, reasons: set[str]) -> None:
    def callback(reason: str):
        def trap(_vm, _args):
            raise VMError(f"MiniMachine trap reached: {reason}")
        return trap

    for reason in sorted(reasons):
        symbol = trap_symbol(reason)
        if symbol not in program.symbol_addresses:
            program.register_service(symbol, callback(reason))


def linux_ecall(vm, args: tuple[int, ...]):
    # Boot-first host ABI:
    #   service 1: write(ptr, len) to the host boot console.
    if len(args) != 3:
        raise VMError(f"Linux ecall expects service,ptr,len; got {len(args)} args")
    service, ptr, size = args
    if service != 1:
        raise VMError(f"unsupported MiniMachine Linux ecall service: {service}")
    if size > 1 << 20:
        raise VMError(f"MiniMachine Linux console write too large: {size}")
    data = bytes(vm.memory.read(ptr + i, 8) for i in range(size))
    text = data.decode("utf-8", errors="replace")
    sys.stdout.write(text)
    sys.stdout.flush()
    return None


def current_instruction(vm):
    if vm.current_function is None or vm.current_block is None:
        return None
    linked = vm.program.functions.get(vm.current_function)
    if linked is None:
        return None
    block = linked.block_map.get(vm.current_block)
    if block is None or vm.ip >= len(block.instructions):
        return None
    return block.instructions[vm.ip]


def main() -> int:
    args = parse_args()
    llvm_text = args.input.read_text()
    linker_contract = LinkerContract.load(args.linker_contract)

    functions, _stats = legalize_module(llvm_text)
    image = parse_module_image(llvm_text)
    image_sections = {
        obj.section for obj in image.objects if obj.section is not None
    }

    strict_source = []
    p3_functions = []
    reasons: set[str] = set()

    for function in functions:
        verify_muir(function)
        reasons.update(trap_reasons(function))
        expanded, _abi_stats = expand_function(function)
        escapes = arch_escapes(expanded)
        if escapes:
            print(
                "BOOT_EXEC_BLOCKED "
                f"stage=arch_escape function={function.name} sites={len(escapes)}"
            )
            return 1
        lowered = lower_function(expanded)
        verify_p3(lowered)
        strict_source.append(function)
        p3_functions.append(lowered)

    program = Program(p3_functions)
    surface = collect_runtime_surface(strict_source)
    install_runtime(program, surface)
    register_traps(program, reasons)

    missing_helpers = tuple(getattr(program, "runtime_missing_helpers", ()))
    missing_systems = tuple(getattr(program, "runtime_missing_systems", ()))
    if missing_helpers or missing_systems:
        print(
            "BOOT_EXEC_BLOCKED "
            f"stage=runtime helpers={len(missing_helpers)} systems={len(missing_systems)}"
        )
        return 1

    install_module_image(
        program,
        image,
        symbol_aliases=dict(linker_contract.aliases),
        linker_contract=linker_contract,
    )

    if args.entry not in program.functions:
        print(f"BOOT_EXEC_BLOCKED stage=entry missing={args.entry}")
        return 1

    print(
        "BOOT_EXEC_START "
        f"entry={args.entry} functions={len(p3_functions)} "
        f"image_objects={len(image.objects)} image_bytes={image.byte_size} "
        f"linker_boundaries={len(linker_contract.active_boundary_symbols(image_sections))}"
    )

    vm = program.new_vm()
    vm.ecall_handler = linux_ecall
    try:
        vm.run_function(
            args.entry,
            (),
            result_count=0,
            max_steps=args.max_steps,
        )
    except VMError as exc:
        inst = current_instruction(vm)
        print(
            "BOOT_EXEC_BLOCKED "
            f"stage=execute steps={vm.steps} function={vm.current_function} "
            f"block={vm.current_block} ip={vm.ip} sp=0x{vm.sp:x} "
            f"error={exc}"
        )
        if inst is not None:
            print(f"BOOT_EXEC_NEXT {inst!r}")
        return 1

    print(
        "BOOT_EXEC_HALTED "
        f"steps={vm.steps} function={vm.current_function} "
        f"block={vm.current_block} ip={vm.ip}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
