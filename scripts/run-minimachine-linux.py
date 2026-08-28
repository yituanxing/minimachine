#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.image import ImageError, install_module_image, parse_module_image
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
    p.add_argument("--max-steps", type=int, default=10_000_000)
    p.add_argument("--progress-every", type=int, default=250_000)
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


def referenced_slots(value):
    seen: set[int] = set()

    def walk(obj):
        identity = id(obj)
        if identity in seen:
            return
        seen.add(identity)

        if isinstance(obj, muir.Slot):
            yield obj.name
            return
        if is_dataclass(obj):
            for field in fields(obj):
                yield from walk(getattr(obj, field.name))
            return
        if isinstance(obj, (tuple, list, set, frozenset)):
            for item in obj:
                yield from walk(item)
            return
        if isinstance(obj, dict):
            for key, item in obj.items():
                yield from walk(key)
                yield from walk(item)

    yield from walk(value)


def dump_instruction_slots(vm, inst) -> None:
    if inst is None or vm.current_function is None:
        return
    linked = vm.program.functions.get(vm.current_function)
    if linked is None:
        return

    for name in sorted(set(referenced_slots(inst))):
        offset = linked.slot_offsets.get(name)
        if offset is None:
            continue
        cell = (vm.sp + offset) & ((1 << 64) - 1)
        value = vm.memory.read(cell, 64)
        preview = bytes(vm.memory.read(value + i, 8) for i in range(16))
        print(
            "BOOT_EXEC_SLOT "
            f"name={name} cell=0x{cell:x} value=0x{value:x} "
            f"mem16={preview.hex()}",
            flush=True,
        )


def dump_linux_memory_state(vm) -> None:
    def word(name: str, index: int = 0):
        address = vm.program.symbol_addresses.get(name)
        if address is None:
            print(f"BOOT_EXEC_MEM symbol={name} missing=1", flush=True)
            return None
        value = vm.memory.read(address + index * 8, 64)
        print(
            f"BOOT_EXEC_MEM symbol={name} index={index} "
            f"address=0x{address + index * 8:x} value={value} hex=0x{value:x}",
            flush=True,
        )
        return value

    for symbol in (
        "memory_start",
        "memory_end",
        "min_low_pfn",
        "max_low_pfn",
        "max_pfn",
        "nr_kernel_pages",
        "nr_all_pages",
    ):
        word(symbol)

    for index in range(2):
        word("arch_zone_lowest_possible_pfn", index)
        word("arch_zone_highest_possible_pfn", index)

    memblock = vm.program.symbol_addresses.get("memblock")
    if memblock is None:
        print("BOOT_EXEC_MEM symbol=memblock missing=1", flush=True)
        return

    # Linux 6.6 struct memblock on this target:
    #   u8 bottom_up; u64 current_limit;
    #   memblock_type memory; memblock_type reserved;
    # and memblock_type is {cnt,max,total_size,regions,name}.
    memory_cnt = vm.memory.read(memblock + 16, 64)
    memory_max = vm.memory.read(memblock + 24, 64)
    memory_total = vm.memory.read(memblock + 32, 64)
    memory_regions = vm.memory.read(memblock + 40, 64)
    reserved_cnt = vm.memory.read(memblock + 56, 64)
    reserved_total = vm.memory.read(memblock + 72, 64)
    reserved_regions = vm.memory.read(memblock + 80, 64)
    print(
        "BOOT_EXEC_MEMBLOCK "
        f"address=0x{memblock:x} memory_cnt={memory_cnt} "
        f"memory_max={memory_max} memory_total={memory_total} "
        f"memory_regions=0x{memory_regions:x} reserved_cnt={reserved_cnt} "
        f"reserved_total={reserved_total} reserved_regions=0x{reserved_regions:x}",
        flush=True,
    )

    for kind, count, regions in (
        ("memory", memory_cnt, memory_regions),
        ("reserved", reserved_cnt, reserved_regions),
    ):
        for index in range(min(count, 4)):
            region = regions + index * 24
            base = vm.memory.read(region + 0, 64)
            size = vm.memory.read(region + 8, 64)
            flags = vm.memory.read(region + 16, 32)
            print(
                "BOOT_EXEC_REGION "
                f"kind={kind} index={index} address=0x{region:x} "
                f"base=0x{base:x} size=0x{size:x} flags=0x{flags:x}",
                flush=True,
            )


def probe_linux_memory_helpers(vm) -> None:
    def run(name: str, args: tuple[int, ...], result_count: int = 1):
        if name not in vm.program.functions:
            print(f"BOOT_EXEC_PROBE function={name} missing=1", flush=True)
            return None
        try:
            result = vm.run_function(
                name,
                args,
                result_count=result_count,
                max_steps=500_000,
            )
        except VMError as exc:
            print(
                f"BOOT_EXEC_PROBE function={name} error={exc}",
                flush=True,
            )
            return None
        print(
            f"BOOT_EXEC_PROBE function={name} result={result}",
            flush=True,
        )
        return result

    memblock = vm.program.symbol_addresses.get("memblock")
    if memblock is not None:
        regions = vm.memory.read(memblock + 40, 64)
        run("memblock_get_region_node", (regions,))
        run("numa_valid_node", (0,))

        index = vm.alloc_bytes(8, align=8)
        direct_start = vm.alloc_bytes(8, align=8)
        direct_end = vm.alloc_bytes(8, align=8)
        direct_nid = vm.alloc_bytes(8, align=8)
        vm.memory.write(index, 32, 0xFFFFFFFF)
        vm.memory.write(direct_start, 64, 0)
        vm.memory.write(direct_end, 64, 0)
        vm.memory.write(direct_nid, 32, 0xFFFFFFFF)
        run(
            "__next_mem_pfn_range",
            (index, 0, direct_start, direct_end, direct_nid),
            result_count=0,
        )
        print(
            "BOOT_EXEC_PROBE_NEXT_PFN "
            f"index={vm.memory.read(index, 32)} "
            f"start={vm.memory.read(direct_start, 64)} "
            f"end={vm.memory.read(direct_end, 64)} "
            f"nid={vm.memory.read(direct_nid, 32)}",
            flush=True,
        )

    start = vm.alloc_bytes(8, align=8)
    end = vm.alloc_bytes(8, align=8)
    run("get_pfn_range_for_nid", (0, start, end), result_count=0)
    start_pfn = vm.memory.read(start, 64)
    end_pfn = vm.memory.read(end, 64)
    print(
        "BOOT_EXEC_PROBE_PFN "
        f"start={start_pfn} end={end_pfn}",
        flush=True,
    )

    if end_pfn >= start_pfn:
        run(
            "__absent_pages_in_range",
            (0, start_pfn, end_pfn),
        )

        zone_start = vm.alloc_bytes(8, align=8)
        zone_end = vm.alloc_bytes(8, align=8)
        run(
            "zone_spanned_pages_in_node",
            (0, 0, start_pfn, end_pfn, zone_start, zone_end),
        )
        print(
            "BOOT_EXEC_PROBE_ZONE "
            f"start={vm.memory.read(zone_start, 64)} "
            f"end={vm.memory.read(zone_end, 64)}",
            flush=True,
        )


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
    blocked_functions: list[tuple[str, int]] = []
    reasons: set[str] = set()

    for function in functions:
        verify_muir(function)
        reasons.update(trap_reasons(function))
        expanded, _abi_stats = expand_function(function)
        escapes = arch_escapes(expanded)
        if escapes:
            blocked_functions.append((function.name, len(escapes)))
            continue
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

    try:
        install_module_image(
            program,
            image,
            symbol_aliases=dict(linker_contract.aliases),
            linker_contract=linker_contract,
        )
    except (ImageError, VMError, ValueError) as exc:
        print(f"BOOT_EXEC_BLOCKED stage=image error={exc}")
        return 1

    if args.entry not in program.functions:
        print(f"BOOT_EXEC_BLOCKED stage=entry missing={args.entry}")
        return 1

    print(
        "BOOT_EXEC_START "
        f"entry={args.entry} functions={len(p3_functions)} "
        f"blocked_functions={len(blocked_functions)} "
        f"arch_escape_sites={sum(sites for _, sites in blocked_functions)} "
        f"image_objects={len(image.objects)} image_bytes={image.byte_size} "
        f"linker_boundaries={len(linker_contract.active_boundary_symbols(image_sections))}"
    )

    vm = program.new_vm()
    vm.ecall_handler = linux_ecall

    original_step = vm.step
    next_progress = args.progress_every

    def traced_step():
        nonlocal next_progress
        if args.progress_every > 0 and vm.steps >= next_progress:
            print(
                "BOOT_EXEC_PROGRESS "
                f"steps={vm.steps} function={vm.current_function} "
                f"block={vm.current_block} ip={vm.ip}",
                flush=True,
            )
            next_progress += args.progress_every
        original_step()

    vm.step = traced_step

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
            print(f"BOOT_EXEC_NEXT {inst!r}", flush=True)
            dump_instruction_slots(vm, inst)
        dump_linux_memory_state(vm)
        probe_linux_memory_helpers(vm)
        return 1

    print(
        "BOOT_EXEC_HALTED "
        f"steps={vm.steps} function={vm.current_function} "
        f"block={vm.current_block} ip={vm.ip}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
