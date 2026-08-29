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
from src.minimachine.abi import CALLER_SP, RESULT_COUNT, RESULT_PTR, RET_PC, expand_function
from src.minimachine.checkpoint import (
    CheckpointError,
    image_fingerprint,
    load_checkpoint,
    save_checkpoint,
)
from src.minimachine.image import ImageError, install_module_image, parse_module_image
from src.minimachine.legalize import legalize_module
from src.minimachine.layout import DataLayout
from src.minimachine.kallsyms import install_p3_kallsyms
from src.minimachine.linker import LinkerContract
from src.minimachine.lower_p3 import lower_function
from src.minimachine.program_cache import (
    ProgramCache,
    ProgramCacheError,
    load_program_cache,
    save_program_cache,
)
from src.minimachine.runtime import (
    accelerate_direct_runtime,
    collect_runtime_surface,
    install_runtime,
)
from src.minimachine.verify import verify_muir, verify_p3
from src.minimachine.vm import HOST_CONTROL_TRANSFER, Program, VMError


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
    p.add_argument(
        "--probe-kallsyms",
        metavar="SYMBOL",
        help="probe Linux kallsyms against one P3 function and exit",
    )
    p.add_argument(
        "--checkpoint-in",
        type=Path,
        help="resume VM state from a checkpoint for the exact linked image",
    )
    p.add_argument(
        "--checkpoint-out",
        type=Path,
        help="write a resumable VM checkpoint during replay",
    )
    p.add_argument(
        "--checkpoint-initcall",
        help="write --checkpoint-out on entry to this Linux initcall",
    )
    p.add_argument(
        "--checkpoint-after-initcall",
        help="write --checkpoint-out when the initcall after this one begins",
    )
    p.add_argument(
        "--checkpoint-function",
        help="write --checkpoint-out on an occurrence of this function entry",
    )
    p.add_argument(
        "--checkpoint-function-hit",
        type=int,
        default=1,
        help="1-based occurrence of --checkpoint-function to capture",
    )
    p.add_argument(
        "--stop-after-checkpoint",
        action="store_true",
        help="stop replay immediately after writing a checkpoint",
    )
    p.add_argument(
        "--program-cache-in",
        type=Path,
        help="load a lowered P3 program cache for the exact linked image",
    )
    p.add_argument(
        "--program-cache-out",
        type=Path,
        help="write the lowered P3 program before runtime callbacks are bound",
    )
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
    #   service 2: context_switch(prev,next,fresh_sp,start_fn,start_arg)
    if not args:
        raise VMError("Linux ecall requires a service number")

    service = args[0]
    if service == 1:
        if len(args) != 3:
            raise VMError(
                f"Linux console ecall expects service,ptr,len; got {len(args)} args"
            )
        _, ptr, size = args
        if size > 1 << 20:
            raise VMError(f"MiniMachine Linux console write too large: {size}")
        data = bytes(vm.memory.read(ptr + i, 8) for i in range(size))
        text = data.decode("utf-8", errors="replace")
        sys.stdout.write(text)
        sys.stdout.flush()
        return None

    if service == 2:
        if len(args) != 6:
            raise VMError(
                "Linux context-switch ecall expects "
                "service,prev,next,fresh_sp,start_fn,start_arg"
            )
        _, prev, next_task, fresh_sp, start_fn, start_arg = args

        result_count = vm.memory.read(vm.sp + RESULT_COUNT, 64)
        if result_count != 1:
            raise VMError(
                f"Linux context switch expects one result, got {result_count}"
            )

        contexts = getattr(vm, "linux_task_contexts", None)
        if contexts is None:
            contexts = {}
            vm.linux_task_contexts = contexts

        # Save where __switch_to() must continue when this task is resumed,
        # together with the system-call result slot carrying the last task.
        contexts[prev] = (
            vm.memory.read(vm.sp + CALLER_SP, 64),
            vm.memory.read(vm.sp + RET_PC, 64),
            vm.memory.read(vm.sp + RESULT_PTR, 64),
        )

        saved = contexts.get(next_task)
        if saved is not None:
            resume_sp, resume_pc, result_ptr = saved
            vm.memory.write(result_ptr, 64, prev)
            vm.sp = resume_sp
            vm.halted = False
            vm._set_code(resume_pc)
            return HOST_CONTROL_TRANSFER

        if not fresh_sp or not start_fn:
            raise VMError(
                "first MiniMachine task switch lacks fresh stack/function: "
                f"next=0x{next_task:x} sp=0x{fresh_sp:x} fn=0x{start_fn:x}"
            )
        if "minimachine_ret_from_fork" not in vm.program.functions:
            raise VMError("MiniMachine ret-from-fork trampoline is missing")

        shadow_stacks = getattr(vm, "linux_task_shadow_stacks", None)
        if shadow_stacks is None:
            shadow_stacks = {}
            vm.linux_task_shadow_stacks = shadow_stacks

        shadow_top = shadow_stacks.get(next_task)
        if shadow_top is None:
            shadow_top = vm.linux_shadow_stack_next
            vm.linux_shadow_stack_next -= 0x01000000
            if vm.linux_shadow_stack_next <= vm.heap_next:
                raise VMError("MiniMachine Linux shadow task stacks exhausted")
            shadow_stacks[next_task] = shadow_top
            print(
                "BOOT_EXEC_TASK_SHADOW_STACK "
                f"task=0x{next_task:x} guest_sp=0x{fresh_sp:x} "
                f"p3_stack_top=0x{shadow_top:x}",
                flush=True,
            )

        vm.enter_function(
            "minimachine_ret_from_fork",
            (prev, start_fn, start_arg),
            stack_top=shadow_top,
            result_count=0,
        )
        return HOST_CONTROL_TRANSFER

    raise VMError(f"unsupported MiniMachine Linux ecall service: {service}")


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


def dump_current_call_frame(vm) -> None:
    sp = vm.sp
    header = (
        ("caller_sp", CALLER_SP),
        ("ret_pc", RET_PC),
        ("entry", 16),
        ("frame_size", 24),
        ("result_ptr", RESULT_PTR),
        ("result_count", RESULT_COUNT),
        ("resume_pc", 48),
        ("arg_count", 56),
    )
    print(
        "BOOT_EXEC_CALL_FRAME "
        + " ".join(
            f"{name}=0x{vm.memory.read(sp + off, 64):x}"
            for name, off in header
        ),
        flush=True,
    )

    caller_sp = vm.memory.read(sp + CALLER_SP, 64)
    if not caller_sp or vm.current_function is None:
        return
    linked = vm.program.functions.get(vm.current_function)
    if linked is None:
        return

    interesting = []
    for name, off in sorted(linked.slot_offsets.items(), key=lambda item: item[1]):
        value = vm.memory.read(caller_sp + off, 64)
        if (
            name.startswith("__abi_")
            or value in vm.program.symbol_addresses.values()
            or value >= 0x10000
        ):
            interesting.append((name, off, value))
    for name, off, value in interesting[-80:]:
        print(
            "BOOT_EXEC_CALLER_SLOT "
            f"name={name} offset={off} address=0x{caller_sp + off:x} "
            f"value=0x{value:x}",
            flush=True,
        )

    entry = vm.memory.read(sp + 16, 64)
    descriptor_candidates = [
        (name, value)
        for name, _off, value in interesting
        if value and vm.memory.read(value, 64) == entry
    ]
    for name, descriptor in descriptor_candidates[-20:]:
        print(
            "BOOT_EXEC_DESCRIPTOR "
            f"slot={name} address=0x{descriptor:x} "
            f"entry=0x{vm.memory.read(descriptor, 64):x} "
            f"frame=0x{vm.memory.read(descriptor + 8, 64):x}",
            flush=True,
        )


def dump_scheduler_indirect_call(vm) -> None:
    if vm.current_function != "dequeue_task":
        return

    frame_size = vm.memory.read(vm.sp + 24, 64)
    argc = vm.memory.read(vm.sp + 56, 64)
    if argc < 2:
        return

    arg_base = vm.sp + frame_size
    rq = vm.memory.read(arg_base, 64)
    task = vm.memory.read(arg_base + 8, 64)
    flags = vm.memory.read(arg_base + 16, 64) if argc >= 3 else 0

    sched_off = getattr(vm, "linux_task_sched_class_offset", None)
    print(
        "BOOT_EXEC_SCHED_CALL "
        f"rq=0x{rq:x} task=0x{task:x} flags=0x{flags:x} "
        f"sched_class_offset={sched_off}",
        flush=True,
    )
    if sched_off is None or not task:
        return

    sched_class = vm.memory.read(task + sched_off, 64)
    dequeue_desc = vm.memory.read(sched_class + 8, 64) if sched_class else 0
    entry = vm.memory.read(dequeue_desc, 64) if dequeue_desc else 0
    desc_frame = vm.memory.read(dequeue_desc + 8, 64) if dequeue_desc else 0
    print(
        "BOOT_EXEC_SCHED_DESCRIPTOR "
        f"sched_class=0x{sched_class:x} dequeue_desc=0x{dequeue_desc:x} "
        f"entry=0x{entry:x} frame_size=0x{desc_frame:x}",
        flush=True,
    )

    for symbol in (
        "dequeue_task_idle",
        "dequeue_task_fair",
        "dequeue_task_rt",
        "dequeue_task_dl",
    ):
        desc = vm.program.symbol_addresses.get(symbol)
        if desc is None:
            continue
        print(
            "BOOT_EXEC_EXPECTED_DESCRIPTOR "
            f"symbol={symbol} address=0x{desc:x} "
            f"entry=0x{vm.memory.read(desc, 64):x} "
            f"frame_size=0x{vm.memory.read(desc + 8, 64):x}",
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
    linked_image_sha256 = image_fingerprint(llvm_text)
    linker_contract = LinkerContract.load(args.linker_contract)

    if args.program_cache_in is not None:
        try:
            cached = load_program_cache(
                args.program_cache_in,
                image_sha256=linked_image_sha256,
            )
        except ProgramCacheError as exc:
            print(f"BOOT_EXEC_BLOCKED stage=program-cache error={exc}")
            return 1
        program = cached.program
        surface = cached.surface
        reasons = set(cached.reasons)
        blocked_functions = list(cached.blocked_functions)
        image = cached.image
        task_sched_class_offset = cached.task_sched_class_offset
        p3_function_count = cached.function_count
        print(
            "BOOT_EXEC_PROGRAM_CACHE_LOADED "
            f"path={args.program_cache_in} functions={p3_function_count}",
            flush=True,
        )
    else:
        functions, _stats = legalize_module(llvm_text)
        image = parse_module_image(llvm_text)

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
        layout = DataLayout.from_module(llvm_text)
        task_info = layout.info("%struct.task_struct")
        task_sched_class_offset = (
            task_info.field_offsets[15]
            if task_info.field_offsets is not None
            and len(task_info.field_offsets) > 15
            else None
        )
        p3_function_count = len(p3_functions)

        if args.program_cache_out is not None:
            save_program_cache(
                ProgramCache(
                    image_sha256=linked_image_sha256,
                    program=program,
                    surface=surface,
                    reasons=frozenset(reasons),
                    blocked_functions=tuple(blocked_functions),
                    image=image,
                    task_sched_class_offset=task_sched_class_offset,
                ),
                args.program_cache_out,
            )
            print(
                "BOOT_EXEC_PROGRAM_CACHE_SAVED "
                f"path={args.program_cache_out} functions={p3_function_count}",
                flush=True,
            )

    image_sections = {
        obj.section for obj in image.objects if obj.section is not None
    }

    install_runtime(program, surface)
    accelerated_runtime = accelerate_direct_runtime(program)
    if accelerated_runtime:
        print(
            "BOOT_EXEC_FAST_RUNTIME symbols=" + ",".join(accelerated_runtime),
            flush=True,
        )
    register_traps(program, reasons)

    missing_helpers = tuple(getattr(program, "runtime_missing_helpers", ()))
    missing_systems = tuple(getattr(program, "runtime_missing_systems", ()))
    if missing_helpers or missing_systems:
        print(
            "BOOT_EXEC_BLOCKED "
            f"stage=runtime helpers={len(missing_helpers)} systems={len(missing_systems)}"
        )
        return 1

    # Native Linux adds kallsyms during final ELF linking. P3 has its own
    # final code-address domain, so synthesize the same generated globals
    # before installing the module image. This ensures semantic _end includes
    # the tables and Linux reserves their storage during setup_arch.
    try:
        installed_kallsyms = install_p3_kallsyms(
            program,
            external_data=image.external_data,
        )
    except VMError as exc:
        print(f"BOOT_EXEC_BLOCKED stage=kallsyms error={exc}")
        return 1
    if installed_kallsyms:
        print(
            "BOOT_EXEC_KALLSYMS "
            f"generated={len(installed_kallsyms)} "
            f"symbols={program.initial_memory.read(program.symbol_addresses['kallsyms_num_syms'], 32)}",
            flush=True,
        )

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
        f"entry={args.entry} functions={p3_function_count} "
        f"blocked_functions={len(blocked_functions)} "
        f"arch_escape_sites={sum(sites for _, sites in blocked_functions)} "
        f"image_objects={len(image.objects)} image_bytes={image.byte_size} "
        f"linker_boundaries={len(linker_contract.active_boundary_symbols(image_sections))}"
    )

    vm = program.new_vm()
    vm.ecall_handler = linux_ecall
    vm.linux_task_contexts = {}
    vm.linux_task_shadow_stacks = {}
    # Reserve the top 16 MiB for the boot task's existing P3 stack. New
    # Linux tasks receive independent P3 continuation stacks below it.
    vm.linux_shadow_stack_next = vm.stack_top - 0x01000000
    if task_sched_class_offset is not None:
        vm.linux_task_sched_class_offset = task_sched_class_offset

    resumed_from_checkpoint = False
    if args.checkpoint_in is not None:
        try:
            load_checkpoint(
                vm,
                args.checkpoint_in,
                image_sha256=linked_image_sha256,
            )
        except CheckpointError as exc:
            print(f"BOOT_EXEC_BLOCKED stage=checkpoint error={exc}")
            return 1
        vm.ecall_handler = linux_ecall
        resumed_from_checkpoint = True
        print(
            "BOOT_EXEC_CHECKPOINT_RESTORED "
            f"path={args.checkpoint_in} steps={vm.steps} "
            f"function={vm.current_function} block={vm.current_block} ip={vm.ip}",
            flush=True,
        )

    if args.probe_kallsyms is not None:
        symbol = args.probe_kallsyms
        linked = program.functions.get(symbol)
        if linked is None or not linked.function.blocks:
            print(f"BOOT_EXEC_BLOCKED stage=kallsyms-probe missing={symbol}")
            return 1
        pc = program.block_code[(symbol, linked.function.blocks[0].label)]
        size_ptr = vm.alloc_bytes(8, align=8)
        offset_ptr = vm.alloc_bytes(8, align=8)
        name_ptr = vm.alloc_bytes(512, align=8)
        try:
            found = vm.run_function(
                "kallsyms_lookup_size_offset",
                (pc, size_ptr, offset_ptr),
                result_count=1,
                max_steps=args.max_steps,
            )
            named = vm.run_function(
                "lookup_symbol_name",
                (pc, name_ptr),
                result_count=1,
                max_steps=args.max_steps,
            )
        except VMError as exc:
            print(f"BOOT_EXEC_BLOCKED stage=kallsyms-probe error={exc}")
            return 1
        raw_name = bytearray()
        for i in range(512):
            byte = vm.memory.read(name_ptr + i, 8)
            if byte == 0:
                break
            raw_name.append(byte)
        decoded = raw_name.decode("utf-8", errors="replace")
        size = vm.memory.read(size_ptr, 64)
        offset = vm.memory.read(offset_ptr, 64)
        print(
            "BOOT_EXEC_KALLSYMS_PROBE "
            f"symbol={symbol} pc=0x{pc:x} found={found[0]} "
            f"name_rc={named[0]} name={decoded} size={size} offset={offset}",
            flush=True,
        )
        if found[0] != 1 or named[0] != 0 or decoded != symbol:
            print("BOOT_EXEC_BLOCKED stage=kallsyms-probe mismatch=1")
            return 1
        return 0

    last_milestone_function = None
    milestone_functions = {
        "sched_init",
        "sched_fork",
        "early_irq_init",
        "init_IRQ",
        "tick_init",
        "init_timers",
        "srcu_init",
        "hrtimers_init",
        "softirq_init",
        "timekeeping_init",
        "time_init",
        "console_init",
        "arch_call_rest_init",
        "rest_init",
        "minimachine_ret_from_fork",
        "kernel_init",
        "kernel_init_freeable",
        "init_post",
        "console_on_rootfs",
        "do_mounts_initrd",
        "prepare_namespace",
        "do_basic_setup",
        "do_pre_smp_initcalls",
        "do_initcalls",
        "do_initcall_level",
        "do_one_initcall",
        "wait_for_initramfs",
        "async_synchronize_full",
        "free_initmem",
        "mark_readonly",
        "run_init_process",
        "try_to_run_init_process",
        "kernel_execve",
        "kthreadd",
    }

    sched_watch_installed = False

    # Track initcall identity only on real function-entry transitions.  This
    # avoids wrapping every P3 instruction just to discover coarse milestones.
    symbols_by_address: dict[int, list[str]] = {}
    for symbol, address in program.symbol_addresses.items():
        symbols_by_address.setdefault(address, []).append(symbol)
    last_initcall_enter_step: int | None = (
        vm.steps if resumed_from_checkpoint and args.checkpoint_initcall else None
    )
    last_initcall_symbol = (
        args.checkpoint_initcall
        if resumed_from_checkpoint and args.checkpoint_initcall
        else "<none>"
    )
    checkpoint_written = False
    checkpoint_after_armed = bool(
        resumed_from_checkpoint
        and args.checkpoint_after_initcall is not None
        and args.checkpoint_initcall == args.checkpoint_after_initcall
    )
    if checkpoint_after_armed:
        print(
            "BOOT_EXEC_CHECKPOINT_ARMED "
            f"after_initcall={args.checkpoint_after_initcall} "
            f"steps={vm.steps} source=resume",
            flush=True,
        )
    checkpoint_function_hits = 0

    def install_sched_watch() -> None:
        nonlocal sched_watch_installed
        if sched_watch_installed:
            return
        linked = vm.program.functions.get("sched_fork")
        sched_off = getattr(vm, "linux_task_sched_class_offset", None)
        if linked is None or sched_off is None:
            return
        frame_size = vm.memory.read(vm.sp + 24, 64)
        argc = vm.memory.read(vm.sp + 56, 64)
        if argc < 2:
            return
        arg_base = vm.sp + frame_size
        task = vm.memory.read(arg_base + 8, 64)
        watched = task + sched_off
        print(
            "BOOT_EXEC_SCHED_WATCH_START "
            f"steps={vm.steps} task=0x{task:x} address=0x{watched:x} "
            f"value=0x{vm.memory.read(watched, 64):x}",
            flush=True,
        )
        original_memory_write = vm.memory.write

        def watched_write(address, bits, value):
            byte_count = bits // 8
            if address <= watched < address + byte_count:
                before = vm.memory.read(watched, 64)
                original_memory_write(address, bits, value)
                after = vm.memory.read(watched, 64)
                print(
                    "BOOT_EXEC_SCHED_WATCH_WRITE "
                    f"steps={vm.steps} function={vm.current_function} "
                    f"block={vm.current_block} ip={vm.ip} "
                    f"address=0x{address:x} bits={bits} "
                    f"before=0x{before:x} after=0x{after:x}",
                    flush=True,
                )
                return
            original_memory_write(address, bits, value)

        vm.memory.write = watched_write
        sched_watch_installed = True

    def observe_function_entry(function: str) -> None:
        nonlocal last_milestone_function
        nonlocal last_initcall_enter_step, last_initcall_symbol
        nonlocal checkpoint_written, checkpoint_after_armed
        nonlocal checkpoint_function_hits

        if function == "do_one_initcall":
            frame_size = vm.memory.read(vm.sp + 24, 64)
            argc = vm.memory.read(vm.sp + 56, 64)
            initcall_ptr = vm.memory.read(vm.sp + frame_size, 64) if argc else 0
            names = symbols_by_address.get(initcall_ptr, ())
            initcall_symbol = "|".join(sorted(names)[:4]) if names else "<unknown>"
            since_previous = (
                vm.steps - last_initcall_enter_step
                if last_initcall_enter_step is not None
                else 0
            )
            print(
                "BOOT_EXEC_INITCALL_ENTER "
                f"steps={vm.steps} since_previous={since_previous} "
                f"ptr=0x{initcall_ptr:x} symbol={initcall_symbol}",
                flush=True,
            )
            previous_initcall_symbol = last_initcall_symbol
            last_initcall_enter_step = vm.steps
            last_initcall_symbol = initcall_symbol

            if (
                not checkpoint_written
                and checkpoint_after_armed
                and args.checkpoint_out is not None
            ):
                save_checkpoint(
                    vm,
                    args.checkpoint_out,
                    image_sha256=linked_image_sha256,
                )
                checkpoint_written = True
                checkpoint_after_armed = False
                print(
                    "BOOT_EXEC_CHECKPOINT_SAVED "
                    f"path={args.checkpoint_out} steps={vm.steps} "
                    f"after_initcall={previous_initcall_symbol} "
                    f"next_initcall={initcall_symbol}",
                    flush=True,
                )

            if (
                not checkpoint_written
                and args.checkpoint_out is not None
                and args.checkpoint_initcall == initcall_symbol
            ):
                save_checkpoint(
                    vm,
                    args.checkpoint_out,
                    image_sha256=linked_image_sha256,
                )
                checkpoint_written = True
                print(
                    "BOOT_EXEC_CHECKPOINT_SAVED "
                    f"path={args.checkpoint_out} steps={vm.steps} "
                    f"initcall={initcall_symbol}",
                    flush=True,
                )

            if (
                not checkpoint_written
                and args.checkpoint_out is not None
                and args.checkpoint_after_initcall == initcall_symbol
            ):
                checkpoint_after_armed = True
                print(
                    "BOOT_EXEC_CHECKPOINT_ARMED "
                    f"after_initcall={initcall_symbol} steps={vm.steps}",
                    flush=True,
                )

        if (
            not checkpoint_written
            and args.checkpoint_function is not None
            and function == args.checkpoint_function
        ):
            checkpoint_function_hits += 1
            if checkpoint_function_hits == args.checkpoint_function_hit:
                if args.checkpoint_out is None:
                    raise VMError(
                        "--checkpoint-function requires --checkpoint-out"
                    )
                save_checkpoint(
                    vm,
                    args.checkpoint_out,
                    image_sha256=linked_image_sha256,
                )
                checkpoint_written = True
                print(
                    "BOOT_EXEC_CHECKPOINT_SAVED "
                    f"path={args.checkpoint_out} steps={vm.steps} "
                    f"function={function} hit={checkpoint_function_hits}",
                    flush=True,
                )
                if args.stop_after_checkpoint:
                    vm.halted = True

        if function == "sched_fork":
            install_sched_watch()

        if function in milestone_functions:
            print(
                "BOOT_EXEC_MILESTONE "
                f"steps={vm.steps} function={function}",
                flush=True,
            )
            last_milestone_function = function

    traced_entry_codes: dict[int, str] = {}
    for function in milestone_functions:
        linked = vm.program.functions.get(function)
        if linked is None or not linked.function.blocks:
            continue
        entry_block = linked.function.blocks[0].label
        traced_entry_codes[
            vm.program.block_code[(function, entry_block)]
        ] = function

    if args.checkpoint_function is not None:
        linked = vm.program.functions.get(args.checkpoint_function)
        if linked is None or not linked.function.blocks:
            print(
                "BOOT_EXEC_BLOCKED stage=checkpoint "
                f"missing_function={args.checkpoint_function}"
            )
            return 1
        entry_block = linked.function.blocks[0].label
        traced_entry_codes[
            vm.program.block_code[(args.checkpoint_function, entry_block)]
        ] = args.checkpoint_function

    original_set_code = vm._set_code

    def traced_set_code(code: int) -> None:
        function = traced_entry_codes.get(code)
        original_set_code(code)
        if function is not None:
            observe_function_entry(function)

    vm._set_code = traced_set_code

    original_enter_function = vm.enter_function

    def traced_enter_function(name, args=(), *, stack_top, result_count=0):
        original_enter_function(
            name,
            args,
            stack_top=stack_top,
            result_count=result_count,
        )
        if name in milestone_functions:
            observe_function_entry(name)

    vm.enter_function = traced_enter_function

    # Progress sampling is optional.  Production replay uses zero so the hot
    # interpreter path is not wrapped once per P3 instruction.
    if args.progress_every > 0:
        original_step = vm.step
        next_progress = args.progress_every

        def progress_step():
            nonlocal next_progress
            if vm.steps >= next_progress:
                print(
                    "BOOT_EXEC_PROGRESS "
                    f"steps={vm.steps} function={vm.current_function} "
                    f"block={vm.current_block} ip={vm.ip} "
                    f"last_initcall={last_initcall_symbol}",
                    flush=True,
                )
                next_progress += args.progress_every
            original_step()

        vm.step = progress_step

    try:
        if resumed_from_checkpoint:
            resume_limit = vm.steps + args.max_steps
            print(
                "BOOT_EXEC_CHECKPOINT_BUDGET "
                f"saved_steps={vm.steps} additional_steps={args.max_steps} "
                f"absolute_limit={resume_limit}",
                flush=True,
            )
            vm.run(max_steps=resume_limit)
        else:
            vm.run_function(
                args.entry,
                (),
                result_count=0,
                max_steps=args.max_steps,
            )
    except VMError as exc:
        if args.checkpoint_function is not None:
            print(
                "BOOT_EXEC_CHECKPOINT_FUNCTION_PROGRESS "
                f"function={args.checkpoint_function} "
                f"hits={checkpoint_function_hits} steps={vm.steps}",
                flush=True,
            )
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
        dump_current_call_frame(vm)
        dump_scheduler_indirect_call(vm)
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
