from __future__ import annotations

from dataclasses import fields, is_dataclass
import struct

from .. import muir
from ..vm import VMError


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


_LINUX_SEMANTIC_STACK_BYTES = 0x01000000
_LINUX_SEMANTIC_STACK_START_GAP = 0x40000000
