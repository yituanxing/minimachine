from __future__ import annotations

from pathlib import Path
import time

from ..abi import CALLER_SP, RESULT_COUNT, RET_PC
from ..vm import HOST_CONTROL_TRANSFER, VMError


_LINUX_SEMANTIC_STACK_BYTES = 0x01000000
_LINUX_SEMANTIC_STACK_START_GAP = 0x40000000


def _linux_semantic_call_stack_top(vm, task: int, depth: int) -> int:
    stacks = getattr(vm, "linux_task_semantic_stacks", None)
    if stacks is None:
        stacks = {}
        vm.linux_task_semantic_stacks = stacks

    key = (int(task), int(depth))
    top = stacks.get(key)
    if top is not None:
        return int(top)

    next_top = getattr(vm, "linux_semantic_stack_next", None)
    if next_top is None:
        next_top = vm.stack_top - _LINUX_SEMANTIC_STACK_START_GAP
    next_top = int(next_top)
    lower = next_top - _LINUX_SEMANTIC_STACK_BYTES
    if lower <= int(vm.heap_next):
        raise VMError(
            "MiniMachine Linux semantic call stacks exhausted: "
            f"task=0x{int(task):x} depth={depth} "
            f"next_top=0x{next_top:x} heap_next=0x{int(vm.heap_next):x}"
        )

    stacks[key] = next_top
    vm.linux_semantic_stack_next = lower
    print(
        "BOOT_EXEC_TASK_SEMANTIC_STACK "
        f"task=0x{int(task):x} depth={depth} "
        f"top=0x{next_top:x} bottom=0x{lower:x}",
        flush=True,
    )
    return next_top


def _call_linux_function_preserving_control(
    vm,
    name: str,
    args: tuple[int, ...],
    *,
    result_count: int,
    max_extra_steps: int = 2_000_000,
    preserve_linux_task_state: bool = False,
    call_task_override: int | None = None,
):
    if name not in vm.program.functions:
        raise VMError(f"rootfs injection missing Linux function: {name}")

    current_addr = vm.program.symbol_addresses.get(
        "minimachine_current_task"
    )
    saved_current = (
        vm.memory.read(current_addr, 64)
        if current_addr is not None
        else None
    )
    saved_contexts = dict(getattr(vm, "linux_task_contexts", {}))
    saved_shadow_stacks = dict(
        getattr(vm, "linux_task_shadow_stacks", {})
    )
    saved_shadow_next = getattr(vm, "linux_shadow_stack_next", None)
    saved = (
        vm.sp,
        vm.current_function,
        vm.current_block,
        vm.ip,
        vm.halted,
        vm.steps,
    )
    saved_active_user_task = int(
        getattr(vm, "active_user_task", 0) or 0
    )
    linked = vm.program.functions[name]
    previous_depth = int(getattr(vm, "_preserved_call_depth", 0))

    # Semantic Linux calls may remain parked while the scheduler runs another
    # task (vfork is the canonical case). A single fixed temporary P3 stack
    # therefore corrupts the parked task as soon as the child performs its own
    # syscall. Give every Linux task and same-task nesting level an independent
    # persistent semantic-call stack.
    # A userspace host callback can remain live while Linux has scheduled a
    # different task underneath an outer blocking semantic call. In that case
    # linux_current_task describes the kernel continuation, not the P3 caller
    # that owns this semantic syscall. Prefer the active userspace continuation
    # owner so task-scoped exec/exit transfers unwind the right call suffix.
    call_task = int(
        call_task_override
        if call_task_override is not None
        else (
            saved_active_user_task
            or getattr(vm, "linux_current_task", 0)
            or (saved_current if saved_current is not None else 0)
        )
    )
    task_depths = getattr(vm, "_preserved_task_depths", None)
    if task_depths is None:
        task_depths = {}
        vm._preserved_task_depths = task_depths
    task_depth = int(task_depths.get(call_task, 0))
    task_depths[call_task] = task_depth + 1

    call_tasks = getattr(vm, "_preserved_call_tasks", None)
    if call_tasks is None:
        call_tasks = []
        vm._preserved_call_tasks = call_tasks
    if len(call_tasks) != previous_depth:
        # Keep legacy callers that only manipulated the numeric depth usable,
        # while making real nested semantic calls explicit from here onward.
        if previous_depth == 0:
            call_tasks.clear()
        elif len(call_tasks) > previous_depth:
            del call_tasks[previous_depth:]
        else:
            call_tasks.extend([call_task] * (previous_depth - len(call_tasks)))
    call_tasks.append(call_task)

    # A semantic Linux call can return control to another task while its
    # kernel continuation remains parked inside the task's semantic stack.
    # Re-entering a syscall for that same task must not reuse the parked arena:
    # doing so overwrites __switch_to()'s dynamic return chain before the task
    # is scheduled back in.
    stack_depth = task_depth
    parked = getattr(vm, "linux_task_contexts", {}).get(call_task)
    while parked is not None:
        parked_sp = int(parked[0])
        semantic_stacks = getattr(vm, "linux_task_semantic_stacks", {})
        candidate_top = semantic_stacks.get((call_task, stack_depth))
        if candidate_top is None:
            break
        candidate_top = int(candidate_top)
        candidate_bottom = candidate_top - _LINUX_SEMANTIC_STACK_BYTES
        if not (candidate_bottom <= parked_sp < candidate_top):
            break
        print(
            "BOOT_EXEC_TASK_SEMANTIC_STACK_BUSY "
            f"task=0x{call_task:x} depth={stack_depth} "
            f"parked_sp=0x{parked_sp:x} "
            f"top=0x{candidate_top:x} bottom=0x{candidate_bottom:x}",
            flush=True,
        )
        stack_depth += 1

    temp_stack_top = _linux_semantic_call_stack_top(
        vm, call_task, stack_depth
    )
    result_words = max(1, result_count)
    total = linked.frame_size + len(args) * 8 + result_words * 8
    callee_sp = temp_stack_top - total
    result_base = callee_sp + linked.frame_size + len(args) * 8

    vm._preserved_call_depth = previous_depth + 1
    transfer_seen = False

    try:
        vm.enter_function(
            name,
            args,
            stack_top=temp_stack_top,
            result_count=result_count,
        )
        vm.run(max_steps=saved[5] + max_extra_steps)
        transfer_stop = getattr(
            vm, "_preserved_transfer_stop_depth", None
        )
        legacy_transfer = bool(
            getattr(vm, "_preserved_nonreturning_transfer", False)
        )
        transfer_seen = bool(
            legacy_transfer
            and (
                transfer_stop is None
                or (previous_depth + 1) > int(transfer_stop)
            )
        )
        if transfer_seen:
            return HOST_CONTROL_TRANSFER
        return tuple(
            vm.memory.read(result_base + i * 8, 64)
            for i in range(result_count)
        )
    except VMError as exc:
        current_inst = None
        linked_now = vm.program.functions.get(vm.current_function)
        if (
            linked_now is not None
            and vm.current_block is not None
            and 0 <= vm.ip
        ):
            block_now = next(
                (
                    block
                    for block in linked_now.function.blocks
                    if block.label == vm.current_block
                ),
                None,
            )
            if block_now is not None and vm.ip < len(block_now.instructions):
                current_inst = block_now.instructions[vm.ip]
        print(
            "BOOT_EXEC_PRESERVED_CALL_ERROR "
            f"target={name} function={vm.current_function} "
            f"block={vm.current_block} ip={vm.ip} sp=0x{vm.sp:x} "
            f"inst={current_inst!r} error={exc}",
            flush=True,
        )
        if vm.current_function == "setup_object" and linked_now is not None:
            interesting = (
                "0", "1", "3", "4", "10", "12", "13", "14", "15",
                "17", "18", "21", "22", "23", "25", "26",
            )
            values = {}
            for slot_name in interesting:
                off = linked_now.slot_offsets.get(slot_name)
                if off is not None:
                    values[slot_name] = vm.memory.read(vm.sp + off, 64)
            cache_ptr = values.get("23") or values.get("10") or 0
            ctor_now = (
                vm.memory.read(cache_ptr + 64, 64)
                if cache_ptr else 0
            )
            ctor_block = vm.program.code_block.get(ctor_now)
            ctor_name = ctor_block[0] if ctor_block is not None else None
            ctor_descriptor = (
                vm.program.symbol_addresses.get(ctor_name, 0)
                if ctor_name is not None else 0
            )
            print(
                "BOOT_EXEC_SETUP_OBJECT_CTOR "
                f"value=0x{ctor_now:x} "
                f"code_block={ctor_block!r} "
                f"code_mem_entry=0x{vm.memory.read(ctor_now, 64):x} "
                f"code_mem_frame=0x{vm.memory.read(ctor_now + 8, 64):x} "
                f"descriptor=0x{ctor_descriptor:x} "
                f"descriptor_entry=0x{vm.memory.read(ctor_descriptor, 64):x} "
                f"descriptor_frame=0x{vm.memory.read(ctor_descriptor + 8, 64):x}",
                flush=True,
            )
            print(
                "BOOT_EXEC_SETUP_OBJECT_STATE "
                + " ".join(
                    f"s{slot}=0x{value:x}"
                    for slot, value in sorted(values.items(), key=lambda item: int(item[0]))
                )
                + f" cache=0x{cache_ptr:x} cache_ctor=0x{ctor_now:x} "
                + f"ctor_code={vm.program.code_block.get(ctor_now)!r} "
                + f"ctor_descriptor={next((hex(address) for symbol, address in vm.program.symbol_addresses.items() if symbol in vm.program.functions and vm.program.block_code.get((symbol, vm.program.functions[symbol].function.blocks[0].label)) == ctor_now), None)}",
                flush=True,
            )
        raise
    finally:
        observed_steps = vm.steps
        transfer_stop = getattr(
            vm, "_preserved_transfer_stop_depth", None
        )
        legacy_transfer = bool(
            getattr(vm, "_preserved_nonreturning_transfer", False)
        )
        transfer_seen = transfer_seen or bool(
            legacy_transfer
            and (
                transfer_stop is None
                or (previous_depth + 1) > int(transfer_stop)
            )
        )
        vm._preserved_call_depth = previous_depth
        if call_tasks:
            call_tasks.pop()
        if task_depth == 0:
            task_depths.pop(call_task, None)
        else:
            task_depths[call_task] = task_depth

        if transfer_seen:
            # Keep the task, P3 PC/SP, scheduler contexts, and newly installed
            # userspace image exactly as established by the control transfer.
            # Stop unwinding at the first outer semantic call owned by another
            # Linux task; that caller remains blocked and its vm.run() resumes.
            vm.steps = observed_steps
            boundary = (
                previous_depth == 0
                if transfer_stop is None
                else previous_depth == int(transfer_stop)
            )
            if boundary:
                vm._preserved_nonreturning_transfer = False
                if hasattr(vm, "_preserved_transfer_stop_depth"):
                    delattr(vm, "_preserved_transfer_stop_depth")
                vm.halted = False
                print(
                    "BOOT_EXEC_PRESERVED_CONTROL_RESUME "
                    f"function={vm.current_function} "
                    f"steps={vm.steps} "
                    f"stop_depth={previous_depth}",
                    flush=True,
                )
        else:
            if not preserve_linux_task_state:
                if current_addr is not None and saved_current is not None:
                    observed_current = vm.memory.read(current_addr, 64)
                    if observed_current != saved_current:
                        print(
                            "BOOT_EXEC_PRESERVE_CURRENT_RESTORE "
                            f"function={name} "
                            f"before=0x{saved_current:x} "
                            f"after=0x{observed_current:x}",
                            flush=True,
                        )
                        vm.memory.write(current_addr, 64, saved_current)
                vm.linux_task_contexts = saved_contexts
                vm.linux_task_shadow_stacks = saved_shadow_stacks
                if saved_shadow_next is not None:
                    vm.linux_shadow_stack_next = saved_shadow_next
            (
                vm.sp,
                vm.current_function,
                vm.current_block,
                vm.ip,
                vm.halted,
                _saved_steps,
            ) = saved
            vm.active_user_task = saved_active_user_task
            vm.steps = observed_steps if preserve_linux_task_state else _saved_steps


def skip_prepare_namespace_after_hot_init(vm) -> None:
    if vm.current_function != "prepare_namespace" or vm.ip != 0:
        raise VMError(
            "prepare_namespace hot skip requires an entry checkpoint; "
            f"got function={vm.current_function} block={vm.current_block} ip={vm.ip}"
        )

    command = vm.program.symbol_addresses.get("ramdisk_execute_command")
    if command is None:
        raise VMError("ramdisk_execute_command symbol is missing")
    initial_command = vm.program.initial_memory.read(command, 64)
    if not initial_command:
        raise VMError("ramdisk_execute_command has no initial /init pointer")
    vm.memory.write(command, 64, initial_command)

    result_count = vm.memory.read(vm.sp + RESULT_COUNT, 64)
    if result_count != 0:
        raise VMError(
            f"prepare_namespace caller expects {result_count} results"
        )
    caller_sp = vm.memory.read(vm.sp + CALLER_SP, 64)
    ret_pc = vm.memory.read(vm.sp + RET_PC, 64)
    vm.sp = caller_sp
    vm.halted = False
    vm._set_code(ret_pc)
    print(
        "BOOT_EXEC_PREPARE_NAMESPACE_SKIPPED "
        f"ramdisk_execute_command=0x{initial_command:x} ret_pc=0x{ret_pc:x}",
        flush=True,
    )


def probe_live_rootfs(vm) -> None:
    def guest_cstring(text: str) -> int:
        data = text.encode("utf-8") + b"\0"
        ptr = vm.alloc_bytes(len(data), align=8)
        for i, byte in enumerate(data):
            vm.memory.write(ptr + i, 8, byte)
        return ptr

    current_addr = vm.program.symbol_addresses.get("minimachine_current_task")
    current = vm.memory.read(current_addr, 64) if current_addr is not None else 0
    print(
        "BOOT_EXEC_ROOTFS_PROBE_CURRENT "
        f"symbol=0x{(current_addr or 0):x} task=0x{current:x}",
        flush=True,
    )

    for path in ("/", "/dev", "/dev/console", "/root", "/init"):
        ptr = guest_cstring(path)
        try:
            result, = _call_linux_function_preserving_control(
                vm,
                "init_eaccess",
                (ptr,),
                result_count=1,
            )
        except VMError as exc:
            print(
                f"BOOT_EXEC_ROOTFS_PROBE path={path} error={exc}",
                flush=True,
            )
            continue
        signed = result - (1 << 64) if result & (1 << 63) else result
        print(
            f"BOOT_EXEC_ROOTFS_PROBE path={path} result={signed}",
            flush=True,
        )

    tmp = guest_cstring("/mm-probe")
    try:
        result, = _call_linux_function_preserving_control(
            vm,
            "init_mkdir",
            (tmp, 0o755),
            result_count=1,
        )
        signed = result - (1 << 64) if result & (1 << 63) else result
        print(
            f"BOOT_EXEC_ROOTFS_PROBE mkdir=/mm-probe result={signed}",
            flush=True,
        )
    except VMError as exc:
        print(
            f"BOOT_EXEC_ROOTFS_PROBE mkdir=/mm-probe error={exc}",
            flush=True,
        )


def install_hot_filp_trace(vm) -> None:
    targets = {
        "filp_open",
        "file_open_name",
        "do_filp_open",
        "path_openat",
        "path_init",
        "link_path_walk",
        "open_last_lookups",
        "lookup_open",
        "lookup_fast",
        "d_lookup",
        "d_alloc_parallel",
        "mnt_want_write",
        "may_o_create",
        "do_open",
        "vfs_open",
    }
    traced_codes: dict[int, tuple[str, str]] = {}
    for code, pair in vm.program.code_block.items():
        if pair[0] in targets:
            traced_codes[code] = pair

    original_set_code = vm._set_code
    original_enter_function = vm.enter_function
    trace_count = 0
    trace_limit = 1200

    def emit(function: str, block: str) -> None:
        nonlocal trace_count
        if trace_count >= trace_limit:
            return
        trace_count += 1
        linked = vm.program.functions.get(function)
        frame_size = vm.memory.read(vm.sp + 24, 64) if linked is not None else 0
        argc = vm.memory.read(vm.sp + 56, 64) if linked is not None else 0
        args = []
        if linked is not None and argc <= 8:
            arg_base = vm.sp + frame_size
            args = [vm.memory.read(arg_base + i * 8, 64) for i in range(argc)]
        if function == "open_last_lookups" and block == "entry" and len(args) >= 3:
            op_ptr = args[2]
            fields = [vm.memory.read(op_ptr + i * 4, 32) for i in range(5)]
            print(
                "BOOT_EXEC_HOT_OPEN_FLAGS "
                f"ptr=0x{op_ptr:x} "
                f"f0=0x{fields[0]:x} f1=0x{fields[1]:x} "
                f"f2=0x{fields[2]:x} f3=0x{fields[3]:x} f4=0x{fields[4]:x}",
                flush=True,
            )
        print(
            "BOOT_EXEC_HOT_FILP_TRACE "
            f"seq={trace_count} steps={vm.steps} "
            f"function={function} block={block} sp=0x{vm.sp:x} "
            f"argc={argc} args={','.join(f'0x{x:x}' for x in args)}",
            flush=True,
        )

    def traced_set_code(code: int) -> None:
        original_set_code(code)
        pair = traced_codes.get(code)
        if pair is not None:
            emit(pair[0], pair[1])

    def traced_enter_function(name, args=(), *, stack_top, result_count=0):
        original_enter_function(
            name,
            args,
            stack_top=stack_top,
            result_count=result_count,
        )
        if name in targets:
            linked = vm.program.functions[name]
            emit(name, linked.function.blocks[0].label)

    vm._set_code = traced_set_code
    vm.enter_function = traced_enter_function
    and32 = vm.program.host_services.get("__mm_and_32")
    if and32 is not None:
        def traced_and32(inner_vm, args):
            result = and32(inner_vm, args)
            if len(args) == 2 and (args[1] & 0xFFFFFFFF) == 579:
                print(
                    "BOOT_EXEC_HOT_AND32 "
                    f"steps={inner_vm.steps} "
                    f"a=0x{args[0] & 0xffffffff:x} "
                    f"b=0x{args[1] & 0xffffffff:x} "
                    f"result=0x{int(result) & 0xffffffff:x}",
                    flush=True,
                )
            return result
        vm.program.host_services["__mm_and_32"] = traced_and32

    print(
        "BOOT_EXEC_HOT_FILP_TRACE_ARMED "
        f"functions={len(targets)} codes={len(traced_codes)} limit={trace_limit}",
        flush=True,
    )


def inject_live_root_init(
    vm,
    path: Path,
    *,
    guest_path_text: str = "/init",
) -> None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VMError(
            f"cannot read injected rootfs image for {guest_path_text}: {exc}"
        ) from exc
    if not data:
        raise VMError(f"injected rootfs image for {guest_path_text} is empty")
    if not guest_path_text.startswith("/"):
        raise VMError("--inject-init-path must be absolute")

    guest_path = guest_path_text.encode("utf-8") + b"\0"
    path_ptr = vm.alloc_bytes(len(guest_path), align=8)
    for i, byte in enumerate(guest_path):
        vm.memory.write(path_ptr + i, 8, byte)

    data_ptr = vm.alloc_bytes(len(data), align=8)
    bulk_write = getattr(vm.memory, "bulk_write", None)
    if bulk_write is not None:
        started = time.perf_counter()
        bulk_write(data_ptr, data)
        print(
            "BOOT_EXEC_HOST_BULK_WRITE "
            f"bytes={len(data)} seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
    else:
        for i, byte in enumerate(data):
            vm.memory.write(data_ptr + i, 8, byte)

    pos_ptr = vm.alloc_bytes(8, align=8)
    vm.memory.write(pos_ptr, 64, 0)

    # Linux asm-generic flags: O_WRONLY | O_CREAT | O_TRUNC.
    open_flags = 0x1 | 0x40 | 0x200
    file_ptr, = _call_linux_function_preserving_control(
        vm,
        "filp_open",
        (path_ptr, open_flags, 0o755),
        result_count=1,
        max_extra_steps=max(
            16_000_000,
            len(data) * 2,
        ),
    )
    print(
        "BOOT_EXEC_ROOTFS_OPEN "
        f"path={guest_path_text} file=0x{file_ptr:x} bytes={len(data)}",
        flush=True,
    )
    if file_ptr >= (1 << 64) - 4095:
        signed = file_ptr - (1 << 64)
        raise VMError(f"filp_open({guest_path_text}) failed: {signed}")

    try:
        written, = _call_linux_function_preserving_control(
            vm,
            "kernel_write",
            (file_ptr, data_ptr, len(data), pos_ptr),
            result_count=1,
            max_extra_steps=max(
                8_000_000,
                len(data) * 32,
            ),
        )
        if written != len(data):
            signed = written - (1 << 64) if written & (1 << 63) else written
            raise VMError(
                f"kernel_write({guest_path_text}) wrote {signed}, expected {len(data)}"
            )
    finally:
        if "__fput_sync" in vm.program.functions:
            _call_linux_function_preserving_control(
                vm,
                "__fput_sync",
                (file_ptr,),
                result_count=0,
            )
            print(
                "BOOT_EXEC_ROOTFS_CLOSE method=__fput_sync",
                flush=True,
            )
        else:
            _call_linux_function_preserving_control(
                vm,
                "fput",
                (file_ptr,),
                result_count=0,
            )
            if "flush_delayed_fput" in vm.program.functions:
                _call_linux_function_preserving_control(
                    vm,
                    "flush_delayed_fput",
                    (),
                    result_count=0,
                )
                print(
                    "BOOT_EXEC_ROOTFS_CLOSE "
                    "method=fput+flush_delayed_fput",
                    flush=True,
                )
            else:
                print(
                    "BOOT_EXEC_ROOTFS_CLOSE method=fput-only",
                    flush=True,
                )

    access, = _call_linux_function_preserving_control(
        vm,
        "init_eaccess",
        (path_ptr,),
        result_count=1,
    )
    if access != 0:
        signed = access - (1 << 64) if access & (1 << 63) else access
        raise VMError(
            f"init_eaccess({guest_path_text}) after injection failed: {signed}"
        )

    print(
        "BOOT_EXEC_ROOTFS_INJECTED "
        f"path={guest_path_text} bytes={len(data)} mode=0755",
        flush=True,
    )


def inject_live_root_initramfs(vm, path: Path) -> None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VMError(f"cannot read injected initramfs: {exc}") from exc
    if not data:
        raise VMError("injected initramfs is empty")
    if "unpack_to_rootfs" not in vm.program.functions:
        raise VMError("Linux unpack_to_rootfs is missing from P3 program")

    data_ptr = vm.alloc_bytes(len(data), align=8)
    bulk_write = getattr(vm.memory, "bulk_write", None)
    if bulk_write is not None:
        started = time.perf_counter()
        bulk_write(data_ptr, data)
        print(
            "BOOT_EXEC_HOST_BULK_WRITE "
            f"bytes={len(data)} seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
    else:
        for i, byte in enumerate(data):
            vm.memory.write(data_ptr + i, 8, byte)

    error_ptr, = _call_linux_function_preserving_control(
        vm,
        "unpack_to_rootfs",
        (data_ptr, len(data)),
        result_count=1,
        max_extra_steps=25_000_000,
    )
    if error_ptr:
        raw = bytearray()
        for i in range(256):
            byte = vm.memory.read(error_ptr + i, 8)
            if byte == 0:
                break
            raw.append(byte)
        message = raw.decode("utf-8", errors="replace")
        raise VMError(
            f"unpack_to_rootfs(initramfs) failed: ptr=0x{error_ptr:x} "
            f"message={message!r}"
        )

    guest_path = b"/init\0"
    path_ptr = vm.alloc_bytes(len(guest_path), align=8)
    for i, byte in enumerate(guest_path):
        vm.memory.write(path_ptr + i, 8, byte)
    access, = _call_linux_function_preserving_control(
        vm,
        "init_eaccess",
        (path_ptr,),
        result_count=1,
    )
    if access != 0:
        signed = access - (1 << 64) if access & (1 << 63) else access
        raise VMError(f"init_eaccess(/init) after initramfs unpack failed: {signed}")

    print(
        "BOOT_EXEC_ROOTFS_UNPACKED "
        f"path=/init archive_bytes={len(data)} via=unpack_to_rootfs",
        flush=True,
    )
