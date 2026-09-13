from __future__ import annotations

from src.minimachine.ctype_surface import resolve_ctype_callback


STDIO_LINE_EXTERNALS = frozenset({"fputs", "fgets"})
_U64_MASK = (1 << 64) - 1


def _signed_u64(value: int) -> int:
    value = int(value) & _U64_MASK
    return value - (1 << 64) if value & (1 << 63) else value


def _set_errno(vm, errno_address: int | None, value: int) -> None:
    if errno_address is not None:
        vm.memory.write(errno_address, 32, int(value) & 0xFFFFFFFF)


def _stream_state(vm, stream: int):
    streams = getattr(vm, "user_file_streams", None)
    if streams is None:
        streams = {}
        vm.user_file_streams = streams
    if stream in {0, 1, 2}:
        return streams.setdefault(
            stream,
            {"fd": stream, "eof": False, "error": False, "ungetc": None},
        )
    return streams.get(stream)


def _read_guest_bytes(vm, address: int, size: int) -> bytes:
    bulk_read = getattr(vm.memory, "bulk_read", None)
    if bulk_read is not None:
        return bytes(bulk_read(address, size))
    return bytes(vm.memory.read(address + offset, 8) for offset in range(size))


def _write_guest_bytes(vm, address: int, payload: bytes) -> None:
    bulk_write = getattr(vm.memory, "bulk_write", None)
    if bulk_write is not None:
        bulk_write(address, payload)
        return
    for offset, byte in enumerate(payload):
        vm.memory.write(address + offset, 8, byte)


def _strlen(vm, address: int) -> int:
    bulk_strlen = getattr(vm.memory, "bulk_strlen", None)
    if bulk_strlen is not None:
        return int(bulk_strlen(address))
    length = 0
    while vm.memory.read(address + length, 8) != 0:
        length += 1
    return length


def _read_buffers(vm) -> dict[int, bytearray]:
    buffers = getattr(vm, "user_stdio_read_buffers", None)
    if buffers is None:
        buffers = {}
        vm.user_stdio_read_buffers = buffers
    return buffers


def resolve_stdio_line_callback(
    original: str,
    *,
    errno_address: int | None,
    vm_error,
    user_syscall,
):
    """Resolve line-oriented stdio and its immediate libc table dependencies.

    The stdio surface deliberately uses the guest Linux read/write syscalls and
    guest FILE* state.  Ctype table accessors are delegated to a deterministic
    guest C-locale surface because the frozen SQLite input loop reaches them
    immediately after reading each line.
    """
    ctype_callback = resolve_ctype_callback(original, vm_error=vm_error)
    if ctype_callback is not None:
        return ctype_callback

    if original not in STDIO_LINE_EXTERNALS:
        return None

    if original == "fputs":
        def user_fputs(vm, args):
            if len(args) != 2:
                raise vm_error("fputs expects string,FILE*")
            ptr, stream = map(int, args)
            state = _stream_state(vm, stream)
            if state is None:
                _set_errno(vm, errno_address, 9)  # EBADF
                return _U64_MASK

            length = _strlen(vm, ptr)
            written = 0
            while written < length:
                raw = user_syscall(
                    vm,
                    (64, int(state["fd"]), ptr + written, length - written, 0, 0, 0),
                )
                if raw is None:
                    raise vm_error("fputs write performed unexpected control transfer")
                signed = _signed_u64(raw)
                if signed < 0:
                    state["error"] = True
                    _set_errno(vm, errno_address, -signed)
                    return _U64_MASK
                if signed == 0:
                    state["error"] = True
                    _set_errno(vm, errno_address, 5)  # EIO
                    return _U64_MASK
                written += signed
            return 0

        return user_fputs

    if original == "fgets":
        def user_fgets(vm, args):
            if len(args) != 3:
                raise vm_error("fgets expects buffer,size,FILE*")
            dst, raw_size, stream = map(int, args)
            size32 = raw_size & 0xFFFFFFFF
            size = size32 - (1 << 32) if size32 & (1 << 31) else size32
            if dst == 0 or size <= 0:
                return 0

            state = _stream_state(vm, stream)
            if state is None:
                _set_errno(vm, errno_address, 9)  # EBADF
                return 0
            if size == 1:
                vm.memory.write(dst, 8, 0)
                state["eof"] = False
                return dst

            pending = _read_buffers(vm).setdefault(stream, bytearray())
            pushed = state.get("ungetc")
            if pushed is not None:
                pending[:0] = bytes((int(pushed) & 0xFF,))
                state["ungetc"] = None

            output = bytearray()
            saw_eof = False
            saw_error = False
            while len(output) < size - 1:
                if pending:
                    remaining = size - 1 - len(output)
                    newline = pending.find(b"\n", 0, remaining)
                    take = newline + 1 if newline >= 0 else min(len(pending), remaining)
                    output.extend(pending[:take])
                    del pending[:take]
                    if output.endswith(b"\n"):
                        break
                    continue

                remaining = size - 1 - len(output)
                request = min(4096, remaining)
                scratch = vm.alloc_bytes(request, align=1)
                raw = user_syscall(
                    vm,
                    (63, int(state["fd"]), scratch, request, 0, 0, 0),
                )
                if raw is None:
                    raise vm_error("fgets read performed unexpected control transfer")
                signed = _signed_u64(raw)
                if signed < 0:
                    state["error"] = True
                    _set_errno(vm, errno_address, -signed)
                    saw_error = True
                    break
                if signed == 0:
                    state["eof"] = True
                    saw_eof = True
                    break
                pending.extend(_read_guest_bytes(vm, scratch, signed))

            if not output and (saw_eof or saw_error):
                return 0

            _write_guest_bytes(vm, dst, bytes(output))
            vm.memory.write(dst + len(output), 8, 0)
            if output:
                state["eof"] = False
            return dst

        return user_fgets

    raise AssertionError(f"unhandled stdio line external: {original}")
