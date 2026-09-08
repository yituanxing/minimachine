from __future__ import annotations

from functools import cmp_to_key
import struct
import time

from ..abi import (
    ARG_COUNT,
    CALLER_SP,
    FRAME_SIZE,
    HEADER_SIZE,
    RESULT_COUNT,
    RESULT_PTR,
    RET_PC,
    WORD,
)
from ..runtime import direct_runtime_callback
from ..vm import HOST_CONTROL_TRANSFER, VMError


def _user_external_prefix(symbol: str) -> str | None:
    if not symbol.startswith("__mm_"):
        return None
    marker = "_ext_"
    marker_at = symbol.find(marker, len("__mm_"))
    if marker_at < 0:
        return None
    return symbol[: marker_at + len(marker)]


def _user_external_original(symbol: str) -> str:
    prefix = _user_external_prefix(symbol)
    return symbol[len(prefix):] if prefix is not None else symbol


def build_user_libc_callback(
    symbol: str,
    errno_address: int | None,
    *,
    user_syscall,
    _call_linux_function_preserving_control,
    _call_guest_descriptor_preserving_control,
    _guest_function_name_from_descriptor,
    _snapshot_p3_call_chain,
    _restore_p3_call_chain,
):
    def _user_libc_callback(nested_symbol: str, nested_errno_address: int | None):
        return build_user_libc_callback(
            nested_symbol,
            nested_errno_address,
            user_syscall=user_syscall,
            _call_linux_function_preserving_control=_call_linux_function_preserving_control,
            _call_guest_descriptor_preserving_control=_call_guest_descriptor_preserving_control,
            _guest_function_name_from_descriptor=_guest_function_name_from_descriptor,
            _snapshot_p3_call_chain=_snapshot_p3_call_chain,
            _restore_p3_call_chain=_restore_p3_call_chain,
        )

    original = _user_external_original(symbol)
    external_prefix = _user_external_prefix(symbol) or "__mm_user_ext_"

    direct = direct_runtime_callback(original)
    if direct is not None:
        return direct

    if original == "bcmp":
        return direct_runtime_callback("memcmp")

    if original == "__errno_location" and errno_address is not None:
        def errno_location(_vm, args):
            if args:
                raise VMError("__errno_location expects no arguments")
            return errno_address
        return errno_location

    if original == "_setjmp":
        def user_setjmp(vm, args):
            if len(args) != 1:
                raise VMError("_setjmp expects jmp_buf")
            env = int(args[0])
            expected = vm.memory.read(vm.sp + RESULT_COUNT, 64)
            if expected != 1:
                raise VMError(
                    f"_setjmp caller expects {expected} results, expected 1"
                )
            state = (
                vm.memory.read(vm.sp + CALLER_SP, 64),
                vm.memory.read(vm.sp + RET_PC, 64),
                vm.memory.read(vm.sp + RESULT_PTR, 64),
                int(vm.heap_next),
            )
            table = getattr(vm, "user_setjmp_states", None)
            if table is None:
                table = {}
                vm.user_setjmp_states = table
            table[env] = state
            print(
                "BOOT_EXEC_USER_SETJMP "
                f"env=0x{env:x} resume_sp=0x{state[0]:x} "
                f"resume_pc=0x{state[1]:x} result_ptr=0x{state[2]:x}",
                flush=True,
            )
            return 0
        return user_setjmp

    if original == "longjmp":
        def user_longjmp(vm, args):
            if len(args) != 2:
                raise VMError("longjmp expects jmp_buf,value")
            env, value = map(int, args)
            table = getattr(vm, "user_setjmp_states", None)
            state = table.get(env) if table is not None else None
            if state is None:
                raise VMError(
                    f"longjmp received unknown jmp_buf 0x{env:x}"
                )
            resume_sp, resume_pc, result_ptr, heap_next = state
            result = value if value != 0 else 1
            vm.memory.write(result_ptr, 64, result)
            vm.heap_next = heap_next
            vm.sp = resume_sp
            vm.halted = False
            vm._set_code(resume_pc)
            print(
                "BOOT_EXEC_USER_LONGJMP "
                f"env=0x{env:x} value={result} "
                f"resume_sp=0x{resume_sp:x} resume_pc=0x{resume_pc:x}",
                flush=True,
            )
            return HOST_CONTROL_TRANSFER
        return user_longjmp

    def set_errno(vm, value: int) -> None:
        if errno_address is not None:
            vm.memory.write(errno_address, 32, value & 0xFFFFFFFF)

    def errno_message(vm) -> bytes:
        err = (
            vm.memory.read(errno_address, 32)
            if errno_address is not None
            else 0
        )
        messages = {
            0: b"Success",
            1: b"Operation not permitted",
            2: b"No such file or directory",
            5: b"Input/output error",
            9: b"Bad file descriptor",
            12: b"Cannot allocate memory",
            13: b"Permission denied",
            17: b"File exists",
            20: b"Not a directory",
            21: b"Is a directory",
            22: b"Invalid argument",
            28: b"No space left on device",
            34: b"Numerical result out of range",
            38: b"Function not implemented",
            95: b"Operation not supported",
        }
        return messages.get(err, f"Unknown error {err}".encode("ascii"))

    def libc_linux_result(vm, raw: int) -> int:
        signed = raw - (1 << 64) if raw & (1 << 63) else raw
        if -4095 <= signed < 0:
            set_errno(vm, -signed)
            return (1 << 64) - 1
        return raw

    def stdio_streams(vm):
        streams = getattr(vm, "user_file_streams", None)
        if streams is None:
            streams = {}
            vm.user_file_streams = streams
        return streams

    def stdio_state(vm, stream: int):
        return stdio_streams(vm).get(stream)

    def stdio_fd(vm, stream: int) -> int:
        # External stdin/stdout/stderr use their Linux fd values directly.
        # Other FILE* values are opaque guest handles backed by Linux fds.
        if stream in {0, 1, 2}:
            return stream
        state = stdio_state(vm, stream)
        return int(state["fd"]) if state is not None else -1

    def read_user_cstring(vm, ptr: int, limit: int = 1 << 20) -> bytes:
        if not ptr:
            return b""
        bulk_strlen = getattr(vm.memory, "bulk_strlen", None)
        bulk_read = getattr(vm.memory, "bulk_read", None)
        if bulk_strlen is not None and bulk_read is not None:
            length = min(int(bulk_strlen(ptr)), limit)
            return bytes(bulk_read(ptr, length))
        data = bytearray()
        for index in range(limit):
            byte = vm.memory.read(ptr + index, 8)
            if byte == 0:
                break
            data.append(byte)
        return bytes(data)

    def render_user_printf(vm, fmt_ptr: int, next_arg) -> bytes:
        fmt = read_user_cstring(vm, fmt_ptr, 4096).decode("latin1")
        out = bytearray()
        index = 0

        def signed_value(value: int, bits: int) -> int:
            mask = (1 << bits) - 1
            value &= mask
            sign = 1 << (bits - 1)
            return value - (1 << bits) if value & sign else value

        while index < len(fmt):
            if fmt[index] != "%":
                out.append(ord(fmt[index]))
                index += 1
                continue
            index += 1
            if index < len(fmt) and fmt[index] == "%":
                out.append(ord("%"))
                index += 1
                continue

            flags = ""
            while index < len(fmt) and fmt[index] in "-+ #0":
                flags += fmt[index]
                index += 1

            if index < len(fmt) and fmt[index] == "*":
                width = signed_value(next_arg(), 32)
                if width < 0:
                    flags += "-"
                    width = -width
                index += 1
            else:
                start = index
                while index < len(fmt) and fmt[index].isdigit():
                    index += 1
                width = int(fmt[start:index] or "0")

            precision = None
            if index < len(fmt) and fmt[index] == ".":
                index += 1
                if index < len(fmt) and fmt[index] == "*":
                    raw_precision = signed_value(next_arg(), 32)
                    precision = None if raw_precision < 0 else raw_precision
                    index += 1
                else:
                    start = index
                    while index < len(fmt) and fmt[index].isdigit():
                        index += 1
                    precision = int(fmt[start:index] or "0")

            length = ""
            if index < len(fmt) and fmt[index] in "hljzt":
                length = fmt[index]
                index += 1
                if (
                    length in {"h", "l"}
                    and index < len(fmt)
                    and fmt[index] == length
                ):
                    length += fmt[index]
                    index += 1
            if index >= len(fmt):
                raise VMError("printf format ends after %")

            spec = fmt[index]
            index += 1
            bits = 64 if length in {"l", "ll", "j", "z", "t"} else 32
            numeric = False

            if spec in "di":
                value = signed_value(next_arg(), bits)
                sign = "-" if value < 0 else (
                    "+" if "+" in flags else (" " if " " in flags else "")
                )
                digits = str(abs(value))
                if precision is not None:
                    digits = digits.rjust(precision, "0")
                piece = sign + digits
                numeric = True
            elif spec in "uoxX":
                value = next_arg() & ((1 << bits) - 1)
                prefix = ""
                if spec == "u":
                    digits = str(value)
                elif spec == "o":
                    digits = format(value, "o")
                    if "#" in flags and not digits.startswith("0"):
                        prefix = "0"
                else:
                    digits = format(value, "x" if spec == "x" else "X")
                    if "#" in flags and value:
                        prefix = "0x" if spec == "x" else "0X"
                if precision is not None:
                    digits = digits.rjust(precision, "0")
                piece = prefix + digits
                numeric = True
            elif spec in "eEfFgG":
                raw_bits = next_arg() & ((1 << 64) - 1)
                value = struct.unpack(
                    "<d", raw_bits.to_bytes(8, "little")
                )[0]
                format_spec = "%" + flags
                if width:
                    format_spec += str(width)
                if precision is not None:
                    format_spec += "." + str(precision)
                format_spec += spec
                piece = format_spec % value
                numeric = True
            elif spec == "p":
                value = next_arg()
                piece = "(nil)" if value == 0 else f"0x{value:x}"
            elif spec == "c":
                piece = chr(next_arg() & 0xFF)
            elif spec == "s":
                raw = read_user_cstring(vm, next_arg())
                if precision is not None:
                    raw = raw[:precision]
                if width > len(raw):
                    padding = b" " * (width - len(raw))
                    raw = raw + padding if "-" in flags else padding + raw
                out.extend(raw)
                continue
            elif spec == "m":
                raw = errno_message(vm)
                if precision is not None:
                    raw = raw[:precision]
                if width > len(raw):
                    padding = b" " * (width - len(raw))
                    raw = raw + padding if "-" in flags else padding + raw
                out.extend(raw)
                continue
            else:
                raise VMError(
                    f"unsupported printf conversion %{length}{spec}"
                )

            raw = piece.encode("latin1")
            if width > len(raw):
                pad_len = width - len(raw)
                if "-" in flags:
                    raw += b" " * pad_len
                elif "0" in flags and precision is None and numeric:
                    lead = 1 if raw[:1] in {b"+", b"-", b" "} else (
                        2 if raw[:2] in {b"0x", b"0X"} else 0
                    )
                    raw = raw[:lead] + b"0" * pad_len + raw[lead:]
                else:
                    raw = b" " * pad_len + raw
            out.extend(raw)
        return bytes(out)

    def write_stdio_payload(vm, stream: int, payload: bytes) -> int:
        if not payload:
            return 0
        ptr = vm.alloc_bytes(len(payload), align=1)
        bulk_write = getattr(vm.memory, "bulk_write", None)
        if bulk_write is not None:
            bulk_write(ptr, payload)
        else:
            for offset, byte in enumerate(payload):
                vm.memory.write(ptr + offset, 8, byte)
        raw = user_syscall(
            vm,
            (64, stdio_fd(vm, stream), ptr, len(payload), 0, 0, 0),
        )
        signed = raw - (1 << 64) if raw & (1 << 63) else raw
        if signed < 0:
            set_errno(vm, -signed)
            return (1 << 64) - 1
        return signed

    if original in {"fopen", "fopen64"}:
        def user_fopen(vm, args):
            if len(args) != 2:
                raise VMError(f"{original} expects path,mode")
            path_ptr, mode_ptr = map(int, args)
            mode = read_user_cstring(vm, mode_ptr, 32).decode(
                "ascii", errors="ignore"
            )
            if not mode or mode[0] not in "rwa":
                set_errno(vm, 22)
                return 0

            if mode[0] == "r":
                flags = 0
            elif mode[0] == "w":
                flags = 0x1 | 0x40 | 0x200  # O_WRONLY|O_CREAT|O_TRUNC
            else:
                flags = 0x1 | 0x40 | 0x400  # O_WRONLY|O_CREAT|O_APPEND
            if "+" in mode:
                flags = (flags & ~0x3) | 0x2  # O_RDWR
            if "x" in mode:
                flags |= 0x80  # O_EXCL
            if "e" in mode:
                flags |= 0x80000  # O_CLOEXEC

            raw = user_syscall(
                vm,
                (
                    56,  # openat
                    (-100) & ((1 << 64) - 1),  # AT_FDCWD
                    path_ptr,
                    flags,
                    0o666,
                    0,
                    0,
                ),
            )
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                set_errno(vm, -signed)
                return 0

            handle = vm.alloc_bytes(32, align=8)
            stdio_streams(vm)[handle] = {
                "fd": int(signed),
                "eof": False,
                "error": False,
                "ungetc": None,
            }
            print(
                "BOOT_EXEC_USER_FOPEN "
                f"kind={original} path_ptr=0x{path_ptr:x} mode={mode!r} "
                f"fd={signed} handle=0x{handle:x}",
                flush=True,
            )
            return handle

        return user_fopen

    if original == "getline":
        def user_getline(vm, args):
            if len(args) != 3:
                raise VMError("getline expects lineptr,n,FILE*")
            lineptr_ptr, capacity_ptr, stream = map(int, args)
            if lineptr_ptr == 0 or capacity_ptr == 0:
                set_errno(vm, 22)
                return (1 << 64) - 1
            state = stdio_state(vm, stream)
            if state is None:
                set_errno(vm, 9)
                return (1 << 64) - 1

            line_ptr = vm.memory.read(lineptr_ptr, 64)
            capacity = vm.memory.read(capacity_ptr, 64)
            allocations = getattr(vm, "user_allocations", None)
            if allocations is None:
                allocations = {}
                vm.user_allocations = allocations

            def ensure_capacity(required: int) -> None:
                nonlocal line_ptr, capacity
                if line_ptr != 0 and capacity >= required:
                    return
                new_capacity = max(128, int(capacity) if capacity else 0)
                while new_capacity < required:
                    new_capacity *= 2
                new_ptr = vm.alloc_bytes(new_capacity, align=16)
                if line_ptr:
                    copy_size = min(int(capacity), required - 1)
                    bulk_copy = getattr(vm.memory, "bulk_copy", None)
                    if bulk_copy is not None and copy_size:
                        bulk_copy(new_ptr, line_ptr, copy_size)
                    else:
                        for index in range(copy_size):
                            vm.memory.write(
                                new_ptr + index,
                                8,
                                vm.memory.read(line_ptr + index, 8),
                            )
                    allocations.pop(line_ptr, None)
                allocations[new_ptr] = new_capacity
                line_ptr = new_ptr
                capacity = new_capacity
                vm.memory.write(lineptr_ptr, 64, line_ptr)
                vm.memory.write(capacity_ptr, 64, capacity)

            length = 0
            while True:
                pushed = state.get("ungetc")
                if pushed is not None:
                    state["ungetc"] = None
                    state["eof"] = False
                    byte = int(pushed) & 0xFF
                else:
                    byte_ptr = vm.alloc_bytes(1, align=1)
                    raw = user_syscall(
                        vm,
                        (
                            63,  # read
                            int(state["fd"]),
                            byte_ptr,
                            1,
                            0,
                            0,
                            0,
                        ),
                    )
                    signed = raw - (1 << 64) if raw & (1 << 63) else raw
                    if signed < 0:
                        state["error"] = True
                        set_errno(vm, -signed)
                        if length == 0:
                            return (1 << 64) - 1
                        break
                    if signed == 0:
                        state["eof"] = True
                        if length == 0:
                            return (1 << 64) - 1
                        break
                    byte = vm.memory.read(byte_ptr, 8)

                ensure_capacity(length + 2)
                vm.memory.write(line_ptr + length, 8, byte)
                length += 1
                if byte == 10:
                    break

            ensure_capacity(length + 1)
            vm.memory.write(line_ptr + length, 8, 0)
            return length

        return user_getline

    if original == "fclose":
        def user_fclose(vm, args):
            if len(args) != 1:
                raise VMError("fclose expects FILE*")
            stream = int(args[0])
            state = stdio_streams(vm).pop(stream, None)
            if state is None:
                set_errno(vm, 9)
                return (1 << 64) - 1
            raw = user_syscall(
                vm,
                (57, int(state["fd"]), 0, 0, 0, 0, 0),
            )
            return libc_linux_result(vm, raw)

        return user_fclose

    if original == "fread":
        def user_fread(vm, args):
            if len(args) != 4:
                raise VMError("fread expects ptr,size,nmemb,FILE*")
            ptr, size, nmemb, stream = map(int, args)
            if size == 0 or nmemb == 0:
                return 0
            state = stdio_state(vm, stream)
            if state is None:
                set_errno(vm, 9)
                return 0

            total = size * nmemb
            copied = 0
            pushed = state.get("ungetc")
            if pushed is not None and total:
                vm.memory.write(ptr, 8, int(pushed) & 0xFF)
                state["ungetc"] = None
                copied = 1

            if copied < total:
                raw = user_syscall(
                    vm,
                    (
                        63,  # read
                        int(state["fd"]),
                        ptr + copied,
                        total - copied,
                        0,
                        0,
                        0,
                    ),
                )
                signed = raw - (1 << 64) if raw & (1 << 63) else raw
                if signed < 0:
                    state["error"] = True
                    set_errno(vm, -signed)
                    return copied // size
                copied += int(signed)
                if int(signed) < total - (1 if pushed is not None else 0):
                    state["eof"] = True

            return copied // size

        return user_fread

    if original in {"getc", "fgetc", "getc_unlocked"}:
        def user_getc(vm, args):
            if len(args) != 1:
                raise VMError(f"{original} expects FILE*")
            stream = int(args[0])
            state = stdio_state(vm, stream)
            if state is None:
                set_errno(vm, 9)
                return (1 << 64) - 1

            pushed = state.get("ungetc")
            if pushed is not None:
                state["ungetc"] = None
                state["eof"] = False
                return int(pushed) & 0xFF

            byte_ptr = vm.alloc_bytes(1, align=1)
            raw = user_syscall(
                vm,
                (63, int(state["fd"]), byte_ptr, 1, 0, 0, 0),
            )
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                state["error"] = True
                set_errno(vm, -signed)
                return (1 << 64) - 1
            if signed == 0:
                state["eof"] = True
                return (1 << 64) - 1
            return vm.memory.read(byte_ptr, 8)

        return user_getc

    if original == "ungetc":
        def user_ungetc(vm, args):
            if len(args) != 2:
                raise VMError("ungetc expects char,FILE*")
            ch, stream = map(int, args)
            state = stdio_state(vm, stream)
            if state is None or ch == ((1 << 64) - 1):
                return (1 << 64) - 1
            if state.get("ungetc") is not None:
                return (1 << 64) - 1
            state["ungetc"] = ch & 0xFF
            state["eof"] = False
            return ch & 0xFF

        return user_ungetc

    if original == "feof":
        def user_feof(vm, args):
            if len(args) != 1:
                raise VMError("feof expects FILE*")
            state = stdio_state(vm, int(args[0]))
            return 1 if state is not None and state.get("eof") else 0

        return user_feof

    if original in {"fseeko", "fseeko64"}:
        def user_fseeko(vm, args):
            if len(args) != 3:
                raise VMError(f"{original} expects FILE*,offset,whence")
            stream, offset, whence = map(int, args)
            state = stdio_state(vm, stream)
            if state is None:
                set_errno(vm, 9)
                return (1 << 64) - 1
            raw = user_syscall(
                vm,
                (62, int(state["fd"]), offset, whence, 0, 0, 0),
            )
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                state["error"] = True
                set_errno(vm, -signed)
                return (1 << 64) - 1
            state["eof"] = False
            state["ungetc"] = None
            return 0

        return user_fseeko

    if original in {"ftello", "ftello64"}:
        def user_ftello(vm, args):
            if len(args) != 1:
                raise VMError(f"{original} expects FILE*")
            stream = int(args[0])
            state = stdio_state(vm, stream)
            if state is None:
                set_errno(vm, 9)
                return (1 << 64) - 1
            raw = user_syscall(
                vm,
                (62, int(state["fd"]), 0, 1, 0, 0, 0),
            )
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                state["error"] = True
                set_errno(vm, -signed)
                return (1 << 64) - 1
            if state.get("ungetc") is not None:
                signed -= 1
            return signed & ((1 << 64) - 1)

        return user_ftello

    if original == "setvbuf":
        def user_setvbuf(vm, args):
            if len(args) != 4:
                raise VMError("setvbuf expects FILE*,buf,mode,size")
            stream = int(args[0])
            if stream not in {0, 1, 2} and stdio_state(vm, stream) is None:
                set_errno(vm, 9)
                return (1 << 64) - 1
            # MiniMachine stdio is intentionally unbuffered; accepting the
            # buffering request preserves observable file contents while
            # avoiding a second host-side buffering layer.
            return 0

        return user_setvbuf

    if original == "openlog":
        def user_openlog(vm, args):
            if len(args) != 3:
                raise VMError("openlog expects ident,option,facility")
            ident_ptr, option, facility = map(int, args)
            ident = (
                read_user_cstring(vm, ident_ptr, 256)
                if ident_ptr
                else b""
            )
            vm.user_syslog_state = {
                "ident": ident,
                "option": option,
                "facility": facility,
            }
            print(
                "BOOT_EXEC_USER_OPENLOG "
                f"ident={ident.decode('utf-8', errors='replace')!r} "
                f"option=0x{option:x} facility=0x{facility:x}",
                flush=True,
            )
            return None

        return user_openlog

    if original == "closelog":
        def user_closelog(vm, args):
            if args:
                raise VMError("closelog expects no arguments")
            vm.user_syslog_state = None
            return None

        return user_closelog

    if original in {"syslog", "vsyslog"}:
        def user_syslog(vm, args):
            if original == "syslog":
                if len(args) < 2:
                    raise VMError("syslog expects priority,format,...")
                priority, fmt_ptr = map(int, args[:2])
                values = iter(int(value) for value in args[2:])

                def next_arg() -> int:
                    try:
                        return next(values)
                    except StopIteration as exc:
                        raise VMError(
                            "syslog format consumes more arguments than supplied"
                        ) from exc
            else:
                if len(args) != 3:
                    raise VMError("vsyslog expects priority,format,va_list")
                priority, fmt_ptr, cursor = map(int, args)

                def next_arg() -> int:
                    nonlocal cursor
                    value = vm.memory.read(cursor, 64)
                    cursor += 8
                    return int(value)

            payload = render_user_printf(vm, fmt_ptr, next_arg)
            state = getattr(vm, "user_syslog_state", None) or {
                "ident": b"",
                "option": 0,
                "facility": 0,
            }
            ident = bytes(state.get("ident", b""))
            option = int(state.get("option", 0))
            prefix = ident + (b": " if ident else b"")
            line = prefix + payload
            if not line.endswith(b"\n"):
                line += b"\n"

            # With no /dev/log endpoint in the guest rootfs, libc syslog
            # calls are allowed to fail to deliver silently. Preserve the
            # observable LOG_PERROR behavior by also writing to stderr.
            if option & 0x20:  # LOG_PERROR
                write_stdio_payload(vm, 2, line)
            print(
                "BOOT_EXEC_USER_SYSLOG "
                f"priority=0x{priority:x} bytes={len(payload)} "
                f"perror={1 if option & 0x20 else 0}",
                flush=True,
            )
            return None

        return user_syslog

    if original == "snprintf":
        def user_snprintf(vm, args):
            if len(args) < 3:
                raise VMError("snprintf expects buffer,size,format,...")
            buf, size, fmt_ptr = map(int, args[:3])
            values = iter(int(value) for value in args[3:])

            def next_arg() -> int:
                try:
                    return next(values)
                except StopIteration as exc:
                    raise VMError(
                        "snprintf format consumes more arguments than supplied"
                    ) from exc

            payload = render_user_printf(vm, fmt_ptr, next_arg)
            if size:
                if buf == 0:
                    raise VMError("snprintf received null buffer with nonzero size")
                count = min(len(payload), size - 1)
                bulk_write = getattr(vm.memory, "bulk_write", None)
                if bulk_write is not None and count:
                    bulk_write(buf, payload[:count])
                else:
                    for offset, byte in enumerate(payload[:count]):
                        vm.memory.write(buf + offset, 8, byte)
                vm.memory.write(buf + count, 8, 0)
            return len(payload)

        return user_snprintf

    if original in {"fprintf", "vfprintf"}:
        def user_fprintf(vm, args):
            if original == "fprintf":
                if len(args) < 2:
                    raise VMError("fprintf expects FILE*,format,...")
                stream, fmt_ptr = map(int, args[:2])
                values = iter(int(value) for value in args[2:])

                def next_arg() -> int:
                    try:
                        return next(values)
                    except StopIteration as exc:
                        raise VMError(
                            "fprintf format consumes more arguments than supplied"
                        ) from exc
            else:
                if len(args) != 3:
                    raise VMError("vfprintf expects FILE*,format,va_list")
                stream, fmt_ptr, cursor = map(int, args)

                def next_arg() -> int:
                    nonlocal cursor
                    value = vm.memory.read(cursor, 64)
                    cursor += 8
                    return int(value)

            payload = render_user_printf(vm, fmt_ptr, next_arg)
            # Keep guest stdout/stderr byte-clean.  BusyBox may assemble one
            # logical line with several libc stdio calls, so host diagnostics
            # here would splice BOOT_EXEC text into the guest stream.
            return write_stdio_payload(vm, stream, payload)

        return user_fprintf

    if original in {"printf", "vprintf"}:
        def user_printf(vm, args):
            if original == "printf":
                if len(args) < 1:
                    raise VMError("printf expects format,...")
                fmt_ptr = int(args[0])
                values = iter(int(value) for value in args[1:])

                def next_arg() -> int:
                    try:
                        return next(values)
                    except StopIteration as exc:
                        raise VMError(
                            "printf format consumes more arguments than supplied"
                        ) from exc
            else:
                if len(args) != 2:
                    raise VMError("vprintf expects format,va_list")
                fmt_ptr, cursor = map(int, args)

                def next_arg() -> int:
                    nonlocal cursor
                    value = vm.memory.read(cursor, 64)
                    cursor += 8
                    return int(value)

            payload = render_user_printf(vm, fmt_ptr, next_arg)
            return write_stdio_payload(vm, 1, payload)

        return user_printf

    if original == "qsort":
        def user_qsort(vm, args):
            if len(args) != 4:
                raise VMError("qsort expects base,nmemb,size,compar")
            base, nmemb, size, compar = map(int, args)
            if nmemb <= 1 or size == 0:
                return None
            if size < 0 or nmemb < 0:
                raise VMError(
                    f"qsort received invalid dimensions nmemb={nmemb} size={size}"
                )

            bulk_read = getattr(vm.memory, "bulk_read", None)
            bulk_write = getattr(vm.memory, "bulk_write", None)

            def read_record(address: int) -> bytes:
                if bulk_read is not None:
                    return bytes(bulk_read(address, size))
                return bytes(
                    vm.memory.read(address + offset, 8)
                    for offset in range(size)
                )

            def write_record(address: int, payload: bytes) -> None:
                if bulk_write is not None:
                    bulk_write(address, payload)
                    return
                for offset, byte in enumerate(payload):
                    vm.memory.write(address + offset, 8, byte)

            records = [
                read_record(base + index * size)
                for index in range(nmemb)
            ]
            scratch = vm.alloc_bytes(size * 2, align=8)
            left_ptr = scratch
            right_ptr = scratch + size
            comparator_name = _guest_function_name_from_descriptor(vm, compar)
            calls = 0

            def compare(left: bytes, right: bytes) -> int:
                nonlocal calls
                write_record(left_ptr, left)
                write_record(right_ptr, right)
                call_result = _call_guest_descriptor_preserving_control(
                    vm,
                    compar,
                    (left_ptr, right_ptr),
                    result_count=1,
                    max_extra_steps=500_000,
                )
                if call_result is HOST_CONTROL_TRANSFER:
                    raise VMError(
                        "qsort comparator performed a non-returning control transfer"
                    )
                raw, = call_result
                calls += 1
                raw32 = raw & 0xFFFFFFFF
                return (
                    raw32 - (1 << 32)
                    if raw32 & (1 << 31)
                    else raw32
                )

            records.sort(key=cmp_to_key(compare))
            for index, payload in enumerate(records):
                write_record(base + index * size, payload)

            print(
                "BOOT_EXEC_USER_QSORT "
                f"compar={comparator_name} nmemb={nmemb} size={size} calls={calls}",
                flush=True,
            )
            return None

        return user_qsort

    if original == "bsearch":
        def user_bsearch(vm, args):
            if len(args) != 5:
                raise VMError("bsearch expects key,base,nmemb,size,compar")
            key, base, nmemb, size, compar = map(int, args)
            if size <= 0:
                return 0

            def preview_cstring(ptr: int, limit: int = 48) -> str:
                if not ptr:
                    return "<null>"
                data = bytearray()
                for i in range(limit):
                    byte = vm.memory.read(ptr + i, 8)
                    if byte == 0:
                        break
                    data.append(byte)
                return data.decode("utf-8", errors="backslashreplace")

            comparator_name = _guest_function_name_from_descriptor(vm, compar)
            key_text = preview_cstring(key)
            lo = 0
            hi = nmemb
            calls = 0
            while lo < hi:
                mid = lo + (hi - lo) // 2
                element = base + mid * size
                element_name_ptr = vm.memory.read(element, 64)
                raw_name = preview_cstring(element_name_ptr)
                first = (
                    vm.memory.read(element_name_ptr, 8)
                    if element_name_ptr else 0
                )
                logical_name = (
                    preview_cstring(element_name_ptr + 1)
                    if 0 < first < 8
                    else raw_name
                )
                raw, = _call_guest_descriptor_preserving_control(
                    vm,
                    compar,
                    (key, element),
                    result_count=1,
                    max_extra_steps=500_000,
                )
                calls += 1
                raw32 = raw & 0xFFFFFFFF
                cmp_value = (
                    raw32 - (1 << 32)
                    if raw32 & (1 << 31)
                    else raw32
                )
                print(
                    "BOOT_EXEC_USER_BSEARCH_CMP "
                    f"compar={comparator_name} key={key_text!r} "
                    f"nmemb={nmemb} size={size} lo={lo} hi={hi} mid={mid} "
                    f"element=0x{element:x} name_ptr=0x{element_name_ptr:x} "
                    f"first=0x{first:x} name={logical_name!r} "
                    f"raw=0x{raw:x} cmp={cmp_value}",
                    flush=True,
                )
                if cmp_value < 0:
                    hi = mid
                elif cmp_value > 0:
                    lo = mid + 1
                else:
                    print(
                        "BOOT_EXEC_USER_BSEARCH "
                        f"compar={comparator_name} key={key_text!r} "
                        f"nmemb={nmemb} size={size} calls={calls} "
                        f"index={mid} result=0x{element:x}",
                        flush=True,
                    )
                    return element

            print(
                "BOOT_EXEC_USER_BSEARCH "
                f"compar={comparator_name} key={key_text!r} "
                f"nmemb={nmemb} size={size} calls={calls} result=0x0",
                flush=True,
            )
            return 0

        return user_bsearch

    if original == "fflush":
        def user_fflush(_vm, args):
            if len(args) != 1:
                raise VMError("fflush expects FILE*")
            return 0
        return user_fflush

    if original == "clearerr":
        def user_clearerr(vm, args):
            if len(args) != 1:
                raise VMError("clearerr expects FILE*")
            state = stdio_state(vm, int(args[0]))
            if state is not None:
                state["eof"] = False
                state["error"] = False
            return None
        return user_clearerr

    if original in {"ferror", "ferror_unlocked"}:
        def user_ferror(vm, args):
            if len(args) != 1:
                raise VMError(f"{original} expects FILE*")
            state = stdio_state(vm, int(args[0]))
            return 1 if state is not None and state.get("error") else 0
        return user_ferror

    if original == "fputs_unlocked":
        def user_fputs(vm, args):
            if len(args) != 2:
                raise VMError("fputs_unlocked expects string,FILE*")
            ptr, stream = map(int, args)
            length = 0
            bulk_strlen = getattr(vm.memory, "bulk_strlen", None)
            if bulk_strlen is not None:
                length = bulk_strlen(ptr)
            else:
                while vm.memory.read(ptr + length, 8) != 0:
                    length += 1
            raw = user_syscall(
                vm,
                (64, stdio_fd(vm, stream), ptr, length, 0, 0, 0),
            )
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                set_errno(vm, -signed)
                return (1 << 64) - 1
            return 0
        return user_fputs

    if original == "fwrite":
        def user_fwrite(vm, args):
            if len(args) != 4:
                raise VMError("fwrite expects ptr,size,nmemb,FILE*")
            ptr, size, nmemb, stream = map(int, args)
            if size == 0 or nmemb == 0:
                return 0
            fd = stdio_fd(vm, stream)
            if fd < 0:
                set_errno(vm, 9)
                return 0

            total = size * nmemb
            written = 0
            state = stdio_state(vm, stream)
            while written < total:
                raw = user_syscall(
                    vm,
                    (
                        64,  # write
                        fd,
                        ptr + written,
                        total - written,
                        0,
                        0,
                        0,
                    ),
                )
                signed = raw - (1 << 64) if raw & (1 << 63) else raw
                if signed < 0:
                    if state is not None:
                        state["error"] = True
                    set_errno(vm, -signed)
                    break
                if signed == 0:
                    break
                written += int(signed)
            return written // size

        return user_fwrite

    if original in {"fputc", "putc_unlocked"}:
        def user_putc(vm, args):
            if len(args) != 2:
                raise VMError(f"{original} expects char,FILE*")
            ch, stream = map(int, args)
            ptr = vm.alloc_bytes(1, align=1)
            vm.memory.write(ptr, 8, ch & 0xFF)
            raw = user_syscall(
                vm,
                (64, stdio_fd(vm, stream), ptr, 1, 0, 0, 0),
            )
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                set_errno(vm, -signed)
                return (1 << 64) - 1
            return ch & 0xFF
        return user_putc

    if original in {"putchar", "putchar_unlocked"}:
        def user_putchar(vm, args):
            if len(args) != 1:
                raise VMError(f"{original} expects char")
            ch = int(args[0])
            ptr = vm.alloc_bytes(1, align=1)
            vm.memory.write(ptr, 8, ch & 0xFF)
            raw = user_syscall(vm, (64, 1, ptr, 1, 0, 0, 0))
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                set_errno(vm, -signed)
                return (1 << 64) - 1
            return ch & 0xFF
        return user_putchar

    if original == "puts":
        def user_puts(vm, args):
            if len(args) != 1:
                raise VMError("puts expects string")
            ptr = int(args[0])
            length = 0
            bulk_strlen = getattr(vm.memory, "bulk_strlen", None)
            if bulk_strlen is not None:
                length = bulk_strlen(ptr)
            else:
                while vm.memory.read(ptr + length, 8) != 0:
                    length += 1
            raw = user_syscall(vm, (64, 1, ptr, length, 0, 0, 0))
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                set_errno(vm, -signed)
                return (1 << 64) - 1
            nl = vm.alloc_bytes(1, align=1)
            vm.memory.write(nl, 8, 10)
            raw = user_syscall(vm, (64, 1, nl, 1, 0, 0, 0))
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                set_errno(vm, -signed)
                return (1 << 64) - 1
            return 0
        return user_puts

    def encode_dev(major: int, minor: int) -> int:
        return (
            (minor & 0xFF)
            | ((major & 0xFFF) << 8)
            | ((minor & ~0xFF) << 12)
            | ((major & ~0xFFF) << 32)
        ) & ((1 << 64) - 1)

    def write_user_stat_from_statx(vm, stat_ptr: int, statx_ptr: int) -> None:
        # RISC-V glibc struct stat (128 bytes), probed with the exact
        # cross libc used by the BusyBox carrier.
        for i in range(128):
            vm.memory.write(stat_ptr + i, 8, 0)

        stx_blksize = vm.memory.read(statx_ptr + 4, 32)
        stx_nlink = vm.memory.read(statx_ptr + 16, 32)
        stx_uid = vm.memory.read(statx_ptr + 20, 32)
        stx_gid = vm.memory.read(statx_ptr + 24, 32)
        stx_mode = vm.memory.read(statx_ptr + 28, 16)
        stx_ino = vm.memory.read(statx_ptr + 32, 64)
        stx_size = vm.memory.read(statx_ptr + 40, 64)
        stx_blocks = vm.memory.read(statx_ptr + 48, 64)
        stx_rdev_major = vm.memory.read(statx_ptr + 128, 32)
        stx_rdev_minor = vm.memory.read(statx_ptr + 132, 32)
        stx_dev_major = vm.memory.read(statx_ptr + 136, 32)
        stx_dev_minor = vm.memory.read(statx_ptr + 140, 32)

        vm.memory.write(
            stat_ptr + 0,
            64,
            encode_dev(stx_dev_major, stx_dev_minor),
        )
        vm.memory.write(stat_ptr + 8, 64, stx_ino)
        vm.memory.write(stat_ptr + 16, 32, stx_mode)
        vm.memory.write(stat_ptr + 20, 32, stx_nlink)
        vm.memory.write(stat_ptr + 24, 32, stx_uid)
        vm.memory.write(stat_ptr + 28, 32, stx_gid)
        vm.memory.write(
            stat_ptr + 32,
            64,
            encode_dev(stx_rdev_major, stx_rdev_minor),
        )
        vm.memory.write(stat_ptr + 48, 64, stx_size)
        vm.memory.write(stat_ptr + 56, 64, stx_blksize)
        vm.memory.write(stat_ptr + 64, 64, stx_blocks)

        # struct statx timestamps: atime@64, btime@80, ctime@96, mtime@112.
        # struct stat timestamps: atim@72, mtim@88, ctim@104.
        for dst_off, src_off in ((72, 64), (88, 112), (104, 96)):
            sec = vm.memory.read(statx_ptr + src_off, 64)
            nsec = vm.memory.read(statx_ptr + src_off + 8, 32)
            vm.memory.write(stat_ptr + dst_off, 64, sec)
            vm.memory.write(stat_ptr + dst_off + 8, 64, nsec)

    if original in {"stat", "stat64", "lstat", "lstat64"}:
        def user_stat(vm, args):
            if len(args) != 2:
                raise VMError(f"{original} expects path,stat")
            path_ptr, stat_ptr = map(int, args)
            if "__se_sys_statx" not in vm.program.functions:
                raise VMError("Linux image is missing __se_sys_statx")

            statx_ptr = vm.alloc_bytes(256, align=8)
            bulk_fill = getattr(vm.memory, "bulk_fill", None)
            if bulk_fill is not None:
                bulk_fill(statx_ptr, 0, 256)
            else:
                for i in range(256):
                    vm.memory.write(statx_ptr + i, 8, 0)

            at_fdcwd = (-100) & ((1 << 64) - 1)
            flags = 0x100 if original in {"lstat", "lstat64"} else 0
            statx_basic_stats = 0x000007FF
            result, = _call_linux_function_preserving_control(
                vm,
                "__se_sys_statx",
                (at_fdcwd, path_ptr, flags, statx_basic_stats, statx_ptr),
                result_count=1,
                max_extra_steps=4_000_000,
            )
            signed = result - (1 << 64) if result & (1 << 63) else result
            if signed < 0:
                set_errno(vm, -signed)
                return (1 << 64) - 1

            write_user_stat_from_statx(vm, stat_ptr, statx_ptr)
            print(
                "BOOT_EXEC_USER_STAT "
                f"kind={original} path_ptr=0x{path_ptr:x} "
                f"stat_ptr=0x{stat_ptr:x} "
                f"mode=0{vm.memory.read(stat_ptr + 16, 32):o} "
                f"size={vm.memory.read(stat_ptr + 48, 64)}",
                flush=True,
            )
            return 0

        return user_stat

    if original == "tcgetattr":
        def user_tcgetattr(vm, args):
            if len(args) != 2:
                raise VMError("tcgetattr expects fd,termios")
            fd, termios_ptr = map(int, args)
            raw = user_syscall(
                vm,
                (29, fd, 0x5401, termios_ptr, 0, 0, 0),
            )
            return libc_linux_result(vm, raw)

        return user_tcgetattr

    if original == "tcsetattr":
        def user_tcsetattr(vm, args):
            if len(args) != 3:
                raise VMError("tcsetattr expects fd,optional_actions,termios")
            fd, optional_actions, termios_ptr = map(int, args)
            request = {
                0: 0x5402,  # TCSANOW -> TCSETS
                1: 0x5403,  # TCSADRAIN -> TCSETSW
                2: 0x5404,  # TCSAFLUSH -> TCSETSF
            }.get(optional_actions)
            if request is None:
                set_errno(vm, 22)
                return (1 << 64) - 1
            raw = user_syscall(
                vm,
                (29, fd, request, termios_ptr, 0, 0, 0),
            )
            return libc_linux_result(vm, raw)

        return user_tcsetattr

    if original == "ioctl":
        def user_ioctl(vm, args):
            if len(args) not in {2, 3}:
                raise VMError("ioctl expects fd,request[,arg]")
            fd = int(args[0])
            request = int(args[1])
            arg = int(args[2]) if len(args) == 3 else 0
            raw = user_syscall(
                vm,
                (29, fd, request, arg, 0, 0, 0),
            )
            return libc_linux_result(vm, raw)

        return user_ioctl

    if original == "isatty":
        def user_isatty(vm, args):
            if len(args) != 1:
                raise VMError("isatty expects fd")
            if "__se_sys_ioctl" not in vm.program.functions:
                raise VMError("Linux image is missing __se_sys_ioctl")
            fd = int(args[0])
            termios = vm.alloc_bytes(64, align=8)
            result, = _call_linux_function_preserving_control(
                vm,
                "__se_sys_ioctl",
                (fd, 0x5401, termios),
                result_count=1,
                max_extra_steps=2_000_000,
            )
            signed = result - (1 << 64) if result & (1 << 63) else result
            if signed < 0:
                set_errno(vm, -signed)
                return 0
            return 1

        return user_isatty

    if original == "strsep":
        def user_strsep(vm, args):
            if len(args) != 2:
                raise VMError("strsep expects char**,delim")
            stringp, delim_ptr = map(int, args)
            current = vm.memory.read(stringp, 64)
            if current == 0:
                return 0

            delimiters = set(read_user_cstring(vm, delim_ptr, 256))
            cursor = current
            while True:
                byte = vm.memory.read(cursor, 8)
                if byte == 0:
                    vm.memory.write(stringp, 64, 0)
                    return current
                if byte in delimiters:
                    vm.memory.write(cursor, 8, 0)
                    vm.memory.write(stringp, 64, cursor + 1)
                    return current
                cursor += 1

        return user_strsep

    if original == "strerror":
        messages = {
            0: b"Success",
            2: b"No such file or directory",
            5: b"Input/output error",
            9: b"Bad file descriptor",
            12: b"Cannot allocate memory",
            13: b"Permission denied",
            22: b"Invalid argument",
            38: b"Function not implemented",
            95: b"Operation not supported",
        }

        def user_strerror(vm, args):
            if len(args) != 1:
                raise VMError("strerror expects errno")
            err = int(args[0]) & 0xFFFFFFFF
            payload = messages.get(
                err,
                f"Unknown error {err}".encode("ascii"),
            ) + b"\0"
            ptr = vm.alloc_bytes(len(payload), align=1)
            bulk_write = getattr(vm.memory, "bulk_write", None)
            if bulk_write is not None:
                bulk_write(ptr, payload)
            else:
                for i, byte in enumerate(payload):
                    vm.memory.write(ptr + i, 8, byte)
            return ptr

        return user_strerror

    if original in {"fcntl", "fcntl64"}:
        def user_fcntl(vm, args):
            if len(args) not in {2, 3}:
                raise VMError(f"{original} expects fd,cmd[,arg]")
            fd, cmd = map(int, args[:2])
            arg = int(args[2]) if len(args) == 3 else 0
            raw = user_syscall(
                vm,
                (25, fd, cmd, arg, 0, 0, 0),
            )
            return libc_linux_result(vm, raw)
        return user_fcntl

    if original in {"dup", "dup2"}:
        def user_dup(vm, args):
            expected = 1 if original == "dup" else 2
            if len(args) != expected:
                raise VMError(f"{original} expects {expected} fd argument(s)")
            target = "__se_sys_dup" if original == "dup" else "__se_sys_dup2"
            if target not in vm.program.functions:
                raise VMError(f"Linux image is missing {target}")
            raw, = _call_linux_function_preserving_control(
                vm,
                target,
                tuple(int(value) for value in args),
                result_count=1,
                max_extra_steps=8_000_000,
            )
            return libc_linux_result(vm, raw)
        return user_dup

    if original == "poll":
        def user_poll(vm, args):
            if len(args) != 3:
                raise VMError("poll expects fds,nfds,timeout")
            if "__se_sys_poll" not in vm.program.functions:
                raise VMError("Linux image is missing __se_sys_poll")
            raw, = _call_linux_function_preserving_control(
                vm,
                "__se_sys_poll",
                tuple(int(value) for value in args),
                result_count=1,
                max_extra_steps=8_000_000,
            )
            return libc_linux_result(vm, raw)
        return user_poll

    if original in {"getrlimit", "getrlimit64", "setrlimit", "setrlimit64"}:
        def user_rlimit(vm, args):
            if len(args) != 2:
                raise VMError(f"{original} expects resource,rlim")
            resource, rlim_ptr = map(int, args)
            target = "__se_sys_prlimit64"
            if target not in vm.program.functions:
                raise VMError(f"Linux image is missing {target}")
            if original.startswith("get"):
                new_ptr = 0
                old_ptr = rlim_ptr
            else:
                new_ptr = rlim_ptr
                old_ptr = 0
            raw, = _call_linux_function_preserving_control(
                vm,
                target,
                (0, resource, new_ptr, old_ptr),
                result_count=1,
                max_extra_steps=8_000_000,
            )
            return libc_linux_result(vm, raw)
        return user_rlimit

    if original == "setsid":
        def user_setsid(vm, args):
            if args:
                raise VMError("setsid expects no arguments")
            raw = user_syscall(
                vm,
                (157, 0, 0, 0, 0, 0, 0),
            )
            return libc_linux_result(vm, raw)

        return user_setsid

    if original in {"open", "open64"}:
        def user_open(vm, args):
            if len(args) not in {2, 3}:
                raise VMError(f"{original} expects path,flags[,mode]")
            path, flags = map(int, args[:2])
            mode = int(args[2]) if len(args) == 3 else 0
            raw = user_syscall(
                vm,
                (
                    56,  # openat
                    (-100) & ((1 << 64) - 1),  # AT_FDCWD
                    path,
                    flags,
                    mode,
                    0,
                    0,
                ),
            )
            return libc_linux_result(vm, raw)
        return user_open

    if original == "opendir":
        def user_opendir(vm, args):
            if len(args) != 1:
                raise VMError("opendir expects path")
            path = int(args[0])
            # asm-generic Linux flags: O_DIRECTORY | O_CLOEXEC.
            raw = user_syscall(
                vm,
                (
                    56,
                    (-100) & ((1 << 64) - 1),
                    path,
                    0x10000 | 0x80000,
                    0,
                    0,
                    0,
                ),
            )
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                set_errno(vm, -signed)
                return 0

            streams = getattr(vm, "user_dir_streams", None)
            if streams is None:
                streams = {}
                vm.user_dir_streams = streams
            handle = vm.alloc_bytes(32, align=8)
            buffer_ptr = vm.alloc_bytes(4096, align=8)
            streams[handle] = {
                "fd": int(raw),
                "buffer": buffer_ptr,
                "size": 0,
                "pos": 0,
            }
            print(
                "BOOT_EXEC_USER_OPENDIR "
                f"path=0x{path:x} fd={int(raw)} handle=0x{handle:x}",
                flush=True,
            )
            return handle
        return user_opendir

    if original in {"readdir", "readdir64"}:
        def user_readdir(vm, args):
            if len(args) != 1:
                raise VMError(f"{original} expects DIR*")
            handle = int(args[0])
            streams = getattr(vm, "user_dir_streams", {})
            state = streams.get(handle)
            if state is None:
                set_errno(vm, 9)
                return 0

            while True:
                if state["pos"] < state["size"]:
                    entry = state["buffer"] + state["pos"]
                    reclen = vm.memory.read(entry + 16, 16)
                    if reclen < 20 or state["pos"] + reclen > state["size"]:
                        raise VMError(
                            "getdents64 returned malformed dirent: "
                            f"pos={state['pos']} size={state['size']} "
                            f"reclen={reclen}"
                        )
                    state["pos"] += reclen
                    return entry

                raw = user_syscall(
                    vm,
                    (
                        61,  # getdents64
                        state["fd"],
                        state["buffer"],
                        4096,
                        0,
                        0,
                        0,
                    ),
                )
                signed = raw - (1 << 64) if raw & (1 << 63) else raw
                if signed < 0:
                    set_errno(vm, -signed)
                    return 0
                if signed == 0:
                    return 0
                state["size"] = int(signed)
                state["pos"] = 0
        return user_readdir

    if original == "closedir":
        def user_closedir(vm, args):
            if len(args) != 1:
                raise VMError("closedir expects DIR*")
            handle = int(args[0])
            streams = getattr(vm, "user_dir_streams", {})
            state = streams.pop(handle, None)
            if state is None:
                set_errno(vm, 9)
                return (1 << 64) - 1
            raw = user_syscall(
                vm,
                (57, state["fd"], 0, 0, 0, 0, 0),
            )
            return libc_linux_result(vm, raw)
        return user_closedir

    if original == "getcwd":
        def user_getcwd(vm, args):
            if len(args) != 2:
                raise VMError("getcwd expects buffer,size")
            buf, size = map(int, args)
            if "__se_sys_getcwd" not in vm.program.functions:
                raise VMError("Linux image is missing __se_sys_getcwd")

            if buf == 0:
                alloc_size = size if size else 4096
                if alloc_size <= 0:
                    set_errno(vm, 22)
                    return 0
                buf = vm.alloc_bytes(alloc_size, align=1)
                size = alloc_size
            elif size == 0:
                set_errno(vm, 22)
                return 0

            result, = _call_linux_function_preserving_control(
                vm,
                "__se_sys_getcwd",
                (buf, size),
                result_count=1,
                max_extra_steps=2_000_000,
            )
            signed = result - (1 << 64) if result & (1 << 63) else result
            if signed < 0:
                set_errno(vm, -signed)
                return 0
            return buf

        return user_getcwd

    def check_signal_number(vm, signum: int) -> bool:
        if 1 <= signum <= 64:
            return True
        set_errno(vm, 22)
        return False

    if original in {
        "sigfillset",
        "sigemptyset",
        "sigaddset",
        "sigdelset",
        "sigismember",
    }:
        def sigset_op(vm, args):
            if original in {"sigfillset", "sigemptyset"}:
                if len(args) != 1:
                    raise VMError(f"{original} expects sigset_t*")
                ptr = int(args[0])
                fill = 0xFF if original == "sigfillset" else 0x00
                bulk_fill = getattr(vm.memory, "bulk_fill", None)
                if bulk_fill is not None:
                    bulk_fill(ptr, fill, 128)
                else:
                    for i in range(128):
                        vm.memory.write(ptr + i, 8, fill)
                return 0

            if len(args) != 2:
                raise VMError(f"{original} expects sigset_t*,signum")
            ptr, signum = map(int, args)
            if not check_signal_number(vm, signum):
                return -1 & ((1 << 64) - 1)
            bit = signum - 1
            byte_addr = ptr + (bit // 8)
            mask = 1 << (bit % 8)
            old = vm.memory.read(byte_addr, 8)
            if original == "sigaddset":
                vm.memory.write(byte_addr, 8, old | mask)
                return 0
            if original == "sigdelset":
                vm.memory.write(byte_addr, 8, old & ~mask)
                return 0
            return 1 if old & mask else 0

        return sigset_op

    def call_rt_sigaction(vm, signum: int, act_ptr: int, old_ptr: int) -> int:
        if "__se_sys_rt_sigaction" not in vm.program.functions:
            raise VMError("Linux image is missing __se_sys_rt_sigaction")
        kact = 0
        kold = 0
        if act_ptr:
            kact = vm.alloc_bytes(24, align=8)
            # glibc/RISC-V struct sigaction:
            #   handler @0, 128-byte mask @8, flags @136, restorer @144.
            # MiniMachine kernel struct sigaction:
            #   handler @0, flags @8, kernel_sigset_t(low64) @16.
            handler = vm.memory.read(act_ptr + 0, 64)
            mask = vm.memory.read(act_ptr + 8, 64)
            flags = vm.memory.read(act_ptr + 136, 32)
            vm.memory.write(kact + 0, 64, handler)
            vm.memory.write(kact + 8, 64, flags)
            vm.memory.write(kact + 16, 64, mask)
        if old_ptr:
            kold = vm.alloc_bytes(24, align=8)

        result, = _call_linux_function_preserving_control(
            vm,
            "__se_sys_rt_sigaction",
            (signum, kact, kold, 8),
            result_count=1,
            max_extra_steps=2_000_000,
        )
        signed = result - (1 << 64) if result & (1 << 63) else result
        if signed < 0:
            set_errno(vm, -signed)
            return result

        if old_ptr:
            handler = vm.memory.read(kold + 0, 64)
            flags = vm.memory.read(kold + 8, 64)
            mask = vm.memory.read(kold + 16, 64)
            vm.memory.write(old_ptr + 0, 64, handler)
            for i in range(128):
                vm.memory.write(old_ptr + 8 + i, 8, 0)
            vm.memory.write(old_ptr + 8, 64, mask)
            vm.memory.write(old_ptr + 136, 32, flags)
            vm.memory.write(old_ptr + 144, 64, 0)
        return result

    if original == "sigaction":
        def user_sigaction(vm, args):
            if len(args) != 3:
                raise VMError("sigaction expects signum,act,oldact")
            signum, act_ptr, old_ptr = map(int, args)
            if not check_signal_number(vm, signum):
                return -1 & ((1 << 64) - 1)
            return call_rt_sigaction(vm, signum, act_ptr, old_ptr)
        return user_sigaction

    if original == "sigprocmask":
        def user_sigprocmask(vm, args):
            if len(args) != 3:
                raise VMError("sigprocmask expects how,set,oldset")
            how, set_ptr, old_ptr = map(int, args)
            if "__se_sys_rt_sigprocmask" not in vm.program.functions:
                raise VMError("Linux image is missing __se_sys_rt_sigprocmask")
            result, = _call_linux_function_preserving_control(
                vm,
                "__se_sys_rt_sigprocmask",
                (how, set_ptr, old_ptr, 8),
                result_count=1,
                max_extra_steps=2_000_000,
            )
            signed = result - (1 << 64) if result & (1 << 63) else result
            if signed < 0:
                set_errno(vm, -signed)
                return -1 & ((1 << 64) - 1)
            return 0
        return user_sigprocmask

    if original == "signal":
        def user_signal(vm, args):
            if len(args) != 2:
                raise VMError("signal expects signum,handler")
            signum, handler = map(int, args)
            if not check_signal_number(vm, signum):
                return ((1 << 64) - 1)
            act = vm.alloc_bytes(152, align=8)
            old = vm.alloc_bytes(152, align=8)
            bulk_fill = getattr(vm.memory, "bulk_fill", None)
            if bulk_fill is not None:
                bulk_fill(act, 0, 152)
                bulk_fill(old, 0, 152)
            else:
                for i in range(152):
                    vm.memory.write(act + i, 8, 0)
                    vm.memory.write(old + i, 8, 0)
            vm.memory.write(act + 0, 64, handler)
            # SA_RESTART, matching the usual libc signal() semantics.
            vm.memory.write(act + 136, 32, 0x10000000)
            result = call_rt_sigaction(vm, signum, act, old)
            signed = result - (1 << 64) if result & (1 << 63) else result
            if signed < 0:
                return ((1 << 64) - 1)
            return vm.memory.read(old + 0, 64)
        return user_signal

    if original == "mallopt":
        def user_mallopt(_vm, args):
            if len(args) != 2:
                raise VMError("mallopt expects parameter,value")
            parameter, value = map(int, args)
            print(
                "BOOT_EXEC_USER_MALLOPT "
                f"parameter={parameter if parameter < (1 << 63) else parameter - (1 << 64)} "
                f"value={value if value < (1 << 63) else value - (1 << 64)}",
                flush=True,
            )
            # BusyBox uses mallopt only to tune the host libc allocator.
            # MiniMachine provides its own userspace allocation arena, so
            # accepting the hint is the correct semantic equivalent.
            return 1
        return user_mallopt

    if original == "getcwd":
        def user_getcwd(vm, args):
            if len(args) != 2:
                raise VMError("getcwd expects buffer,size")
            buf, size = map(int, args)
            allocated = False
            capacity = size

            if buf == 0:
                # GNU/POSIX extension used by BusyBox ash: getcwd(NULL, 0)
                # asks libc to allocate a sufficiently large result buffer.
                capacity = size if size else 4096
                buf = vm.alloc_bytes(capacity, align=16)
                allocations = getattr(vm, "user_allocations", None)
                if allocations is None:
                    allocations = {}
                    vm.user_allocations = allocations
                allocations[buf] = capacity
                allocated = True
            elif size == 0:
                set_errno(vm, 22)
                return 0

            raw = user_syscall(
                vm,
                (17, buf, capacity, 0, 0, 0, 0),
            )
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                set_errno(vm, -signed)
                if allocated:
                    allocations = getattr(vm, "user_allocations", None)
                    if allocations is not None:
                        allocations.pop(buf, None)
                return 0

            preview = bytearray()
            for index in range(min(capacity, 512)):
                byte = vm.memory.read(buf + index, 8)
                if byte == 0:
                    break
                preview.append(byte)
            print(
                "BOOT_EXEC_USER_GETCWD "
                f"ptr=0x{buf:x} size={capacity} "
                f"result={preview.decode('utf-8', errors='replace')!r}",
                flush=True,
            )
            return buf

        return user_getcwd

    if original == "malloc":
        def malloc(vm, args):
            if len(args) != 1:
                raise VMError("malloc expects size")
            size = int(args[0])
            ptr = vm.alloc_bytes(size, align=16)
            allocations = getattr(vm, "user_allocations", None)
            if allocations is None:
                allocations = {}
                vm.user_allocations = allocations
            allocations[ptr] = size
            return ptr
        return malloc

    if original == "free":
        def free(vm, args):
            if len(args) != 1:
                raise VMError("free expects pointer")
            ptr = int(args[0])
            allocations = getattr(vm, "user_allocations", None)
            if allocations is not None:
                allocations.pop(ptr, None)
            return None
        return free

    if original == "realloc":
        def realloc(vm, args):
            if len(args) != 2:
                raise VMError("realloc expects pointer,size")
            ptr, size = map(int, args)
            allocations = getattr(vm, "user_allocations", None)
            if allocations is None:
                allocations = {}
                vm.user_allocations = allocations
            if ptr == 0:
                out = vm.alloc_bytes(size, align=16)
                allocations[out] = size
                return out
            if size == 0:
                allocations.pop(ptr, None)
                return 0
            old_size = allocations.get(ptr)
            if old_size is None:
                raise VMError(
                    f"realloc received unknown userspace allocation 0x{ptr:x}"
                )
            out = vm.alloc_bytes(size, align=16)
            copy_size = min(old_size, size)
            bulk = getattr(vm.memory, "bulk_copy", None)
            if bulk is not None:
                bulk(out, ptr, copy_size)
            else:
                for i in range(copy_size):
                    vm.memory.write(
                        out + i,
                        8,
                        vm.memory.read(ptr + i, 8),
                    )
            allocations.pop(ptr, None)
            allocations[out] = size
            return out
        return realloc

    if original == "atoi":
        def user_atoi(vm, args):
            if len(args) != 1:
                raise VMError("atoi expects string")
            ptr = int(args[0])
            data = bytearray()
            for index in range(256):
                byte = vm.memory.read(ptr + index, 8)
                if byte == 0:
                    break
                data.append(byte)
            text = data.decode("ascii", errors="ignore")
            pos = 0
            while pos < len(text) and text[pos].isspace():
                pos += 1
            sign = 1
            if pos < len(text) and text[pos] in "+-":
                if text[pos] == "-":
                    sign = -1
                pos += 1
            value = 0
            while pos < len(text) and text[pos].isdigit():
                value = value * 10 + (ord(text[pos]) - ord("0"))
                pos += 1
            value *= sign
            print(
                "BOOT_EXEC_USER_ATOI "
                f"text={text!r} result={value}",
                flush=True,
            )
            return value & ((1 << 64) - 1)

        return user_atoi

    if original in {
        "strtol", "strtoll", "strtoul", "strtoull",
        "__isoc23_strtol", "__isoc23_strtoll",
        "__isoc23_strtoul", "__isoc23_strtoull",
    }:
        def user_strto(vm, args):
            if len(args) != 3:
                raise VMError(f"{original} expects string,endptr,base")
            nptr, endptr, base = map(int, args)
            raw_text = read_user_cstring(vm, nptr, 4096)
            text = raw_text.decode("ascii", errors="ignore")
            pos = 0
            while pos < len(text) and text[pos].isspace():
                pos += 1

            negative = False
            if pos < len(text) and text[pos] in "+-":
                negative = text[pos] == "-"
                pos += 1

            c23 = original.startswith("__isoc23_")
            signed = original.replace("__isoc23_", "") in {"strtol", "strtoll"}

            def digit_value(ch: str) -> int:
                code = ord(ch)
                if 48 <= code <= 57:
                    return code - 48
                if 65 <= code <= 90:
                    return code - 65 + 10
                if 97 <= code <= 122:
                    return code - 97 + 10
                return -1

            if base != 0 and not (2 <= base <= 36):
                set_errno(vm, 22)
                if endptr:
                    vm.memory.write(endptr, 64, nptr)
                return 0

            prefix_pos = pos
            if base == 0:
                if (
                    pos + 2 < len(text)
                    and text[pos] == "0"
                    and text[pos + 1] in "xX"
                    and 0 <= digit_value(text[pos + 2]) < 16
                ):
                    base = 16
                    pos += 2
                elif (
                    c23
                    and pos + 2 < len(text)
                    and text[pos] == "0"
                    and text[pos + 1] in "bB"
                    and 0 <= digit_value(text[pos + 2]) < 2
                ):
                    base = 2
                    pos += 2
                elif pos < len(text) and text[pos] == "0":
                    base = 8
                else:
                    base = 10
            elif (
                base == 16
                and pos + 2 < len(text)
                and text[pos] == "0"
                and text[pos + 1] in "xX"
                and 0 <= digit_value(text[pos + 2]) < 16
            ):
                pos += 2
            elif (
                c23
                and base == 2
                and pos + 2 < len(text)
                and text[pos] == "0"
                and text[pos + 1] in "bB"
                and 0 <= digit_value(text[pos + 2]) < 2
            ):
                pos += 2

            digit_start = pos
            value = 0
            while pos < len(text):
                digit = digit_value(text[pos])
                if digit < 0 or digit >= base:
                    break
                value = value * base + digit
                pos += 1

            if pos == digit_start:
                if endptr:
                    vm.memory.write(endptr, 64, nptr)
                result = 0
            else:
                if signed:
                    limit = (1 << 63) if negative else (1 << 63) - 1
                    if value > limit:
                        set_errno(vm, 34)
                        result = -(1 << 63) if negative else (1 << 63) - 1
                    else:
                        result = -value if negative else value
                else:
                    if value > (1 << 64) - 1:
                        set_errno(vm, 34)
                        result = (1 << 64) - 1
                    else:
                        result = (-value if negative else value) & ((1 << 64) - 1)
                if endptr:
                    vm.memory.write(endptr, 64, nptr + pos)

            print(
                "BOOT_EXEC_USER_STRTO "
                f"name={original} text={text!r} base={base} "
                f"end={pos if pos != digit_start else prefix_pos} "
                f"result={result}",
                flush=True,
            )
            return result & ((1 << 64) - 1)

        return user_strto

    if original in {"getopt_long", "getopt_long_only"}:
        def user_getopt_long(vm, args):
            if len(args) != 5:
                raise VMError(
                    f"{original} expects argc,argv,optstring,longopts,longindex"
                )
            argc, argv, optstring_ptr, longopts_ptr, longindex_ptr = map(
                int, args
            )
            optstring = read_user_cstring(
                vm, optstring_ptr, 4096
            ).decode("latin1")
            long_only = original == "getopt_long_only"

            def data_address(name: str) -> int:
                address = vm.program.symbol_addresses.get(
                    f"{external_prefix}{name}"
                )
                if address is None:
                    raise VMError(
                        f"{original} missing userspace external data {name}"
                    )
                return int(address)

            optarg_addr = data_address("optarg")
            optind_addr = data_address("optind")
            optopt_addr = data_address("optopt")

            def finish(index: int, next_pos: int, result: int) -> int:
                state["index"] = index
                state["next"] = next_pos
                state["optind_seen"] = index
                vm.memory.write(optind_addr, 32, index)
                return result & ((1 << 64) - 1)

            def parse_long_options():
                options = []
                if not longopts_ptr:
                    return options
                # glibc/musl struct option on 64-bit:
                #   const char *name; int has_arg; int *flag; int val;
                # with natural padding -> 32 bytes.
                for option_index in range(4096):
                    base = longopts_ptr + option_index * 32
                    name_ptr = vm.memory.read(base, 64)
                    if name_ptr == 0:
                        break
                    name = read_user_cstring(
                        vm, name_ptr, 4096
                    ).decode("latin1")
                    has_arg = vm.memory.read(base + 8, 32)
                    flag_ptr = vm.memory.read(base + 16, 64)
                    value = vm.memory.read(base + 24, 32)
                    options.append(
                        (option_index, name, has_arg, flag_ptr, value)
                    )
                else:
                    raise VMError(
                        f"{original} long option table is not terminated"
                    )
                return options

            long_options = parse_long_options()
            optind = int(vm.memory.read(optind_addr, 32))
            if optind <= 0:
                optind = 1

            state = getattr(vm, "user_getopt_state", None)
            if (
                state is None
                or state.get("argv") != argv
                or state.get("optstring") != optstring_ptr
                or state.get("longopts") != longopts_ptr
                or state.get("long_only") != long_only
                or state.get("argc") != argc
                or optind == 1 and state.get("optind_seen") != 1
            ):
                state = {
                    "argv": argv,
                    "optstring": optstring_ptr,
                    "longopts": longopts_ptr,
                    "long_only": long_only,
                    "argc": argc,
                    "index": optind,
                    "next": 0,
                    "optind_seen": optind,
                }
                vm.user_getopt_state = state
            elif state.get("index") != optind and state.get("next", 0) == 0:
                state["index"] = optind

            vm.memory.write(optarg_addr, 64, 0)

            while True:
                index = int(state["index"])
                next_pos = int(state.get("next", 0))
                if index >= argc:
                    return finish(index, 0, (1 << 64) - 1)

                arg_ptr = vm.memory.read(argv + index * 8, 64)
                if not arg_ptr:
                    return finish(index, 0, (1 << 64) - 1)
                arg = read_user_cstring(vm, arg_ptr, 4096).decode("latin1")

                if next_pos == 0:
                    if len(arg) < 2 or arg[0] != "-" or arg == "-":
                        return finish(index, 0, (1 << 64) - 1)
                    if arg == "--":
                        return finish(index + 1, 0, (1 << 64) - 1)

                    is_double_dash = arg.startswith("--")
                    long_candidate = is_double_dash
                    if long_only and not is_double_dash:
                        first = arg[1:2]
                        search_start = 1 if optstring.startswith(":") else 0
                        if (
                            len(arg) > 2
                            or not first
                            or optstring.find(first, search_start) < 0
                        ):
                            long_candidate = True

                    if long_candidate:
                        token_start = 2 if is_double_dash else 1
                        token = arg[token_start:]
                        name, sep, inline_value = token.partition("=")
                        exact = [
                            option
                            for option in long_options
                            if option[1] == name
                        ]
                        matches = exact or [
                            option
                            for option in long_options
                            if option[1].startswith(name)
                        ]
                        if len(matches) != 1:
                            return finish(index + 1, 0, ord("?"))

                        (
                            option_index,
                            option_name,
                            has_arg,
                            flag_ptr,
                            value,
                        ) = matches[0]
                        if longindex_ptr:
                            vm.memory.write(
                                longindex_ptr, 32, option_index
                            )
                        vm.memory.write(optopt_addr, 32, 0)

                        if has_arg == 0:
                            if sep:
                                return finish(index + 1, 0, ord("?"))
                            index += 1
                        elif has_arg == 1:
                            if sep:
                                value_offset = token_start + len(name) + 1
                                vm.memory.write(
                                    optarg_addr,
                                    64,
                                    arg_ptr + value_offset,
                                )
                                index += 1
                            elif index + 1 < argc:
                                value_ptr = vm.memory.read(
                                    argv + (index + 1) * 8,
                                    64,
                                )
                                vm.memory.write(
                                    optarg_addr, 64, value_ptr
                                )
                                index += 2
                            else:
                                index += 1
                                missing = (
                                    ord(":")
                                    if optstring.startswith(":")
                                    else ord("?")
                                )
                                return finish(index, 0, missing)
                        elif has_arg == 2:
                            if sep:
                                value_offset = token_start + len(name) + 1
                                vm.memory.write(
                                    optarg_addr,
                                    64,
                                    arg_ptr + value_offset,
                                )
                            index += 1
                        else:
                            raise VMError(
                                f"{original} invalid has_arg={has_arg} "
                                f"for --{option_name}"
                            )

                        print(
                            "BOOT_EXEC_USER_GETOPT_LONG "
                            f"name={option_name!r} index={option_index} "
                            f"has_arg={has_arg} optind={index} "
                            f"optarg=0x{vm.memory.read(optarg_addr, 64):x} "
                            f"flag=0x{flag_ptr:x} value={value}",
                            flush=True,
                        )
                        if flag_ptr:
                            vm.memory.write(flag_ptr, 32, value)
                            return finish(index, 0, 0)
                        return finish(index, 0, value)

                    next_pos = 1

                option = arg[next_pos]
                next_pos += 1
                option_code = ord(option) & 0xFF
                vm.memory.write(optopt_addr, 32, option_code)

                search_start = 1 if optstring.startswith(":") else 0
                pos = optstring.find(option, search_start)
                if option == ":" or pos < 0:
                    if next_pos >= len(arg):
                        index += 1
                        next_pos = 0
                    return finish(index, next_pos, ord("?"))

                requires_arg = (
                    pos + 1 < len(optstring)
                    and optstring[pos + 1] == ":"
                )
                optional_arg = (
                    requires_arg
                    and pos + 2 < len(optstring)
                    and optstring[pos + 2] == ":"
                )

                if requires_arg:
                    if next_pos < len(arg):
                        vm.memory.write(
                            optarg_addr, 64, arg_ptr + next_pos
                        )
                        index += 1
                        next_pos = 0
                    elif optional_arg:
                        index += 1
                        next_pos = 0
                    elif index + 1 < argc:
                        index += 1
                        value_ptr = vm.memory.read(argv + index * 8, 64)
                        vm.memory.write(optarg_addr, 64, value_ptr)
                        index += 1
                        next_pos = 0
                    else:
                        index += 1
                        next_pos = 0
                        missing = (
                            ord(":")
                            if optstring.startswith(":")
                            else ord("?")
                        )
                        return finish(index, next_pos, missing)
                elif next_pos >= len(arg):
                    index += 1
                    next_pos = 0

                print(
                    "BOOT_EXEC_USER_GETOPT_LONG_SHORT "
                    f"option={option!r} optind={index} "
                    f"optarg=0x{vm.memory.read(optarg_addr, 64):x}",
                    flush=True,
                )
                return finish(index, next_pos, option_code)

        return user_getopt_long

    if original == "getopt":
        def user_getopt(vm, args):
            if len(args) != 3:
                raise VMError("getopt expects argc,argv,optstring")
            argc, argv, optstring_ptr = map(int, args)
            optstring = read_user_cstring(
                vm, optstring_ptr, 4096
            ).decode("latin1")

            def data_address(name: str) -> int:
                address = vm.program.symbol_addresses.get(
                    f"{external_prefix}{name}"
                )
                if address is None:
                    raise VMError(
                        f"getopt missing userspace external data {name}"
                    )
                return int(address)

            optarg_addr = data_address("optarg")
            optind_addr = data_address("optind")
            optopt_addr = data_address("optopt")

            optind = int(vm.memory.read(optind_addr, 32))
            if optind <= 0:
                optind = 1

            state = getattr(vm, "user_getopt_state", None)
            if (
                state is None
                or state.get("argv") != argv
                or state.get("optstring") != optstring_ptr
                or state.get("argc") != argc
                or optind == 1 and state.get("optind_seen") != 1
            ):
                state = {
                    "argv": argv,
                    "optstring": optstring_ptr,
                    "argc": argc,
                    "index": optind,
                    "next": 0,
                    "optind_seen": optind,
                }
                vm.user_getopt_state = state
            elif state.get("index") != optind and state.get("next", 0) == 0:
                state["index"] = optind

            vm.memory.write(optarg_addr, 64, 0)

            while True:
                index = int(state["index"])
                next_pos = int(state.get("next", 0))

                if index >= argc:
                    vm.memory.write(optind_addr, 32, index)
                    state["optind_seen"] = index
                    return (1 << 64) - 1

                arg_ptr = vm.memory.read(argv + index * 8, 64)
                if not arg_ptr:
                    vm.memory.write(optind_addr, 32, index)
                    state["optind_seen"] = index
                    return (1 << 64) - 1
                arg = read_user_cstring(vm, arg_ptr, 4096).decode("latin1")

                if next_pos == 0:
                    if len(arg) < 2 or arg[0] != "-" or arg == "-":
                        vm.memory.write(optind_addr, 32, index)
                        state["optind_seen"] = index
                        return (1 << 64) - 1
                    if arg == "--":
                        index += 1
                        state["index"] = index
                        state["next"] = 0
                        vm.memory.write(optind_addr, 32, index)
                        state["optind_seen"] = index
                        return (1 << 64) - 1
                    next_pos = 1

                option = arg[next_pos]
                next_pos += 1
                option_code = ord(option) & 0xFF
                vm.memory.write(optopt_addr, 32, option_code)

                search_start = 1 if optstring.startswith(":") else 0
                pos = optstring.find(option, search_start)
                if option == ":" or pos < 0:
                    if next_pos >= len(arg):
                        index += 1
                        next_pos = 0
                    state["index"] = index
                    state["next"] = next_pos
                    vm.memory.write(optind_addr, 32, index)
                    state["optind_seen"] = index
                    return ord("?")

                requires_arg = (
                    pos + 1 < len(optstring)
                    and optstring[pos + 1] == ":"
                )
                optional_arg = (
                    requires_arg
                    and pos + 2 < len(optstring)
                    and optstring[pos + 2] == ":"
                )

                if requires_arg:
                    if next_pos < len(arg):
                        vm.memory.write(
                            optarg_addr, 64, arg_ptr + next_pos
                        )
                        index += 1
                        next_pos = 0
                    elif optional_arg:
                        vm.memory.write(optarg_addr, 64, 0)
                        index += 1
                        next_pos = 0
                    elif index + 1 < argc:
                        index += 1
                        value_ptr = vm.memory.read(argv + index * 8, 64)
                        vm.memory.write(optarg_addr, 64, value_ptr)
                        index += 1
                        next_pos = 0
                    else:
                        index += 1
                        next_pos = 0
                        state["index"] = index
                        state["next"] = next_pos
                        vm.memory.write(optind_addr, 32, index)
                        state["optind_seen"] = index
                        return ord(":") if optstring.startswith(":") else ord("?")
                elif next_pos >= len(arg):
                    index += 1
                    next_pos = 0

                state["index"] = index
                state["next"] = next_pos
                vm.memory.write(optind_addr, 32, index)
                state["optind_seen"] = index
                print(
                    "BOOT_EXEC_USER_GETOPT "
                    f"option={option!r} optind={index} "
                    f"optarg=0x{vm.memory.read(optarg_addr, 64):x}",
                    flush=True,
                )
                return option_code

        return user_getopt

    if original == "gnu_dev_major":
        def user_gnu_dev_major(vm, args):
            if len(args) != 1:
                raise VMError("gnu_dev_major expects dev")
            dev = int(args[0]) & ((1 << 64) - 1)
            return ((dev >> 8) & 0xFFF) | ((dev >> 32) & 0xFFFFF000)

        return user_gnu_dev_major

    if original == "gnu_dev_minor":
        def user_gnu_dev_minor(vm, args):
            if len(args) != 1:
                raise VMError("gnu_dev_minor expects dev")
            dev = int(args[0]) & ((1 << 64) - 1)
            return (dev & 0xFF) | ((dev >> 12) & 0xFFFFFF00)

        return user_gnu_dev_minor

    if original == "gnu_dev_makedev":
        def user_gnu_dev_makedev(vm, args):
            if len(args) != 2:
                raise VMError("gnu_dev_makedev expects major,minor")
            major, minor = (int(value) & 0xFFFFFFFF for value in args)
            return (
                (minor & 0xFF)
                | ((major & 0xFFF) << 8)
                | ((minor & ~0xFF) << 12)
                | ((major & ~0xFFF) << 32)
            ) & ((1 << 64) - 1)

        return user_gnu_dev_makedev

    if original == "vasprintf":
        def user_vasprintf(vm, args):
            if len(args) != 3:
                raise VMError("vasprintf expects strp,format,va_list")
            strp, fmt_ptr, ap = map(int, args)
            if not strp:
                set_errno(vm, 22)
                return (1 << 64) - 1

            renderer = _user_libc_callback(
                f"{external_prefix}vsnprintf",
                errno_address,
            )
            if renderer is None:
                raise VMError("vasprintf is missing vsnprintf renderer")

            length = renderer(vm, (0, 0, fmt_ptr, ap))
            if length is HOST_CONTROL_TRANSFER:
                return HOST_CONTROL_TRANSFER
            length = int(length)
            if length < 0:
                return (1 << 64) - 1

            buffer = vm.alloc_bytes(length + 1, align=1)
            written = renderer(vm, (buffer, length + 1, fmt_ptr, ap))
            if written is HOST_CONTROL_TRANSFER:
                return HOST_CONTROL_TRANSFER
            written = int(written)
            if written < 0:
                return (1 << 64) - 1
            vm.memory.write(strp, 64, buffer)
            print(
                "BOOT_EXEC_USER_VASPRINTF "
                f"fmt=0x{fmt_ptr:x} ap=0x{ap:x} "
                f"buffer=0x{buffer:x} length={written}",
                flush=True,
            )
            return written

        return user_vasprintf

    if original == "vsnprintf":
        def user_vsnprintf(vm, args):
            if len(args) != 4:
                raise VMError("vsnprintf expects buffer,size,format,va_list")
            outbuf, size, fmt_ptr, ap = map(int, args)

            def read_cstring(ptr: int, limit: int = 1 << 20) -> bytes:
                if not ptr:
                    return b""
                data = bytearray()
                bulk_strlen = getattr(vm.memory, "bulk_strlen", None)
                if bulk_strlen is not None:
                    length = min(int(bulk_strlen(ptr)), limit)
                    bulk_read = getattr(vm.memory, "bulk_read", None)
                    if bulk_read is not None:
                        return bytes(bulk_read(ptr, length))
                for index in range(limit):
                    byte = vm.memory.read(ptr + index, 8)
                    if byte == 0:
                        break
                    data.append(byte)
                return bytes(data)

            fmt = read_cstring(fmt_ptr, 4096).decode("latin1")
            cursor = ap

            def next_arg() -> int:
                nonlocal cursor
                value = vm.memory.read(cursor, 64)
                cursor += 8
                return int(value)

            def signed_value(value: int, bits: int) -> int:
                mask = (1 << bits) - 1
                value &= mask
                sign = 1 << (bits - 1)
                return value - (1 << bits) if value & sign else value

            def unsigned_value(value: int, bits: int) -> int:
                return value & ((1 << bits) - 1)

            out = bytearray()
            index = 0
            while index < len(fmt):
                if fmt[index] != "%":
                    out.append(ord(fmt[index]))
                    index += 1
                    continue

                index += 1
                if index < len(fmt) and fmt[index] == "%":
                    out.append(ord("%"))
                    index += 1
                    continue

                flags = ""
                while index < len(fmt) and fmt[index] in "-+ #0":
                    flags += fmt[index]
                    index += 1

                if index < len(fmt) and fmt[index] == "*":
                    width = signed_value(next_arg(), 32)
                    if width < 0:
                        flags += "-"
                        width = -width
                    index += 1
                else:
                    width_start = index
                    while index < len(fmt) and fmt[index].isdigit():
                        index += 1
                    width = int(fmt[width_start:index] or "0")

                precision = None
                if index < len(fmt) and fmt[index] == ".":
                    index += 1
                    if index < len(fmt) and fmt[index] == "*":
                        raw_precision = signed_value(next_arg(), 32)
                        precision = None if raw_precision < 0 else raw_precision
                        index += 1
                    else:
                        precision_start = index
                        while index < len(fmt) and fmt[index].isdigit():
                            index += 1
                        precision = int(fmt[precision_start:index] or "0")

                length = ""
                if index < len(fmt) and fmt[index] in "hljzt":
                    length = fmt[index]
                    index += 1
                    if (
                        length in {"h", "l"}
                        and index < len(fmt)
                        and fmt[index] == length
                    ):
                        length += fmt[index]
                        index += 1

                if index >= len(fmt):
                    raise VMError("vsnprintf format ends after %")
                spec = fmt[index]
                index += 1
                bits = 64 if length in {"l", "ll", "j", "z", "t"} else 32
                prefix = ""
                numeric = False

                if spec in "di":
                    value = signed_value(next_arg(), bits)
                    sign = "-" if value < 0 else ("+" if "+" in flags else (" " if " " in flags else ""))
                    digits = str(abs(value))
                    if precision is not None:
                        digits = digits.rjust(precision, "0")
                    piece = sign + digits
                    numeric = True
                elif spec in "uoxX":
                    value = unsigned_value(next_arg(), bits)
                    if spec == "u":
                        digits = str(value)
                    elif spec == "o":
                        digits = format(value, "o")
                        if "#" in flags and (not digits.startswith("0")):
                            prefix = "0"
                    else:
                        digits = format(value, "x" if spec == "x" else "X")
                        if "#" in flags and value:
                            prefix = "0x" if spec == "x" else "0X"
                    if precision is not None:
                        digits = digits.rjust(precision, "0")
                    piece = prefix + digits
                    numeric = True
                elif spec == "p":
                    value = next_arg()
                    piece = "(nil)" if value == 0 else f"0x{value:x}"
                elif spec == "c":
                    piece = chr(next_arg() & 0xFF)
                elif spec == "s":
                    raw = read_cstring(next_arg())
                    if precision is not None:
                        raw = raw[:precision]
                    piece_bytes = raw
                    if width > len(piece_bytes):
                        padding = b" " * (width - len(piece_bytes))
                        piece_bytes = (
                            piece_bytes + padding
                            if "-" in flags
                            else padding + piece_bytes
                        )
                    out.extend(piece_bytes)
                    continue
                elif spec == "m":
                    piece_bytes = errno_message(vm)
                    if precision is not None:
                        piece_bytes = piece_bytes[:precision]
                    if width > len(piece_bytes):
                        padding = b" " * (width - len(piece_bytes))
                        piece_bytes = (
                            piece_bytes + padding
                            if "-" in flags
                            else padding + piece_bytes
                        )
                    out.extend(piece_bytes)
                    continue
                else:
                    raise VMError(
                        f"unsupported vsnprintf conversion %{length}{spec}"
                    )

                piece_bytes = piece.encode("latin1")
                if width > len(piece_bytes):
                    pad_len = width - len(piece_bytes)
                    if "-" in flags:
                        piece_bytes += b" " * pad_len
                    elif "0" in flags and precision is None and numeric:
                        lead = 0
                        if piece_bytes[:1] in {b"+", b"-", b" "}:
                            lead = 1
                        elif piece_bytes[:2] in {b"0x", b"0X"}:
                            lead = 2
                        piece_bytes = (
                            piece_bytes[:lead]
                            + b"0" * pad_len
                            + piece_bytes[lead:]
                        )
                    else:
                        piece_bytes = b" " * pad_len + piece_bytes
                out.extend(piece_bytes)

            payload = bytes(out)
            if size > 0:
                written = payload[: max(0, size - 1)]
                bulk_write = getattr(vm.memory, "bulk_write", None)
                if bulk_write is not None:
                    if written:
                        bulk_write(outbuf, written)
                else:
                    for offset, byte in enumerate(written):
                        vm.memory.write(outbuf + offset, 8, byte)
                vm.memory.write(outbuf + len(written), 8, 0)
            else:
                written = b""

            print(
                "BOOT_EXEC_USER_VSNPRINTF "
                f"fmt={fmt!r} result={len(payload)} written={len(written)} "
                f"ap=0x{ap:x}",
                flush=True,
            )
            return len(payload)

        return user_vsnprintf

    if original == "putenv":
        def user_putenv(vm, args):
            if len(args) != 1:
                raise VMError("putenv expects string")
            string_ptr = int(args[0])
            raw = read_user_cstring(vm, string_ptr, 1 << 16)
            if not raw:
                set_errno(vm, 22)
                return (1 << 64) - 1

            if b"=" in raw:
                name, _value = raw.split(b"=", 1)
                remove = False
            else:
                name = raw
                remove = True
            if not name or b"=" in name:
                set_errno(vm, 22)
                return (1 << 64) - 1

            environ_symbol = f"{external_prefix}environ"
            environ_addr = vm.program.symbol_addresses.get(environ_symbol)
            if environ_addr is None:
                set_errno(vm, 38)
                return (1 << 64) - 1

            envp = vm.memory.read(environ_addr, 64)
            entries = []
            match_index = None
            prefix = name + b"="
            if envp:
                for index in range(4096):
                    entry = vm.memory.read(envp + index * 8, 64)
                    if entry == 0:
                        break
                    entries.append(entry)
                    entry_raw = read_user_cstring(vm, entry, 1 << 16)
                    if match_index is None and entry_raw.startswith(prefix):
                        match_index = index
                else:
                    raise VMError("putenv environment vector is not terminated")

            if remove:
                if match_index is None:
                    return 0
                entries.pop(match_index)
            elif match_index is not None:
                entries[match_index] = string_ptr
            else:
                entries.append(string_ptr)

            new_envp = vm.alloc_bytes((len(entries) + 1) * 8, align=8)
            for index, entry in enumerate(entries):
                vm.memory.write(new_envp + index * 8, 64, entry)
            vm.memory.write(new_envp + len(entries) * 8, 64, 0)
            vm.memory.write(environ_addr, 64, new_envp)
            vm.user_envp = new_envp
            return 0

        return user_putenv

    if original == "getenv":
        def user_getenv(vm, args):
            if len(args) != 1:
                raise VMError("getenv expects name")
            name_ptr = int(args[0])
            name = read_user_cstring(vm, name_ptr, 4096)
            if not name or b"=" in name:
                return 0

            environ_symbol = f"{external_prefix}environ"
            environ_addr = vm.program.symbol_addresses.get(environ_symbol)
            if environ_addr is None:
                return 0
            envp = vm.memory.read(environ_addr, 64)
            if not envp:
                return 0

            prefix = name + b"="
            for index in range(4096):
                entry = vm.memory.read(envp + index * 8, 64)
                if entry == 0:
                    return 0
                raw = read_user_cstring(vm, entry, 1 << 16)
                if raw.startswith(prefix):
                    return entry + len(prefix)
            raise VMError("getenv environment vector is not terminated")

        return user_getenv

    if original == "time":
        def user_time(vm, args):
            if len(args) != 1:
                raise VMError("time expects time_t*")
            tloc = int(args[0])

            # Implement libc time(2) through Linux's real timekeeping path.
            # RISC-V/asm-generic has gettimeofday(2), whose tv_sec is the
            # required time_t value. Do not substitute host wall-clock time.
            timeval = vm.alloc_bytes(16, align=8)
            raw = user_syscall(
                vm,
                (169, timeval, 0, 0, 0, 0, 0),
            )
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                set_errno(vm, -signed)
                return (1 << 64) - 1

            seconds = vm.memory.read(timeval, 64)
            if tloc:
                vm.memory.write(tloc, 64, seconds)
            return seconds

        return user_time

    if original == "reboot":
        def user_reboot(vm, args):
            if len(args) != 1:
                raise VMError("reboot expects command")
            cmd = int(args[0])
            # glibc reboot(2) supplies the Linux magic values around the
            # single libc command argument. Keep the real permission and
            # reboot-command semantics in Linux instead of acknowledging the
            # request in the host runtime.
            result = user_syscall(
                vm,
                (
                    142,          # asm-generic __NR_reboot
                    0xFEE1DEAD,   # LINUX_REBOOT_MAGIC1
                    0x28121969,   # LINUX_REBOOT_MAGIC2
                    cmd,
                    0,
                    0,
                    0,
                ),
            )
            signed = result - (1 << 64) if result & (1 << 63) else result
            if signed < 0:
                set_errno(vm, -signed)
                return (1 << 64) - 1
            return result
        return user_reboot

    if original in {"exit", "_exit"}:
        def user_exit(vm, args):
            if len(args) != 1:
                raise VMError(f"{original} expects status")
            status = int(args[0]) & 0xFF
            vm.user_exit_status = status
            vm._user_exit_task = int(
                getattr(vm, "linux_current_task", 0) or 0
            )
            vm._user_exit_status = status
            print(
                "BOOT_EXEC_USER_EXIT_REQUEST "
                f"kind={original} status={status} "
                f"task=0x{vm._user_exit_task:x}",
                flush=True,
            )

            task_pids = getattr(vm, "user_task_pids", {})
            parent_pids = getattr(vm, "user_task_parent_pids", {})
            child_pid = int(task_pids.get(vm._user_exit_task, 0) or 0)
            parent_pid = int(parent_pids.get(vm._user_exit_task, 0) or 0)
            if child_pid:
                exits = getattr(vm, "user_wait_exits", None)
                if exits is None:
                    exits = []
                    vm.user_wait_exits = exits
                if not any(int(item[3]) == vm._user_exit_task for item in exits):
                    exits.append((child_pid, parent_pid, status, vm._user_exit_task))
                    print(
                        "BOOT_EXEC_USER_EXIT_TRACKED "
                        f"task=0x{vm._user_exit_task:x} "
                        f"pid={child_pid} ppid={parent_pid} status={status}",
                        flush=True,
                    )

            raw = user_syscall(
                vm,
                (93, status, 0, 0, 0, 0, 0),
            )
            if raw is HOST_CONTROL_TRANSFER:
                return HOST_CONTROL_TRANSFER

            # Linux do_exit() is noreturn. If the semantic syscall returns,
            # clear the arm and report the unexpected result rather than
            # silently halting the whole VM.
            vm._user_exit_task = 0
            vm._user_exit_status = 0
            signed = raw - (1 << 64) if raw & (1 << 63) else raw
            if signed < 0:
                set_errno(vm, -signed)
                return (1 << 64) - 1
            raise VMError(
                f"Linux exit syscall unexpectedly returned {signed}"
            )

        return user_exit

    if original == "execvp":
        def user_execvp(vm, args):
            if len(args) != 2:
                raise VMError("execvp expects file,argv")
            file_ptr, argv_ptr = map(int, args)
            file_name = read_user_cstring(vm, file_ptr, 4096)
            if not file_name:
                set_errno(vm, 2)
                return (1 << 64) - 1

            environ_symbol = f"{external_prefix}environ"
            environ_addr = vm.program.symbol_addresses.get(environ_symbol)
            envp = (
                vm.memory.read(environ_addr, 64)
                if environ_addr is not None
                else int(getattr(vm, "user_envp", 0) or 0)
            )

            def try_exec(path_ptr: int):
                raw = user_syscall(
                    vm,
                    (221, path_ptr, argv_ptr, envp, 0, 0, 0),
                )
                if raw is HOST_CONTROL_TRANSFER:
                    return HOST_CONTROL_TRANSFER, 0
                signed = raw - (1 << 64) if raw & (1 << 63) else raw
                return raw, -signed if -4095 <= signed < 0 else 0

            if b"/" in file_name:
                raw, error = try_exec(file_ptr)
                if raw is HOST_CONTROL_TRANSFER:
                    return HOST_CONTROL_TRANSFER
                if error:
                    set_errno(vm, error)
                    return (1 << 64) - 1
                return raw

            path_value = None
            if envp:
                for index in range(4096):
                    entry = vm.memory.read(envp + index * 8, 64)
                    if entry == 0:
                        break
                    raw_entry = read_user_cstring(vm, entry, 1 << 16)
                    if raw_entry.startswith(b"PATH="):
                        path_value = raw_entry[5:]
                        break
                else:
                    raise VMError("execvp environment vector is not terminated")
            if path_value is None:
                path_value = b"/bin:/usr/bin"

            saw_eacces = False
            for directory in path_value.split(b":"):
                candidate = (
                    file_name
                    if not directory
                    else directory.rstrip(b"/") + b"/" + file_name
                )
                candidate_ptr = vm.alloc_bytes(len(candidate) + 1, align=1)
                bulk_write = getattr(vm.memory, "bulk_write", None)
                if bulk_write is not None and candidate:
                    bulk_write(candidate_ptr, candidate)
                else:
                    for offset, byte in enumerate(candidate):
                        vm.memory.write(candidate_ptr + offset, 8, byte)
                vm.memory.write(candidate_ptr + len(candidate), 8, 0)

                raw, error = try_exec(candidate_ptr)
                if raw is HOST_CONTROL_TRANSFER:
                    return HOST_CONTROL_TRANSFER
                if error in {2, 20}:  # ENOENT / ENOTDIR
                    continue
                if error == 13:  # EACCES: keep searching, but remember it.
                    saw_eacces = True
                    continue
                if error:
                    set_errno(vm, error)
                    return (1 << 64) - 1
                return raw

            set_errno(vm, 13 if saw_eacces else 2)
            return (1 << 64) - 1

        return user_execvp

    if original == "waitpid":
        def user_waitpid(vm, args):
            if len(args) != 3:
                raise VMError("waitpid expects pid,status,options")
            pid, status_ptr, options = map(int, args)

            # A CLONE_VM child shares the concrete P3 userspace stack addresses
            # with its parent. Snapshot the waiter *before* any scheduler drive:
            # the child may run and reuse this very ext_waitpid frame before we
            # ever reach the real wait4 bridge.
            waiter_stack = _snapshot_p3_call_chain(vm)

            # A normal CLONE_VM fork returns to the parent before the child
            # necessarily wins a scheduler slot. Keep the real Linux scheduler
            # in charge, but drive bounded schedule() rounds until service 2
            # consumes the pending P3 child continuation.
            pending = getattr(vm, "pending_user_fork_continuation", None)
            if pending is not None and "schedule" in vm.program.functions:
                for round_no in range(1, 9):
                    before = getattr(vm, "pending_user_fork_continuation", None)
                    if before is None:
                        break
                    scheduler_owner = int(
                        getattr(vm, "linux_current_task", 0)
                        or 0
                    )
                    scheduled = _call_linux_function_preserving_control(
                        vm,
                        "schedule",
                        (),
                        result_count=0,
                        max_extra_steps=12_000_000,
                        preserve_linux_task_state=True,
                        # This is an internal kernel scheduler drive, not a
                        # syscall owned by the active userspace continuation.
                        # Bind its semantic stack to the Linux task that is
                        # actually executing schedule() so context-switch
                        # frames from different tasks cannot alias.
                        call_task_override=(
                            scheduler_owner if scheduler_owner else None
                        ),
                    )
                    print(
                        "BOOT_EXEC_USER_WAIT_SCHEDULE "
                        f"round={round_no} "
                        f"pending_before={1 if before is not None else 0} "
                        f"pending_after={1 if getattr(vm, 'pending_user_fork_continuation', None) is not None else 0} "
                        f"transfer={1 if scheduled is HOST_CONTROL_TRANSFER else 0}",
                        flush=True,
                    )
                    if scheduled is HOST_CONTROL_TRANSFER:
                        return HOST_CONTROL_TRANSFER

            # The scheduler drive above may already have let the child reuse
            # the parent P3 stack. Restore the original waiter frames before
            # entering wait4, then restore the same snapshot once more after
            # wait4 returns because Linux may schedule the child again there.
            _restore_p3_call_chain(vm, waiter_stack)
            if waiter_stack:
                print(
                    "BOOT_EXEC_USER_WAIT_PRE_RESTORE "
                    f"frames={len(waiter_stack)} "
                    f"bytes={sum(len(payload) for _, payload in waiter_stack)} "
                    f"pid={pid}",
                    flush=True,
                )

            raw = user_syscall(
                vm,
                (
                    260,  # wait4
                    pid,
                    status_ptr,
                    options,
                    0,  # struct rusage *
                    0,
                    0,
                ),
            )
            if raw is HOST_CONTROL_TRANSFER:
                return HOST_CONTROL_TRANSFER
            _restore_p3_call_chain(vm, waiter_stack)
            if waiter_stack:
                print(
                    "BOOT_EXEC_USER_WAIT_STACK_RESTORE "
                    f"frames={len(waiter_stack)} "
                    f"bytes={sum(len(payload) for _, payload in waiter_stack)} "
                    f"pid={pid}",
                    flush=True,
                )

            signed_raw = raw - (1 << 64) if raw & (1 << 63) else raw
            requested_pid = pid - (1 << 64) if pid & (1 << 63) else pid
            if signed_raw in {-38, -10}:
                parent_task = int(
                    getattr(vm, "active_user_task", 0)
                    or getattr(vm, "linux_current_task", 0)
                    or 0
                )
                parent_pid = int(
                    getattr(vm, "user_task_pids", {}).get(parent_task, 0)
                    or 0
                )
                exits = getattr(vm, "user_wait_exits", None)
                if exits:
                    match_index = None
                    for index, item in enumerate(exits):
                        child_pid, child_parent_pid, child_status, child_task = item
                        if parent_pid and child_parent_pid and child_parent_pid != parent_pid:
                            continue
                        if requested_pid > 0 and child_pid != requested_pid:
                            continue
                        match_index = index
                        break
                    if match_index is not None:
                        child_pid, child_parent_pid, child_status, child_task = exits.pop(match_index)
                        if status_ptr:
                            vm.memory.write(
                                status_ptr,
                                32,
                                (int(child_status) & 0xFF) << 8,
                            )
                        raw = int(child_pid)
                        print(
                            "BOOT_EXEC_USER_WAIT_EXIT_REPLAY "
                            f"parent_task=0x{parent_task:x} "
                            f"parent_pid={parent_pid} "
                            f"child_task=0x{int(child_task):x} "
                            f"child_pid={int(child_pid)} "
                            f"status={int(child_status)} "
                            f"kernel_result={signed_raw}",
                            flush=True,
                        )
            return libc_linux_result(vm, raw)

        return user_waitpid

    if original in {"fork", "vfork"}:
        def user_fork(vm, args):
            if args:
                raise VMError(f"{original} expects no arguments")
            if "__se_sys_clone" not in vm.program.functions:
                raise VMError("Linux image is missing __se_sys_clone")
            if getattr(vm, "pending_user_fork_continuation", None) is not None:
                raise VMError("nested MiniMachine userspace fork is not supported")

            expected = vm.memory.read(vm.sp + RESULT_COUNT, 64)
            if expected != 1:
                raise VMError(
                    f"{original} caller expects {expected} results, expected 1"
                )
            continuation = (
                vm.memory.read(vm.sp + CALLER_SP, 64),
                vm.memory.read(vm.sp + RET_PC, 64),
                vm.memory.read(vm.sp + RESULT_PTR, 64),
            )
            vm.pending_user_fork_continuation = continuation

            # MiniMachine Linux is NOMMU, so both forms share the Linux
            # address space. Keep the semantic distinction between fork and
            # vfork, though: ordinary fork creates a separate SIGCHLD task and
            # lets the parent block naturally in wait4; only vfork holds the
            # parent inside CLONE_VFORK until child exec/exit. P3 userspace
            # stacks/continuations are already task-scoped above this layer.
            clone_vm = 0x00000100
            clone_vfork = 0x00004000
            sigchld = 17
            flags = clone_vm | sigchld
            if original == "vfork":
                flags |= clone_vfork
            restart_errors = {-512, -513, -514, -516}
            attempt = 0
            while True:
                parent_stack = _snapshot_p3_call_chain(vm)
                try:
                    call_result = _call_linux_function_preserving_control(
                        vm,
                        "__se_sys_clone",
                        (flags, 0, 0, 0, 0),
                        result_count=1,
                        max_extra_steps=180_000_000,
                        preserve_linux_task_state=True,
                    )
                except Exception:
                    if getattr(vm, "pending_user_fork_continuation", None) is continuation:
                        vm.pending_user_fork_continuation = None
                    raise

                if call_result is HOST_CONTROL_TRANSFER:
                    # The first-run child path consumes the continuation in
                    # service 2. Keep it armed across the control transfer.
                    return HOST_CONTROL_TRANSFER

                _restore_p3_call_chain(vm, parent_stack)
                print(
                    "BOOT_EXEC_USER_FORK_STACK_RESTORE "
                    f"frames={len(parent_stack)} "
                    f"bytes={sum(len(payload) for _, payload in parent_stack)}",
                    flush=True,
                )
                result, = call_result
                signed_result = (
                    result - (1 << 64)
                    if result & (1 << 63)
                    else result
                )

                # Directly invoking __se_sys_clone bypasses Linux's normal
                # syscall-exit restart handling. Internal ERESTART* values
                # must never escape to BusyBox. Retry only while the child
                # continuation is still unconsumed, which means no first-run
                # child has taken ownership of it yet.
                if (
                    signed_result in restart_errors
                    and getattr(vm, "pending_user_fork_continuation", None) is continuation
                    and attempt < 3
                ):
                    attempt += 1
                    print(
                        "BOOT_EXEC_USER_FORK_RESTART "
                        f"attempt={attempt} result={signed_result}",
                        flush=True,
                    )
                    continue

                if (
                    signed_result < 0
                    and getattr(vm, "pending_user_fork_continuation", None) is continuation
                ):
                    # A real clone failure cannot produce a child first-run.
                    vm.pending_user_fork_continuation = None
                break

            print(
                "BOOT_EXEC_USER_FORK_CLONE "
                f"kind={original} flags=0x{flags:x} result={result} "
                f"pending={1 if getattr(vm, 'pending_user_fork_continuation', None) is continuation else 0}",
                flush=True,
            )
            return libc_linux_result(vm, result)

        return user_fork

    syscall_map = {
        "read": (63, 3),
        "write": (64, 3),
        "chdir": (49, 1),
        "close": (57, 1),
        "gettimeofday": (169, 2),
        "uname": (160, 1),
        "umask": (166, 1),
        "times": (153, 1),
        "getpid": (172, 0),
        "getppid": (173, 0),
        "getuid": (174, 0),
        "geteuid": (175, 0),
        "getgid": (176, 0),
        "getegid": (177, 0),
        "execve": (221, 3),
    }
    syscall_spec = syscall_map.get(original)
    if syscall_spec is not None:
        nr, argc = syscall_spec

        def libc_syscall(vm, args):
            if len(args) != argc:
                raise VMError(
                    f"{original} expects {argc} arguments, got {len(args)}"
                )
            padded = tuple(args) + (0,) * (6 - len(args))
            raw = user_syscall(vm, (nr, *padded))
            if raw is HOST_CONTROL_TRANSFER:
                return HOST_CONTROL_TRANSFER
            return libc_linux_result(vm, raw)
        return libc_syscall

    def unimplemented(_vm, args):
        preview = ",".join(f"0x{x:x}" for x in args[:8])
        print(
            "BOOT_EXEC_USER_EXTERNAL "
            f"name={original} argc={len(args)} args={preview}",
            flush=True,
        )
        raise VMError(
            f"unimplemented userspace external {original}"
        )

    return unimplemented
