from __future__ import annotations

import unittest

from src.minimachine.stdio_surface import resolve_stdio_line_callback
from src.minimachine.vm import Program, VMError


class FakeSyscalls:
    def __init__(self, payload: bytes = b""):
        self.input = bytearray(payload)
        self.writes: list[tuple[int, bytes]] = []
        self.read_calls = 0

    def __call__(self, vm, args):
        nr, fd, ptr, count, *_ = map(int, args)
        if nr == 64:
            bulk_read = getattr(vm.memory, "bulk_read", None)
            if bulk_read is not None:
                payload = bytes(bulk_read(ptr, count))
            else:
                payload = bytes(vm.memory.read(ptr + i, 8) for i in range(count))
            self.writes.append((fd, payload))
            return count
        if nr == 63:
            self.read_calls += 1
            amount = min(count, len(self.input))
            payload = bytes(self.input[:amount])
            del self.input[:amount]
            bulk_write = getattr(vm.memory, "bulk_write", None)
            if bulk_write is not None:
                bulk_write(ptr, payload)
            else:
                for i, byte in enumerate(payload):
                    vm.memory.write(ptr + i, 8, byte)
            return amount
        raise AssertionError(f"unexpected syscall {nr}")


def alloc_bytes(vm, payload: bytes) -> int:
    ptr = vm.alloc_bytes(len(payload), align=1)
    bulk_write = getattr(vm.memory, "bulk_write", None)
    if bulk_write is not None:
        bulk_write(ptr, payload)
    else:
        for i, byte in enumerate(payload):
            vm.memory.write(ptr + i, 8, byte)
    return ptr


def read_cstring(vm, ptr: int) -> bytes:
    out = bytearray()
    while True:
        byte = vm.memory.read(ptr + len(out), 8)
        if byte == 0:
            return bytes(out)
        out.append(byte)


def callback(name: str, syscalls: FakeSyscalls, errno_address: int | None = None):
    result = resolve_stdio_line_callback(
        name,
        errno_address=errno_address,
        vm_error=VMError,
        user_syscall=syscalls,
    )
    assert result is not None
    return result


class StdioSurfaceTests(unittest.TestCase):
    def test_fputs_writes_exact_string_without_newline(self):
        vm = Program().new_vm()
        syscalls = FakeSyscalls()
        text = alloc_bytes(vm, b"sqlite> \0")
        result = callback("fputs", syscalls)(vm, (text, 1))
        self.assertEqual(result, 0)
        self.assertEqual(syscalls.writes, [(1, b"sqlite> ")])

    def test_fputs_uses_guest_file_handle(self):
        vm = Program().new_vm()
        vm.user_file_streams = {
            0x1230: {"fd": 7, "eof": False, "error": False, "ungetc": None}
        }
        syscalls = FakeSyscalls()
        text = alloc_bytes(vm, b"hello\0")
        self.assertEqual(callback("fputs", syscalls)(vm, (text, 0x1230)), 0)
        self.assertEqual(syscalls.writes, [(7, b"hello")])

    def test_fgets_stops_at_newline_and_buffers_following_bytes(self):
        vm = Program().new_vm()
        syscalls = FakeSyscalls(b"alpha\nbeta\n")
        first = vm.alloc_bytes(32, align=1)
        second = vm.alloc_bytes(32, align=1)
        fgets = callback("fgets", syscalls)
        self.assertEqual(fgets(vm, (first, 32, 0)), first)
        self.assertEqual(read_cstring(vm, first), b"alpha\n")
        first_reads = syscalls.read_calls
        self.assertEqual(fgets(vm, (second, 32, 0)), second)
        self.assertEqual(read_cstring(vm, second), b"beta\n")
        self.assertEqual(syscalls.read_calls, first_reads)

    def test_fgets_eof_after_data_returns_partial_line(self):
        vm = Program().new_vm()
        syscalls = FakeSyscalls(b"tail")
        dst = vm.alloc_bytes(16, align=1)
        fgets = callback("fgets", syscalls)
        self.assertEqual(fgets(vm, (dst, 16, 0)), dst)
        self.assertEqual(read_cstring(vm, dst), b"tail")
        other = vm.alloc_bytes(16, align=1)
        self.assertEqual(fgets(vm, (other, 16, 0)), 0)

    def test_bad_stream_sets_ebadf(self):
        vm = Program().new_vm()
        errno_address = vm.alloc_bytes(4, align=4)
        syscalls = FakeSyscalls()
        text = alloc_bytes(vm, b"x\0")
        self.assertEqual(
            callback("fputs", syscalls, errno_address)(vm, (text, 0xDEAD)),
            (1 << 64) - 1,
        )
        self.assertEqual(vm.memory.read(errno_address, 32), 9)
        dst = vm.alloc_bytes(8, align=1)
        self.assertEqual(
            callback("fgets", syscalls, errno_address)(vm, (dst, 8, 0xDEAD)),
            0,
        )
        self.assertEqual(vm.memory.read(errno_address, 32), 9)

    def test_unrelated_stdio_name_is_not_hidden(self):
        self.assertIsNone(
            resolve_stdio_line_callback(
                "popen",
                errno_address=0x1000,
                vm_error=VMError,
                user_syscall=FakeSyscalls(),
            )
        )


if __name__ == "__main__":
    unittest.main()
