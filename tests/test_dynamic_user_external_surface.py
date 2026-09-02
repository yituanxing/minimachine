from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys
import struct
import unittest

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.lower_p3 import lower_function
from src.minimachine.native_vm import NativeVM
from src.minimachine.vm import Program


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run-minimachine-linux.py"


def load_runner():
    spec = spec_from_file_location("run_minimachine_linux_dynamic_test", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DynamicUserExternalSurfaceTests(unittest.TestCase):
    def test_runtime_registered_external_descriptor_is_live_immediately(self):
        runner = load_runner()
        program = Program()
        vm = program.new_vm()

        image = SimpleNamespace(
            external_functions=("__mm_user_ext_malloc",),
            external_data=("__mm_user_ext_environ",),
        )
        user_image = SimpleNamespace(image=image)

        runner.install_user_external_surface(vm, user_image, 0x12345678)

        descriptor = program.symbol_addresses["__mm_user_ext_malloc"]
        initial_entry = program.initial_memory.read(descriptor, 64)
        initial_frame = program.initial_memory.read(descriptor + 8, 64)
        self.assertNotEqual(initial_entry, 0)
        self.assertEqual(vm.memory.read(descriptor, 64), initial_entry)
        self.assertEqual(vm.memory.read(descriptor + 8, 64), initial_frame)

        environ = program.symbol_addresses["__mm_user_ext_environ"]
        self.assertEqual(vm.memory.read(environ, 64), 0x12345678)


    def test_linux_context_switch_accepts_first_run_user_task(self):
        runner = load_runner()

        def lowered(name):
            fn = muir.Function(
                name,
                [muir.Block("entry", [muir.Ret(None)])],
                set(),
            )
            expanded, _ = expand_function(fn)
            return lower_function(expanded)

        program = Program(
            (
                lowered("minimachine_ret_from_fork"),
                lowered("resume_after_fork"),
            )
        )
        vm = program.new_vm()
        vm.linux_shadow_stack_next = 0xF0000000

        resume_code = program.block_code[("resume_after_fork", "entry")]
        vm.pending_user_fork_continuation = (
            0xAB1000,
            resume_code,
            0xAB2000,
        )

        vm.memory.write(vm.sp + runner.RESULT_COUNT, 64, 1)
        vm.memory.write(vm.sp + runner.CALLER_SP, 64, 0xABC000)
        vm.memory.write(vm.sp + runner.RET_PC, 64, program.halt_code)
        vm.memory.write(vm.sp + runner.RESULT_PTR, 64, 0xABD000)

        result = runner.linux_ecall(
            vm,
            (2, 0x1000, 0x2000, 0x3000, 0, 0),
        )
        self.assertIs(result, runner.HOST_CONTROL_TRANSFER)
        self.assertEqual(vm.linux_current_task, 0x2000)
        self.assertEqual(vm.current_function, "minimachine_ret_from_fork")
        self.assertEqual(vm.linux_task_shadow_stacks[0x2000], 0xF0000000)
        self.assertIsNone(
            getattr(vm, "pending_user_fork_continuation", None)
        )
        self.assertIn(0x2000, vm.linux_user_fork_continuations)

        result = runner.linux_ecall(vm, (3, 0x3000))
        self.assertIs(result, runner.HOST_CONTROL_TRANSFER)
        self.assertEqual(vm.sp, 0xAB1000)
        self.assertEqual(vm.current_function, "resume_after_fork")
        self.assertEqual(vm.memory.read(0xAB2000, 64), 0)
        self.assertNotIn(0x2000, vm.linux_user_fork_continuations)

    def test_fork_adapts_to_nommu_vfork_clone_flags(self):
        runner = load_runner()
        fn = muir.Function(
            "__se_sys_clone",
            [muir.Block("entry", [muir.Ret(muir.Imm(0))])],
            set(),
        )
        expanded, _ = expand_function(fn)
        program = Program((lower_function(expanded),))
        vm = program.new_vm()

        vm.memory.write(vm.sp + runner.RESULT_COUNT, 64, 1)
        vm.memory.write(vm.sp + runner.CALLER_SP, 64, 0xAC1000)
        vm.memory.write(vm.sp + runner.RET_PC, 64, program.halt_code)
        vm.memory.write(vm.sp + runner.RESULT_PTR, 64, 0xAC2000)

        seen = {}

        def fake_call(vm_arg, name, args, **kwargs):
            self.assertIs(vm_arg, vm)
            seen["name"] = name
            seen["args"] = args
            seen["kwargs"] = kwargs
            return (321,)

        runner._call_linux_function_preserving_control = fake_call
        for original in ("fork", "vfork"):
            with self.subTest(original=original):
                callback = runner._user_libc_callback(
                    f"__mm_user_ext_{original}",
                    None,
                )
                self.assertIsNotNone(callback)
                assert callback is not None
                self.assertEqual(callback(vm, ()), 321)
                self.assertEqual(seen["name"], "__se_sys_clone")
                self.assertEqual(seen["args"], (0x4111, 0, 0, 0, 0))
                self.assertTrue(seen["kwargs"]["preserve_linux_task_state"])
                self.assertIsNone(
                    getattr(vm, "pending_user_fork_continuation", None)
                )

    def test_setsid_syscall_falls_back_to_linux_sys_setsid(self):
        runner = load_runner()
        fn = muir.Function(
            "sys_setsid",
            [muir.Block("entry", [muir.Ret(muir.Imm(0))])],
            set(),
        )
        expanded, _ = expand_function(fn)
        program = Program((lower_function(expanded),))
        vm = program.new_vm()
        seen = {}

        def fake_call(vm_arg, name, args, **kwargs):
            self.assertIs(vm_arg, vm)
            seen["name"] = name
            seen["args"] = args
            return (77,)

        runner._call_linux_function_preserving_control = fake_call
        self.assertEqual(
            runner.user_syscall(vm, (157, 0, 0, 0, 0, 0, 0)),
            77,
        )
        self.assertEqual(seen["name"], "sys_setsid")
        self.assertEqual(seen["args"], ())

    def test_time_libc_wrapper_uses_linux_gettimeofday(self):
        runner = load_runner()
        program = Program()
        vm = program.new_vm()
        seen = {}

        def fake_user_syscall(vm_arg, args):
            self.assertIs(vm_arg, vm)
            seen["args"] = args
            timeval = args[1]
            vm.memory.write(timeval, 64, 123456789)
            vm.memory.write(timeval + 8, 64, 654321)
            return 0

        runner.user_syscall = fake_user_syscall
        callback = runner._user_libc_callback("__mm_lua54_ext_time", None)
        self.assertIsNotNone(callback)
        assert callback is not None

        tloc = 0xD200
        self.assertEqual(callback(vm, (tloc,)), 123456789)
        self.assertEqual(vm.memory.read(tloc, 64), 123456789)
        self.assertEqual(
            seen["args"],
            (169, seen["args"][1], 0, 0, 0, 0, 0),
        )

    def test_snprintf_formats_double_and_reports_untruncated_length(self):
        runner = load_runner()
        program = Program()
        vm = program.new_vm()

        fmt = 0xD400
        buf = 0xD480
        for offset, byte in enumerate(b"%.14g\0"):
            vm.memory.write(fmt + offset, 8, byte)

        callback = runner._user_libc_callback(
            "__mm_lua54_ext_snprintf",
            None,
        )
        self.assertIsNotNone(callback)
        assert callback is not None

        raw_double = int.from_bytes(
            struct.pack("<d", 1234.5),
            "little",
        )
        full = b"1234.5"
        self.assertEqual(
            callback(vm, (buf, 32, fmt, raw_double)),
            len(full),
        )
        self.assertEqual(
            bytes(vm.memory.read(buf + i, 8) for i in range(len(full) + 1)),
            full + b"\0",
        )

        self.assertEqual(
            callback(vm, (buf, 5, fmt, raw_double)),
            len(full),
        )
        self.assertEqual(
            bytes(vm.memory.read(buf + i, 8) for i in range(5)),
            b"1234\0",
        )

    def test_getline_grows_guest_buffer_and_reaches_eof(self):
        runner = load_runner()
        program = Program()
        vm = program.new_vm()

        path_ptr = 0xD800
        mode_ptr = 0xD880
        lineptr_ptr = 0xD900
        capacity_ptr = 0xD908
        for offset, byte in enumerate(b"/etc/test\0"):
            vm.memory.write(path_ptr + offset, 8, byte)
        for offset, byte in enumerate(b"r\0"):
            vm.memory.write(mode_ptr + offset, 8, byte)
        vm.memory.write(lineptr_ptr, 64, 0)
        vm.memory.write(capacity_ptr, 64, 0)

        content = b"first line\nsecond\n"
        position = 0

        def fake_user_syscall(vm_arg, args):
            nonlocal position
            self.assertIs(vm_arg, vm)
            nr = int(args[0])
            if nr == 56:
                return 7
            if nr == 63:
                ptr = int(args[2])
                count = int(args[3])
                payload = content[position : position + count]
                for index, byte in enumerate(payload):
                    vm.memory.write(ptr + index, 8, byte)
                position += len(payload)
                return len(payload)
            if nr == 57:
                return 0
            self.fail(f"unexpected syscall {args}")

        runner.user_syscall = fake_user_syscall
        fopen = runner._user_libc_callback("__mm_user_ext_fopen64", None)
        getline = runner._user_libc_callback("__mm_user_ext_getline", None)
        fclose = runner._user_libc_callback("__mm_user_ext_fclose", None)
        self.assertIsNotNone(fopen)
        self.assertIsNotNone(getline)
        self.assertIsNotNone(fclose)
        assert fopen is not None and getline is not None and fclose is not None

        stream = fopen(vm, (path_ptr, mode_ptr))
        self.assertNotEqual(stream, 0)

        first_len = getline(vm, (lineptr_ptr, capacity_ptr, stream))
        self.assertEqual(first_len, len(b"first line\n"))
        line = vm.memory.read(lineptr_ptr, 64)
        capacity = vm.memory.read(capacity_ptr, 64)
        self.assertGreaterEqual(capacity, first_len + 1)
        self.assertEqual(
            bytes(vm.memory.read(line + i, 8) for i in range(first_len + 1)),
            b"first line\n\0",
        )

        second_len = getline(vm, (lineptr_ptr, capacity_ptr, stream))
        self.assertEqual(second_len, len(b"second\n"))
        self.assertEqual(
            bytes(vm.memory.read(line + i, 8) for i in range(second_len + 1)),
            b"second\n\0",
        )
        self.assertEqual(
            getline(vm, (lineptr_ptr, capacity_ptr, stream)),
            (1 << 64) - 1,
        )
        self.assertEqual(fclose(vm, (stream,)), 0)

    def test_fwrite_and_setsid_use_linux_syscalls(self):
        runner = load_runner()
        program = Program()
        vm = program.new_vm()
        calls = []

        def fake_user_syscall(vm_arg, args):
            self.assertIs(vm_arg, vm)
            calls.append(args)
            if args[0] == 64:
                return int(args[3])
            if args[0] == 157:
                return 1
            self.fail(f"unexpected syscall {args}")

        runner.user_syscall = fake_user_syscall
        fwrite = runner._user_libc_callback("__mm_lua54_ext_fwrite", None)
        setsid = runner._user_libc_callback("__mm_user_ext_setsid", None)
        self.assertIsNotNone(fwrite)
        self.assertIsNotNone(setsid)
        assert fwrite is not None and setsid is not None

        ptr = 0xD580
        for index, byte in enumerate(b"hello"):
            vm.memory.write(ptr + index, 8, byte)

        self.assertEqual(fwrite(vm, (ptr, 1, 5, 1)), 5)
        self.assertEqual(setsid(vm, ()), 1)
        self.assertEqual(
            calls,
            [
                (64, 1, ptr, 5, 0, 0, 0),
                (157, 0, 0, 0, 0, 0, 0),
            ],
        )

    def test_guest_stdio_streams_share_linux_fd_state(self):
        runner = load_runner()
        program = Program()
        vm = program.new_vm()

        def put_cstring(address: int, payload: bytes) -> None:
            for offset, byte in enumerate(payload + b"\0"):
                vm.memory.write(address + offset, 8, byte)

        path_ptr = 0xD600
        mode_ptr = 0xD680
        data_ptr = 0xD700
        put_cstring(path_ptr, b"/tmp/script.lua")
        put_cstring(mode_ptr, b"r")

        content = b"abc"
        position = 0
        closed = []
        calls = []

        def fake_user_syscall(vm_arg, args):
            nonlocal position
            self.assertIs(vm_arg, vm)
            calls.append(args)
            nr = args[0]
            if nr == 56:
                self.assertEqual(args[1], (-100) & ((1 << 64) - 1))
                self.assertEqual(args[2], path_ptr)
                self.assertEqual(args[3], 0)
                return 5
            if nr == 63:
                fd, ptr, count = map(int, args[1:4])
                self.assertEqual(fd, 5)
                payload = content[position : position + count]
                for index, byte in enumerate(payload):
                    vm.memory.write(ptr + index, 8, byte)
                position += len(payload)
                return len(payload)
            if nr == 62:
                fd, offset, whence = map(int, args[1:4])
                self.assertEqual(fd, 5)
                signed_offset = (
                    offset - (1 << 64)
                    if offset & (1 << 63)
                    else offset
                )
                if whence == 0:
                    position = signed_offset
                elif whence == 1:
                    position += signed_offset
                elif whence == 2:
                    position = len(content) + signed_offset
                else:
                    return (-22) & ((1 << 64) - 1)
                return position
            if nr == 57:
                closed.append(int(args[1]))
                return 0
            self.fail(f"unexpected syscall {args}")

        runner.user_syscall = fake_user_syscall

        fopen = runner._user_libc_callback("__mm_lua54_ext_fopen64", None)
        getc = runner._user_libc_callback("__mm_lua54_ext_getc", None)
        ungetc = runner._user_libc_callback("__mm_lua54_ext_ungetc", None)
        fread = runner._user_libc_callback("__mm_lua54_ext_fread", None)
        feof = runner._user_libc_callback("__mm_lua54_ext_feof", None)
        clearerr = runner._user_libc_callback("__mm_lua54_ext_clearerr", None)
        ftello = runner._user_libc_callback("__mm_lua54_ext_ftello64", None)
        fseeko = runner._user_libc_callback("__mm_lua54_ext_fseeko64", None)
        setvbuf = runner._user_libc_callback("__mm_lua54_ext_setvbuf", None)
        fclose = runner._user_libc_callback("__mm_lua54_ext_fclose", None)
        callbacks = (
            fopen,
            getc,
            ungetc,
            fread,
            feof,
            clearerr,
            ftello,
            fseeko,
            setvbuf,
            fclose,
        )
        self.assertTrue(all(callback is not None for callback in callbacks))
        assert all(callback is not None for callback in callbacks)

        handle = fopen(vm, (path_ptr, mode_ptr))
        self.assertNotEqual(handle, 0)
        self.assertEqual(getc(vm, (handle,)), ord("a"))
        self.assertEqual(ungetc(vm, (ord("a"), handle)), ord("a"))
        self.assertEqual(getc(vm, (handle,)), ord("a"))

        self.assertEqual(fread(vm, (data_ptr, 1, 2, handle)), 2)
        self.assertEqual(
            bytes(vm.memory.read(data_ptr + i, 8) for i in range(2)),
            b"bc",
        )
        self.assertEqual(ftello(vm, (handle,)), 3)

        self.assertEqual(fseeko(vm, (handle, 0, 0)), 0)
        self.assertEqual(setvbuf(vm, (handle, 0, 2, 0)), 0)
        self.assertEqual(fread(vm, (data_ptr, 1, 4, handle)), 3)
        self.assertEqual(
            bytes(vm.memory.read(data_ptr + i, 8) for i in range(3)),
            b"abc",
        )
        self.assertEqual(feof(vm, (handle,)), 1)
        clearerr(vm, (handle,))
        self.assertEqual(feof(vm, (handle,)), 0)

        self.assertEqual(fclose(vm, (handle,)), 0)
        self.assertEqual(closed, [5])

    def test_termios_libc_wrappers_use_linux_ioctl(self):
        runner = load_runner()
        program = Program()
        vm = program.new_vm()
        calls = []

        def fake_user_syscall(vm_arg, args):
            self.assertIs(vm_arg, vm)
            calls.append(args)
            return 0

        runner.user_syscall = fake_user_syscall
        tcgetattr = runner._user_libc_callback(
            "__mm_user_ext_tcgetattr",
            None,
        )
        tcsetattr = runner._user_libc_callback(
            "__mm_user_ext_tcsetattr",
            None,
        )
        self.assertIsNotNone(tcgetattr)
        self.assertIsNotNone(tcsetattr)
        assert tcgetattr is not None and tcsetattr is not None

        self.assertEqual(tcgetattr(vm, (0, 0xD500)), 0)
        self.assertEqual(tcsetattr(vm, (0, 0, 0xD500)), 0)
        self.assertEqual(tcsetattr(vm, (0, 1, 0xD500)), 0)
        self.assertEqual(tcsetattr(vm, (0, 2, 0xD500)), 0)
        self.assertEqual(
            calls,
            [
                (29, 0, 0x5401, 0xD500, 0, 0, 0),
                (29, 0, 0x5402, 0xD500, 0, 0, 0),
                (29, 0, 0x5403, 0xD500, 0, 0, 0),
                (29, 0, 0x5404, 0xD500, 0, 0, 0),
            ],
        )

    def test_ioctl_libc_wrapper_uses_linux_syscall(self):
        runner = load_runner()
        program = Program()
        vm = program.new_vm()
        seen = {}

        def fake_user_syscall(vm_arg, args):
            self.assertIs(vm_arg, vm)
            seen["args"] = args
            return 7

        runner.user_syscall = fake_user_syscall
        callback = runner._user_libc_callback("__mm_user_ext_ioctl", None)
        self.assertIsNotNone(callback)
        assert callback is not None

        self.assertEqual(callback(vm, (0, 0x5600, 0xD400)), 7)
        self.assertEqual(
            seen["args"],
            (29, 0, 0x5600, 0xD400, 0, 0, 0),
        )

    def test_reboot_libc_wrapper_uses_linux_magic_syscall(self):
        runner = load_runner()
        program = Program()
        vm = program.new_vm()
        seen = {}

        def fake_user_syscall(vm_arg, args):
            self.assertIs(vm_arg, vm)
            seen["args"] = args
            return 0

        runner.user_syscall = fake_user_syscall
        callback = runner._user_libc_callback("__mm_user_ext_reboot", None)
        self.assertIsNotNone(callback)
        assert callback is not None
        self.assertEqual(callback(vm, (0,)), 0)
        self.assertEqual(
            seen["args"],
            (142, 0xFEE1DEAD, 0x28121969, 0, 0, 0, 0),
        )

    def test_reboot_syscall_falls_back_to_linux_wrapper(self):
        runner = load_runner()
        fn = muir.Function(
            "__se_sys_reboot",
            [muir.Block("entry", [muir.Ret(muir.Imm(0))])],
            set(),
        )
        expanded, _ = expand_function(fn)
        program = Program((lower_function(expanded),))
        vm = program.new_vm()
        seen = {}

        def fake_call(vm_arg, name, args, **kwargs):
            self.assertIs(vm_arg, vm)
            seen["name"] = name
            seen["args"] = args
            return (0,)

        runner._call_linux_function_preserving_control = fake_call
        result = runner.user_syscall(vm, (142, 1, 2, 3, 4, 5, 6))
        self.assertEqual(result, 0)
        self.assertEqual(seen["name"], "__se_sys_reboot")
        self.assertEqual(seen["args"], (1, 2, 3, 4))

    def test_getopt_tracks_guest_short_option_state(self):
        runner = load_runner()
        program = Program()
        for name, value in (
            ("optarg", 0),
            ("opterr", 1),
            ("optind", 1),
            ("optopt", 0),
        ):
            program.define_data_symbol(
                f"__mm_user_ext_{name}",
                int(value).to_bytes(8, "little"),
                align=8,
            )
        vm = program.new_vm()
        callback = runner._user_libc_callback("__mm_user_ext_getopt", None)
        self.assertIsNotNone(callback)
        assert callback is not None

        def put(address: int, data: bytes) -> None:
            for index, byte in enumerate(data + b"\0"):
                vm.memory.write(address + index, 8, byte)

        argv = 0xCE00
        prog = 0xCF00
        cluster = 0xCF20
        optstring = 0xCF40
        put(prog, b"prog")
        put(cluster, b"-ab")
        put(optstring, b"ab")
        vm.memory.write(argv + 0, 64, prog)
        vm.memory.write(argv + 8, 64, cluster)
        vm.memory.write(argv + 16, 64, 0)

        optind = program.symbol_addresses["__mm_user_ext_optind"]
        optarg = program.symbol_addresses["__mm_user_ext_optarg"]
        self.assertEqual(callback(vm, (2, argv, optstring)), ord("a"))
        self.assertEqual(vm.memory.read(optind, 32), 1)
        self.assertEqual(callback(vm, (2, argv, optstring)), ord("b"))
        self.assertEqual(vm.memory.read(optind, 32), 2)
        self.assertEqual(callback(vm, (2, argv, optstring)), (1 << 64) - 1)

        option = 0xCF60
        value = 0xCF80
        optstring2 = 0xCFA0
        put(option, b"-n")
        put(value, b"42")
        put(optstring2, b"n:")
        vm.memory.write(argv + 8, 64, option)
        vm.memory.write(argv + 16, 64, value)
        vm.memory.write(argv + 24, 64, 0)
        vm.memory.write(optind, 32, 1)

        self.assertEqual(callback(vm, (3, argv, optstring2)), ord("n"))
        self.assertEqual(vm.memory.read(optind, 32), 3)
        self.assertEqual(vm.memory.read(optarg, 64), value)
        self.assertEqual(callback(vm, (3, argv, optstring2)), (1 << 64) - 1)

    def test_directory_callbacks_iterate_linux_dirent64(self):
        runner = load_runner()
        vm = Program().new_vm()
        errno_address = 0xD000
        path = 0xD100
        for index, byte in enumerate(b"/bin\0"):
            vm.memory.write(path + index, 8, byte)

        calls = {"getdents": 0}

        def fake_user_syscall(vm_arg, args):
            self.assertIs(vm_arg, vm)
            nr, a0, a1, a2, _a3, _a4, _a5 = map(int, args)
            if nr == 56:
                self.assertEqual(a0, ((-100) & ((1 << 64) - 1)))
                self.assertEqual(a1, path)
                self.assertTrue(a2 & 0x10000)
                return 7
            if nr == 61:
                self.assertEqual(a0, 7)
                calls["getdents"] += 1
                if calls["getdents"] > 1:
                    return 0
                buf = a1
                vm.memory.write(buf + 0, 64, 123)
                vm.memory.write(buf + 8, 64, 1)
                vm.memory.write(buf + 16, 16, 24)
                vm.memory.write(buf + 18, 8, 10)
                for index, byte in enumerate(b"sh\0"):
                    vm.memory.write(buf + 19 + index, 8, byte)
                return 24
            if nr == 57:
                self.assertEqual(a0, 7)
                return 0
            self.fail(f"unexpected syscall {nr}")

        runner.user_syscall = fake_user_syscall
        opendir = runner._user_libc_callback("__mm_user_ext_opendir", errno_address)
        readdir = runner._user_libc_callback("__mm_user_ext_readdir64", errno_address)
        closedir = runner._user_libc_callback("__mm_user_ext_closedir", errno_address)
        self.assertIsNotNone(opendir)
        self.assertIsNotNone(readdir)
        self.assertIsNotNone(closedir)
        assert opendir is not None and readdir is not None and closedir is not None

        handle = opendir(vm, (path,))
        self.assertNotEqual(handle, 0)
        entry = readdir(vm, (handle,))
        self.assertNotEqual(entry, 0)
        self.assertEqual(vm.memory.read(entry + 0, 64), 123)
        self.assertEqual(vm.memory.read(entry + 16, 16), 24)
        self.assertEqual(
            bytes(vm.memory.read(entry + 19 + i, 8) for i in range(3)),
            b"sh\0",
        )
        self.assertEqual(readdir(vm, (handle,)), 0)
        self.assertEqual(closedir(vm, (handle,)), 0)
        self.assertEqual(readdir(vm, (handle,)), 0)
        self.assertEqual(vm.memory.read(errno_address, 32), 9)

    def test_isoc23_strtoul_updates_endptr_and_errno(self):
        runner = load_runner()
        vm = Program().new_vm()
        errno_address = 0xC000
        callback = runner._user_libc_callback(
            "__mm_user_ext___isoc23_strtoul",
            errno_address,
        )
        self.assertIsNotNone(callback)
        assert callback is not None

        def put(address: int, data: bytes) -> None:
            for index, byte in enumerate(data + b"\0"):
                vm.memory.write(address + index, 8, byte)

        text = 0xC100
        endptr = 0xC200
        put(text, b"  0b101x")
        self.assertEqual(callback(vm, (text, endptr, 0)), 5)
        self.assertEqual(vm.memory.read(endptr, 64), text + 7)

        put(text, b"-1")
        self.assertEqual(
            callback(vm, (text, endptr, 10)),
            (1 << 64) - 1,
        )

        vm.memory.write(errno_address, 32, 0)
        put(text, b"18446744073709551616")
        self.assertEqual(
            callback(vm, (text, endptr, 10)),
            (1 << 64) - 1,
        )
        self.assertEqual(vm.memory.read(errno_address, 32), 34)

    def test_vsnprintf_percent_m_uses_guest_errno_without_consuming_arg(self):
        runner = load_runner()
        vm = Program().new_vm()
        errno_address = 0xC300
        callback = runner._user_libc_callback(
            "__mm_user_ext_vsnprintf",
            errno_address,
        )
        self.assertIsNotNone(callback)
        assert callback is not None

        fmt = 0xC400
        ap = 0xC500
        out = 0xC600
        for index, byte in enumerate(b"%m:%d\0"):
            vm.memory.write(fmt + index, 8, byte)
        vm.memory.write(ap, 64, 7)
        vm.memory.write(errno_address, 32, 2)

        result = callback(vm, (out, 128, fmt, ap))
        expected = b"No such file or directory:7"
        self.assertEqual(result, len(expected))
        self.assertEqual(
            bytes(vm.memory.read(out + i, 8) for i in range(len(expected) + 1)),
            expected + b"\0",
        )

    def _mov64_function(self, name: str):
        src = muir.Slot("src")
        dst = muir.Slot("dst")
        fn = muir.Function(
            name,
            [
                muir.Block(
                    "entry",
                    [
                        muir.Mov(muir.Width.I64, dst, src),
                        muir.Ret(dst),
                    ],
                )
            ],
            {"src", "dst"},
            ("src",),
        )
        expanded, _ = expand_function(fn)
        return lower_function(expanded)

    def test_native_vm_preserves_i64_slot_move_in_initial_program(self):
        program = Program((self._mov64_function("mov64_initial"),))
        vm = NativeVM(program)
        self.assertEqual(
            vm.run_function("mov64_initial", (0x18ACBEA,), result_count=1),
            (0x18ACBEA,),
        )

    def test_native_vm_preserves_i64_slot_move_in_appended_segment(self):
        program = Program()
        vm = NativeVM(program)
        program.add_function(self._mov64_function("mov64_appended"))
        self.assertEqual(
            vm.run_function("mov64_appended", (0x18ACBEA,), result_count=1),
            (0x18ACBEA,),
        )

    def test_native_vm_can_call_service_registered_after_vm_creation(self):
        program = Program()
        vm = NativeVM(program)

        program.register_service("__mm_user_ext_probe", lambda _vm, _args: 77)
        result = muir.Slot("result")
        fn = muir.Function(
            "__mm_user_probe_caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol="__mm_user_ext_probe"),
                            (),
                            result,
                        ),
                        muir.Ret(result),
                    ],
                )
            ],
            {"result"},
        )
        expanded, _ = expand_function(fn)
        program.add_function(lower_function(expanded))

        descriptor = program.symbol_addresses["__mm_user_ext_probe"]
        self.assertNotEqual(program.initial_memory.read(descriptor, 64), 0)
        self.assertEqual(vm.memory.read(descriptor, 64), 0)

        self.assertEqual(
            vm.run_function("__mm_user_probe_caller", result_count=1),
            (77,),
        )
        self.assertEqual(
            vm.memory.read(descriptor, 64),
            program.initial_memory.read(descriptor, 64),
        )

if __name__ == "__main__":
    unittest.main()
