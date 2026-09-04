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
from src.minimachine.image import ImageObject, ModuleImage
from src.minimachine.native_vm import NativeVM
from src.minimachine.user_bundle import namespace_user_program
from src.minimachine.user_image import UserProgramImage, pack_user_program
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
    def test_user_external_callback_tracks_dynamic_active_linux_task(self):
        runner = load_runner()
        program = Program()
        boot_task = 0xB4C000
        shell_task = 0xB91880
        child_task = 0xB91EA0
        current_addr = program.define_data_symbol(
            "minimachine_current_task",
            boot_task.to_bytes(8, "little"),
            align=8,
        )
        vm = program.new_vm()
        vm.linux_current_task = boot_task

        main = muir.Function(
            "unused",
            [muir.Block("entry", [muir.Ret(None)])],
            set(),
        )
        expanded, _ = expand_function(main)
        image = UserProgramImage(
            "unused",
            (lower_function(expanded),),
            ModuleImage(
                objects=(),
                aliases=(),
                external_data=(),
                external_functions=("__mm_shell_ext_getpid",),
                skipped_linker_metadata=(),
            ),
            "none",
            (),
        )

        seen = []

        def fake_user_syscall(vm_arg, args):
            self.assertIs(vm_arg, vm)
            seen.append(
                (
                    vm_arg.linux_current_task,
                    vm_arg.memory.read(current_addr, 64),
                    args,
                )
            )
            return 15

        runner.user_syscall = fake_user_syscall
        runner.install_user_external_surface(vm, image, 0)
        callback = program.host_services["__mm_shell_ext_getpid"]

        vm.active_user_task = shell_task
        self.assertEqual(callback(vm, ()), 15)
        self.assertEqual(
            seen[-1],
            (shell_task, shell_task, (172, 0, 0, 0, 0, 0, 0)),
        )

        # A vfork child inherits the exact same userspace namespace before
        # exec. The callback must follow the dynamic continuation owner rather
        # than the task that happened to install this namespace.
        vm.active_user_task = child_task
        vm.linux_current_task = shell_task
        vm.memory.write(current_addr, 64, shell_task)
        self.assertEqual(callback(vm, ()), 15)
        self.assertEqual(
            seen[-1],
            (child_task, child_task, (172, 0, 0, 0, 0, 0, 0)),
        )
        self.assertEqual(vm.linux_current_task, child_task)
        self.assertEqual(vm.memory.read(current_addr, 64), child_task)

    def test_service3_isolates_exec_instances_by_linux_task(self):
        runner = load_runner()

        main = muir.Function(
            "main",
            [muir.Block("entry", [muir.Ret(None)])],
            set(),
        )
        helper = muir.Function(
            "helper",
            [muir.Block("entry", [muir.Ret(None)])],
            set(),
        )
        functions = []
        for function in (main, helper):
            expanded, _ = expand_function(function)
            functions.append(lower_function(expanded))

        base = UserProgramImage(
            "main",
            tuple(functions),
            ModuleImage(
                objects=(
                    ImageObject(
                        "global_counter",
                        "i64",
                        (7).to_bytes(8, "little"),
                        8,
                        None,
                        False,
                        (),
                    ),
                ),
                aliases=(),
                external_data=("environ",),
                external_functions=("write",),
                skipped_linker_metadata=(),
            ),
            "none",
            (),
        )
        named = namespace_user_program(
            base,
            internal_prefix="__mm_user_",
            external_prefix="__mm_user_ext_",
        )
        payload = pack_user_program(named)
        program = Program()
        vm = program.new_vm()

        pc = 0x02000000
        for offset, byte in enumerate(payload):
            vm.memory.write(pc + offset, 8, byte)

        def handoff(task: int, regs: int, user_sp: int):
            vm.linux_current_task = task
            vm.memory.write(regs + 0, 64, pc)
            vm.memory.write(regs + 8, 64, user_sp)
            vm.memory.write(regs + 80, 64, 1)
            return runner.linux_ecall(vm, (3, regs))

        first = handoff(0x1000, 0x03000000, 0x0E000000)
        self.assertIs(first, runner.HOST_CONTROL_TRANSFER)
        self.assertEqual(vm.current_function, "__mm_user_main")
        first_global = program.symbol_addresses["__mm_user_global_counter"]
        self.assertEqual(vm.memory.read(first_global, 64), 7)
        vm.memory.write(first_global, 64, 99)

        second = handoff(0x2000, 0x03000100, 0x0D000000)
        self.assertIs(second, runner.HOST_CONTROL_TRANSFER)
        second_entry = vm.current_function
        self.assertIsNotNone(second_entry)
        assert second_entry is not None
        self.assertTrue(second_entry.startswith("__mm_exec_2000_"))
        self.assertTrue(second_entry.endswith("_main"))

        instances = vm.user_exec_instances
        self.assertEqual(len(instances), 2)
        second_instance = instances[next(
            key for key in instances if key[0] == 0x2000
        )]
        namespace = second_instance["namespace"]
        self.assertIsNotNone(namespace)
        assert namespace is not None
        second_global = program.symbol_addresses[
            f"__mm_{namespace}_global_counter"
        ]
        self.assertNotEqual(first_global, second_global)
        self.assertEqual(vm.memory.read(first_global, 64), 99)
        self.assertEqual(vm.memory.read(second_global, 64), 7)

        vm.memory.write(second_global, 64, 1234)
        functions_before = len(program.functions)
        data_end_before = program._next_data
        third = handoff(0x2000, 0x03000200, 0x0C000000)
        self.assertIs(third, runner.HOST_CONTROL_TRANSFER)
        self.assertEqual(vm.current_function, second_entry)
        self.assertEqual(len(program.functions), functions_before)
        self.assertEqual(program._next_data, data_end_before)
        self.assertEqual(vm.memory.read(second_global, 64), 7)
        self.assertEqual(vm.memory.read(first_global, 64), 99)

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

    def test_exit_and__exit_route_through_linux_task_exit(self):
        runner = load_runner()
        vm = Program().new_vm()
        vm.linux_current_task = 0xBEEF
        vm.active_user_task = 0xBEEF
        vm.user_task_pids = {0xBEEF: 16}
        vm.user_task_parent_pids = {0xBEEF: 15}
        calls = []

        def fake_user_syscall(vm_arg, args):
            self.assertIs(vm_arg, vm)
            calls.append(args)
            return runner.HOST_CONTROL_TRANSFER

        runner.user_syscall = fake_user_syscall
        for original, status in (("exit", 7), ("_exit", 9)):
            with self.subTest(original=original):
                callback = runner._user_libc_callback(
                    f"__mm_user_ext_{original}",
                    None,
                )
                self.assertIsNotNone(callback)
                assert callback is not None
                self.assertIs(
                    callback(vm, (status,)),
                    runner.HOST_CONTROL_TRANSFER,
                )
                self.assertEqual(vm._user_exit_task, 0xBEEF)
                self.assertEqual(vm._user_exit_status, status)

        self.assertEqual(
            calls,
            [
                (93, 7, 0, 0, 0, 0, 0),
                (93, 9, 0, 0, 0, 0, 0),
            ],
        )
        self.assertEqual(
            vm.user_wait_exits,
            [(16, 15, 7, 0xBEEF)],
        )

    def test_semantic_linux_call_owner_prefers_active_userspace_task(self):
        runner = load_runner()

        fn = muir.Function(
            "echo",
            [
                muir.Block(
                    "entry",
                    [muir.Ret(muir.Slot("value"))],
                )
            ],
            {"value"},
            ("value",),
        )
        expanded, _ = expand_function(fn)
        program = Program((lower_function(expanded),))
        vm = program.new_vm()

        kernel_task = 0xB4C000
        user_task = 0xB91880
        vm.linux_current_task = kernel_task
        vm.active_user_task = user_task

        value = 0x123456789ABCDEF0
        self.assertEqual(
            runner._call_linux_function_preserving_control(
                vm,
                "echo",
                (value,),
                result_count=1,
            ),
            (value,),
        )
        self.assertIn((user_task, 0), vm.linux_task_semantic_stacks)
        self.assertNotIn((kernel_task, 0), vm.linux_task_semantic_stacks)

    def test_semantic_linux_call_stacks_do_not_clobber_parked_tasks(self):
        runner = load_runner()

        fn = muir.Function(
            "echo",
            [
                muir.Block(
                    "entry",
                    [muir.Ret(muir.Slot("value"))],
                )
            ],
            {"value"},
            ("value",),
        )
        expanded, _ = expand_function(fn)
        program = Program((lower_function(expanded),))
        vm = program.new_vm()

        parent = 0xB100
        child = 0xB200

        vm.linux_current_task = parent
        self.assertEqual(
            runner._call_linux_function_preserving_control(
                vm,
                "echo",
                (0x1111222233334444,),
                result_count=1,
            ),
            (0x1111222233334444,),
        )
        parent_top = vm.linux_task_semantic_stacks[(parent, 0)]
        parent_linked = program.functions["echo"]
        parent_sp = (
            parent_top
            - parent_linked.frame_size
            - 8  # one argument
            - 8  # one result word
        )
        parent_arg = parent_sp + parent_linked.frame_size
        self.assertEqual(
            vm.memory.read(parent_arg, 64),
            0x1111222233334444,
        )

        # Model the parent being parked in its semantic-call stack while the
        # child performs another Linux call. The child must get a distinct
        # persistent P3 stack instead of reusing the parent's arena.
        vm.linux_current_task = child
        self.assertEqual(
            runner._call_linux_function_preserving_control(
                vm,
                "echo",
                (0xAAAABBBBCCCCDDDD,),
                result_count=1,
            ),
            (0xAAAABBBBCCCCDDDD,),
        )
        child_top = vm.linux_task_semantic_stacks[(child, 0)]
        self.assertNotEqual(child_top, parent_top)
        self.assertEqual(
            vm.memory.read(parent_arg, 64),
            0x1111222233334444,
        )

    def test_semantic_linux_call_stack_skips_same_task_parked_frame(self):
        runner = load_runner()

        fn = muir.Function(
            "echo",
            [
                muir.Block(
                    "entry",
                    [muir.Ret(muir.Slot("value"))],
                )
            ],
            {"value"},
            ("value",),
        )
        expanded, _ = expand_function(fn)
        program = Program((lower_function(expanded),))
        vm = program.new_vm()

        task = 0xB91880
        vm.linux_current_task = task
        first_value = 0x1111222233334444
        self.assertEqual(
            runner._call_linux_function_preserving_control(
                vm,
                "echo",
                (first_value,),
                result_count=1,
            ),
            (first_value,),
        )

        depth0_top = vm.linux_task_semantic_stacks[(task, 0)]
        linked = program.functions["echo"]
        depth0_sp = depth0_top - linked.frame_size - 8 - 8
        depth0_arg = depth0_sp + linked.frame_size
        self.assertEqual(vm.memory.read(depth0_arg, 64), first_value)

        # Model the task being switched out while its depth-0 semantic call
        # frame is still the scheduler resume point. A later syscall from the
        # same userspace task must use another arena instead of clobbering it.
        vm.linux_task_contexts = {
            task: (depth0_sp, program.halt_code, depth0_arg),
        }
        second_value = 0xAAAABBBBCCCCDDDD
        self.assertEqual(
            runner._call_linux_function_preserving_control(
                vm,
                "echo",
                (second_value,),
                result_count=1,
            ),
            (second_value,),
        )

        self.assertIn((task, 1), vm.linux_task_semantic_stacks)
        self.assertNotEqual(
            vm.linux_task_semantic_stacks[(task, 1)],
            depth0_top,
        )
        self.assertEqual(vm.memory.read(depth0_arg, 64), first_value)

    def test_exiting_user_task_switch_marks_nonreturning_transfer(self):
        runner = load_runner()

        fn = muir.Function(
            "resume_parent",
            [muir.Block("entry", [muir.Ret(None)])],
            set(),
        )
        expanded, _ = expand_function(fn)
        program = Program((lower_function(expanded),))
        vm = program.new_vm()

        prev = 0xB100
        next_task = 0xB200
        resume_sp = 0xAB0000
        result_ptr = 0xAB1000
        resume_pc = program.block_code[("resume_parent", "entry")]
        vm.linux_task_contexts = {
            next_task: (resume_sp, resume_pc, result_ptr),
        }
        vm._preserved_call_depth = 1
        vm._user_exit_task = prev
        vm._user_exit_status = 23

        vm.memory.write(vm.sp + runner.RESULT_COUNT, 64, 1)
        vm.memory.write(vm.sp + runner.CALLER_SP, 64, 0xAC0000)
        vm.memory.write(vm.sp + runner.RET_PC, 64, program.halt_code)
        vm.memory.write(vm.sp + runner.RESULT_PTR, 64, 0xAC1000)

        result = runner.linux_ecall(
            vm,
            (2, prev, next_task, 0, 0, 0),
        )
        self.assertIs(result, runner.HOST_CONTROL_TRANSFER)
        self.assertEqual(vm.linux_current_task, next_task)
        self.assertEqual(vm.current_function, "resume_parent")
        self.assertEqual(vm.sp, resume_sp)
        self.assertEqual(vm.memory.read(result_ptr, 64), prev)
        self.assertTrue(vm._preserved_nonreturning_transfer)
        self.assertTrue(vm.halted)
        self.assertEqual(vm._user_exit_task, 0)
        self.assertEqual(vm._user_exit_status, 0)

    def test_execvp_searches_guest_path_and_propagates_exec_transfer(self):
        runner = load_runner()
        program = Program()
        program.define_data_symbol(
            "__mm_user_ext_environ",
            (0).to_bytes(8, "little"),
            align=8,
        )
        vm = program.new_vm()

        def put(address: int, payload: bytes) -> None:
            for offset, byte in enumerate(payload + b"\0"):
                vm.memory.write(address + offset, 8, byte)

        file_ptr = 0xCC00
        argv_ptr = 0xCC80
        envp = 0xCD00
        path_entry = 0xCD80
        put(file_ptr, b"sh")
        put(path_entry, b"PATH=/missing:/bin")
        vm.memory.write(envp, 64, path_entry)
        vm.memory.write(envp + 8, 64, 0)
        environ_addr = program.symbol_addresses["__mm_user_ext_environ"]
        vm.memory.write(environ_addr, 64, envp)

        seen = []

        def fake_user_syscall(vm_arg, args):
            self.assertIs(vm_arg, vm)
            self.assertEqual(args[0], 221)
            candidate = runner._user_libc_callback
            path = bytearray()
            ptr = int(args[1])
            for index in range(256):
                byte = vm.memory.read(ptr + index, 8)
                if byte == 0:
                    break
                path.append(byte)
            seen.append((bytes(path), int(args[2]), int(args[3])))
            if len(seen) == 1:
                return ((1 << 64) - 2)
            return runner.HOST_CONTROL_TRANSFER

        runner.user_syscall = fake_user_syscall
        callback = runner._user_libc_callback("__mm_user_ext_execvp", None)
        self.assertIsNotNone(callback)
        assert callback is not None

        result = callback(vm, (file_ptr, argv_ptr))
        self.assertIs(result, runner.HOST_CONTROL_TRANSFER)
        self.assertEqual(
            seen,
            [
                (b"/missing/sh", argv_ptr, envp),
                (b"/bin/sh", argv_ptr, envp),
            ],
        )

    def test_execve_libc_bridge_propagates_nonreturning_transfer(self):
        runner = load_runner()
        program = Program()
        vm = program.new_vm()

        runner.user_syscall = (
            lambda _vm, _args: runner.HOST_CONTROL_TRANSFER
        )
        callback = runner._user_libc_callback("__mm_user_ext_execve", None)
        self.assertIsNotNone(callback)
        assert callback is not None
        self.assertIs(
            callback(vm, (0xD000, 0xD100, 0xD200)),
            runner.HOST_CONTROL_TRANSFER,
        )

    def test_preserved_call_does_not_restore_pre_exec_control(self):
        runner = load_runner()
        program = Program()

        replacement = muir.Function(
            "replacement_user",
            [muir.Block("entry", [muir.Ret(None)])],
            set(),
        )
        expanded, _ = expand_function(replacement)
        program.add_function(lower_function(expanded))

        def transfer(vm_arg, args):
            self.assertEqual(args, ())
            vm_arg.enter_function(
                "replacement_user",
                (),
                stack_top=0xEE0000,
                result_count=0,
            )
            vm_arg._preserved_nonreturning_transfer = True
            vm_arg.halted = True
            return runner.HOST_CONTROL_TRANSFER

        program.register_service("__mm_exec_transfer_probe", transfer)
        outer = muir.Function(
            "linux_exec_probe",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol="__mm_exec_transfer_probe"),
                            (),
                            None,
                        ),
                        muir.Ret(None),
                    ],
                )
            ],
            set(),
        )
        expanded, _ = expand_function(outer)
        program.add_function(lower_function(expanded))

        vm = program.new_vm()
        old_sp = vm.sp
        result = runner._call_linux_function_preserving_control(
            vm,
            "linux_exec_probe",
            (),
            result_count=0,
        )
        self.assertIs(result, runner.HOST_CONTROL_TRANSFER)
        self.assertEqual(vm.current_function, "replacement_user")
        self.assertNotEqual(vm.sp, old_sp)
        self.assertFalse(vm.halted)
        self.assertEqual(getattr(vm, "_preserved_call_depth", -1), 0)
        self.assertFalse(
            getattr(vm, "_preserved_nonreturning_transfer", True)
        )

    def test_task_scoped_transfer_stops_at_outer_waiter(self):
        runner = load_runner()
        fn = muir.Function(
            "child_semantic_call",
            [muir.Block("entry", [muir.Ret(None)])],
            set(),
        )
        expanded, _ = expand_function(fn)
        program = Program((lower_function(expanded),))
        vm = program.new_vm()

        parent = 0xB100
        child = 0xB200
        vm.linux_current_task = child
        vm._preserved_call_depth = 1
        vm._preserved_call_tasks = [parent]

        def fake_run(*, max_steps):
            self.assertGreater(max_steps, 0)
            self.assertEqual(
                vm._preserved_call_tasks,
                [parent, child],
            )
            self.assertTrue(
                runner._arm_preserved_task_transfer(
                    vm,
                    child,
                    reason="test-child-exec",
                )
            )
            self.assertEqual(vm._preserved_transfer_stop_depth, 1)
            self.assertTrue(vm.halted)

        vm.run = fake_run
        result = runner._call_linux_function_preserving_control(
            vm,
            "child_semantic_call",
            (),
            result_count=0,
        )

        self.assertIs(result, runner.HOST_CONTROL_TRANSFER)
        self.assertEqual(vm._preserved_call_depth, 1)
        self.assertEqual(vm._preserved_call_tasks, [parent])
        self.assertFalse(
            getattr(vm, "_preserved_nonreturning_transfer", False)
        )
        self.assertFalse(
            hasattr(vm, "_preserved_transfer_stop_depth")
        )
        self.assertFalse(vm.halted)

    def test_blocking_syscall_fallback_propagates_control_transfer(self):
        runner = load_runner()
        fn = muir.Function(
            "__se_sys_wait4",
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
            seen["kwargs"] = kwargs
            return runner.HOST_CONTROL_TRANSFER

        runner._call_linux_function_preserving_control = fake_call
        result = runner.user_syscall(
            vm,
            (260, 14, 0xD340, 0, 0, 0, 0),
        )
        self.assertIs(result, runner.HOST_CONTROL_TRANSFER)
        self.assertEqual(seen["name"], "__se_sys_wait4")
        self.assertEqual(seen["args"], (14, 0xD340, 0, 0))
        self.assertTrue(seen["kwargs"]["preserve_linux_task_state"])

    def test_waitpid_uses_linux_wait4_and_guest_status(self):
        runner = load_runner()
        vm = Program().new_vm()
        status_ptr = 0xD340
        seen = {}

        original_ret_pc = 0x123456789ABCDEF0
        vm.memory.write(vm.sp + runner.FRAME_SIZE, 64, runner.HEADER_SIZE)
        vm.memory.write(vm.sp + runner.ARG_COUNT, 64, 0)
        vm.memory.write(vm.sp + runner.RET_PC, 64, original_ret_pc)

        def fake_user_syscall(vm_arg, args):
            self.assertIs(vm_arg, vm)
            seen["args"] = args
            vm.memory.write(status_ptr, 32, 0x2A00)
            # Model a child reusing the parent's concrete P3 userspace stack
            # while wait4 has the parent blocked.
            vm.memory.write(vm.sp + runner.RET_PC, 64, 0xDEADBEEF)
            return 14

        runner.user_syscall = fake_user_syscall
        callback = runner._user_libc_callback("__mm_user_ext_waitpid", None)
        self.assertIsNotNone(callback)
        assert callback is not None

        self.assertEqual(callback(vm, (14, status_ptr, 0)), 14)
        self.assertEqual(
            seen["args"],
            (260, 14, status_ptr, 0, 0, 0, 0),
        )
        self.assertEqual(vm.memory.read(status_ptr, 32), 0x2A00)
        self.assertEqual(vm.memory.read(vm.sp + runner.RET_PC, 64), original_ret_pc)

    def test_waitpid_drives_scheduler_until_pending_fork_child_is_armed(self):
        runner = load_runner()
        program = Program()
        vm = program.new_vm()
        vm.program.functions["schedule"] = object()
        vm.pending_user_fork_continuation = (0x1000, 0x2000, 0x3000)
        vm.active_user_task = 0xB91880
        vm.linux_current_task = 0xB4C000
        vm.memory.write(vm.sp + runner.FRAME_SIZE, 64, runner.HEADER_SIZE)
        vm.memory.write(vm.sp + runner.ARG_COUNT, 64, 0)
        vm.memory.write(vm.sp + runner.RESULT_COUNT, 64, 1)
        vm.memory.write(vm.sp + runner.CALLER_SP, 64, 0)
        vm.memory.write(vm.sp + runner.RET_PC, 64, 0x1234)
        vm.memory.write(vm.sp + runner.RESULT_PTR, 64, 0xD000)
        calls = []

        def fake_call(vm_arg, name, args, **kwargs):
            self.assertIs(vm_arg, vm)
            calls.append((name, args, kwargs))
            if name == "schedule":
                # Model a CLONE_VM child reusing the parent's ext_waitpid
                # frame before the parent reaches wait4.
                vm.memory.write(vm.sp + runner.RESULT_COUNT, 64, 16)
                vm.memory.write(vm.sp + runner.RET_PC, 64, 0xDEADBEEF)
                if len(calls) == 2:
                    vm.pending_user_fork_continuation = None
            return ()

        def fake_user_syscall(vm_arg, args):
            self.assertIs(vm_arg, vm)
            self.assertEqual(args[0], 260)
            return (1 << 64) - 10

        runner._call_linux_function_preserving_control = fake_call
        runner.user_syscall = fake_user_syscall
        callback = runner._user_libc_callback("__mm_user_ext_waitpid", None)
        self.assertIsNotNone(callback)
        assert callback is not None

        self.assertEqual(
            callback(vm, ((1 << 64) - 1, 0, 0)),
            (1 << 64) - 1,
        )
        schedule_calls = [call for call in calls if call[0] == "schedule"]
        self.assertEqual(len(schedule_calls), 2)
        self.assertTrue(
            all(call[2]["preserve_linux_task_state"] for call in schedule_calls)
        )
        self.assertTrue(
            all(
                call[2]["call_task_override"] == 0xB4C000
                for call in schedule_calls
            )
        )
        self.assertEqual(vm.memory.read(vm.sp + runner.RESULT_COUNT, 64), 1)
        self.assertEqual(vm.memory.read(vm.sp + runner.RET_PC, 64), 0x1234)

    def test_waitpid_replays_tracked_child_exit_when_wait4_bridge_is_unavailable(self):
        runner = load_runner()
        vm = Program().new_vm()
        parent_task = 0xB91880
        child_task = 0xB91EA0
        status_ptr = 0xD3C0
        vm.active_user_task = parent_task
        vm.linux_current_task = parent_task
        vm.user_task_pids = {parent_task: 15, child_task: 16}
        vm.user_wait_exits = [(16, 15, 0, child_task)]

        def fake_user_syscall(vm_arg, args):
            self.assertIs(vm_arg, vm)
            self.assertEqual(args[0], 260)
            return (1 << 64) - 38

        runner.user_syscall = fake_user_syscall
        callback = runner._user_libc_callback("__mm_user_ext_waitpid", None)
        self.assertIsNotNone(callback)
        assert callback is not None

        self.assertEqual(
            callback(vm, ((1 << 64) - 1, status_ptr, 0)),
            16,
        )
        self.assertEqual(vm.memory.read(status_ptr, 32), 0)
        self.assertEqual(vm.user_wait_exits, [])

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
        for original, expected_flags in (("fork", 0x111), ("vfork", 0x4111)):
            with self.subTest(original=original):
                callback = runner._user_libc_callback(
                    f"__mm_user_ext_{original}",
                    None,
                )
                self.assertIsNotNone(callback)
                assert callback is not None
                original_ret_pc = 0x123456789ABCDEF0
                vm.memory.write(vm.sp + runner.FRAME_SIZE, 64, runner.HEADER_SIZE)
                vm.memory.write(vm.sp + runner.ARG_COUNT, 64, 0)
                vm.memory.write(vm.sp + runner.RET_PC, 64, original_ret_pc)

                def fake_call_with_stack_damage(vm_arg, name, args, **kwargs):
                    fake_call(vm_arg, name, args, **kwargs)
                    vm_arg.memory.write(vm_arg.sp + runner.RET_PC, 64, 0xDEADBEEF)
                    return (321,)

                runner._call_linux_function_preserving_control = fake_call_with_stack_damage
                self.assertEqual(callback(vm, ()), 321)
                self.assertEqual(vm.memory.read(vm.sp + runner.RET_PC, 64), original_ret_pc)
                self.assertEqual(seen["name"], "__se_sys_clone")
                self.assertEqual(seen["args"], (expected_flags, 0, 0, 0, 0))
                self.assertTrue(seen["kwargs"]["preserve_linux_task_state"])
                self.assertIsNotNone(
                    getattr(vm, "pending_user_fork_continuation", None)
                )
                # Model service 2 consuming the first-run child continuation
                # before the next subtest starts another fork.
                vm.pending_user_fork_continuation = None

    def test_fork_retries_internal_restart_without_dropping_child_continuation(self):
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
        vm.memory.write(vm.sp + runner.FRAME_SIZE, 64, runner.HEADER_SIZE)
        vm.memory.write(vm.sp + runner.ARG_COUNT, 64, 0)

        results = iter(((1 << 64) - 513, 17))
        calls = []

        def fake_call(vm_arg, name, args, **kwargs):
            self.assertIs(vm_arg, vm)
            calls.append((name, args, kwargs))
            return (next(results),)

        runner._call_linux_function_preserving_control = fake_call
        callback = runner._user_libc_callback("__mm_user_ext_vfork", None)
        self.assertIsNotNone(callback)
        assert callback is not None

        self.assertEqual(callback(vm, ()), 17)
        self.assertEqual(len(calls), 2)
        self.assertIsNotNone(
            getattr(vm, "pending_user_fork_continuation", None)
        )

    def test_getpid_and_getppid_track_active_userspace_task(self):
        runner = load_runner()
        vm = Program().new_vm()
        task = 0xB91EA0
        vm.active_user_task = task
        vm.linux_current_task = 0xB4C000

        results = {
            "minimachine_user_syscall": (1 << 64) - 38,
            "sys_getpid": 16,
            "sys_getppid": 15,
        }

        def fake_call(vm_arg, name, args, **kwargs):
            self.assertIs(vm_arg, vm)
            return (results[name],)

        runner._call_linux_function_preserving_control = fake_call
        vm.program.functions["minimachine_user_syscall"] = object()
        vm.program.functions["sys_getpid"] = object()
        vm.program.functions["sys_getppid"] = object()

        self.assertEqual(
            runner.user_syscall(vm, (172, 0, 0, 0, 0, 0, 0)),
            16,
        )
        self.assertEqual(
            runner.user_syscall(vm, (173, 0, 0, 0, 0, 0, 0)),
            15,
        )
        self.assertEqual(vm.user_task_pids[task], 16)
        self.assertEqual(vm.user_task_parent_pids[task], 15)

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

    def test_qsort_orders_guest_records_through_guest_comparator(self):
        runner = load_runner()
        vm = Program().new_vm()
        base = 0xD600
        compar = 0xD700
        values = [7, 1, 9, 3]
        for index, value in enumerate(values):
            vm.memory.write(base + index * 8, 64, value)

        original_call = runner._call_guest_descriptor_preserving_control
        original_name = runner._guest_function_name_from_descriptor

        def fake_name(vm_arg, descriptor):
            self.assertIs(vm_arg, vm)
            self.assertEqual(descriptor, compar)
            return "guest_cmp"

        def fake_call(vm_arg, descriptor, args, **kwargs):
            self.assertIs(vm_arg, vm)
            self.assertEqual(descriptor, compar)
            self.assertEqual(kwargs["result_count"], 1)
            left = vm.memory.read(args[0], 64)
            right = vm.memory.read(args[1], 64)
            result = -1 if left < right else (1 if left > right else 0)
            return (result & ((1 << 64) - 1),)

        runner._guest_function_name_from_descriptor = fake_name
        runner._call_guest_descriptor_preserving_control = fake_call
        try:
            callback = runner._user_libc_callback(
                "__mm_user_ext_qsort",
                None,
            )
            self.assertIsNotNone(callback)
            assert callback is not None
            self.assertIsNone(callback(vm, (base, len(values), 8, compar)))
        finally:
            runner._call_guest_descriptor_preserving_control = original_call
            runner._guest_function_name_from_descriptor = original_name

        self.assertEqual(
            [vm.memory.read(base + index * 8, 64) for index in range(len(values))],
            sorted(values),
        )

    def test_vasprintf_allocates_and_formats_guest_string(self):
        runner = load_runner()
        vm = Program().new_vm()

        strp = 0xD800
        fmt = 0xD880
        text = 0xD900
        ap = 0xD980
        for offset, byte in enumerate(b"%s:%d\0"):
            vm.memory.write(fmt + offset, 8, byte)
        for offset, byte in enumerate(b"item\0"):
            vm.memory.write(text + offset, 8, byte)
        vm.memory.write(ap, 64, text)
        vm.memory.write(ap + 8, 64, 42)

        callback = runner._user_libc_callback(
            "__mm_user_ext_vasprintf",
            None,
        )
        self.assertIsNotNone(callback)
        assert callback is not None

        self.assertEqual(callback(vm, (strp, fmt, ap)), 7)
        result = vm.memory.read(strp, 64)
        self.assertNotEqual(result, 0)
        self.assertEqual(
            bytes(vm.memory.read(result + index, 8) for index in range(8)),
            b"item:42\0",
        )

    def test_gnu_dev_helpers_round_trip_linux_dev_t_encoding(self):
        runner = load_runner()
        vm = Program().new_vm()

        major_cb = runner._user_libc_callback(
            "__mm_user_ext_gnu_dev_major",
            None,
        )
        minor_cb = runner._user_libc_callback(
            "__mm_user_ext_gnu_dev_minor",
            None,
        )
        make_cb = runner._user_libc_callback(
            "__mm_user_ext_gnu_dev_makedev",
            None,
        )
        self.assertIsNotNone(major_cb)
        self.assertIsNotNone(minor_cb)
        self.assertIsNotNone(make_cb)
        assert major_cb is not None
        assert minor_cb is not None
        assert make_cb is not None

        major = 0x12345
        minor = 0x6789A
        dev = make_cb(vm, (major, minor))
        self.assertEqual(major_cb(vm, (dev,)), major)
        self.assertEqual(minor_cb(vm, (dev,)), minor)

        # Also cover the compact legacy layout used by ordinary tty/dev nodes.
        small_dev = make_cb(vm, (240, 7))
        self.assertEqual(small_dev, (240 << 8) | 7)
        self.assertEqual(major_cb(vm, (small_dev,)), 240)
        self.assertEqual(minor_cb(vm, (small_dev,)), 7)

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

    def test_strsep_splits_in_place_and_preserves_empty_tokens(self):
        runner = load_runner()
        program = Program()
        vm = program.new_vm()

        stringp = 0xD100
        text_ptr = 0xD180
        delim_ptr = 0xD200
        for offset, byte in enumerate(b"a::bc\0"):
            vm.memory.write(text_ptr + offset, 8, byte)
        for offset, byte in enumerate(b":\0"):
            vm.memory.write(delim_ptr + offset, 8, byte)
        vm.memory.write(stringp, 64, text_ptr)

        callback = runner._user_libc_callback("__mm_user_ext_strsep", None)
        self.assertIsNotNone(callback)
        assert callback is not None

        first = callback(vm, (stringp, delim_ptr))
        second = callback(vm, (stringp, delim_ptr))
        third = callback(vm, (stringp, delim_ptr))
        final = callback(vm, (stringp, delim_ptr))

        self.assertEqual(first, text_ptr)
        self.assertEqual(second, text_ptr + 2)
        self.assertEqual(third, text_ptr + 3)
        self.assertEqual(final, 0)
        self.assertEqual(
            bytes(vm.memory.read(text_ptr + i, 8) for i in range(6)),
            b"a\0\0bc\0",
        )
        self.assertEqual(vm.memory.read(stringp, 64), 0)

    def test_syslog_state_preserves_log_perror_output(self):
        runner = load_runner()
        program = Program()
        vm = program.new_vm()
        calls = []

        ident_ptr = 0xD300
        fmt_ptr = 0xD340
        string_ptr = 0xD380
        for base, payload in (
            (ident_ptr, b"init\0"),
            (fmt_ptr, b"message %s %d\0"),
            (string_ptr, b"ready\0"),
        ):
            for offset, byte in enumerate(payload):
                vm.memory.write(base + offset, 8, byte)

        def fake_user_syscall(vm_arg, args):
            self.assertIs(vm_arg, vm)
            calls.append(args)
            self.assertEqual(args[0], 64)
            self.assertEqual(args[1], 2)
            return int(args[3])

        runner.user_syscall = fake_user_syscall
        openlog = runner._user_libc_callback("__mm_user_ext_openlog", None)
        syslog = runner._user_libc_callback("__mm_user_ext_syslog", None)
        closelog = runner._user_libc_callback("__mm_user_ext_closelog", None)
        self.assertIsNotNone(openlog)
        self.assertIsNotNone(syslog)
        self.assertIsNotNone(closelog)
        assert openlog is not None and syslog is not None and closelog is not None

        openlog(vm, (ident_ptr, 0x20, 0x18))
        self.assertEqual(
            vm.user_syslog_state,
            {"ident": b"init", "option": 0x20, "facility": 0x18},
        )
        syslog(vm, (5, fmt_ptr, string_ptr, 7))
        self.assertEqual(len(calls), 1)
        ptr = int(calls[0][2])
        size = int(calls[0][3])
        self.assertEqual(
            bytes(vm.memory.read(ptr + i, 8) for i in range(size)),
            b"init: message ready 7\n",
        )
        closelog(vm, ())
        self.assertIsNone(vm.user_syslog_state)

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

    def test_getopt_long_handles_short_and_struct_option_paths(self):
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
        callback = runner._user_libc_callback(
            "__mm_user_ext_getopt_long",
            None,
        )
        self.assertIsNotNone(callback)
        assert callback is not None

        def put(address: int, data: bytes) -> None:
            for index, byte in enumerate(data + b"\0"):
                vm.memory.write(address + index, 8, byte)

        argv = 0xD200
        prog = 0xD300
        short_arg = 0xD320
        optstring = 0xD340
        long_arg = 0xD380
        long_name = 0xD3C0
        longopts = 0xD400
        longindex = 0xD480
        flag = 0xD4C0

        put(prog, b"uname")
        put(short_arg, b"-s")
        put(optstring, b"sn")
        vm.memory.write(argv + 0, 64, prog)
        vm.memory.write(argv + 8, 64, short_arg)
        vm.memory.write(argv + 16, 64, 0)

        optind = program.symbol_addresses["__mm_user_ext_optind"]
        optarg = program.symbol_addresses["__mm_user_ext_optarg"]
        self.assertEqual(
            callback(vm, (2, argv, optstring, 0, 0)),
            ord("s"),
        )
        self.assertEqual(vm.memory.read(optind, 32), 2)
        self.assertEqual(vm.memory.read(optarg, 64), 0)
        self.assertEqual(
            callback(vm, (2, argv, optstring, 0, 0)),
            (1 << 64) - 1,
        )

        # struct option { "output", required_argument, NULL, 'o' }
        put(long_arg, b"--output=value")
        put(long_name, b"output")
        vm.memory.write(longopts + 0, 64, long_name)
        vm.memory.write(longopts + 8, 32, 1)
        vm.memory.write(longopts + 16, 64, 0)
        vm.memory.write(longopts + 24, 32, ord("o"))
        vm.memory.write(longopts + 32, 64, 0)
        vm.memory.write(argv + 8, 64, long_arg)
        vm.memory.write(optind, 32, 1)

        self.assertEqual(
            callback(vm, (2, argv, optstring, longopts, longindex)),
            ord("o"),
        )
        self.assertEqual(vm.memory.read(optind, 32), 2)
        self.assertEqual(vm.memory.read(longindex, 32), 0)
        value_ptr = vm.memory.read(optarg, 64)
        self.assertEqual(
            bytes(vm.memory.read(value_ptr + i, 8) for i in range(6)),
            b"value\0",
        )

        # struct option { "quiet", no_argument, &flag, 7 }
        quiet_arg = 0xD500
        quiet_name = 0xD540
        put(quiet_arg, b"--quiet")
        put(quiet_name, b"quiet")
        vm.memory.write(longopts + 0, 64, quiet_name)
        vm.memory.write(longopts + 8, 32, 0)
        vm.memory.write(longopts + 16, 64, flag)
        vm.memory.write(longopts + 24, 32, 7)
        vm.memory.write(longopts + 32, 64, 0)
        vm.memory.write(argv + 8, 64, quiet_arg)
        vm.memory.write(optind, 32, 1)
        vm.memory.write(flag, 32, 0)

        self.assertEqual(
            callback(vm, (2, argv, optstring, longopts, longindex)),
            0,
        )
        self.assertEqual(vm.memory.read(flag, 32), 7)
        self.assertEqual(vm.memory.read(optind, 32), 2)

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
