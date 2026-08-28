import unittest

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.legalize import legalize_module
from src.minimachine.lower_p3 import lower_function
from src.minimachine.runtime import collect_runtime_surface, install_runtime
from src.minimachine.vm import Program


def executable(functions):
    p3_functions = []
    for fn in functions:
        expanded, _ = expand_function(fn)
        p3_functions.append(lower_function(expanded))
    program = Program(p3_functions)
    install_runtime(program, collect_runtime_surface(functions))
    return program


class RuntimeTests(unittest.TestCase):
    def test_scalar_helpers_execute_end_to_end(self):
        functions, _ = legalize_module(
            """
            define i64 @scalar(i64 %a, i64 %b) {
            entry:
              %x = and i64 %a, %b
              %y = shl i64 %x, 3
              %z = mul i64 %y, 5
              %c = icmp ult i64 %z, 1000
              %r = select i1 %c, i64 %z, i64 1000
              ret i64 %r
            }
            """
        )
        program = executable(functions)
        vm = program.new_vm()
        self.assertEqual(vm.run_function("scalar", (0x3F, 0x0F)), (600,))

    def test_signed_division_matches_llvm_truncation(self):
        functions, _ = legalize_module(
            """
            define i64 @sdiv_test(i64 %a, i64 %b) {
            entry:
              %q = sdiv i64 %a, %b
              ret i64 %q
            }
            """
        )
        program = executable(functions)
        result = program.new_vm().run_function(
            "sdiv_test",
            ((-7) & ((1 << 64) - 1), 3),
        )
        self.assertEqual(result, (((-2) & ((1 << 64) - 1)),))

    def test_supervisor_state_system_service_executes(self):
        fn = muir.Function(
            "state_user",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys("state_write_status", (muir.Imm(0x12),), None),
                        muir.Sys("state_set_status", (muir.Imm(0x80),), None),
                        muir.Sys("state_read_status", (), muir.Slot("out")),
                        muir.Ret(muir.Slot("out")),
                    ],
                )
            ],
            {"out"},
        )
        program = executable([fn])
        self.assertEqual(
            program.new_vm().run_function("state_user"),
            (0x92,),
        )

    def test_atomic_add_mutates_memory_and_returns_old(self):
        fn = muir.Function(
            "atomic_user",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys(
                            "atomic_add_i32_relaxed",
                            (muir.Slot("p"), muir.Slot("v")),
                            muir.Slot("old"),
                        ),
                        muir.Ret(muir.Slot("old")),
                    ],
                )
            ],
            {"p", "v", "old"},
            ("p", "v"),
        )
        program = executable([fn])
        vm = program.new_vm()
        vm.memory.write(0x2000, 32, 5)
        self.assertEqual(vm.run_function("atomic_user", (0x2000, 3)), (5,))
        self.assertEqual(vm.memory.read(0x2000, 32), 8)

    def test_faultable_load_returns_error_then_value(self):
        fn = muir.Function(
            "uget",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys(
                            "faultable_load_i32",
                            (muir.Slot("p"), muir.Imm(0)),
                            (muir.Slot("err"), muir.Slot("value")),
                        ),
                        muir.Ret(muir.Slot("value")),
                    ],
                )
            ],
            {"p", "err", "value"},
            ("p",),
        )
        program = executable([fn])
        vm = program.new_vm()
        vm.memory.write(0x3000, 32, 0x12345678)
        self.assertEqual(vm.run_function("uget", (0x3000,)), (0x12345678,))

    def test_static_key_controls_real_p3_branch(self):
        fn = muir.Function(
            "static_user",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys(
                            "static_branch",
                            (muir.Slot("key"),),
                            muir.Slot("enabled"),
                        ),
                        muir.Br(
                            muir.Width.I8,
                            muir.Cond.EQ,
                            muir.Slot("enabled"),
                            muir.Imm(0),
                            muir.Target(label="off"),
                            muir.Target(label="on"),
                        ),
                    ],
                ),
                muir.Block("off", [muir.Ret(muir.Imm(10))]),
                muir.Block("on", [muir.Ret(muir.Imm(20))]),
            ],
            {"key", "enabled"},
            ("key",),
        )
        program = executable([fn])

        off_vm = program.new_vm()
        self.assertEqual(off_vm.run_function("static_user", (7,)), (10,))

        on_vm = program.new_vm()
        on_vm.static_keys[7] = 1
        self.assertEqual(on_vm.run_function("static_user", (7,)), (20,))

    def test_ecall_requires_explicit_handler(self):
        fn = muir.Function(
            "ecall_user",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys(
                            "ecall",
                            (muir.Slot("x"),),
                            (muir.Slot("a"), muir.Slot("b")),
                        ),
                        muir.Sub(
                            muir.Width.I64,
                            muir.Slot("d"),
                            muir.Slot("b"),
                            muir.Slot("a"),
                        ),
                        muir.Ret(muir.Slot("d")),
                    ],
                )
            ],
            {"x", "a", "b", "d"},
            ("x",),
        )
        program = executable([fn])
        vm = program.new_vm()
        with self.assertRaisesRegex(Exception, "ecall reached without"):
            vm.run_function("ecall_user", (9,))

        vm = program.new_vm()
        vm.ecall_handler = lambda _vm, args: (args[0], args[0] + 4)
        self.assertEqual(vm.run_function("ecall_user", (9,)), (4,))

    def test_vector_state_snapshot_and_restore_execute(self):
        fn = muir.Function(
            "vector_state_user",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys(
                            "vector_state_snapshot",
                            (),
                            (
                                muir.Slot("a"),
                                muir.Slot("b"),
                                muir.Slot("c"),
                                muir.Slot("d"),
                                muir.Slot("e"),
                            ),
                        ),
                        muir.Sys(
                            "vector_state_restore",
                            (
                                muir.Imm(11),
                                muir.Imm(12),
                                muir.Imm(13),
                                muir.Imm(14),
                            ),
                            None,
                        ),
                        muir.Sys(
                            "vector_state_snapshot",
                            (),
                            (
                                muir.Slot("a2"),
                                muir.Slot("b2"),
                                muir.Slot("c2"),
                                muir.Slot("d2"),
                                muir.Slot("e2"),
                            ),
                        ),
                        muir.Ret(muir.Slot("e2")),
                    ],
                )
            ],
            {"a","b","c","d","e","a2","b2","c2","d2","e2"},
        )
        program = executable([fn])
        vm = program.new_vm()
        vm.vector_state = (1, 2, 3, 4, 64)
        self.assertEqual(vm.run_function("vector_state_user"), (64,))
        self.assertEqual(vm.vector_state, (11, 12, 13, 14, 64))

    def test_llvm_fixed_register_reads_execute(self):
        functions, stats = legalize_module(
            """
            define i64 @read_sp() {
            entry:
              %v = call i64 @llvm.read_register.i64(metadata !0)
              ret i64 %v
            }

            define i64 @read_tp() {
            entry:
              %v = call i64 @llvm.read_register.i64(metadata !1)
              ret i64 %v
            }

            !0 = !{!"sp"}
            !1 = !{!"tp"}
            """
        )
        self.assertEqual(stats.lowered_read_sp, 1)
        self.assertEqual(stats.lowered_read_thread_pointer, 1)

        program = executable(functions)

        sp_vm = program.new_vm()
        sp_result = sp_vm.run_function("read_sp")
        self.assertEqual(sp_result[0], sp_vm.stack_top - program.functions["read_sp"].frame_size - 8)

        tp_vm = program.new_vm()
        tp_vm.system_state["thread_pointer"] = 0xABCDEF
        self.assertEqual(tp_vm.run_function("read_tp"), (0xABCDEF,))

    def test_aggregate_load_store_round_trip_executes(self):
        functions, _ = legalize_module(
            """
            %Pair = type { i32, i32 }

            define void @copy_pair(ptr %src, ptr %dst) {
            entry:
              %v = load %Pair, ptr %src
              store %Pair %v, ptr %dst
              ret void
            }
            """
        )
        program = executable(functions)
        vm = program.new_vm()
        source = 0x4000
        dest = 0x5000
        raw = bytes([1, 2, 3, 4, 0xAA, 0xBB, 0xCC, 0xDD])
        for i, byte in enumerate(raw):
            vm.memory.write(source + i, 8, byte)

        self.assertEqual(
            vm.run_function("copy_pair", (source, dest), result_count=0),
            (),
        )
        copied = bytes(vm.memory.read(dest + i, 8) for i in range(len(raw)))
        self.assertEqual(copied, raw)

    def test_pointer_scaled_helper_executes(self):
        fn = muir.Function(
            "ptr",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Helper(
                            "__mm_ptr_add_scaled_24",
                            (muir.Slot("base"), muir.Slot("index")),
                            muir.Slot("out"),
                        ),
                        muir.Ret(muir.Slot("out")),
                    ],
                )
            ],
            {"base", "index", "out"},
            ("base", "index"),
        )
        program = executable([fn])
        self.assertEqual(
            program.new_vm().run_function("ptr", (1000, 7)),
            (1168,),
        )


if __name__ == "__main__":
    unittest.main()
