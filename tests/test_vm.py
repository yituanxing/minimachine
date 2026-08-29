import unittest

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.legalize import legalize_module
from src.minimachine.lower_p3 import lower_function
from src.minimachine.vm import HOST_CONTROL_TRANSFER, MASK64, Program


def machine(function: muir.Function):
    expanded, _ = expand_function(function)
    return lower_function(expanded)


class VMTests(unittest.TestCase):
    def test_arguments_sub_and_branch_execute(self):
        fn = muir.Function(
            "eq64",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sub(
                            muir.Width.I64,
                            muir.Slot("diff"),
                            muir.Slot("a"),
                            muir.Slot("b"),
                        ),
                        muir.Br(
                            muir.Width.I64,
                            muir.Cond.EQ,
                            muir.Slot("diff"),
                            muir.Imm(0),
                            muir.Target(label="yes"),
                            muir.Target(label="no"),
                        ),
                    ],
                ),
                muir.Block("yes", [muir.Ret(muir.Imm(1))]),
                muir.Block("no", [muir.Ret(muir.Imm(0))]),
            ],
            {"a", "b", "diff"},
            ("a", "b"),
        )
        program = Program([machine(fn)])

        self.assertEqual(
            program.new_vm().run_function("eq64", (5, 5)),
            (1,),
        )
        self.assertEqual(
            program.new_vm().run_function("eq64", (7, 5)),
            (0,),
        )

    def test_function_descriptor_call_executes(self):
        dec = muir.Function(
            "dec",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sub(
                            muir.Width.I64,
                            muir.Slot("out"),
                            muir.Slot("x"),
                            muir.Imm(1),
                        ),
                        muir.Ret(muir.Slot("out")),
                    ],
                )
            ],
            {"x", "out"},
            ("x",),
        )
        caller = muir.Function(
            "caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol="dec"),
                            (muir.Slot("x"),),
                            muir.Slot("r"),
                        ),
                        muir.Ret(muir.Slot("r")),
                    ],
                )
            ],
            {"x", "r"},
            ("x",),
        )

        program = Program([machine(dec), machine(caller)])
        self.assertEqual(
            program.new_vm().run_function("caller", (41,)),
            (40,),
        )

    def test_existing_function_descriptor_can_use_host_fast_path(self):
        dec = muir.Function(
            "dec",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sub(
                            muir.Width.I64,
                            muir.Slot("out"),
                            muir.Slot("x"),
                            muir.Imm(1),
                        ),
                        muir.Ret(muir.Slot("out")),
                    ],
                )
            ],
            {"x", "out"},
            ("x",),
        )
        caller = muir.Function(
            "caller",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Call(
                            muir.Callee(symbol="dec"),
                            (muir.Slot("x"),),
                            muir.Slot("r"),
                        ),
                        muir.Ret(muir.Slot("r")),
                    ],
                )
            ],
            {"x", "r"},
            ("x",),
        )
        program = Program([machine(dec), machine(caller)])
        program.replace_function_with_service("dec", lambda vm, args: args[0] + 100)

        self.assertEqual(program.new_vm().run_function("caller", (41,)), (141,))
        # Direct test entry still exercises the linked P3 body; only descriptor
        # calls are redirected for replay acceleration.
        self.assertEqual(program.new_vm().run_function("dec", (41,)), (40,))

    def test_runtime_loaded_function_installs_into_live_vm(self):
        base = muir.Function(
            "base",
            [muir.Block("entry", [muir.Ret(muir.Imm(1))])],
            set(),
        )
        runtime = muir.Function(
            "runtime_user",
            [muir.Block("entry", [muir.Ret(muir.Imm(7))])],
            set(),
        )
        program = Program([machine(base)])
        vm = program.new_vm()

        vm.install_function(machine(runtime))

        descriptor = program.symbol_addresses["runtime_user"]
        self.assertNotEqual(vm.memory.read(descriptor, 64), 0)
        self.assertEqual(vm.run_function("runtime_user", ()), (7,))

    def test_nested_function_call_preserves_outer_p3_continuation(self):
        inner = muir.Function(
            "inner",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sub(
                            muir.Width.I64,
                            muir.Slot("out"),
                            muir.Slot("x"),
                            muir.Imm(1),
                        ),
                        muir.Ret(muir.Slot("out")),
                    ],
                )
            ],
            {"x", "out"},
            ("x",),
        )
        outer = muir.Function(
            "outer",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys("nested", (muir.Slot("x"),), muir.Slot("r")),
                        muir.Ret(muir.Slot("r")),
                    ],
                )
            ],
            {"x", "r"},
            ("x",),
        )
        expanded_inner, _ = expand_function(inner)
        expanded_outer, _ = expand_function(outer)
        program = Program([
            lower_function(expanded_inner),
            lower_function(expanded_outer),
        ])

        def nested(vm, args):
            return vm.call_function_nested(
                "inner",
                args,
                stack_top=vm.stack_top - 0x10000,
                result_count=1,
            )[0]

        program.register_system("nested", nested)
        self.assertEqual(program.new_vm().run_function("outer", (41,)), (40,))

    def test_multi_result_system_service_executes(self):
        fn = muir.Function(
            "sys_pair_user",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys(
                            "pair",
                            (muir.Slot("x"),),
                            (muir.Slot("lo"), muir.Slot("hi")),
                        ),
                        muir.Sub(
                            muir.Width.I64,
                            muir.Slot("delta"),
                            muir.Slot("hi"),
                            muir.Slot("lo"),
                        ),
                        muir.Ret(muir.Slot("delta")),
                    ],
                )
            ],
            {"x", "lo", "hi", "delta"},
            ("x",),
        )
        program = Program([machine(fn)])
        program.register_system(
            "pair",
            lambda vm, args: (args[0], (args[0] + 1) & MASK64),
        )

        self.assertEqual(
            program.new_vm().run_function("sys_pair_user", (123,)),
            (1,),
        )

    def test_host_control_transfer_starts_fresh_activation(self):
        source = muir.Function(
            "source",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys("jump", (), None),
                        muir.Sys("bad", (), None),
                        muir.Ret(None),
                    ],
                )
            ],
            set(),
        )
        target = muir.Function(
            "target",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sys("good", (), None),
                        muir.Ret(None),
                    ],
                )
            ],
            set(),
        )
        program = Program([machine(source), machine(target)])
        seen = []

        def jump(vm, args):
            self.assertEqual(args, ())
            vm.enter_function(
                "target",
                (),
                stack_top=vm.stack_top - 0x1000,
                result_count=0,
            )
            return HOST_CONTROL_TRANSFER

        program.register_system("jump", jump)
        program.register_system("good", lambda vm, args: seen.append("good"))
        program.register_system("bad", lambda vm, args: seen.append("bad"))

        program.new_vm().run_function("source", (), result_count=0)
        self.assertEqual(seen, ["good"])

    def test_llvm_sext_i1_executes_with_exact_source_width(self):
        functions, _ = legalize_module(
            """
            define i64 @sx(i1 %x) {
            entry:
              %y = sext i1 %x to i64
              ret i64 %y
            }
            """
        )
        self.assertEqual(len(functions), 1)
        program = Program([machine(functions[0])])

        self.assertEqual(program.new_vm().run_function("sx", (0,)), (0,))
        self.assertEqual(
            program.new_vm().run_function("sx", (1,)),
            (MASK64,),
        )


    def test_llvm_indirectbr_executes_through_p3_target_slot(self):
        functions, stats = legalize_module(
            """
            define i64 @jump(ptr %target) {
            entry:
              indirectbr ptr %target, [label %left, label %right]

            left:
              ret i64 11

            right:
              ret i64 22
            }
            """
        )
        self.assertEqual(stats.lowered_indirectbr, 1)
        self.assertEqual(stats.arch_escapes, 0)

        program = Program([machine(functions[0])])
        left = program.block_code[("jump", "left")]
        right = program.block_code[("jump", "right")]

        self.assertEqual(
            program.new_vm().run_function("jump", (left,)),
            (11,),
        )
        self.assertEqual(
            program.new_vm().run_function("jump", (right,)),
            (22,),
        )


if __name__ == "__main__":
    unittest.main()
