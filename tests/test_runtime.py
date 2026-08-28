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
