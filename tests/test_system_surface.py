import unittest

from src.minimachine import muir, p3
from src.minimachine.system_surface import (
    referenced_system_symbols,
    resolve_system_surface,
    system_op_from_symbol,
)
from src.minimachine.vm import Program


def function_loading(symbol: str) -> p3.Function:
    return p3.Function(
        "main",
        [
            p3.Block(
                "entry",
                [
                    p3.Mov(
                        muir.Width.I64,
                        muir.Slot("entry"),
                        p3.Mem(
                            muir.Address(muir.Symbol(symbol), 0),
                            muir.Width.I64,
                        ),
                    )
                ],
            )
        ],
        {"entry"},
    )


class SystemSurfaceTests(unittest.TestCase):
    def test_resolves_known_system_descriptor_generically(self):
        fn = function_loading("__mm_sys_fence")
        program = Program()
        program.add_function(fn)

        result = resolve_system_surface(program, (fn,))

        self.assertEqual(result.referenced, ("__mm_sys_fence",))
        self.assertEqual(result.added, ("__mm_sys_fence",))
        self.assertEqual(result.unsupported, ())
        self.assertIn("__mm_sys_fence", program.symbol_addresses)
        self.assertIn("__mm_sys_fence", program.host_services)

    def test_reports_unknown_system_descriptor_without_guessing_semantics(self):
        fn = function_loading("__mm_sys_future_semantics")
        program = Program()
        program.add_function(fn)

        result = resolve_system_surface(program, (fn,))

        self.assertEqual(result.added, ())
        self.assertEqual(result.unsupported, ("__mm_sys_future_semantics",))
        self.assertNotIn("__mm_sys_future_semantics", program.symbol_addresses)

    def test_non_system_symbols_are_ignored(self):
        fn = function_loading("ordinary_symbol")
        self.assertEqual(referenced_system_symbols((fn,)), ())
        self.assertIsNone(system_op_from_symbol("ordinary_symbol"))


if __name__ == "__main__":
    unittest.main()
