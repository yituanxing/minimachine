import unittest

from src.minimachine import muir
from src.minimachine.lower_p3 import MachineLoweringError, lower_function
from src.minimachine.verify import verify_muir, verify_p3


class IRContractTests(unittest.TestCase):
    def test_direct_core_lowers_one_to_one(self):
        a = muir.Slot("a")
        b = muir.Slot("b")
        x = muir.Slot("x")
        fn = muir.Function(
            "f",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Sub(muir.Width.I64, x, a, b),
                        muir.Br(
                            muir.Width.I64,
                            muir.Cond.EQ,
                            x,
                            muir.Imm(0),
                            muir.Target(label="yes"),
                            muir.Target(label="no"),
                        ),
                    ],
                ),
                muir.Block(
                    "yes",
                    [
                        muir.Br(
                            muir.Width.I64,
                            muir.Cond.EQ,
                            muir.Imm(0),
                            muir.Imm(0),
                            muir.Target(label="yes"),
                            muir.Target(label="yes"),
                        )
                    ],
                ),
                muir.Block(
                    "no",
                    [
                        muir.Br(
                            muir.Width.I64,
                            muir.Cond.EQ,
                            muir.Imm(0),
                            muir.Imm(0),
                            muir.Target(label="no"),
                            muir.Target(label="no"),
                        )
                    ],
                ),
            ],
            {"a", "b", "x"},
        )
        verify_muir(fn)
        out = lower_function(fn)
        verify_p3(out)
        self.assertEqual(len(out.blocks[0].instructions), 2)

    def test_helper_cannot_leak_into_p3(self):
        fn = muir.Function(
            "g",
            [
                muir.Block(
                    "entry",
                    [
                        muir.Helper("__mm_mul_i64", (muir.Imm(2), muir.Imm(3)), muir.Slot("x")),
                        muir.Ret(muir.Slot("x")),
                    ],
                )
            ],
            {"x"},
        )
        verify_muir(fn)
        with self.assertRaises(MachineLoweringError):
            lower_function(fn)


if __name__ == "__main__":
    unittest.main()
