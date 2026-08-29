import unittest

from src.minimachine import muir, p3
from src.minimachine.user_image import (
    FORMAT,
    UserImageError,
    dumps_function,
    function_to_obj,
    loads_function,
)


class UserImageTests(unittest.TestCase):
    def test_strict_p3_function_round_trips(self):
        fn = p3.Function(
            "user_init",
            [
                p3.Block(
                    "entry",
                    [
                        p3.Mov(
                            muir.Width.I64,
                            muir.Slot("sp"),
                            muir.Special.SP,
                        ),
                        p3.Mov(
                            muir.Width.I8,
                            p3.Mem(muir.Address(muir.Slot("sp"), -1), muir.Width.I8),
                            muir.Imm(0x24),
                        ),
                        p3.Sub(
                            muir.Width.I64,
                            muir.Slot("next"),
                            muir.Slot("sp"),
                            muir.Imm(8),
                        ),
                        p3.Br(
                            muir.Width.I64,
                            muir.Cond.EQ,
                            muir.Slot("next"),
                            muir.Reloc("limit", 4),
                            muir.Target(label="done"),
                            muir.Target(symbol="__mm_user_syscall"),
                        ),
                    ],
                ),
                p3.Block(
                    "done",
                    [
                        p3.Br(
                            muir.Width.I64,
                            muir.Cond.EQ,
                            muir.Imm(0),
                            muir.Imm(0),
                            muir.Target(slot=muir.Slot("next")),
                            muir.Target(address=muir.Address(muir.Slot("sp"), 8)),
                        )
                    ],
                ),
            ],
            {"sp", "next"},
        )

        encoded = dumps_function(fn)
        decoded = loads_function(encoded)

        self.assertEqual(decoded, fn)
        self.assertEqual(function_to_obj(fn)["format"], FORMAT)
        self.assertLess(len(encoded), 4096)

    def test_invalid_format_is_rejected(self):
        with self.assertRaises(UserImageError):
            loads_function(b'{"format":"other","function":{}}')

    def test_invalid_json_is_rejected(self):
        with self.assertRaises(UserImageError):
            loads_function(b"not-json")


if __name__ == "__main__":
    unittest.main()
