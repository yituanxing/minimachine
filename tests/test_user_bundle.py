import unittest

from src.minimachine import muir, p3
from src.minimachine.image import (
    ImageAlias,
    ImageObject,
    ModuleImage,
    Relocation,
    SymbolExpr,
)
from src.minimachine.user_bundle import namespace_user_program
from src.minimachine.user_image import UserProgramImage


class UserBundleTests(unittest.TestCase):
    def test_namespace_rewrites_functions_globals_and_externals(self):
        helper = p3.Function(
            "helper",
            [
                p3.Block(
                    "entry",
                    [
                        p3.Br(
                            muir.Width.I8,
                            muir.Cond.EQ,
                            muir.Imm(0),
                            muir.Imm(0),
                            muir.Target(symbol="write"),
                            muir.Target(symbol="write"),
                        )
                    ],
                )
            ],
            set(),
        )
        main = p3.Function(
            "ash_main",
            [
                p3.Block(
                    "entry",
                    [
                        p3.Mov(
                            muir.Width.I64,
                            muir.Slot("p"),
                            muir.Symbol("message"),
                        ),
                        p3.Br(
                            muir.Width.I8,
                            muir.Cond.EQ,
                            muir.Imm(0),
                            muir.Imm(0),
                            muir.Target(symbol="helper"),
                            muir.Target(symbol="helper"),
                        ),
                    ],
                )
            ],
            {"p"},
        )
        image = ModuleImage(
            objects=(
                ImageObject(
                    "message",
                    "[2 x i8]",
                    b"x\0",
                    1,
                    ".rodata",
                    True,
                    (),
                ),
                ImageObject(
                    "message_ptr",
                    "ptr",
                    b"\0" * 8,
                    8,
                    ".data",
                    False,
                    (Relocation(0, 8, SymbolExpr("message")),),
                ),
            ),
            aliases=(ImageAlias("alias", SymbolExpr("message")),),
            external_data=("environ",),
            external_functions=("write",),
            skipped_linker_metadata=(),
        )
        original = UserProgramImage(
            "ash_main",
            (main, helper),
            image,
            "linux-main",
            ("__mm_fmul_64",),
        )
        namespaced = namespace_user_program(original)

        self.assertEqual(namespaced.entry, "__mm_user_ash_main")
        self.assertEqual(
            {fn.name for fn in namespaced.functions},
            {"__mm_user_ash_main", "__mm_user_helper"},
        )
        self.assertEqual(
            {obj.name for obj in namespaced.image.objects},
            {"__mm_user_message", "__mm_user_message_ptr"},
        )
        self.assertEqual(
            namespaced.image.external_functions,
            ("__mm_user_ext_write",),
        )
        self.assertEqual(
            namespaced.image.external_data,
            ("__mm_user_ext_environ",),
        )
        self.assertEqual(
            namespaced.runtime_helpers,
            ("__mm_fmul_64",),
        )


if __name__ == "__main__":
    unittest.main()
