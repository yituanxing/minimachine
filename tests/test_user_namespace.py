import unittest

from src.minimachine import muir, p3
from src.minimachine.image import ImageObject, ModuleImage, Relocation, SymbolExpr
from src.minimachine.user_bundle import (
    namespace_user_program,
    rebase_user_program_namespace,
)
from src.minimachine.user_image import UserProgramImage


class UserNamespaceTests(unittest.TestCase):
    def test_rebase_moves_functions_data_externals_and_relocations(self):
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
                            muir.Target(label="entry"),
                            muir.Target(label="entry"),
                        )
                    ],
                )
            ],
            set(),
        )
        main = p3.Function(
            "main",
            [
                p3.Block(
                    "entry",
                    [
                        p3.Br(
                            muir.Width.I8,
                            muir.Cond.EQ,
                            muir.Imm(0),
                            muir.Imm(0),
                            muir.Target(symbol="helper"),
                            muir.Target(symbol="helper"),
                        )
                    ],
                )
            ],
            set(),
        )
        image = ModuleImage(
            objects=(
                ImageObject(
                    "global",
                    "ptr",
                    b"\0" * 8,
                    8,
                    None,
                    False,
                    (Relocation(0, 8, SymbolExpr("helper")),),
                ),
            ),
            aliases=(),
            external_data=("environ",),
            external_functions=("write",),
            skipped_linker_metadata=(),
        )
        base = UserProgramImage(
            "main",
            (main, helper),
            image,
            "linux-main",
            (),
        )
        named = namespace_user_program(
            base,
            internal_prefix="__mm_user_",
            external_prefix="__mm_user_ext_",
        )
        rebased = rebase_user_program_namespace(
            named,
            namespace="user_task_b91880",
        )

        self.assertEqual(rebased.entry, "__mm_user_task_b91880_main")
        self.assertEqual(
            {fn.name for fn in rebased.functions},
            {
                "__mm_user_task_b91880_main",
                "__mm_user_task_b91880_helper",
            },
        )
        self.assertEqual(
            rebased.image.objects[0].name,
            "__mm_user_task_b91880_global",
        )
        self.assertEqual(
            rebased.image.objects[0].relocations[0].target.symbol,
            "__mm_user_task_b91880_helper",
        )
        self.assertEqual(
            rebased.image.external_data,
            ("__mm_user_task_b91880_ext_environ",),
        )
        self.assertEqual(
            rebased.image.external_functions,
            ("__mm_user_task_b91880_ext_write",),
        )

    def test_distinct_program_namespaces_do_not_collide(self):
        main = p3.Function(
            "main",
            [
                p3.Block(
                    "entry",
                    [
                        p3.Br(
                            muir.Width.I8,
                            muir.Cond.EQ,
                            muir.Imm(0),
                            muir.Imm(0),
                            muir.Target(symbol="helper"),
                            muir.Target(symbol="helper"),
                        )
                    ],
                )
            ],
            set(),
        )
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
                            muir.Target(label="entry"),
                            muir.Target(label="entry"),
                        )
                    ],
                )
            ],
            set(),
        )
        program = UserProgramImage(
            "main",
            (main, helper),
            ModuleImage(
                objects=(),
                aliases=(),
                external_data=(),
                external_functions=("write",),
                skipped_linker_metadata=(),
            ),
            "linux-main",
            (),
        )

        busybox = namespace_user_program(
            program,
            internal_prefix="__mm_busybox_",
            external_prefix="__mm_busybox_ext_",
        )
        lua = namespace_user_program(
            program,
            internal_prefix="__mm_lua54_",
            external_prefix="__mm_lua54_ext_",
        )

        busybox_names = {fn.name for fn in busybox.functions}
        lua_names = {fn.name for fn in lua.functions}
        self.assertTrue(busybox_names.isdisjoint(lua_names))
        self.assertEqual(busybox.entry, "__mm_busybox_main")
        self.assertEqual(lua.entry, "__mm_lua54_main")
        self.assertEqual(
            busybox.image.external_functions,
            ("__mm_busybox_ext_write",),
        )
        self.assertEqual(
            lua.image.external_functions,
            ("__mm_lua54_ext_write",),
        )


if __name__ == "__main__":
    unittest.main()
