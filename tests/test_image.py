import unittest
from pathlib import Path

from src.minimachine import muir, p3
from src.minimachine.image import (
    ImageObject,
    ModuleImage,
    Relocation,
    SymbolExpr,
    install_module_image,
)
from src.minimachine.linker import (
    BoundarySymbol,
    LinkerContract,
    SectionGroup,
    SemanticBoundary,
)
from src.minimachine.vm import Program


class ImageTests(unittest.TestCase):
    def test_target_symbol_alias_patches_relocation(self):
        image = ModuleImage(
            objects=(
                ImageObject(
                    name="jiffies_64",
                    ty="i64",
                    data=(123).to_bytes(8, "little"),
                    align=8,
                    section=".data",
                    constant=False,
                    relocations=(),
                ),
                ImageObject(
                    name="jiffies_ref",
                    ty="ptr",
                    data=bytes(8),
                    align=8,
                    section=".data",
                    constant=False,
                    relocations=(
                        Relocation(0, 8, SymbolExpr("jiffies")),
                    ),
                ),
            ),
            aliases=(),
            external_data=("jiffies",),
            external_functions=(),
            skipped_linker_metadata=(),
        )

        program = Program()
        install_module_image(
            program,
            image,
            symbol_aliases={"jiffies": "jiffies_64"},
        )

        self.assertEqual(
            program.symbol_addresses["jiffies"],
            program.symbol_addresses["jiffies_64"],
        )
        self.assertEqual(
            program.new_vm().memory.read(
                program.symbol_addresses["jiffies_ref"],
                64,
            ),
            program.symbol_addresses["jiffies_64"],
        )


    def test_linker_contract_groups_sections_and_defines_boundaries(self):
        image = ModuleImage(
            objects=(
                # Deliberately reverse llvm-link order. The target linker
                # contract must still place level 0 before level 1.
                ImageObject(
                    name="init1",
                    ty="ptr",
                    data=bytes(8),
                    align=8,
                    section=".initcall1.init",
                    constant=False,
                    relocations=(),
                ),
                ImageObject(
                    name="init0",
                    ty="ptr",
                    data=bytes(8),
                    align=8,
                    section=".initcall0.init",
                    constant=False,
                    relocations=(),
                ),
                ImageObject(
                    name="boundary_ref",
                    ty="ptr",
                    data=bytes(8),
                    align=8,
                    section=".data",
                    constant=False,
                    relocations=(
                        Relocation(
                            0,
                            8,
                            SymbolExpr("__initcall0_start"),
                        ),
                    ),
                ),
            ),
            aliases=(),
            external_data=("__initcall0_start",),
            external_functions=(),
            skipped_linker_metadata=(),
        )
        contract = LinkerContract(
            aliases={},
            groups=(
                SectionGroup("init0", (".initcall0.init",), 8),
                SectionGroup("init1", (".initcall1.init",), 8),
                SectionGroup(
                    "empty_after_init1",
                    (".missing.init",),
                    8,
                    "init1",
                ),
            ),
            boundaries=(
                BoundarySymbol("__initcall0_start", "init0", "start"),
                BoundarySymbol("__initcall1_start", "init1", "start"),
                BoundarySymbol("__initcall_end", "init1", "end"),
                BoundarySymbol(
                    "__empty_start",
                    "empty_after_init1",
                    "start",
                ),
                BoundarySymbol(
                    "__empty_end",
                    "empty_after_init1",
                    "end",
                ),
            ),
        )

        program = Program()
        install_module_image(
            program,
            image,
            linker_contract=contract,
        )

        self.assertEqual(
            program.symbol_addresses["__initcall0_start"],
            program.symbol_addresses["init0"],
        )
        self.assertEqual(
            program.symbol_addresses["__initcall1_start"],
            program.symbol_addresses["init1"],
        )
        self.assertLess(
            program.symbol_addresses["init0"],
            program.symbol_addresses["init1"],
        )
        self.assertEqual(
            program.symbol_addresses["__initcall_end"],
            program.symbol_addresses["init1"] + 8,
        )
        self.assertEqual(
            program.symbol_addresses["__empty_start"],
            program.symbol_addresses["__initcall_end"],
        )
        self.assertEqual(
            program.symbol_addresses["__empty_end"],
            program.symbol_addresses["__initcall_end"],
        )
        self.assertEqual(
            program.new_vm().memory.read(
                program.symbol_addresses["boundary_ref"],
                64,
            ),
            program.symbol_addresses["init0"],
        )


    def test_linux_contract_keeps_empty_module_version_boundaries(self):
        contract = LinkerContract.load(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "linux-6.6.143-minimachine-linker.json"
        )
        active = contract.active_boundary_symbols({"__param"})
        self.assertIn("__start___modver", active)
        self.assertIn("__stop___modver", active)

        image = ModuleImage(
            objects=(
                ImageObject(
                    name="param",
                    ty="i64",
                    data=bytes(8),
                    align=8,
                    section="__param",
                    constant=False,
                    relocations=(),
                ),
                ImageObject(
                    name="modver_start_ref",
                    ty="ptr",
                    data=bytes(8),
                    align=8,
                    section=".data",
                    constant=False,
                    relocations=(
                        Relocation(0, 8, SymbolExpr("__start___modver")),
                    ),
                ),
            ),
            aliases=(),
            external_data=("__start___modver", "__stop___modver"),
            external_functions=(),
            skipped_linker_metadata=(),
        )
        program = Program()
        install_module_image(program, image, linker_contract=contract)
        self.assertEqual(
            program.symbol_addresses["__start___modver"],
            program.symbol_addresses["__stop___modver"],
        )

    def test_semantic_boundaries_follow_program_layout(self):
        image = ModuleImage(
            objects=(
                ImageObject(
                    name="payload",
                    ty="i64",
                    data=(7).to_bytes(8, "little"),
                    align=8,
                    section=".data",
                    constant=False,
                    relocations=(),
                ),
            ),
            aliases=(),
            external_data=("_sdata", "_end"),
            external_functions=(),
            skipped_linker_metadata=(),
        )
        contract = LinkerContract(
            aliases={},
            groups=(),
            boundaries=(),
            semantic_boundaries=(
                SemanticBoundary("_stext", "code_start"),
                SemanticBoundary("_etext", "code_end"),
                SemanticBoundary("_sdata", "data_start"),
                SemanticBoundary("_end", "data_end"),
            ),
        )

        program = Program(
            [
                p3.Function(
                    "entry",
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
                                ),
                            ],
                        ),
                    ],
                    set(),
                ),
            ]
        )
        install_module_image(program, image, linker_contract=contract)

        self.assertEqual(
            program.symbol_addresses["_stext"],
            min(program.code_block),
        )
        self.assertEqual(
            program.symbol_addresses["_etext"],
            max(program.code_block) + 8,
        )
        self.assertEqual(
            program.symbol_addresses["_sdata"],
            program.symbol_addresses["payload"],
        )
        self.assertEqual(
            program.symbol_addresses["_end"],
            program.symbol_addresses["payload"] + 8,
        )


if __name__ == "__main__":
    unittest.main()
