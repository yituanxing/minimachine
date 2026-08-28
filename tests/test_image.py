import unittest

from src.minimachine.image import (
    ImageObject,
    ModuleImage,
    Relocation,
    SymbolExpr,
    install_module_image,
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


if __name__ == "__main__":
    unittest.main()
