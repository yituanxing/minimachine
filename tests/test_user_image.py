import unittest

from src.minimachine import muir, p3
from src.minimachine.image import (
    ImageAlias,
    ImageObject,
    ModuleImage,
    Relocation,
    SymbolExpr,
)
from src.minimachine.user_image import (
    BFLT_DATA_ALIGN,
    BFLT_HEADER_SIZE,
    BFLT_MAGIC,
    BFLT_VERSION,
    FORMAT,
    PROGRAM_FORMAT,
    USER_PAYLOAD_MAGIC,
    UserImageError,
    UserProgramImage,
    build_bflt,
    build_bflt_program,
    dumps_function,
    dumps_program,
    extract_bflt_payload,
    function_to_obj,
    loads_function,
    loads_program,
    unpack_user_image,
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

    def test_multi_function_user_program_round_trips(self):
        main = p3.Function(
            "__mm_user_main",
            [
                p3.Block(
                    "entry",
                    [
                        p3.Br(
                            muir.Width.I8,
                            muir.Cond.EQ,
                            muir.Imm(0),
                            muir.Imm(0),
                            muir.Target(symbol="__mm_user_helper"),
                            muir.Target(symbol="__mm_user_helper"),
                        )
                    ],
                )
            ],
            set(),
        )
        helper = p3.Function(
            "__mm_user_helper",
            [
                p3.Block(
                    "entry",
                    [
                        p3.Br(
                            muir.Width.I8,
                            muir.Cond.EQ,
                            muir.Imm(0),
                            muir.Imm(0),
                            muir.Target(symbol="__mm_user_syscall"),
                            muir.Target(symbol="__mm_user_syscall"),
                        )
                    ],
                )
            ],
            set(),
        )
        program = UserProgramImage(main.name, (main, helper))

        encoded = dumps_program(program)
        decoded = loads_program(encoded)
        self.assertEqual(decoded, program)
        self.assertEqual(
            __import__("json").loads(encoded.decode())["format"],
            PROGRAM_FORMAT,
        )

        image = build_bflt_program(
            program,
            stack_size=0x20000,
            compress_payload=True,
        )
        data_start = int.from_bytes(image[12:16], "big")
        unpacked = unpack_user_image(image[BFLT_HEADER_SIZE:data_start])
        self.assertEqual(unpacked, program)

    def test_multi_function_program_round_trips_global_image(self):
        main = p3.Function(
            "__mm_user_main",
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
        image = ModuleImage(
            objects=(
                ImageObject(
                    name="message",
                    ty="[6 x i8]",
                    data=b"hello\0",
                    align=1,
                    section=".rodata",
                    constant=True,
                    relocations=(),
                ),
                ImageObject(
                    name="message_ptr",
                    ty="ptr",
                    data=b"\0" * 8,
                    align=8,
                    section=".data",
                    constant=False,
                    relocations=(
                        Relocation(
                            0,
                            8,
                            SymbolExpr("message"),
                        ),
                    ),
                ),
            ),
            aliases=(
                ImageAlias("message_alias", SymbolExpr("message")),
            ),
            external_data=("environ",),
            external_functions=("write",),
            skipped_linker_metadata=(".llvm_addrsig",),
            undef_bytes=3,
        )
        program = UserProgramImage(main.name, (main,), image)

        encoded = dumps_program(program)
        decoded = loads_program(encoded)
        self.assertEqual(decoded, program)

        wrapped = build_bflt_program(
            program,
            stack_size=0x20000,
            compress_payload=True,
        )
        data_start = int.from_bytes(wrapped[12:16], "big")
        unpacked = unpack_user_image(
            wrapped[BFLT_HEADER_SIZE:data_start]
        )
        self.assertEqual(unpacked, program)

    def test_bflt_wraps_the_same_p3_payload(self):
        fn = p3.Function(
            "init",
            [
                p3.Block(
                    "entry",
                    [
                        p3.Br(
                            muir.Width.I8,
                            muir.Cond.EQ,
                            muir.Imm(0),
                            muir.Imm(0),
                            muir.Target(symbol="__mm_sys_linux_syscall"),
                            muir.Target(symbol="__mm_sys_linux_syscall"),
                        )
                    ],
                )
            ],
            set(),
        )

        image = build_bflt(fn, stack_size=0x20000, ktrace=True)

        self.assertEqual(image[:4], BFLT_MAGIC)
        self.assertEqual(int.from_bytes(image[4:8], "big"), BFLT_VERSION)
        self.assertEqual(int.from_bytes(image[8:12], "big"), BFLT_HEADER_SIZE)
        self.assertEqual(image[BFLT_HEADER_SIZE:BFLT_HEADER_SIZE + 4], USER_PAYLOAD_MAGIC)
        data_start = int.from_bytes(image[12:16], "big")
        self.assertEqual(data_start % BFLT_DATA_ALIGN, 0)
        self.assertEqual(len(image), data_start)
        self.assertEqual(extract_bflt_payload(image), fn)

    def test_bflt_rejects_bad_magic(self):
        fn = p3.Function("init", [p3.Block("entry", [])], set())
        image = bytearray(build_bflt(fn))
        image[0:4] = b"nope"
        with self.assertRaises(UserImageError):
            extract_bflt_payload(bytes(image))

    def test_invalid_format_is_rejected(self):
        with self.assertRaises(UserImageError):
            loads_function(b'{"format":"other","function":{}}')

    def test_invalid_json_is_rejected(self):
        with self.assertRaises(UserImageError):
            loads_function(b"not-json")


if __name__ == "__main__":
    unittest.main()
