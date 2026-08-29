import struct
import unittest

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.kallsyms import build_p3_kallsyms, install_p3_kallsyms
from src.minimachine.lower_p3 import lower_function
from src.minimachine.vm import Program


def machine(function: muir.Function):
    expanded, _ = expand_function(function)
    return lower_function(expanded)


def decode_name(image, index: int) -> str:
    marker_index = index >> 8
    marker = struct.unpack_from("<I", image.markers, marker_index * 4)[0]
    off = marker
    for _ in range(index & 0xFF):
        first = image.names[off]
        if first & 0x80:
            length = (first & 0x7F) | (image.names[off + 1] << 7)
            off += 2 + length
        else:
            off += 1 + first

    first = image.names[off]
    if first & 0x80:
        length = (first & 0x7F) | (image.names[off + 1] << 7)
        data = image.names[off + 2 : off + 2 + length]
    else:
        length = first
        data = image.names[off + 1 : off + 1 + length]

    out = bytearray()
    for token in data:
        token_off = struct.unpack_from("<H", image.token_index, token * 2)[0]
        end = image.token_table.index(0, token_off)
        out.extend(image.token_table[token_off:end])
    return out[1:].decode("utf-8")


class KallsymsTests(unittest.TestCase):
    def program(self):
        alpha = muir.Function(
            "alpha",
            [muir.Block("entry", [muir.Ret(None)])],
            set(),
        )
        zeta = muir.Function(
            "zeta_long_symbol_name",
            [muir.Block("entry", [muir.Ret(None)])],
            set(),
        )
        return Program([machine(alpha), machine(zeta)])

    def test_build_p3_kallsyms_round_trip(self):
        program = self.program()
        image = build_p3_kallsyms(program)

        self.assertEqual(len(image.symbols), 2)
        self.assertEqual(
            [decode_name(image, i) for i in range(len(image.symbols))],
            [name for name, _address in image.symbols],
        )

        for index, (_name, address) in enumerate(image.symbols):
            offset = struct.unpack_from("<I", image.offsets, index * 4)[0]
            self.assertEqual(image.relative_base + offset, address)

        seqs = [
            int.from_bytes(image.seqs_of_names[i : i + 3], "big")
            for i in range(0, len(image.seqs_of_names), 3)
        ]
        self.assertEqual(
            [image.symbols[index][0] for index in seqs],
            sorted(name for name, _address in image.symbols),
        )

    def test_install_only_when_linux_requests_kallsyms(self):
        program = self.program()
        self.assertEqual(install_p3_kallsyms(program, external_data=()), ())

        installed = install_p3_kallsyms(
            program,
            external_data=("kallsyms_num_syms",),
        )
        self.assertIn("kallsyms_num_syms", installed)
        self.assertIn("kallsyms_offsets", installed)
        count_addr = program.symbol_addresses["kallsyms_num_syms"]
        self.assertEqual(program.initial_memory.read(count_addr, 32), 2)


if __name__ == "__main__":
    unittest.main()
