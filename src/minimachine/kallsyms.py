from __future__ import annotations

import struct
from dataclasses import dataclass

from .vm import MASK64, Program, VMError


_KALLSYMS_REQUIRED = {
    "kallsyms_num_syms",
    "kallsyms_offsets",
    "kallsyms_names",
    "kallsyms_relative_base",
    "kallsyms_token_table",
    "kallsyms_token_index",
    "kallsyms_markers",
    "kallsyms_seqs_of_names",
}


@dataclass(frozen=True)
class KallsymsImage:
    symbols: tuple[tuple[str, int], ...]
    relative_base: int
    offsets: bytes
    names: bytes
    markers: bytes
    token_table: bytes
    token_index: bytes
    seqs_of_names: bytes


def _encode_len(length: int) -> bytes:
    if length <= 0:
        raise VMError("kallsyms symbol has empty encoded name")
    if length <= 0x7F:
        return bytes((length,))
    if length > 0x3FFF:
        raise VMError(f"kallsyms symbol encoding too large: {length}")
    return bytes(((length & 0x7F) | 0x80, (length >> 7) & 0x7F))


def _function_symbols(program: Program) -> tuple[tuple[str, int], ...]:
    entries: list[tuple[str, int]] = []
    for name, linked in program.functions.items():
        if not linked.function.blocks:
            continue
        entry = program.block_code[(name, linked.function.blocks[0].label)]
        entries.append((name, entry & MASK64))
    entries.sort(key=lambda item: (item[1], item[0]))
    return tuple(entries)


def build_p3_kallsyms(program: Program) -> KallsymsImage:
    """Build Linux 6.6-style base-relative kallsyms data for P3 code.

    The native Linux build normally synthesizes these tables from the final
    ELF symbol map in scripts/kallsyms.c. MiniMachine has a different final
    code-address domain, so generate the same runtime data contract from the
    final linked P3 function entry addresses instead.
    """
    symbols = _function_symbols(program)
    if not symbols:
        raise VMError("cannot build kallsyms without P3 functions")

    relative_base = min(address for _name, address in symbols)

    offsets = bytearray()
    for name, address in symbols:
        offset = address - relative_base
        if offset < 0 or offset > 0xFFFFFFFF:
            raise VMError(
                "P3 kallsyms code span exceeds base-relative u32 range: "
                f"{name}=0x{address:x} base=0x{relative_base:x}"
            )
        offsets.extend(struct.pack("<I", offset))

    # Use a legal, deliberately simple token table: each byte that occurs in
    # a symbol expands to itself. This forgoes build-time compression but keeps
    # Linux's normal kallsyms decoder and lookup implementation unchanged.
    used = {ord("t")}
    encoded_names: list[bytes] = []
    for name, _address in symbols:
        raw = ("t" + name).encode("utf-8")
        if any(byte == 0 for byte in raw):
            raise VMError(f"NUL in P3 kallsyms symbol name: {name!r}")
        used.update(raw)
        encoded_names.append(raw)

    token_table = bytearray(b"\0")
    token_offsets = [0] * 256
    for byte in sorted(used):
        token_offsets[byte] = len(token_table)
        token_table.extend((byte, 0))

    token_index = bytearray()
    for offset in token_offsets:
        if offset > 0xFFFF:
            raise VMError("P3 kallsyms token table exceeds u16 index range")
        token_index.extend(struct.pack("<H", offset))

    names = bytearray()
    marker_values: list[int] = []
    for index, raw in enumerate(encoded_names):
        if index % 256 == 0:
            marker_values.append(len(names))
        names.extend(_encode_len(len(raw)))
        names.extend(raw)

    markers = b"".join(struct.pack("<I", value) for value in marker_values)

    # Linux 6.6 stores the original address-order sequence number as a
    # 24-bit big-endian value in a table sorted by symbol name.
    if len(symbols) > 0xFFFFFF:
        raise VMError("too many P3 symbols for Linux kallsyms sequence table")
    name_order = sorted(range(len(symbols)), key=lambda index: symbols[index][0])
    seqs = bytearray()
    for index in name_order:
        seqs.extend(((index >> 16) & 0xFF, (index >> 8) & 0xFF, index & 0xFF))

    return KallsymsImage(
        symbols=symbols,
        relative_base=relative_base,
        offsets=bytes(offsets),
        names=bytes(names),
        markers=bytes(markers),
        token_table=bytes(token_table),
        token_index=bytes(token_index),
        seqs_of_names=bytes(seqs),
    )


def install_p3_kallsyms(
    program: Program,
    *,
    external_data: set[str] | frozenset[str] | tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Install generated kallsyms globals when the linked Linux image needs them."""
    required = set(external_data)
    if "kallsyms_num_syms" not in required:
        return ()

    missing = _KALLSYMS_REQUIRED - set(program.symbol_addresses)
    if not missing:
        return tuple(sorted(_KALLSYMS_REQUIRED))

    image = build_p3_kallsyms(program)
    payloads = (
        ("kallsyms_num_syms", struct.pack("<I", len(image.symbols)), 4),
        ("kallsyms_offsets", image.offsets, 4),
        ("kallsyms_names", image.names, 1),
        ("kallsyms_relative_base", struct.pack("<Q", image.relative_base), 8),
        ("kallsyms_token_table", image.token_table, 1),
        ("kallsyms_token_index", image.token_index, 2),
        ("kallsyms_markers", image.markers, 4),
        ("kallsyms_seqs_of_names", image.seqs_of_names, 1),
    )

    installed: list[str] = []
    for name, data, align in payloads:
        if name in program.symbol_addresses:
            continue
        program.define_data_symbol(name, data, align=align)
        installed.append(name)
    return tuple(installed)
