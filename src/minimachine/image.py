from __future__ import annotations

from dataclasses import dataclass
import re
import struct
from typing import Iterable

from . import muir
from .layout import DataLayout, LayoutError
from .legalize import _const_gep_value, _split_top_commas, _split_typed_value
from .vm import MASK64, Program, VMError


class ImageError(ValueError):
    pass


@dataclass(frozen=True)
class SymbolExpr:
    symbol: str
    addend: int = 0


@dataclass(frozen=True)
class BlockExpr:
    function: str
    label: str
    addend: int = 0


@dataclass(frozen=True)
class Relocation:
    offset: int
    size: int
    target: SymbolExpr | BlockExpr


@dataclass(frozen=True)
class ImageObject:
    name: str
    ty: str
    data: bytes
    align: int
    section: str | None
    constant: bool
    relocations: tuple[Relocation, ...]

    @property
    def size(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class ImageAlias:
    name: str
    target: SymbolExpr


@dataclass(frozen=True)
class ModuleImage:
    objects: tuple[ImageObject, ...]
    aliases: tuple[ImageAlias, ...]
    external_data: tuple[str, ...]
    external_functions: tuple[str, ...]
    skipped_linker_metadata: tuple[str, ...]
    undef_bytes: int = 0

    @property
    def relocation_count(self) -> int:
        return sum(len(obj.relocations) for obj in self.objects)

    @property
    def byte_size(self) -> int:
        return sum(obj.size for obj in self.objects)


_GLOBAL_RE = re.compile(r"^@([-A-Za-z$._0-9]+)\s*=\s*(.+)$")
_FUNCTION_DECL_RE = re.compile(
    r"^declare\s+.*?@([-A-Za-z$._0-9]+)\s*\("
)
_BLOCKADDRESS_RE = re.compile(
    r"^blockaddress\s*\(\s*@([-A-Za-z$._0-9]+)\s*,\s*%([-A-Za-z$._0-9]+)\s*\)$"
)


def _decode_cstring(text: str) -> bytes:
    if not (text.startswith('c"') and text.endswith('"')):
        raise ImageError(f"invalid LLVM cstring: {text[:80]}")
    body = text[2:-1]
    out = bytearray()
    i = 0
    while i < len(body):
        ch = body[i]
        if ch != "\\":
            out.extend(ch.encode("utf-8"))
            i += 1
            continue
        if i + 2 < len(body) and all(
            c in "0123456789abcdefABCDEF" for c in body[i + 1 : i + 3]
        ):
            out.append(int(body[i + 1 : i + 3], 16))
            i += 3
            continue
        if i + 1 >= len(body):
            raise ImageError("trailing backslash in LLVM cstring")
        # Be permissive for the printable escaped forms accepted by the LLVM
        # assembler, although kernel strings normally use hex escapes.
        out.extend(body[i + 1].encode("utf-8"))
        i += 2
    return bytes(out)


def _write_int(data: bytearray, offset: int, size: int, value: int) -> None:
    if size < 1:
        raise ImageError("zero-sized scalar")
    mask = (1 << (size * 8)) - 1
    value &= mask
    data[offset : offset + size] = value.to_bytes(size, "little")


def _symbol_token(value: str) -> str | None:
    m = re.fullmatch(r"@([-A-Za-z$._0-9]+)", value.strip())
    return m.group(1) if m else None


def _cast_inner(value: str, opname: str) -> tuple[str, str] | None:
    prefix = opname + " ("
    raw = value.strip()
    if not raw.startswith(prefix) or not raw.endswith(")"):
        return None
    body = raw[len(prefix) : -1].strip()

    # Find the top-level " to " separator.
    depth = 0
    for i in range(len(body) - 3):
        c = body[i]
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        elif depth == 0 and body.startswith(" to ", i):
            return body[:i].strip(), body[i + 4 :].strip()
    raise ImageError(f"cannot split constant cast: {value}")


def _eval_scalar(
    layout: DataLayout,
    ty: str,
    value: str,
) -> int | SymbolExpr | BlockExpr:
    raw = value.strip()

    if raw in {"null", "zeroinitializer", "false"}:
        return 0
    if raw == "true":
        return 1
    if raw == "undef":
        return 0
    if raw == "poison":
        raise ImageError("live poison global scalar initializer")

    symbol = _symbol_token(raw)
    if symbol is not None:
        return SymbolExpr(symbol)

    bm = _BLOCKADDRESS_RE.fullmatch(raw)
    if bm:
        return BlockExpr(bm.group(1), bm.group(2))

    if raw.startswith("getelementptr"):
        try:
            lowered = _const_gep_value(raw, layout)
        except (ValueError, LayoutError) as exc:
            raise ImageError(str(exc)) from exc
        if isinstance(lowered, muir.Symbol):
            return SymbolExpr(lowered.name)
        if isinstance(lowered, muir.Reloc):
            return SymbolExpr(lowered.symbol, lowered.addend)
        if isinstance(lowered, muir.Imm):
            return lowered.value
        raise ImageError(f"unsupported global GEP result: {lowered!r}")

    cast = _cast_inner(raw, "inttoptr")
    if cast is not None:
        source, _dst_ty = cast
        src_ty, src_value = _split_typed_value(source)
        result = _eval_scalar(layout, src_ty, src_value)
        if not isinstance(result, int):
            raise ImageError("inttoptr source is not an integer constant")
        return result

    cast = _cast_inner(raw, "ptrtoint")
    if cast is not None:
        source, _dst_ty = cast
        src_ty, src_value = _split_typed_value(source)
        if not src_ty.startswith("ptr"):
            raise ImageError(f"ptrtoint source is not ptr: {source}")
        return _eval_scalar(layout, src_ty, src_value)

    for opname in ("bitcast", "addrspacecast"):
        cast = _cast_inner(raw, opname)
        if cast is not None:
            source, _dst_ty = cast
            src_ty, src_value = _split_typed_value(source)
            return _eval_scalar(layout, src_ty, src_value)

    # LLVM integer constants are decimal (possibly signed). Python's int also
    # handles an optional leading +/-. Hex constants are accepted for
    # robustness with hand-written IR.
    try:
        return int(raw, 0)
    except ValueError:
        pass

    # Floating-point globals are rare in the kernel but zero values can still
    # appear in generic data structures. Support standard decimal and LLVM
    # hexadecimal bit-pattern spelling for the common scalar widths.
    if ty in {"half", "float", "double"}:
        if raw.startswith("0x"):
            return int(raw, 16)
        try:
            f = float(raw)
        except ValueError as exc:
            raise ImageError(f"unsupported scalar initializer {ty} {raw}") from exc
        if ty == "float":
            return int.from_bytes(struct.pack("<f", f), "little")
        if ty == "double":
            return int.from_bytes(struct.pack("<d", f), "little")
        # Python has no native half pack in every supported runtime; struct
        # format 'e' is available on current CPython.
        return int.from_bytes(struct.pack("<e", f), "little")

    raise ImageError(f"unsupported scalar initializer {ty} {raw[:120]}")


def _encode_initializer(
    layout: DataLayout,
    ty: str,
    value: str,
    data: bytearray,
    base: int,
    relocs: list[Relocation],
    stats: dict[str, int],
) -> None:
    ty = ty.strip()
    raw = value.strip()
    info = layout.info(ty)

    if raw == "zeroinitializer":
        return
    if raw == "undef":
        stats["undef_bytes"] = stats.get("undef_bytes", 0) + info.size
        return
    if raw == "poison":
        raise ImageError(f"live poison initializer for {ty}")

    if info.fields is not None:
        body = raw
        if body.startswith("<{") and body.endswith("}>"):
            body = body[2:-2].strip()
        elif body.startswith("{") and body.endswith("}"):
            body = body[1:-1].strip()
        else:
            raise ImageError(f"expected struct initializer for {ty}: {raw[:120]}")

        fields = _split_top_commas(body) if body else []
        if len(fields) != len(info.fields):
            raise ImageError(
                f"struct initializer field count mismatch for {ty}: "
                f"{len(fields)} != {len(info.fields)}"
            )
        assert info.field_offsets is not None
        for index, (field_ty, field_text) in enumerate(
            zip(info.fields, fields)
        ):
            actual_ty, actual_value = _split_typed_value(field_text)
            if actual_ty != field_ty:
                raise ImageError(
                    f"struct field type mismatch for {ty}[{index}]: "
                    f"{actual_ty} != {field_ty}"
                )
            _encode_initializer(
                layout,
                field_ty,
                actual_value,
                data,
                base + info.field_offsets[index],
                relocs,
                stats,
            )
        return

    if info.element is not None:
        if raw.startswith('c"'):
            if info.element != "i8":
                raise ImageError(f"cstring used for non-i8 aggregate {ty}")
            encoded = _decode_cstring(raw)
            if len(encoded) != info.size:
                raise ImageError(
                    f"cstring size mismatch for {ty}: "
                    f"{len(encoded)} != {info.size}"
                )
            data[base : base + info.size] = encoded
            return

        if raw.startswith("[") and raw.endswith("]"):
            body = raw[1:-1].strip()
        elif raw.startswith("<") and raw.endswith(">"):
            body = raw[1:-1].strip()
        else:
            raise ImageError(f"expected array/vector initializer for {ty}: {raw[:120]}")

        items = _split_top_commas(body) if body else []
        if info.count is None or len(items) != info.count:
            raise ImageError(
                f"aggregate initializer count mismatch for {ty}: "
                f"{len(items)} != {info.count}"
            )
        elem_info = layout.info(info.element)
        stride = ((elem_info.size + elem_info.align - 1) // elem_info.align) * elem_info.align
        for index, item in enumerate(items):
            actual_ty, actual_value = _split_typed_value(item)
            if actual_ty != info.element:
                raise ImageError(
                    f"aggregate element type mismatch for {ty}[{index}]: "
                    f"{actual_ty} != {info.element}"
                )
            _encode_initializer(
                layout,
                info.element,
                actual_value,
                data,
                base + stride * index,
                relocs,
                stats,
            )
        return

    scalar = _eval_scalar(layout, ty, raw)
    if isinstance(scalar, int):
        _write_int(data, base, info.size, scalar)
        return

    if info.size < 1 or info.size > 16:
        raise ImageError(
            f"relocation field width unsupported for {ty}: {info.size}"
        )
    relocs.append(Relocation(base, info.size, scalar))


def _parse_section(parts: list[str]) -> str | None:
    for part in parts:
        m = re.fullmatch(r'section\s+"([^"]+)"', part.strip())
        if m:
            return m.group(1)
    return None


def _parse_align(parts: list[str], default: int) -> int:
    for part in parts:
        m = re.fullmatch(r"align\s+(\d+)", part.strip())
        if m:
            return int(m.group(1))
    return default


def _find_top_level_keyword(
    text: str,
    keywords: set[str],
) -> tuple[str, int, int] | None:
    in_string = False
    escape = False
    depth = 0
    i = 0
    while i < len(text):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            i += 1
            continue

        if c == '"':
            in_string = True
            i += 1
            continue
        if c in "([{<":
            depth += 1
            i += 1
            continue
        if c in ")]}>":
            depth = max(0, depth - 1)
            i += 1
            continue

        if depth == 0 and (c.isalpha() or c == "_"):
            j = i + 1
            while j < len(text) and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            if word in keywords:
                return word, i, j
            i = j
            continue
        i += 1
    return None


def _parse_alias(name: str, rhs: str) -> ImageAlias:
    m = re.search(r"\balias\b", rhs)
    if not m:
        raise ImageError(f"alias keyword missing: @{name}")
    parts = _split_top_commas(rhs[m.end() :].strip())
    if len(parts) != 2:
        raise ImageError(f"cannot parse alias @{name}: {rhs}")
    _alias_ty = parts[0].strip()
    ptr_ty, ptr_value = _split_typed_value(parts[1])
    if not ptr_ty.startswith("ptr"):
        raise ImageError(f"alias target is not pointer: @{name}")
    target = _eval_scalar(DataLayout({}), ptr_ty, ptr_value)
    if not isinstance(target, SymbolExpr):
        raise ImageError(f"alias target is not direct symbol: @{name}")
    return ImageAlias(name, target)


def _logical_module_records(text: str):
    """Yield top-level LLVM module records, preserving multiline globals.

    llvm-dis may emit long global initializers (especially cstrings and
    aggregate constants) across physical lines. Split only when not inside a
    quoted string or nested delimiter.
    """
    buf: list[str] = []
    depth = 0
    in_string = False
    escape = False

    def feed(line: str) -> None:
        nonlocal depth, in_string, escape
        for ch in line:
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch in "([{<":
                depth += 1
            elif ch in ")]}>":
                depth = max(0, depth - 1)

    for raw in text.splitlines():
        if not buf:
            stripped = raw.lstrip()
            # Only aggregate top-level records we care about. Function bodies
            # and metadata remain one physical line at a time.
            if not (
                stripped.startswith("@")
                or stripped.startswith("declare ")
            ):
                yield raw
                continue

        buf.append(raw)
        feed(raw)

        if not in_string and depth == 0:
            yield "\n".join(buf)
            buf.clear()

    if buf:
        yield "\n".join(buf)


def parse_module_image(text: str) -> ModuleImage:
    layout = DataLayout.from_module(text)
    objects: list[ImageObject] = []
    aliases: list[ImageAlias] = []
    external_data: set[str] = set()
    external_functions: set[str] = set()
    skipped: list[str] = []
    stats: dict[str, int] = {}

    for raw_line in _logical_module_records(text):
        line = raw_line.strip()
        if not line:
            continue

        fm = _FUNCTION_DECL_RE.match(line)
        if fm:
            external_functions.add(fm.group(1))
            continue

        gm = _GLOBAL_RE.match(line)
        if not gm:
            continue
        name, rhs = gm.groups()

        alias_kw = _find_top_level_keyword(rhs, {"alias"})
        if alias_kw is not None:
            aliases.append(_parse_alias(name, rhs))
            continue
        if _find_top_level_keyword(rhs, {"ifunc"}) is not None:
            raise ImageError(f"ifunc is not supported in image: @{name}")

        kind_match = _find_top_level_keyword(rhs, {"global", "constant"})
        if kind_match is None:
            # Metadata-like global value forms that are not runtime storage.
            continue
        kind, kind_start, kind_end = kind_match

        first = rhs.split(None, 1)[0] if rhs.split() else ""
        if first in {"external", "extern_weak"}:
            external_data.add(name)
            continue

        if name in {"llvm.compiler.used", "llvm.used"}:
            skipped.append(name)
            continue

        tail = rhs[kind_end:].strip()
        parts = _split_top_commas(tail)
        if not parts:
            raise ImageError(f"global @{name} has no initializer")

        try:
            ty, initializer = _split_typed_value(parts[0])
            info = layout.info(ty)
        except (ValueError, LayoutError) as exc:
            raise ImageError(f"@{name}: {exc}") from exc

        data = bytearray(info.size)
        relocs: list[Relocation] = []
        try:
            _encode_initializer(
                layout,
                ty,
                initializer,
                data,
                0,
                relocs,
                stats,
            )
        except (ValueError, LayoutError, ImageError) as exc:
            raise ImageError(f"@{name}: {exc}") from exc

        objects.append(
            ImageObject(
                name=name,
                ty=ty,
                data=bytes(data),
                align=_parse_align(parts[1:], info.align),
                section=_parse_section(parts[1:]),
                constant=(kind == "constant"),
                relocations=tuple(relocs),
            )
        )

    return ModuleImage(
        objects=tuple(objects),
        aliases=tuple(aliases),
        external_data=tuple(sorted(external_data)),
        external_functions=tuple(sorted(external_functions)),
        skipped_linker_metadata=tuple(skipped),
        undef_bytes=stats.get("undef_bytes", 0),
    )


def _resolve_target(program: Program, target: SymbolExpr | BlockExpr) -> int:
    if isinstance(target, SymbolExpr):
        try:
            base = program.symbol_addresses[target.symbol]
        except KeyError as exc:
            raise VMError(f"unresolved image symbol: {target.symbol}") from exc
        return (base + target.addend) & MASK64

    try:
        base = program.block_code[(target.function, target.label)]
    except KeyError as exc:
        raise VMError(
            f"unresolved image blockaddress: "
            f"{target.function}:{target.label}"
        ) from exc
    return (base + target.addend) & MASK64


def install_module_image(
    program: Program,
    image: ModuleImage,
    *,
    external_symbols: dict[str, int] | None = None,
    symbol_aliases: dict[str, str] | None = None,
) -> None:
    external_symbols = external_symbols or {}
    symbol_aliases = symbol_aliases or {}

    for name, address in external_symbols.items():
        if name in program.symbol_addresses:
            if program.symbol_addresses[name] != (address & MASK64):
                raise VMError(f"external symbol conflicts with existing symbol: {name}")
            continue
        program.symbol_addresses[name] = address & MASK64

    # Allocate all storage before applying relocations so self- and
    # cross-references are resolvable.
    object_addresses: dict[str, int] = {}
    for obj in image.objects:
        address = program.define_data_symbol(
            obj.name,
            obj.data,
            align=obj.align,
        )
        object_addresses[obj.name] = address

    # Aliases may chain. Resolve until fixed point.  Linker-defined aliases
    # (for example Linux's jiffies = jiffies_64) use the same relocation
    # machinery as LLVM aliases, but are supplied by the target link contract.
    pending = list(image.aliases)
    pending.extend(
        ImageAlias(name, SymbolExpr(target))
        for name, target in sorted(symbol_aliases.items())
    )
    while pending:
        next_pending: list[ImageAlias] = []
        progress = False
        for alias in pending:
            target = alias.target
            if target.symbol not in program.symbol_addresses:
                next_pending.append(alias)
                continue
            if alias.name in program.symbol_addresses:
                raise VMError(f"duplicate image alias symbol: {alias.name}")
            program.symbol_addresses[alias.name] = (
                program.symbol_addresses[target.symbol] + target.addend
            ) & MASK64
            progress = True
        if not progress:
            missing = ", ".join(
                f"{a.name}->{a.target.symbol}" for a in next_pending[:8]
            )
            raise VMError(f"unresolved image aliases: {missing}")
        pending = next_pending

    for obj in image.objects:
        base = object_addresses[obj.name]
        for reloc in obj.relocations:
            value = _resolve_target(program, reloc.target)
            mask = (1 << (reloc.size * 8)) - 1
            raw = (value & mask).to_bytes(reloc.size, "little")
            for i, byte in enumerate(raw):
                program.initial_memory.write(
                    base + reloc.offset + i,
                    8,
                    byte,
                )
