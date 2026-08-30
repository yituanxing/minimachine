from __future__ import annotations

from dataclasses import dataclass
import base64
import json
import struct
import zlib

from . import muir, p3
from .image import (
    BlockExpr,
    ImageAlias,
    ImageObject,
    ModuleImage,
    Relocation,
    SymbolExpr,
)

FORMAT = "minimachine-p3-v1"
PROGRAM_FORMAT = "minimachine-p3-program-v1"

BFLT_MAGIC = b"bFLT"
BFLT_VERSION = 4
BFLT_HEADER_SIZE = 64
BFLT_FLAG_KTRACE = 0x0010
USER_PAYLOAD_MAGIC = b"MMP3"
USER_PAYLOAD_VERSION = 1
USER_PAYLOAD_ZLIB_VERSION = 2
USER_PROGRAM_PAYLOAD_VERSION = 3
USER_PROGRAM_PAYLOAD_ZLIB_VERSION = 4
USER_PAYLOAD_HEADER_SIZE = 12
BFLT_DATA_ALIGN = 0x20


class UserImageError(ValueError):
    pass


@dataclass(frozen=True)
class UserProgramImage:
    entry: str
    functions: tuple[p3.Function, ...]
    image: ModuleImage | None = None

    def __post_init__(self) -> None:
        names = tuple(fn.name for fn in self.functions)
        if not names:
            raise UserImageError("P3 user program has no functions")
        if len(set(names)) != len(names):
            raise UserImageError("P3 user program has duplicate function names")
        if self.entry not in names:
            raise UserImageError(
                f"P3 user program entry is missing: {self.entry}"
            )


def _width(bits: int) -> muir.Width:
    try:
        return muir.Width(bits)
    except ValueError as exc:
        raise UserImageError(f"unsupported width: {bits}") from exc


def _value_to_obj(value):
    if isinstance(value, muir.Slot):
        return {"kind": "slot", "name": value.name}
    if isinstance(value, muir.Imm):
        return {"kind": "imm", "value": value.value}
    if isinstance(value, muir.Symbol):
        return {"kind": "symbol", "name": value.name}
    if isinstance(value, muir.Reloc):
        return {"kind": "reloc", "symbol": value.symbol, "addend": value.addend}
    if isinstance(value, muir.BlockAddr):
        return {"kind": "block_addr", "function": value.function, "label": value.label}
    if isinstance(value, muir.Special):
        return {"kind": "special", "name": value.value}
    raise UserImageError(f"unsupported P3 value: {type(value).__name__}")


def _value_from_obj(obj):
    kind = obj.get("kind")
    if kind == "slot":
        return muir.Slot(obj["name"])
    if kind == "imm":
        return muir.Imm(int(obj["value"]))
    if kind == "symbol":
        return muir.Symbol(obj["name"])
    if kind == "reloc":
        return muir.Reloc(obj["symbol"], int(obj.get("addend", 0)))
    if kind == "block_addr":
        return muir.BlockAddr(obj["function"], obj["label"])
    if kind == "special":
        try:
            return muir.Special(obj["name"])
        except ValueError as exc:
            raise UserImageError(f"unsupported special value: {obj['name']}") from exc
    raise UserImageError(f"unsupported P3 value kind: {kind!r}")


def _address_to_obj(address: muir.Address):
    return {"base": _value_to_obj(address.base), "offset": address.offset}


def _address_from_obj(obj):
    return muir.Address(_value_from_obj(obj["base"]), int(obj.get("offset", 0)))


def _operand_to_obj(operand):
    if isinstance(operand, p3.Mem):
        return {
            "kind": "mem",
            "width": operand.width.value,
            "address": _address_to_obj(operand.address),
        }
    return {"kind": "value", "value": _value_to_obj(operand)}


def _operand_from_obj(obj):
    kind = obj.get("kind")
    if kind == "mem":
        return p3.Mem(_address_from_obj(obj["address"]), _width(int(obj["width"])))
    if kind == "value":
        return _value_from_obj(obj["value"])
    raise UserImageError(f"unsupported P3 operand kind: {kind!r}")


def _target_to_obj(target: muir.Target):
    if target.label is not None:
        return {"kind": "label", "label": target.label}
    if target.symbol is not None:
        return {"kind": "symbol", "symbol": target.symbol}
    if target.slot is not None:
        return {"kind": "slot", "slot": target.slot.name}
    if target.address is not None:
        return {"kind": "address", "address": _address_to_obj(target.address)}
    raise UserImageError("invalid empty P3 target")


def _target_from_obj(obj):
    kind = obj.get("kind")
    if kind == "label":
        return muir.Target(label=obj["label"])
    if kind == "symbol":
        return muir.Target(symbol=obj["symbol"])
    if kind == "slot":
        return muir.Target(slot=muir.Slot(obj["slot"]))
    if kind == "address":
        return muir.Target(address=_address_from_obj(obj["address"]))
    raise UserImageError(f"unsupported P3 target kind: {kind!r}")


def _instr_to_obj(inst):
    if isinstance(inst, p3.Mov):
        out = {
            "op": "mov",
            "width": inst.width.value,
            "dst": _operand_to_obj(inst.dst),
            "src": _operand_to_obj(inst.src),
        }
        if inst.extend is not None:
            out["extend"] = inst.extend
        if inst.src_bits is not None:
            out["src_bits"] = inst.src_bits
        return out
    if isinstance(inst, p3.Sub):
        return {
            "op": "sub",
            "width": inst.width.value,
            "dst": inst.dst.name,
            "a": _value_to_obj(inst.a),
            "b": _value_to_obj(inst.b),
        }
    if isinstance(inst, p3.Br):
        return {
            "op": "br",
            "width": inst.width.value,
            "cond": inst.cond.value,
            "a": _value_to_obj(inst.a),
            "b": _value_to_obj(inst.b),
            "true": _target_to_obj(inst.true_target),
            "false": _target_to_obj(inst.false_target),
        }
    raise UserImageError(f"unsupported P3 instruction: {type(inst).__name__}")


def _instr_from_obj(obj):
    op = obj.get("op")
    if op == "mov":
        return p3.Mov(
            _width(int(obj["width"])),
            _operand_from_obj(obj["dst"]),
            _operand_from_obj(obj["src"]),
            obj.get("extend"),
            int(obj["src_bits"]) if obj.get("src_bits") is not None else None,
        )
    if op == "sub":
        return p3.Sub(
            _width(int(obj["width"])),
            muir.Slot(obj["dst"]),
            _value_from_obj(obj["a"]),
            _value_from_obj(obj["b"]),
        )
    if op == "br":
        try:
            cond = muir.Cond(obj["cond"])
        except ValueError as exc:
            raise UserImageError(f"unsupported branch condition: {obj['cond']}") from exc
        return p3.Br(
            _width(int(obj["width"])),
            cond,
            _value_from_obj(obj["a"]),
            _value_from_obj(obj["b"]),
            _target_from_obj(obj["true"]),
            _target_from_obj(obj["false"]),
        )
    raise UserImageError(f"unsupported P3 instruction op: {op!r}")


def function_to_obj(function: p3.Function) -> dict:
    return {
        "format": FORMAT,
        "function": {
            "name": function.name,
            "frame_slots": sorted(function.frame_slots),
            "blocks": [
                {
                    "label": block.label,
                    "instructions": [_instr_to_obj(inst) for inst in block.instructions],
                }
                for block in function.blocks
            ],
        },
    }


def function_from_obj(obj: dict) -> p3.Function:
    if obj.get("format") != FORMAT:
        raise UserImageError(f"unsupported user image format: {obj.get('format')!r}")
    fn = obj.get("function")
    if not isinstance(fn, dict):
        raise UserImageError("missing P3 function payload")
    blocks = [
        p3.Block(
            block["label"],
            [_instr_from_obj(inst) for inst in block.get("instructions", ())],
        )
        for block in fn.get("blocks", ())
    ]
    if not blocks:
        raise UserImageError("P3 user image has no blocks")
    return p3.Function(fn["name"], blocks, set(fn.get("frame_slots", ())))


def dumps_function(function: p3.Function) -> bytes:
    return json.dumps(
        function_to_obj(function),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def loads_function(data: bytes) -> p3.Function:
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UserImageError("invalid P3 user image JSON") from exc
    if not isinstance(obj, dict):
        raise UserImageError("P3 user image must be a JSON object")
    return function_from_obj(obj)

def _image_target_to_obj(target: SymbolExpr | BlockExpr) -> dict:
    if isinstance(target, SymbolExpr):
        return {
            "kind": "symbol",
            "symbol": target.symbol,
            "addend": target.addend,
        }
    if isinstance(target, BlockExpr):
        return {
            "kind": "block",
            "function": target.function,
            "label": target.label,
            "addend": target.addend,
        }
    raise UserImageError(
        f"unsupported image relocation target: {type(target).__name__}"
    )


def _image_target_from_obj(obj) -> SymbolExpr | BlockExpr:
    kind = obj.get("kind")
    if kind == "symbol":
        return SymbolExpr(
            obj["symbol"],
            int(obj.get("addend", 0)),
        )
    if kind == "block":
        return BlockExpr(
            obj["function"],
            obj["label"],
            int(obj.get("addend", 0)),
        )
    raise UserImageError(f"unsupported image target kind: {kind!r}")


def _module_image_to_obj(image: ModuleImage | None):
    if image is None:
        return None
    return {
        "objects": [
            {
                "name": obj.name,
                "ty": obj.ty,
                "data": base64.b64encode(obj.data).decode("ascii"),
                "align": obj.align,
                "section": obj.section,
                "constant": obj.constant,
                "relocations": [
                    {
                        "offset": reloc.offset,
                        "size": reloc.size,
                        "target": _image_target_to_obj(reloc.target),
                    }
                    for reloc in obj.relocations
                ],
            }
            for obj in image.objects
        ],
        "aliases": [
            {
                "name": alias.name,
                "target": _image_target_to_obj(alias.target),
            }
            for alias in image.aliases
        ],
        "external_data": list(image.external_data),
        "external_functions": list(image.external_functions),
        "skipped_linker_metadata": list(image.skipped_linker_metadata),
        "undef_bytes": image.undef_bytes,
    }


def _module_image_from_obj(obj) -> ModuleImage | None:
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise UserImageError("P3 user program image must be an object")
    objects = []
    for raw in obj.get("objects", ()):
        try:
            data = base64.b64decode(raw["data"], validate=True)
        except (KeyError, ValueError) as exc:
            raise UserImageError("invalid user image object data") from exc
        objects.append(
            ImageObject(
                name=raw["name"],
                ty=raw["ty"],
                data=data,
                align=int(raw["align"]),
                section=raw.get("section"),
                constant=bool(raw.get("constant", False)),
                relocations=tuple(
                    Relocation(
                        int(reloc["offset"]),
                        int(reloc["size"]),
                        _image_target_from_obj(reloc["target"]),
                    )
                    for reloc in raw.get("relocations", ())
                ),
            )
        )
    aliases = tuple(
        ImageAlias(
            raw["name"],
            _image_target_from_obj(raw["target"]),
        )
        for raw in obj.get("aliases", ())
    )
    if any(not isinstance(alias.target, SymbolExpr) for alias in aliases):
        raise UserImageError("user image alias target must be a symbol")
    return ModuleImage(
        objects=tuple(objects),
        aliases=aliases,
        external_data=tuple(obj.get("external_data", ())),
        external_functions=tuple(obj.get("external_functions", ())),
        skipped_linker_metadata=tuple(
            obj.get("skipped_linker_metadata", ())
        ),
        undef_bytes=int(obj.get("undef_bytes", 0)),
    )


def program_to_obj(program: UserProgramImage) -> dict:
    return {
        "format": PROGRAM_FORMAT,
        "entry": program.entry,
        "functions": [
            function_to_obj(function)["function"]
            for function in program.functions
        ],
        "image": _module_image_to_obj(program.image),
    }


def program_from_obj(obj: dict) -> UserProgramImage:
    if obj.get("format") != PROGRAM_FORMAT:
        raise UserImageError(
            f"unsupported user program format: {obj.get('format')!r}"
        )
    raw_functions = obj.get("functions")
    if not isinstance(raw_functions, list):
        raise UserImageError("P3 user program functions must be a list")
    functions = tuple(
        function_from_obj({"format": FORMAT, "function": fn})
        for fn in raw_functions
    )
    entry = obj.get("entry")
    if not isinstance(entry, str) or not entry:
        raise UserImageError("P3 user program entry is missing")
    return UserProgramImage(
        entry,
        functions,
        _module_image_from_obj(obj.get("image")),
    )


def dumps_program(program: UserProgramImage) -> bytes:
    return json.dumps(
        program_to_obj(program),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def loads_program(data: bytes) -> UserProgramImage:
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UserImageError("invalid P3 user program JSON") from exc
    if not isinstance(obj, dict):
        raise UserImageError("P3 user program must be a JSON object")
    return program_from_obj(obj)



def _align_up(value: int, align: int) -> int:
    if align <= 0 or (align & (align - 1)):
        raise UserImageError("alignment must be a power of two")
    return (value + align - 1) & ~(align - 1)


def pack_user_payload(
    function: p3.Function,
    *,
    compress: bool = False,
) -> bytes:
    raw = dumps_function(function)
    if compress:
        body = zlib.compress(raw, level=9)
        version = USER_PAYLOAD_ZLIB_VERSION
    else:
        body = raw
        version = USER_PAYLOAD_VERSION
    header = struct.pack(
        ">4sII",
        USER_PAYLOAD_MAGIC,
        version,
        len(body),
    )
    return header + body


def pack_user_program(
    program: UserProgramImage,
    *,
    compress: bool = False,
) -> bytes:
    raw = dumps_program(program)
    if compress:
        body = zlib.compress(raw, level=9)
        version = USER_PROGRAM_PAYLOAD_ZLIB_VERSION
    else:
        body = raw
        version = USER_PROGRAM_PAYLOAD_VERSION
    header = struct.pack(
        ">4sII",
        USER_PAYLOAD_MAGIC,
        version,
        len(body),
    )
    return header + body


def unpack_user_image(data: bytes) -> UserProgramImage:
    if len(data) < USER_PAYLOAD_HEADER_SIZE:
        raise UserImageError("truncated MiniMachine user payload")
    magic, version, size = struct.unpack(
        ">4sII", data[:USER_PAYLOAD_HEADER_SIZE]
    )
    if magic != USER_PAYLOAD_MAGIC:
        raise UserImageError(f"bad MiniMachine user payload magic: {magic!r}")
    supported = {
        USER_PAYLOAD_VERSION,
        USER_PAYLOAD_ZLIB_VERSION,
        USER_PROGRAM_PAYLOAD_VERSION,
        USER_PROGRAM_PAYLOAD_ZLIB_VERSION,
    }
    if version not in supported:
        raise UserImageError(
            f"unsupported MiniMachine user payload version: {version}"
        )
    end = USER_PAYLOAD_HEADER_SIZE + size
    if end > len(data):
        raise UserImageError("truncated MiniMachine user payload body")
    body = data[USER_PAYLOAD_HEADER_SIZE:end]
    if version in {
        USER_PAYLOAD_ZLIB_VERSION,
        USER_PROGRAM_PAYLOAD_ZLIB_VERSION,
    }:
        try:
            body = zlib.decompress(body)
        except zlib.error as exc:
            raise UserImageError(
                "invalid compressed MiniMachine user payload"
            ) from exc
    if version in {USER_PAYLOAD_VERSION, USER_PAYLOAD_ZLIB_VERSION}:
        function = loads_function(body)
        return UserProgramImage(function.name, (function,))
    return loads_program(body)


def unpack_user_payload(data: bytes) -> p3.Function:
    if len(data) < USER_PAYLOAD_HEADER_SIZE:
        raise UserImageError("truncated MiniMachine user payload")
    magic, version, size = struct.unpack(
        ">4sII", data[:USER_PAYLOAD_HEADER_SIZE]
    )
    if magic != USER_PAYLOAD_MAGIC:
        raise UserImageError(f"bad MiniMachine user payload magic: {magic!r}")
    if version not in {USER_PAYLOAD_VERSION, USER_PAYLOAD_ZLIB_VERSION}:
        raise UserImageError(
            f"unsupported MiniMachine user payload version: {version}"
        )
    end = USER_PAYLOAD_HEADER_SIZE + size
    if end > len(data):
        raise UserImageError("truncated MiniMachine user payload body")
    body = data[USER_PAYLOAD_HEADER_SIZE:end]
    if version == USER_PAYLOAD_ZLIB_VERSION:
        try:
            body = zlib.decompress(body)
        except zlib.error as exc:
            raise UserImageError(
                "invalid compressed MiniMachine user payload"
            ) from exc
    return loads_function(body)


def _wrap_bflt_payload(
    payload: bytes,
    *,
    stack_size: int,
    ktrace: bool,
) -> bytes:
    if stack_size <= 0 or stack_size >= (1 << 28):
        raise UserImageError(f"invalid bFLT stack size: {stack_size}")

    entry = BFLT_HEADER_SIZE
    text_end = _align_up(entry + len(payload), BFLT_DATA_ALIGN)
    flags = BFLT_FLAG_KTRACE if ktrace else 0

    header = struct.pack(
        ">4s15I",
        BFLT_MAGIC,
        BFLT_VERSION,
        entry,
        text_end,
        text_end,
        text_end,
        stack_size,
        text_end,
        0,
        flags,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    if len(header) != BFLT_HEADER_SIZE:
        raise AssertionError(f"unexpected bFLT header size: {len(header)}")

    image = header + payload
    image += b"\0" * (text_end - len(image))
    return image


def build_bflt_program(
    program: UserProgramImage,
    *,
    stack_size: int = 64 * 1024,
    ktrace: bool = False,
    compress_payload: bool = False,
) -> bytes:
    payload = pack_user_program(program, compress=compress_payload)
    return _wrap_bflt_payload(
        payload,
        stack_size=stack_size,
        ktrace=ktrace,
    )


def build_bflt(
    function: p3.Function,
    *,
    stack_size: int = 64 * 1024,
    ktrace: bool = False,
    compress_payload: bool = False,
) -> bytes:
    payload = pack_user_payload(function, compress=compress_payload)
    return _wrap_bflt_payload(
        payload,
        stack_size=stack_size,
        ktrace=ktrace,
    )


def extract_bflt_payload(image: bytes) -> p3.Function:
    if len(image) < BFLT_HEADER_SIZE:
        raise UserImageError("truncated bFLT image")

    fields = struct.unpack(">4s15I", image[:BFLT_HEADER_SIZE])
    magic = fields[0]
    (
        rev,
        entry,
        data_start,
        data_end,
        bss_end,
        _stack_size,
        reloc_start,
        reloc_count,
        _flags,
        _build_date,
        *_filler,
    ) = fields[1:]

    if magic != BFLT_MAGIC:
        raise UserImageError(f"bad bFLT magic: {magic!r}")
    if rev != BFLT_VERSION:
        raise UserImageError(f"unsupported bFLT version: {rev}")
    if entry != BFLT_HEADER_SIZE:
        raise UserImageError(f"unexpected MiniMachine bFLT entry: {entry}")
    if not (entry <= data_start <= data_end <= bss_end <= len(image)):
        raise UserImageError("invalid bFLT section bounds")
    if reloc_count != 0 or reloc_start != data_end:
        raise UserImageError("MiniMachine bootstrap bFLT must be relocation-free")

    return unpack_user_payload(image[entry:data_start])
