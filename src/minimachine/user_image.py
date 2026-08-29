from __future__ import annotations

import json

from . import muir, p3

FORMAT = "minimachine-p3-v1"


class UserImageError(ValueError):
    pass


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
