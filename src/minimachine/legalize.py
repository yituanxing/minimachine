from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re

from . import muir
from .llvm_text import TextBlock, TextFunction, TextInst, parse_module


_LOCAL_RE = re.compile(r"%[-A-Za-z$._0-9]+")
_VALUE_TOKEN_RE = re.compile(
    r"(%[-A-Za-z$._0-9]+|@[-A-Za-z$._0-9]+|-?[0-9]+|true|false|null)"
)
_INT_TYPE_RE = re.compile(r"\bi(1|8|16|32|64)\b")
_PHI_IN_RE = re.compile(r"\[\s*([^,]+),\s*%([-A-Za-z$._0-9]+)\s*\]")
_LABEL_USE_RE = re.compile(r"label\s+%([-A-Za-z$._0-9]+)")


class LegalizeError(ValueError):
    def __init__(self, function: str, block: str, opcode: str, reason: str):
        self.function = function
        self.block = block
        self.opcode = opcode
        self.reason = reason
        super().__init__(f"{function}:{block}:{opcode}: {reason}")


@dataclass
class LegalizeStats:
    llvm_instructions: int = 0
    muir_instructions: int = 0
    fused_icmp_br: int = 0
    phi_edge_moves: int = 0
    folded_gep_mem: int = 0
    temporary_helpers: int = 0
    regular_calls: int = 0
    arch_escapes: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "llvm_instructions": self.llvm_instructions,
            "muir_instructions": self.muir_instructions,
            "fused_icmp_br": self.fused_icmp_br,
            "phi_edge_moves": self.phi_edge_moves,
            "folded_gep_mem": self.folded_gep_mem,
            "temporary_helpers": self.temporary_helpers,
            "regular_calls": self.regular_calls,
            "arch_escapes": self.arch_escapes,
        }


def _slot(name: str) -> muir.Slot:
    return muir.Slot(name[1:] if name.startswith("%") else name)


def _width(bits: int) -> muir.Width:
    if bits == 1:
        # Materialized i1 values are represented as 0/1 bytes. Fused compares
        # keep the compared operand width and never need an i1 machine width.
        return muir.Width.I8
    return {
        8: muir.Width.I8,
        16: muir.Width.I16,
        32: muir.Width.I32,
        64: muir.Width.I64,
    }[bits]


def _first_width(text: str, *, default_pointer: bool = False) -> muir.Width:
    m = _INT_TYPE_RE.search(text)
    if m:
        return _width(int(m.group(1)))
    if default_pointer and re.search(r"\bptr\b", text):
        return muir.Width.I64
    raise ValueError(f"cannot determine supported width from: {text}")


def _value(segment: str) -> muir.Value:
    tokens = _VALUE_TOKEN_RE.findall(segment)
    if not tokens:
        raise ValueError(f"cannot parse scalar value from: {segment}")
    token = tokens[-1]
    if token.startswith("%"):
        return _slot(token)
    if token.startswith("@"):
        return muir.Symbol(token[1:])
    if token == "true":
        return muir.Imm(1)
    if token in {"false", "null"}:
        return muir.Imm(0)
    return muir.Imm(int(token))


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def _icmp_basis(
    pred: str,
    width: muir.Width,
    a: muir.Value,
    b: muir.Value,
    true_target: muir.Target,
    false_target: muir.Target,
) -> muir.Br:
    if pred == "eq":
        return muir.Br(width, muir.Cond.EQ, a, b, true_target, false_target)
    if pred == "ne":
        return muir.Br(width, muir.Cond.EQ, a, b, false_target, true_target)
    if pred == "slt":
        return muir.Br(width, muir.Cond.SLT, a, b, true_target, false_target)
    if pred == "sgt":
        return muir.Br(width, muir.Cond.SLT, b, a, true_target, false_target)
    if pred == "sle":
        return muir.Br(width, muir.Cond.SLT, b, a, false_target, true_target)
    if pred == "sge":
        return muir.Br(width, muir.Cond.SLT, a, b, false_target, true_target)
    if pred == "ult":
        return muir.Br(width, muir.Cond.ULT, a, b, true_target, false_target)
    if pred == "ugt":
        return muir.Br(width, muir.Cond.ULT, b, a, true_target, false_target)
    if pred == "ule":
        return muir.Br(width, muir.Cond.ULT, b, a, false_target, true_target)
    if pred == "uge":
        return muir.Br(width, muir.Cond.ULT, a, b, false_target, true_target)
    raise ValueError(f"unsupported icmp predicate: {pred}")


def _parse_icmp(inst: TextInst):
    m = re.search(
        r"icmp\s+([a-z]+)\s+(i(?:1|8|16|32|64)|ptr)\s+(.+?),\s*(.+?)(?:,\s*!.*)?$",
        inst.text,
    )
    if not m:
        raise ValueError(f"cannot parse icmp: {inst.text}")
    pred, ty, lhs, rhs = m.groups()
    width = muir.Width.I64 if ty == "ptr" else _width(int(ty[1:]))
    return pred, width, _value(lhs), _value(rhs)


def _use_counts(fn: TextFunction) -> Counter[str]:
    counts: Counter[str] = Counter()
    for block in fn.blocks:
        for inst in block.instructions:
            text = inst.text
            for name in _LOCAL_RE.findall(text):
                if inst.result == name:
                    continue
                counts[name] += 1
    return counts


def _result_defs(fn: TextFunction) -> dict[str, TextInst]:
    return {
        inst.result: inst
        for block in fn.blocks
        for inst in block.instructions
        if inst.result is not None
    }


def _parse_phi(inst: TextInst):
    width = _first_width(inst.text, default_pointer=True)
    incoming = []
    for value_text, pred in _PHI_IN_RE.findall(inst.text):
        incoming.append((_value(value_text), pred))
    if not incoming:
        raise ValueError(f"cannot parse phi: {inst.text}")
    return width, incoming


def _scalar_size(ty: str) -> int | None:
    ty = ty.strip()
    m = re.fullmatch(r"i(8|16|32|64)", ty)
    if m:
        return int(m.group(1)) // 8
    if ty == "ptr":
        return 8
    m = re.fullmatch(r"\[(\d+)\s+x\s+i(8|16|32|64)\]", ty)
    if m:
        return int(m.group(1)) * (int(m.group(2)) // 8)
    return None


def _parse_gep(inst: TextInst):
    text = inst.text
    # Named structs/complex literal aggregate types are deliberately surfaced
    # as layout blockers in this first legalizer rather than guessed.
    m = re.match(
        r"getelementptr(?:\s+inbounds)?\s+(.+?),\s+ptr(?:\s+addrspace\(\d+\))?\s+([^,]+)(.*)$",
        text,
    )
    if not m:
        raise ValueError(f"cannot parse gep: {text}")
    source_ty, base_text, rest = m.groups()
    base = _value(base_text)
    raw_indices = re.findall(r",\s+i(?:32|64)\s+([^,]+)", rest)
    indices = [_value(x) for x in raw_indices]
    if not indices:
        return base, 0, None

    # Common scalar pointer arithmetic: GEP iT, ptr base, idx.
    scalar = _scalar_size(source_ty)
    if scalar is not None and not source_ty.strip().startswith("["):
        if len(indices) != 1:
            raise ValueError(f"complex scalar gep indices: {text}")
        idx = indices[0]
        if isinstance(idx, muir.Imm):
            return base, idx.value * scalar, None
        return base, 0, (idx, scalar)

    # Common array form: GEP [N x iT], ptr base, 0, idx.
    am = re.fullmatch(r"\[(\d+)\s+x\s+i(8|16|32|64)\]", source_ty.strip())
    if am:
        n = int(am.group(1))
        elem = int(am.group(2)) // 8
        if len(indices) == 1:
            idx = indices[0]
            scale = n * elem
        elif len(indices) == 2 and isinstance(indices[0], muir.Imm) and indices[0].value == 0:
            idx = indices[1]
            scale = elem
        else:
            raise ValueError(f"complex array gep indices: {text}")
        if isinstance(idx, muir.Imm):
            return base, idx.value * scale, None
        return base, 0, (idx, scale)

    # Zero-only GEP is representation-preserving even when the aggregate
    # layout is not yet modeled.
    if all(isinstance(i, muir.Imm) and i.value == 0 for i in indices):
        return base, 0, None

    raise ValueError(f"aggregate layout required: {source_ty}")


def _call_args(text: str, callee_end: int) -> tuple[muir.Value, ...]:
    open_paren = text.find("(", callee_end)
    if open_paren < 0:
        raise ValueError(f"call has no argument list: {text}")
    depth = 0
    close = None
    for i in range(open_paren, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                close = i
                break
    if close is None:
        raise ValueError(f"unterminated call argument list: {text}")
    body = text[open_paren + 1 : close].strip()
    if not body:
        return ()

    args = []
    part = []
    depth = 0
    for c in body + ",":
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        if c == "," and depth == 0:
            segment = "".join(part).strip()
            if segment:
                args.append(_value(segment))
            part = []
        else:
            part.append(c)
    return tuple(args)


def _parse_call(inst: TextInst):
    text = inst.text
    if re.search(r"\basm\b", text):
        return "inline_asm", None, (), None

    direct = re.search(r"@([-A-Za-z$._0-9]+)\s*\(", text)
    if direct:
        symbol = direct.group(1)
        args = _call_args(text, direct.end() - 1)
        return "direct", symbol, args, None

    # Indirect call: find the SSA callee immediately before '('.
    indirect = re.search(r"(%[-A-Za-z$._0-9]+)\s*\(", text)
    if indirect:
        args = _call_args(text, indirect.end() - 1)
        return "indirect", None, args, _slot(indirect.group(1))

    raise ValueError(f"cannot parse call target: {text}")


_DROP_INTRINSICS = (
    "llvm.lifetime.start",
    "llvm.lifetime.end",
    "llvm.assume",
    "llvm.dbg.",
)


def _parallel_copies(copies, temp_index: list[int]):
    pending = [
        [dst, src, width]
        for dst, src, width in copies
        if not (isinstance(src, muir.Slot) and src == dst)
    ]
    out: list[muir.Mov] = []

    while pending:
        source_slots = {
            src.name
            for _, src, _ in pending
            if isinstance(src, muir.Slot)
        }
        safe = next(
            (i for i, (dst, _, _) in enumerate(pending) if dst.name not in source_slots),
            None,
        )
        if safe is not None:
            dst, src, width = pending.pop(safe)
            out.append(muir.Mov(width, dst, src))
            continue

        dst, _, width = pending[0]
        temp_index[0] += 1
        temp = muir.Slot(f"__phi_tmp{temp_index[0]}")
        out.append(muir.Mov(width, temp, dst))
        for item in pending:
            if isinstance(item[1], muir.Slot) and item[1] == dst:
                item[1] = temp

    return out


def legalize_function(fn: TextFunction) -> tuple[muir.Function, LegalizeStats]:
    stats = LegalizeStats()
    uses = _use_counts(fn)
    defs = _result_defs(fn)
    frame_slots = {arg[1:] for arg in fn.args}
    aliases: dict[str, muir.Address] = {}
    phis_by_target: dict[str, list[tuple[muir.Slot, muir.Value, str, muir.Width]]] = defaultdict(list)
    out_blocks: dict[str, muir.Block] = {}
    temp_counter = [0]

    # Pre-read PHIs so predecessor edge copies are known independently of
    # textual block order.
    for block in fn.blocks:
        for inst in block.instructions:
            if inst.opcode != "phi":
                continue
            if inst.result is None:
                raise LegalizeError(fn.name, block.label, "phi", "phi has no result")
            try:
                width, incoming = _parse_phi(inst)
            except ValueError as e:
                raise LegalizeError(fn.name, block.label, "phi", str(e)) from e
            dst = _slot(inst.result)
            frame_slots.add(dst.name)
            for src, pred in incoming:
                phis_by_target[block.label].append((dst, src, pred, width))

    for block in fn.blocks:
        out: list[muir.Instr] = []
        for inst in block.instructions:
            stats.llvm_instructions += 1
            op = inst.opcode
            result = _slot(inst.result) if inst.result else None
            if result:
                frame_slots.add(result.name)

            try:
                if op == "phi":
                    continue

                if op == "alloca":
                    # Frame object details are refined by the ABI/frame pass.
                    continue

                if op == "sub":
                    m = re.search(r"sub(?:\s+\w+)*\s+(i(?:8|16|32|64))\s+(.+?),\s*(.+)$", inst.text)
                    if not m or result is None:
                        raise ValueError(f"cannot parse sub: {inst.text}")
                    width = _width(int(m.group(1)[1:]))
                    out.append(muir.Sub(width, result, _value(m.group(2)), _value(m.group(3))))
                    continue

                if op == "add":
                    m = re.search(r"add(?:\s+\w+)*\s+(i(?:8|16|32|64))\s+(.+?),\s*(.+)$", inst.text)
                    if not m or result is None:
                        raise ValueError(f"cannot parse add: {inst.text}")
                    width = _width(int(m.group(1)[1:]))
                    a, b = _value(m.group(2)), _value(m.group(3))
                    if isinstance(b, muir.Imm):
                        out.append(muir.Sub(width, result, a, muir.Imm(-b.value)))
                    elif isinstance(a, muir.Imm):
                        temp_counter[0] += 1
                        neg = muir.Slot(f"__add_neg{temp_counter[0]}")
                        frame_slots.add(neg.name)
                        out.append(muir.Sub(width, neg, muir.Imm(0), b))
                        out.append(muir.Sub(width, result, a, neg))
                    else:
                        temp_counter[0] += 1
                        neg = muir.Slot(f"__add_neg{temp_counter[0]}")
                        frame_slots.add(neg.name)
                        out.append(muir.Sub(width, neg, muir.Imm(0), b))
                        out.append(muir.Sub(width, result, a, neg))
                    continue

                if op == "getelementptr":
                    if result is None:
                        raise ValueError("GEP has no result")
                    base, offset, dynamic = _parse_gep(inst)
                    # Fold only when the GEP has exactly one use. The load/store
                    # legalizer consumes this alias as a memory operand.
                    if uses[inst.result] == 1 and dynamic is None:
                        aliases[inst.result] = muir.Address(base, offset)
                        continue
                    if dynamic is None:
                        if offset == 0:
                            out.append(muir.Mov(muir.Width.I64, result, base))
                        else:
                            out.append(muir.Sub(muir.Width.I64, result, base, muir.Imm(-offset)))
                    else:
                        idx, scale = dynamic
                        stats.temporary_helpers += 1
                        out.append(
                            muir.Helper(
                                f"__mm_ptr_add_scaled_{scale}",
                                (base, idx),
                                result,
                            )
                        )
                    continue

                if op == "load":
                    m = re.search(r"load(?:\s+volatile)?\s+(i(?:1|8|16|32|64)|ptr)\s*,\s*ptr\s+([^,]+)", inst.text)
                    if not m or result is None:
                        raise ValueError(f"cannot parse load: {inst.text}")
                    width = muir.Width.I64 if m.group(1) == "ptr" else _width(int(m.group(1)[1:]))
                    ptr_text = m.group(2).strip()
                    addr = aliases.get(ptr_text)
                    if addr is None:
                        addr = muir.Address(_value(ptr_text), 0)
                    else:
                        stats.folded_gep_mem += 1
                    out.append(muir.Mov(width, result, muir.Mem(addr, width)))
                    continue

                if op == "store":
                    m = re.search(
                        r"store(?:\s+volatile)?\s+(i(?:1|8|16|32|64)|ptr)\s+(.+?),\s*ptr\s+([^,]+)",
                        inst.text,
                    )
                    if not m:
                        raise ValueError(f"cannot parse store: {inst.text}")
                    width = muir.Width.I64 if m.group(1) == "ptr" else _width(int(m.group(1)[1:]))
                    src = _value(m.group(2))
                    ptr_text = m.group(3).strip()
                    addr = aliases.get(ptr_text)
                    if addr is None:
                        addr = muir.Address(_value(ptr_text), 0)
                    else:
                        stats.folded_gep_mem += 1
                    out.append(muir.Mov(width, muir.Mem(addr, width), src))
                    continue

                if op == "icmp":
                    if inst.result and uses[inst.result] == 1:
                        # The consuming conditional BR emits the fused compare.
                        continue
                    if result is None:
                        raise ValueError("icmp has no result")
                    pred, width, a, b = _parse_icmp(inst)
                    stats.temporary_helpers += 1
                    out.append(muir.Helper(f"__mm_icmp_{pred}_{width.value}", (a, b), result))
                    continue

                if op == "br":
                    labels = _LABEL_USE_RE.findall(inst.text)
                    if inst.text.startswith("br label"):
                        if len(labels) != 1:
                            raise ValueError(f"cannot parse unconditional br: {inst.text}")
                        t = muir.Target(label=labels[0])
                        out.append(muir.Br(muir.Width.I8, muir.Cond.EQ, muir.Imm(0), muir.Imm(0), t, t))
                        continue
                    m = re.search(r"br\s+i1\s+(%[-A-Za-z$._0-9]+|true|false)", inst.text)
                    if not m or len(labels) != 2:
                        raise ValueError(f"cannot parse conditional br: {inst.text}")
                    cond_text = m.group(1)
                    tt, ft = muir.Target(label=labels[0]), muir.Target(label=labels[1])
                    cond_def = defs.get(cond_text)
                    if cond_def and cond_def.opcode == "icmp" and uses[cond_text] == 1:
                        pred, width, a, b = _parse_icmp(cond_def)
                        out.append(_icmp_basis(pred, width, a, b, tt, ft))
                        stats.fused_icmp_br += 1
                    else:
                        cond = _value(cond_text)
                        out.append(
                            muir.Br(
                                muir.Width.I8,
                                muir.Cond.EQ,
                                cond,
                                muir.Imm(0),
                                ft,
                                tt,
                            )
                        )
                    continue

                if op in {"zext", "sext", "trunc"}:
                    m = re.search(
                        rf"{op}\s+(i(?:1|8|16|32|64))\s+(.+?)\s+to\s+(i(?:1|8|16|32|64))",
                        inst.text,
                    )
                    if not m or result is None:
                        raise ValueError(f"cannot parse {op}: {inst.text}")
                    dst_width = _width(int(m.group(3)[1:]))
                    mode = {"zext": "zext", "sext": "sext", "trunc": "trunc"}[op]
                    out.append(muir.Mov(dst_width, result, _value(m.group(2)), extend=mode))
                    continue

                if op in {"ptrtoint", "inttoptr", "bitcast", "addrspacecast"}:
                    m = re.search(rf"{op}\s+(.+?)\s+to\s+(.+)$", inst.text)
                    if not m or result is None:
                        raise ValueError(f"cannot parse {op}: {inst.text}")
                    src = _value(m.group(1))
                    width = _first_width(m.group(2), default_pointer=True)
                    out.append(muir.Mov(width, result, src))
                    continue

                if op == "freeze":
                    m = re.search(r"freeze\s+(i(?:1|8|16|32|64)|ptr)\s+(.+)$", inst.text)
                    if not m or result is None:
                        raise ValueError(f"cannot parse freeze: {inst.text}")
                    width = muir.Width.I64 if m.group(1) == "ptr" else _width(int(m.group(1)[1:]))
                    out.append(muir.Mov(width, result, _value(m.group(2))))
                    continue

                if op == "select":
                    m = re.search(
                        r"select\s+i1\s+(.+?),\s+(i(?:1|8|16|32|64)|ptr)\s+(.+?),\s+\2\s+(.+?)(?:,\s*!.*)?$",
                        inst.text,
                    )
                    if not m or result is None:
                        raise ValueError(f"cannot parse select: {inst.text}")
                    width = muir.Width.I64 if m.group(2) == "ptr" else _width(int(m.group(2)[1:]))
                    stats.temporary_helpers += 1
                    out.append(
                        muir.Helper(
                            f"__mm_select_{width.value}",
                            (_value(m.group(1)), _value(m.group(3)), _value(m.group(4))),
                            result,
                        )
                    )
                    continue

                if op in {"and", "or", "xor", "shl", "lshr", "ashr", "mul", "udiv", "sdiv", "urem", "srem"}:
                    m = re.search(
                        rf"{op}(?:\s+\w+)*\s+(i(?:8|16|32|64))\s+(.+?),\s*(.+)$",
                        inst.text,
                    )
                    if not m or result is None:
                        raise ValueError(f"cannot parse helper op {op}: {inst.text}")
                    width = int(m.group(1)[1:])
                    stats.temporary_helpers += 1
                    out.append(
                        muir.Helper(
                            f"__mm_{op}_{width}",
                            (_value(m.group(2)), _value(m.group(3))),
                            result,
                        )
                    )
                    continue

                if op == "call":
                    kind, symbol, args, callee_slot = _parse_call(inst)
                    if kind == "inline_asm":
                        stats.arch_escapes += 1
                        stats.temporary_helpers += 1
                        out.append(
                            muir.Helper(
                                "__mm_arch_inline_asm",
                                (),
                                result,
                            )
                        )
                        continue
                    assert kind in {"direct", "indirect"}
                    if symbol and symbol.startswith(_DROP_INTRINSICS):
                        continue
                    if symbol and symbol.startswith("llvm."):
                        stats.temporary_helpers += 1
                        out.append(
                            muir.Helper(
                                "__mm_" + _sanitize(symbol),
                                args,
                                result,
                            )
                        )
                        continue
                    callee = (
                        muir.Callee(symbol=symbol)
                        if kind == "direct"
                        else muir.Callee(slot=callee_slot)
                    )
                    out.append(muir.Call(callee, args, result))
                    stats.regular_calls += 1
                    continue

                if op == "ret":
                    if re.match(r"ret\s+void\b", inst.text):
                        out.append(muir.Ret(None))
                    else:
                        m = re.match(r"ret\s+(?:i(?:1|8|16|32|64)|ptr)\s+(.+)$", inst.text)
                        if not m:
                            raise ValueError(f"cannot parse ret: {inst.text}")
                        out.append(muir.Ret(_value(m.group(1))))
                    continue

                if op == "unreachable":
                    out.append(muir.Trap("llvm.unreachable"))
                    continue

                if op in {"extractvalue", "insertvalue"}:
                    if result is None:
                        raise ValueError(f"{op} has no result")
                    stats.temporary_helpers += 1
                    values = tuple(_value(x) for x in _LOCAL_RE.findall(inst.text))
                    out.append(muir.Helper(f"__mm_{op}", values, result))
                    continue

                if op in {"callbr", "indirectbr"}:
                    stats.arch_escapes += 1
                    raise ValueError(f"arch/control escape not yet lowered: {inst.text}")

                raise ValueError(f"opcode not handled: {op}")

            except ValueError as e:
                raise LegalizeError(fn.name, block.label, op, str(e)) from e

        out_blocks[block.label] = muir.Block(block.label, out)

    # Insert PHI edge copies immediately before predecessor terminators.
    copies_by_pred: dict[str, list[tuple[muir.Slot, muir.Value, muir.Width]]] = defaultdict(list)
    for target, phis in phis_by_target.items():
        for dst, src, pred, width in phis:
            if pred not in out_blocks:
                raise LegalizeError(fn.name, target, "phi", f"unknown predecessor: {pred}")
            copies_by_pred[pred].append((dst, src, width))

    for pred, copies in copies_by_pred.items():
        block = out_blocks[pred]
        if not block.instructions:
            raise LegalizeError(fn.name, pred, "phi", "predecessor has no terminator")
        term = block.instructions[-1]
        if not isinstance(term, (muir.Br, muir.Ret, muir.Trap)):
            raise LegalizeError(fn.name, pred, "phi", "predecessor terminator is not lowered")
        moves = _parallel_copies(copies, temp_counter)
        for move in moves:
            if isinstance(move.dst, muir.Slot):
                frame_slots.add(move.dst.name)
        block.instructions[-1:-1] = moves
        stats.phi_edge_moves += len(moves)

    result_fn = muir.Function(fn.name, [out_blocks[b.label] for b in fn.blocks], frame_slots)
    stats.muir_instructions = sum(len(b.instructions) for b in result_fn.blocks)
    return result_fn, stats


def legalize_module(text: str):
    functions = parse_module(text)
    out = []
    total = LegalizeStats()
    for fn in functions:
        lowered, stats = legalize_function(fn)
        out.append(lowered)
        for key, value in stats.as_dict().items():
            setattr(total, key, getattr(total, key) + value)
    return out, total
