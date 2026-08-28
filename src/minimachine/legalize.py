from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re

from . import muir
from .llvm_text import TextBlock, TextFunction, TextInst, parse_module
from .layout import DataLayout, LayoutError


_LOCAL_RE = re.compile(r"%[-A-Za-z$._0-9]+")
_VALUE_TOKEN_RE = re.compile(
    r"(%[-A-Za-z$._0-9]+|@[-A-Za-z$._0-9]+|-?[0-9]+|true|false|null)"
)
_INT_TYPE_RE = re.compile(r"(?<![%@A-Za-z0-9_.$])i(\d+)(?![-A-Za-z0-9_.$])")
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


def _storage_width(bits: int) -> muir.Width:
    if bits <= 8:
        return muir.Width.I8
    if bits <= 16:
        return muir.Width.I16
    if bits <= 32:
        return muir.Width.I32
    if bits <= 64:
        return muir.Width.I64
    raise ValueError(f"integer width exceeds one MiniMachine slot: i{bits}")


def _first_width(text: str, *, default_pointer: bool = False) -> muir.Width:
    m = _INT_TYPE_RE.search(text)
    if m:
        return _storage_width(int(m.group(1)))
    if default_pointer and re.search(r"\bptr\b", text):
        return muir.Width.I64
    raise ValueError(f"cannot determine supported width from: {text}")


def _value(segment: str) -> muir.Value:
    if re.search(r"\bpoison\b", segment):
        return muir.Arbitrary("poison")
    if re.search(r"\bundef\b", segment):
        return muir.Arbitrary("undef")
    if "zeroinitializer" in segment:
        return muir.Imm(0)
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
        r"icmp\s+([a-z]+)\s+(i\d+|ptr)\s+(.+?),\s*(.+?)(?:,\s*!.*)?$",
        inst.text,
    )
    if not m:
        raise ValueError(f"cannot parse icmp: {inst.text}")
    pred, ty, lhs, rhs = m.groups()
    if ty == "ptr":
        return pred, 64, muir.Width.I64, _value(lhs), _value(rhs)
    bits = int(ty[1:])
    width = _width(bits) if bits in {1, 8, 16, 32, 64} else None
    return pred, bits, width, _value(lhs), _value(rhs)


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


def _split_top_commas(text: str) -> list[str]:
    parts=[]
    start=0
    stack=[]
    close_to_open={")":"(", "]":"[", "}":"{", ">":"<"}
    for i,c in enumerate(text):
        if c in "([{<":
            stack.append(c)
        elif c in close_to_open:
            if stack and stack[-1] == close_to_open[c]:
                stack.pop()
        elif c == "," and not stack:
            parts.append(text[start:i].strip())
            start=i+1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def _const_gep_value(text: str, layout: DataLayout) -> muir.Value:
    raw=text.strip()
    m=re.match(r"getelementptr(\s+inbounds)?\s*\((.*)\)$", raw)
    if not m:
        raise ValueError(f"not a constant GEP expression: {text}")
    normalized="getelementptr" + (m.group(1) or "") + " " + m.group(2)
    base, offset, dynamic=_parse_gep(TextInst(None, "getelementptr", normalized), layout)
    if dynamic:
        raise ValueError(f"dynamic constant-expression GEP: {text}")
    if isinstance(base, muir.Symbol):
        return muir.Reloc(base.name, offset)
    if isinstance(base, muir.Reloc):
        return muir.Reloc(base.symbol, base.addend + offset)
    if offset == 0:
        return base
    raise ValueError(f"non-symbol constant-expression GEP needs materialization: {text}")


def _parse_phi(inst: TextInst, layout: DataLayout):
    width = _first_width(inst.text, default_pointer=True)
    incoming=[]
    text=inst.text
    stack=[]
    start=None
    for i,c in enumerate(text):
        if c=="[":
            if not stack:
                start=i+1
            stack.append(c)
        elif c=="]" and stack:
            stack.pop()
            if not stack and start is not None:
                body=text[start:i]
                parts=_split_top_commas(body)
                if len(parts)==2 and re.fullmatch(r"%[-A-Za-z$._0-9]+", parts[1]):
                    value_text=parts[0].strip()
                    if value_text.startswith("getelementptr"):
                        value=_const_gep_value(value_text, layout)
                    else:
                        value=_value(value_text)
                    incoming.append((value, parts[1][1:]))
                start=None
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


def _parse_gep(inst: TextInst, layout: DataLayout):
    text = inst.text
    body=re.sub(r"^getelementptr(?:\s+inbounds)?\s+", "", text, count=1)
    parts=_split_top_commas(body)
    if len(parts) < 2:
        raise ValueError(f"cannot parse gep: {text}")
    source_ty=parts[0].strip()
    pm=re.match(r"ptr(?:\s+addrspace\(\d+\))?\s+(.+)$", parts[1])
    if not pm:
        raise ValueError(f"cannot parse gep base: {text}")
    base_text=pm.group(1).strip()
    if base_text.startswith("getelementptr"):
        base=_const_gep_value(base_text,layout)
    else:
        base=_value(base_text)
    indices=[]
    for raw in parts[2:]:
        m=re.match(r"i(?:32|64)\s+(.+)$", raw)
        if not m:
            raise ValueError(f"cannot parse gep index: {raw} in {text}")
        indices.append(_value(m.group(1)))
    if not indices:
        return base,0,[]

    constant_offset=0
    dynamic_terms: list[tuple[muir.Value,int]]=[]

    info=layout.info(source_ty)
    first_scale=((info.size+info.align-1)//info.align)*info.align
    first=indices[0]
    if isinstance(first,muir.Imm):
        constant_offset += first.value*first_scale
    else:
        dynamic_terms.append((first,first_scale))

    current_ty=source_ty
    for index in indices[1:]:
        info=layout.info(current_ty)
        if info.fields is not None:
            if not isinstance(index,muir.Imm):
                raise LayoutError(f"dynamic struct GEP index in {current_ty}; inst={text}")
            assert info.field_offsets is not None
            n=index.value
            if n<0 or n>=len(info.fields):
                raise LayoutError(
                    f"struct GEP index {n} out of range for {current_ty} "
                    f"fields={len(info.fields)} inst={text}"
                )
            constant_offset += info.field_offsets[n]
            current_ty=info.fields[n]
            continue
        if info.element is not None:
            elem=layout.info(info.element)
            scale=((elem.size+elem.align-1)//elem.align)*elem.align
            if isinstance(index,muir.Imm):
                constant_offset += index.value*scale
            else:
                dynamic_terms.append((index,scale))
            current_ty=info.element
            continue
        raise LayoutError(f"cannot descend GEP through scalar {current_ty}; inst={text}")

    return base,constant_offset,dynamic_terms


def _call_args(text: str, callee_end: int, layout: DataLayout) -> tuple[muir.Value, ...]:
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
                if "getelementptr" in segment:
                    pos = segment.find("getelementptr")
                    args.append(_const_gep_value(segment[pos:], layout))
                else:
                    args.append(_value(segment))
            part = []
        else:
            part.append(c)
    return tuple(args)


def _parse_call(inst: TextInst, layout: DataLayout):
    text = inst.text
    if re.search(r"\basm\b", text):
        return "inline_asm", None, (), None

    direct = re.search(r"@([-A-Za-z$._0-9]+)\s*\(", text)
    if direct:
        symbol = direct.group(1)
        args = _call_args(text, direct.end() - 1, layout)
        return "direct", symbol, args, None

    # Indirect call: find the SSA callee immediately before '('.
    indirect = re.search(r"(%[-A-Za-z$._0-9]+)\s*\(", text)
    if indirect:
        args = _call_args(text, indirect.end() - 1, layout)
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


def legalize_function(fn: TextFunction, layout: DataLayout) -> tuple[muir.Function, LegalizeStats]:
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
                width, incoming = _parse_phi(inst, layout)
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
                    m = re.search(r"add(?:\s+\w+)*\s+(i\d+)\s+(.+?),\s*(.+)$", inst.text)
                    if not m or result is None:
                        raise ValueError(f"cannot parse add: {inst.text}")
                    bits = int(m.group(1)[1:])
                    a, b = _value(m.group(2)), _value(m.group(3))
                    if bits > 64:
                        stats.temporary_helpers += 1
                        out.append(muir.Helper(f"__mm_add_{bits}", (a, b), result))
                        continue
                    width = _storage_width(bits)
                    if bits not in {8,16,32,64}:
                        stats.temporary_helpers += 1
                        out.append(muir.Helper(f"__mm_add_{bits}", (a, b), result))
                    elif isinstance(b, muir.Imm):
                        out.append(muir.Sub(width, result, a, muir.Imm(-b.value)))
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
                    base, offset, dynamic_terms = _parse_gep(inst, layout)
                    # Fold only when the GEP has exactly one use. The load/store
                    # legalizer consumes this alias as a memory operand.
                    if uses[inst.result] == 1 and not dynamic_terms:
                        aliases[inst.result] = muir.Address(base, offset)
                        continue

                    current: muir.Value = base
                    if offset:
                        temp_counter[0] += 1
                        off_dst = result if not dynamic_terms else muir.Slot(f"__gep_off{temp_counter[0]}")
                        frame_slots.add(off_dst.name)
                        out.append(muir.Sub(muir.Width.I64, off_dst, current, muir.Imm(-offset)))
                        current = off_dst

                    for n, (idx, scale) in enumerate(dynamic_terms):
                        last = n == len(dynamic_terms) - 1
                        temp_counter[0] += 1
                        dst = result if last else muir.Slot(f"__gep_dyn{temp_counter[0]}_{n}")
                        frame_slots.add(dst.name)
                        if scale == 1:
                            neg = muir.Slot(f"__gep_neg{temp_counter[0]}")
                            frame_slots.add(neg.name)
                            out.append(muir.Sub(muir.Width.I64, neg, muir.Imm(0), idx))
                            out.append(muir.Sub(muir.Width.I64, dst, current, neg))
                        else:
                            stats.temporary_helpers += 1
                            out.append(muir.Helper(f"__mm_ptr_add_scaled_{scale}", (current, idx), dst))
                        current = dst

                    if not offset and not dynamic_terms:
                        out.append(muir.Mov(muir.Width.I64, result, base))
                    elif dynamic_terms and current != result:
                        out.append(muir.Mov(muir.Width.I64, result, current))
                    continue

                if op == "load":
                    body=inst.text[len("load "):]
                    parts=_split_top_commas(body)
                    if len(parts) < 2 or result is None:
                        raise ValueError(f"cannot parse load: {inst.text}")
                    ty=parts[0].strip()
                    ptr_part=parts[1].strip()
                    pm=re.match(r"ptr(?:\s+addrspace\(\d+\))?\s+(.+)$",ptr_part)
                    if not pm:
                        raise ValueError(f"cannot parse load pointer: {inst.text}")
                    ptr_text=pm.group(1).strip()
                    if ptr_text.startswith("getelementptr"):
                        cv=_const_gep_value(ptr_text,layout)
                        if isinstance(cv,muir.Reloc):
                            addr=muir.Address(muir.Symbol(cv.symbol),cv.addend)
                        else:
                            addr=muir.Address(cv,0)
                    else:
                        addr=aliases.get(ptr_text)
                        if addr is None:
                            addr=muir.Address(_value(ptr_text),0)
                        else:
                            stats.folded_gep_mem += 1

                    if ty=="ptr" or re.fullmatch(r"i\d+",ty):
                        if ty=="ptr":
                            width=muir.Width.I64
                            out.append(muir.Mov(width,result,muir.Mem(addr,width)))
                        else:
                            bits=int(ty[1:])
                            if bits in {1,8,16,32,64}:
                                width=_storage_width(bits)
                                out.append(muir.Mov(width,result,muir.Mem(addr,width)))
                            else:
                                base=addr.base
                                if addr.offset:
                                    temp_counter[0]+=1
                                    ap=muir.Slot(f"__odd_addr{temp_counter[0]}")
                                    frame_slots.add(ap.name)
                                    out.append(muir.Sub(muir.Width.I64,ap,base,muir.Imm(-addr.offset)))
                                    base=ap
                                stats.temporary_helpers += 1
                                out.append(muir.Helper(f"__mm_load_i{bits}",(base,),result))
                    else:
                        stats.temporary_helpers += 1
                        base=addr.base
                        if addr.offset:
                            temp_counter[0]+=1
                            ap=muir.Slot(f"__agg_addr{temp_counter[0]}")
                            frame_slots.add(ap.name)
                            out.append(muir.Sub(muir.Width.I64,ap,base,muir.Imm(-addr.offset)))
                            base=ap
                        out.append(muir.Helper("__mm_load_aggregate",(base,),result))
                    continue

                if op == "store":
                    body=inst.text[len("store "):]
                    parts=_split_top_commas(body)
                    if len(parts) < 2:
                        raise ValueError(f"cannot parse store: {inst.text}")
                    vm=re.match(r"(.+?)\s+(.+)$",parts[0].strip())
                    pm=re.match(r"ptr(?:\s+addrspace\(\d+\))?\s+(.+)$",parts[1].strip())
                    if not vm or not pm:
                        raise ValueError(f"cannot parse store operands: {inst.text}")
                    ty=vm.group(1).strip()
                    src=_value(vm.group(2))
                    ptr_text=pm.group(1).strip()
                    if ptr_text.startswith("getelementptr"):
                        cv=_const_gep_value(ptr_text,layout)
                        if isinstance(cv,muir.Reloc):
                            addr=muir.Address(muir.Symbol(cv.symbol),cv.addend)
                        else:
                            addr=muir.Address(cv,0)
                    else:
                        addr=aliases.get(ptr_text)
                        if addr is None:
                            addr=muir.Address(_value(ptr_text),0)
                        else:
                            stats.folded_gep_mem += 1

                    if ty=="ptr" or re.fullmatch(r"i\d+",ty):
                        if ty=="ptr":
                            width=muir.Width.I64
                            out.append(muir.Mov(width,muir.Mem(addr,width),src))
                        else:
                            bits=int(ty[1:])
                            if bits in {1,8,16,32,64}:
                                width=_storage_width(bits)
                                out.append(muir.Mov(width,muir.Mem(addr,width),src))
                            else:
                                base=addr.base
                                if addr.offset:
                                    temp_counter[0]+=1
                                    ap=muir.Slot(f"__odd_addr{temp_counter[0]}")
                                    frame_slots.add(ap.name)
                                    out.append(muir.Sub(muir.Width.I64,ap,base,muir.Imm(-addr.offset)))
                                    base=ap
                                stats.temporary_helpers += 1
                                out.append(muir.Helper(f"__mm_store_i{bits}",(base,src),None))
                    else:
                        base=addr.base
                        if addr.offset:
                            temp_counter[0]+=1
                            ap=muir.Slot(f"__agg_addr{temp_counter[0]}")
                            frame_slots.add(ap.name)
                            out.append(muir.Sub(muir.Width.I64,ap,base,muir.Imm(-addr.offset)))
                            base=ap
                        stats.temporary_helpers += 1
                        out.append(muir.Helper("__mm_store_aggregate",(base,src),None))
                    continue


                if op == "icmp":
                    if result is None:
                        raise ValueError("icmp has no result")
                    pred, bits, width, a, b = _parse_icmp(inst)
                    if inst.result and uses[inst.result] == 1 and width is not None:
                        # The consuming conditional BR emits the fused compare.
                        continue
                    stats.temporary_helpers += 1
                    out.append(muir.Helper(f"__mm_icmp_{pred}_{bits}", (a, b), result))
                    continue

                if op == "br":
                    labels = _LABEL_USE_RE.findall(inst.text)
                    if inst.text.startswith("br label"):
                        if len(labels) != 1:
                            raise ValueError(f"cannot parse unconditional br: {inst.text}")
                        t = muir.Target(label=labels[0])
                        out.append(muir.Br(muir.Width.I8, muir.Cond.EQ, muir.Imm(0), muir.Imm(0), t, t))
                        continue
                    cm = re.search(
                        r"br\s+i1\s+icmp\s+([a-z]+)\s*\((.+?),\s*(.+?)\),\s*label",
                        inst.text,
                    )
                    if cm and len(labels) == 2:
                        pred = cm.group(1)
                        lhs_text, rhs_text = cm.group(2), cm.group(3)
                        width = _first_width(lhs_text, default_pointer=True)
                        tt, ft = muir.Target(label=labels[0]), muir.Target(label=labels[1])
                        out.append(_icmp_basis(pred, width, _value(lhs_text), _value(rhs_text), tt, ft))
                        stats.fused_icmp_br += 1
                        continue
                    m = re.search(r"br\s+i1\s+(%[-A-Za-z$._0-9]+|true|false)", inst.text)
                    if not m or len(labels) != 2:
                        raise ValueError(f"cannot parse conditional br: {inst.text}")
                    cond_text = m.group(1)
                    tt, ft = muir.Target(label=labels[0]), muir.Target(label=labels[1])
                    cond_def = defs.get(cond_text)
                    if cond_def and cond_def.opcode == "icmp" and uses[cond_text] == 1:
                        pred, bits, width, a, b = _parse_icmp(cond_def)
                        if width is not None:
                            out.append(_icmp_basis(pred, width, a, b, tt, ft))
                            stats.fused_icmp_br += 1
                        else:
                            # Odd-width compare was materialized by a helper.
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
                        rf"{op}(?:\s+\w+)*\s+(i\d+)\s+(.+?)\s+to\s+(i\d+)",
                        inst.text,
                    )
                    if not m or result is None:
                        raise ValueError(f"cannot parse {op}: {inst.text}")
                    src_bits=int(m.group(1)[1:])
                    dst_bits=int(m.group(3)[1:])
                    if src_bits in {1,8,16,32,64} and dst_bits in {1,8,16,32,64}:
                        dst_width=_width(dst_bits)
                        out.append(muir.Mov(dst_width,result,_value(m.group(2)),extend=op))
                    else:
                        stats.temporary_helpers += 1
                        out.append(
                            muir.Helper(
                                f"__mm_{op}_{src_bits}_{dst_bits}",
                                (_value(m.group(2)),),
                                result,
                            )
                        )
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
                    if result is None:
                        raise ValueError("freeze has no result")
                    m = re.match(r"freeze\s+(i\d+|ptr)\s+(.+)$", inst.text)
                    if m:
                        ty=m.group(1)
                        value=_value(m.group(2))
                        if ty=="ptr":
                            out.append(muir.Mov(muir.Width.I64,result,value))
                        else:
                            bits=int(ty[1:])
                            if bits in {1,8,16,32,64}:
                                out.append(muir.Mov(_storage_width(bits),result,value))
                            else:
                                stats.temporary_helpers += 1
                                out.append(muir.Helper(f"__mm_freeze_{bits}",(value,),result))
                    else:
                        value=_value(inst.text)
                        stats.temporary_helpers += 1
                        out.append(muir.Helper("__mm_freeze_aggregate",(value,),result))
                    continue


                if op == "select":
                    if result is None:
                        raise ValueError("select has no result")
                    body=inst.text[len("select "):]
                    parts=_split_top_commas(body)
                    semantic_parts=[p for p in parts if not p.lstrip().startswith("!")]
                    if len(semantic_parts) != 3:
                        raise ValueError(f"cannot parse select: {inst.text}")
                    parts=semantic_parts
                    cm=re.match(r"i1\s+(.+)$",parts[0])
                    if not cm:
                        raise ValueError(f"cannot parse select condition: {inst.text}")
                    cond=_value(cm.group(1))
                    try:
                        tv=_value(parts[1])
                        fv=_value(parts[2])
                    except ValueError as e:
                        raise ValueError(f"cannot parse select values: {inst.text}") from e
                    type_tag=_sanitize(re.sub(r"\s+(?:%[-A-Za-z$._0-9]+|@[-A-Za-z$._0-9]+|-?\d+|poison|undef|zeroinitializer)\s*$","",parts[1]))
                    stats.temporary_helpers += 1
                    out.append(muir.Helper(f"__mm_select_{type_tag or 'opaque'}",(cond,tv,fv),result))
                    continue


                if op in {"and", "or", "xor", "shl", "lshr", "ashr", "mul", "udiv", "sdiv", "urem", "srem"}:
                    m = re.search(
                        rf"{op}(?:\s+\w+)*\s+(i\d+)\s+(.+?),\s*(.+)$",
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
                    kind, symbol, args, callee_slot = _parse_call(inst, layout)
                    if kind == "inline_asm":
                        stats.arch_escapes += 1
                        out.append(muir.ArchEscape("inline_asm", inst.text))
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
                        try:
                            out.append(muir.Ret(_value(inst.text)))
                        except ValueError as e:
                            raise ValueError(f"cannot parse ret: {inst.text}") from e
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
                    labels = _LABEL_USE_RE.findall(inst.text)
                    if not labels:
                        raise ValueError(f"arch/control escape has no successor: {inst.text}")
                    stats.arch_escapes += 1
                    out.append(
                        muir.ArchEscape(
                            op,
                            inst.text,
                            tuple(muir.Target(label=x) for x in labels),
                        )
                    )
                    continue

                raise ValueError(f"opcode not handled: {op}")

            except ValueError as e:
                raise LegalizeError(fn.name, block.label, op, str(e)) from e

        out_blocks[block.label] = muir.Block(block.label, out)

    # Insert PHI edge copies immediately before predecessor terminators.
    copies_by_pred: dict[str, list[tuple[muir.Slot, muir.Value, muir.Width]]] = defaultdict(list)
    for target, phis in phis_by_target.items():
        for dst, src, pred, width in phis:
            resolved_pred = pred
            if resolved_pred not in out_blocks and resolved_pred.isdigit() and "entry" in out_blocks:
                resolved_pred = "entry"
            if resolved_pred not in out_blocks:
                raise LegalizeError(fn.name, target, "phi", f"unknown predecessor: {pred}")
            copies_by_pred[resolved_pred].append((dst, src, width))

    for pred, copies in copies_by_pred.items():
        block = out_blocks[pred]
        if not block.instructions:
            raise LegalizeError(fn.name, pred, "phi", "predecessor has no terminator")
        term = block.instructions[-1]
        if not isinstance(term, (muir.Br, muir.Ret, muir.Trap, muir.ArchEscape)):
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
    layout = DataLayout.from_module(text)
    out = []
    total = LegalizeStats()
    for fn in functions:
        lowered, stats = legalize_function(fn, layout)
        out.append(lowered)
        for key, value in stats.as_dict().items():
            setattr(total, key, getattr(total, key) + value)
    return out, total
