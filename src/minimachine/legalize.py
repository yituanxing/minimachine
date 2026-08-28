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
    dropped_compiler_barriers: int = 0
    lowered_linux_bug: int = 0
    dropped_pause_hints: int = 0
    lowered_counter_reads: int = 0
    lowered_plain_asm_memory: int = 0
    lowered_identity_asm: int = 0
    lowered_divzero_constant: int = 0
    lowered_system_fence: int = 0
    lowered_system_state: int = 0
    lowered_system_tlb: int = 0
    lowered_system_wait: int = 0
    lowered_system_atomic: int = 0
    lowered_static_branch: int = 0
    lowered_system_ecall: int = 0
    lowered_system_faultable: int = 0
    lowered_system_lrsc: int = 0
    lowered_cpu_feature_branch: int = 0
    lowered_system_icache: int = 0
    lowered_system_vector: int = 0
    lowered_alternative_tlb: int = 0
    lowered_generic_csr: int = 0
    lowered_faultable_atomic: int = 0
    lowered_read_sp: int = 0
    lowered_read_thread_pointer: int = 0
    lowered_indirectbr: int = 0
    lowered_undef_indirectbr: int = 0
    lowered_switch: int = 0

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
            "dropped_compiler_barriers": self.dropped_compiler_barriers,
            "lowered_linux_bug": self.lowered_linux_bug,
            "dropped_pause_hints": self.dropped_pause_hints,
            "lowered_counter_reads": self.lowered_counter_reads,
            "lowered_plain_asm_memory": self.lowered_plain_asm_memory,
            "lowered_identity_asm": self.lowered_identity_asm,
            "lowered_divzero_constant": self.lowered_divzero_constant,
            "lowered_system_fence": self.lowered_system_fence,
            "lowered_system_state": self.lowered_system_state,
            "lowered_system_tlb": self.lowered_system_tlb,
            "lowered_system_wait": self.lowered_system_wait,
            "lowered_system_atomic": self.lowered_system_atomic,
            "lowered_static_branch": self.lowered_static_branch,
            "lowered_system_ecall": self.lowered_system_ecall,
            "lowered_system_faultable": self.lowered_system_faultable,
            "lowered_system_lrsc": self.lowered_system_lrsc,
            "lowered_cpu_feature_branch": self.lowered_cpu_feature_branch,
            "lowered_system_icache": self.lowered_system_icache,
            "lowered_system_vector": self.lowered_system_vector,
            "lowered_alternative_tlb": self.lowered_alternative_tlb,
            "lowered_generic_csr": self.lowered_generic_csr,
            "lowered_faultable_atomic": self.lowered_faultable_atomic,
            "lowered_read_sp": self.lowered_read_sp,
            "lowered_read_thread_pointer": self.lowered_read_thread_pointer,
            "lowered_indirectbr": self.lowered_indirectbr,
            "lowered_undef_indirectbr": self.lowered_undef_indirectbr,
            "lowered_switch": self.lowered_switch,
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


def _strip_memory_qualifiers(text: str) -> str:
    value = text.strip()
    while True:
        changed = False
        for qualifier in ("volatile ", "atomic "):
            if value.startswith(qualifier):
                value = value[len(qualifier):].lstrip()
                changed = True
        if not changed:
            return value


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


def _parse_switch(inst: TextInst):
    text = re.sub(r"\s+", " ", inst.text.strip())
    m = re.fullmatch(
        r"switch\s+i(\d+)\s+(.+?),\s*label\s+%([-A-Za-z$._0-9]+)\s*\[(.*)\]",
        text,
    )
    if not m:
        raise ValueError(f"cannot parse switch: {inst.text}")

    bits = int(m.group(1))
    selector = _value(m.group(2))
    default = m.group(3)
    case_body = m.group(4).strip()
    cases: list[tuple[int, str]] = []

    if case_body:
        case_re = re.compile(
            rf"i{bits}\s+(-?[0-9]+)\s*,\s*label\s+%([-A-Za-z$._0-9]+)"
        )
        pos = 0
        for cm in case_re.finditer(case_body):
            between = case_body[pos:cm.start()].strip()
            if between:
                raise ValueError(f"cannot parse switch cases: {inst.text}")
            cases.append((int(cm.group(1)), cm.group(2)))
            pos = cm.end()
        if case_body[pos:].strip():
            raise ValueError(f"cannot parse switch cases: {inst.text}")

    return bits, selector, default, cases


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


def _load_store_address_uses(fn: TextFunction) -> set[str]:
    """Results whose only use is a load/store memory address.

    Constant GEP folding is only valid when the GEP value is consumed as the
    address operand itself.  A pointer stored *as data* must still be
    materialized; otherwise the store reads an uninitialized frame slot.
    """

    addresses: set[str] = set()
    for block in fn.blocks:
        for inst in block.instructions:
            if inst.opcode not in {"load", "store"}:
                continue

            body = inst.text[len(inst.opcode) :].lstrip()
            parts = _split_top_commas(body)
            if len(parts) < 2:
                continue

            pm = re.match(
                r"ptr(?:\s+addrspace\(\d+\))?\s+(%[-A-Za-z$._0-9]+)\s*$",
                parts[1].strip(),
            )
            if pm:
                addresses.add(pm.group(1))

    return addresses


def _result_defs(fn: TextFunction) -> dict[str, TextInst]:
    return {
        inst.result: inst
        for block in fn.blocks
        for inst in block.instructions
        if inst.result is not None
    }


def _fusable_icmp_results(
    fn: TextFunction,
    uses: Counter[str],
    defs: dict[str, TextInst],
) -> set[str]:
    fused: set[str] = set()
    for block in fn.blocks:
        for inst in block.instructions:
            if inst.opcode != "br":
                continue
            m = re.search(
                r"br\s+i1\s+(%[-A-Za-z$._0-9]+)\s*,\s*label",
                inst.text,
            )
            if not m:
                continue
            cond = m.group(1)
            defined = defs.get(cond)
            if (
                uses[cond] == 1
                and defined is not None
                and defined.opcode == "icmp"
            ):
                fused.add(cond)
    return fused


def _split_top_commas(text: str) -> list[str]:
    parts = []
    start = 0
    stack = []
    close_to_open = {")": "(", "]": "[", "}": "{", ">": "<"}
    in_string = False
    escape = False

    for i, c in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue

        if c == '"':
            in_string = True
        elif c in "([{<":
            stack.append(c)
        elif c in close_to_open:
            if stack and stack[-1] == close_to_open[c]:
                stack.pop()
        elif c == "," and not stack:
            parts.append(text[start:i].strip())
            start = i + 1

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


def _const_scalar_value(text: str, layout: DataLayout) -> muir.Value:
    text = text.strip()
    if text.startswith("getelementptr"):
        return _const_gep_value(text, layout)

    cast = re.match(
        r"^(ptrtoint|inttoptr|bitcast|addrspacecast)\s*\((.+)\s+to\s+.+\)$",
        text,
    )
    if cast:
        _ty, value_text = _split_typed_value(cast.group(2).strip())
        return _const_scalar_value(value_text, layout)

    binary = re.match(r"^(add|sub)(?:\s+\w+)*\s*\((.*)\)$", text)
    if binary:
        parts = _split_top_commas(binary.group(2))
        if len(parts) != 2:
            raise ValueError(f"cannot parse constant expression: {text}")
        _lhs_ty, lhs_text = _split_typed_value(parts[0])
        _rhs_ty, rhs_text = _split_typed_value(parts[1])
        lhs = _const_scalar_value(lhs_text, layout)
        rhs = _const_scalar_value(rhs_text, layout)

        if isinstance(lhs, muir.Imm) and isinstance(rhs, muir.Imm):
            value = (
                lhs.value + rhs.value
                if binary.group(1) == "add"
                else lhs.value - rhs.value
            )
            return muir.Imm(value)

        if isinstance(lhs, muir.Symbol):
            lhs = muir.Reloc(lhs.name, 0)
        if isinstance(rhs, muir.Symbol):
            rhs = muir.Reloc(rhs.name, 0)

        if isinstance(lhs, muir.Reloc) and isinstance(rhs, muir.Imm):
            delta = rhs.value if binary.group(1) == "add" else -rhs.value
            return muir.Reloc(lhs.symbol, lhs.addend + delta)
        if (
            binary.group(1) == "add"
            and isinstance(lhs, muir.Imm)
            and isinstance(rhs, muir.Reloc)
        ):
            return muir.Reloc(rhs.symbol, rhs.addend + lhs.value)

        raise ValueError(f"unsupported relocatable constant expression: {text}")

    return _value(text)


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


def _inline_asm_template(text: str) -> str | None:
    m = re.search(r'\basm\b(?:\s+\w+)*\s+"((?:\\.|[^"])*)"', text)
    if not m:
        return None
    template = m.group(1)
    template = template.replace(r"\0A", "\n").replace(r"\09", "\t")
    return template


def _is_linux_bug_asm(template: str) -> bool:
    return "ebreak" in template and "__bug_table" in template


def _normalize_inline_asm(template: str) -> str:
    return re.sub(r"\s+", " ", template.replace("\t", " ").strip())


def _counter_read_sysop(template: str) -> str | None:
    normalized = _normalize_inline_asm(template)
    mapping = {
        "csrr $0, 0xc00": "counter_cycle",
        "csrr $0, cycle": "counter_cycle",
        "csrr $0, 0xc01": "counter_time",
        "csrr $0, time": "counter_time",
        "csrr $0, 0xc02": "counter_instret",
        "csrr $0, instret": "counter_instret",
    }
    return mapping.get(normalized)


_FENCE_BITS = {"i": 1, "o": 2, "r": 4, "w": 8}


def _fence_mask(spec: str) -> int:
    mask = 0
    for ch in spec:
        mask |= _FENCE_BITS[ch]
    return mask


def _lower_simple_fence_sys(template: str, result: muir.Slot | None) -> muir.Sys | None:
    if result is not None:
        return None
    normalized = _normalize_inline_asm(template).rstrip(";").strip()
    m = re.fullmatch(r"fence\s+([iorw]+)\s*,\s*([iorw]+)", normalized)
    if not m:
        return None
    return muir.Sys(
        "fence",
        (muir.Imm(_fence_mask(m.group(1))), muir.Imm(_fence_mask(m.group(2)))),
        None,
    )


_STATE_NAMES = {
    "sstatus": "status",
    "0x100": "status",
    "sie": "interrupt_enable",
    "0x104": "interrupt_enable",
    "stvec": "trap_vector",
    "0x105": "trap_vector",
    "senvcfg": "env_config",
    "0x10a": "env_config",
    "sscratch": "scratch",
    "0x140": "scratch",
    "sepc": "exception_pc",
    "0x141": "exception_pc",
    "scause": "exception_cause",
    "0x142": "exception_cause",
    "stval": "exception_value",
    "0x143": "exception_value",
    "sip": "interrupt_pending",
    "0x144": "interrupt_pending",
    "stimecmp": "timer_compare",
    "0x14d": "timer_compare",
    "satp": "address_space",
    "0x180": "address_space",
}


def _state_name(token: str) -> str | None:
    return _STATE_NAMES.get(token.lower())


def _lower_simple_state_sys(
    template: str,
    text: str,
    result: muir.Slot | None,
    layout: DataLayout,
) -> muir.Sys | None:
    normalized = _normalize_inline_asm(template).rstrip(";").strip()
    args = _inline_asm_args(text, layout)

    m = re.fullmatch(r"csrr \$0,\s*([A-Za-z0-9x]+)", normalized)
    if m and result is not None and not args:
        state = _state_name(m.group(1))
        if state:
            return muir.Sys(f"state_read_{state}", (), result)

    for mnemonic, action in (
        ("csrw", "write"),
        ("csrs", "set"),
        ("csrc", "clear"),
    ):
        m = re.fullmatch(rf"{mnemonic}\s+([A-Za-z0-9x]+),\s*\$0", normalized)
        if m and result is None and len(args) == 1:
            state = _state_name(m.group(1))
            if state:
                return muir.Sys(f"state_{action}_{state}", args, None)

    for mnemonic, action in (
        ("csrrw", "swap"),
        ("csrrs", "read_set"),
        ("csrrc", "read_clear"),
    ):
        m = re.fullmatch(rf"{mnemonic}\s+\$0,\s*([A-Za-z0-9x]+),\s*\$1", normalized)
        if m and result is not None and len(args) == 1:
            state = _state_name(m.group(1))
            if state:
                return muir.Sys(f"state_{action}_{state}", args, result)

    return None


def _eval_csr_const(expr: str) -> int | None:
    expr = expr.strip()
    if not re.fullmatch(r"(?:0x[0-9A-Fa-f]+|[0-9]+)(?:\s*\+\s*(?:0x[0-9A-Fa-f]+|[0-9]+))*", expr):
        return None
    return sum(int(part.strip(), 0) for part in expr.split("+"))


def _lower_generic_csr_sys(
    template: str,
    text: str,
    result: muir.Slot | None,
    layout: DataLayout,
) -> muir.Sys | None:
    normalized = _normalize_inline_asm(template).rstrip(";").strip()
    args = _inline_asm_args(text, layout)

    m = re.fullmatch(r"csrr\s+\$0,\s*(.+)", normalized)
    if m and result is not None and not args:
        csr = _eval_csr_const(m.group(1))
        if csr is not None:
            return muir.Sys("csr_read", (muir.Imm(csr),), result)

    for mnemonic, op in (
        ("csrw", "csr_write"),
        ("csrs", "csr_set"),
        ("csrc", "csr_clear"),
    ):
        m = re.fullmatch(rf"{mnemonic}\s+(.+?),\s*\$0", normalized)
        if m and result is None and len(args) == 1:
            csr = _eval_csr_const(m.group(1))
            if csr is not None:
                return muir.Sys(op, (muir.Imm(csr), args[0]), None)

    for mnemonic, op in (
        ("csrrw", "csr_swap"),
        ("csrrs", "csr_read_set"),
        ("csrrc", "csr_read_clear"),
    ):
        m = re.fullmatch(rf"{mnemonic}\s+\$0,\s*(.+?),\s*\$1", normalized)
        if m and result is not None and len(args) == 1:
            csr = _eval_csr_const(m.group(1))
            if csr is not None:
                return muir.Sys(op, (muir.Imm(csr), args[0]), result)

    return None


def _lower_simple_tlb_sys(
    template: str,
    text: str,
    result: muir.Slot | None,
    layout: DataLayout,
) -> muir.Sys | None:
    if result is not None:
        return None
    normalized = _normalize_inline_asm(template).rstrip(";").strip()
    args = _inline_asm_args(text, layout)

    if normalized == "sfence.vma" and not args:
        return muir.Sys("tlb_flush_all", (), None)
    if normalized == "sfence.vma $0" and len(args) == 1:
        return muir.Sys("tlb_flush_address", args, None)
    if normalized == "sfence.vma x0, $0" and len(args) == 1:
        return muir.Sys("tlb_flush_asid", args, None)
    if normalized == "sfence.vma $0, $1" and len(args) == 2:
        return muir.Sys("tlb_flush_address_asid", args, None)
    return None


def _lower_simple_atomic_sys(
    template: str,
    text: str,
    result: muir.Slot | None,
    layout: DataLayout,
) -> muir.Sys | None:
    normalized = _normalize_inline_asm(template).strip().rstrip(";").strip()
    post_acquire_fence = False
    m_fence = re.fullmatch(
        r"(.+?)\s*(?:;\s*)?fence\s+r\s*,\s*rw",
        normalized,
    )
    if m_fence:
        normalized = m_fence.group(1).strip()
        post_acquire_fence = True

    m = re.fullmatch(
        r"amo(add|or|and|xor|swap)\.(w|d)(?:\.(aqrl|aq|rl))?\s+"
        r"(zero|\$\d+),\s*\$\d+,\s*\$\d+",
        normalized,
    )
    if not m:
        return None

    kind, width_tag, ordering, dest = m.groups()
    args = _inline_asm_args(text, layout)
    if len(args) != 3 or args[0] != args[2]:
        return None
    address, value, _ = args

    if dest == "zero":
        if result is not None:
            return None
    elif result is None:
        return None

    bits = 32 if width_tag == "w" else 64
    order = {
        None: "relaxed",
        "aq": "acquire",
        "rl": "release",
        "aqrl": "acq_rel",
    }[ordering]
    if post_acquire_fence:
        order = order + "_post_acquire_fence"

    return muir.Sys(
        f"atomic_{kind}_i{bits}_{order}",
        (address, value),
        result,
    )


_FAULT_WIDTH = {
    "lb": 8,
    "lh": 16,
    "lw": 32,
    "ld": 64,
    "sb": 8,
    "sh": 16,
    "sw": 32,
    "sd": 64,
}


def _faultable_sys_spec(template: str) -> tuple[str, int] | None:
    if "__ex_table" not in template:
        return None
    normalized = _normalize_inline_asm(template)
    m = re.match(r"1:\s*;?\s*(lb|lh|lw|ld|sb|sh|sw|sd)\b", normalized)
    if not m:
        return None
    mnemonic = m.group(1)
    kind = "load" if mnemonic.startswith("l") else "store"
    return kind, _FAULT_WIDTH[mnemonic]


def _faultable_atomic_sys_spec(template: str) -> tuple[str, int, str] | None:
    if "__ex_table" not in template:
        return None
    normalized = _normalize_inline_asm(template)

    m = re.search(r"\bamo(add|and|or|xor|swap)\.(w|d)\.aqrl\b", normalized)
    if m:
        kind, width_tag = m.groups()
        return kind, 32 if width_tag == "w" else 64, "acq_rel"

    m = re.search(r"\blr\.(w|d)\.aqrl\b", normalized)
    if m and re.search(r"\bsc\.[wd]\.aqrl\b", normalized) and re.search(r"\bbne\b", normalized):
        return "cmpxchg", 32 if m.group(1) == "w" else 64, "acq_rel"

    return None


def _lrsc_sys_spec(template: str) -> tuple[str, int, str] | None:
    normalized = _normalize_inline_asm(template).strip()
    pre_release = False
    m_pre = re.match(
        r"fence\s+rw\s*,\s*w(?=\s*(?:;\s*)?0:)",
        normalized,
    )
    if m_pre:
        pre_release = True
        normalized = normalized[m_pre.end():].lstrip(" ;")

    m = re.match(r"0:\s*;?\s*lr\.(w|d)\b", normalized)
    if not m:
        return None
    bits = 32 if m.group(1) == "w" else 64

    if (
        re.search(r"\bbltz\b", normalized)
        and re.search(r"addi\s+\$1,\s*\$0,\s*1\b", normalized)
        and re.search(r"\bsc\.[wd]", normalized)
    ):
        kind = "add1_if_nonnegative"
    elif (
        re.search(r"\bbgtz\b", normalized)
        and re.search(r"addi\s+\$1,\s*\$0,\s*-1\b", normalized)
        and re.search(r"\bsc\.[wd]", normalized)
    ):
        kind = "sub1_if_nonpositive"
    elif re.search(r"\bbne\b", normalized) and re.search(r"\bsc\.[wd]", normalized):
        kind = "cmpxchg"
    elif re.search(r"\bbeq\b", normalized) and re.search(r"\badd\b", normalized):
        kind = "add_unless"
    elif "addi $1, $0, -1" in normalized and re.search(r"\bbltz\b", normalized):
        kind = "dec_if_positive"
    else:
        return None

    if re.search(r"\bsc\.[wd]\.rl\b", normalized) and "fence rw, rw" in normalized:
        ordering = "release_post_full_fence"
    elif pre_release:
        ordering = "pre_release"
    elif re.search(r"\bsc\.[wd]\b", normalized):
        ordering = "relaxed"
    else:
        return None
    return kind, bits, ordering


def _lower_alternative_tlb_sys(
    template: str,
    text: str,
    result: muir.Slot | None,
    layout: DataLayout,
) -> muir.Sys | None:
    if result is not None or ".alternative" not in template:
        return None
    normalized = _normalize_inline_asm(template)
    primary = normalized.split("887 :", 1)[0]
    args = _inline_asm_args(text, layout)

    if re.search(r"sfence\.vma\s+\$0\s*,\s*\$1", primary) and len(args) == 2:
        return muir.Sys("tlb_flush_address_asid", args, None)
    if re.search(r"sfence\.vma\s+x0\s*,\s*\$0", primary) and len(args) == 1:
        return muir.Sys("tlb_flush_asid", args, None)
    if re.search(r"sfence\.vma\s+\$0", primary) and len(args) == 1:
        return muir.Sys("tlb_flush_address", args, None)
    return None


def _lower_vector_system_sys(
    template: str,
    text: str,
    inst: TextInst,
    scalar_result: muir.Slot | None,
    layout: DataLayout,
    frame_slots: set[str],
    aggregate_results: dict[str, tuple[tuple[muir.Slot, muir.Width], ...]],
) -> muir.Sys | None:
    normalized = _normalize_inline_asm(template).rstrip(";").strip()

    if normalized == "csrr $0, 0xc22" and scalar_result is not None:
        return muir.Sys("vector_length_bytes", (), scalar_result)

    if (
        "csrr $0, 0x8" in normalized
        and "csrr $1, 0xc21" in normalized
        and "csrr $2, 0xc20" in normalized
        and "csrr $3, 0xf" in normalized
        and "csrr $4, 0xc22" in normalized
    ):
        sys_result, widths = _prepare_inline_asm_result(
            inst, scalar_result, frame_slots, aggregate_results
        )
        if len(widths) == 5:
            return muir.Sys("vector_state_snapshot", (), sys_result)

    if (
        "vsetvl x0, $2, $1" in normalized
        and "csrw 0x8, $0" in normalized
        and "csrw 0xf, $3" in normalized
    ):
        args = _inline_asm_args(text, layout)
        if scalar_result is None and len(args) == 4:
            return muir.Sys("vector_state_restore", args, None)

    return None


def _lower_wait_sys(template: str, result: muir.Slot | None) -> muir.Sys | None:
    if result is None and _normalize_inline_asm(template) == "wfi":
        return muir.Sys("wait_interrupt", (), None)
    return None


def _scan_quoted(text: str, start: int) -> int:
    assert text[start] == '"'
    i = start + 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == '"':
            return i
        i += 1
    raise ValueError("unterminated quoted inline-asm string")


def _inline_asm_constraints(text: str) -> str:
    asm = re.search(r"\basm\b", text)
    if not asm:
        raise ValueError("inline asm marker missing")
    q1 = text.find('"', asm.end())
    if q1 < 0:
        raise ValueError("inline asm template missing")
    q1e = _scan_quoted(text, q1)
    q2 = text.find('"', q1e + 1)
    if q2 < 0:
        raise ValueError("inline asm constraints missing")
    q2e = _scan_quoted(text, q2)
    return text[q2 + 1 : q2e]


def _inline_asm_args(text: str, layout: DataLayout) -> tuple[muir.Value, ...]:
    asm = re.search(r"\basm\b", text)
    if not asm:
        raise ValueError("inline asm marker missing")
    q1 = text.find('"', asm.end())
    if q1 < 0:
        raise ValueError("inline asm template missing")
    q1e = _scan_quoted(text, q1)
    q2 = text.find('"', q1e + 1)
    if q2 < 0:
        raise ValueError("inline asm constraints missing")
    q2e = _scan_quoted(text, q2)
    open_paren = text.find("(", q2e + 1)
    if open_paren < 0:
        raise ValueError("inline asm argument list missing")

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
        raise ValueError("unterminated inline asm argument list")

    body = text[open_paren + 1 : close].strip()
    if not body:
        return ()
    args = []
    for segment in _split_top_commas(body):
        if "getelementptr" in segment:
            pos = segment.find("getelementptr")
            args.append(_const_gep_value(segment[pos:], layout))
        else:
            args.append(_value(segment))
    return tuple(args)


_PLAIN_ASM_LOADS = {
    "lb $0, 0($1)": muir.Width.I8,
    "lh $0, 0($1)": muir.Width.I16,
    "lw $0, 0($1)": muir.Width.I32,
    "ld $0, 0($1)": muir.Width.I64,
}
_PLAIN_ASM_STORES = {
    "sb $0, 0($1)": muir.Width.I8,
    "sh $0, 0($1)": muir.Width.I16,
    "sw $0, 0($1)": muir.Width.I32,
    "sd $0, 0($1)": muir.Width.I64,
}


def _lower_plain_asm_memory(
    template: str,
    text: str,
    result: muir.Slot | None,
    layout: DataLayout,
) -> muir.Mov | None:
    normalized = _normalize_inline_asm(template)
    if normalized in _PLAIN_ASM_LOADS:
        if result is None:
            return None
        args = _inline_asm_args(text, layout)
        if len(args) != 1:
            return None
        width = _PLAIN_ASM_LOADS[normalized]
        return muir.Mov(width, result, muir.Mem(muir.Address(args[0], 0), width))

    if normalized in _PLAIN_ASM_STORES:
        if result is not None:
            return None
        args = _inline_asm_args(text, layout)
        if len(args) != 2:
            return None
        width = _PLAIN_ASM_STORES[normalized]
        value, address = args
        return muir.Mov(width, muir.Mem(muir.Address(address, 0), width), value)

    return None


def _lower_cpu_feature_callbr(
    inst: TextInst,
    layout: DataLayout,
    temp_counter: list[int],
    frame_slots: set[str],
) -> list[muir.Instr] | None:
    template = _inline_asm_template(inst.text)
    if template is None or ".alternative" not in template:
        return None
    normalized = _normalize_inline_asm(template)
    if "__jump_table" in normalized:
        return None

    labels = _LABEL_USE_RE.findall(inst.text)
    args = _inline_asm_args(inst.text, layout)
    if len(labels) != 2 or len(args) != 1:
        return None

    primary = normalized.split("887 :", 1)[0]
    replacement = normalized.split("888 :", 1)[1] if "888 :" in normalized else ""
    primary_jumps = re.search(r"\bj\s+\$\{?1(?::l)?\}?", primary) is not None
    replacement_jumps = re.search(r"\bj\s+\$\{?1(?::l)?\}?", replacement) is not None
    if primary_jumps == replacement_jumps:
        return None

    temp_counter[0] += 1
    cond = muir.Slot(f"__cpu_feature{temp_counter[0]}")
    frame_slots.add(cond.name)
    fallthrough, alt_target = labels

    # SYS cpu_feature returns 1 when the alternative feature is available.
    true_target = alt_target if replacement_jumps else fallthrough
    false_target = alt_target if primary_jumps else fallthrough

    return [
        muir.Sys("cpu_feature", (args[0],), cond),
        muir.Br(
            muir.Width.I8,
            muir.Cond.EQ,
            cond,
            muir.Imm(0),
            muir.Target(label=false_target),
            muir.Target(label=true_target),
        ),
    ]


def _lower_jump_label_callbr(
    inst: TextInst,
    layout: DataLayout,
    temp_counter: list[int],
    frame_slots: set[str],
) -> list[muir.Instr] | None:
    template = _inline_asm_template(inst.text)
    if template is None or "__jump_table" not in template:
        return None

    labels = _LABEL_USE_RE.findall(inst.text)
    if len(labels) != 2:
        return None
    args = _inline_asm_args(inst.text, layout)
    if len(args) != 1:
        return None

    temp_counter[0] += 1
    cond = muir.Slot(f"__static_branch{temp_counter[0]}")
    frame_slots.add(cond.name)

    fallthrough, patched_target = labels
    return [
        muir.Sys("static_branch", (args[0],), cond),
        muir.Br(
            muir.Width.I8,
            muir.Cond.EQ,
            cond,
            muir.Imm(0),
            muir.Target(label=fallthrough),
            muir.Target(label=patched_target),
        ),
    ]


def _call_return_types(text: str) -> tuple[muir.Width, ...]:
    m = re.search(r"\bcall\s+(.+?)\s+asm\b", text)
    if not m:
        raise ValueError(f"cannot parse inline-asm return type: {text}")
    ty = m.group(1).strip()
    if ty == "void":
        return ()
    if ty.startswith("{") and ty.endswith("}"):
        fields = _split_top_commas(ty[1:-1].strip())
        return tuple(_first_width(field, default_pointer=True) for field in fields)
    return (_first_width(ty, default_pointer=True),)


def _aggregate_result_slots(
    result_name: str,
    widths: tuple[muir.Width, ...],
    frame_slots: set[str],
) -> tuple[muir.Slot, ...]:
    base = result_name[1:] if result_name.startswith("%") else result_name
    slots = tuple(muir.Slot(f"{base}.__r{i}") for i in range(len(widths)))
    frame_slots.update(slot.name for slot in slots)
    return slots


def _prepare_inline_asm_result(
    inst: TextInst,
    scalar_result: muir.Slot | None,
    frame_slots: set[str],
    aggregate_results: dict[str, tuple[tuple[muir.Slot, muir.Width], ...]],
) -> tuple[muir.Result, tuple[muir.Width, ...]]:
    widths = _call_return_types(inst.text)
    if not widths:
        return None, ()
    if len(widths) == 1:
        if scalar_result is None:
            raise ValueError("scalar inline-asm result missing")
        return scalar_result, widths
    if inst.result is None:
        raise ValueError("aggregate inline-asm result missing")
    slots = _aggregate_result_slots(inst.result, widths, frame_slots)
    aggregate_results[inst.result] = tuple(zip(slots, widths))
    return slots, widths


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


def _split_typed_value(text: str) -> tuple[str, str]:
    raw = text.strip()
    if not raw:
        raise ValueError("empty typed value")

    if raw[0] in "[{<":
        stack: list[str] = []
        close_to_open = {"]": "[", "}": "{", ">": "<"}
        for i, ch in enumerate(raw):
            if ch in "[{<":
                stack.append(ch)
            elif ch in close_to_open:
                if not stack or stack[-1] != close_to_open[ch]:
                    raise ValueError(f"unbalanced typed value: {text}")
                stack.pop()
                if not stack:
                    ty = raw[: i + 1].strip()
                    value = raw[i + 1 :].strip()
                    if not value:
                        raise ValueError(f"typed value has no value: {text}")
                    return ty, value
        raise ValueError(f"unterminated aggregate type: {text}")

    m = re.match(r"(%[-A-Za-z$._0-9]+|ptr(?:\s+addrspace\(\d+\))?|i\d+|half|float|double|fp128)\s+(.+)$", raw)
    if not m:
        raise ValueError(f"cannot split typed value: {text}")
    return m.group(1).strip(), m.group(2).strip()


def _typed_aggregate_operand(text: str) -> tuple[str, muir.Value]:
    ty, value_text = _split_typed_value(text)
    if value_text.startswith(("[", "{", "<")):
        raise ValueError(f"aggregate literal needs materialization: {text}")
    return ty, _value(value_text)


def _literal_initializers(
    layout: DataLayout,
    ty: str,
    value_text: str,
    overwrite_path: tuple[int, ...] | None,
    base_offset: int = 0,
) -> list[tuple[int, int, muir.Value]]:
    value = value_text.strip()

    # Empty path means this entire subtree is replaced by insertvalue.
    if overwrite_path == ():
        return []
    if value == "zeroinitializer":
        return []
    if value in {"poison", "undef"}:
        if overwrite_path is not None:
            raise ValueError("poison/undef survives insertvalue")
        raise ValueError("live poison/undef aggregate literal")

    info = layout.info(ty)

    if info.fields is not None:
        if not (value.startswith("{") and value.endswith("}")):
            raise ValueError(f"expected struct literal for {ty}: {value_text}")
        fields = _split_top_commas(value[1:-1].strip())
        if len(fields) != len(info.fields):
            raise ValueError(f"struct literal field count mismatch: {value_text}")
        assert info.field_offsets is not None
        out: list[tuple[int, int, muir.Value]] = []
        selected = overwrite_path[0] if overwrite_path is not None else None
        rest = overwrite_path[1:] if overwrite_path is not None else None
        for i, (field_ty, field_text) in enumerate(zip(info.fields, fields)):
            parsed_ty, parsed_value = _split_typed_value(field_text)
            if parsed_ty != field_ty:
                raise ValueError(
                    f"struct literal type mismatch: expected {field_ty}, got {parsed_ty}"
                )
            child_path = rest if selected == i else None
            out.extend(
                _literal_initializers(
                    layout,
                    field_ty,
                    parsed_value,
                    child_path,
                    base_offset + info.field_offsets[i],
                )
            )
        return out

    if info.element is not None:
        if not (
            (value.startswith("[") and value.endswith("]"))
            or (value.startswith("<") and value.endswith(">"))
        ):
            raise ValueError(f"expected aggregate literal for {ty}: {value_text}")
        items = _split_top_commas(value[1:-1].strip())
        if info.count is None or len(items) != info.count:
            raise ValueError(f"aggregate literal item count mismatch: {value_text}")
        _, scale, _ = layout.gep_step(ty, 0)
        out: list[tuple[int, int, muir.Value]] = []
        selected = overwrite_path[0] if overwrite_path is not None else None
        rest = overwrite_path[1:] if overwrite_path is not None else None
        for i, item in enumerate(items):
            parsed_ty, parsed_value = _split_typed_value(item)
            if parsed_ty != info.element:
                raise ValueError(
                    f"aggregate literal type mismatch: expected {info.element}, got {parsed_ty}"
                )
            child_path = rest if selected == i else None
            out.extend(
                _literal_initializers(
                    layout,
                    info.element,
                    parsed_value,
                    child_path,
                    base_offset + scale * i,
                )
            )
        return out

    # Scalar leaf outside the overwritten path must be concrete.
    scalar = _value(value)
    if isinstance(scalar, muir.Arbitrary):
        raise ValueError("live poison/undef scalar in aggregate literal")
    if info.size > 8:
        raise ValueError(
            f"wide scalar aggregate literal needs explicit blob encoding: {ty}"
        )
    if isinstance(scalar, muir.Imm) and scalar.value == 0:
        return []
    return [(base_offset, info.size, scalar)]


def _insert_base_operand(
    layout: DataLayout,
    typed_text: str,
    indices: tuple[int, ...],
    temp_counter: list[int],
    frame_slots: set[str],
) -> tuple[str, muir.Value, list[muir.Instr]]:
    ty, value_text = _split_typed_value(typed_text)
    if not value_text.startswith(("[", "{", "<")):
        return ty, _value(value_text), []

    initializers = _literal_initializers(
        layout,
        ty,
        value_text,
        indices,
    )
    if not initializers:
        return ty, muir.Imm(0), []

    total_size = layout.info(ty).size
    temp_counter[0] += 1
    dst = muir.Slot(f"__agg_literal{temp_counter[0]}")
    frame_slots.add(dst.name)

    args: list[muir.Value] = [muir.Imm(total_size)]
    for offset, size, value in initializers:
        args.extend((muir.Imm(offset), value, muir.Imm(size)))

    return (
        ty,
        dst,
        [muir.Helper("__mm_aggregate_literal", tuple(args), dst)],
    )


def _aggregate_index_layout(
    layout: DataLayout,
    aggregate_ty: str,
    indices: tuple[int, ...],
) -> tuple[int, str, int, bool]:
    current = aggregate_ty.strip()
    offset = 0
    for index in indices:
        next_ty, scale, field_offset = layout.gep_step(current, index)
        if scale:
            offset += scale * index
        else:
            offset += field_offset
        current = next_ty
    info = layout.info(current)
    is_blob = (
        info.size > 8
        or info.fields is not None
        or info.element is not None
    )
    return offset, current, info.size, is_blob


def _materialize_wide_value(
    value: muir.Value,
    bits: int,
    out: list[muir.Instr],
    temp_counter: list[int],
    frame_slots: set[str],
) -> muir.Value:
    if bits <= 64 or not isinstance(value, muir.Imm):
        return value

    masked = value.value & ((1 << bits) - 1)
    chunks = tuple(
        muir.Imm((masked >> shift) & ((1 << 64) - 1))
        for shift in range(0, bits, 64)
    )
    temp_counter[0] += 1
    dst = muir.Slot(f"__wide_const{temp_counter[0]}")
    frame_slots.add(dst.name)
    out.append(
        muir.Helper(
            f"__mm_wide_const_{bits}",
            chunks,
            dst,
        )
    )
    return dst


def _register_metadata(text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("!") or " = !{!" not in line:
            continue
        left, right = line.split("=", 1)
        ident = left.strip()[1:]
        if not ident.isdigit():
            continue
        marker = '!{!"'
        if marker not in right:
            continue
        start_quote = right.find(marker) + len(marker)
        end_quote = right.find('"', start_quote)
        if end_quote < 0:
            continue
        out[int(ident)] = right[start_quote:end_quote]
    return out


def legalize_function(
    fn: TextFunction,
    layout: DataLayout,
    register_metadata: dict[int, str] | None = None,
) -> tuple[muir.Function, LegalizeStats]:
    stats = LegalizeStats()
    register_metadata = register_metadata or {}
    uses = _use_counts(fn)
    load_store_address_uses = _load_store_address_uses(fn)
    defs = _result_defs(fn)
    fusable_icmps = _fusable_icmp_results(fn, uses, defs)
    frame_slots = {arg[1:] for arg in fn.args}
    aliases: dict[str, muir.Address] = {}
    aggregate_results: dict[str, tuple[tuple[muir.Slot, muir.Width], ...]] = {}
    phis_by_target: dict[str, list[tuple[muir.Slot, muir.Value, str, muir.Width]]] = defaultdict(list)
    out_blocks: dict[str, muir.Block] = {}
    extra_blocks_after: dict[str, list[str]] = defaultdict(list)
    switch_edge_blocks: dict[tuple[str, str], list[str]] = defaultdict(list)
    used_block_labels = {b.label for b in fn.blocks}
    temp_counter = [0]

    def fresh_block_label(base: str) -> str:
        label = base
        index = 0
        while label in used_block_labels:
            index += 1
            label = f"{base}_{index}"
        used_block_labels.add(label)
        return label

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
                    if result is None:
                        raise ValueError("alloca has no result")
                    body = inst.text[len("alloca "):]
                    parts = _split_top_commas(body)
                    if not parts:
                        raise ValueError(f"cannot parse alloca: {inst.text}")

                    ty = parts[0].strip()
                    for prefix in ("inalloca ", "swifterror "):
                        if ty.startswith(prefix):
                            ty = ty[len(prefix):].strip()

                    info = layout.info(ty)
                    count: muir.Value = muir.Imm(1)
                    align = info.align

                    for part in parts[1:]:
                        item = part.strip()
                        am = re.fullmatch(r"align\s+(\d+)", item)
                        if am:
                            align = int(am.group(1))
                            continue
                        if item.startswith("addrspace("):
                            # MiniMachine v1 currently has a flat address space.
                            continue
                        count_ty, count_text = _split_typed_value(item)
                        if not re.fullmatch(r"i\d+", count_ty):
                            raise ValueError(
                                f"alloca count is not integer typed: {item}"
                            )
                        count = _value(count_text)

                    stats.temporary_helpers += 1
                    out.append(
                        muir.Helper(
                            "__mm_alloca",
                            (
                                muir.Imm(info.size),
                                count,
                                muir.Imm(align),
                            ),
                            result,
                        )
                    )
                    continue

                if op == "sub":
                    m = re.match(
                        r"sub(?:\s+\w+)*\s+(i(?:8|16|32|64))\s+(.+)$",
                        inst.text,
                    )
                    if not m or result is None:
                        raise ValueError(f"cannot parse sub: {inst.text}")
                    operands = _split_top_commas(m.group(2))
                    if len(operands) != 2:
                        raise ValueError(f"cannot parse sub operands: {inst.text}")
                    width = _width(int(m.group(1)[1:]))
                    out.append(
                        muir.Sub(
                            width,
                            result,
                            _const_scalar_value(operands[0], layout),
                            _const_scalar_value(operands[1], layout),
                        )
                    )
                    continue

                if op == "add":
                    m = re.match(r"add(?:\s+\w+)*\s+(i\d+)\s+(.+)$", inst.text)
                    if not m or result is None:
                        raise ValueError(f"cannot parse add: {inst.text}")
                    operands = _split_top_commas(m.group(2))
                    if len(operands) != 2:
                        raise ValueError(f"cannot parse add operands: {inst.text}")
                    bits = int(m.group(1)[1:])
                    a = _const_scalar_value(operands[0], layout)
                    b = _const_scalar_value(operands[1], layout)
                    if bits > 64:
                        a = _materialize_wide_value(
                            a, bits, out, temp_counter, frame_slots
                        )
                        b = _materialize_wide_value(
                            b, bits, out, temp_counter, frame_slots
                        )
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
                    if (
                        uses[inst.result] == 1
                        and inst.result in load_store_address_uses
                        and not dynamic_terms
                    ):
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
                    ty=_strip_memory_qualifiers(parts[0])
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
                        agg_size = layout.info(ty).size
                        out.append(
                            muir.Helper(
                                "__mm_load_aggregate",
                                (base, muir.Imm(agg_size)),
                                result,
                            )
                        )
                    continue

                if op == "store":
                    body=inst.text[len("store "):]
                    parts=_split_top_commas(body)
                    if len(parts) < 2:
                        raise ValueError(f"cannot parse store: {inst.text}")
                    value_part = _strip_memory_qualifiers(parts[0])
                    try:
                        ty, value_text = _split_typed_value(value_part)
                    except ValueError as e:
                        raise ValueError(
                            f"cannot parse store value: {inst.text}"
                        ) from e
                    pm=re.match(r"ptr(?:\s+addrspace\(\d+\))?\s+(.+)$",parts[1].strip())
                    if not pm:
                        raise ValueError(f"cannot parse store pointer: {inst.text}")
                    if value_text.startswith("getelementptr"):
                        src = _const_gep_value(value_text, layout)
                    else:
                        src = _value(value_text)
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
                                if bits > 64:
                                    src = _materialize_wide_value(
                                        src, bits, out, temp_counter, frame_slots
                                    )
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
                        agg_size = layout.info(ty).size
                        out.append(
                            muir.Helper(
                                "__mm_store_aggregate",
                                (base, src, muir.Imm(agg_size)),
                                None,
                            )
                        )
                    continue


                if op == "icmp":
                    if result is None:
                        raise ValueError("icmp has no result")
                    pred, bits, width, a, b = _parse_icmp(inst)
                    if inst.result and inst.result in fusable_icmps and width is not None:
                        # Only an actual one-use conditional BR owns this
                        # fusion. Other consumers need a materialized i1.
                        continue
                    if bits > 64:
                        a = _materialize_wide_value(
                            a, bits, out, temp_counter, frame_slots
                        )
                        b = _materialize_wide_value(
                            b, bits, out, temp_counter, frame_slots
                        )
                    stats.temporary_helpers += 1
                    out.append(muir.Helper(f"__mm_icmp_{pred}_{bits}", (a, b), result))
                    continue

                if op == "switch":
                    bits, selector, default_label, cases = _parse_switch(inst)
                    if bits > 64:
                        raise ValueError(
                            f"switch selector wider than one MiniMachine slot: i{bits}"
                        )
                    width = _storage_width(bits)
                    stem = _sanitize(block.label) or "entry"

                    edge_specs = [(value, target) for value, target in cases]
                    default_edge = fresh_block_label(
                        f"__switch_{stem}_default_edge"
                    )
                    default_target = muir.Target(label=default_label)
                    out_blocks[default_edge] = muir.Block(
                        default_edge,
                        [
                            muir.Br(
                                muir.Width.I8,
                                muir.Cond.EQ,
                                muir.Imm(0),
                                muir.Imm(0),
                                default_target,
                                default_target,
                            )
                        ],
                    )
                    switch_edge_blocks[(block.label, default_label)].append(
                        default_edge
                    )

                    case_edges: list[tuple[int, str, str]] = []
                    for index, (value, target_label) in enumerate(edge_specs):
                        edge_label = fresh_block_label(
                            f"__switch_{stem}_case{index}_edge"
                        )
                        target = muir.Target(label=target_label)
                        out_blocks[edge_label] = muir.Block(
                            edge_label,
                            [
                                muir.Br(
                                    muir.Width.I8,
                                    muir.Cond.EQ,
                                    muir.Imm(0),
                                    muir.Imm(0),
                                    target,
                                    target,
                                )
                            ],
                        )
                        switch_edge_blocks[(block.label, target_label)].append(
                            edge_label
                        )
                        case_edges.append((value, target_label, edge_label))

                    compare_labels = [
                        fresh_block_label(f"__switch_{stem}_cmp{index}")
                        for index in range(1, len(case_edges))
                    ]

                    for index, (value, _target_label, edge_label) in enumerate(
                        case_edges
                    ):
                        false_label = (
                            compare_labels[index]
                            if index < len(compare_labels)
                            else default_edge
                        )
                        branch = muir.Br(
                            width,
                            muir.Cond.EQ,
                            selector,
                            muir.Imm(value),
                            muir.Target(label=edge_label),
                            muir.Target(label=false_label),
                        )
                        if index == 0:
                            out.append(branch)
                        else:
                            compare_label = compare_labels[index - 1]
                            out_blocks[compare_label] = muir.Block(
                                compare_label, [branch]
                            )

                    if not case_edges:
                        out.append(
                            muir.Br(
                                muir.Width.I8,
                                muir.Cond.EQ,
                                muir.Imm(0),
                                muir.Imm(0),
                                muir.Target(label=default_edge),
                                muir.Target(label=default_edge),
                            )
                        )

                    extra_blocks_after[block.label].extend(compare_labels)
                    extra_blocks_after[block.label].extend(
                        edge_label for _, _, edge_label in case_edges
                    )
                    extra_blocks_after[block.label].append(default_edge)
                    stats.lowered_switch += 1
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
                    if cond_def and cond_def.opcode == "icmp" and cond_text in fusable_icmps:
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
                        out.append(
                            muir.Mov(
                                dst_width,
                                result,
                                _value(m.group(2)),
                                extend=op,
                                src_bits=src_bits,
                            )
                        )
                    else:
                        src_value = _value(m.group(2))
                        if src_bits > 64:
                            src_value = _materialize_wide_value(
                                src_value,
                                src_bits,
                                out,
                                temp_counter,
                                frame_slots,
                            )
                        stats.temporary_helpers += 1
                        out.append(
                            muir.Helper(
                                f"__mm_{op}_{src_bits}_{dst_bits}",
                                (src_value,),
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
                                if bits > 64:
                                    value = _materialize_wide_value(
                                        value,
                                        bits,
                                        out,
                                        temp_counter,
                                        frame_slots,
                                    )
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
                    type_text = re.sub(
                        r"\s+(?:%[-A-Za-z$._0-9]+|@[-A-Za-z$._0-9]+|-?\d+|poison|undef|zeroinitializer)\s*$",
                        "",
                        parts[1],
                    ).strip()
                    type_tag=_sanitize(type_text)
                    wm = re.fullmatch(r"i(\d+)", type_text)
                    if wm and int(wm.group(1)) > 64:
                        wide_bits = int(wm.group(1))
                        tv = _materialize_wide_value(
                            tv, wide_bits, out, temp_counter, frame_slots
                        )
                        fv = _materialize_wide_value(
                            fv, wide_bits, out, temp_counter, frame_slots
                        )
                    stats.temporary_helpers += 1
                    out.append(muir.Helper(f"__mm_select_{type_tag or 'opaque'}",(cond,tv,fv),result))
                    continue


                if op in {"and", "or", "xor", "shl", "lshr", "ashr", "mul", "udiv", "sdiv", "urem", "srem"}:
                    m = re.match(
                        rf"{op}(?:\s+\w+)*\s+(i\d+)\s+(.+)$",
                        inst.text,
                    )
                    if not m or result is None:
                        raise ValueError(f"cannot parse helper op {op}: {inst.text}")
                    operands = _split_top_commas(m.group(2))
                    if len(operands) != 2:
                        raise ValueError(
                            f"cannot parse helper operands {op}: {inst.text}"
                        )
                    width = int(m.group(1)[1:])
                    lhs = _const_scalar_value(operands[0], layout)
                    rhs = _const_scalar_value(operands[1], layout)
                    if width > 64:
                        lhs = _materialize_wide_value(
                            lhs, width, out, temp_counter, frame_slots
                        )
                        rhs = _materialize_wide_value(
                            rhs, width, out, temp_counter, frame_slots
                        )
                    stats.temporary_helpers += 1
                    out.append(
                        muir.Helper(
                            f"__mm_{op}_{width}",
                            (lhs, rhs),
                            result,
                        )
                    )
                    continue

                if op == "call":
                    kind, symbol, args, callee_slot = _parse_call(inst, layout)

                    if (
                        kind == "direct"
                        and symbol is not None
                        and symbol.startswith("llvm.read_register.")
                        and result is not None
                    ):
                        mm = re.search(r"metadata\s+!(\d+)", inst.text)
                        reg_name = (
                            register_metadata.get(int(mm.group(1)))
                            if mm is not None
                            else None
                        )
                        if reg_name in {"sp", "x2"}:
                            stats.lowered_read_sp += 1
                            out.append(
                                muir.Mov(
                                    muir.Width.I64,
                                    result,
                                    muir.Special.SP,
                                )
                            )
                            continue
                        if reg_name in {"tp", "x4"}:
                            stats.lowered_read_thread_pointer += 1
                            out.append(
                                muir.Sys(
                                    "thread_pointer",
                                    (),
                                    result,
                                )
                            )
                            continue

                    if kind == "inline_asm":
                        template = _inline_asm_template(inst.text)
                        if result is None and template is not None and not template.strip():
                            # LLVM's empty sideeffect asm with a memory clobber
                            # is a compiler barrier. O2 has already observed it,
                            # and MiniMachine's P3 backend performs no memory-op
                            # reordering, so no runtime instruction is required.
                            stats.dropped_compiler_barriers += 1
                            continue
                        if result is not None and template is not None and not template.strip():
                            constraints = _inline_asm_constraints(inst.text)
                            args = _inline_asm_args(inst.text, layout)
                            if constraints == "=r,0" and len(args) == 1:
                                stats.lowered_identity_asm += 1
                                out.append(muir.Mov(muir.Width.I64, result, args[0]))
                                continue
                        if result is not None and template is not None and _normalize_inline_asm(template) == "div $0, $0, zero":
                            constraints = _inline_asm_constraints(inst.text)
                            args = _inline_asm_args(inst.text, layout)
                            if constraints == "=r" and not args:
                                stats.lowered_divzero_constant += 1
                                out.append(muir.Mov(muir.Width.I32, result, muir.Imm(-1)))
                                continue
                        if result is None and template is not None and _normalize_inline_asm(template) == "pause":
                            stats.dropped_pause_hints += 1
                            continue
                        if result is not None and template is not None:
                            counter_op = _counter_read_sysop(template)
                            if counter_op is not None:
                                stats.lowered_counter_reads += 1
                                out.append(muir.Sys(counter_op, (), result))
                                continue
                        if template is not None:
                            plain_mem = _lower_plain_asm_memory(template, inst.text, result, layout)
                            if plain_mem is not None:
                                stats.lowered_plain_asm_memory += 1
                                out.append(plain_mem)
                                continue

                            fence_sys = _lower_simple_fence_sys(template, result)
                            if fence_sys is not None:
                                stats.lowered_system_fence += 1
                                out.append(fence_sys)
                                continue

                            state_sys = _lower_simple_state_sys(template, inst.text, result, layout)
                            if state_sys is not None:
                                stats.lowered_system_state += 1
                                out.append(state_sys)
                                continue

                            generic_csr_sys = _lower_generic_csr_sys(
                                template, inst.text, result, layout
                            )
                            if generic_csr_sys is not None:
                                stats.lowered_generic_csr += 1
                                out.append(generic_csr_sys)
                                continue

                            if _normalize_inline_asm(template).rstrip(";").strip() == "fence.i" and result is None:
                                stats.lowered_system_icache += 1
                                out.append(muir.Sys("icache_sync", (), None))
                                continue

                            vector_sys = _lower_vector_system_sys(
                                template,
                                inst.text,
                                inst,
                                result,
                                layout,
                                frame_slots,
                                aggregate_results,
                            )
                            if vector_sys is not None:
                                stats.lowered_system_vector += 1
                                out.append(vector_sys)
                                continue

                            alt_tlb_sys = _lower_alternative_tlb_sys(
                                template, inst.text, result, layout
                            )
                            if alt_tlb_sys is not None:
                                stats.lowered_alternative_tlb += 1
                                out.append(alt_tlb_sys)
                                continue

                            tlb_sys = _lower_simple_tlb_sys(template, inst.text, result, layout)
                            if tlb_sys is not None:
                                stats.lowered_system_tlb += 1
                                out.append(tlb_sys)
                                continue

                            atomic_sys = _lower_simple_atomic_sys(template, inst.text, result, layout)
                            if atomic_sys is not None:
                                stats.lowered_system_atomic += 1
                                out.append(atomic_sys)
                                continue

                            wait_sys = _lower_wait_sys(template, result)
                            if wait_sys is not None:
                                stats.lowered_system_wait += 1
                                out.append(wait_sys)
                                continue

                            normalized_asm = _normalize_inline_asm(template).rstrip(";").strip()
                            if normalized_asm == "ecall":
                                sys_result, _ = _prepare_inline_asm_result(
                                    inst, result, frame_slots, aggregate_results
                                )
                                ecall_args = _inline_asm_args(inst.text, layout)
                                out.append(muir.Sys("ecall", ecall_args, sys_result))
                                stats.lowered_system_ecall += 1
                                continue

                            fault_spec = _faultable_sys_spec(template)
                            if fault_spec is not None:
                                kind, bits = fault_spec
                                sys_result, _ = _prepare_inline_asm_result(
                                    inst, result, frame_slots, aggregate_results
                                )
                                fault_args = _inline_asm_args(inst.text, layout)
                                out.append(
                                    muir.Sys(
                                        f"faultable_{kind}_i{bits}",
                                        fault_args,
                                        sys_result,
                                    )
                                )
                                stats.lowered_system_faultable += 1
                                continue

                            fault_atomic_spec = _faultable_atomic_sys_spec(template)
                            if fault_atomic_spec is not None:
                                kind, bits, ordering = fault_atomic_spec
                                sys_result, _ = _prepare_inline_asm_result(
                                    inst, result, frame_slots, aggregate_results
                                )
                                fault_atomic_args = _inline_asm_args(inst.text, layout)
                                # Linux futex asm carries an error accumulator
                                # initialized to zero and repeats the memory
                                # operand. The MiniMachine system contract
                                # exposes only semantic inputs.
                                if (
                                    len(fault_atomic_args) >= 4
                                    and fault_atomic_args[-1] == fault_atomic_args[0]
                                    and isinstance(fault_atomic_args[-2], muir.Imm)
                                    and fault_atomic_args[-2].value == 0
                                ):
                                    fault_atomic_args = fault_atomic_args[:-2]
                                out.append(
                                    muir.Sys(
                                        f"faultable_atomic_{kind}_i{bits}_{ordering}",
                                        fault_atomic_args,
                                        sys_result,
                                    )
                                )
                                stats.lowered_faultable_atomic += 1
                                continue

                            lrsc_spec = _lrsc_sys_spec(template)
                            if lrsc_spec is not None:
                                kind, bits, ordering = lrsc_spec
                                sys_result, _ = _prepare_inline_asm_result(
                                    inst, result, frame_slots, aggregate_results
                                )
                                lrsc_args = _inline_asm_args(inst.text, layout)
                                # Linux's constraints repeat the memory operand
                                # as the final input. The system contract needs
                                # one address, not two aliases of the same cell.
                                if len(lrsc_args) >= 2 and lrsc_args[-1] == lrsc_args[0]:
                                    lrsc_args = lrsc_args[:-1]
                                out.append(
                                    muir.Sys(
                                        f"atomic_{kind}_i{bits}_{ordering}",
                                        lrsc_args,
                                        sys_result,
                                    )
                                )
                                stats.lowered_system_lrsc += 1
                                continue
                        if result is None and template is not None and _is_linux_bug_asm(template):
                            # Linux BUG() is a deliberate non-returning trap.
                            # Preserve the runtime fault, not the RISC-V ebreak
                            # encoding or __bug_table assembler metadata.
                            stats.lowered_linux_bug += 1
                            out.append(muir.Trap("linux_bug"))
                            # BUG() is non-returning. LLVM inline asm itself
                            # does not encode that fact strongly enough for
                            # every optimization shape, so dead tail may still
                            # remain in the textual block. Semantic legalization
                            # makes the control-flow contract explicit here.
                            break
                        stats.arch_escapes += 1
                        out.append(muir.ArchEscape("inline_asm", inst.text))
                        continue
                    assert kind in {"direct", "indirect"}
                    if symbol and symbol.startswith(_DROP_INTRINSICS):
                        continue
                    if symbol == "llvm.va_start":
                        stats.temporary_helpers += 1
                        out.append(
                            muir.Helper(
                                "__mm_llvm_va_start",
                                args + (muir.Imm(len(fn.args)),),
                                None,
                            )
                        )
                        continue
                    if symbol == "llvm.va_copy":
                        stats.temporary_helpers += 1
                        out.append(
                            muir.Helper(
                                "__mm_llvm_va_copy",
                                args,
                                None,
                            )
                        )
                        continue
                    if symbol == "llvm.va_end":
                        stats.temporary_helpers += 1
                        out.append(
                            muir.Helper(
                                "__mm_llvm_va_end",
                                args,
                                None,
                            )
                        )
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
                    if out and isinstance(out[-1], muir.Trap):
                        continue
                    out.append(muir.Trap("llvm.unreachable"))
                    continue

                if op == "extractvalue":
                    if result is None:
                        raise ValueError("extractvalue has no result")

                    # Aggregate results from multi-result SYS/CALL already have
                    # concrete result slots and do not use blob storage.
                    m = re.search(
                        r"extractvalue\s+.+?\s+(%[-A-Za-z$._0-9]+)\s*,\s*(\d+)\s*$",
                        inst.text,
                    )
                    if m and m.group(1) in aggregate_results:
                        source, index_text = m.groups()
                        index = int(index_text)
                        fields = aggregate_results[source]
                        if index < 0 or index >= len(fields):
                            raise ValueError(f"extractvalue index out of range: {inst.text}")
                        src_slot, width = fields[index]
                        out.append(muir.Mov(width, result, src_slot))
                        continue

                    body = inst.text[len("extractvalue "):]
                    parts = _split_top_commas(body)
                    if len(parts) < 2:
                        raise ValueError(f"cannot parse extractvalue: {inst.text}")
                    aggregate_ty, aggregate_value = _typed_aggregate_operand(parts[0])
                    try:
                        indices = tuple(int(x.strip(), 0) for x in parts[1:])
                    except ValueError as e:
                        raise ValueError(f"non-constant extractvalue index: {inst.text}") from e
                    offset, field_ty, field_size, field_is_blob = _aggregate_index_layout(
                        layout, aggregate_ty, indices
                    )
                    stats.temporary_helpers += 1
                    out.append(
                        muir.Helper(
                            "__mm_extractvalue",
                            (
                                aggregate_value,
                                muir.Imm(offset),
                                muir.Imm(field_size),
                                muir.Imm(1 if field_is_blob else 0),
                            ),
                            result,
                        )
                    )
                    continue

                if op == "insertvalue":
                    if result is None:
                        raise ValueError("insertvalue has no result")
                    body = inst.text[len("insertvalue "):]
                    parts = _split_top_commas(body)
                    if len(parts) < 3:
                        raise ValueError(f"cannot parse insertvalue: {inst.text}")
                    try:
                        indices = tuple(int(x.strip(), 0) for x in parts[2:])
                    except ValueError as e:
                        raise ValueError(f"non-constant insertvalue index: {inst.text}") from e
                    aggregate_ty, aggregate_value, base_prelude = _insert_base_operand(
                        layout,
                        parts[0],
                        indices,
                        temp_counter,
                        frame_slots,
                    )
                    out.extend(base_prelude)
                    _, inserted_value = _typed_aggregate_operand(parts[1])

                    aggregate_size = layout.info(aggregate_ty).size
                    offset, field_ty, field_size, field_is_blob = _aggregate_index_layout(
                        layout, aggregate_ty, indices
                    )
                    stats.temporary_helpers += 1
                    out.append(
                        muir.Helper(
                            "__mm_insertvalue",
                            (
                                aggregate_value,
                                muir.Imm(aggregate_size),
                                muir.Imm(offset),
                                inserted_value,
                                muir.Imm(field_size),
                                muir.Imm(1 if field_is_blob else 0),
                            ),
                            result,
                        )
                    )
                    continue

                if op == "indirectbr":
                    labels = _LABEL_USE_RE.findall(inst.text)
                    if not labels:
                        raise ValueError(
                            f"indirectbr has no declared successor: {inst.text}"
                        )

                    body = inst.text[len("indirectbr") :].strip()
                    parts = _split_top_commas(body)
                    if not parts:
                        raise ValueError(f"cannot parse indirectbr: {inst.text}")

                    address_text = parts[0]
                    blockaddress_match = re.search(
                        r"blockaddress\s*\(\s*@([-A-Za-z$._0-9]+)\s*,\s*%([-A-Za-z$._0-9]+)\s*\)",
                        address_text,
                    )
                    if blockaddress_match is not None:
                        function_name, label = blockaddress_match.groups()
                        if function_name != fn.name:
                            raise ValueError(
                                "indirectbr blockaddress must target current function"
                            )
                        target = muir.Target(label=label)
                    else:
                        address = _value(address_text)
                        if isinstance(address, muir.Arbitrary) and address.kind == "undef":
                            # LLVM undef may be refined independently at each
                            # use. Choose one declared valid successor rather
                            # than preserving an unconstrained pointer that
                            # could also trigger LLVM UB.
                            target = muir.Target(label=labels[0])
                            stats.lowered_undef_indirectbr += 1
                        elif isinstance(address, muir.Slot):
                            target = muir.Target(slot=address)
                        else:
                            raise ValueError(
                                "dynamic indirectbr address must be a local SSA value: "
                                + address_text
                            )

                    # LLVM indirectbr is a register jump whose destination set
                    # is CFG metadata.  P3 already has an indirect BR target,
                    # so semantic descent only needs an unconditional indirect
                    # branch; invalid destinations retain LLVM's UB contract.
                    out.append(
                        muir.Br(
                            muir.Width.I8,
                            muir.Cond.EQ,
                            muir.Imm(0),
                            muir.Imm(0),
                            target,
                            target,
                        )
                    )
                    stats.lowered_indirectbr += 1
                    continue

                if op == "callbr":
                    cpu_feature_branch = _lower_cpu_feature_callbr(
                        inst, layout, temp_counter, frame_slots
                    )
                    if cpu_feature_branch is not None:
                        stats.lowered_cpu_feature_branch += 1
                        out.extend(cpu_feature_branch)
                        continue

                    static_branch = _lower_jump_label_callbr(
                        inst, layout, temp_counter, frame_slots
                    )
                    if static_branch is not None:
                        stats.lowered_static_branch += 1
                        out.extend(static_branch)
                        continue

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
            switch_edges = switch_edge_blocks.get((resolved_pred, target))
            if switch_edges:
                for edge_pred in switch_edges:
                    copies_by_pred[edge_pred].append((dst, src, width))
            else:
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

    ordered_blocks: list[muir.Block] = []
    for text_block in fn.blocks:
        ordered_blocks.append(out_blocks[text_block.label])
        ordered_blocks.extend(
            out_blocks[label] for label in extra_blocks_after.get(text_block.label, [])
        )

    result_fn = muir.Function(
        fn.name,
        ordered_blocks,
        frame_slots,
        tuple(arg[1:] if arg.startswith("%") else arg for arg in fn.args),
    )
    stats.muir_instructions = sum(len(b.instructions) for b in result_fn.blocks)
    return result_fn, stats


def legalize_module(text: str):
    functions = parse_module(text)
    layout = DataLayout.from_module(text)
    register_metadata = _register_metadata(text)
    out = []
    total = LegalizeStats()
    for fn in functions:
        lowered, stats = legalize_function(fn, layout, register_metadata)
        out.append(lowered)
        for key, value in stats.as_dict().items():
            setattr(total, key, getattr(total, key) + value)
    return out, total
