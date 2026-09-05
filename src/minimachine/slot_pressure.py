from __future__ import annotations

from collections import defaultdict

from . import muir, p3
from .abi import ENTRY, RET_PC


def _slots_in_value(value):
    return {value.name} if isinstance(value, muir.Slot) else set()


def _slots_in_address(address):
    return _slots_in_value(address.base)


def _slots_in_operand(operand):
    if isinstance(operand, p3.Mem):
        return _slots_in_address(operand.address)
    return _slots_in_value(operand)


def _slots_in_target(target):
    out = set()
    if target.slot is not None:
        out.add(target.slot.name)
    if target.address is not None:
        out |= _slots_in_address(target.address)
    return out


def _uses_defs(inst):
    uses = set()
    defs = set()
    if isinstance(inst, p3.Mov):
        uses |= _slots_in_operand(inst.src)
        if isinstance(inst.dst, muir.Slot):
            defs.add(inst.dst.name)
        elif isinstance(inst.dst, p3.Mem):
            uses |= _slots_in_address(inst.dst.address)
    elif isinstance(inst, p3.Sub):
        uses |= _slots_in_value(inst.a)
        uses |= _slots_in_value(inst.b)
        defs.add(inst.dst.name)
    elif isinstance(inst, p3.Br):
        uses |= _slots_in_value(inst.a)
        uses |= _slots_in_value(inst.b)
        uses |= _slots_in_target(inst.true_target)
        uses |= _slots_in_target(inst.false_target)
    else:
        raise TypeError(type(inst))
    return uses, defs


def _is_call_terminator(br):
    if not isinstance(br, p3.Br):
        return False
    for target in (br.true_target, br.false_target):
        if target.address is None:
            return False
        if (
            target.address.offset != ENTRY
            or target.address.base is not muir.Special.SP
        ):
            return False
    return True


def _hidden_call_continuation(function_name, block):
    if not block.instructions or not _is_call_terminator(block.instructions[-1]):
        return None
    for inst in reversed(block.instructions[:-1]):
        if (
            isinstance(inst, p3.Mov)
            and isinstance(inst.dst, p3.Mem)
            and inst.dst.address.offset == RET_PC
            and isinstance(inst.src, muir.BlockAddr)
            and inst.src.function == function_name
        ):
            return inst.src.label
    return None


def analyze_function(function):
    blocks = {block.label: block for block in function.blocks}
    successors = {}
    instruction_ud = {}
    block_use = {}
    block_def = {}

    for block in function.blocks:
        succ = set()
        if block.instructions:
            br = block.instructions[-1]
            if isinstance(br, p3.Br):
                for target in (br.true_target, br.false_target):
                    if target.is_direct() and target.label in blocks:
                        succ.add(target.label)
            continuation = _hidden_call_continuation(function.name, block)
            if continuation in blocks:
                succ.add(continuation)
        successors[block.label] = succ

        seen_def = set()
        uses = set()
        defs = set()
        uds = []
        for inst in block.instructions:
            inst_use, inst_def = _uses_defs(inst)
            uds.append((inst_use, inst_def))
            uses |= inst_use - seen_def
            seen_def |= inst_def
            defs |= inst_def
        instruction_ud[block.label] = uds
        block_use[block.label] = uses
        block_def[block.label] = defs

    live_in = {label: set() for label in blocks}
    live_out = {label: set() for label in blocks}
    order = [block.label for block in function.blocks]
    changed = True
    while changed:
        changed = False
        for label in reversed(order):
            out = set()
            for succ in successors[label]:
                out |= live_in[succ]
            incoming = block_use[label] | (out - block_def[label])
            if out != live_out[label] or incoming != live_in[label]:
                live_out[label] = out
                live_in[label] = incoming
                changed = True

    graph = defaultdict(set)
    peak_live = 0
    referenced = set()

    def add_clique(values):
        values = list(values)
        for index, left in enumerate(values):
            for right in values[index + 1 :]:
                if left != right:
                    graph[left].add(right)
                    graph[right].add(left)

    for block in function.blocks:
        live = set(live_out[block.label])
        peak_live = max(peak_live, len(live))
        referenced |= live
        add_clique(live)
        for uses, defs in reversed(instruction_ud[block.label]):
            referenced |= uses | defs
            for defined in defs:
                for resident in live:
                    if defined != resident:
                        graph[defined].add(resident)
                        graph[resident].add(defined)
            before = (live - defs) | uses
            peak_live = max(peak_live, len(before))
            add_clique(before)
            live = before

    for slot in referenced:
        graph[slot]

    colors = {}
    for slot in sorted(graph, key=lambda name: (-len(graph[name]), name)):
        occupied = {colors[other] for other in graph[slot] if other in colors}
        color = 0
        while color in occupied:
            color += 1
        colors[slot] = color

    physical_slots = max(colors.values()) + 1 if colors else 0
    return {
        "logical_slots": len(function.frame_slots),
        "peak_live": peak_live,
        "colors": physical_slots,
        "color_map": dict(colors),
        "hidden_call_edges": sum(
            _hidden_call_continuation(function.name, block) is not None
            for block in function.blocks
        ),
    }


def analyze_linked_function(linked):
    return analyze_function(linked.function)
