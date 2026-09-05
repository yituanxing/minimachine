from __future__ import annotations

import argparse
import gzip
import json
import pickle
import statistics
from collections import defaultdict
from pathlib import Path

from src.minimachine import muir, p3
from src.minimachine.abi import ENTRY, RET_PC


def slots_in_value(value):
    return {value.name} if isinstance(value, muir.Slot) else set()


def slots_in_address(address):
    return slots_in_value(address.base)


def slots_in_operand(operand):
    if isinstance(operand, p3.Mem):
        return slots_in_address(operand.address)
    return slots_in_value(operand)


def slots_in_target(target):
    out = set()
    if target.slot is not None:
        out.add(target.slot.name)
    if target.address is not None:
        out |= slots_in_address(target.address)
    return out


def uses_defs(inst):
    uses = set()
    defs = set()
    if isinstance(inst, p3.Mov):
        uses |= slots_in_operand(inst.src)
        if isinstance(inst.dst, muir.Slot):
            defs.add(inst.dst.name)
        elif isinstance(inst.dst, p3.Mem):
            uses |= slots_in_address(inst.dst.address)
    elif isinstance(inst, p3.Sub):
        uses |= slots_in_value(inst.a)
        uses |= slots_in_value(inst.b)
        defs.add(inst.dst.name)
    elif isinstance(inst, p3.Br):
        uses |= slots_in_value(inst.a)
        uses |= slots_in_value(inst.b)
        uses |= slots_in_target(inst.true_target)
        uses |= slots_in_target(inst.false_target)
    else:
        raise TypeError(type(inst))
    return uses, defs


def direct_labels(inst):
    if not isinstance(inst, p3.Br):
        return set()
    out = set()
    for target in (inst.true_target, inst.false_target):
        if target.is_direct() and target.label is not None:
            out.add(target.label)
    return out


def is_call_terminator(br):
    if not isinstance(br, p3.Br):
        return False
    for target in (br.true_target, br.false_target):
        if target.address is None:
            return False
        address = target.address
        if address.offset != ENTRY or address.base is not muir.Special.SP:
            return False
    return True


def hidden_call_continuation(function_name, block):
    if not block.instructions or not is_call_terminator(block.instructions[-1]):
        return None
    for inst in reversed(block.instructions[:-1]):
        if not isinstance(inst, p3.Mov):
            continue
        if not isinstance(inst.dst, p3.Mem):
            continue
        if inst.dst.address.offset != RET_PC:
            continue
        if isinstance(inst.src, muir.BlockAddr) and inst.src.function == function_name:
            return inst.src.label
    return None


def analyze_function(linked):
    function = linked.function
    blocks = {block.label: block for block in function.blocks}

    successors = {}
    for block in function.blocks:
        succ = set()
        if block.instructions:
            succ |= direct_labels(block.instructions[-1])
            continuation = hidden_call_continuation(function.name, block)
            if continuation is not None:
                succ.add(continuation)
        successors[block.label] = {label for label in succ if label in blocks}

    block_use = {}
    block_def = {}
    instruction_ud = {}
    for block in function.blocks:
        seen_def = set()
        uses = set()
        defs = set()
        uds = []
        for inst in block.instructions:
            inst_use, inst_def = uses_defs(inst)
            uds.append((inst_use, inst_def))
            uses |= inst_use - seen_def
            seen_def |= inst_def
            defs |= inst_def
        block_use[block.label] = uses
        block_def[block.label] = defs
        instruction_ud[block.label] = uds

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
    peak = 0
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
        peak = max(peak, len(live))
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
            peak = max(peak, len(before))
            add_clique(before)
            live = before

    for slot in referenced:
        graph[slot]

    nodes = sorted(graph, key=lambda slot: (-len(graph[slot]), slot))
    color = {}
    for slot in nodes:
        used = {color[other] for other in graph[slot] if other in color}
        candidate = 0
        while candidate in used:
            candidate += 1
        color[slot] = candidate

    colors = max(color.values()) + 1 if color else 0
    hidden_edges = sum(
        hidden_call_continuation(function.name, block) is not None
        for block in function.blocks
    )
    return {
        "name": function.name,
        "logical_slots": len(function.frame_slots),
        "peak_live": peak,
        "colors": colors,
        "blocks": len(function.blocks),
        "instructions": sum(len(block.instructions) for block in function.blocks),
        "hidden_call_edges": hidden_edges,
    }


def quantile(sorted_values, fraction):
    if not sorted_values:
        return 0
    index = int(round((len(sorted_values) - 1) * fraction))
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def summarize(rows, key):
    values = sorted(row[key] for row in rows)
    if not values:
        return {}
    return {
        "min": values[0],
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "p999": quantile(values, 0.999),
        "max": values[-1],
        "mean": statistics.fmean(values),
        "coverage": {
            str(size): sum(value <= size for value in values) / len(values)
            for size in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache")
    parser.add_argument("--out", default="slot-coloring.json")
    args = parser.parse_args()

    with gzip.open(args.cache, "rb") as handle:
        payload = pickle.load(handle)

    cache = payload["cache"]
    rows = []
    for index, linked in enumerate(cache.program.functions.values(), 1):
        rows.append(analyze_function(linked))
        if index % 1000 == 0:
            print("SLOT_ANALYZE_PROGRESS", index, flush=True)

    result = {
        "functions": len(rows),
        "hidden_call_edges": sum(row["hidden_call_edges"] for row in rows),
        "logical_slots": summarize(rows, "logical_slots"),
        "peak_live": summarize(rows, "peak_live"),
        "colors": summarize(rows, "colors"),
        "color_equals_peak_fraction": (
            sum(row["colors"] == row["peak_live"] for row in rows) / len(rows)
        ),
        "function_colors": {
            row["name"]: row["colors"] for row in rows
        },
        "top_colors": sorted(rows, key=lambda row: (row["colors"], row["peak_live"], row["logical_slots"]), reverse=True)[:30],
        "top_peak": sorted(rows, key=lambda row: (row["peak_live"], row["colors"], row["logical_slots"]), reverse=True)[:30],
        "top_logical": sorted(rows, key=lambda row: row["logical_slots"], reverse=True)[:30],
    }

    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    compact = {key: result[key] for key in ("functions", "hidden_call_edges", "logical_slots", "peak_live", "colors", "color_equals_peak_fraction")}
    print("SLOT_ANALYZE_RESULT", json.dumps(compact, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
