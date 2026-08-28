from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from . import muir
from .vm import MASK64, Program, VM, VMError


@dataclass(frozen=True)
class RuntimeSurface:
    helpers: frozenset[str]
    system_ops: frozenset[str]


def collect_runtime_surface(functions: Iterable[muir.Function]) -> RuntimeSurface:
    helpers: set[str] = set()
    systems: set[str] = set()
    for function in functions:
        for block in function.blocks:
            for inst in block.instructions:
                if isinstance(inst, muir.Helper):
                    helpers.add(inst.symbol)
                elif isinstance(inst, muir.Sys):
                    systems.add(inst.op)
    return RuntimeSurface(frozenset(helpers), frozenset(systems))


def _mask(bits: int) -> int:
    if bits < 1 or bits > 64:
        raise VMError(f"runtime helper width outside one P3 word: {bits}")
    return (1 << bits) - 1


def _signed(value: int, bits: int) -> int:
    value &= _mask(bits)
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def _binary_integer(op: str, bits: int, a: int, b: int) -> int:
    mask = _mask(bits)
    a &= mask
    b &= mask

    if op == "and":
        return a & b
    if op == "or":
        return a | b
    if op == "xor":
        return a ^ b
    if op == "mul":
        return (a * b) & mask
    if op in {"shl", "lshr", "ashr"}:
        if b >= bits:
            raise VMError(f"LLVM poison shift amount: {b} for i{bits}")
        if op == "shl":
            return (a << b) & mask
        if op == "lshr":
            return a >> b
        return (_signed(a, bits) >> b) & mask
    if op in {"udiv", "urem"}:
        if b == 0:
            raise VMError("LLVM poison unsigned division by zero")
        return (a // b if op == "udiv" else a % b) & mask
    if op in {"sdiv", "srem"}:
        sb = _signed(b, bits)
        sa = _signed(a, bits)
        if sb == 0:
            raise VMError("LLVM poison signed division by zero")
        if sa == -(1 << (bits - 1)) and sb == -1:
            raise VMError("LLVM poison signed division overflow")
        # LLVM signed division truncates toward zero.
        q = abs(sa) // abs(sb)
        if (sa < 0) != (sb < 0):
            q = -q
        if op == "sdiv":
            return q & mask
        return (sa - q * sb) & mask
    raise VMError(f"unknown integer helper operation: {op}")


def _icmp(pred: str, bits: int, a: int, b: int) -> int:
    mask = _mask(bits)
    ua, ub = a & mask, b & mask
    sa, sb = _signed(a, bits), _signed(b, bits)
    table = {
        "eq": ua == ub,
        "ne": ua != ub,
        "ult": ua < ub,
        "ule": ua <= ub,
        "ugt": ua > ub,
        "uge": ua >= ub,
        "slt": sa < sb,
        "sle": sa <= sb,
        "sgt": sa > sb,
        "sge": sa >= sb,
    }
    try:
        return int(table[pred])
    except KeyError as exc:
        raise VMError(f"unknown icmp predicate: {pred}") from exc


def helper_callback(symbol: str):
    m = re.fullmatch(
        r"__mm_(and|or|xor|shl|lshr|ashr|mul|udiv|sdiv|urem|srem)_(\d+)",
        symbol,
    )
    if m:
        op, width_text = m.groups()
        bits = int(width_text)

        def binary(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError(f"{symbol} expects 2 arguments")
            return _binary_integer(op, bits, args[0], args[1])

        return binary

    m = re.fullmatch(r"__mm_icmp_([a-z]+)_(\d+)", symbol)
    if m:
        pred, width_text = m.groups()
        bits = int(width_text)

        def compare(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError(f"{symbol} expects 2 arguments")
            return _icmp(pred, bits, args[0], args[1])

        return compare

    m = re.fullmatch(r"__mm_ptr_add_scaled_(\d+)", symbol)
    if m:
        scale = int(m.group(1))

        def ptr_add(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError(f"{symbol} expects 2 arguments")
            return (args[0] + args[1] * scale) & MASK64

        return ptr_add

    if re.fullmatch(r"__mm_select_.+", symbol):
        def select(vm: VM, args: tuple[int, ...]):
            if len(args) != 3:
                raise VMError(f"{symbol} expects condition,true,false")
            return args[1] if (args[0] & 1) else args[2]

        return select

    m = re.fullmatch(r"__mm_(zext|sext|trunc)_(\d+)_(\d+)", symbol)
    if m:
        op, src_text, dst_text = m.groups()
        src_bits, dst_bits = int(src_text), int(dst_text)

        def cast(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError(f"{symbol} expects 1 argument")
            value = args[0] & _mask(src_bits)
            if op == "sext":
                value = _signed(value, src_bits)
            return value & _mask(dst_bits)

        return cast

    m = re.fullmatch(r"__mm_freeze_(\d+)", symbol)
    if m:
        bits = int(m.group(1))

        def freeze(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError(f"{symbol} expects 1 argument")
            return args[0] & _mask(bits)

        return freeze

    m = re.fullmatch(r"__mm_add_(\d+)", symbol)
    if m:
        bits = int(m.group(1))

        def add(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError(f"{symbol} expects 2 arguments")
            return (args[0] + args[1]) & _mask(bits)

        return add

    m = re.fullmatch(r"__mm_llvm_bswap_i(16|32|64)", symbol)
    if m:
        bits = int(m.group(1))

        def bswap(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError(f"{symbol} expects 1 argument")
            size = bits // 8
            raw = (args[0] & _mask(bits)).to_bytes(size, "little")
            return int.from_bytes(raw[::-1], "little")

        return bswap

    m = re.fullmatch(r"__mm_llvm_(u|s)(min|max)_i(8|16|32|64)", symbol)
    if m:
        sign, direction, bits_text = m.groups()
        bits = int(bits_text)

        def minmax(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError(f"{symbol} expects 2 arguments")
            a, b = args
            if sign == "s":
                ka, kb = _signed(a, bits), _signed(b, bits)
            else:
                ka, kb = a & _mask(bits), b & _mask(bits)
            choose_a = ka <= kb if direction == "min" else ka >= kb
            return (a if choose_a else b) & _mask(bits)

        return minmax

    if symbol.startswith("__mm_llvm_memset_"):
        def memset(vm: VM, args: tuple[int, ...]):
            if len(args) < 3:
                raise VMError(f"{symbol} expects dst,value,length[,volatile]")
            dst, value, length = args[:3]
            byte = value & 0xFF
            for i in range(length):
                vm.memory.write((dst + i) & MASK64, 8, byte)
            return None

        return memset

    if symbol.startswith("__mm_llvm_memcpy_"):
        def memcpy(vm: VM, args: tuple[int, ...]):
            if len(args) < 3:
                raise VMError(f"{symbol} expects dst,src,length[,volatile]")
            dst, src, length = args[:3]
            data = [vm.memory.read((src + i) & MASK64, 8) for i in range(length)]
            for i, byte in enumerate(data):
                vm.memory.write((dst + i) & MASK64, 8, byte)
            return None

        return memcpy

    if symbol.startswith("__mm_llvm_memmove_"):
        def memmove(vm: VM, args: tuple[int, ...]):
            if len(args) < 3:
                raise VMError(f"{symbol} expects dst,src,length[,volatile]")
            dst, src, length = args[:3]
            data = [vm.memory.read((src + i) & MASK64, 8) for i in range(length)]
            for i, byte in enumerate(data):
                vm.memory.write((dst + i) & MASK64, 8, byte)
            return None

        return memmove

    # These helpers are intentionally not guessed. Their current μIR call
    # shape lacks enough executable type/layout information.
    if symbol in {
        "__mm_load_aggregate",
        "__mm_store_aggregate",
        "__mm_extractvalue",
        "__mm_insertvalue",
        "__mm_freeze_aggregate",
    }:
        def incomplete(vm: VM, args: tuple[int, ...]):
            raise VMError(
                f"{symbol} requires aggregate layout metadata before execution"
            )

        return incomplete

    return None


def _state_key(op: str) -> str:
    for prefix in (
        "state_read_clear_",
        "state_read_set_",
        "state_read_",
        "state_write_",
        "state_set_",
        "state_clear_",
        "state_swap_",
    ):
        if op.startswith(prefix):
            return op[len(prefix):]
    return op


def system_callback(op: str):
    if op in {"fence", "icache_sync"} or op.startswith("tlb_flush_"):
        def ordering(vm: VM, args: tuple[int, ...]):
            # The reference VM is single-threaded and executes memory accesses
            # in program order, so fences have no additional runtime action.
            return None
        return ordering

    if op == "wait_interrupt":
        def wait(vm: VM, args: tuple[int, ...]):
            return None
        return wait

    if op in {"counter_cycle", "counter_time", "counter_instret"}:
        def counter(vm: VM, args: tuple[int, ...]):
            return vm.steps & MASK64
        return counter

    if op.startswith("state_"):
        def state(vm: VM, args: tuple[int, ...]):
            key = _state_key(op)
            old = vm.system_state.get(key, 0)
            if op.startswith("state_read_clear_"):
                if len(args) != 1:
                    raise VMError(f"{op} expects one mask")
                vm.system_state[key] = old & ~args[0] & MASK64
                return old
            if op.startswith("state_read_set_"):
                if len(args) != 1:
                    raise VMError(f"{op} expects one mask")
                vm.system_state[key] = (old | args[0]) & MASK64
                return old
            if op.startswith("state_read_"):
                return old
            if op.startswith("state_write_"):
                if len(args) != 1:
                    raise VMError(f"{op} expects one value")
                vm.system_state[key] = args[0] & MASK64
                return None
            if op.startswith("state_set_"):
                if len(args) != 1:
                    raise VMError(f"{op} expects one mask")
                vm.system_state[key] = (old | args[0]) & MASK64
                return None
            if op.startswith("state_clear_"):
                if len(args) != 1:
                    raise VMError(f"{op} expects one mask")
                vm.system_state[key] = old & ~args[0] & MASK64
                return None
            if op.startswith("state_swap_"):
                if len(args) != 1:
                    raise VMError(f"{op} expects one value")
                vm.system_state[key] = args[0] & MASK64
                return old
            raise VMError(f"unsupported state operation: {op}")
        return state

    if op == "csr_read":
        def csr_read(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError("csr_read expects csr id")
            return vm.csr.get(args[0], 0)
        return csr_read

    if op in {"csr_write", "csr_set", "csr_clear"}:
        def csr_mutate(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError(f"{op} expects csr id,value")
            csr, value = args
            old = vm.csr.get(csr, 0)
            if op == "csr_write":
                new = value
            elif op == "csr_set":
                new = old | value
            else:
                new = old & ~value
            vm.csr[csr] = new & MASK64
            return None
        return csr_mutate

    if op in {"csr_swap", "csr_read_set", "csr_read_clear"}:
        def csr_rmw(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError(f"{op} expects csr id,value")
            csr, value = args
            old = vm.csr.get(csr, 0)
            if op == "csr_swap":
                new = value
            elif op == "csr_read_set":
                new = old | value
            else:
                new = old & ~value
            vm.csr[csr] = new & MASK64
            return old
        return csr_rmw

    return None


def install_runtime(program: Program, surface: RuntimeSurface) -> None:
    missing_helpers = []
    for symbol in sorted(surface.helpers):
        callback = helper_callback(symbol)
        if callback is None:
            missing_helpers.append(symbol)
            continue
        program.register_service(symbol, callback)

    missing_systems = []
    for op in sorted(surface.system_ops):
        callback = system_callback(op)
        if callback is None:
            missing_systems.append(op)
            continue
        program.register_system(op, callback)

    # Missing services are deliberately left unresolved. Execution reaches an
    # explicit linker/runtime error instead of silently using wrong semantics.
    program.runtime_missing_helpers = tuple(missing_helpers)
    program.runtime_missing_systems = tuple(missing_systems)
