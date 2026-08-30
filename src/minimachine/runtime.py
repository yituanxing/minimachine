from __future__ import annotations

from dataclasses import dataclass
import math
import re
import struct
from typing import Iterable

from . import muir
from .abi import CALLER_SP, RET_PC
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


def _int_mask(bits: int) -> int:
    if bits < 1:
        raise VMError(f"invalid integer width: {bits}")
    return (1 << bits) - 1


def _signed(value: int, bits: int) -> int:
    value &= _int_mask(bits)
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def _binary_integer(op: str, bits: int, a: int, b: int) -> int:
    mask = _int_mask(bits)
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
    mask = _int_mask(bits)
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


def _fp_decode(bits: int, value: int) -> float:
    if bits == 32:
        return struct.unpack(">f", (value & 0xFFFFFFFF).to_bytes(4, "big"))[0]
    if bits == 64:
        return struct.unpack(">d", (value & MASK64).to_bytes(8, "big"))[0]
    raise VMError(f"unsupported floating width: {bits}")


def _fp_encode(bits: int, value: float) -> int:
    fmt = ">f" if bits == 32 else ">d" if bits == 64 else None
    if fmt is None:
        raise VMError(f"unsupported floating width: {bits}")
    try:
        packed = struct.pack(fmt, value)
    except OverflowError:
        packed = struct.pack(
            fmt,
            math.copysign(math.inf, value),
        )
    return int.from_bytes(packed, "big")


def _fp_binary(op: str, bits: int, a_raw: int, b_raw: int) -> int:
    a = _fp_decode(bits, a_raw)
    b = _fp_decode(bits, b_raw)
    if op == "fadd":
        value = a + b
    elif op == "fsub":
        value = a - b
    elif op == "fmul":
        value = a * b
    elif op == "fdiv":
        if b == 0.0:
            if math.isnan(a) or a == 0.0:
                value = math.nan
            else:
                sign = (
                    math.copysign(1.0, a)
                    * math.copysign(1.0, b)
                )
                value = math.copysign(math.inf, sign)
        else:
            value = a / b
    else:
        raise VMError(f"unsupported floating binary op: {op}")
    return _fp_encode(bits, value)


def _fp_compare(pred: str, bits: int, a_raw: int, b_raw: int) -> int:
    a = _fp_decode(bits, a_raw)
    b = _fp_decode(bits, b_raw)
    unordered = math.isnan(a) or math.isnan(b)
    ordered = not unordered

    table = {
        "false": False,
        "oeq": ordered and a == b,
        "ogt": ordered and a > b,
        "oge": ordered and a >= b,
        "olt": ordered and a < b,
        "ole": ordered and a <= b,
        "one": ordered and a != b,
        "ord": ordered,
        "ueq": unordered or a == b,
        "ugt": unordered or a > b,
        "uge": unordered or a >= b,
        "ult": unordered or a < b,
        "ule": unordered or a <= b,
        "une": unordered or a != b,
        "uno": unordered,
        "true": True,
    }
    try:
        return int(table[pred])
    except KeyError as exc:
        raise VMError(f"unsupported floating predicate: {pred}") from exc


def _int_abi_align(bits: int) -> int:
    size = max(1, (bits + 7) // 8)
    if bits <= 8:
        return 1
    if bits <= 16:
        return 2
    if bits <= 32:
        return 4
    if bits <= 64:
        return 8
    return 16


def _decode_integer(vm: VM, encoded: int, bits: int) -> int:
    if bits <= 64:
        return encoded & _int_mask(bits)
    size = (bits + 7) // 8
    value = 0
    for i in range(size):
        value |= vm.memory.read(encoded + i, 8) << (8 * i)
    return value & _int_mask(bits)


def _encode_integer(vm: VM, bits: int, value: int) -> int:
    value &= _int_mask(bits)
    if bits <= 64:
        return value
    size = (bits + 7) // 8
    blob = vm.alloc_bytes(size, align=_int_abi_align(bits))
    for i in range(size):
        vm.memory.write(blob + i, 8, (value >> (8 * i)) & 0xFF)
    return blob


def _overflow_blob(vm: VM, bits: int, value: int, overflow: bool) -> int:
    size = max(1, (bits + 7) // 8)
    align = _int_abi_align(bits)
    overflow_offset = size
    total_size = ((overflow_offset + 1 + align - 1) // align) * align
    blob = vm.alloc_bytes(total_size, align=align)
    for i in range(total_size):
        vm.memory.write(blob + i, 8, 0)
    masked = value & _int_mask(bits)
    for i in range(size):
        vm.memory.write(
            blob + i,
            8,
            (masked >> (8 * i)) & 0xFF,
        )
    vm.memory.write(blob + overflow_offset, 8, int(overflow))
    return blob


def _caller_frame(vm: VM, depth: int) -> int:
    if depth < 0:
        raise VMError("negative frame depth")
    frame = vm.memory.read(vm.sp + CALLER_SP, 64)
    for _ in range(depth):
        parent = vm.memory.read(frame + CALLER_SP, 64)
        if parent == 0 or parent == frame:
            raise VMError("frame depth exceeds MiniMachine call chain")
        frame = parent
    return frame


def helper_callback(symbol: str):
    if symbol == "__mm_alloca":
        def alloca(vm: VM, args: tuple[int, ...]):
            if len(args) != 3:
                raise VMError("__mm_alloca expects element_size,count,align")
            element_size, count, align = args
            if align == 0 or (align & (align - 1)):
                raise VMError(f"invalid alloca alignment: {align}")
            if count > MASK64 // max(1, element_size):
                raise VMError("alloca size overflow")
            return vm.alloc_bytes(element_size * count, align=align)

        return alloca

    m = re.fullmatch(r"__mm_wide_const_(\d+)", symbol)
    if m:
        bits = int(m.group(1))

        def wide_const(vm: VM, args: tuple[int, ...]):
            expected = (bits + 63) // 64
            if len(args) != expected:
                raise VMError(
                    f"{symbol} expects {expected} 64-bit chunks"
                )
            value = 0
            for i, chunk in enumerate(args):
                value |= (chunk & MASK64) << (64 * i)
            return _encode_integer(vm, bits, value)

        return wide_const

    if symbol == "__mm_llvm_stacksave_p0":
        def stack_save(vm: VM, args: tuple[int, ...]):
            if args:
                raise VMError("__mm_llvm_stacksave_p0 expects no arguments")
            return vm.heap_next

        return stack_save

    if symbol == "__mm_llvm_stackrestore_p0":
        def stack_restore(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError("__mm_llvm_stackrestore_p0 expects one pointer")
            saved = args[0]
            if saved > vm.heap_next or saved >= vm.stack_top:
                raise VMError(
                    "__mm_llvm_stackrestore_p0 received invalid saved stack"
                )
            vm.heap_next = saved
            return None

        return stack_restore

    if symbol == "__mm_llvm_va_start":
        def va_start(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError("__mm_llvm_va_start expects va_list,fixed_count")
            va_list, fixed_count = args
            from .abi import CALLER_SP, FRAME_SIZE, ARG_COUNT, WORD
            caller_sp = vm.memory.read(vm.sp + CALLER_SP, 64)
            frame_size = vm.memory.read(caller_sp + FRAME_SIZE, 64)
            total_args = vm.memory.read(caller_sp + ARG_COUNT, 64)
            if fixed_count > total_args:
                raise VMError(
                    f"va_start fixed_count {fixed_count} exceeds argc {total_args}"
                )
            first_vararg = caller_sp + frame_size + fixed_count * WORD
            vm.memory.write(va_list, 64, first_vararg)
            return None

        return va_start

    if symbol == "__mm_llvm_va_copy":
        def va_copy(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError("__mm_llvm_va_copy expects dst,src")
            dst, src = args
            vm.memory.write(dst, 64, vm.memory.read(src, 64))
            return None

        return va_copy

    if symbol == "__mm_llvm_va_end":
        def va_end(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError("__mm_llvm_va_end expects va_list")
            return None

        return va_end


    m = re.fullmatch(r"__mm_llvm_fabs_f(32|64)", symbol)
    if m:
        bits = int(m.group(1))
        sign_bit = 1 << (bits - 1)
        mask = (1 << bits) - 1

        def llvm_fabs(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError(f"{symbol} expects 1 argument")
            return args[0] & (mask ^ sign_bit)

        return llvm_fabs

    m = re.fullmatch(r"__mm_llvm_fmuladd_f(32|64)", symbol)
    if m:
        bits = int(m.group(1))

        def fp_muladd(vm: VM, args: tuple[int, ...]):
            if len(args) != 3:
                raise VMError(f"{symbol} expects 3 arguments")
            a = _fp_decode(bits, args[0])
            b = _fp_decode(bits, args[1])
            c = _fp_decode(bits, args[2])
            # llvm.fmuladd permits target contraction; the reference runtime
            # uses the non-contracted fmul+fadd semantics, which is a valid
            # implementation of this intrinsic.
            return _fp_encode(bits, a * b + c)

        return fp_muladd

    m = re.fullmatch(r"__mm_(fadd|fsub|fmul|fdiv)_(32|64)", symbol)
    if m:
        op, bits_text = m.groups()
        bits = int(bits_text)

        def fp_binary(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError(f"{symbol} expects 2 arguments")
            return _fp_binary(op, bits, args[0], args[1])

        return fp_binary

    m = re.fullmatch(r"__mm_fneg_(32|64)", symbol)
    if m:
        bits = int(m.group(1))
        sign_bit = 1 << (bits - 1)
        mask = (1 << bits) - 1

        def fp_neg(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError(f"{symbol} expects 1 argument")
            return (args[0] ^ sign_bit) & mask

        return fp_neg

    m = re.fullmatch(
        r"__mm_fcmp_(false|oeq|ogt|oge|olt|ole|one|ord|"
        r"ueq|ugt|uge|ult|ule|une|uno|true)_(32|64)",
        symbol,
    )
    if m:
        pred, bits_text = m.groups()
        bits = int(bits_text)

        def fp_compare(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError(f"{symbol} expects 2 arguments")
            return _fp_compare(pred, bits, args[0], args[1])

        return fp_compare

    m = re.fullmatch(r"__mm_(sitofp|uitofp)_(\d+)_(32|64)", symbol)
    if m:
        op, src_text, dst_text = m.groups()
        src_bits = int(src_text)
        dst_bits = int(dst_text)
        src_mask = (1 << src_bits) - 1

        def int_to_fp(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError(f"{symbol} expects 1 argument")
            value = args[0] & src_mask
            if op == "sitofp":
                value = _signed(value, src_bits)
            return _fp_encode(dst_bits, float(value))

        return int_to_fp

    m = re.fullmatch(r"__mm_(fptosi|fptoui)_(32|64)_(\d+)", symbol)
    if m:
        op, src_text, dst_text = m.groups()
        src_bits = int(src_text)
        dst_bits = int(dst_text)
        dst_mask = (1 << dst_bits) - 1

        def fp_to_int(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError(f"{symbol} expects 1 argument")
            value = _fp_decode(src_bits, args[0])
            if not math.isfinite(value):
                raise VMError(f"{symbol} reached LLVM poison non-finite input")
            integer = math.trunc(value)
            if op == "fptosi":
                lo = -(1 << (dst_bits - 1))
                hi = 1 << (dst_bits - 1)
                if not (lo <= integer < hi):
                    raise VMError(
                        f"{symbol} reached LLVM poison out-of-range input"
                    )
            else:
                if not (0 <= integer < (1 << dst_bits)):
                    raise VMError(
                        f"{symbol} reached LLVM poison out-of-range input"
                    )
            return integer & dst_mask

        return fp_to_int

    m = re.fullmatch(r"__mm_(fpext|fptrunc)_(32|64)_(32|64)", symbol)
    if m:
        _op, src_text, dst_text = m.groups()
        src_bits = int(src_text)
        dst_bits = int(dst_text)

        def fp_cast(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError(f"{symbol} expects 1 argument")
            return _fp_encode(dst_bits, _fp_decode(src_bits, args[0]))

        return fp_cast

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
            a = _decode_integer(vm, args[0], bits)
            b = _decode_integer(vm, args[1], bits)
            return _encode_integer(
                vm,
                bits,
                _binary_integer(op, bits, a, b),
            )

        return binary

    m = re.fullmatch(r"__mm_icmp_([a-z]+)_(\d+)", symbol)
    if m:
        pred, width_text = m.groups()
        bits = int(width_text)

        def compare(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError(f"{symbol} expects 2 arguments")
            a = _decode_integer(vm, args[0], bits)
            b = _decode_integer(vm, args[1], bits)
            return _icmp(pred, bits, a, b)

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
            value = _decode_integer(vm, args[0], src_bits)
            if op == "sext":
                value = _signed(value, src_bits)
            return _encode_integer(vm, dst_bits, value)

        return cast

    m = re.fullmatch(r"__mm_freeze_(\d+)", symbol)
    if m:
        bits = int(m.group(1))

        def freeze(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError(f"{symbol} expects 1 argument")
            value = _decode_integer(vm, args[0], bits)
            return _encode_integer(vm, bits, value)

        return freeze

    m = re.fullmatch(r"__mm_add_(\d+)", symbol)
    if m:
        bits = int(m.group(1))

        def add(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError(f"{symbol} expects 2 arguments")
            a = _decode_integer(vm, args[0], bits)
            b = _decode_integer(vm, args[1], bits)
            return _encode_integer(vm, bits, a + b)

        return add

    m = re.fullmatch(r"__mm_llvm_objectsize_i(32|64)_p\d+", symbol)
    if m:
        bits = int(m.group(1))
        mask = _mask(bits)

        def llvm_objectsize(vm: VM, args: tuple[int, ...]):
            if len(args) != 4:
                raise VMError(
                    f"{symbol} expects object,min,nullunknown,dynamic"
                )
            pointer, minimum, null_unknown, _dynamic = args
            if pointer == 0 and not null_unknown:
                return 0
            # The reference VM has no source-level allocation provenance for
            # an arbitrary pointer. LLVM specifies unknown size as 0 for the
            # minimum query and -1 for the maximum query.
            return 0 if minimum else mask

        return llvm_objectsize

    m = re.fullmatch(r"__mm_llvm_expect(?:_with_probability)?_.+", symbol)
    if m:
        def llvm_expect(vm: VM, args: tuple[int, ...]):
            if len(args) not in {2, 3}:
                raise VMError(f"{symbol} expects value,expected[,probability]")
            return args[0]

        return llvm_expect

    m = re.fullmatch(r"__mm_llvm_is_constant_.+", symbol)
    if m:
        def llvm_is_constant(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError(f"{symbol} expects one value")
            # Reaching the reference VM means compile-time constant folding
            # did not prove the operand manifestly constant.
            return 0

        return llvm_is_constant

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

    m = re.fullmatch(r"__mm_llvm_(fshl|fshr)_i(8|16|32|64)", symbol)
    if m:
        direction, bits_text = m.groups()
        bits = int(bits_text)
        mask = _mask(bits)

        def funnel(vm: VM, args: tuple[int, ...]):
            if len(args) != 3:
                raise VMError(f"{symbol} expects a,b,shift")
            a, b, shift = args
            a &= mask
            b &= mask
            amount = shift % bits
            if amount == 0:
                return a if direction == "fshl" else b
            if direction == "fshl":
                return ((a << amount) | (b >> (bits - amount))) & mask
            return ((a << (bits - amount)) | (b >> amount)) & mask

        return funnel

    m = re.fullmatch(r"__mm_llvm_(cttz|ctlz|ctpop)_i(8|16|32|64)", symbol)
    if m:
        op, bits_text = m.groups()
        bits = int(bits_text)
        mask = _mask(bits)

        def bitcount(vm: VM, args: tuple[int, ...]):
            if op == "ctpop":
                if len(args) != 1:
                    raise VMError(f"{symbol} expects one argument")
                return (args[0] & mask).bit_count()

            if len(args) not in {1, 2}:
                raise VMError(f"{symbol} expects value[,zero_is_poison]")
            value = args[0] & mask
            zero_is_poison = bool(args[1]) if len(args) == 2 else False
            if value == 0:
                if zero_is_poison:
                    raise VMError(f"{symbol} reached LLVM poison zero input")
                return bits
            if op == "cttz":
                return (value & -value).bit_length() - 1
            return bits - value.bit_length()

        return bitcount

    m = re.fullmatch(r"__mm_(load|store)_i(\d+)", symbol)
    if m:
        kind, bits_text = m.groups()
        bits = int(bits_text)
        size = (bits + 7) // 8

        if kind == "load":
            def integer_load(vm: VM, args: tuple[int, ...]):
                if len(args) != 1:
                    raise VMError(f"{symbol} expects address")
                address = args[0]
                value = 0
                for i in range(size):
                    value |= vm.memory.read(address + i, 8) << (8 * i)
                return _encode_integer(vm, bits, value)
            return integer_load

        def integer_store(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError(f"{symbol} expects address,value")
            address, encoded = args
            value = _decode_integer(vm, encoded, bits)
            for i in range(size):
                vm.memory.write(
                    address + i,
                    8,
                    (value >> (8 * i)) & 0xFF,
                )
            return None
        return integer_store

    if symbol.startswith("__mm_llvm_prefetch_"):
        def prefetch(vm: VM, args: tuple[int, ...]):
            # LLVM prefetch has no observable program semantics; the reference
            # VM deliberately models it as a no-op performance hint.
            return None
        return prefetch

    m = re.fullmatch(r"__mm_llvm_abs_i(8|16|32|64)", symbol)
    if m:
        bits = int(m.group(1))
        mask = _mask(bits)
        minimum = -(1 << (bits - 1))

        def llvm_abs(vm: VM, args: tuple[int, ...]):
            if len(args) not in {1, 2}:
                raise VMError(f"{symbol} expects value[,min_is_poison]")
            value = _signed(args[0], bits)
            min_is_poison = bool(args[1]) if len(args) == 2 else False
            if value == minimum and min_is_poison:
                raise VMError(f"{symbol} reached LLVM poison INT_MIN input")
            return abs(value) & mask

        return llvm_abs

    m = re.fullmatch(r"__mm_llvm_(uadd|usub|sadd|ssub)_sat_i(8|16|32|64)", symbol)
    if m:
        op, bits_text = m.groups()
        bits = int(bits_text)
        mask = _mask(bits)
        signed_min = -(1 << (bits - 1))
        signed_max = (1 << (bits - 1)) - 1

        def saturating(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError(f"{symbol} expects two operands")
            if op == "uadd":
                return min((args[0] & mask) + (args[1] & mask), mask)
            if op == "usub":
                return max((args[0] & mask) - (args[1] & mask), 0)
            a = _signed(args[0], bits)
            b = _signed(args[1], bits)
            raw = a + b if op == "sadd" else a - b
            return min(max(raw, signed_min), signed_max) & mask

        return saturating

    m = re.fullmatch(
        r"__mm_llvm_([us])(add|sub|mul)_with_overflow_i(\d+)",
        symbol,
    )
    if m:
        signedness, op, bits_text = m.groups()
        bits = int(bits_text)
        mask = _int_mask(bits)
        signed_min = -(1 << (bits - 1))
        signed_max = (1 << (bits - 1)) - 1

        def with_overflow(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError(f"{symbol} expects two operands")
            a_raw = _decode_integer(vm, args[0], bits)
            b_raw = _decode_integer(vm, args[1], bits)
            if signedness == "u":
                a = a_raw & mask
                b = b_raw & mask
                if op == "add":
                    raw = a + b
                    overflow = raw > mask
                elif op == "sub":
                    raw = a - b
                    overflow = a < b
                else:
                    raw = a * b
                    overflow = raw > mask
                return _overflow_blob(vm, bits, raw, overflow)

            a = _signed(a_raw, bits)
            b = _signed(b_raw, bits)
            if op == "add":
                raw = a + b
            elif op == "sub":
                raw = a - b
            else:
                raw = a * b
            overflow = raw < signed_min or raw > signed_max
            return _overflow_blob(vm, bits, raw, overflow)

        return with_overflow

    m = re.fullmatch(r"__mm_llvm_bitreverse_i(\d+)", symbol)
    if m:
        bits = int(m.group(1))
        if bits <= 64:
            mask = _mask(bits)

            def bitreverse(vm: VM, args: tuple[int, ...]):
                if len(args) != 1:
                    raise VMError(f"{symbol} expects one argument")
                value = args[0] & mask
                out = 0
                for i in range(bits):
                    out = (out << 1) | ((value >> i) & 1)
                return out & mask

            return bitreverse

    if symbol == "__mm_llvm_returnaddress":
        def returnaddress(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError("llvm.returnaddress expects depth")
            frame = _caller_frame(vm, args[0])
            return vm.memory.read(frame + RET_PC, 64)
        return returnaddress

    if symbol.startswith("__mm_llvm_frameaddress"):
        def frameaddress(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError("llvm.frameaddress expects depth")
            return _caller_frame(vm, args[0])
        return frameaddress

    if symbol == "__mm_llvm_experimental_noalias_scope_decl":
        def noalias_scope_decl(vm: VM, args: tuple[int, ...]):
            return None
        return noalias_scope_decl

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
            bulk = getattr(vm.memory, "bulk_set", None)
            if bulk is not None:
                bulk(dst, byte, length)
            else:
                for i in range(length):
                    vm.memory.write((dst + i) & MASK64, 8, byte)
            return None

        return memset

    if symbol.startswith("__mm_llvm_memcpy_"):
        def memcpy(vm: VM, args: tuple[int, ...]):
            if len(args) < 3:
                raise VMError(f"{symbol} expects dst,src,length[,volatile]")
            dst, src, length = args[:3]
            bulk = getattr(vm.memory, "bulk_copy", None)
            if bulk is not None:
                bulk(dst, src, length)
            else:
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
            bulk = getattr(vm.memory, "bulk_move", None)
            if bulk is not None:
                bulk(dst, src, length)
            else:
                data = [vm.memory.read((src + i) & MASK64, 8) for i in range(length)]
                for i, byte in enumerate(data):
                    vm.memory.write((dst + i) & MASK64, 8, byte)
            return None

        return memmove

    if symbol == "__mm_aggregate_literal":
        def aggregate_literal(vm: VM, args: tuple[int, ...]):
            if not args or (len(args) - 1) % 3:
                raise VMError(
                    "__mm_aggregate_literal expects total,(offset,value,size)*"
                )
            total_size = args[0]
            blob = vm.alloc_bytes(total_size)
            for i in range(total_size):
                vm.memory.write(blob + i, 8, 0)

            for n in range(1, len(args), 3):
                offset, value, size = args[n : n + 3]
                if size < 1 or size > 8:
                    raise VMError(
                        f"aggregate literal scalar size outside one word: {size}"
                    )
                if offset + size > total_size:
                    raise VMError("aggregate literal initializer exceeds blob")
                for byte_index in range(size):
                    vm.memory.write(
                        blob + offset + byte_index,
                        8,
                        (value >> (8 * byte_index)) & 0xFF,
                    )
            return blob
        return aggregate_literal

    if symbol == "__mm_load_aggregate":
        def load_aggregate(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError("__mm_load_aggregate expects address,size")
            source, size = args
            blob = vm.alloc_bytes(size)
            for i in range(size):
                vm.memory.write(blob + i, 8, vm.memory.read(source + i, 8))
            return blob
        return load_aggregate

    if symbol == "__mm_store_aggregate":
        def store_aggregate(vm: VM, args: tuple[int, ...]):
            if len(args) != 3:
                raise VMError("__mm_store_aggregate expects address,blob,size")
            destination, blob, size = args
            for i in range(size):
                vm.memory.write(
                    destination + i,
                    8,
                    vm.memory.read(blob + i, 8),
                )
            return None
        return store_aggregate

    if symbol == "__mm_freeze_aggregate":
        def freeze_aggregate(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError("__mm_freeze_aggregate expects blob")
            return args[0]
        return freeze_aggregate

    if symbol == "__mm_extractvalue":
        def extractvalue(vm: VM, args: tuple[int, ...]):
            if len(args) != 4:
                raise VMError(
                    "__mm_extractvalue expects blob,offset,size,is_blob"
                )
            blob, offset, size, is_blob = args
            if is_blob:
                out = vm.alloc_bytes(size)
                for i in range(size):
                    byte = 0 if blob == 0 else vm.memory.read(blob + offset + i, 8)
                    vm.memory.write(out + i, 8, byte)
                return out
            if size < 1 or size > 8:
                raise VMError(
                    f"scalar extractvalue size outside one word: {size}"
                )
            value = 0
            if blob != 0:
                for i in range(size):
                    value |= vm.memory.read(blob + offset + i, 8) << (8 * i)
            return value
        return extractvalue

    if symbol == "__mm_insertvalue":
        def insertvalue(vm: VM, args: tuple[int, ...]):
            if len(args) != 6:
                raise VMError(
                    "__mm_insertvalue expects blob,total,offset,value,size,is_blob"
                )
            blob, total_size, offset, value, field_size, field_is_blob = args
            if offset + field_size > total_size:
                raise VMError("__mm_insertvalue field exceeds aggregate")
            out = vm.alloc_bytes(total_size)
            for i in range(total_size):
                byte = 0 if blob == 0 else vm.memory.read(blob + i, 8)
                vm.memory.write(out + i, 8, byte)

            if field_is_blob:
                for i in range(field_size):
                    byte = 0 if value == 0 else vm.memory.read(value + i, 8)
                    vm.memory.write(out + offset + i, 8, byte)
            else:
                if field_size < 1 or field_size > 8:
                    raise VMError(
                        f"scalar insertvalue size outside one word: {field_size}"
                    )
                for i in range(field_size):
                    vm.memory.write(
                        out + offset + i,
                        8,
                        (value >> (8 * i)) & 0xFF,
                    )
            return out
        return insertvalue

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



def _expected_results(vm: VM) -> int:
    from .abi import RESULT_COUNT
    return vm.memory.read(vm.sp + RESULT_COUNT, 64)


def _atomic_result(vm: VM, old: int, status: int = 0):
    count = _expected_results(vm)
    if count == 0:
        return None
    if count == 1:
        return old
    if count == 2:
        return (old, status)
    raise VMError(f"atomic service expects unsupported result count: {count}")


def _atomic_callback(op: str):
    m = re.fullmatch(
        r"atomic_(add|or|and|xor|swap)_i(32|64)_(.+)",
        op,
    )
    if m:
        kind, bits_text, ordering = m.groups()
        bits = int(bits_text)
        mask = _mask(bits)

        def amo(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError(f"{op} expects address,value")
            address, value = args
            old = vm.memory.read(address, bits)
            rhs = value & mask
            if kind == "add":
                new = (old + rhs) & mask
            elif kind == "or":
                new = old | rhs
            elif kind == "and":
                new = old & rhs
            elif kind == "xor":
                new = old ^ rhs
            else:
                new = rhs
            vm.memory.write(address, bits, new)
            return _atomic_result(vm, old)

        return amo

    m = re.fullmatch(
        r"atomic_cmpxchg_i(32|64)_(.+)",
        op,
    )
    if m:
        bits = int(m.group(1))
        mask = _mask(bits)

        def cmpxchg(vm: VM, args: tuple[int, ...]):
            if len(args) != 3:
                raise VMError(f"{op} expects address,expected,desired")
            address, expected, desired = args
            old = vm.memory.read(address, bits)
            success = (old & mask) == (expected & mask)
            if success:
                vm.memory.write(address, bits, desired)
            # LR/SC status is 0 on store success; use 1 for the compare-fail
            # path so the multi-result contract remains deterministic.
            return _atomic_result(vm, old, 0 if success else 1)

        return cmpxchg

    m = re.fullmatch(
        r"atomic_(add_unless|dec_if_positive|add1_if_nonnegative|sub1_if_nonpositive)_i(32|64)_(.+)",
        op,
    )
    if m:
        kind, bits_text, ordering = m.groups()
        bits = int(bits_text)
        mask = _mask(bits)

        def conditional(vm: VM, args: tuple[int, ...]):
            if not args:
                raise VMError(f"{op} missing address")
            address = args[0]
            old = vm.memory.read(address, bits)
            sold = _signed(old, bits)
            changed = False
            new = old

            if kind == "add_unless":
                if len(args) != 3:
                    raise VMError(f"{op} expects address,delta,unless")
                delta, unless = args[1], args[2]
                if old != (unless & mask):
                    new = (old + delta) & mask
                    changed = True
            elif kind == "dec_if_positive":
                if old != 0 and sold > 0:
                    new = (old - 1) & mask
                    changed = True
            elif kind == "add1_if_nonnegative":
                if sold >= 0:
                    new = (old + 1) & mask
                    changed = True
            elif kind == "sub1_if_nonpositive":
                if sold <= 0:
                    new = (old - 1) & mask
                    changed = True

            if changed:
                vm.memory.write(address, bits, new)
            return _atomic_result(vm, old, 0 if changed else 1)

        return conditional

    return None


def _faultable_callback(op: str):
    m = re.fullmatch(r"faultable_(load|store)_i(8|16|32|64)", op)
    if m:
        kind, bits_text = m.groups()
        bits = int(bits_text)

        if kind == "load":
            def load(vm: VM, args: tuple[int, ...]):
                if len(args) < 1:
                    raise VMError(f"{op} missing address")
                value = vm.memory.read(args[0], bits)
                count = _expected_results(vm)
                if count == 2:
                    return (0, value)
                if count == 1:
                    return value
                raise VMError(f"{op} unexpected result count: {count}")
            return load

        def store(vm: VM, args: tuple[int, ...]):
            if len(args) < 2:
                raise VMError(f"{op} expects address,value[,error_init]")
            vm.memory.write(args[0], bits, args[1])
            count = _expected_results(vm)
            if count == 0:
                return None
            if count == 1:
                return 0
            raise VMError(f"{op} unexpected result count: {count}")
        return store

    m = re.fullmatch(
        r"faultable_atomic_(add|and|or|xor|swap|cmpxchg)_i(32|64)_(.+)",
        op,
    )
    if not m:
        return None
    kind, bits_text, ordering = m.groups()
    bits = int(bits_text)
    mask = _mask(bits)

    def fault_atomic(vm: VM, args: tuple[int, ...]):
        if not args:
            raise VMError(f"{op} missing address")
        address = args[0]
        old = vm.memory.read(address, bits)
        status = 0

        if kind == "cmpxchg":
            if len(args) != 3:
                raise VMError(f"{op} expects address,expected,desired")
            expected, desired = args[1], args[2]
            success = old == (expected & mask)
            if success:
                vm.memory.write(address, bits, desired)
            status = 0 if success else 1
        else:
            if len(args) != 2:
                raise VMError(f"{op} expects address,value")
            value = args[1] & mask
            if kind == "add":
                new = (old + value) & mask
            elif kind == "and":
                new = old & value
            elif kind == "or":
                new = old | value
            elif kind == "xor":
                new = old ^ value
            else:
                new = value
            vm.memory.write(address, bits, new)

        count = _expected_results(vm)
        if count == 2:
            # Linux futex AMO aggregate is {error, old}.
            return (0, old)
        if count == 3:
            # Linux futex cmpxchg aggregate is {error, old, sc_status}.
            return (0, old, status)
        if count == 1:
            return old
        if count == 0:
            return None
        raise VMError(f"{op} unexpected result count: {count}")

    return fault_atomic


def system_callback(op: str):
    atomic = _atomic_callback(op)
    if atomic is not None:
        return atomic

    faultable = _faultable_callback(op)
    if faultable is not None:
        return faultable

    if op == "static_branch":
        def static_branch(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError("static_branch expects key")
            return int(bool(vm.static_keys.get(args[0], 0)))
        return static_branch

    if op == "cpu_feature":
        def cpu_feature(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError("cpu_feature expects feature id")
            return int(args[0] in vm.cpu_features)
        return cpu_feature

    if op == "ecall":
        def ecall(vm: VM, args: tuple[int, ...]):
            if vm.ecall_handler is None:
                raise VMError("ecall reached without a configured handler")
            return vm.ecall_handler(vm, args)
        return ecall

    if op == "vector_state_snapshot":
        def vector_snapshot(vm: VM, args: tuple[int, ...]):
            if args:
                raise VMError("vector_state_snapshot expects no arguments")
            return vm.vector_state
        return vector_snapshot

    if op == "vector_state_restore":
        def vector_restore(vm: VM, args: tuple[int, ...]):
            if len(args) != 4:
                raise VMError("vector_state_restore expects vstart,vtype,vl,vcsr")
            vm.vector_state = (
                args[0] & MASK64,
                args[1] & MASK64,
                args[2] & MASK64,
                args[3] & MASK64,
                vm.vector_state[4],
            )
            return None
        return vector_restore

    if op == "vector_length_bytes":
        def vector_length(vm: VM, args: tuple[int, ...]):
            if args:
                raise VMError("vector_length_bytes expects no arguments")
            return vm.vector_state[4]
        return vector_length

    if op == "thread_pointer":
        def thread_pointer(vm: VM, args: tuple[int, ...]):
            if args:
                raise VMError("thread_pointer expects no arguments")
            return vm.system_state.get("thread_pointer", 0)
        return thread_pointer

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


# Portable C-runtime entry points that a real MiniMachine target can later
# provide as P3 code.  The reference VM exposes them as host services so the
# executable whole-image gate is not coupled to RISC-V's assembly lib/string
# implementations.
_DIRECT_RUNTIME_SYMBOLS = (
    "memcpy",
    "__memcpy",
    "memmove",
    "__memmove",
    "memset",
    "__memset",
    "memcmp",
    "strlen",
    "strcmp",
    "strncmp",
    "ror32",
)


def direct_runtime_callback(symbol: str):
    base = symbol[2:] if symbol in {"__memcpy", "__memmove", "__memset"} else symbol

    if base == "memcpy":
        def memcpy(vm: VM, args: tuple[int, ...]):
            if len(args) != 3:
                raise VMError(f"{symbol} expects dst,src,size")
            dst, src, size = args
            bulk = getattr(vm.memory, "bulk_copy", None)
            if bulk is not None:
                bulk(dst, src, size)
            else:
                for i in range(size):
                    vm.memory.write(dst + i, 8, vm.memory.read(src + i, 8))
            return dst
        return memcpy

    if base == "memmove":
        def memmove(vm: VM, args: tuple[int, ...]):
            if len(args) != 3:
                raise VMError(f"{symbol} expects dst,src,size")
            dst, src, size = args
            bulk = getattr(vm.memory, "bulk_move", None)
            if bulk is not None:
                bulk(dst, src, size)
            else:
                data = [vm.memory.read(src + i, 8) for i in range(size)]
                for i, byte in enumerate(data):
                    vm.memory.write(dst + i, 8, byte)
            return dst
        return memmove

    if base == "memset":
        def memset(vm: VM, args: tuple[int, ...]):
            if len(args) != 3:
                raise VMError(f"{symbol} expects dst,value,size")
            dst, value, size = args
            byte = value & 0xFF
            bulk = getattr(vm.memory, "bulk_set", None)
            if bulk is not None:
                bulk(dst, byte, size)
            else:
                for i in range(size):
                    vm.memory.write(dst + i, 8, byte)
            return dst
        return memset

    if base == "memcmp":
        def memcmp(vm: VM, args: tuple[int, ...]):
            if len(args) != 3:
                raise VMError("memcmp expects a,b,size")
            a, b, size = args
            bulk = getattr(vm.memory, "bulk_compare", None)
            if bulk is not None:
                return bulk(a, b, size) & MASK64
            for i in range(size):
                av = vm.memory.read(a + i, 8)
                bv = vm.memory.read(b + i, 8)
                if av != bv:
                    return (av - bv) & MASK64
            return 0
        return memcmp

    if base == "strlen":
        def strlen(vm: VM, args: tuple[int, ...]):
            if len(args) != 1:
                raise VMError("strlen expects string")
            ptr = args[0]
            bulk = getattr(vm.memory, "bulk_strlen", None)
            if bulk is not None:
                return bulk(ptr)
            size = 0
            while vm.memory.read(ptr + size, 8) != 0:
                size += 1
                if size > MASK64:
                    raise VMError("strlen address wrapped")
            return size
        return strlen

    if base == "strcmp":
        def strcmp(vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError("strcmp expects a,b")
            a, b = args
            i = 0
            while True:
                av = vm.memory.read(a + i, 8)
                bv = vm.memory.read(b + i, 8)
                if av != bv:
                    return (av - bv) & MASK64
                if av == 0:
                    return 0
                i += 1
        return strcmp

    if base == "strncmp":
        def strncmp(vm: VM, args: tuple[int, ...]):
            if len(args) != 3:
                raise VMError("strncmp expects a,b,size")
            a, b, size = args
            for i in range(size):
                av = vm.memory.read(a + i, 8)
                bv = vm.memory.read(b + i, 8)
                if av != bv:
                    return (av - bv) & MASK64
                if av == 0:
                    return 0
            return 0
        return strncmp

    if base == "ror32":
        def ror32(_vm: VM, args: tuple[int, ...]):
            if len(args) != 2:
                raise VMError("ror32 expects word,shift")
            word = args[0] & 0xFFFFFFFF
            shift = args[1] & 31
            return (
                (word >> shift) |
                ((word << ((-shift) & 31)) & 0xFFFFFFFF)
            ) & 0xFFFFFFFF
        return ror32

    return None


def install_direct_runtime(program: Program) -> None:
    for symbol in _DIRECT_RUNTIME_SYMBOLS:
        if symbol in program.symbol_addresses:
            continue
        callback = direct_runtime_callback(symbol)
        if callback is None:
            continue
        program.register_service(symbol, callback)


def accelerate_direct_runtime(
    program: Program,
    symbols: tuple[str, ...] = (
        "memcpy",
        "memmove",
        "memset",
        "memcmp",
        "strlen",
        "strcmp",
        "strncmp",
        "ror32",
    ),
) -> tuple[str, ...]:
    """Fast-path selected portable runtime functions in the reference VM.

    Linux provides P3 bodies for these symbols, so the normal runtime installer leaves
    them alone.  Dynamic boot replay can safely redirect descriptor-mediated calls to
    the equivalent host callbacks to avoid spending millions of interpreter steps in
    byte-copy loops.  The original P3 bodies remain linked for structural validation.
    """
    accelerated = []
    for symbol in symbols:
        if symbol not in _DIRECT_RUNTIME_SYMBOLS:
            raise VMError(f"unsupported direct runtime acceleration: {symbol}")
        if symbol not in program.functions:
            continue
        callback = direct_runtime_callback(symbol)
        if callback is None:
            continue
        program.replace_function_with_service(symbol, callback)
        accelerated.append(symbol)
    return tuple(accelerated)


def install_runtime(program: Program, surface: RuntimeSurface) -> None:
    install_direct_runtime(program)

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
