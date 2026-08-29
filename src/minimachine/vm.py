from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Iterable

from . import muir, p3
from .abi import (
    ARG_COUNT,
    CALLER_SP,
    ENTRY,
    FRAME_SIZE,
    HEADER_SIZE,
    RESUME_PC,
    RESULT_COUNT,
    RESULT_PTR,
    RET_PC,
    WORD,
)
from .verify import verify_p3


MASK64 = (1 << 64) - 1
CODE_BASE = 0x8000_0000_0000_0000
DATA_BASE = 0x0000_0000_0001_0000
DEFAULT_STACK_TOP = 0x0000_0001_0000_0000
HEAP_BASE = 0x0000_0000_1000_0000


class VMError(RuntimeError):
    pass


class VMHalt(Exception):
    pass


class _HostControlTransfer:
    pass


HOST_CONTROL_TRANSFER = _HostControlTransfer()


class SparseMemory:
    def __init__(self, initial: dict[int, int] | None = None):
        self.bytes: dict[int, int] = dict(initial or {})

    def clone(self) -> "SparseMemory":
        return SparseMemory(self.bytes)

    def read(self, address: int, bits: int) -> int:
        if bits not in {8, 16, 32, 64}:
            raise VMError(f"unsupported memory width: {bits}")
        value = 0
        for i in range(bits // 8):
            value |= self.bytes.get(address + i, 0) << (8 * i)
        return value

    def write(self, address: int, bits: int, value: int) -> None:
        if bits not in {8, 16, 32, 64}:
            raise VMError(f"unsupported memory width: {bits}")
        value &= (1 << bits) - 1
        for i in range(bits // 8):
            self.bytes[address + i] = (value >> (8 * i)) & 0xFF


HostService = Callable[["VM", tuple[int, ...]], Iterable[int] | int | None]


@dataclass(frozen=True)
class LinkedFunction:
    function: p3.Function
    slot_offsets: dict[str, int]
    frame_size: int
    block_map: dict[str, p3.Block]


class Program:
    """Linked executable P3 image.

    P3 slots are concrete activation-frame cells. Function symbols resolve to
    descriptors containing an entry code address and fixed frame size, exactly
    matching the ABI lowerer's descriptor contract.
    """

    def __init__(self, functions: Iterable[p3.Function] = ()):
        self.functions: dict[str, LinkedFunction] = {}
        self.block_code: dict[tuple[str, str], int] = {}
        self.code_block: dict[int, tuple[str, str]] = {}
        self.host_code: dict[int, str] = {}
        self.host_services: dict[str, HostService] = {}
        self.symbol_addresses: dict[str, int] = {}
        self.initial_memory = SparseMemory()
        self._next_code = CODE_BASE
        self._next_data = DATA_BASE
        self.halt_code = self._alloc_code()
        self.host_code[self.halt_code] = "__halt__"

        for function in functions:
            self.add_function(function)

    def _alloc_code(self) -> int:
        value = self._next_code
        self._next_code += 8
        return value

    def _alloc_data(self, size: int, align: int = 8) -> int:
        start = (self._next_data + align - 1) // align * align
        self._next_data = start + size
        return start

    def add_function(self, function: p3.Function) -> None:
        verify_p3(function)
        if function.name in self.functions or function.name in self.symbol_addresses:
            raise VMError(f"duplicate program symbol: {function.name}")

        slot_offsets = {
            name: HEADER_SIZE + i * WORD
            for i, name in enumerate(sorted(function.frame_slots))
        }
        frame_size = HEADER_SIZE + len(slot_offsets) * WORD
        block_map = {block.label: block for block in function.blocks}
        linked = LinkedFunction(function, slot_offsets, frame_size, block_map)
        self.functions[function.name] = linked

        for block in function.blocks:
            code = self._alloc_code()
            self.block_code[(function.name, block.label)] = code
            self.code_block[code] = (function.name, block.label)

        descriptor = self._alloc_data(16)
        self.symbol_addresses[function.name] = descriptor
        entry = self.block_code[(function.name, function.blocks[0].label)]
        self.initial_memory.write(descriptor + 0, 64, entry)
        self.initial_memory.write(descriptor + 8, 64, frame_size)

    def register_service(self, symbol: str, callback: HostService) -> None:
        if symbol in self.symbol_addresses:
            raise VMError(f"duplicate program symbol: {symbol}")
        code = self._alloc_code()
        self.host_code[code] = symbol
        self.host_services[symbol] = callback
        descriptor = self._alloc_data(16)
        self.symbol_addresses[symbol] = descriptor
        self.initial_memory.write(descriptor + 0, 64, code)
        self.initial_memory.write(descriptor + 8, 64, HEADER_SIZE)

    def replace_function_with_service(self, symbol: str, callback: HostService) -> None:
        """Redirect calls through an existing function descriptor to a host service.

        The P3 body stays linked so structural/executable gates can still inspect it.
        Only descriptor-mediated calls are accelerated, which preserves the machine ABI
        while letting the reference VM fast-path proven-equivalent runtime routines.
        """
        if symbol not in self.functions:
            raise VMError(f"cannot replace non-function symbol: {symbol}")
        descriptor = self.symbol_addresses[symbol]
        host_symbol = f"__mm_fast_{symbol}"
        if host_symbol in self.host_services:
            raise VMError(f"host service already registered: {host_symbol}")
        code = self._alloc_code()
        self.host_code[code] = host_symbol
        self.host_services[host_symbol] = callback
        self.initial_memory.write(descriptor + 0, 64, code)
        self.initial_memory.write(descriptor + 8, 64, HEADER_SIZE)

    def register_system(self, op: str, callback: HostService) -> None:
        tag = re.sub(r"[^A-Za-z0-9_]+", "_", op).strip("_")
        if not tag:
            raise VMError("empty system operation")
        self.register_service("__mm_sys_" + tag, callback)

    def define_data_symbol(self, symbol: str, data: bytes, *, align: int = 8) -> int:
        if symbol in self.symbol_addresses:
            raise VMError(f"duplicate program symbol: {symbol}")
        address = self._alloc_data(len(data), align)
        self.symbol_addresses[symbol] = address
        for i, byte in enumerate(data):
            self.initial_memory.write(address + i, 8, byte)
        return address

    def new_vm(self, *, stack_top: int = DEFAULT_STACK_TOP) -> "VM":
        return VM(self, self.initial_memory.clone(), stack_top=stack_top)


class VM:
    def __init__(self, program: Program, memory: SparseMemory, *, stack_top: int):
        self.program = program
        self.memory = memory
        self.stack_top = stack_top & MASK64
        self.sp = self.stack_top
        self.current_function: str | None = None
        self.current_block: str | None = None
        self.ip = 0
        self.halted = False
        self.steps = 0
        self.system_state: dict[str, int] = {}
        self.csr: dict[int, int] = {}
        self.static_keys: dict[int, int] = {}
        self.cpu_features: set[int] = set()
        self.ecall_handler = None
        # Abstract vector architectural state used by the system contract:
        # (vstart, vtype, vl, vcsr, vlenb)
        self.vector_state = (0, 0, 0, 0, 0)
        self.heap_next = HEAP_BASE

    def enter_function(
        self,
        name: str,
        args: Iterable[int] = (),
        *,
        stack_top: int,
        result_count: int = 0,
    ) -> None:
        """Transfer execution to a fresh P3 activation on a caller-owned stack.

        This is the machine-level primitive used by host/platform code for
        control transfers that do not return through the current host-service
        frame, such as starting a newly scheduled kernel task.
        """
        if name not in self.program.functions:
            raise VMError(f"unknown transfer target: {name}")
        if result_count < 0:
            raise VMError("negative result count")

        linked = self.program.functions[name]
        argv = tuple(value & MASK64 for value in args)
        result_words = max(1, result_count)
        total = linked.frame_size + len(argv) * WORD + result_words * WORD
        callee_sp = (stack_top - total) & MASK64
        arg_base = callee_sp + linked.frame_size
        result_base = arg_base + len(argv) * WORD
        entry = self.program.block_code[
            (name, linked.function.blocks[0].label)
        ]

        # A fresh machine task has no P3 caller. Point CALLER_SP at the frame
        # itself so an unexpected return writes RESUME_PC locally and halts.
        self.memory.write(callee_sp + CALLER_SP, 64, callee_sp)
        self.memory.write(callee_sp + RET_PC, 64, self.program.halt_code)
        self.memory.write(callee_sp + ENTRY, 64, entry)
        self.memory.write(callee_sp + FRAME_SIZE, 64, linked.frame_size)
        self.memory.write(callee_sp + RESULT_PTR, 64, result_base)
        self.memory.write(callee_sp + RESULT_COUNT, 64, result_count)
        self.memory.write(callee_sp + RESUME_PC, 64, self.program.halt_code)
        self.memory.write(callee_sp + ARG_COUNT, 64, len(argv))
        for i, value in enumerate(argv):
            self.memory.write(arg_base + i * WORD, 64, value)

        self.sp = callee_sp
        self.current_function = name
        self.current_block = linked.function.blocks[0].label
        self.ip = 0
        self.halted = False

    def install_function(self, function: p3.Function) -> None:
        """Install a P3 function after this VM has already been created.

        Runtime-loaded MiniMachine userspace arrives through Linux bFLT, so its
        P3 body is not necessarily known when Program/new_vm() are built.  Add
        the function to the shared Program and mirror the newly allocated
        descriptor bytes into this VM's live memory image.
        """
        before_data = self.program._next_data
        self.program.add_function(function)
        after_data = self.program._next_data
        for address in range(before_data, after_data):
            self.memory.bytes[address] = self.program.initial_memory.bytes.get(address, 0)

    def alloc_bytes(self, size: int, *, align: int = 8) -> int:
        if size < 0:
            raise VMError("negative allocation size")
        if align <= 0 or (align & (align - 1)):
            raise VMError("allocation alignment must be a power of two")
        address = (self.heap_next + align - 1) & ~(align - 1)
        self.heap_next = address + max(1, size)
        if self.heap_next >= self.stack_top:
            raise VMError("VM heap collided with stack")
        return address

    @staticmethod
    def _mask(bits: int) -> int:
        return (1 << bits) - 1

    @staticmethod
    def _signed(value: int, bits: int) -> int:
        value &= (1 << bits) - 1
        sign = 1 << (bits - 1)
        return value - (1 << bits) if value & sign else value

    def _linked(self) -> LinkedFunction:
        if self.current_function is None:
            raise VMError("no current P3 function")
        return self.program.functions[self.current_function]

    def _slot_address(self, slot: muir.Slot) -> int:
        linked = self._linked()
        try:
            offset = linked.slot_offsets[slot.name]
        except KeyError as exc:
            raise VMError(
                f"unknown slot {slot.name} in {self.current_function}"
            ) from exc
        return (self.sp + offset) & MASK64

    def _resolve_symbol(self, symbol: str) -> int:
        try:
            return self.program.symbol_addresses[symbol]
        except KeyError as exc:
            raise VMError(f"unresolved symbol: {symbol}") from exc

    def _read_value(self, value: p3.Value) -> int:
        if isinstance(value, muir.Imm):
            return value.value & MASK64
        if isinstance(value, muir.Slot):
            return self.memory.read(self._slot_address(value), 64)
        if isinstance(value, muir.Special):
            if value is muir.Special.SP:
                return self.sp
            raise VMError(f"unsupported special value: {value}")
        if isinstance(value, muir.Symbol):
            return self._resolve_symbol(value.name)
        if isinstance(value, muir.Reloc):
            return (self._resolve_symbol(value.symbol) + value.addend) & MASK64
        if isinstance(value, muir.BlockAddr):
            try:
                return self.program.block_code[(value.function, value.label)]
            except KeyError as exc:
                raise VMError(
                    f"unresolved block address: {value.function}:{value.label}"
                ) from exc
        raise VMError(f"unsupported P3 value: {value!r}")

    def _address(self, address: muir.Address) -> int:
        return (self._read_value(address.base) + address.offset) & MASK64

    def _read_operand(self, operand: p3.Operand) -> int:
        if isinstance(operand, p3.Mem):
            return self.memory.read(
                self._address(operand.address),
                operand.width.value,
            )
        return self._read_value(operand)

    def _write_operand(self, operand: p3.Operand, value: int, bits: int) -> None:
        value &= self._mask(bits)
        if isinstance(operand, p3.Mem):
            self.memory.write(
                self._address(operand.address),
                operand.width.value,
                value,
            )
            return
        if isinstance(operand, muir.Slot):
            # A slot is one 64-bit frame cell. Narrow writes replace the
            # logical slot value; high bits do not retain stale state.
            self.memory.write(self._slot_address(operand), 64, value)
            return
        if isinstance(operand, muir.Special):
            if operand is muir.Special.SP:
                self.sp = value & MASK64
                return
        raise VMError(f"illegal MOV destination: {operand!r}")

    def _mov_value(self, inst: p3.Mov) -> int:
        raw = self._read_operand(inst.src)
        dst_bits = inst.width.value

        if inst.extend is None:
            return raw & self._mask(dst_bits)

        src_bits = inst.src_bits
        if src_bits is None:
            if isinstance(inst.src, p3.Mem):
                src_bits = inst.src.width.value
            else:
                raise VMError(
                    f"{inst.extend} MOV is missing source bit width"
                )
        if not (1 <= src_bits <= 64):
            raise VMError(f"invalid MOV source width: {src_bits}")

        raw &= self._mask(src_bits)
        if inst.extend == "zext":
            return raw & self._mask(dst_bits)
        if inst.extend == "sext":
            return self._signed(raw, src_bits) & self._mask(dst_bits)
        if inst.extend == "trunc":
            return raw & self._mask(dst_bits)
        raise VMError(f"unknown MOV extension mode: {inst.extend}")

    def _condition(self, inst: p3.Br) -> bool:
        bits = inst.width.value
        a = self._read_value(inst.a) & self._mask(bits)
        b = self._read_value(inst.b) & self._mask(bits)
        if inst.cond is muir.Cond.EQ:
            return a == b
        if inst.cond is muir.Cond.ULT:
            return a < b
        if inst.cond is muir.Cond.SLT:
            return self._signed(a, bits) < self._signed(b, bits)
        raise VMError(f"unsupported branch condition: {inst.cond}")

    def _target_code(self, target: muir.Target) -> int:
        if target.is_direct():
            assert target.label is not None
            if self.current_function is None:
                raise VMError("local branch without current function")
            try:
                return self.program.block_code[
                    (self.current_function, target.label)
                ]
            except KeyError as exc:
                raise VMError(f"unknown local target: {target.label}") from exc

        if target.is_external():
            assert target.symbol is not None
            # Direct external BR is used by trap endpoints. Host services have
            # descriptors for calls, but also expose their host entry here.
            for code, symbol in self.program.host_code.items():
                if symbol == target.symbol:
                    return code
            raise VMError(f"unresolved external branch: {target.symbol}")

        if target.slot is not None:
            return self._read_value(target.slot)
        if target.address is not None:
            return self.memory.read(self._address(target.address), 64)
        raise VMError(f"invalid branch target: {target!r}")

    def _set_code(self, code: int) -> None:
        if code == self.program.halt_code:
            self.halted = True
            return
        if code in self.program.host_code:
            symbol = self.program.host_code[code]
            if symbol == "__halt__":
                self.halted = True
                return
            self._invoke_host(symbol)
            return
        try:
            function, block = self.program.code_block[code]
        except KeyError as exc:
            raise VMError(f"invalid code address: 0x{code:x}") from exc
        self.current_function = function
        self.current_block = block
        self.ip = 0

    def _invoke_host(self, symbol: str) -> None:
        callback = self.program.host_services.get(symbol)
        if callback is None:
            raise VMError(f"host service is not registered: {symbol}")

        frame_size = self.memory.read(self.sp + FRAME_SIZE, 64)
        argc = self.memory.read(self.sp + ARG_COUNT, 64)
        arg_base = self.sp + frame_size
        args = tuple(
            self.memory.read(arg_base + i * WORD, 64)
            for i in range(argc)
        )

        raw_result = callback(self, args)
        if raw_result is HOST_CONTROL_TRANSFER:
            return
        if raw_result is None:
            results: tuple[int, ...] = ()
        elif isinstance(raw_result, int):
            results = (raw_result,)
        else:
            results = tuple(raw_result)

        expected = self.memory.read(self.sp + RESULT_COUNT, 64)
        if len(results) != expected:
            raise VMError(
                f"{symbol} returned {len(results)} values, caller expects {expected}"
            )
        result_ptr = self.memory.read(self.sp + RESULT_PTR, 64)
        for i, value in enumerate(results):
            self.memory.write(result_ptr + i * WORD, 64, value)

        caller_sp = self.memory.read(self.sp + CALLER_SP, 64)
        ret_pc = self.memory.read(self.sp + RET_PC, 64)
        self.sp = caller_sp
        self._set_code(ret_pc)

    def step(self) -> None:
        if self.halted:
            return
        linked = self._linked()
        if self.current_block is None:
            raise VMError("no current P3 block")
        block = linked.block_map[self.current_block]
        if self.ip >= len(block.instructions):
            raise VMError(
                f"fell off P3 block {self.current_function}:{self.current_block}"
            )

        inst = block.instructions[self.ip]
        self.steps += 1

        if isinstance(inst, p3.Mov):
            value = self._mov_value(inst)
            self._write_operand(inst.dst, value, inst.width.value)
            self.ip += 1
            return

        if isinstance(inst, p3.Sub):
            bits = inst.width.value
            value = (
                self._read_value(inst.a) - self._read_value(inst.b)
            ) & self._mask(bits)
            self._write_operand(inst.dst, value, bits)
            self.ip += 1
            return

        if isinstance(inst, p3.Br):
            target = inst.true_target if self._condition(inst) else inst.false_target
            self._set_code(self._target_code(target))
            return

        raise VMError(f"non-P3 instruction at runtime: {type(inst).__name__}")

    def run(self, *, max_steps: int = 1_000_000) -> None:
        while not self.halted:
            if self.steps >= max_steps:
                raise VMError(f"step limit exceeded: {max_steps}")
            self.step()

    def run_function(
        self,
        name: str,
        args: Iterable[int] = (),
        *,
        result_count: int = 1,
        max_steps: int = 1_000_000,
    ) -> tuple[int, ...]:
        if name not in self.program.functions:
            raise VMError(f"unknown entry function: {name}")
        if result_count < 0:
            raise VMError("negative result count")

        linked = self.program.functions[name]
        argv = tuple(value & MASK64 for value in args)
        result_words = max(1, result_count)
        total = linked.frame_size + len(argv) * WORD + result_words * WORD
        caller_sp = self.stack_top
        callee_sp = (caller_sp - total) & MASK64
        arg_base = callee_sp + linked.frame_size
        result_base = arg_base + len(argv) * WORD
        entry = self.program.block_code[(name, linked.function.blocks[0].label)]

        self.memory.write(callee_sp + CALLER_SP, 64, caller_sp)
        self.memory.write(callee_sp + RET_PC, 64, self.program.halt_code)
        self.memory.write(callee_sp + ENTRY, 64, entry)
        self.memory.write(callee_sp + FRAME_SIZE, 64, linked.frame_size)
        self.memory.write(callee_sp + RESULT_PTR, 64, result_base)
        self.memory.write(callee_sp + RESULT_COUNT, 64, result_count)
        self.memory.write(callee_sp + ARG_COUNT, 64, len(argv))
        for i, value in enumerate(argv):
            self.memory.write(arg_base + i * WORD, 64, value)

        self.sp = callee_sp
        self.current_function = name
        self.current_block = linked.function.blocks[0].label
        self.ip = 0
        self.halted = False
        self.steps = 0
        self.run(max_steps=max_steps)

        return tuple(
            self.memory.read(result_base + i * WORD, 64)
            for i in range(result_count)
        )
