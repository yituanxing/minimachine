from __future__ import annotations

import ctypes
import os
from pathlib import Path

from . import muir, p3
from .vm import DEFAULT_STACK_TOP, MASK64, VM, VMError

MM_OP_MOV = 1
MM_OP_SUB = 2
MM_OP_BR = 3

MM_V_IMM = 1
MM_V_SLOT = 2
MM_V_SP = 3
MM_V_MEM = 4
MM_V_INVALID = 5

MM_EXT_NONE = 0
MM_EXT_ZEXT = 1
MM_EXT_SEXT = 2
MM_EXT_TRUNC = 3

MM_COND_EQ = 1
MM_COND_ULT = 2
MM_COND_SLT = 3

MM_T_CODE = 1
MM_T_SLOT = 2
MM_T_MEM = 3

MM_STATUS_LIMIT = 0
MM_STATUS_HALT = 1
MM_STATUS_HOST = 2
MM_STATUS_WATCH = 3
MM_STATUS_ERROR = 4


class COperand(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_uint8),
        ("width", ctypes.c_uint8),
        ("base_kind", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8),
        ("offset", ctypes.c_int64),
        ("value", ctypes.c_uint64),
        ("base_value", ctypes.c_uint64),
    ]


class CInst(ctypes.Structure):
    _fields_ = [
        ("opcode", ctypes.c_uint8),
        ("width", ctypes.c_uint8),
        ("cond", ctypes.c_uint8),
        ("extend", ctypes.c_uint8),
        ("src_bits", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 3),
        ("dst", COperand),
        ("src", COperand),
        ("a", COperand),
        ("b", COperand),
        ("t", COperand),
        ("f", COperand),
    ]


class CBlock(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint64),
        ("first", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
    ]


class CRunResult(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int),
        ("error", ctypes.c_int),
        ("target_code", ctypes.c_uint64),
        ("block_code", ctypes.c_uint64),
        ("sp", ctypes.c_uint64),
        ("steps", ctypes.c_uint64),
        ("ip", ctypes.c_uint32),
    ]


def _load_library():
    candidates = []
    override = os.environ.get("MINIMACHINE_NATIVE_LIB")
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            root / "build" / "libminimachine_p3vm.so",
            root / "native" / "libminimachine_p3vm.so",
        ]
    )
    for path in candidates:
        if path.is_file():
            lib = ctypes.CDLL(str(path))
            break
    else:
        raise VMError(
            "native P3 VM library not found; compile native/p3vm.c or set "
            "MINIMACHINE_NATIVE_LIB"
        )

    lib.mm_vm_create.argtypes = [
        ctypes.POINTER(CInst), ctypes.c_size_t,
        ctypes.POINTER(CBlock), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
        ctypes.c_uint64,
    ]
    lib.mm_vm_create.restype = ctypes.c_void_p
    lib.mm_vm_destroy.argtypes = [ctypes.c_void_p]
    lib.mm_vm_destroy.restype = None
    lib.mm_vm_replace_program.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(CInst), ctypes.c_size_t,
        ctypes.POINTER(CBlock), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
        ctypes.c_uint64,
    ]
    lib.mm_vm_replace_program.restype = ctypes.c_int
    lib.mm_vm_load_bytes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.mm_vm_load_bytes.restype = ctypes.c_int
    lib.mm_vm_mem_read.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint
    ]
    lib.mm_vm_mem_read.restype = ctypes.c_uint64
    lib.mm_vm_mem_write.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint, ctypes.c_uint64
    ]
    lib.mm_vm_mem_write.restype = None
    lib.mm_vm_set_watches.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t
    ]
    lib.mm_vm_set_watches.restype = ctypes.c_int
    lib.mm_vm_set_state.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32,
        ctypes.c_uint64, ctypes.c_uint64,
    ]
    lib.mm_vm_set_state.restype = None
    lib.mm_vm_run.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.mm_vm_run.restype = CRunResult
    return lib


class NativeMemory:
    def __init__(self, lib, handle):
        self._lib = lib
        self._handle = handle

    def read(self, address: int, bits: int) -> int:
        if bits not in {8, 16, 32, 64}:
            raise VMError(f"unsupported memory width: {bits}")
        return int(
            self._lib.mm_vm_mem_read(
                self._handle, address & MASK64, bits
            )
        )

    def write(self, address: int, bits: int, value: int) -> None:
        if bits not in {8, 16, 32, 64}:
            raise VMError(f"unsupported memory width: {bits}")
        self._lib.mm_vm_mem_write(
            self._handle,
            address & MASK64,
            bits,
            value & MASK64,
        )


class NativeVM(VM):
    """C-backed strict-P3 executor preserving the Python VM control contract."""

    def __init__(self, program, *, stack_top: int = DEFAULT_STACK_TOP):
        self._lib = _load_library()
        self._packed = None
        insts, blocks, hosts = self._pack_program(program)
        handle = self._lib.mm_vm_create(
            insts,
            len(insts),
            blocks,
            len(blocks),
            hosts,
            len(hosts),
            program.halt_code,
        )
        if not handle:
            raise VMError("cannot create native P3 VM")
        self._handle = handle
        memory = NativeMemory(self._lib, handle)
        super().__init__(program, memory, stack_top=stack_top)
        self._program_shape = self._shape(program)
        self._watch_codes: tuple[int, ...] = ()
        self._load_initial_memory(program)

    def __del__(self):
        handle = getattr(self, "_handle", None)
        lib = getattr(self, "_lib", None)
        if handle and lib:
            try:
                lib.mm_vm_destroy(handle)
            except Exception:
                pass
            self._handle = None

    @staticmethod
    def _shape(program):
        return (
            len(program.functions),
            len(program.block_code),
            len(program.host_code),
        )

    def _load_initial_memory(self, program) -> None:
        items = list(program.initial_memory.bytes.items())
        if not items:
            return
        addresses = (ctypes.c_uint64 * len(items))(
            *(address & MASK64 for address, _ in items)
        )
        values = (ctypes.c_uint8 * len(items))(
            *(value & 0xFF for _, value in items)
        )
        if not self._lib.mm_vm_load_bytes(
            self._handle, addresses, values, len(items)
        ):
            raise VMError("cannot load native P3 initial memory")

    def _resolved_value(self, value, linked, program):
        if isinstance(value, muir.Imm):
            return MM_V_IMM, value.value & MASK64
        if isinstance(value, muir.Slot):
            try:
                return MM_V_SLOT, linked.slot_offsets[value.name]
            except KeyError as exc:
                raise VMError(
                    f"unknown slot {value.name} while packing native P3"
                ) from exc
        if isinstance(value, muir.Special):
            if value is muir.Special.SP:
                return MM_V_SP, 0
            raise VMError(f"unsupported native special value: {value}")
        if isinstance(value, muir.Symbol):
            address = program.symbol_addresses.get(value.name)
            if address is None:
                return MM_V_INVALID, 0
            return MM_V_IMM, address
        if isinstance(value, muir.Reloc):
            base = program.symbol_addresses.get(value.symbol)
            if base is None:
                return MM_V_INVALID, 0
            return MM_V_IMM, (base + value.addend) & MASK64
        if isinstance(value, muir.BlockAddr):
            try:
                return (
                    MM_V_IMM,
                    program.block_code[(value.function, value.label)],
                )
            except KeyError as exc:
                raise VMError(
                    "unresolved native block address: "
                    f"{value.function}:{value.label}"
                ) from exc
        raise VMError(f"unsupported native P3 value: {value!r}")

    def _operand(self, operand, linked, program):
        out = COperand()
        if isinstance(operand, p3.Mem):
            out.kind = MM_V_MEM
            out.width = operand.width.value
            base_kind, base_value = self._resolved_value(
                operand.address.base, linked, program
            )
            out.base_kind = base_kind
            out.base_value = base_value
            out.offset = operand.address.offset
            return out
        kind, value = self._resolved_value(operand, linked, program)
        out.kind = kind
        out.value = value
        return out

    def _target(
        self,
        target,
        function_name,
        linked,
        program,
        host_by_symbol,
    ):
        out = COperand()
        if target.is_direct():
            out.kind = MM_T_CODE
            out.value = program.block_code[
                (function_name, target.label)
            ]
            return out
        if target.is_external():
            code = host_by_symbol.get(target.symbol)
            if code is None:
                out.kind = 255
                return out
            out.kind = MM_T_CODE
            out.value = code
            return out
        if target.slot is not None:
            out.kind = MM_T_SLOT
            out.value = linked.slot_offsets[target.slot.name]
            return out
        if target.address is not None:
            out.kind = MM_T_MEM
            base_kind, base_value = self._resolved_value(
                target.address.base, linked, program
            )
            out.base_kind = base_kind
            out.base_value = base_value
            out.offset = target.address.offset
            out.width = 64
            return out
        raise VMError(f"invalid native branch target: {target!r}")

    def _pack_program(self, program):
        host_by_symbol = {
            symbol: code for code, symbol in program.host_code.items()
        }
        packed_insts = []
        packed_blocks = []

        for code, (function_name, block_name) in sorted(
            program.code_block.items()
        ):
            linked = program.functions[function_name]
            block = linked.block_map[block_name]
            first = len(packed_insts)

            for inst in block.instructions:
                out = CInst()
                if isinstance(inst, p3.Mov):
                    out.opcode = MM_OP_MOV
                    out.width = inst.width.value
                    out.extend = {
                        None: MM_EXT_NONE,
                        "zext": MM_EXT_ZEXT,
                        "sext": MM_EXT_SEXT,
                        "trunc": MM_EXT_TRUNC,
                    }[inst.extend]
                    out.src_bits = inst.src_bits or 0
                    out.dst = self._operand(
                        inst.dst, linked, program
                    )
                    out.src = self._operand(
                        inst.src, linked, program
                    )
                elif isinstance(inst, p3.Sub):
                    out.opcode = MM_OP_SUB
                    out.width = inst.width.value
                    out.dst = self._operand(
                        inst.dst, linked, program
                    )
                    out.a = self._operand(
                        inst.a, linked, program
                    )
                    out.b = self._operand(
                        inst.b, linked, program
                    )
                elif isinstance(inst, p3.Br):
                    out.opcode = MM_OP_BR
                    out.width = inst.width.value
                    out.cond = {
                        muir.Cond.EQ: MM_COND_EQ,
                        muir.Cond.ULT: MM_COND_ULT,
                        muir.Cond.SLT: MM_COND_SLT,
                    }[inst.cond]
                    out.a = self._operand(
                        inst.a, linked, program
                    )
                    out.b = self._operand(
                        inst.b, linked, program
                    )
                    out.t = self._target(
                        inst.true_target,
                        function_name,
                        linked,
                        program,
                        host_by_symbol,
                    )
                    out.f = self._target(
                        inst.false_target,
                        function_name,
                        linked,
                        program,
                        host_by_symbol,
                    )
                else:
                    raise VMError(
                        "non-P3 instruction reached native packer: "
                        f"{type(inst).__name__}"
                    )
                packed_insts.append(out)

            packed_blocks.append(
                CBlock(
                    code=code,
                    first=first,
                    count=len(block.instructions),
                )
            )

        inst_array = (CInst * len(packed_insts))(*packed_insts)
        block_array = (CBlock * len(packed_blocks))(*packed_blocks)
        host_codes = sorted(program.host_code)
        host_array = (ctypes.c_uint64 * len(host_codes))(
            *host_codes
        )
        self._packed = (
            inst_array,
            block_array,
            host_array,
        )
        return inst_array, block_array, host_array

    def _ensure_program_current(self):
        shape = self._shape(self.program)
        if shape == self._program_shape:
            return
        insts, blocks, hosts = self._pack_program(self.program)
        if not self._lib.mm_vm_replace_program(
            self._handle,
            insts,
            len(insts),
            blocks,
            len(blocks),
            hosts,
            len(hosts),
            self.program.halt_code,
        ):
            raise VMError("cannot refresh native P3 program")
        self._program_shape = shape
        self.set_watch_codes(self._watch_codes)

    def set_watch_codes(self, codes) -> None:
        self._watch_codes = tuple(
            sorted(set(int(code) & MASK64 for code in codes))
        )
        array = (ctypes.c_uint64 * len(self._watch_codes))(
            *self._watch_codes
        )
        if not self._lib.mm_vm_set_watches(
            self._handle,
            array,
            len(array),
        ):
            raise VMError("cannot install native P3 watch codes")

    def _current_code(self) -> int:
        if self.current_function is None or self.current_block is None:
            raise VMError("native P3 VM has no current block")
        try:
            return self.program.block_code[
                (self.current_function, self.current_block)
            ]
        except KeyError as exc:
            raise VMError(
                "native current block is not linked: "
                f"{self.current_function}:{self.current_block}"
            ) from exc

    def _sync_block(self, code: int, ip: int) -> None:
        try:
            function, block = self.program.code_block[code]
        except KeyError:
            return
        self.current_function = function
        self.current_block = block
        self.ip = int(ip)

    def run(self, *, max_steps: int = 1_000_000) -> None:
        while not self.halted:
            if self.steps >= max_steps:
                raise VMError(
                    f"step limit exceeded: {max_steps}"
                )

            self._ensure_program_current()
            code = self._current_code()
            self._lib.mm_vm_set_state(
                self._handle,
                code,
                self.ip,
                self.sp & MASK64,
                self.steps,
            )
            result = self._lib.mm_vm_run(
                self._handle,
                max_steps,
            )
            self.sp = int(result.sp)
            self.steps = int(result.steps)
            self._sync_block(
                int(result.block_code),
                int(result.ip),
            )

            if result.status == MM_STATUS_LIMIT:
                raise VMError(
                    f"step limit exceeded: {max_steps}"
                )
            if result.status == MM_STATUS_HALT:
                self.halted = True
                return
            if result.status == MM_STATUS_HOST:
                self._set_code(
                    int(result.target_code)
                )
                continue
            if result.status == MM_STATUS_WATCH:
                self._set_code(
                    int(result.target_code)
                )
                continue
            if result.status == MM_STATUS_ERROR:
                raise VMError(
                    "native P3 execution error: "
                    f"code={result.error} "
                    f"target=0x{int(result.target_code):x}"
                )
            raise VMError(
                f"unknown native P3 run status: {result.status}"
            )
