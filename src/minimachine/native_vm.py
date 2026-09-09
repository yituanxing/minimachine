from __future__ import annotations

import ctypes
import hashlib
import mmap
import os
from pathlib import Path
import re
import struct
import time

from . import muir, p3
from .abi import CALLER_SP, RET_PC
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
MM_T_LOCAL_BLOCK = 4

MM_INTR_NONE = 0
MM_INTR_AND = 1
MM_INTR_OR = 2
MM_INTR_XOR = 3
MM_INTR_SHL = 4
MM_INTR_LSHR = 5
MM_INTR_ASHR = 6
MM_INTR_ADD = 7
MM_INTR_MUL = 8
MM_INTR_ICMP = 9
MM_INTR_ALLOCA = 10
MM_INTR_FREE = 11
MM_INTR_EXPECT = 12
MM_INTR_PTR_ADD_SCALED = 13
MM_INTR_ROR32 = 14
MM_INTR_LOAD_I128 = 15
MM_INTR_STORE_I128 = 16
MM_INTR_IS_CONSTANT = 17
MM_INTR_PREFETCH = 18
MM_INTR_UDIV = 19
MM_INTR_SDIV = 20
MM_INTR_UREM = 21
MM_INTR_SREM = 22
MM_INTR_WIDE_CONST_I128 = 23
MM_INTR_ICMP_EQ_I128 = 24
MM_INTR_MEMSET = 25
MM_INTR_MEMCPY = 26

MM_IPRED_EQ = 1
MM_IPRED_NE = 2
MM_IPRED_ULT = 3
MM_IPRED_ULE = 4
MM_IPRED_UGT = 5
MM_IPRED_UGE = 6
MM_IPRED_SLT = 7
MM_IPRED_SLE = 8
MM_IPRED_SGT = 9
MM_IPRED_SGE = 10

MM_STATUS_LIMIT = 0
MM_STATUS_HALT = 1
MM_STATUS_HOST = 2
MM_STATUS_WATCH = 3
MM_STATUS_ERROR = 4
_NATIVE_PAGE_SIZE = 65536
_NATIVE_PACK_MAGIC = b"MMP3NP1\0"
_NATIVE_PACK_VERSION = 1
_NATIVE_PACK_HEADER = struct.Struct("<8sIIIIQQQQ32s40x")
_NATIVE_PACK_HEADER_SIZE = _NATIVE_PACK_HEADER.size


def _native_local_targets_enabled() -> bool:
    return os.environ.get(
        "MINIMACHINE_NATIVE_LOCAL_TARGETS", "0"
    ).lower() not in {"0", "false", "no", "off", ""}


def _native_host_intrinsics_enabled() -> bool:
    return os.environ.get(
        "MINIMACHINE_NATIVE_HOST_INTRINSICS", "1"
    ).lower() not in {"0", "false", "no", "off", ""}


def _native_free_intrinsic_enabled() -> bool:
    return os.environ.get(
        "MINIMACHINE_NATIVE_FREE_INTRINSIC", "1"
    ).lower() not in {"0", "false", "no", "off", ""}


def _native_simple_intrinsics_enabled() -> bool:
    return os.environ.get(
        "MINIMACHINE_NATIVE_SIMPLE_INTRINSICS", "1"
    ).lower() not in {"0", "false", "no", "off", ""}


def _native_memory_intrinsics_enabled() -> bool:
    return os.environ.get(
        "MINIMACHINE_NATIVE_MEMORY_INTRINSICS", "1"
    ).lower() not in {"0", "false", "no", "off", ""}


def _native_i128_value_intrinsics_enabled() -> bool:
    return os.environ.get(
        "MINIMACHINE_NATIVE_I128_VALUE_INTRINSICS", "1"
    ).lower() not in {"0", "false", "no", "off", ""}


def _native_scalar_intrinsics_enabled() -> bool:
    return os.environ.get(
        "MINIMACHINE_NATIVE_SCALAR_INTRINSICS", "1"
    ).lower() not in {"0", "false", "no", "off", ""}


def _native_ror32_intrinsic_enabled() -> bool:
    return os.environ.get(
        "MINIMACHINE_NATIVE_ROR32_INTRINSIC", "1"
    ).lower() not in {"0", "false", "no", "off", ""}


def _native_i128_intrinsics_enabled() -> bool:
    return os.environ.get(
        "MINIMACHINE_NATIVE_I128_INTRINSICS", "1"
    ).lower() not in {"0", "false", "no", "off", ""}


def native_pack_schema_fingerprint() -> str:
    """Fingerprint everything that can change the packed native code image."""
    from .program_cache import lowering_fingerprint

    root = Path(__file__).resolve().parent
    repo = root.parents[1]
    digest = hashlib.sha256()
    digest.update(lowering_fingerprint().encode("ascii"))
    digest.update(b"\0")
    for path in (
        root / "native_vm.py",
        root / "runtime.py",
        root / "vm.py",
        root / "kallsyms.py",
        root / "linker.py",
        repo / "scripts" / "run-minimachine-linux.py",
    ):
        digest.update(str(path.relative_to(repo)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def native_pack_cache_key(
    *,
    image_sha256: str,
    initramfs_sha256: str | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(image_sha256.encode("ascii"))
    digest.update(b"\0")
    digest.update((initramfs_sha256 or "-").encode("ascii"))
    digest.update(b"\0")
    digest.update(native_pack_schema_fingerprint().encode("ascii"))
    digest.update(b"\0local-targets=")
    digest.update(b"1" if _native_local_targets_enabled() else b"0")
    return digest.hexdigest()


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


class CHostIntrinsic(ctypes.Structure):
    _fields_ = [
        ("op", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("pred", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8),
        ("imm", ctypes.c_uint32),
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
    lib.mm_vm_add_segment.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(CInst), ctypes.c_size_t,
        ctypes.POINTER(CBlock), ctypes.c_size_t,
    ]
    lib.mm_vm_add_segment.restype = ctypes.c_int
    lib.mm_vm_set_host_codes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    ]
    lib.mm_vm_set_host_codes.restype = ctypes.c_int
    lib.mm_vm_set_host_intrinsics.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(CHostIntrinsic),
        ctypes.c_size_t,
    ]
    lib.mm_vm_set_host_intrinsics.restype = ctypes.c_int
    lib.mm_vm_set_heap_state.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64
    ]
    lib.mm_vm_set_heap_state.restype = None
    lib.mm_vm_get_heap_next.argtypes = [ctypes.c_void_p]
    lib.mm_vm_get_heap_next.restype = ctypes.c_uint64
    lib.mm_vm_alloc_bytes.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.mm_vm_alloc_bytes.restype = ctypes.c_int
    lib.mm_vm_freed_count.argtypes = [ctypes.c_void_p]
    lib.mm_vm_freed_count.restype = ctypes.c_size_t
    lib.mm_vm_freed_data.argtypes = [ctypes.c_void_p]
    lib.mm_vm_freed_data.restype = ctypes.POINTER(ctypes.c_uint64)
    lib.mm_vm_clear_freed.argtypes = [ctypes.c_void_p]
    lib.mm_vm_clear_freed.restype = None
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
    lib.mm_vm_mem_read_blob.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
    ]
    lib.mm_vm_mem_read_blob.restype = ctypes.c_int
    lib.mm_vm_mem_write.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint, ctypes.c_uint64
    ]
    lib.mm_vm_mem_write.restype = None
    lib.mm_vm_mem_write_blob.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
    ]
    lib.mm_vm_mem_write_blob.restype = ctypes.c_int
    lib.mm_vm_mem_fill.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint8, ctypes.c_uint64
    ]
    lib.mm_vm_mem_fill.restype = ctypes.c_int
    lib.mm_vm_mem_copy.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64
    ]
    lib.mm_vm_mem_copy.restype = ctypes.c_int
    lib.mm_vm_mem_move.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64
    ]
    lib.mm_vm_mem_move.restype = ctypes.c_int
    lib.mm_vm_mem_compare.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64
    ]
    lib.mm_vm_mem_compare.restype = ctypes.c_int
    lib.mm_vm_mem_strlen.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64
    ]
    lib.mm_vm_mem_strlen.restype = ctypes.c_uint64
    lib.mm_vm_mem_strcmp.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64
    ]
    lib.mm_vm_mem_strcmp.restype = ctypes.c_int
    lib.mm_vm_mem_strncmp.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64
    ]
    lib.mm_vm_mem_strncmp.restype = ctypes.c_int
    lib.mm_vm_mem_page_count.argtypes = [ctypes.c_void_p]
    lib.mm_vm_mem_page_count.restype = ctypes.c_size_t
    lib.mm_vm_mem_export_pages.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.mm_vm_mem_export_pages.restype = ctypes.c_size_t
    lib.mm_vm_mem_restore_pages.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.mm_vm_mem_restore_pages.restype = ctypes.c_int
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

    def bulk_read(self, src: int, size: int) -> bytes:
        if size < 0:
            raise VMError("negative native bulk read size")
        if not size:
            return b""
        buf = (ctypes.c_uint8 * size)()
        if not self._lib.mm_vm_mem_read_blob(
            self._handle,
            src & MASK64,
            buf,
            size,
        ):
            raise VMError("native bulk read failed")
        return bytes(buf)

    def bulk_write(self, dst: int, data: bytes | bytearray | memoryview) -> None:
        if not data:
            return
        if isinstance(data, bytearray):
            size = len(data)
            buf = (ctypes.c_uint8 * size).from_buffer(data)
        else:
            raw = bytes(data)
            size = len(raw)
            buf = (ctypes.c_uint8 * size).from_buffer_copy(raw)
        if not self._lib.mm_vm_mem_write_blob(
            self._handle,
            dst & MASK64,
            buf,
            size,
        ):
            raise VMError("native bulk write failed")

    def bulk_set(self, dst: int, value: int, size: int) -> None:
        if size < 0:
            raise VMError("negative native bulk memset size")
        if not self._lib.mm_vm_mem_fill(
            self._handle,
            dst & MASK64,
            value & 0xFF,
            size,
        ):
            raise VMError("native bulk memset failed")

    def bulk_copy(self, dst: int, src: int, size: int) -> None:
        if size < 0:
            raise VMError("negative native bulk memcpy size")
        if not self._lib.mm_vm_mem_copy(
            self._handle,
            dst & MASK64,
            src & MASK64,
            size,
        ):
            raise VMError("native bulk memcpy failed")

    def bulk_move(self, dst: int, src: int, size: int) -> None:
        if size < 0:
            raise VMError("negative native bulk memmove size")
        if not self._lib.mm_vm_mem_move(
            self._handle,
            dst & MASK64,
            src & MASK64,
            size,
        ):
            raise VMError("native bulk memmove failed")

    def bulk_compare(self, a: int, b: int, size: int) -> int:
        if size < 0:
            raise VMError("negative native bulk memcmp size")
        return int(
            self._lib.mm_vm_mem_compare(
                self._handle,
                a & MASK64,
                b & MASK64,
                size,
            )
        )

    def bulk_strlen(self, ptr: int) -> int:
        return int(
            self._lib.mm_vm_mem_strlen(
                self._handle,
                ptr & MASK64,
            )
        )

    def bulk_strcmp(self, a: int, b: int) -> int:
        return int(
            self._lib.mm_vm_mem_strcmp(
                self._handle,
                a & MASK64,
                b & MASK64,
            )
        )

    def bulk_strncmp(self, a: int, b: int, size: int) -> int:
        if size < 0:
            raise VMError("negative native strncmp size")
        return int(
            self._lib.mm_vm_mem_strncmp(
                self._handle,
                a & MASK64,
                b & MASK64,
                size,
            )
        )


    def snapshot_pages(self):
        count = int(self._lib.mm_vm_mem_page_count(self._handle))
        page_nos = (ctypes.c_uint64 * count)()
        raw = (ctypes.c_uint8 * (count * _NATIVE_PAGE_SIZE))()
        written = int(
            self._lib.mm_vm_mem_export_pages(
                self._handle,
                page_nos,
                raw,
                count,
            )
        )
        if written != count:
            raise VMError(
                f"native checkpoint page export truncated: {written}/{count}"
            )
        return {
            "page_size": _NATIVE_PAGE_SIZE,
            "page_nos": tuple(int(page_nos[i]) for i in range(count)),
            "data": bytes(raw),
        }

    def restore_pages(self, snapshot) -> None:
        page_size = int(snapshot["page_size"])
        if page_size != _NATIVE_PAGE_SIZE:
            raise VMError(
                f"native checkpoint page size mismatch: {page_size}"
            )
        page_nos_tuple = tuple(int(x) for x in snapshot["page_nos"])
        data = bytes(snapshot["data"])
        expected = len(page_nos_tuple) * page_size
        if len(data) != expected:
            raise VMError(
                "native checkpoint page data size mismatch: "
                f"{len(data)} != {expected}"
            )
        page_nos = (ctypes.c_uint64 * len(page_nos_tuple))(
            *page_nos_tuple
        )
        raw = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        if not self._lib.mm_vm_mem_restore_pages(
            self._handle,
            page_nos,
            raw,
            len(page_nos_tuple),
            page_size,
        ):
            raise VMError("native checkpoint page restore failed")


class NativeVM(VM):
    """C-backed strict-P3 executor preserving the Python VM control contract."""

    def __init__(
        self,
        program,
        *,
        stack_top: int = DEFAULT_STACK_TOP,
        pack_cache_in: Path | None = None,
        pack_cache_out: Path | None = None,
        pack_cache_key: str | None = None,
        append_pack_cache_in_dir: Path | None = None,
        append_pack_cache_out_dir: Path | None = None,
        load_initial_memory: bool = True,
    ):
        self._lib = _load_library()
        self._packed = None
        self._packed_mmap = None
        self._extra_packed = []
        self._append_pack_mmaps = []
        self._host_packed = None
        self._pack_cache_key = pack_cache_key
        self._append_pack_cache_in_dir = append_pack_cache_in_dir
        self._append_pack_cache_out_dir = append_pack_cache_out_dir
        self._append_pack_index = 0
        self.native_append_cache_context = None
        self._host_profile_enabled = os.environ.get(
            "MINIMACHINE_NATIVE_HOST_PROFILE", ""
        ).lower() in {"1", "true", "yes", "on"}
        self._host_profile_counts: dict[str, int] = {}
        self._host_profile_seconds: dict[str, float] = {}
        self._host_profile_total_calls = 0
        self._host_profile_total_seconds = 0.0
        if pack_cache_in is not None:
            if pack_cache_key is None:
                raise VMError("native pack cache input requires a cache key")
            insts, blocks, hosts = self._load_packed_cache(
                program,
                pack_cache_in,
                cache_key=pack_cache_key,
            )
        else:
            insts, blocks, hosts = self._pack_program(program)
            if pack_cache_out is not None:
                if pack_cache_key is None:
                    raise VMError("native pack cache output requires a cache key")
                self._save_packed_cache(
                    program,
                    pack_cache_out,
                    insts,
                    blocks,
                    hosts,
                    cache_key=pack_cache_key,
                )
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
        self._sync_native_heap_state()
        self._host_intrinsic_packed = None
        self._install_native_host_intrinsics()
        self._program_shape = self._shape(program)
        self._packed_block_codes = set(program.code_block)
        self._packed_function_names = set(program.functions)
        self._packed_host_codes = set(program.host_code)
        self._watch_codes: tuple[int, ...] = ()
        self.native_report_every = 0
        self.native_report_slots: tuple[str, ...] = ()
        self._native_profile_run_calls = 0
        self._native_profile_c_ns = 0
        self._native_profile_host_returns = 0
        self._native_profile_watch_returns = 0
        if load_initial_memory:
            self._load_initial_memory(program)
        else:
            print(
                "BOOT_EXEC_NATIVE_INITIAL_MEMORY skipped=checkpoint-restore "
                f"bytes={len(program.initial_memory.bytes)}",
                flush=True,
            )
        self._synced_data_end = program._next_data

    def __del__(self):
        handle = getattr(self, "_handle", None)
        lib = getattr(self, "_lib", None)
        if handle and lib:
            try:
                lib.mm_vm_destroy(handle)
            except Exception:
                pass
            self._handle = None

    def _sync_native_heap_state(self) -> None:
        self._lib.mm_vm_set_heap_state(
            self._handle,
            int(self.heap_next) & MASK64,
            int(self.stack_top) & MASK64,
        )

    def _sync_python_heap_state(self) -> None:
        self.heap_next = int(self._lib.mm_vm_get_heap_next(self._handle))

    def alloc_bytes(self, size: int, *, align: int = 8) -> int:
        if size < 0:
            raise VMError("negative allocation size")
        if align <= 0 or (align & (align - 1)):
            raise VMError("allocation alignment must be a power of two")
        self._sync_native_heap_state()
        address = ctypes.c_uint64()
        if not self._lib.mm_vm_alloc_bytes(
            self._handle,
            int(size),
            int(align),
            ctypes.byref(address),
        ):
            raise VMError("VM heap collided with stack")
        self._sync_python_heap_state()
        return int(address.value)

    def _sync_native_frees(self) -> None:
        count = int(self._lib.mm_vm_freed_count(self._handle))
        if count == 0:
            return
        data = self._lib.mm_vm_freed_data(self._handle)
        if not data:
            raise VMError("native free queue is unavailable")
        allocations = getattr(self, "user_allocations", None)
        if allocations is not None:
            for index in range(count):
                allocations.pop(int(data[index]), None)
        self._lib.mm_vm_clear_freed(self._handle)

    @staticmethod
    def _native_intrinsic_for_symbol(symbol: str) -> CHostIntrinsic:
        out = CHostIntrinsic()
        if not _native_host_intrinsics_enabled():
            return out

        if symbol == "__mm_alloca":
            out.op = MM_INTR_ALLOCA
            return out

        if (
            _native_free_intrinsic_enabled()
            and (symbol == "__mm_user_ext_free" or symbol.endswith("_ext_free"))
        ):
            out.op = MM_INTR_FREE
            return out

        if (
            _native_ror32_intrinsic_enabled()
            and symbol == "__mm_fast_ror32"
        ):
            out.op = MM_INTR_ROR32
            return out

        if _native_i128_intrinsics_enabled():
            if symbol == "__mm_load_i128":
                out.op = MM_INTR_LOAD_I128
                return out
            if symbol == "__mm_store_i128":
                out.op = MM_INTR_STORE_I128
                return out

        if _native_simple_intrinsics_enabled():
            if re.fullmatch(
                r"__mm_llvm_expect(?:_with_probability)?_.+",
                symbol,
            ):
                out.op = MM_INTR_EXPECT
                return out

            match = re.fullmatch(r"__mm_ptr_add_scaled_(\d+)", symbol)
            if match:
                scale = int(match.group(1))
                if 0 <= scale <= 0xFFFFFFFF:
                    out.op = MM_INTR_PTR_ADD_SCALED
                    out.imm = scale
                return out

        if _native_memory_intrinsics_enabled():
            if symbol.startswith("__mm_llvm_memset_"):
                out.op = MM_INTR_MEMSET
                return out
            if symbol.startswith("__mm_llvm_memcpy_"):
                out.op = MM_INTR_MEMCPY
                return out

        if _native_i128_value_intrinsics_enabled():
            if symbol == "__mm_wide_const_128":
                out.op = MM_INTR_WIDE_CONST_I128
                return out
            if symbol == "__mm_icmp_eq_128":
                out.op = MM_INTR_ICMP_EQ_I128
                return out

        if _native_scalar_intrinsics_enabled():
            if re.fullmatch(r"__mm_llvm_is_constant_.+", symbol):
                out.op = MM_INTR_IS_CONSTANT
                return out

            if symbol.startswith("__mm_llvm_prefetch_"):
                out.op = MM_INTR_PREFETCH
                return out

            scalar_binary_ops = {
                "udiv": MM_INTR_UDIV,
                "sdiv": MM_INTR_SDIV,
                "urem": MM_INTR_UREM,
                "srem": MM_INTR_SREM,
            }
            match = re.fullmatch(
                r"__mm_(udiv|sdiv|urem|srem)_(32|64)",
                symbol,
            )
            if match:
                op_name, bits_text = match.groups()
                out.op = scalar_binary_ops[op_name]
                out.bits = int(bits_text)
                return out

        binary_ops = {
            "and": MM_INTR_AND,
            "or": MM_INTR_OR,
            "xor": MM_INTR_XOR,
            "shl": MM_INTR_SHL,
            "lshr": MM_INTR_LSHR,
            "ashr": MM_INTR_ASHR,
            "add": MM_INTR_ADD,
            "mul": MM_INTR_MUL,
        }
        match = re.fullmatch(
            r"__mm_(and|or|xor|shl|lshr|ashr|add|mul)_(\d+)",
            symbol,
        )
        if match:
            op_name, bits_text = match.groups()
            bits = int(bits_text)
            if 1 <= bits <= 64:
                out.op = binary_ops[op_name]
                out.bits = bits
            return out

        match = re.fullmatch(r"__mm_icmp_([a-z]+)_(\d+)", symbol)
        if match:
            pred_name, bits_text = match.groups()
            bits = int(bits_text)
            pred = {
                "eq": MM_IPRED_EQ,
                "ne": MM_IPRED_NE,
                "ult": MM_IPRED_ULT,
                "ule": MM_IPRED_ULE,
                "ugt": MM_IPRED_UGT,
                "uge": MM_IPRED_UGE,
                "slt": MM_IPRED_SLT,
                "sle": MM_IPRED_SLE,
                "sgt": MM_IPRED_SGT,
                "sge": MM_IPRED_SGE,
            }.get(pred_name)
            if pred is not None and 1 <= bits <= 64:
                out.op = MM_INTR_ICMP
                out.bits = bits
                out.pred = pred
            return out

        return out

    def _install_native_host_intrinsics(self) -> None:
        host_codes = sorted(self.program.host_code)
        array = (CHostIntrinsic * len(host_codes))()
        mapped = 0
        free_mapped = 0
        for index, code in enumerate(host_codes):
            symbol = self.program.host_code[code]
            descriptor = self._native_intrinsic_for_symbol(symbol)
            array[index] = descriptor
            if descriptor.op != MM_INTR_NONE:
                mapped += 1
            if descriptor.op == MM_INTR_FREE:
                free_mapped += 1
        if not self._lib.mm_vm_set_host_intrinsics(
            self._handle,
            array,
            len(array),
        ):
            raise VMError("cannot install native host intrinsic table")
        self._host_intrinsic_packed = array
        if _native_host_intrinsics_enabled():
            print(
                "BOOT_EXEC_NATIVE_HOST_INTRINSICS "
                f"mapped={mapped} hosts={len(host_codes)} "
                f"free_mapped={free_mapped}",
                flush=True,
            )

    def _save_packed_cache(
        self,
        program,
        path: Path,
        insts,
        blocks,
        hosts,
        *,
        cache_key: str,
        log_label: str = "BOOT_EXEC_NATIVE_PACK_CACHE_SAVED",
    ) -> None:
        started = time.perf_counter()
        try:
            key_bytes = bytes.fromhex(cache_key)
        except ValueError as exc:
            raise VMError("native pack cache key is not hexadecimal") from exc
        if len(key_bytes) != 32:
            raise VMError("native pack cache key must be a SHA-256 digest")

        path.parent.mkdir(parents=True, exist_ok=True)
        header = _NATIVE_PACK_HEADER.pack(
            _NATIVE_PACK_MAGIC,
            _NATIVE_PACK_VERSION,
            ctypes.sizeof(CInst),
            ctypes.sizeof(CBlock),
            ctypes.sizeof(ctypes.c_uint64),
            len(insts),
            len(blocks),
            len(hosts),
            program.halt_code & MASK64,
            key_bytes,
        )
        with path.open("wb") as handle:
            handle.write(header)
            handle.write(memoryview(insts).cast("B"))
            handle.write(memoryview(blocks).cast("B"))
            handle.write(memoryview(hosts).cast("B"))
        elapsed = time.perf_counter() - started
        print(
            f"{log_label} "
            f"path={path} seconds={elapsed:.3f} bytes={path.stat().st_size} "
            f"insts={len(insts)} blocks={len(blocks)} hosts={len(hosts)}",
            flush=True,
        )

    def _load_packed_cache(
        self,
        program,
        path: Path,
        *,
        cache_key: str,
    ):
        started = time.perf_counter()
        try:
            key_bytes = bytes.fromhex(cache_key)
        except ValueError as exc:
            raise VMError("native pack cache key is not hexadecimal") from exc
        if len(key_bytes) != 32:
            raise VMError("native pack cache key must be a SHA-256 digest")

        try:
            handle = path.open("rb")
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY)
            handle.close()
        except OSError as exc:
            raise VMError(f"cannot map native pack cache: {exc}") from exc
        try:
            if len(mapped) < _NATIVE_PACK_HEADER_SIZE:
                raise VMError("native pack cache is truncated")
            (
                magic,
                version,
                inst_size,
                block_size,
                host_size,
                inst_count,
                block_count,
                host_count,
                halt_code,
                stored_key,
            ) = _NATIVE_PACK_HEADER.unpack_from(mapped, 0)
            if magic != _NATIVE_PACK_MAGIC or version != _NATIVE_PACK_VERSION:
                raise VMError(
                    "native pack cache format mismatch: "
                    f"magic={magic!r} version={version}"
                )
            if (
                inst_size != ctypes.sizeof(CInst)
                or block_size != ctypes.sizeof(CBlock)
                or host_size != ctypes.sizeof(ctypes.c_uint64)
            ):
                raise VMError("native pack cache ABI layout mismatch")
            if stored_key != key_bytes:
                raise VMError("native pack cache fingerprint mismatch")
            if halt_code != (program.halt_code & MASK64):
                raise VMError("native pack cache halt code mismatch")

            inst_bytes = inst_count * inst_size
            block_bytes = block_count * block_size
            host_bytes = host_count * host_size
            expected = (
                _NATIVE_PACK_HEADER_SIZE
                + inst_bytes
                + block_bytes
                + host_bytes
            )
            if len(mapped) != expected:
                raise VMError(
                    "native pack cache size mismatch: "
                    f"{len(mapped)} != {expected}"
                )

            inst_offset = _NATIVE_PACK_HEADER_SIZE
            block_offset = inst_offset + inst_bytes
            host_offset = block_offset + block_bytes
            insts = (CInst * inst_count).from_buffer(mapped, inst_offset)
            blocks = (CBlock * block_count).from_buffer(mapped, block_offset)
            if host_count:
                hosts = (ctypes.c_uint64 * host_count).from_buffer(
                    mapped, host_offset
                )
            else:
                hosts = (ctypes.c_uint64 * 0)()

            current_hosts = tuple(sorted(program.host_code))
            cached_hosts = tuple(int(hosts[i]) for i in range(host_count))
            if cached_hosts != current_hosts:
                raise VMError("native pack cache host table mismatch")
        except Exception:
            mapped.close()
            raise

        self._packed_mmap = mapped
        self._packed = (insts, blocks, hosts)
        elapsed = time.perf_counter() - started
        print(
            "BOOT_EXEC_NATIVE_PACK_CACHE_LOADED "
            f"path={path} seconds={elapsed:.3f} bytes={len(mapped)} "
            f"insts={inst_count} blocks={block_count} hosts={host_count}",
            flush=True,
        )
        return insts, blocks, hosts

    def _append_cache_key(
        self,
        *,
        index: int,
        new_block_codes,
        current_hosts,
    ) -> str | None:
        context = getattr(self, "native_append_cache_context", None)
        if not context:
            return None
        digest = hashlib.sha256()
        digest.update((self._pack_cache_key or "-").encode("ascii"))
        digest.update(b"\0")
        digest.update(str(context).encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            struct.pack(
                "<QQQQ",
                index,
                int(self.program._next_data) & MASK64,
                len(new_block_codes),
                len(current_hosts),
            )
        )
        for code in new_block_codes:
            digest.update(struct.pack("<Q", int(code) & MASK64))
        for code in sorted(current_hosts):
            digest.update(struct.pack("<Q", int(code) & MASK64))
        return digest.hexdigest()

    @staticmethod
    def _append_cache_path(directory: Path, index: int) -> Path:
        return directory / f"append-{index:03d}.bin"

    def _load_append_packed_cache(
        self,
        path: Path,
        *,
        cache_key: str,
        current_hosts,
    ):
        started = time.perf_counter()
        key_bytes = bytes.fromhex(cache_key)
        if len(key_bytes) != 32:
            raise VMError("native append pack cache key must be SHA-256")
        try:
            handle = path.open("rb")
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY)
            handle.close()
        except OSError as exc:
            raise VMError(f"cannot map native append pack cache: {exc}") from exc
        try:
            if len(mapped) < _NATIVE_PACK_HEADER_SIZE:
                raise VMError("native append pack cache is truncated")
            (
                magic,
                version,
                inst_size,
                block_size,
                host_size,
                inst_count,
                block_count,
                host_count,
                halt_code,
                stored_key,
            ) = _NATIVE_PACK_HEADER.unpack_from(mapped, 0)
            if magic != _NATIVE_PACK_MAGIC or version != _NATIVE_PACK_VERSION:
                raise VMError("native append pack cache format mismatch")
            if (
                inst_size != ctypes.sizeof(CInst)
                or block_size != ctypes.sizeof(CBlock)
                or host_size != ctypes.sizeof(ctypes.c_uint64)
            ):
                raise VMError("native append pack cache ABI layout mismatch")
            if stored_key != key_bytes:
                raise VMError("native append pack cache fingerprint mismatch")
            if halt_code != (self.program.halt_code & MASK64):
                raise VMError("native append pack cache halt code mismatch")

            inst_bytes = inst_count * inst_size
            block_bytes = block_count * block_size
            host_bytes = host_count * host_size
            expected = (
                _NATIVE_PACK_HEADER_SIZE
                + inst_bytes
                + block_bytes
                + host_bytes
            )
            if len(mapped) != expected:
                raise VMError(
                    "native append pack cache size mismatch: "
                    f"{len(mapped)} != {expected}"
                )
            inst_offset = _NATIVE_PACK_HEADER_SIZE
            block_offset = inst_offset + inst_bytes
            host_offset = block_offset + block_bytes
            insts = (CInst * inst_count).from_buffer(mapped, inst_offset)
            blocks = (CBlock * block_count).from_buffer(mapped, block_offset)
            if host_count:
                hosts = (ctypes.c_uint64 * host_count).from_buffer(
                    mapped, host_offset
                )
            else:
                hosts = (ctypes.c_uint64 * 0)()
            cached_hosts = tuple(int(hosts[i]) for i in range(host_count))
            if cached_hosts != tuple(sorted(current_hosts)):
                raise VMError("native append pack cache host table mismatch")
        except Exception:
            mapped.close()
            raise

        self._append_pack_mmaps.append(mapped)
        elapsed = time.perf_counter() - started
        print(
            "BOOT_EXEC_NATIVE_APPEND_CACHE_LOADED "
            f"path={path} seconds={elapsed:.3f} bytes={len(mapped)} "
            f"insts={inst_count} blocks={block_count} hosts={host_count}",
            flush=True,
        )
        return insts, blocks, hosts

    @staticmethod
    def _shape(program):
        return (
            len(program.functions),
            len(program.block_code),
            len(program.host_code),
        )

    def _write_sparse_initial_items(self, items) -> tuple[int, int]:
        start = None
        previous = None
        data = bytearray()
        total = 0
        runs = 0

        def flush() -> None:
            nonlocal start, previous, data, runs
            if start is None or not data:
                return
            self.memory.bulk_write(start, data)
            runs += 1
            start = None
            previous = None
            data = bytearray()

        for address, value in items:
            address = int(address) & MASK64
            if (
                start is None
                or previous is None
                or address != ((previous + 1) & MASK64)
                or len(data) >= 4 * 1024 * 1024
            ):
                flush()
                start = address
            data.append(int(value) & 0xFF)
            previous = address
            total += 1
        flush()
        return total, runs

    def _load_initial_memory(self, program) -> None:
        started = time.perf_counter()
        total, runs = self._write_sparse_initial_items(
            program.initial_memory.bytes.items()
        )
        print(
            "BOOT_EXEC_NATIVE_INITIAL_MEMORY "
            f"seconds={time.perf_counter() - started:.3f} "
            f"bytes={total} runs={runs}",
            flush=True,
        )

    def _sync_appended_initial_memory(self) -> int:
        start = self._synced_data_end
        end = self.program._next_data
        if end <= start:
            return 0

        total, runs = self._write_sparse_initial_items(
            (address, value)
            for address, value in self.program.initial_memory.bytes.items()
            if start <= address < end
        )
        self._synced_data_end = end
        print(
            "BOOT_EXEC_NATIVE_DATA_APPEND "
            f"start=0x{start:x} end=0x{end:x} bytes={total} runs={runs}",
            flush=True,
        )
        return total

    def _trace_user_descriptor_after_sync(self, stage: str) -> None:
        symbol = "__mm_user_ext_getcwd"
        descriptor = self.program.symbol_addresses.get(symbol)
        if descriptor is None:
            return
        print(
            "BOOT_EXEC_NATIVE_USER_DESCRIPTOR "
            f"stage={stage} descriptor=0x{descriptor:x} "
            f"initial_entry=0x{self.program.initial_memory.read(descriptor, 64):x} "
            f"live_entry=0x{self.memory.read(descriptor, 64):x} "
            f"initial_frame={self.program.initial_memory.read(descriptor + 8, 64)} "
            f"live_frame={self.memory.read(descriptor + 8, 64)}",
            flush=True,
        )

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
        local_block_index,
    ):
        out = COperand()
        if target.is_direct():
            key = (function_name, target.label)
            local_index = local_block_index.get(key)
            if _native_local_targets_enabled() and local_index is not None:
                out.kind = MM_T_LOCAL_BLOCK
                out.value = local_index
                return out
            out.kind = MM_T_CODE
            out.value = program.block_code[key]
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

    def _pack_program(
        self,
        program,
        ordered_blocks=None,
        *,
        retain=True,
        log_label="BOOT_EXEC_NATIVE_PACK",
    ):
        started = time.perf_counter()
        host_by_symbol = {
            symbol: code for code, symbol in program.host_code.items()
        }
        if ordered_blocks is None:
            ordered_blocks = sorted(program.code_block.items())
        else:
            ordered_blocks = list(ordered_blocks)
        local_block_index = {
            pair: index
            for index, (_, pair) in enumerate(ordered_blocks)
        }
        total_insts = 0
        for _, (function_name, block_name) in ordered_blocks:
            linked = program.functions[function_name]
            total_insts += len(linked.block_map[block_name].instructions)

        # Allocate the final native-compatible buffers once.  The previous
        # implementation first built Python lists of ctypes objects and then
        # copied them into arrays, temporarily doubling an already-large Linux
        # P3 image.
        inst_array = (CInst * total_insts)()
        block_array = (CBlock * len(ordered_blocks))()
        inst_index = 0

        for block_index, (code, (function_name, block_name)) in enumerate(
            ordered_blocks
        ):
            linked = program.functions[function_name]
            block = linked.block_map[block_name]
            first = inst_index

            for inst in block.instructions:
                out = inst_array[inst_index]
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
                    out.dst = self._operand(inst.dst, linked, program)
                    out.src = self._operand(inst.src, linked, program)
                elif isinstance(inst, p3.Sub):
                    out.opcode = MM_OP_SUB
                    out.width = inst.width.value
                    out.dst = self._operand(inst.dst, linked, program)
                    out.a = self._operand(inst.a, linked, program)
                    out.b = self._operand(inst.b, linked, program)
                elif isinstance(inst, p3.Br):
                    out.opcode = MM_OP_BR
                    out.width = inst.width.value
                    out.cond = {
                        muir.Cond.EQ: MM_COND_EQ,
                        muir.Cond.ULT: MM_COND_ULT,
                        muir.Cond.SLT: MM_COND_SLT,
                    }[inst.cond]
                    out.a = self._operand(inst.a, linked, program)
                    out.b = self._operand(inst.b, linked, program)
                    out.t = self._target(
                        inst.true_target,
                        function_name,
                        linked,
                        program,
                        host_by_symbol,
                        local_block_index,
                    )
                    out.f = self._target(
                        inst.false_target,
                        function_name,
                        linked,
                        program,
                        host_by_symbol,
                        local_block_index,
                    )
                else:
                    raise VMError(
                        "non-P3 instruction reached native packer: "
                        f"{type(inst).__name__}"
                    )
                inst_index += 1

            block_array[block_index] = CBlock(
                code=code,
                first=first,
                count=len(block.instructions),
            )

        host_codes = sorted(program.host_code)
        host_array = (ctypes.c_uint64 * len(host_codes))(*host_codes)
        if retain:
            self._packed = (inst_array, block_array, host_array)
        elapsed = time.perf_counter() - started
        print(
            f"{log_label} "
            f"seconds={elapsed:.3f} insts={total_insts} "
            f"blocks={len(ordered_blocks)} hosts={len(host_codes)} "
            f"inst_bytes={ctypes.sizeof(CInst) * total_insts} "
            f"block_bytes={ctypes.sizeof(CBlock) * len(ordered_blocks)} "
            f"local_targets={int(_native_local_targets_enabled())}",
            flush=True,
        )
        return inst_array, block_array, host_array

    def _ensure_program_current(self):
        shape = self._shape(self.program)
        if shape == self._program_shape:
            return

        current_blocks = set(self.program.code_block)
        current_functions = set(self.program.functions)
        current_hosts = set(self.program.host_code)
        new_block_codes = sorted(current_blocks - self._packed_block_codes)
        new_functions = current_functions - self._packed_function_names
        new_hosts = sorted(current_hosts - self._packed_host_codes)
        host_append_only = self._packed_host_codes.issubset(current_hosts)

        if (
            current_blocks == self._packed_block_codes
            and current_functions == self._packed_function_names
            and host_append_only
            and new_hosts
        ):
            host_codes = sorted(current_hosts)
            host_array = (ctypes.c_uint64 * len(host_codes))(*host_codes)
            if not self._lib.mm_vm_set_host_codes(
                self._handle,
                host_array,
                len(host_array),
            ):
                raise VMError("cannot append native P3 host service table")
            self._host_packed = host_array
            self._install_native_host_intrinsics()
            self._sync_appended_initial_memory()
            self._trace_user_descriptor_after_sync("host-append")
            self._packed_host_codes = current_hosts
            self._program_shape = shape
            print(
                "BOOT_EXEC_NATIVE_HOST_APPEND "
                f"hosts={len(new_hosts)} total={len(current_hosts)}",
                flush=True,
            )
            return

        append_only = (
            self._packed_block_codes.issubset(current_blocks)
            and self._packed_function_names.issubset(current_functions)
            and host_append_only
            and bool(new_block_codes)
            and bool(new_functions)
        )

        if append_only:
            ordered_blocks = [
                (code, self.program.code_block[code])
                for code in new_block_codes
            ]
            append_index = self._append_pack_index
            append_key = self._append_cache_key(
                index=append_index,
                new_block_codes=new_block_codes,
                current_hosts=current_hosts,
            )
            append_cache_path = (
                self._append_cache_path(
                    self._append_pack_cache_in_dir,
                    append_index,
                )
                if self._append_pack_cache_in_dir is not None
                else None
            )
            if (
                append_cache_path is not None
                and append_cache_path.is_file()
                and append_key is not None
            ):
                insts, blocks, hosts = self._load_append_packed_cache(
                    append_cache_path,
                    cache_key=append_key,
                    current_hosts=current_hosts,
                )
            else:
                insts, blocks, hosts = self._pack_program(
                    self.program,
                    ordered_blocks,
                    retain=False,
                    log_label="BOOT_EXEC_NATIVE_APPEND_PACK",
                )
                if (
                    self._append_pack_cache_out_dir is not None
                    and append_key is not None
                ):
                    out_path = self._append_cache_path(
                        self._append_pack_cache_out_dir,
                        append_index,
                    )
                    self._save_packed_cache(
                        self.program,
                        out_path,
                        insts,
                        blocks,
                        hosts,
                        cache_key=append_key,
                        log_label="BOOT_EXEC_NATIVE_APPEND_CACHE_SAVED",
                    )
            self._append_pack_index += 1
            if not self._lib.mm_vm_add_segment(
                self._handle,
                insts,
                len(insts),
                blocks,
                len(blocks),
            ):
                raise VMError("cannot append native P3 program segment")
            self._extra_packed.append((insts, blocks))
            if new_hosts:
                if not self._lib.mm_vm_set_host_codes(
                    self._handle,
                    hosts,
                    len(hosts),
                ):
                    raise VMError("cannot append native P3 host service table")
                self._host_packed = hosts
                self._install_native_host_intrinsics()
            self._sync_appended_initial_memory()
            self._trace_user_descriptor_after_sync("segment-append")

            self._packed_block_codes = current_blocks
            self._packed_function_names = current_functions
            self._packed_host_codes = current_hosts
            self._program_shape = shape
            print(
                "BOOT_EXEC_NATIVE_APPEND "
                f"functions={len(new_functions)} "
                f"blocks={len(new_block_codes)} "
                f"hosts={len(new_hosts)}",
                flush=True,
            )
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
        self._extra_packed.clear()
        self._host_packed = None
        self._install_native_host_intrinsics()
        self._sync_appended_initial_memory()
        self._trace_user_descriptor_after_sync("program-replace")
        self._packed_block_codes = current_blocks
        self._packed_function_names = current_functions
        self._packed_host_codes = current_hosts
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

    def host_profile_summary(self, *, limit: int = 24) -> tuple[str, ...]:
        lines: list[str] = []

        if getattr(self, "_profile_host", False):
            native_ms = self._native_profile_c_ns / 1_000_000
            lines.append(
                "BOOT_EXEC_NATIVE_PROFILE "
                f"run_calls={self._native_profile_run_calls} "
                f"host_returns={self._native_profile_host_returns} "
                f"watch_returns={self._native_profile_watch_returns} "
                f"c_run_ms={native_ms:.3f}"
            )
            lines.extend(super().host_profile_summary())

        if self._host_profile_enabled:
            rows = sorted(
                self._host_profile_seconds,
                key=lambda name: (
                    -self._host_profile_seconds[name],
                    -self._host_profile_counts.get(name, 0),
                    name,
                ),
            )
            lines.append(
                "BOOT_EXEC_NATIVE_HOST_PROFILE "
                f"calls={self._host_profile_total_calls} "
                f"seconds={self._host_profile_total_seconds:.6f} "
                f"symbols={len(rows)}"
            )
            for name in rows[:max(0, limit)]:
                count = self._host_profile_counts.get(name, 0)
                seconds = self._host_profile_seconds.get(name, 0.0)
                lines.append(
                    "BOOT_EXEC_NATIVE_HOST_PROFILE_SYMBOL "
                    f"name={name} calls={count} seconds={seconds:.6f} "
                    f"avg_us={(seconds / count * 1_000_000) if count else 0.0:.3f}"
                )

        return tuple(lines)

    def run(self, *, max_steps: int = 1_000_000) -> None:
        native_limit = MASK64 if max_steps <= 0 else max_steps
        report_every = int(getattr(self, "native_report_every", 0) or 0)
        report_started = time.perf_counter()
        report_start_steps = self.steps

        while not self.halted:
            if self.steps >= native_limit:
                raise VMError(f"step limit exceeded: {native_limit}")

            self._ensure_program_current()
            self._sync_native_heap_state()
            code = self._current_code()
            self._lib.mm_vm_set_state(
                self._handle,
                code,
                self.ip,
                self.sp & MASK64,
                self.steps,
            )
            batch_limit = native_limit
            is_report_boundary = False
            trace_remaining = int(
                getattr(self, "_single_step_trace_remaining", 0) or 0
            )
            if trace_remaining > 0:
                batch_limit = min(native_limit, self.steps + 1)
            if report_every > 0:
                batch_limit = min(
                    native_limit,
                    ((self.steps // report_every) + 1) * report_every,
                )
                is_report_boundary = batch_limit < native_limit

            if getattr(self, "_profile_host", False):
                native_started = time.perf_counter_ns()
                result = self._lib.mm_vm_run(self._handle, batch_limit)
                self._native_profile_c_ns += time.perf_counter_ns() - native_started
                self._native_profile_run_calls += 1
                if result.status == MM_STATUS_HOST:
                    self._native_profile_host_returns += 1
                elif result.status == MM_STATUS_WATCH:
                    self._native_profile_watch_returns += 1
            else:
                result = self._lib.mm_vm_run(self._handle, batch_limit)
            self.sp = int(result.sp)
            self.steps = int(result.steps)
            self._sync_python_heap_state()
            self._sync_native_frees()
            self._sync_block(int(result.block_code), int(result.ip))

            if result.status == MM_STATUS_LIMIT:
                trace_remaining = int(
                    getattr(self, "_single_step_trace_remaining", 0) or 0
                )
                if trace_remaining > 0 and self.steps == batch_limit:
                    descriptor = self.program.symbol_addresses.get(
                        "__mm_user_ext_getcwd"
                    )
                    desc_entry = (
                        self.memory.read(descriptor, 64)
                        if descriptor is not None
                        else 0
                    )
                    frame16 = self.memory.read(self.sp + 16, 64)
                    frame24 = self.memory.read(self.sp + 24, 64)
                    print(
                        "BOOT_EXEC_NATIVE_SINGLE_STEP "
                        f"remaining={trace_remaining} steps={self.steps} "
                        f"function={self.current_function} "
                        f"block={self.current_block} ip={self.ip} "
                        f"sp=0x{self.sp:x} "
                        f"descriptor=0x{descriptor or 0:x} "
                        f"desc_entry=0x{desc_entry:x} "
                        f"sp16=0x{frame16:x} sp24=0x{frame24:x}",
                        flush=True,
                    )
                    self._single_step_trace_remaining = trace_remaining - 1
                    continue
                if is_report_boundary and self.steps == batch_limit:
                    now = time.perf_counter()
                    delta_steps = self.steps - report_start_steps
                    delta_s = now - report_started
                    msteps = (
                        delta_steps / delta_s / 1_000_000
                        if delta_s > 0
                        else 0.0
                    )
                    caller_sp = self.memory.read(
                        self.sp + CALLER_SP, 64
                    )
                    ret_pc = self.memory.read(self.sp + RET_PC, 64)
                    caller_pair = self.program.code_block.get(ret_pc)
                    caller = (
                        f"{caller_pair[0]}:{caller_pair[1]}"
                        if caller_pair is not None
                        else "<host-or-root>"
                    )
                    slot_parts = []
                    linked = (
                        self.program.functions.get(self.current_function)
                        if self.current_function is not None
                        else None
                    )
                    if linked is not None:
                        for slot_name in self.native_report_slots:
                            offset = linked.slot_offsets.get(slot_name)
                            if offset is None:
                                continue
                            value = self.memory.read(self.sp + offset, 64)
                            slot_parts.append(
                                f"{slot_name}=0x{value:x}"
                            )
                    slot_text = (
                        " slots=" + ",".join(slot_parts)
                        if slot_parts
                        else ""
                    )
                    print(
                        "BOOT_EXEC_NATIVE_PROGRESS "
                        f"steps={self.steps} "
                        f"function={self.current_function} "
                        f"block={self.current_block} ip={self.ip} "
                        f"sp=0x{self.sp:x} caller_sp=0x{caller_sp:x} "
                        f"ret_pc=0x{ret_pc:x} caller={caller}"
                        f"{slot_text} "
                        f"chunk_steps={delta_steps} "
                        f"chunk_s={delta_s:.3f} "
                        f"msteps_s={msteps:.3f}",
                        flush=True,
                    )
                    report_started = now
                    report_start_steps = self.steps
                    continue
                raise VMError(f"step limit exceeded: {native_limit}")
            if result.status == MM_STATUS_HALT:
                self.halted = True
                return
            if result.status == MM_STATUS_HOST:
                target_code = int(result.target_code)
                if self._host_profile_enabled:
                    symbol = self.program.host_code.get(
                        target_code,
                        f"<0x{target_code:x}>",
                    )
                    started = time.perf_counter()
                    self._set_code(target_code)
                    elapsed = time.perf_counter() - started
                    self._host_profile_counts[symbol] = (
                        self._host_profile_counts.get(symbol, 0) + 1
                    )
                    self._host_profile_seconds[symbol] = (
                        self._host_profile_seconds.get(symbol, 0.0) + elapsed
                    )
                    self._host_profile_total_calls += 1
                    self._host_profile_total_seconds += elapsed
                else:
                    self._set_code(target_code)
                continue
            if result.status == MM_STATUS_WATCH:
                target_code = int(result.target_code)
                self._set_code(target_code)
                trace_watch = int(
                    getattr(self, "trace_single_step_watch_code", 0) or 0
                )
                if trace_watch and target_code == trace_watch:
                    self._single_step_trace_remaining = int(
                        getattr(self, "trace_single_step_count", 18) or 18
                    )
                    descriptor = self.program.symbol_addresses.get(
                        "__mm_user_ext_getcwd"
                    )
                    print(
                        "BOOT_EXEC_NATIVE_SINGLE_STEP_ARMED "
                        f"code=0x{target_code:x} "
                        f"function={self.current_function} "
                        f"block={self.current_block} "
                        f"sp=0x{self.sp:x} "
                        f"descriptor=0x{descriptor or 0:x} "
                        f"desc_entry=0x{self.memory.read(descriptor, 64) if descriptor is not None else 0:x}",
                        flush=True,
                    )
                    self.set_watch_codes(
                        code for code in self._watch_codes
                        if code != trace_watch
                    )
                continue
            if result.status == MM_STATUS_ERROR:
                raise VMError(
                    "native P3 execution error: "
                    f"code={result.error} "
                    f"target=0x{int(result.target_code):x}"
                )
            raise VMError(f"unknown native P3 run status: {result.status}")

