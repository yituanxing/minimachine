from __future__ import annotations

import gzip
import hashlib
import pickle
from pathlib import Path

from . import p3
from .program_cache import ProgramCache, lowering_fingerprint
from .vm import LinkedFunction, Program, SparseMemory


NATIVE_HOT_CACHE_VERSION = 1


class NativeHotCacheError(RuntimeError):
    pass


def _open_cache(path: Path, mode: str):
    if path.suffix == ".gz":
        if "w" in mode:
            return gzip.open(path, mode, compresslevel=3)
        return gzip.open(path, mode)
    return path.open(mode)


def native_hot_cache_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    digest.update(lowering_fingerprint().encode("ascii"))
    digest.update(b"\0")
    for name in ("native_hot_cache.py", "vm.py"):
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _slim_program(program: Program) -> Program:
    """Keep runtime/link metadata while dropping base P3 instruction bodies.

    Native hot replay executes the frozen base image from the mmap-backed
    packed cache.  Python still needs function entry/frame metadata, global
    code maps, symbols, and the allocation cursors so host callbacks and
    dynamically appended userspace code preserve the exact machine ABI.
    """
    slim = Program()
    slim.functions = {}

    for name, linked in program.functions.items():
        if not linked.function.blocks:
            raise NativeHotCacheError(
                f"cannot slim function without entry block: {name}"
            )
        entry_label = linked.function.blocks[0].label
        entry_block = p3.Block(entry_label, [])
        stub = p3.Function(
            name=name,
            blocks=[entry_block],
            frame_slots=set(),
        )
        slim.functions[name] = LinkedFunction(
            function=stub,
            slot_offsets=dict(linked.slot_offsets),
            frame_size=linked.frame_size,
            block_map={entry_label: entry_block},
        )

    slim.block_code = dict(program.block_code)
    slim.code_block = dict(program.code_block)
    slim.host_code = dict(program.host_code)
    slim.host_services = {}
    slim.symbol_addresses = dict(program.symbol_addresses)
    slim.initial_memory = SparseMemory(program.initial_memory.bytes)
    slim._next_code = program._next_code
    slim._next_data = program._next_data
    slim.halt_code = program.halt_code
    return slim


def slim_program_cache(cache: ProgramCache) -> ProgramCache:
    return ProgramCache(
        image_sha256=cache.image_sha256,
        program=_slim_program(cache.program),
        surface=cache.surface,
        reasons=cache.reasons,
        blocked_functions=cache.blocked_functions,
        image=cache.image,
        task_sched_class_offset=cache.task_sched_class_offset,
    )


def save_native_hot_cache(cache: ProgramCache, path: Path) -> None:
    payload = {
        "version": NATIVE_HOT_CACHE_VERSION,
        "schema_sha256": native_hot_cache_fingerprint(),
        "cache": slim_program_cache(cache),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_cache(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_native_hot_cache(
    path: Path,
    *,
    image_sha256: str,
) -> ProgramCache:
    try:
        with _open_cache(path, "rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError) as exc:
        raise NativeHotCacheError(
            f"cannot read native hot metadata cache: {exc}"
        ) from exc

    if payload.get("version") != NATIVE_HOT_CACHE_VERSION:
        raise NativeHotCacheError(
            "native hot metadata cache version mismatch: "
            f"{payload.get('version')} != {NATIVE_HOT_CACHE_VERSION}"
        )
    actual_schema = payload.get("schema_sha256")
    expected_schema = native_hot_cache_fingerprint()
    if actual_schema != expected_schema:
        raise NativeHotCacheError(
            "native hot metadata cache schema fingerprint mismatch: "
            f"{actual_schema} != {expected_schema}"
        )
    cache = payload.get("cache")
    if not isinstance(cache, ProgramCache):
        raise NativeHotCacheError(
            "native hot metadata cache payload has wrong type"
        )
    if cache.image_sha256 != image_sha256:
        raise NativeHotCacheError(
            "native hot metadata cache linked-image fingerprint mismatch"
        )
    return cache
