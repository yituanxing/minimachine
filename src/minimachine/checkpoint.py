from __future__ import annotations

import gzip
import hashlib
import pickle
from pathlib import Path

from .vm import VM


CHECKPOINT_VERSION = 3
_LINUX_ATTRS = (
    "linux_task_contexts",
    "linux_task_shadow_stacks",
    "linux_shadow_stack_next",
    "linux_task_semantic_stacks",
    "linux_semantic_stack_next",
    "linux_task_sched_class_offset",
)


class CheckpointError(RuntimeError):
    pass


def image_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _snapshot(
    vm: VM,
    *,
    image_sha256: str,
    initramfs_sha256: str | None = None,
) -> dict:
    linux_state = {
        name: getattr(vm, name)
        for name in _LINUX_ATTRS
        if hasattr(vm, name)
    }
    if hasattr(vm.memory, "snapshot_pages"):
        memory_payload = {
            "memory_format": "native-pages",
            "memory_pages": vm.memory.snapshot_pages(),
        }
    else:
        baseline = vm.program.initial_memory.bytes
        memory_delta = {
            address: value
            for address, value in vm.memory.bytes.items()
            if baseline.get(address, 0) != value
        }
        memory_payload = {
            "memory_format": "byte-delta",
            "memory_delta": memory_delta,
        }

    return {
        "version": CHECKPOINT_VERSION,
        "image_sha256": image_sha256,
        "initramfs_sha256": initramfs_sha256,
        "stack_top": vm.stack_top,
        **memory_payload,
        "sp": vm.sp,
        "current_function": vm.current_function,
        "current_block": vm.current_block,
        "ip": vm.ip,
        "halted": vm.halted,
        "steps": vm.steps,
        "system_state": vm.system_state,
        "csr": vm.csr,
        "static_keys": vm.static_keys,
        "cpu_features": vm.cpu_features,
        "vector_state": vm.vector_state,
        "heap_next": vm.heap_next,
        "linux_state": linux_state,
    }


def save_checkpoint(
    vm: VM,
    path: Path,
    *,
    image_sha256: str,
    initramfs_sha256: str | None = None,
) -> None:
    payload = _snapshot(
        vm,
        image_sha256=image_sha256,
        initramfs_sha256=initramfs_sha256,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=3) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(
    vm: VM,
    path: Path,
    *,
    image_sha256: str,
    initramfs_sha256: str | None = None,
) -> None:
    try:
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError) as exc:
        raise CheckpointError(f"cannot read VM checkpoint: {exc}") from exc

    version = payload.get("version")
    if version not in {1, 2, CHECKPOINT_VERSION}:
        raise CheckpointError(
            "checkpoint version mismatch: "
            f"{version} not in {{1,2,{CHECKPOINT_VERSION}}}"
        )
    if payload.get("image_sha256") != image_sha256:
        raise CheckpointError("checkpoint linked-image fingerprint mismatch")
    checkpoint_initramfs = payload.get("initramfs_sha256")
    if (
        checkpoint_initramfs is not None
        and initramfs_sha256 is not None
        and checkpoint_initramfs != initramfs_sha256
    ):
        raise CheckpointError("checkpoint initramfs fingerprint mismatch")
    if payload.get("stack_top") != vm.stack_top:
        raise CheckpointError("checkpoint stack layout mismatch")

    if version == 1:
        # Backward compatibility with the first full-memory checkpoint format.
        if hasattr(vm.memory, "restore_pages"):
            raise CheckpointError(
                "legacy full-memory checkpoint cannot restore into native VM"
            )
        vm.memory.bytes = dict(payload["memory"])
    elif version == 2:
        if hasattr(vm.memory, "restore_pages"):
            # Old checkpoints store only byte deltas.  Rebuild the native
            # memory image from the program baseline plus that delta.
            merged = dict(vm.program.initial_memory.bytes)
            merged.update(payload["memory_delta"])
            page_size = 65536
            page_map: dict[int, bytearray] = {}
            for address, value in merged.items():
                page_no = address // page_size
                page = page_map.setdefault(page_no, bytearray(page_size))
                page[address % page_size] = value
            page_nos = tuple(sorted(page_map))
            vm.memory.restore_pages(
                {
                    "page_size": page_size,
                    "page_nos": page_nos,
                    "data": b"".join(bytes(page_map[n]) for n in page_nos),
                }
            )
        else:
            vm.memory.bytes = dict(vm.program.initial_memory.bytes)
            vm.memory.bytes.update(payload["memory_delta"])
    else:
        memory_format = payload.get("memory_format")
        if memory_format == "native-pages":
            if hasattr(vm.memory, "restore_pages"):
                vm.memory.restore_pages(payload["memory_pages"])
            else:
                snapshot = payload["memory_pages"]
                page_size = int(snapshot["page_size"])
                out: dict[int, int] = {}
                data = bytes(snapshot["data"])
                for index, page_no in enumerate(snapshot["page_nos"]):
                    base = int(page_no) * page_size
                    chunk = data[index * page_size:(index + 1) * page_size]
                    for offset, value in enumerate(chunk):
                        if value:
                            out[base + offset] = value
                vm.memory.bytes = out
        elif memory_format == "byte-delta":
            if hasattr(vm.memory, "restore_pages"):
                merged = dict(vm.program.initial_memory.bytes)
                merged.update(payload["memory_delta"])
                page_size = 65536
                page_map: dict[int, bytearray] = {}
                for address, value in merged.items():
                    page_no = address // page_size
                    page = page_map.setdefault(page_no, bytearray(page_size))
                    page[address % page_size] = value
                page_nos = tuple(sorted(page_map))
                vm.memory.restore_pages(
                    {
                        "page_size": page_size,
                        "page_nos": page_nos,
                        "data": b"".join(
                            bytes(page_map[n]) for n in page_nos
                        ),
                    }
                )
            else:
                vm.memory.bytes = dict(vm.program.initial_memory.bytes)
                vm.memory.bytes.update(payload["memory_delta"])
        else:
            raise CheckpointError(
                f"unknown checkpoint memory format: {memory_format!r}"
            )
    vm.sp = payload["sp"]
    vm.current_function = payload["current_function"]
    vm.current_block = payload["current_block"]
    vm.ip = payload["ip"]
    vm.halted = payload["halted"]
    vm.steps = payload["steps"]
    vm.system_state = dict(payload["system_state"])
    vm.csr = dict(payload["csr"])
    vm.static_keys = dict(payload["static_keys"])
    vm.cpu_features = set(payload["cpu_features"])
    vm.vector_state = tuple(payload["vector_state"])
    vm.heap_next = payload["heap_next"]

    for name in _LINUX_ATTRS:
        if hasattr(vm, name):
            delattr(vm, name)
    for name, value in payload.get("linux_state", {}).items():
        setattr(vm, name, value)
