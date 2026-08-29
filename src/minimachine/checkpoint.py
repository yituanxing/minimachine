from __future__ import annotations

import gzip
import hashlib
import pickle
from pathlib import Path

from .vm import VM


CHECKPOINT_VERSION = 2
_LINUX_ATTRS = (
    "linux_task_contexts",
    "linux_task_shadow_stacks",
    "linux_shadow_stack_next",
    "linux_task_sched_class_offset",
)


class CheckpointError(RuntimeError):
    pass


def image_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _snapshot(vm: VM, *, image_sha256: str) -> dict:
    linux_state = {
        name: getattr(vm, name)
        for name in _LINUX_ATTRS
        if hasattr(vm, name)
    }
    baseline = vm.program.initial_memory.bytes
    memory_delta = {
        address: value
        for address, value in vm.memory.bytes.items()
        if baseline.get(address, 0) != value
    }
    return {
        "version": CHECKPOINT_VERSION,
        "image_sha256": image_sha256,
        "stack_top": vm.stack_top,
        "memory_delta": memory_delta,
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
) -> None:
    payload = _snapshot(vm, image_sha256=image_sha256)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=3) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(
    vm: VM,
    path: Path,
    *,
    image_sha256: str,
) -> None:
    try:
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError) as exc:
        raise CheckpointError(f"cannot read VM checkpoint: {exc}") from exc

    version = payload.get("version")
    if version not in {1, CHECKPOINT_VERSION}:
        raise CheckpointError(
            "checkpoint version mismatch: "
            f"{version} not in {{1,{CHECKPOINT_VERSION}}}"
        )
    if payload.get("image_sha256") != image_sha256:
        raise CheckpointError("checkpoint linked-image fingerprint mismatch")
    if payload.get("stack_top") != vm.stack_top:
        raise CheckpointError("checkpoint stack layout mismatch")

    if version == 1:
        # Backward compatibility with the first full-memory checkpoint format.
        vm.memory.bytes = dict(payload["memory"])
    else:
        vm.memory.bytes = dict(vm.program.initial_memory.bytes)
        vm.memory.bytes.update(payload["memory_delta"])
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
