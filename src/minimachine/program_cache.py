from __future__ import annotations

import gzip
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

from .image import ModuleImage
from .runtime import RuntimeSurface
from .vm import Program


PROGRAM_CACHE_VERSION = 2

_LOWERING_FINGERPRINT_FILES = (
    "llvm_text.py",
    "layout.py",
    "legalize.py",
    "abi.py",
    "lower_p3.py",
    "muir.py",
    "p3.py",
    "verify.py",
    "image.py",
)


def lowering_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _LOWERING_FINGERPRINT_FILES:
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class ProgramCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProgramCache:
    image_sha256: str
    program: Program
    surface: RuntimeSurface
    reasons: frozenset[str]
    blocked_functions: tuple[tuple[str, int], ...]
    image: ModuleImage
    task_sched_class_offset: int | None

    @property
    def function_count(self) -> int:
        return len(self.program.functions)


def save_program_cache(cache: ProgramCache, path: Path) -> None:
    payload = {
        "version": PROGRAM_CACHE_VERSION,
        "lowering_sha256": lowering_fingerprint(),
        "cache": cache,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=3) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_program_cache(
    path: Path,
    *,
    image_sha256: str,
) -> ProgramCache:
    try:
        with gzip.open(path, "rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError) as exc:
        raise ProgramCacheError(f"cannot read P3 program cache: {exc}") from exc

    if payload.get("version") != PROGRAM_CACHE_VERSION:
        raise ProgramCacheError(
            "P3 program cache version mismatch: "
            f"{payload.get('version')} != {PROGRAM_CACHE_VERSION}"
        )
    actual_lowering = payload.get("lowering_sha256")
    expected_lowering = lowering_fingerprint()
    if actual_lowering != expected_lowering:
        raise ProgramCacheError(
            "P3 program cache lowering fingerprint mismatch: "
            f"{actual_lowering} != {expected_lowering}"
        )
    cache = payload.get("cache")
    if not isinstance(cache, ProgramCache):
        raise ProgramCacheError("P3 program cache payload has wrong type")
    if cache.image_sha256 != image_sha256:
        raise ProgramCacheError("P3 program cache linked-image fingerprint mismatch")
    return cache
