from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path

from .image import ModuleImage
from .runtime import RuntimeSurface
from .vm import Program


PROGRAM_CACHE_VERSION = 1


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
    cache = payload.get("cache")
    if not isinstance(cache, ProgramCache):
        raise ProgramCacheError("P3 program cache payload has wrong type")
    if cache.image_sha256 != image_sha256:
        raise ProgramCacheError("P3 program cache linked-image fingerprint mismatch")
    return cache
