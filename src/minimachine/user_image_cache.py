from __future__ import annotations

import gzip
import hashlib
import pickle
from pathlib import Path

from . import p3
from .user_image import UserProgramImage


USER_IMAGE_CACHE_VERSION = 1
_SCHEMA_FILES = (
    "user_image.py",
    "user_bundle.py",
    "p3.py",
    "muir.py",
    "image.py",
)


class UserImageCacheError(RuntimeError):
    pass


def _open_cache(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, compresslevel=3)
    return path.open(mode)


def slim_user_image_metadata(image: UserProgramImage) -> UserProgramImage:
    """Drop P3 instruction bodies while preserving native replay link metadata."""
    functions = tuple(
        p3.Function(
            name=function.name,
            blocks=[
                p3.Block(block.label, [])
                for block in function.blocks
            ],
            frame_slots=set(function.frame_slots),
        )
        for function in image.functions
    )
    return UserProgramImage(
        entry=image.entry,
        functions=functions,
        image=image.image,
        entry_args=image.entry_args,
        runtime_helpers=image.runtime_helpers,
    )


def user_image_schema_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _SCHEMA_FILES:
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def save_user_image_cache(
    image: UserProgramImage,
    path: Path,
    *,
    payload_sha256: str,
    slim_metadata: bool = False,
) -> None:
    if slim_metadata:
        image = slim_user_image_metadata(image)
    payload = {
        "version": USER_IMAGE_CACHE_VERSION,
        "schema_sha256": user_image_schema_fingerprint(),
        "payload_sha256": payload_sha256,
        "slim_metadata": bool(slim_metadata),
        "image": image,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_cache(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_user_image_cache(
    path: Path,
    *,
    payload_sha256: str,
    require_slim_metadata: bool | None = None,
) -> UserProgramImage:
    try:
        with _open_cache(path, "rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError) as exc:
        raise UserImageCacheError(
            f"cannot read userspace image cache: {exc}"
        ) from exc
    if payload.get("version") != USER_IMAGE_CACHE_VERSION:
        raise UserImageCacheError(
            "userspace image cache version mismatch: "
            f"{payload.get('version')} != {USER_IMAGE_CACHE_VERSION}"
        )
    actual_schema = payload.get("schema_sha256")
    expected_schema = user_image_schema_fingerprint()
    if actual_schema != expected_schema:
        raise UserImageCacheError(
            "userspace image cache schema fingerprint mismatch: "
            f"{actual_schema} != {expected_schema}"
        )
    if payload.get("payload_sha256") != payload_sha256:
        raise UserImageCacheError(
            "userspace image cache payload fingerprint mismatch"
        )
    stored_slim = bool(payload.get("slim_metadata", False))
    if (
        require_slim_metadata is not None
        and stored_slim != bool(require_slim_metadata)
    ):
        raise UserImageCacheError(
            "userspace image cache metadata mode mismatch: "
            f"stored={int(stored_slim)} "
            f"required={int(bool(require_slim_metadata))}"
        )
    image = payload.get("image")
    if not isinstance(image, UserProgramImage):
        raise UserImageCacheError("userspace image cache payload has wrong type")
    return image
