from __future__ import annotations

import gzip
import hashlib
import pickle
from pathlib import Path

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
) -> None:
    payload = {
        "version": USER_IMAGE_CACHE_VERSION,
        "schema_sha256": user_image_schema_fingerprint(),
        "payload_sha256": payload_sha256,
        "image": image,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=3) as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_user_image_cache(
    path: Path,
    *,
    payload_sha256: str,
) -> UserProgramImage:
    try:
        with gzip.open(path, "rb") as handle:
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
    image = payload.get("image")
    if not isinstance(image, UserProgramImage):
        raise UserImageCacheError("userspace image cache payload has wrong type")
    return image
