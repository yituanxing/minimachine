from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.minimachine import muir
from src.minimachine.abi import expand_function
from src.minimachine.lower_p3 import lower_function
from src.minimachine.user_image import UserProgramImage, pack_user_program, unpack_user_image
from src.minimachine.user_image_cache import (
    UserImageCacheError,
    load_user_image_cache,
    save_user_image_cache,
)


class UserImageCacheTests(unittest.TestCase):
    def _payload(self):
        fn = muir.Function(
            "main",
            [muir.Block("entry", [muir.Ret(None)])],
            set(),
        )
        expanded, _ = expand_function(fn)
        image = UserProgramImage("main", (lower_function(expanded),))
        payload = pack_user_program(image)
        return payload, unpack_user_image(payload)

    def test_round_trip_is_payload_bound(self):
        payload, image = self._payload()
        digest = hashlib.sha256(payload).hexdigest()
        with TemporaryDirectory() as tmp:
            for name in ("user.pkl", "user.pkl.gz"):
                path = Path(tmp) / name
                save_user_image_cache(image, path, payload_sha256=digest)
                loaded = load_user_image_cache(path, payload_sha256=digest)
                self.assertEqual(loaded.entry, image.entry)
                self.assertEqual(len(loaded.functions), len(image.functions))
                with self.assertRaises(UserImageCacheError):
                    load_user_image_cache(path, payload_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
