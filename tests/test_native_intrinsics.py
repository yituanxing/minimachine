from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.minimachine.native_vm import (
    MM_INTR_EXPECT,
    MM_INTR_NONE,
    MM_INTR_PTR_ADD_SCALED,
    NativeVM,
)


class NativeIntrinsicMappingTests(unittest.TestCase):
    def test_simple_intrinsics_are_opt_in_for_ab(self):
        with patch.dict(
            os.environ,
            {"MINIMACHINE_NATIVE_SIMPLE_INTRINSICS": "0"},
            clear=False,
        ):
            self.assertEqual(
                NativeVM._native_intrinsic_for_symbol(
                    "__mm_llvm_expect_i64"
                ).op,
                MM_INTR_NONE,
            )
            self.assertEqual(
                NativeVM._native_intrinsic_for_symbol(
                    "__mm_ptr_add_scaled_8"
                ).op,
                MM_INTR_NONE,
            )

    def test_expect_and_scaled_pointer_map_to_native_descriptors(self):
        with patch.dict(
            os.environ,
            {"MINIMACHINE_NATIVE_SIMPLE_INTRINSICS": "1"},
            clear=False,
        ):
            expect = NativeVM._native_intrinsic_for_symbol(
                "__mm_llvm_expect_i64"
            )
            self.assertEqual(expect.op, MM_INTR_EXPECT)

            pointer = NativeVM._native_intrinsic_for_symbol(
                "__mm_ptr_add_scaled_1264"
            )
            self.assertEqual(pointer.op, MM_INTR_PTR_ADD_SCALED)
            self.assertEqual(pointer.imm, 1264)


if __name__ == "__main__":
    unittest.main()
