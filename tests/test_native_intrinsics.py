from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.minimachine.native_vm import (
    MM_INTR_EXPECT,
    MM_INTR_NONE,
    MM_INTR_PTR_ADD_SCALED,
    MM_INTR_ROR32,
    MM_INTR_LOAD_I128,
    MM_INTR_STORE_I128,
    MM_INTR_IS_CONSTANT,
    MM_INTR_PREFETCH,
    MM_INTR_UDIV,
    MM_INTR_SDIV,
    MM_INTR_UREM,
    MM_INTR_SREM,
    MM_INTR_WIDE_CONST_I128,
    MM_INTR_ICMP_EQ_I128,
    MM_INTR_MEMSET,
    MM_INTR_MEMCPY,
    MM_INTR_FAST_MEMCPY,
    MM_INTR_FAST_MEMSET,
    MM_INTR_FAST_STRLEN,
    NativeVM,
)


class NativeIntrinsicMappingTests(unittest.TestCase):
    def test_simple_intrinsics_can_be_disabled_for_ab(self):
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

    def test_ror32_intrinsic_is_enabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                NativeVM._native_intrinsic_for_symbol(
                    "__mm_fast_ror32"
                ).op,
                MM_INTR_ROR32,
            )

    def test_ror32_intrinsic_mapping_is_independent(self):
        with patch.dict(
            os.environ,
            {
                "MINIMACHINE_NATIVE_ROR32_INTRINSIC": "0",
                "MINIMACHINE_NATIVE_SIMPLE_INTRINSICS": "1",
            },
            clear=False,
        ):
            self.assertEqual(
                NativeVM._native_intrinsic_for_symbol(
                    "__mm_fast_ror32"
                ).op,
                MM_INTR_NONE,
            )

        with patch.dict(
            os.environ,
            {
                "MINIMACHINE_NATIVE_ROR32_INTRINSIC": "1",
                "MINIMACHINE_NATIVE_SIMPLE_INTRINSICS": "1",
            },
            clear=False,
        ):
            self.assertEqual(
                NativeVM._native_intrinsic_for_symbol(
                    "__mm_fast_ror32"
                ).op,
                MM_INTR_ROR32,
            )

    def test_i128_intrinsics_are_enabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                NativeVM._native_intrinsic_for_symbol(
                    "__mm_load_i128"
                ).op,
                MM_INTR_LOAD_I128,
            )
            self.assertEqual(
                NativeVM._native_intrinsic_for_symbol(
                    "__mm_store_i128"
                ).op,
                MM_INTR_STORE_I128,
            )

    def test_i128_intrinsic_mapping_can_be_disabled_for_ab(self):
        with patch.dict(
            os.environ,
            {"MINIMACHINE_NATIVE_I128_INTRINSICS": "0"},
            clear=False,
        ):
            self.assertEqual(
                NativeVM._native_intrinsic_for_symbol(
                    "__mm_load_i128"
                ).op,
                MM_INTR_NONE,
            )
            self.assertEqual(
                NativeVM._native_intrinsic_for_symbol(
                    "__mm_store_i128"
                ).op,
                MM_INTR_NONE,
            )

        with patch.dict(
            os.environ,
            {"MINIMACHINE_NATIVE_I128_INTRINSICS": "1"},
            clear=False,
        ):
            self.assertEqual(
                NativeVM._native_intrinsic_for_symbol(
                    "__mm_load_i128"
                ).op,
                MM_INTR_LOAD_I128,
            )
            self.assertEqual(
                NativeVM._native_intrinsic_for_symbol(
                    "__mm_store_i128"
                ).op,
                MM_INTR_STORE_I128,
            )

    def test_fast_memory_intrinsics_are_enabled_by_default_and_can_be_disabled(self):
        symbols = {
            "__mm_fast_memcpy": MM_INTR_FAST_MEMCPY,
            "__mm_fast_memset": MM_INTR_FAST_MEMSET,
            "__mm_fast_strlen": MM_INTR_FAST_STRLEN,
        }
        with patch.dict(os.environ, {}, clear=True):
            for symbol, expected in symbols.items():
                self.assertEqual(
                    NativeVM._native_intrinsic_for_symbol(symbol).op,
                    expected,
                )

        with patch.dict(
            os.environ,
            {"MINIMACHINE_NATIVE_FAST_MEMORY_INTRINSICS": "0"},
            clear=False,
        ):
            for symbol in symbols:
                self.assertEqual(
                    NativeVM._native_intrinsic_for_symbol(symbol).op,
                    MM_INTR_NONE,
                )

    def test_memory_intrinsics_are_enabled_by_default_and_can_be_disabled(self):
        symbols = {
            "__mm_llvm_memset_p0_i64": MM_INTR_MEMSET,
            "__mm_llvm_memcpy_p0_p0_i64": MM_INTR_MEMCPY,
        }
        with patch.dict(os.environ, {}, clear=True):
            for symbol, expected in symbols.items():
                self.assertEqual(
                    NativeVM._native_intrinsic_for_symbol(symbol).op,
                    expected,
                )

        with patch.dict(
            os.environ,
            {"MINIMACHINE_NATIVE_MEMORY_INTRINSICS": "0"},
            clear=False,
        ):
            for symbol in symbols:
                self.assertEqual(
                    NativeVM._native_intrinsic_for_symbol(symbol).op,
                    MM_INTR_NONE,
                )

    def test_i128_value_intrinsics_are_enabled_by_default_and_can_be_disabled(self):
        symbols = {
            "__mm_wide_const_128": MM_INTR_WIDE_CONST_I128,
            "__mm_icmp_eq_128": MM_INTR_ICMP_EQ_I128,
        }
        with patch.dict(os.environ, {}, clear=True):
            for symbol, expected in symbols.items():
                self.assertEqual(
                    NativeVM._native_intrinsic_for_symbol(symbol).op,
                    expected,
                )

        with patch.dict(
            os.environ,
            {"MINIMACHINE_NATIVE_I128_VALUE_INTRINSICS": "0"},
            clear=False,
        ):
            for symbol in symbols:
                self.assertEqual(
                    NativeVM._native_intrinsic_for_symbol(symbol).op,
                    MM_INTR_NONE,
                )

    def test_scalar_intrinsics_are_enabled_by_default_and_can_be_disabled(self):
        symbols = {
            "__mm_llvm_is_constant_i32": MM_INTR_IS_CONSTANT,
            "__mm_llvm_prefetch_p0": MM_INTR_PREFETCH,
            "__mm_udiv_64": MM_INTR_UDIV,
            "__mm_sdiv_64": MM_INTR_SDIV,
            "__mm_urem_32": MM_INTR_UREM,
            "__mm_srem_32": MM_INTR_SREM,
        }
        with patch.dict(os.environ, {}, clear=True):
            for symbol, expected in symbols.items():
                self.assertEqual(
                    NativeVM._native_intrinsic_for_symbol(symbol).op,
                    expected,
                )

        with patch.dict(
            os.environ,
            {"MINIMACHINE_NATIVE_SCALAR_INTRINSICS": "0"},
            clear=False,
        ):
            for symbol in symbols:
                self.assertEqual(
                    NativeVM._native_intrinsic_for_symbol(symbol).op,
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
