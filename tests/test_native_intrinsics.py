from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.minimachine import native_vm as native_vm_module
from src.minimachine.native_vm import (
    CBlock,
    CHostIntrinsic,
    CInst,
    COperand,
    CRunResult,
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


class NativeABIContractTests(unittest.TestCase):
    _PYTHON_ONLY_NATIVE_CONSTANTS = {"MM_V_INVALID"}
    _CONSTANT_PREFIXES = (
        "MM_OP_",
        "MM_V_",
        "MM_EXT_",
        "MM_COND_",
        "MM_T_",
        "MM_INTR_",
        "MM_IPRED_",
        "MM_STATUS_",
    )

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.c_source = (cls.repo_root / "native" / "p3vm.c").read_text(
            encoding="utf-8"
        )

    def _compile_layout_probe(self) -> dict[str, int]:
        compiler = shutil.which("cc") or shutil.which("gcc")
        if compiler is None:
            self.skipTest("native ABI layout probe requires cc or gcc")

        probe_source = r'''
#include <stddef.h>
#include <stdio.h>
#include "p3vm.c"

#define PRINT_SIZE(type) \
  printf("sizeof.%s=%zu\n", #type, sizeof(type))
#define PRINT_OFFSET(type, field) \
  printf("offsetof.%s.%s=%zu\n", #type, #field, offsetof(type, field))

int main(void) {
  PRINT_SIZE(MMOperand);
  PRINT_OFFSET(MMOperand, kind);
  PRINT_OFFSET(MMOperand, width);
  PRINT_OFFSET(MMOperand, base_kind);
  PRINT_OFFSET(MMOperand, _pad);
  PRINT_OFFSET(MMOperand, offset);
  PRINT_OFFSET(MMOperand, value);
  PRINT_OFFSET(MMOperand, base_value);

  PRINT_SIZE(MMInst);
  PRINT_OFFSET(MMInst, opcode);
  PRINT_OFFSET(MMInst, width);
  PRINT_OFFSET(MMInst, cond);
  PRINT_OFFSET(MMInst, extend);
  PRINT_OFFSET(MMInst, src_bits);
  PRINT_OFFSET(MMInst, _pad);
  PRINT_OFFSET(MMInst, op0);
  PRINT_OFFSET(MMInst, op1);
  PRINT_OFFSET(MMInst, op2);
  PRINT_OFFSET(MMInst, op3);

  PRINT_SIZE(MMBlock);
  PRINT_OFFSET(MMBlock, code);
  PRINT_OFFSET(MMBlock, first);
  PRINT_OFFSET(MMBlock, count);

  PRINT_SIZE(MMHostIntrinsic);
  PRINT_OFFSET(MMHostIntrinsic, op);
  PRINT_OFFSET(MMHostIntrinsic, bits);
  PRINT_OFFSET(MMHostIntrinsic, pred);
  PRINT_OFFSET(MMHostIntrinsic, _pad);
  PRINT_OFFSET(MMHostIntrinsic, imm);

  PRINT_SIZE(MMRunResult);
  PRINT_OFFSET(MMRunResult, status);
  PRINT_OFFSET(MMRunResult, error);
  PRINT_OFFSET(MMRunResult, target_code);
  PRINT_OFFSET(MMRunResult, block_code);
  PRINT_OFFSET(MMRunResult, sp);
  PRINT_OFFSET(MMRunResult, steps);
  PRINT_OFFSET(MMRunResult, ip);
  return 0;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            probe = tmp_path / "native_abi_probe.c"
            executable = tmp_path / "native_abi_probe"
            probe.write_text(probe_source, encoding="utf-8")
            compile_result = subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    "-O0",
                    "-I",
                    str(self.repo_root / "native"),
                    str(probe),
                    "-o",
                    str(executable),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(
                compile_result.returncode,
                0,
                msg=f"native ABI probe compilation failed:\n{compile_result.stderr}",
            )
            run_result = subprocess.run(
                [str(executable)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(
                run_result.returncode,
                0,
                msg=f"native ABI probe execution failed:\n{run_result.stderr}",
            )

        values: dict[str, int] = {}
        for line in run_result.stdout.splitlines():
            key, value = line.split("=", 1)
            values[key] = int(value)
        return values

    @staticmethod
    def _ctypes_layout(struct_type, fields: tuple[str, ...]) -> tuple[int, dict[str, int]]:
        return (
            ctypes.sizeof(struct_type),
            {field: getattr(struct_type, field).offset for field in fields},
        )

    def test_python_and_c_struct_layouts_match(self):
        c_layout = self._compile_layout_probe()
        self.assertEqual(
            [name for name, _ctype in CInst._fields_ if name.startswith("op")],
            ["opcode", "op0", "op1", "op2", "op3"],
        )

        contracts = (
            (
                "MMOperand",
                COperand,
                ("kind", "width", "base_kind", "_pad", "offset", "value", "base_value"),
            ),
            (
                "MMInst",
                CInst,
                ("opcode", "width", "cond", "extend", "src_bits", "_pad", "op0", "op1", "op2", "op3"),
            ),
            ("MMBlock", CBlock, ("code", "first", "count")),
            ("MMHostIntrinsic", CHostIntrinsic, ("op", "bits", "pred", "_pad", "imm")),
            (
                "MMRunResult",
                CRunResult,
                ("status", "error", "target_code", "block_code", "sp", "steps", "ip"),
            ),
        )
        for c_name, py_type, fields in contracts:
            with self.subTest(struct=c_name):
                py_size, py_offsets = self._ctypes_layout(py_type, fields)
                self.assertEqual(c_layout[f"sizeof.{c_name}"], py_size)
                for field, py_offset in py_offsets.items():
                    self.assertEqual(
                        c_layout[f"offsetof.{c_name}.{field}"],
                        py_offset,
                        msg=f"ABI offset drift: {c_name}.{field}",
                    )

    def test_python_and_c_numeric_native_constants_match(self):
        pattern = re.compile(
            r"^#define\s+(MM_[A-Z0-9_]+)\s+([0-9]+)[uUlL]*\s*$",
            re.MULTILINE,
        )
        c_values = {
            name: int(value)
            for name, value in pattern.findall(self.c_source)
            if name.startswith(self._CONSTANT_PREFIXES)
        }
        python_values = {
            name: value
            for name, value in vars(native_vm_module).items()
            if name.startswith(self._CONSTANT_PREFIXES)
            and isinstance(value, int)
            and name not in self._PYTHON_ONLY_NATIVE_CONSTANTS
        }

        self.assertTrue(c_values, "no native C ABI constants were discovered")
        self.assertEqual(
            set(c_values),
            set(python_values),
            msg=(
                "native ABI constant names drifted between native/p3vm.c and "
                "src/minimachine/native_vm.py"
            ),
        )
        for name, c_value in c_values.items():
            with self.subTest(constant=name):
                self.assertEqual(c_value, python_values[name])


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
