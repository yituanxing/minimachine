# Compiler / System-Contract Freeze

This document freezes the boundary reached by the Linux 6.6.143 RISC-V
full-all corpus before starting the `arch/minimachine` port.

## Frozen validation baseline

Source corpus:

- Linux 6.6.143
- RISC-V defconfig
- LLVM 18 normalized IR
- 2053 frozen translation units

Frozen full-all ABI/P3 run:

- workflow: `Linux Full-All ABI`
- run: `33143130985`
- head: `ca1bb49b3eda3644cfe2d66cc1b9021930531bd1`

Results:

- TU pass: **2053 / 2053**
- functions: **39,894**
- strict-P3 functions: **39,799**
- functions intentionally blocked by arch escape: **95**
- arch escape sites: **165**
- remaining escape template groups: **10**
- P3 instructions emitted: **12,594,658**

The strict-P3 coverage at the function level is therefore approximately
**99.76%**.

## What is already a MiniMachine system contract

The compiler no longer treats these operations as opaque RISC-V assembly.
They have explicit μIR `SYS` semantics and are expanded through the
MiniMachine ABI before strict P3:

- memory fences and ordering
- instruction-cache synchronization
- atomic AMO operations with ordering
- LR/SC semantic atomic operations
- faultable loads and stores
- faultable futex atomic operations
- supervisor state access
- generic CSR read/write/set/clear/read-modify-write
- TLB invalidation
- counters
- wait-for-interrupt
- ecall
- Linux static keys
- CPU-feature conditional branches
- vector state metadata/snapshot/restore contracts where the operation is
  architectural state rather than a concrete vector register implementation

CALL, HELPER, SYS, RET and TRAP are ABI pseudos. They must be removed before
strict P3. Strict P3 remains only:

- `MOV`
- `SUB`
- `BR`

## Why the remaining 165 escapes are frozen

The remaining sites are not generic LLVM/compiler gaps. They are concrete
RISC-V implementation mechanisms that should disappear when Linux gains an
`arch/minimachine` implementation.

The 165 sites are concentrated in ten templates:

- 99 sites: RISC-V runtime-alternative address/feature calculation emitted
  through arch headers and inlined into generic driver TUs
- 49 sites: RISC-V MM/page-table runtime-alternative implementation
- 6 sites: RISC-V non-coherent DMA / PMEM cache-block alternative sequences
- 7 sites: concrete RISC-V vector-register save/load/init sequences
- 2 sites: RISC-V IRQ/softirq stack-switch assembly
- 1 site: RISC-V alternative-wrapped CSR implementation
- 1 site: indirect control transfer

These are deliberately **not** converted into generic `SYS` operations merely
to obtain a cosmetic 100% score.

## Frozen invariant

Compiler side:

> Linux full-all generic semantic lowering remains 2053/2053, and no generic
> atomic, faultable-access, or ecall family may regress back into ArchEscape.

Architecture side:

> Replace the remaining RISC-V implementation sites with Linux
> `arch/minimachine` code and MiniMachine system-runtime services.

The full-all CI gate enforces the frozen floor:

- `p3_function_pass >= 39799`
- `p3_function_skip_escape <= 95`
- `arch_escape_sites <= 165`
- no remaining generic atomic/faultable/ecall escape family

## Next stage

The next work is no longer broad compiler legalization.

The first `arch/minimachine` milestone should intentionally be small:

1. define target configuration and Kconfig boundary;
2. define the MiniMachine low-level system-runtime ABI;
3. provide minimal boot/entry, memory-layout, interrupt and timer contracts;
4. use a deliberately minimal first Linux configuration;
5. compile that target through the frozen LLVM → μIR → P3 pipeline;
6. replace architecture escapes at their source instead of teaching the
   Legalizer RISC-V implementation details.

This freeze is a design boundary, not a claim that a MiniMachine Linux kernel
already boots.
