# MiniMachine

MiniMachine is a compiler-driven minimal-machine research project.

The goal is not to minimize opcode count at any cost. The goal is to design the smallest practical machine boundary that makes lowering from LLVM mechanical, total where possible, and easy to audit.

## Current research question

Can we design a machine competitive with SUBLEQ-style minimal machines in implementation size, while substantially reducing LLVM lowering complexity?

The current candidate is intentionally provisional:

- `MOV` — data movement, direct/indirect memory access, width-aware loads/stores
- `SUB` — integer state transformation
- `BR` — two-target conditional control flow, shaped to match LLVM `icmp + br`

Nothing in the repository treats this P3 candidate as frozen.

## Linux corpus contract

The primary reproducible corpus is pinned to:

- Linux **6.6.143**
- `ARCH=riscv`
- RISC-V `defconfig`
- official kernel.org source archive
- SHA-256 `dace1f8dc9c0dbf5df14f47e3229cd62c298e83049681731ef229f2ba7592932`

This intentionally matches the pinned Linux driver used by MiniC so that the source/configuration variable stays fixed while we study LLVM-to-machine lowering.

## Pipeline

```
Linux C
  |
  | Kbuild + preprocessing (expensive, cached)
  v
frozen .i
  |
  | Clang/LLVM (cached)
  v
raw LLVM bitcode
  |
  | normalization passes (cached)
  v
normalized LLVM
  |
  +--> census / semantic pressure
  |
  +--> P3 lowering experiments
  |
  +--> future tiny VM
```

Linux source, generated headers, `.i`, `.bc`, and normalized artifacts are reproducible cache material and are not committed to Git.

## Scale plan

The corpus is expanded in stable checkpoints:

- focused16 — fast semantic iteration
- full100 — first representative scale gate
- full500 — primary design gate
- full-all — eventual whole configured kernel corpus

Every checkpoint must report:

- attempted TUs
- materialized TUs
- raw/normalized LLVM instruction counts
- opcode/semantic-class count
- direct/structured/helper/arch-escape classification
- generic unsupported count
- cache hit/miss per pipeline stage

## Design rule

A new machine primitive is justified only when the reduction in total system complexity outweighs the added machine/VM complexity.

```
Total complexity =
    machine
  + VM
  + lowering
  + runtime
```

The project therefore measures lowering pressure before growing the ISA.
