# MiniMachine

MiniMachine is a compiler-driven minimal-machine research project.

The goal is not to minimize opcode count at any cost. The goal is to design the smallest practical machine boundary that makes lowering from LLVM mechanical, total where possible, and easy to audit.

## v1.0 status

MiniMachine has reached its v1.0 functional freeze. The v1.0 machine boundary is
P3:

- `MOV` — data movement, direct/indirect memory access, width-aware loads/stores
- `SUB` — integer state transformation
- `BR` — two-target conditional control flow, shaped to match LLVM `icmp + br`

For v1.0, this boundary is frozen. New primitives are not added merely to make a
new workload easier; any future ISA change must again justify itself against total
machine + VM + lowering + runtime complexity.

The runtime freeze point is commit
`710cf462d74fd9840d0f634d8996ba5e620722fa`. At that point the same code and
inputs passed the four release gates three consecutive times:

- Native Dynamic Service Contract;
- Lua Runtime Hot;
- BusyBox Real Software Hot;
- Linux MiniMachine Target Gate.

The BusyBox gate reaches `MINIMACHINE_REAL_EXTERNAL_END`, exits with status 0,
and rejects any `BOOT_EXEC_BLOCKED` marker. See
`docs/V1_RELEASE_STATUS.md` for the frozen evidence.

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
  v
P3 lowering
  |
  v
native P3 VM + checkpoint/replay
  |
  v
Linux 6.6.143 + real userspace
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


## Real-software development driver

MiniMachine is developed against complete upstream software, not by enumerating
application behaviors one by one.

The primary progression rule is:

1. boot the pinned Linux image;
2. run a real BusyBox multicall userspace through the normal Linux exec path;
3. let BusyBox init, rcS, the shell, and external applets expose the next missing
   machine/runtime semantic;
4. fix that semantic at the lowest reusable layer;
5. rerun the whole software workload.

Small shell/process cases are diagnostic and regression tests only. They must
not become the development roadmap, and a BusyBox/ash-specific host shortcut is
not considered a completed feature unless it represents a reusable ABI/runtime
contract.

The v1.0 software ladder is Linux -> BusyBox init/userspace -> Lua. BusyBox and
Lua are release gates and are stable at the v1.0 freeze point. SQLite and broader
network/application coverage are intentionally deferred to post-v1 work rather
than extending the v1.0 finish line.
