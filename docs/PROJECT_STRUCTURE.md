# MiniMachine project structure reset

This document defines the intended post-v1 repository structure before any
semantic or performance refactor. It separates the permanent machine/runtime
implementation from build tooling, release validation, performance experiments,
and historical diagnostics.

## Current problem

The repository is functionally healthy but structurally noisy:

- 98 GitHub Actions workflows live in one flat directory;
- many probe/inspect/scout/hot/cold workflows are historical frontier-finding
  artifacts rather than current release gates;
- the main execution driver, `scripts/run-minimachine-linux.py`, has grown to
  roughly 270 KiB and currently mixes image loading, Linux platform behavior,
  userspace/process semantics, diagnostics, checkpoint control, and CLI glue;
- runtime/platform responsibilities are split between `runtime.py`,
  `native_vm.py`, `run-minimachine-linux.py`, user-image modules, and
  workflow-specific orchestration;
- cache and benchmark preparation are treated as workflow implementation
  details even though they are now important performance dimensions.

The v1.0 release status already states that historical probe/inspect/scout
workflows are engineering evidence, not extra release gates. This reset makes
that distinction explicit in the repository structure.

## Architectural layers

### 1. core

Permanent compiler/machine semantics. Small, auditable, workload-independent.

Current modules:

- `llvm_text.py`
- `legalize.py`
- `muir.py`
- `layout.py`
- `abi.py`
- `lower_p3.py`
- `p3.py`
- `verify.py`
- `linker.py`

Rule: Linux, BusyBox, Lua, CI, checkpoint policy, and benchmark-specific logic
must not leak into this layer.

### 2. engine

Execution implementation of the frozen P3 contract.

Current modules:

- `vm.py` — reference Python engine;
- `native_vm.py` — Python/native binding and packed-program bridge;
- `native/p3vm.c` — native execution engine.

Rule: this layer implements machine execution and generic host-transition
mechanics. Linux process policy does not belong here.

### 3. platform

Guest system and process semantics above the machine.

Current responsibilities are spread across:

- `runtime.py`;
- large sections of `scripts/run-minimachine-linux.py`;
- `image.py`, `user_image.py`, `user_bundle.py`,
  `user_image_cache.py`;
- Linux-specific task/scheduler/syscall/VFS/user-transition code.

Target split:

- `platform/linux.py` — Linux boot/platform contract;
- `platform/process.py` — fork/vfork/exec/wait/task ownership;
- `platform/syscall.py` — syscall dispatch;
- `platform/libc.py` — reusable libc/external ABI services;
- `platform/userspace.py` — user image namespace and handoff.

The exact file split should follow dependency boundaries, not file size alone.

### 4. state/cache

Reusable artifact and replay state:

- `program_cache.py`
- `native_hot_cache.py`
- `checkpoint.py`
- user image caches

Target rule: cache construction, serialization, validation and restoration are
first-class APIs. Workflows should invoke these APIs rather than reimplementing
cache policy.

### 5. tooling

CLI/build/corpus tools only.

Current `scripts/` should eventually become thin entry points over importable
modules. In particular, `run-minimachine-linux.py` should become orchestration,
not the location where platform semantics live.

### 6. validation

Only stable contracts belong in the permanent validation surface.

Frozen v1 release gates:

- Native Dynamic Service Contract;
- Lua Runtime Hot;
- BusyBox Real Software Hot;
- Linux MiniMachine Target Gate.

Current performance-development gates may be retained separately, but must not
be presented as release requirements.

## Workflow policy

The 98 workflows should be classified into three groups before removal:

### release

Permanent correctness gates corresponding to the frozen release contract.

### development

A small active set for:
- current cold baseline;
- current hot baseline;
- native host profile;
- cache/checkpoint integrity;
- current performance A/B experiments.

### historical

Probe, inspect, scout, one-off ABI/frontier and superseded hot/cold workflows.
These should be moved out of `.github/workflows/` so GitHub does not present
them as active CI. Preserve them under a history/evidence area or through Git
history rather than deleting evidence blindly.

## Performance model

Performance is tracked as separate dimensions rather than one "boot time":

1. artifact/build preparation;
2. program/native-pack preparation;
3. kernel reset -> kernel_execve;
4. userspace kernel_execve -> real external software end;
5. host callback time versus pure native execution;
6. peak RSS and serialized artifact size.

Current measured examples motivating this split:

- native hot metadata build: about 55 s, near 10 GiB peak RSS;
- cached kernel reset with ror32+i128 candidate: about 50 s for
  815,095,846 unchanged P3 steps;
- corrected real BusyBox downstream: about 142 s;
- host-profile hot replay shows process/syscall host transitions
  (vfork/waitpid/ecall/exec) dominate callback time.

These are different optimization problems and should not be combined into one
benchmark number.

## Refactor order

1. Inventory and classify workflows and scripts.
2. Freeze a minimal active CI manifest.
3. Extract platform/process semantics from the Linux CLI driver without
   changing behavior.
4. Make build/cache timing a first-class benchmark.
5. Only then perform the next execution-performance refactor.

No P3 ISA change is part of this structural reset.
