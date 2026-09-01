# Real-software validation ladder

MiniMachine is driven by complete upstream software.  Small synthetic cases are
diagnostic/regression tools only; they do not decide the roadmap.

## Selection rubric

A program is promoted into the main ladder when it adds a materially different
semantic surface while keeping failures attributable:

- **semantic novelty (35%)** — adds machine/runtime/OS behavior not already
  exercised heavily by earlier stages;
- **diagnostic clarity (25%)** — a failure can usually be reduced to a reusable
  ABI/runtime/kernel mechanism instead of application-specific behavior;
- **dependency independence (15%)** — avoids pulling a large dependency graph
  before the machine can support it;
- **reproducibility (15%)** — upstream release can be pinned and rebuilt
  deterministically;
- **real-world value (10%)** — passing it is meaningful evidence that ordinary
  software can run.

The ladder is intentionally not a collection of "small C programs".  Each
stage should open a new class of workload.

## Current ladder

### 0. Linux 6.6.143 + BusyBox 1.37.0

Role: operating-system and baseline userspace gate.

Adds:
- Linux boot and normal exec path;
- initramfs/VFS/file descriptors;
- BusyBox init, rcS, ash;
- fork/exec/wait, pipes and shell process control;
- a broad POSIX syscall/libc surface.

BusyBox remains the first gate because later programs should be launched by a
normal userspace environment rather than by a special MiniMachine harness.

### 1. Lua 5.4.9

Role: compact language-runtime gate.

Why it is next:
- whole interpreter and standard libraries are ANSI C with a small dependency
  surface;
- adds bytecode dispatch, indirect calls, closures and coroutine machinery;
- stresses allocation and garbage collection far more than BusyBox;
- exercises setjmp/longjmp-style error unwinding;
- adds floating point, numeric conversion, varargs and math-library pressure;
- can run non-trivial scripts without requiring threads or networking first.

Success criterion: the unmodified upstream interpreter bundle runs a real Lua
script through the Linux userspace path.  Missing functionality must be fixed
as generic libc/ABI/runtime behavior, not Lua-specific callbacks.

### 2. SQLite 3.53.4

Role: persistent file/VFS and 64-bit data-structure gate.

Why it follows Lua:
- amalgamation gives a reproducible, largely self-contained real application;
- file-backed databases exercise seek/truncate/sync/locking/time/error paths;
- parser, VM and B-tree code add large control/data-flow pressure;
- stresses 64-bit integer behavior and careful filesystem semantics;
- default SQLite behavior is a better VFS correctness test than more shell
  utilities, which largely overlap BusyBox.

Success criterion: the upstream CLI creates a file-backed database, performs
schema/data updates, closes it, reopens it, and verifies the persisted result.

### 3. Dropbear 2026.94

Role: network/event/process integration gate.

Why it is later:
- deliberately small and embedded-oriented, so it is a much cleaner network
  target than curl, OpenSSH, nginx or Git;
- adds sockets, bind/listen/accept/connect, select/poll-style waiting;
- adds cryptographic workloads, randomness, PTY/session handling and process
  interaction;
- depends on the process and file semantics proven by BusyBox and the richer
  libc/runtime semantics exposed by Lua/SQLite.

Success criterion: a real Dropbear client/server exchange over the MiniMachine
Linux networking path, without host-side application shortcuts.

## Deliberately deferred

- **cJSON** — useful compiler smoke test, but too much semantic overlap with
  already-proven parsing/allocation code to justify a main runtime stage.
- **zlib-only workloads** — good compute tests but weak OS/runtime coverage;
  keep as optional regression/benchmark material.
- **GNU coreutils/toybox** — substantial overlap with BusyBox and therefore low
  marginal coverage.
- **CPython/Git/OpenSSH/nginx** — valuable eventual stress targets, but their
  dependency and subsystem breadth make early failures poorly attributable.

## Development rule

When a ladder workload fails:

1. record the first failure from the complete upstream program;
2. classify it as ISA/lowering, generic runtime/libc ABI, Linux architecture
   contract, syscall/VFS/process/network semantics, or tooling;
3. repair the lowest reusable layer;
4. add only the smallest regression needed to freeze that repair;
5. rerun the complete workload.

A regression test may explain a failure, but passing that regression is not the
milestone.  The milestone is the real program moving forward.


## Evidence log

### Lua 5.4.9 — bundle milestone

The first complete Lua attempt immediately exposed a generic LLVM-text parser
defect rather than a Lua-specific problem. LLVM 18 emits profiled switch
terminators in the form `switch ... [ ... ], !prof !N`. MiniMachine had
incorrectly treated a switch as multiline until the whole instruction ended
with `]`, so it consumed the following basic blocks into the switch text.
The switch legalizer also rejected metadata after the closing case bracket.

The parser and legalizer were fixed generically and regression coverage was
added. Re-running the unchanged upstream Lua workload then passed the complete
bundle stage:

- normal RISC-V GCC/QEMU execution: PASS;
- whole-program LLVM LTO: PASS;
- MiniMachine lowering and bFLT/MMP3 bundle: PASS;
- 543 P3 functions;
- 684 global objects;
- 495 relocations;
- 113 runtime helpers;
- 108 external functions;
- 2,525,468-byte compressed payload.

This is the intended development pattern: a real program discovers a missing
generic mechanism; the repair is made below the application layer; the same
unmodified program advances.
