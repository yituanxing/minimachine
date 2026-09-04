# MiniMachine v1.0 release status

## Functional freeze

MiniMachine v1.0 is functionally frozen at runtime commit
`710cf462d74fd9840d0f634d8996ba5e620722fa`.

The v1.0 goal is deliberately bounded: a minimal P3 machine and runtime must
lower and execute the pinned Linux system, enter a real BusyBox userspace, run
real external commands through Linux process semantics, and execute an
independent real language runtime. It is not a claim of complete POSIX, glibc,
network, desktop, or arbitrary-application compatibility.

## Frozen machine boundary

P3 consists of:

- `MOV`;
- `SUB`;
- `BR`.

The ISA is frozen for v1.0. A post-v1 workload is not, by itself, justification
for growing the machine.

## Release gates

The same runtime commit and inputs were rerun three times without code changes.

| Gate | GitHub Actions run | Attempt 1 | Attempt 2 | Attempt 3 |
| --- | ---: | :---: | :---: | :---: |
| Native Dynamic Service Contract | `33883760844` | PASS | PASS | PASS |
| Lua Runtime Hot | `33883760758` | PASS | PASS | PASS |
| BusyBox Real Software Hot | `33883760746` | PASS | PASS | PASS |
| Linux MiniMachine Target Gate | `33883760887` | PASS | PASS | PASS |

The Native Dynamic Service Contract contains 48 focused regressions at the
freeze point.

## BusyBox end-to-end evidence

The formal BusyBox gate boots from the generation-matched checkpoint and drives
the normal Linux userspace path. The accepted run must:

- reach BusyBox rcS;
- print `MINIMACHINE_REAL_EXTERNAL_BEGIN`;
- execute real external `uname`, `ls`, and `cat` commands;
- create and schedule real fork children;
- reach `MINIMACHINE_REAL_EXTERNAL_END`;
- complete wait/exit cleanup;
- finish with `BUSYBOX_HOT_STATUS=0`;
- report `BUSYBOX_HOT_FRONTIER=external-software-passed`;
- contain no `BOOT_EXEC_BLOCKED`.

All three freeze attempts satisfy that contract.

## Process-semantics closure

The final v1.0 blocker was not an application-specific libc gap. It was
CLONE_VM process-stack ownership: a scheduled child could reuse concrete P3
userspace addresses before its parent reached wait4, corrupting the parent's
host-backed waitpid frame.

The freeze repair snapshots the parent P3 call chain before child scheduling,
restores it before entering wait4, and restores it again after the blocking
window. Together with task-scoped semantic stacks and scheduler ownership, this
stabilizes the exercised fork/exec/wait/exit path across repeated real-software
runs.

## What is release-critical

The following are v1.0 release-critical:

1. P3 lowering and native VM execution;
2. pinned Linux 6.6.143 target gate;
3. BusyBox init/rcS/shell and external-command process path;
4. Lua runtime gate;
5. the focused native dynamic service contract;
6. generation-matched cache/checkpoint validation.

The repository contains many historical probe, inspect, cold/hot scout, and
diagnostic workflows used to discover earlier frontiers. They are useful
engineering evidence, but they are not additional v1.0 release gates.

## Deferred beyond v1.0

The following are explicitly outside the v1.0 finish line:

- SQLite file-backed database coverage;
- Dropbear/network integration;
- complete POSIX/glibc coverage;
- arbitrary BusyBox applet completeness;
- dynamic-linker completeness;
- performance work that changes semantics;
- additional machine primitives without a new complexity justification.

These can become post-v1 milestones without reopening the v1.0 definition of
done.

## Release rule

After this freeze, runtime or ISA changes require evidence of a regression in a
frozen release gate or a separately declared post-v1 objective. Documentation,
packaging, CI cleanup, and reproducibility improvements may continue as long as
they do not silently expand the v1.0 contract.
