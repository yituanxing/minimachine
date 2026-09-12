# Native P3 Phase 1 closure

## Scope

Native P3 Phase 1 is a post-v1 implementation/performance milestone. It does
not reopen or expand the MiniMachine v1.0 functional contract documented in
`V1_RELEASE_STATUS.md`.

The phase moves the established P3 execution path toward the C native VM while
preserving the frozen machine semantics and the Linux/real-software acceptance
boundary.

## Effective runtime chain

The compact native instruction ABI was introduced and synchronized through:

- C compact layout: `bd01d0604b8b25315b223771daca2c979016848f`;
- Python native pack synchronization: `501d4440d42881fb48c65dcb9c34f2938bf04a59`;
- compact ABI regression coverage: `6fd1b28908fc2a2dbe48f4ae4c18f23babdc35be`.

The phase includes the native ror32, i128 load/store, scalar, i128-value,
memory, and fast-memory paths together with generation-matched native pack and
checkpoint handling.

## Formal acceptance

The frozen acceptance evidence is GitHub Actions run `34682892461` from
`.github/workflows/busybox-multicall-cold.yml`.

The main cold job and dependent real-software job both passed. The accepted
pipeline verifies:

1. native intrinsic mappings and native VM build;
2. scalar/i128/memory edge cases;
3. generation-matched Linux, P3 cache, and BusyBox inputs;
4. real BusyBox rootfs cold boot to the first `kernel_execve` checkpoint;
5. reset, ror32, i128, scalar, i128-value, memory, and fast-memory cached-cold
   candidates;
6. exact checkpoint-step equivalence at every A/B boundary;
7. checkpoint restore and real BusyBox rcS/shell/external-software execution;
8. no `BOOT_EXEC_BLOCKED` on the accepted real-software path.

The exact generation identity and gate state are machine-readable in
`baselines/native-p3-phase1-acceptance.json`.

## ABI closure gate

Phase closure adds a direct Python/C ABI contract test to
`tests/test_native_intrinsics.py`. The test compiles a tiny C probe against the
actual `native/p3vm.c` definitions and compares `sizeof`/`offsetof` values with
Python `ctypes` structures. It also compares shared numeric native constants.

This gate is intentionally part of the existing formal cold validation step so
a C/Python layout or numeric-ABI drift fails before a Linux boot is attempted.
The corrected contract first passed in formal run `34685872480` at commit
`1fdee4e0a2574e43da4fae83cea3b5ad41d8c830`.

## Performance evidence rule

Cold, cached-cold, and hot/checkpoint timings are different measurements and
must not be mixed.

The last previously recorded trusted cold value is 49.07 s. The historical hot
checkpoint A/B was 8.73 s -> 7.52 s (-13.86%), and is not a cold result. A
52.65 s result from the old ash workflow is explicitly rejected because its
lowering generation did not match the cache generation.

Run `34682892461` passed the formal correctness gates, but its small generation
manifest did not persist every `*_WALL` line. GitHub Actions step duration must
not be substituted for the benchmark value. `scripts/summarize-native-p3-cold.py`
exists to turn the real benchmark logs into a compact machine-readable summary.

## Exit criteria

Native P3 Phase 1 is closed when all of the following remain true:

- the Python/C ABI contract passes;
- the formal generation-matched cold pipeline passes;
- every candidate reaches the same `kernel_execve` step count as its control;
- real BusyBox external software passes after checkpoint restore;
- exact benchmark values are derived from `/usr/bin/time` lines in the VM logs,
  not CI step duration;
- further native fast paths are justified by a fresh profile rather than by
  adding intrinsics speculatively.

## Next phase

The default post-closure direction is real-software breadth rather than more
LLVM legalizer coverage. The first target should be SQLite, followed by more
musl CLI/process/thread workloads. A new performance phase should begin only
after profiling the closed Native P3 baseline and should attack the dominant
measured hotspots.

Large-file refactoring (`run-minimachine-linux.py`, then `runtime.py`) is a
separate maintainability track and should be characterization-test driven; it
must not be mixed with semantic or performance changes.
