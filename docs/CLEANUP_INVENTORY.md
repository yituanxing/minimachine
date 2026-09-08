# Repository cleanup inventory

This inventory is intentionally conservative. It records confirmed structural
redundancy before files are moved or removed.

## Repository scale at the reset point

- 213 tracked files;
- 98 active GitHub Actions workflow files;
- 28 scripts;
- 21 core Python modules;
- 20 test files.

The workflow count is much larger than the permanent validation contract.

## Confirmed permanent release gates

The v1.0 release document names exactly four frozen release gates:

1. `native-dynamic-service-contract.yml`
2. `lua-runtime-hot.yml`
3. `busybox-real-hot.yml`
4. `linux-minimachine-target.yml`

These are not cleanup candidates.

## Current infrastructure / active performance workflows

These are not release gates, but they currently produce inputs or measurements
used by active performance work and must be kept until their dependencies are
collapsed into a smaller pipeline:

- `core-ir-contract.yml`
- `initramfs-build.yml`
- `refresh-console-p3-cache.yml`
- `busybox-multicall-bundle.yml`
- `busybox-multicall-cold.yml`
- `native-checkpoint-roundtrip.yml`
- `perf-native-host-profile.yml`
- `runtime-contract.yml`
- `userspace-image-contract.yml`

This list is deliberately broader than the eventual target. The next step is
to collapse artifact production and profiling, not delete their producers
prematurely.

## Confirmed historical workflow families

The following naming families are frontier-finding or one-off diagnostics and
should not remain indefinitely as active CI:

- `*-probe.yml`
- `*-inspect.yml`
- `*-scout.yml`
- `*-diag.yml`
- old staged `hot-*` / `after-pty-*` checkpoint chains;
- one-off fixed-run-id workflows;
- superseded BusyBox ash bring-up workflows;
- superseded Linux frontier workflows.

Several of these still contain hard-coded old run IDs or branch-specific
`workflow_run` chains. That is strong evidence that they are historical
reproduction tools rather than current CI.

They should first be moved out of `.github/workflows/` into an evidence
archive or documented as recoverable from Git history.

## Confirmed duplicate test

Two files test the same runtime-suite script:

- `tests/test_minimachine-runtime-suite.py`
- `tests/test_minimachine_runtime_suite.py`

The hyphenated file is stale:

- it expects 12 core cases;
- the current suite contains 16;
- it expects the older `run_case id level script` interface;
- the current suite uses `run_case id script`.

The underscore-named test matches the current script. The hyphenated file is a
safe cleanup candidate after the structural branch has a baseline test run.

## Main source-structure debt

### `scripts/run-minimachine-linux.py`

At roughly 270 KiB it is no longer merely a CLI. It contains responsibilities
that belong to reusable platform modules. This is the highest-priority source
refactor, but it should be split by dependency boundary rather than rewritten.

### `runtime.py`

Generic LLVM/runtime helper semantics and application/platform host services are
too close together. The generic helper layer should remain independent from
Linux process/VFS/userspace policy.

### workflow orchestration

Generation matching, artifact discovery, cache construction and benchmark
timing are repeatedly encoded in YAML shell snippets. These should become
stable Python tooling APIs and leave workflows thin.

## Cleanup sequence

1. Establish this inventory and structure document.
2. Run/record baseline tests on the structure branch.
3. Remove confirmed duplicate/stale tests.
4. Move historical workflow families out of active CI in batches.
5. Collapse active artifact-producing workflows around a small explicit build
   graph.
6. Extract Linux platform/process code from the 270 KiB driver.
7. Re-run the four frozen release gates after every semantic-moving batch.

The cleanup must not change the frozen P3 ISA.
