# Linux corpus and cache contract

## Why Linux 6.6.143

MiniMachine deliberately starts with the same frozen Linux release used by
MiniC's reproducible external driver:

- Linux 6.6.143
- RISC-V defconfig
- official kernel.org tarball
- pinned SHA-256

The purpose is not to claim this is the best Linux release. It removes a source
of experimental drift while the machine/lowering boundary is still changing.

## Expensive boundary

The expensive path is materialized once:

```
source archive
  -> extracted source
  -> generated Kbuild/config headers
  -> Clang kernel compilation
  -> .i + .bc + .s + .o
```

MiniMachine experiments operate below that boundary:

```
frozen raw .bc
  -> normalized .bc/.ll
  -> census
  -> lowering model
  -> P3 experiments
```

A P3 change must not cause Linux preprocessing or Clang frontend work to rerun.

## Corpus checkpoints

A configured build may contain more TUs than any one experiment needs.
Manifests select deterministic path-sorted prefixes:

- focused16
- full100
- full500
- full-all

The underlying materialized build is shared by all four.

## Cache identity

A materialized corpus is invalid if any of these change:

- Linux archive SHA-256
- architecture/config
- exact generated `.config`
- exact Clang version
- Kbuild compile command for the TU
- normalization pipeline version

Every manifest should eventually record those identities and hashes.

## Source policy

Do not commit:

- Linux source
- generated kernel headers
- preprocessed `.i`
- raw/normalized `.bc`
- object files

Commit only scripts, manifests of identities/results, reports, and design code.
