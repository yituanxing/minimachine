# Linux 6.6.143 RISC-V full-all static analysis

Frozen evidence:

- Linux: 6.6.143
- ARCH: riscv
- config: defconfig
- LLVM: Ubuntu Clang/LLVM 18.1.3
- real Kbuild C translation units: **2053**
- full compile/materialization run: **33088602844**
- full-all analysis run: **33091706490**
- normalization: LLVM middle-end `default<O2>` + `lowerswitch`
- normalization parallelism: 4 workers
- generic unsupported: **0**

One build-oriented TU, `kernel/configs.bc`, contains nine module-level
assembler lines including `.incbin "kernel/config_data.gz"`. It remains in
the 2053-TU corpus and is recorded as an explicit module-asm/build escape.
Only module asm is stripped from the analysis copy before middle-end
normalization.

## Saturation

| Metric | full500 | full-all | Growth |
| --- | ---: | ---: | ---: |
| TUs | 500 | 2053 | 4.106x |
| O2 LLVM instructions | 296,128 | 2,587,814 | 8.739x |
| opcode kinds | 33 | **34** | +1 |
| generic unsupported | 0 | **0** | 0 |

The only opcode kind newly observed after moving from 500 to all 2053 TUs is:

- `indirectbr`: **1**

No opcode kind present in full500 disappeared.

This is strong evidence that semantic-kind saturation occurs well before the
full configured kernel corpus.

## Full-all opcode census

Top generic LLVM instructions:

| Opcode | Count | Share |
| --- | ---: | ---: |
| br | 444,174 | 17.16% |
| getelementptr | 380,099 | 14.69% |
| load | 378,109 | 14.61% |
| icmp | 315,015 | 12.17% |
| call | 309,605 | 11.96% |
| store | 153,231 | 5.92% |
| phi | 112,562 | 4.35% |
| and | 87,535 | 3.38% |
| add | 73,969 | 2.86% |
| zext | 45,539 | 1.76% |
| select | 40,311 | 1.56% |
| ret | 39,833 | 1.54% |

All 34 observed kinds:

```
add alloca and ashr br call callbr extractvalue freeze getelementptr
icmp indirectbr insertvalue inttoptr load lshr mul or phi ptrtoint
ret sdiv select sext shl srem store sub trunc udiv unreachable urem
xor zext
```

## P3 route map

| Route | Count | Share |
| --- | ---: | ---: |
| P3 native / fused | 1,472,579 | **56.90%** |
| structured lowering | 813,963 | **31.45%** |
| helper / expand | 188,931 | **7.30%** |
| arch escape | 72,790 | **2.81%** |
| drop | 37,022 | 1.43% |
| special runtime | 2,529 | 0.10% |
| unsupported | **0** | **0.00%** |

Therefore:

- direct + deterministic structured lowering: **88.36%**
- direct + structured + helper route: **95.66%**
- generic unsupported: **0**

The lower direct percentage compared with full500 is mainly caused by the
full corpus containing more arch/driver-heavy code and more helper-heavy
integer code; it is not caused by new generic LLVM semantic kinds.

## Control-flow pressure

- `icmp`: 315,015
- fused directly into P3 `BR.cc`: 254,463
- materialized comparison result: 60,552
- fusion rate: **80.78%**

This validates a two-target compare-and-branch primitive as a high-ROI machine
semantic.

Predicate distribution:

| Predicate | Count |
| --- | ---: |
| eq | 230,817 |
| slt | 25,855 |
| ult | 19,457 |
| ugt | 17,132 |
| ne | 9,693 |
| sgt | 9,675 |
| sle | 941 |
| ule | 764 |
| uge | 431 |
| sge | 250 |

The complete LLVM integer predicate set is still derivable from the
`EQ / SLT / ULT` basis by operand and target inversion.

## Addressing pressure

- GEP total: 380,099
- constant-index GEP: 334,615
- directly foldable into a load/store memory operand: 222,038
- memory-fold rate: **58.42%**
- dynamic GEP with a single load/store user: 25,445

Base+constant addressing on `MOV` remains strongly justified. Scaled-index
addressing is a candidate for measurement, not yet a required primitive.

## Helper pressure

The 188,931 helper/expand sites decompose exactly as:

| Helper class | Sites | Share of helper | Share of all IR |
| --- | ---: | ---: | ---: |
| AND / OR / XOR | 121,265 | **64.18%** | **4.69%** |
| SHL / LSHR / ASHR | 34,887 | 18.47% | 1.35% |
| MUL / DIV / REM | 7,634 | 4.04% | 0.29% |
| helper intrinsics | 25,145 | 13.31% | 0.97% |

This changes the priority of future ISA experiments:

1. bitwise support deserves an A/B complexity measurement;
2. shifts are the second candidate;
3. multiply/divide should remain software initially.

This is not yet a decision to grow P3. The strict `MOV/SUB/BR` machine
remains the baseline.

## Arch/build escape pressure

The instruction-level arch-escape count is exactly:

- inline asm call sites: 52,685
- arch intrinsics: 16,848
- `callbr`: 3,257
- total: **72,790**

In addition, one TU has module-level build assembler
(`kernel/configs.bc`, nine module-asm lines).

These are not evidence that the generic MiniMachine ISA is missing ordinary
computation semantics. They are evidence that a Linux `arch/minimachine`
port must replace RISC-V-specific system contracts.

## Atomics

The current normalized RISC-V defconfig corpus reports no generic LLVM
`atomicrmw/cmpxchg/fence` instructions. This does **not** prove that a future
SMP MiniMachine can omit atomic semantics: Linux/RISC-V atomics are currently
largely represented through target-specific asm/intrinsics and arch code.

For the first Linux target, NOMMU/UP plus software/runtime lowering remains the
cleanest contract.

## Static emitted P3 proxy

A deliberately conservative static expansion model gives:

- low estimate: **2,482,938 P3 instruction sites**
- high estimate: **5,039,050 P3 instruction sites**
- relative to normalized LLVM: **0.96x - 1.95x**

This is a static code-site proxy, not a cycle/performance estimate.

It excludes:

- shared helper routine bodies;
- `arch/minimachine` replacement code;
- dynamic loop counts inside software helpers.

The main dynamic-performance uncertainty is therefore concentrated in
bitwise/shift helpers, not in the generic control/memory lowering surface.

## Source LOC forecast

Recommended implementation assumes LLVM libraries are used to read bitcode
instead of implementing a standalone LLVM parser.

| Component | Forecast |
| --- | ---: |
| LLVM adapter/reader | 1.5k - 3k |
| LLVM legalizer | **5.2k - 8.9k** |
| μIR + verifier | 2.5k - 4.5k |
| μIR -> P3 | 1k - 2.5k |
| ABI/call/frame | 2k - 4k |
| runtime helpers | 3k - 7k |
| reference VM | **0.6k - 1.5k** |
| **recommended core total** | **15.8k - 31.4k** |

Expected center of mass for the core is approximately **22k-26k LOC** if the
34-kind semantic surface remains closed and we reuse LLVM's reader/APIs.

Additional ranges:

- standalone LLVM parser: **+5k - 10k LOC**
- tests/CI/corpus infrastructure: **8k - 20k LOC**
- first Linux NOMMU/UP `arch/minimachine`: **5k - 10k LOC**
- mature MMU/SMP arch port: **12k - 25k LOC**

A practical first Linux-capable repository is therefore expected around
**35k-55k LOC**, with only roughly half of that being compiler/machine core.

## Candidate machine scenarios

### P3 strict: MOV / SUB / BR

Keep all bitwise, shift, mul/div/rem operations in helper/runtime lowering.

Advantages:

- smallest VM and machine contract;
- strongest research result;
- compiler surface already closes at unsupported=0.

Risk:

- dynamic cost of bitwise and shift helpers may dominate execution.

### P3 + bitwise experiment

A small ALU/bitwise mode could directly absorb 121,265 current helper sites,
or **64.18% of all helper pressure**.

Expected implementation delta:

- VM/machine: roughly +100-300 LOC
- legalizer/emitter: roughly +100-300 LOC
- runtime: potentially -300 to -1000 LOC

This candidate has the highest static ROI and deserves measurement, but should
not be adopted before executable P3 cost data exists.

### P3 + shift experiment

Would absorb 34,887 sites, 18.47% of helper pressure.

Lower ROI than bitwise, but shifts may have high dynamic software cost.

### Hardware MUL/DIV

Only 7,634 sites, 4.04% of helper pressure and 0.29% of all IR.

Keep software initially.

## Current decision

Do **not** slim Linux because of compiler semantic coverage.

Do **not** grow the ISA because opcode kinds increased.

The evidence currently supports:

1. freeze 34 generic LLVM kinds as the first semantic contract;
2. implement strict LLVM -> μIR -> P3 lowering;
3. make `unsupported=0` an executable verifier property, not only a census property;
4. benchmark bitwise/shift helpers on real lowered code;
5. grow the machine only if measured total complexity or runtime cost justifies it;
6. handle RISC-V asm/system contracts in a future `arch/minimachine` port.
