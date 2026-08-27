# Lowering architecture

MiniMachine intentionally has **three representations**, **two lowering
boundaries**, and **two verifier boundaries**.

```
             rich / source-shaped semantics
                         |
                         v
                    LLVM IR
                         |
            [semantic legalization]
                         |
                         v
                       μIR
                         |
             [machine lowering / ABI]
                         |
                         v
                       P3
                         |
                  [P3 verifier]
                         |
                         v
                     μVM / code
             minimal executable semantics
```

`Legalizer` is a transformation, not another IR. `Verifier` is a checker,
not another IR.

Therefore the actual IR layers are:

1. LLVM IR
2. μIR
3. P3

The project performs two real descents:

1. LLVM -> μIR: **semantic descent**
2. μIR -> P3: **machine descent**

A μIR verifier runs between them. A P3 verifier runs after machine lowering.

## Layer 0: normalized LLVM input

Input is frozen LLVM bitcode from the real Linux Kbuild corpus and normalized
with LLVM middle-end O2 plus `lowerswitch`.

LLVM is allowed to remain rich here:

- SSA and PHI
- GEP
- select
- integer casts
- calls/returns
- intrinsics
- aggregates
- target/build escapes

This layer is not MiniMachine's executable contract.

## Descent A: LLVM -> μIR semantic legalization

This is where most compiler complexity belongs.

The Legalizer must remove source/compiler-shaped structure. It must not simply
rename LLVM opcodes.

By the time an instruction reaches μIR:

- `phi` has become edge moves;
- `icmp + br` is fused where legal;
- `select` has become control flow plus moves;
- `getelementptr` has become a memory address form or explicit address
  arithmetic;
- `add` has become SUB-based arithmetic;
- casts have become typed move semantics or disappear;
- aggregates have become explicit value-slot copies/materialization;
- bitwise/shift/mul/div/rem have become helper calls unless a later measured
  machine experiment adopts a primitive;
- LLVM intrinsics have become drop/helper/runtime/arch routes;
- `alloca` has become frame-layout metadata or explicit stack-area logic.

The normal μIR computational vocabulary is deliberately tiny:

```
MOV      typed copy / memory movement
SUB      fixed-width modular subtraction
BR       compare-and-branch
CALL     ABI pseudo
RET      ABI pseudo
HELPER   runtime-call pseudo
TRAP     defined failure/UB endpoint pseudo
```

`CALL / RET / HELPER / TRAP` are explicitly **pseudos**, not hidden machine
instructions. They must disappear in the next descent.

μIR also carries metadata that is not executable instructions:

- blocks and CFG edges;
- value slots;
- frame objects;
- widths;
- signed/unsigned extension mode;
- memory address forms;
- direct/indirect branch target kinds;
- source/LLVM provenance for diagnostics.

### Why keep μIR at all?

Without μIR, LLVM legalization and the three-instruction machine become
entangled. Every experiment on P3 would require touching LLVM-specific logic.

μIR creates a hard complexity boundary:

```
LLVM complexity -> Legalizer -> μIR invariant -> mechanical P3 lowering
```

P3 can later be replaced by P3+BIT, P4, or another experiment without
re-implementing PHI/GEP/select/intrinsic legalization.

## μIR verifier

The μIR verifier proves that semantic descent is complete enough for machine
lowering.

It rejects, among other things:

- remaining PHI/GEP/select/LLVM intrinsics;
- unknown widths;
- invalid memory address forms;
- CFG edges with missing PHI copies;
- malformed parallel-copy lowering;
- branch condition/operand width mismatch;
- unresolved generic LLVM operations;
- CALL/RET without valid ABI metadata;
- HELPER without a declared runtime symbol.

This is where `generic unsupported = 0` becomes an executable property rather
than a census classification.

## Descent B: μIR -> P3 machine lowering

This stage must be small and mechanical.

P3 has only three instruction **families**:

```
MOV.<mode,width> dst, src
SUB.<width>       dst, a, b
BR.<cc,width>     a, b, true_target, false_target
```

The family modes are part of the machine contract and must be counted as
machine complexity. The project must not hide complexity by claiming that
every operand/addressing mode is "free".

Current intended machine semantics:

### MOV

Supports typed movement between value/frame slots and memory operands.

Candidate address form:

```
[base + constant_offset]
```

Widths begin with 8/16/32/64. Loads may specify zero/sign extension as a MOV
mode.

### SUB

Three-address fixed-width modular subtraction:

```
SUB dst, a, b      # dst = a - b (mod 2^width)
```

Three-address form is intentional because LLVM/μIR values are SSA-like and a
two-address destructive form would insert copies.

### BR

Two-target compare-and-branch:

```
BR.EQ   a, b, T, F
BR.SLT  a, b, T, F
BR.ULT  a, b, T, F
```

Other LLVM integer predicates are derived by operand/target swapping.

An unconditional branch is:

```
BR.EQ 0, 0, T, T
```

The target operand must eventually support direct labels and an explicitly
defined indirect-target mode, because calls, returns, function pointers, and
the single observed Linux `indirectbr` require it.

## Where each LLVM semantic disappears

| LLVM semantic | Semantic descent: LLVM -> μIR | Machine descent: μIR -> P3 |
| --- | --- | --- |
| load/store | typed μIR MOV with memory operand | 1:1 P3 MOV |
| sub | μIR SUB | 1:1 P3 SUB |
| add | one/two μIR SUB operations | 1:1 P3 SUB |
| icmp + br | fuse to μIR BR.cc where legal | 1:1 P3 BR |
| standalone icmp | materialize boolean with BR + MOV CFG | MOV/BR |
| phi | predecessor-edge parallel MOVs | P3 MOV |
| select | CFG split + MOV | MOV/BR |
| constant GEP + memory | fold into μIR address operand | P3 MOV [base+off] |
| dynamic GEP | explicit address arithmetic | SUB + MOV |
| zext/sext/trunc | typed MOV mode or no-op | P3 MOV mode |
| bitcast/ptr casts | representation-preserving MOV/no-op where legal | MOV or none |
| alloca | frame object metadata | frame offsets in MOV/address operands |
| call | μIR CALL pseudo + explicit args/result | ABI MOV/BR sequence |
| ret | μIR RET pseudo | ABI MOV/indirect BR sequence |
| and/or/xor | HELPER pseudo initially | ABI sequence + helper body in P3 |
| shl/lshr/ashr | HELPER pseudo initially | ABI sequence + helper body in P3 |
| mul/div/rem | HELPER pseudo initially | ABI sequence + helper body in P3 |
| memcpy/memset/etc | HELPER/runtime pseudo | ABI sequence + helper body |
| unreachable | TRAP/runtime endpoint | branch to trap runtime / defined halt contract |
| callbr / inline asm | arch escape | future arch/minimachine contract |
| indirectbr | μIR BR with indirect target set | P3 indirect-target BR mode |
| module asm | build/arch escape | not generic P3 lowering |

## Examples

### LLVM compare + branch

```llvm
%c = icmp ult i64 %a, %b
br i1 %c, label %yes, label %no
```

Legalizer:

```
BR.ULT.i64 a, b, yes, no
```

P3 lowering is 1:1:

```
BR.ULT.i64 a, b, yes, no
```

### LLVM PHI

```llvm
%x = phi i64 [ %a, %left ], [ %b, %right ]
```

Legalizer places copies on predecessor edges:

```
left:
    MOV.i64 x, a
    ...
right:
    MOV.i64 x, b
    ...
merge:
```

P3 sees no PHI at all.

### LLVM GEP + load

```llvm
%p2 = getelementptr i64, ptr %p, i64 3
%x = load i64, ptr %p2
```

Legalizer:

```
MOV.i64 x, [p + 24]
```

P3:

```
MOV.i64 x, [p + 24]
```

### LLVM add

```llvm
%x = add i64 %a, %b
```

Legalizer:

```
SUB.i64 neg_b, 0, b
SUB.i64 x, a, neg_b
```

P3 lowering is 1:1 for both instructions.

### LLVM multiply

```llvm
%x = mul i64 %a, %b
```

Legalizer does **not** invent a μIR MUL. It emits:

```
HELPER __mm_mul_i64(a, b) -> x
```

Machine lowering expands the call ABI to MOV/BR. The helper implementation is
itself compiled/lowered to P3.

This is the boundary that lets strict P3 remain strict.

## Calls and frames

Calls are intentionally not solved in the LLVM Legalizer.

Semantic descent makes calling intent explicit:

- argument values;
- result destination;
- direct or indirect callee;
- continuation block;
- frame objects that must survive the call.

The ABI lowerer then decides the concrete frame/stack convention and expands
CALL/RET into P3 MOV/BR sequences.

This prevents Linux/LLVM call semantics from leaking into the three machine
instruction families.

## Verifier after P3

The P3 verifier accepts only:

- MOV
- SUB
- BR

No CALL, RET, HELPER, TRAP, PHI, GEP, select, LLVM intrinsic, or generic
operation may remain.

It also checks:

- legal widths and modes;
- valid direct/indirect branch targets;
- memory operand legality;
- frame offsets after layout;
- every block terminates;
- helper/ABI pseudos are fully expanded;
- no unresolved symbol remains except explicitly permitted external/system
  interfaces.

The first strict Linux gate is therefore:

```
2053/2053 LLVM parse
2053/2053 semantic legalize
2053/2053 μIR verify
2053/2053 P3 lower
2053/2053 P3 verify
generic unsupported = 0
```

Only after this gate should executable/differential VM testing become the
primary measure.

## Complexity rule

A primitive may be added to P3 only if measured total complexity improves:

```
machine + VM + legalizer + P3 lowerer + runtime
```

Static full-all data currently says the first candidate worth A/B measurement
is bitwise support, not MUL/DIV. Strict MOV/SUB/BR remains the frozen baseline
until executable evidence says otherwise.
