from __future__ import annotations

from dataclasses import dataclass
import re

from . import muir
from .verify import VerifyError, verify_muir


# Function pointer representation:
#   symbol/function-pointer value -> descriptor address
# Descriptor:
#   +0  entry code address
#   +8  fixed frame size (header + local value cells, excluding call arguments)
DESC_ENTRY = 0
DESC_FRAME_SIZE = 8

# Every activation frame starts with this fixed ABI header.
CALLER_SP = 0
RET_PC = 8
ENTRY = 16
FRAME_SIZE = 24
RETVAL = 32
RESUME_PC = 40
ARG_COUNT = 48
HEADER_SIZE = 64
WORD = 8


@dataclass
class AbiStats:
    calls: int = 0
    helpers: int = 0
    system_ops: int = 0
    returns: int = 0
    continuation_blocks: int = 0
    argument_loads: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "helpers": self.helpers,
            "system_ops": self.system_ops,
            "returns": self.returns,
            "continuation_blocks": self.continuation_blocks,
            "argument_loads": self.argument_loads,
        }


class AbiError(ValueError):
    pass


def _mem(base: muir.Value, offset: int) -> muir.Mem:
    return muir.Mem(muir.Address(base, offset), muir.Width.I64)


def _indirect_from_sp(offset: int) -> muir.Target:
    return muir.Target(address=muir.Address(muir.Special.SP, offset))


def _system_symbol(reason: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_]+", "_", reason).strip("_")
    return "__mm_trap_" + (tag or "unknown")


def _sysop_symbol(op: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_]+", "_", op).strip("_")
    if not tag:
        raise AbiError("empty system operation")
    return "__mm_sys_" + tag


def _uncond(target: muir.Target) -> muir.Br:
    return muir.Br(
        muir.Width.I8,
        muir.Cond.EQ,
        muir.Imm(0),
        muir.Imm(0),
        target,
        target,
    )


class _Builder:
    def __init__(self, function: muir.Function):
        self.function = function
        self.frame_slots = set(function.frame_slots)
        self.serial = 0

    def temp(self, stem: str) -> muir.Slot:
        self.serial += 1
        slot = muir.Slot(f"__abi_{stem}{self.serial}")
        self.frame_slots.add(slot.name)
        return slot

    def label(self, block: str, stem: str) -> str:
        self.serial += 1
        return f"{block}.__abi_{stem}{self.serial}"


def _descriptor(callee: muir.Callee) -> muir.Value:
    if callee.is_direct():
        assert callee.symbol is not None
        return muir.Symbol(callee.symbol)
    if callee.is_indirect():
        assert callee.slot is not None
        return callee.slot
    raise AbiError("invalid callee variant")


def _prologue(builder: _Builder, args: tuple[str, ...], stats: AbiStats) -> list[muir.Instr]:
    if not args:
        return []

    frame_size = builder.temp("frame_size")
    neg_size = builder.temp("neg_frame_size")
    arg_base = builder.temp("arg_base")
    out: list[muir.Instr] = [
        muir.Mov(muir.Width.I64, frame_size, _mem(muir.Special.SP, FRAME_SIZE)),
        muir.Sub(muir.Width.I64, neg_size, muir.Imm(0), frame_size),
        muir.Sub(muir.Width.I64, arg_base, muir.Special.SP, neg_size),
    ]
    for i, name in enumerate(args):
        slot = muir.Slot(name)
        builder.frame_slots.add(name)
        out.append(muir.Mov(muir.Width.I64, slot, _mem(arg_base, i * WORD)))
        stats.argument_loads += 1
    return out


def _call_sequence(
    builder: _Builder,
    callee: muir.Callee,
    args: tuple[muir.Value, ...],
    continuation: str,
) -> list[muir.Instr]:
    desc = _descriptor(callee)
    frame_size = builder.temp("callee_frame_size")
    entry = builder.temp("callee_entry")
    total_size = builder.temp("call_size")
    new_sp = builder.temp("new_sp")
    neg_frame_size = builder.temp("neg_callee_frame_size")
    arg_base = builder.temp("arg_base")

    arg_bytes = len(args) * WORD
    out: list[muir.Instr] = [
        muir.Mov(muir.Width.I64, entry, _mem(desc, DESC_ENTRY)),
        muir.Mov(muir.Width.I64, frame_size, _mem(desc, DESC_FRAME_SIZE)),
        # total_size = frame_size + argument bytes
        muir.Sub(muir.Width.I64, total_size, frame_size, muir.Imm(-arg_bytes)),
        # Stack grows downward.
        muir.Sub(muir.Width.I64, new_sp, muir.Special.SP, total_size),
        muir.Mov(muir.Width.I64, _mem(new_sp, CALLER_SP), muir.Special.SP),
        muir.Mov(
            muir.Width.I64,
            _mem(new_sp, RET_PC),
            muir.BlockAddr(builder.function.name, continuation),
        ),
        muir.Mov(muir.Width.I64, _mem(new_sp, ENTRY), entry),
        muir.Mov(muir.Width.I64, _mem(new_sp, FRAME_SIZE), frame_size),
        muir.Mov(muir.Width.I64, _mem(new_sp, ARG_COUNT), muir.Imm(len(args))),
    ]

    if args:
        # arg_base = new_sp + fixed frame_size. Arguments live directly after
        # the callee's fixed frame, so variadic call sites do not change local
        # slot offsets.
        out.extend(
            [
                muir.Sub(muir.Width.I64, neg_frame_size, muir.Imm(0), frame_size),
                muir.Sub(muir.Width.I64, arg_base, new_sp, neg_frame_size),
            ]
        )
        for i, value in enumerate(args):
            out.append(muir.Mov(muir.Width.I64, _mem(arg_base, i * WORD), value))

    out.extend(
        [
            muir.Mov(muir.Width.I64, muir.Special.SP, new_sp),
            _uncond(_indirect_from_sp(ENTRY)),
        ]
    )
    return out


def _return_sequence(builder: _Builder, value: muir.Value | None) -> list[muir.Instr]:
    caller_sp = builder.temp("caller_sp")
    out: list[muir.Instr] = [
        muir.Mov(muir.Width.I64, caller_sp, _mem(muir.Special.SP, CALLER_SP)),
        # Copy the continuation into the still-live caller frame before SP is
        # restored. After the MOV to SP, the branch can read it at a fixed
        # caller-frame offset without relying on a callee temporary.
        muir.Mov(
            muir.Width.I64,
            _mem(caller_sp, RESUME_PC),
            _mem(muir.Special.SP, RET_PC),
        ),
    ]
    if value is not None:
        out.append(muir.Mov(muir.Width.I64, _mem(caller_sp, RETVAL), value))
    out.extend(
        [
            muir.Mov(muir.Width.I64, muir.Special.SP, caller_sp),
            _uncond(_indirect_from_sp(RESUME_PC)),
        ]
    )
    return out


def expand_function(function: muir.Function) -> tuple[muir.Function, AbiStats]:
    """Expand CALL/HELPER/RET into the MOV/SUB/BR ABI.

    This is a sub-pass of the μIR -> P3 machine descent, not a new IR layer.
    Trap is lowered to an external system BR. ArchEscape intentionally remains
    for the future arch/minimachine replacement boundary.
    """
    verify_muir(function)
    stats = AbiStats()
    builder = _Builder(function)
    output: list[muir.Block] = []
    entry_label = function.blocks[0].label if function.blocks else None

    for original in function.blocks:
        current_label = original.label
        pending = list(original.instructions)
        if current_label == entry_label:
            pending = _prologue(builder, function.args, stats) + pending

        while True:
            call_index = next(
                (
                    i
                    for i, inst in enumerate(pending)
                    if isinstance(inst, (muir.Call, muir.Helper, muir.Sys))
                ),
                None,
            )
            if call_index is None:
                tail: list[muir.Instr] = []
                for inst in pending:
                    if isinstance(inst, muir.Ret):
                        stats.returns += 1
                        tail.extend(_return_sequence(builder, inst.value))
                    elif isinstance(inst, muir.Trap):
                        tail.append(_uncond(muir.Target(symbol=_system_symbol(inst.reason))))
                    else:
                        tail.append(inst)
                output.append(muir.Block(current_label, tail))
                break

            before = pending[:call_index]
            pseudo = pending[call_index]
            after = pending[call_index + 1 :]
            continuation = builder.label(original.label, "cont")
            stats.continuation_blocks += 1

            if isinstance(pseudo, muir.Call):
                callee = pseudo.callee
                args = pseudo.args
                result = pseudo.result
                stats.calls += 1
            elif isinstance(pseudo, muir.Helper):
                callee = muir.Callee(symbol=pseudo.symbol)
                args = pseudo.args
                result = pseudo.result
                stats.helpers += 1
            else:
                assert isinstance(pseudo, muir.Sys)
                callee = muir.Callee(symbol=_sysop_symbol(pseudo.op))
                args = pseudo.args
                result = pseudo.result
                stats.system_ops += 1

            sequence = before + _call_sequence(builder, callee, args, continuation)
            output.append(muir.Block(current_label, sequence))

            resume: list[muir.Instr] = []
            if result is not None:
                resume.append(
                    muir.Mov(
                        muir.Width.I64,
                        result,
                        _mem(muir.Special.SP, RETVAL),
                    )
                )
            pending = resume + after
            current_label = continuation

    result = muir.Function(
        function.name,
        output,
        builder.frame_slots,
        function.args,
    )
    verify_muir(result)
    verify_abi_normalized(result)
    return result, stats


def verify_abi_normalized(function: muir.Function) -> None:
    for block in function.blocks:
        for inst in block.instructions:
            if isinstance(inst, (muir.Call, muir.Helper, muir.Sys, muir.Ret, muir.Trap)):
                raise VerifyError(
                    f"ABI pseudo survived in {function.name}:{block.label}: "
                    f"{type(inst).__name__}"
                )
