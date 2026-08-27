from __future__ import annotations

from . import muir, p3
from .verify import VerifyError, verify_muir, verify_p3


class MachineLoweringError(ValueError):
    pass


def _lower_operand(op: muir.Operand) -> p3.Operand:
    if isinstance(op, muir.Mem):
        return p3.Mem(op.address, op.width)
    return op


def lower_function(function: muir.Function) -> p3.Function:
    """Lower already-legalized μIR into strict P3.

    CALL/RET/HELPER/TRAP deliberately fail here until the ABI/runtime expander
    has removed them. That failure is part of the architectural boundary, not
    an implementation gap to silently paper over.
    """
    verify_muir(function)

    blocks: list[p3.Block] = []
    for block in function.blocks:
        out: list[p3.Instr] = []
        for inst in block.instructions:
            if isinstance(inst, muir.Mov):
                out.append(
                    p3.Mov(
                        width=inst.width,
                        dst=_lower_operand(inst.dst),
                        src=_lower_operand(inst.src),
                        extend=inst.extend,
                    )
                )
            elif isinstance(inst, muir.Sub):
                out.append(p3.Sub(inst.width, inst.dst, inst.a, inst.b))
            elif isinstance(inst, muir.Br):
                out.append(
                    p3.Br(
                        inst.width,
                        inst.cond,
                        inst.a,
                        inst.b,
                        inst.true_target,
                        inst.false_target,
                    )
                )
            else:
                raise MachineLoweringError(
                    f"pseudo must be expanded before strict P3: {type(inst).__name__}"
                )

        blocks.append(p3.Block(block.label, out))

    result = p3.Function(function.name, blocks, set(function.frame_slots))
    verify_p3(result)
    return result
