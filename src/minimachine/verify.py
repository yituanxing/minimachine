from __future__ import annotations

from . import muir, p3


class VerifyError(ValueError):
    pass


def _check_target(target: muir.Target, labels: set[str]) -> None:
    if target.is_direct():
        if target.label not in labels:
            raise VerifyError(f"unknown direct target: {target.label}")
        return
    if target.is_indirect():
        return
    raise VerifyError("target must be exactly one of local label, indirect slot, or indirect address")


def verify_muir(function: muir.Function) -> None:
    labels = {block.label for block in function.blocks}
    if len(labels) != len(function.blocks):
        raise VerifyError("duplicate μIR block label")

    for block in function.blocks:
        if not block.instructions:
            raise VerifyError(f"empty μIR block: {block.label}")

        for inst in block.instructions:
            if isinstance(inst, muir.Br):
                _check_target(inst.true_target, labels)
                _check_target(inst.false_target, labels)
            elif isinstance(inst, muir.Call):
                if not (inst.callee.is_direct() or inst.callee.is_indirect()):
                    raise VerifyError("callee must be exactly one of direct symbol or indirect slot")
            elif isinstance(inst, muir.Helper):
                if not inst.symbol:
                    raise VerifyError("HELPER requires a runtime symbol")
            elif isinstance(inst, muir.ArchEscape):
                for target in inst.targets:
                    _check_target(target, labels)

        terminator = block.instructions[-1]
        if isinstance(terminator, muir.ArchEscape):
            if not terminator.targets:
                raise VerifyError(f"non-terminating arch escape ends μIR block: {block.label}")
        elif not isinstance(terminator, (muir.Br, muir.Ret, muir.Trap)):
            raise VerifyError(f"μIR block has no terminator: {block.label}")


def verify_p3(function: p3.Function) -> None:
    labels = {block.label for block in function.blocks}
    if len(labels) != len(function.blocks):
        raise VerifyError("duplicate P3 block label")

    for block in function.blocks:
        if not block.instructions:
            raise VerifyError(f"empty P3 block: {block.label}")

        for inst in block.instructions:
            if not isinstance(inst, (p3.Mov, p3.Sub, p3.Br)):
                raise VerifyError(f"non-P3 instruction survived: {type(inst).__name__}")
            if isinstance(inst, p3.Br):
                _check_target(inst.true_target, labels)
                _check_target(inst.false_target, labels)

        if not isinstance(block.instructions[-1], p3.Br):
            raise VerifyError(
                f"strict P3 block must end in BR after ABI/runtime expansion: {block.label}"
            )
