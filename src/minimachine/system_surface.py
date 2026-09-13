from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re

from . import muir, p3
from .runtime import system_callback
from .vm import Program


_SYSTEM_PREFIX = "__mm_sys_"


@dataclass(frozen=True)
class SystemSurfaceResolution:
    referenced: tuple[str, ...]
    added: tuple[str, ...]
    unsupported: tuple[str, ...]


def system_symbol(op: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_]+", "_", op).strip("_")
    if not tag:
        raise ValueError("empty system operation")
    return _SYSTEM_PREFIX + tag


def system_op_from_symbol(symbol: str) -> str | None:
    if not symbol.startswith(_SYSTEM_PREFIX):
        return None
    op = symbol[len(_SYSTEM_PREFIX):]
    if not op or system_symbol(op) != symbol:
        return None
    return op


def _value_symbols(value) -> set[str]:
    if isinstance(value, muir.Symbol):
        return {value.name}
    if isinstance(value, muir.Reloc):
        return {value.symbol}
    if isinstance(value, muir.BlockAddr):
        return {value.function}
    return set()


def _address_symbols(address: muir.Address) -> set[str]:
    return _value_symbols(address.base)


def _operand_symbols(operand) -> set[str]:
    if isinstance(operand, p3.Mem):
        return _address_symbols(operand.address)
    return _value_symbols(operand)


def _target_symbols(target: muir.Target) -> set[str]:
    if target.symbol is not None:
        return {target.symbol}
    if target.address is not None:
        return _address_symbols(target.address)
    return set()


def referenced_symbols(functions: Iterable[p3.Function]) -> set[str]:
    symbols: set[str] = set()
    for function in functions:
        for block in function.blocks:
            for inst in block.instructions:
                if isinstance(inst, p3.Mov):
                    symbols.update(_operand_symbols(inst.dst))
                    symbols.update(_operand_symbols(inst.src))
                elif isinstance(inst, p3.Sub):
                    symbols.update(_value_symbols(inst.a))
                    symbols.update(_value_symbols(inst.b))
                elif isinstance(inst, p3.Br):
                    symbols.update(_value_symbols(inst.a))
                    symbols.update(_value_symbols(inst.b))
                    symbols.update(_target_symbols(inst.true_target))
                    symbols.update(_target_symbols(inst.false_target))
                else:
                    raise TypeError(
                        f"unsupported P3 instruction in system surface: "
                        f"{type(inst).__name__}"
                    )
    return symbols


def referenced_system_symbols(
    functions: Iterable[p3.Function],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            symbol
            for symbol in referenced_symbols(functions)
            if symbol.startswith(_SYSTEM_PREFIX)
        )
    )


def resolve_system_surface(
    program: Program,
    functions: Iterable[p3.Function],
) -> SystemSurfaceResolution:
    referenced = referenced_system_symbols(functions)
    added: list[str] = []
    unsupported: list[str] = []

    for symbol in referenced:
        if symbol in program.symbol_addresses:
            continue
        op = system_op_from_symbol(symbol)
        callback = system_callback(op) if op is not None else None
        if callback is None:
            unsupported.append(symbol)
            continue
        program.register_system(op, callback)
        if symbol not in program.symbol_addresses:
            raise RuntimeError(
                f"system surface resolver registered {op!r} but not {symbol!r}"
            )
        added.append(symbol)

    return SystemSurfaceResolution(
        referenced=referenced,
        added=tuple(added),
        unsupported=tuple(unsupported),
    )
