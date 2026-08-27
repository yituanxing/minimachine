from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .muir import Address, Cond, Imm, Slot, Target, Width


Value = Union[Slot, Imm]


@dataclass(frozen=True)
class Mem:
    address: Address
    width: Width


Operand = Union[Value, Mem]


@dataclass(frozen=True)
class Mov:
    width: Width
    dst: Operand
    src: Operand
    extend: str | None = None


@dataclass(frozen=True)
class Sub:
    width: Width
    dst: Slot
    a: Value
    b: Value


@dataclass(frozen=True)
class Br:
    width: Width
    cond: Cond
    a: Value
    b: Value
    true_target: Target
    false_target: Target


Instr = Union[Mov, Sub, Br]


@dataclass
class Block:
    label: str
    instructions: list[Instr]


@dataclass
class Function:
    name: str
    blocks: list[Block]
    frame_slots: set[str]
