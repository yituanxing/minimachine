from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


class Width(Enum):
    I8 = 8
    I16 = 16
    I32 = 32
    I64 = 64


class Cond(Enum):
    EQ = "eq"
    SLT = "slt"
    ULT = "ult"


@dataclass(frozen=True)
class Slot:
    name: str


@dataclass(frozen=True)
class Imm:
    value: int


@dataclass(frozen=True)
class Symbol:
    name: str


Value = Union[Slot, Imm, Symbol]


@dataclass(frozen=True)
class Address:
    base: Value
    offset: int = 0


@dataclass(frozen=True)
class Mem:
    address: Address
    width: Width


Operand = Union[Value, Mem]


@dataclass(frozen=True)
class Target:
    label: str | None = None
    slot: Slot | None = None

    def is_direct(self) -> bool:
        return self.label is not None and self.slot is None

    def is_indirect(self) -> bool:
        return self.slot is not None and self.label is None


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


@dataclass(frozen=True)
class Callee:
    symbol: str | None = None
    slot: Slot | None = None

    def is_direct(self) -> bool:
        return self.symbol is not None and self.slot is None

    def is_indirect(self) -> bool:
        return self.slot is not None and self.symbol is None


@dataclass(frozen=True)
class Call:
    callee: Callee
    args: tuple[Value, ...]
    result: Slot | None
    continuation: str


@dataclass(frozen=True)
class Ret:
    value: Value | None


@dataclass(frozen=True)
class Helper:
    symbol: str
    args: tuple[Value, ...]
    result: Slot | None


@dataclass(frozen=True)
class Trap:
    reason: str


Instr = Union[Mov, Sub, Br, Call, Ret, Helper, Trap]


@dataclass
class Block:
    label: str
    instructions: list[Instr]


@dataclass
class Function:
    name: str
    blocks: list[Block]
    frame_slots: set[str]
