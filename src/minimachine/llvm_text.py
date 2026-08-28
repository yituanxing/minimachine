from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TextInst:
    result: str | None
    opcode: str
    text: str


@dataclass
class TextBlock:
    label: str
    instructions: list[TextInst]


@dataclass
class TextFunction:
    name: str
    args: tuple[str, ...]
    blocks: list[TextBlock]


_RESULT_RE = re.compile(r"^(%[-A-Za-z$._0-9]+)\s*=\s*(.*)$")
_LABEL_RE = re.compile(r"^([-A-Za-z$._0-9]+):(?:\s*;.*)?$")
_DEFINE_RE = re.compile(r"^define\b.*@([-A-Za-z$._0-9]+)\s*\((.*)\).*\{\s*$")
_OPCODE_RE = re.compile(
    r"^(?:(?:tail|musttail|notail)\s+)?"
    r"(callbr|call|ret|br|switch|indirectbr|unreachable|"
    r"add|sub|mul|udiv|sdiv|urem|srem|shl|lshr|ashr|and|or|xor|"
    r"alloca|load|store|getelementptr|trunc|zext|sext|ptrtoint|inttoptr|"
    r"bitcast|addrspacecast|icmp|phi|select|freeze|extractvalue|insertvalue)\b"
)


class LLVMTextError(ValueError):
    pass


def _opcode(body: str) -> str:
    m = _OPCODE_RE.match(body.strip())
    if not m:
        word = body.strip().split(None, 1)[0] if body.strip() else "<empty>"
        raise LLVMTextError(f"cannot identify LLVM opcode from: {word}: {body}")
    return m.group(1)


def _arg_names(args: str) -> tuple[str, ...]:
    return tuple(re.findall(r"%[-A-Za-z$._0-9]+", args))


def parse_module(text: str) -> list[TextFunction]:
    functions: list[TextFunction] = []
    current: TextFunction | None = None
    block: TextBlock | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue

        if current is None:
            m = _DEFINE_RE.match(line)
            if m:
                current = TextFunction(m.group(1), _arg_names(m.group(2)), [])
                block = None
            continue

        if line == "}":
            if block is not None:
                current.blocks.append(block)
            if not current.blocks:
                current.blocks.append(TextBlock("entry", []))
            functions.append(current)
            current = None
            block = None
            continue

        # llvm-dis prints callbr/asm-goto terminator successors on a
        # continuation line beginning with "to label". Keep it attached to
        # the callbr instruction instead of treating it as a new opcode.
        if line.startswith("to label ") and block is not None and block.instructions:
            prev = block.instructions[-1]
            if prev.opcode == "callbr":
                block.instructions[-1] = TextInst(prev.result, prev.opcode, prev.text + " " + line)
                continue

        # switch is printed as one logical instruction over multiple lines:
        #   switch i32 %x, label %default [
        #     i32 0, label %zero
        #   ]
        # Keep all case rows attached to the terminator.
        if block is not None and block.instructions:
            prev = block.instructions[-1]
            if (
                prev.opcode == "switch"
                and "[" in prev.text
                and not prev.text.rstrip().endswith("]")
            ):
                block.instructions[-1] = TextInst(
                    prev.result, prev.opcode, prev.text + " " + line
                )
                continue

        lm = _LABEL_RE.match(line)
        if lm:
            if block is not None:
                current.blocks.append(block)
            block = TextBlock(lm.group(1), [])
            continue

        # LLVM permits an implicit first entry block.
        if block is None:
            block = TextBlock("entry", [])

        result = None
        body = line
        rm = _RESULT_RE.match(line)
        if rm:
            result = rm.group(1)
            body = rm.group(2)

        try:
            op = _opcode(body)
        except LLVMTextError:
            # Declarations, attributes and metadata do not occur inside a
            # function body; anything unknown here is a real parser boundary.
            raise
        block.instructions.append(TextInst(result, op, body))

    if current is not None:
        raise LLVMTextError(f"unterminated function: {current.name}")
    return functions
