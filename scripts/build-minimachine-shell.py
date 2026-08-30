#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.minimachine import muir, p3
from src.minimachine.abi import expand_function
from src.minimachine.lower_p3 import lower_function
from src.minimachine.user_image import build_bflt
from src.minimachine.verify import verify_muir, verify_p3


SYS_READ = 63
SYS_WRITE = 64


def _store_bytes(dst: muir.Slot, data: bytes) -> list[muir.Instr]:
    return [
        muir.Mov(
            muir.Width.I8,
            muir.Mem(muir.Address(dst, index), muir.Width.I8),
            muir.Imm(byte),
        )
        for index, byte in enumerate(data)
    ]


def syscall(
    nr: int,
    args: tuple[muir.Value, ...],
    result: muir.Slot,
) -> muir.Call:
    padded = args + (muir.Imm(0),) * (6 - len(args))
    return muir.Call(
        muir.Callee(symbol="__mm_user_syscall"),
        (muir.Imm(nr),) + padded,
        result,
    )


def build_prompt_probe() -> p3.Function:
    prompt = b"mmsh> "
    buf = muir.Slot("buf")
    result = muir.Slot("result")
    fn = muir.Function(
        "__mm_shell_prompt_probe",
        [
            muir.Block(
                "entry",
                [
                    muir.Sub(
                        muir.Width.I64,
                        buf,
                        muir.Special.SP,
                        muir.Imm(256),
                    ),
                    *_store_bytes(buf, prompt),
                    syscall(
                        SYS_WRITE,
                        (muir.Imm(1), buf, muir.Imm(len(prompt))),
                        result,
                    ),
                    muir.Ret(None),
                ],
            )
        ],
        {"buf", "result"},
        (),
    )
    verify_muir(fn)
    expanded, _ = expand_function(fn)
    lowered = lower_function(expanded)
    verify_p3(lowered)
    return lowered


def build_echo_shell() -> p3.Function:
    prompt = b"mmsh> "
    buf = muir.Slot("buf")
    count = muir.Slot("count")
    result = muir.Slot("result")
    fn = muir.Function(
        "__mm_shell_echo",
        [
            muir.Block(
                "entry",
                [
                    muir.Sub(
                        muir.Width.I64,
                        buf,
                        muir.Special.SP,
                        muir.Imm(512),
                    ),
                    *_store_bytes(buf, prompt),
                    muir.Br(
                        muir.Width.I8,
                        muir.Cond.EQ,
                        muir.Imm(0),
                        muir.Imm(0),
                        muir.Target(label="prompt"),
                        muir.Target(label="prompt"),
                    ),
                ],
            ),
            muir.Block(
                "prompt",
                [
                    syscall(
                        SYS_WRITE,
                        (muir.Imm(1), buf, muir.Imm(len(prompt))),
                        result,
                    ),
                    syscall(
                        SYS_READ,
                        (muir.Imm(0), buf, muir.Imm(128)),
                        count,
                    ),
                    muir.Br(
                        muir.Width.I64,
                        muir.Cond.SLT,
                        count,
                        muir.Imm(1),
                        muir.Target(label="done"),
                        muir.Target(label="echo"),
                    ),
                ],
            ),
            muir.Block(
                "echo",
                [
                    syscall(
                        SYS_WRITE,
                        (muir.Imm(1), buf, count),
                        result,
                    ),
                    muir.Br(
                        muir.Width.I8,
                        muir.Cond.EQ,
                        muir.Imm(0),
                        muir.Imm(0),
                        muir.Target(label="entry"),
                        muir.Target(label="entry"),
                    ),
                ],
            ),
            muir.Block("done", [muir.Ret(None)]),
        ],
        {"buf", "count", "result"},
        (),
    )
    verify_muir(fn)
    expanded, _ = expand_function(fn)
    lowered = lower_function(expanded)
    verify_p3(lowered)
    return lowered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build MiniMachine userspace shell milestones."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mode",
        choices=("prompt-probe", "echo"),
        default="prompt-probe",
    )
    args = parser.parse_args()

    fn = build_prompt_probe() if args.mode == "prompt-probe" else build_echo_shell()
    image = build_bflt(fn, stack_size=256 * 1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(image)
    print(
        f"MINIMACHINE_SHELL_READY path={args.output} mode={args.mode} "
        f"bytes={len(image)} function={fn.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
