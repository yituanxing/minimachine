from __future__ import annotations

from . import muir, p3
from .abi import expand_function
from .image import (
    BlockExpr,
    ImageAlias,
    ImageObject,
    ModuleImage,
    Relocation,
    SymbolExpr,
    parse_module_image,
)
from .legalize import legalize_module
from .lower_p3 import lower_function
from .runtime import RuntimeSurface, collect_runtime_surface
from .user_image import UserProgramImage
from .verify import verify_muir, verify_p3


def _rewrite_value(value, mapping: dict[str, str]):
    if isinstance(value, muir.Symbol):
        return muir.Symbol(mapping.get(value.name, value.name))
    if isinstance(value, muir.Reloc):
        return muir.Reloc(
            mapping.get(value.symbol, value.symbol),
            value.addend,
        )
    if isinstance(value, muir.BlockAddr):
        return muir.BlockAddr(
            mapping.get(value.function, value.function),
            value.label,
        )
    return value


def _rewrite_address(address: muir.Address, mapping: dict[str, str]):
    return muir.Address(
        _rewrite_value(address.base, mapping),
        address.offset,
    )


def _rewrite_operand(operand, mapping: dict[str, str]):
    if isinstance(operand, p3.Mem):
        return p3.Mem(
            _rewrite_address(operand.address, mapping),
            operand.width,
        )
    return _rewrite_value(operand, mapping)


def _rewrite_target(target: muir.Target, mapping: dict[str, str]):
    if target.label is not None:
        return muir.Target(label=target.label)
    if target.symbol is not None:
        return muir.Target(
            symbol=mapping.get(target.symbol, target.symbol)
        )
    if target.slot is not None:
        return muir.Target(slot=target.slot)
    if target.address is not None:
        return muir.Target(
            address=_rewrite_address(target.address, mapping)
        )
    raise ValueError("cannot rewrite empty P3 target")


def _rewrite_instruction(inst, mapping: dict[str, str]):
    if isinstance(inst, p3.Mov):
        return p3.Mov(
            inst.width,
            _rewrite_operand(inst.dst, mapping),
            _rewrite_operand(inst.src, mapping),
            inst.extend,
            inst.src_bits,
        )
    if isinstance(inst, p3.Sub):
        return p3.Sub(
            inst.width,
            inst.dst,
            _rewrite_value(inst.a, mapping),
            _rewrite_value(inst.b, mapping),
        )
    if isinstance(inst, p3.Br):
        return p3.Br(
            inst.width,
            inst.cond,
            _rewrite_value(inst.a, mapping),
            _rewrite_value(inst.b, mapping),
            _rewrite_target(inst.true_target, mapping),
            _rewrite_target(inst.false_target, mapping),
        )
    raise TypeError(f"unsupported P3 instruction: {type(inst).__name__}")


def _rewrite_function(
    function: p3.Function,
    mapping: dict[str, str],
) -> p3.Function:
    rewritten = p3.Function(
        mapping.get(function.name, function.name),
        [
            p3.Block(
                block.label,
                [
                    _rewrite_instruction(inst, mapping)
                    for inst in block.instructions
                ],
            )
            for block in function.blocks
        ],
        set(function.frame_slots),
    )
    verify_p3(rewritten)
    return rewritten


def _rewrite_image_target(
    target: SymbolExpr | BlockExpr,
    mapping: dict[str, str],
):
    if isinstance(target, SymbolExpr):
        return SymbolExpr(
            mapping.get(target.symbol, target.symbol),
            target.addend,
        )
    return BlockExpr(
        mapping.get(target.function, target.function),
        target.label,
        target.addend,
    )


def _rewrite_image(
    image: ModuleImage | None,
    mapping: dict[str, str],
) -> ModuleImage | None:
    if image is None:
        return None
    return ModuleImage(
        objects=tuple(
            ImageObject(
                mapping.get(obj.name, obj.name),
                obj.ty,
                obj.data,
                obj.align,
                obj.section,
                obj.constant,
                tuple(
                    Relocation(
                        reloc.offset,
                        reloc.size,
                        _rewrite_image_target(reloc.target, mapping),
                    )
                    for reloc in obj.relocations
                ),
            )
            for obj in image.objects
        ),
        aliases=tuple(
            ImageAlias(
                mapping.get(alias.name, alias.name),
                SymbolExpr(
                    mapping.get(alias.target.symbol, alias.target.symbol),
                    alias.target.addend,
                ),
            )
            for alias in image.aliases
        ),
        external_data=tuple(
            mapping.get(name, name) for name in image.external_data
        ),
        external_functions=tuple(
            mapping.get(name, name) for name in image.external_functions
        ),
        skipped_linker_metadata=image.skipped_linker_metadata,
        undef_bytes=image.undef_bytes,
    )


def namespace_user_program(
    program: UserProgramImage,
    *,
    internal_prefix: str = "__mm_user_",
    external_prefix: str = "__mm_user_ext_",
) -> UserProgramImage:
    mapping: dict[str, str] = {}

    for function in program.functions:
        mapping[function.name] = internal_prefix + function.name

    if program.image is not None:
        for obj in program.image.objects:
            mapping.setdefault(obj.name, internal_prefix + obj.name)
        for alias in program.image.aliases:
            mapping.setdefault(alias.name, internal_prefix + alias.name)
        for name in program.image.external_data:
            mapping.setdefault(name, external_prefix + name)
        for name in program.image.external_functions:
            mapping.setdefault(name, external_prefix + name)

    return UserProgramImage(
        entry=mapping.get(program.entry, program.entry),
        functions=tuple(
            _rewrite_function(function, mapping)
            for function in program.functions
        ),
        image=_rewrite_image(program.image, mapping),
        entry_args=program.entry_args,
        runtime_helpers=program.runtime_helpers,
    )


def rebase_user_program_namespace(
    program: UserProgramImage,
    *,
    namespace: str,
) -> UserProgramImage:
    if (
        not namespace
        or not all(ch.isalnum() or ch == "_" for ch in namespace)
        or namespace[0].isdigit()
    ):
        raise ValueError(
            "userspace namespace must start with a letter/underscore and "
            "contain only letters, digits, or underscores"
        )
    if program.image is None:
        raise ValueError("cannot rebase userspace namespace without module image")

    external_prefixes: set[str] = set()
    for name in (
        *program.image.external_data,
        *program.image.external_functions,
    ):
        if not name.startswith("__mm_"):
            continue
        marker_at = name.find("_ext_", len("__mm_"))
        if marker_at >= 0:
            external_prefixes.add(name[: marker_at + len("_ext_")])
    if len(external_prefixes) != 1:
        raise ValueError(
            "rebasing requires exactly one existing userspace external namespace"
        )

    old_external = next(iter(external_prefixes))
    old_internal = old_external[:-len("ext_")]
    new_internal = f"__mm_{namespace}_"
    new_external = f"__mm_{namespace}_ext_"
    mapping: dict[str, str] = {}

    for function in program.functions:
        if function.name.startswith(old_internal):
            mapping[function.name] = (
                new_internal + function.name[len(old_internal):]
            )

    for obj in program.image.objects:
        if obj.name.startswith(old_internal):
            mapping[obj.name] = new_internal + obj.name[len(old_internal):]
    for alias in program.image.aliases:
        if alias.name.startswith(old_internal):
            mapping[alias.name] = (
                new_internal + alias.name[len(old_internal):]
            )
    for name in program.image.external_data:
        if name.startswith(old_external):
            mapping[name] = new_external + name[len(old_external):]
    for name in program.image.external_functions:
        if name.startswith(old_external):
            mapping[name] = new_external + name[len(old_external):]

    return UserProgramImage(
        entry=mapping.get(program.entry, program.entry),
        functions=tuple(
            _rewrite_function(function, mapping)
            for function in program.functions
        ),
        image=_rewrite_image(program.image, mapping),
        entry_args=program.entry_args,
        runtime_helpers=program.runtime_helpers,
    )


def build_user_program_from_llvm(
    text: str,
    *,
    entry: str,
    entry_args: str = "linux-main",
    namespace: str = "user",
) -> tuple[UserProgramImage, RuntimeSurface]:
    if (
        not namespace
        or not all(ch.isalnum() or ch == "_" for ch in namespace)
        or namespace[0].isdigit()
    ):
        raise ValueError(
            "userspace namespace must start with a letter/underscore and "
            "contain only letters, digits, or underscores"
        )

    functions, _stats = legalize_module(text)
    if not any(function.name == entry for function in functions):
        raise ValueError(f"userspace entry function is missing: {entry}")

    for function in functions:
        verify_muir(function)

    surface = collect_runtime_surface(functions)
    machine_functions = []
    for function in functions:
        expanded, _ = expand_function(function)
        machine = lower_function(expanded)
        verify_p3(machine)
        machine_functions.append(machine)

    image = parse_module_image(text)
    program = UserProgramImage(
        entry=entry,
        functions=tuple(machine_functions),
        image=image,
        entry_args=entry_args,
        runtime_helpers=tuple(sorted(surface.helpers)),
    )
    return namespace_user_program(
        program,
        internal_prefix=f"__mm_{namespace}_",
        external_prefix=f"__mm_{namespace}_ext_",
    ), surface
