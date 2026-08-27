from __future__ import annotations

from dataclasses import dataclass
import re


class LayoutError(ValueError):
    pass


@dataclass(frozen=True)
class TypeInfo:
    size: int
    align: int
    fields: tuple[str, ...] | None = None
    field_offsets: tuple[int, ...] | None = None
    element: str | None = None
    count: int | None = None


def _round_up(value: int, align: int) -> int:
    if align <= 1:
        return value
    return (value + align - 1) // align * align


def _split_top(text: str) -> list[str]:
    out=[]
    start=0
    stack=[]
    pairs={"]":"[","}":"{",")":"("," >":"<"}
    opens=set("[{(<")
    close_to_open={"]":"[","}":"{",")":"(",">":"<"}
    for i,c in enumerate(text):
        if c in opens:
            stack.append(c)
        elif c in close_to_open:
            if stack and stack[-1]==close_to_open[c]:
                stack.pop()
        elif c=="," and not stack:
            out.append(text[start:i].strip())
            start=i+1
    tail=text[start:].strip()
    if tail:
        out.append(tail)
    return out


class DataLayout:
    """Small LLVM type-layout evaluator for the frozen RV64 corpus.

    It models ABI byte size/alignment for the integer/pointer/array/struct
    types needed by GEP legalization. It is intentionally not a replacement
    for LLVM's full DataLayout API; unknown type forms fail loudly.
    """

    def __init__(self, definitions: dict[str, str]):
        self.definitions=definitions
        self._cache: dict[str, TypeInfo]={}

    @classmethod
    def from_module(cls, text: str) -> "DataLayout":
        defs={}
        lines=text.splitlines()
        i=0
        while i < len(lines):
            line=lines[i].strip()
            m=re.match(r"^(%[-A-Za-z$._0-9]+)\s*=\s*type\s+(.+)$", line)
            if not m:
                i+=1
                continue
            name, body=m.groups()
            # Most llvm-dis type definitions are one line. Handle wrapped
            # aggregate definitions as well by balancing delimiters.
            acc=body
            def balance(s: str) -> int:
                return sum(s.count(x) for x in "[{(<") - sum(s.count(x) for x in "]})>")
            bal=balance(acc)
            while bal>0 and i+1 < len(lines):
                i+=1
                more=lines[i].strip()
                acc += " " + more
                bal=balance(acc)
            defs[name]=acc.strip()
            i+=1
        return cls(defs)

    def info(self, ty: str) -> TypeInfo:
        ty=ty.strip()
        if ty in self._cache:
            return self._cache[ty]

        # Named type.
        if ty.startswith("%"):
            body=self.definitions.get(ty)
            if body is None:
                raise LayoutError(f"unknown named LLVM type: {ty}")
            if body=="opaque":
                raise LayoutError(f"opaque LLVM type has no layout: {ty}")
            info=self.info(body)
            self._cache[ty]=info
            return info

        # Pointer (opaque-pointer LLVM).
        if re.fullmatch(r"ptr(?:\s+addrspace\(\d+\))?", ty):
            return TypeInfo(8,8)

        m=re.fullmatch(r"i(\d+)",ty)
        if m:
            bits=int(m.group(1))
            size=max(1,(bits+7)//8)
            if bits<=8: align=1
            elif bits<=16: align=2
            elif bits<=32: align=4
            elif bits<=64: align=8
            elif bits<=128: align=16
            else: align=min(16, 1 << (size-1).bit_length())
            return TypeInfo(size,align)

        if ty=="half":
            return TypeInfo(2,2)
        if ty=="float":
            return TypeInfo(4,4)
        if ty=="double":
            return TypeInfo(8,8)
        if ty=="fp128":
            return TypeInfo(16,16)

        # Array.
        m=re.fullmatch(r"\[(\d+)\s+x\s+(.+)\]",ty)
        if m:
            count=int(m.group(1)); elem_ty=m.group(2).strip()
            elem=self.info(elem_ty)
            stride=_round_up(elem.size,elem.align)
            return TypeInfo(stride*count,elem.align,element=elem_ty,count=count)

        # Fixed vector. This is enough for layout/GEP; vector operations are
        # still a separate semantic question.
        m=re.fullmatch(r"<(\d+)\s+x\s+(.+)>",ty)
        if m:
            count=int(m.group(1)); elem_ty=m.group(2).strip()
            elem=self.info(elem_ty)
            size=elem.size*count
            align=min(16, max(elem.align, 1 << (max(1,size)-1).bit_length()))
            return TypeInfo(size,align,element=elem_ty,count=count)

        packed=False
        body=None
        if ty.startswith("<{") and ty.endswith("}>"):
            packed=True
            body=ty[2:-2].strip()
        elif ty.startswith("{") and ty.endswith("}"):
            body=ty[1:-1].strip()
        if body is not None:
            fields=tuple(_split_top(body)) if body else ()
            offsets=[]
            cursor=0
            struct_align=1
            for field in fields:
                fi=self.info(field)
                if not packed:
                    cursor=_round_up(cursor,fi.align)
                    struct_align=max(struct_align,fi.align)
                offsets.append(cursor)
                cursor += fi.size
            if packed:
                struct_align=1
            size=_round_up(cursor,struct_align)
            return TypeInfo(size,struct_align,fields=fields,field_offsets=tuple(offsets))

        raise LayoutError(f"unsupported LLVM type layout: {ty}")

    def gep_step(self, current_ty: str, index: int | None):
        """Return (next_type, scale, constant_field_offset).

        For arrays/vectors, scale is the element allocation size and index may
        be dynamic. For structs, index must be constant and field offset is
        returned.
        """
        info=self.info(current_ty)
        if info.fields is not None:
            if index is None:
                raise LayoutError(f"dynamic struct index: {current_ty}")
            if index<0 or index>=len(info.fields):
                raise LayoutError(f"struct index {index} out of range: {current_ty}")
            assert info.field_offsets is not None
            return info.fields[index], 0, info.field_offsets[index]
        if info.element is not None:
            elem=self.info(info.element)
            scale=_round_up(elem.size,elem.align)
            return info.element,scale,0
        raise LayoutError(f"cannot index into scalar type: {current_ty}")
