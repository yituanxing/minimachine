from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


class LinkerContractError(ValueError):
    pass


@dataclass(frozen=True)
class SectionGroup:
    name: str
    sections: tuple[str, ...]
    align: int = 1
    empty_after: str | None = None


@dataclass(frozen=True)
class BoundarySymbol:
    symbol: str
    group: str
    edge: str  # "start" or "end"


@dataclass(frozen=True)
class SemanticBoundary:
    symbol: str
    point: str  # code_start/code_end/data_start/data_end


@dataclass(frozen=True)
class LinkerContract:
    aliases: dict[str, str]
    groups: tuple[SectionGroup, ...]
    boundaries: tuple[BoundarySymbol, ...]
    semantic_boundaries: tuple[SemanticBoundary, ...] = ()

    def __post_init__(self) -> None:
        group_names: set[str] = set()
        claimed_sections: dict[str, str] = {}

        for group in self.groups:
            if not group.name:
                raise LinkerContractError("section group requires a name")
            if group.name in group_names:
                raise LinkerContractError(
                    f"duplicate section group: {group.name}"
                )
            group_names.add(group.name)
            if group.align <= 0 or group.align & (group.align - 1):
                raise LinkerContractError(
                    f"section group alignment must be power of two: "
                    f"{group.name}={group.align}"
                )
            for section in group.sections:
                if not section:
                    raise LinkerContractError(
                        f"empty section selector in group {group.name}"
                    )
                owner = claimed_sections.get(section)
                if owner is not None:
                    raise LinkerContractError(
                        f"section {section} belongs to both "
                        f"{owner} and {group.name}"
                    )
                claimed_sections[section] = group.name

        group_order = {group.name: i for i, group in enumerate(self.groups)}
        for group in self.groups:
            if group.empty_after is None:
                continue
            if group.empty_after not in group_names:
                raise LinkerContractError(
                    f"section group {group.name} anchors after unknown group "
                    f"{group.empty_after}"
                )
            if group_order[group.empty_after] >= group_order[group.name]:
                raise LinkerContractError(
                    f"section group {group.name} must anchor after an earlier group"
                )

        boundary_names: set[str] = set()
        for boundary in self.boundaries:
            if not boundary.symbol:
                raise LinkerContractError("boundary symbol requires a name")
            if boundary.symbol in boundary_names:
                raise LinkerContractError(
                    f"duplicate boundary symbol: {boundary.symbol}"
                )
            boundary_names.add(boundary.symbol)
            if boundary.group not in group_names:
                raise LinkerContractError(
                    f"boundary {boundary.symbol} references unknown group "
                    f"{boundary.group}"
                )
            if boundary.edge not in {"start", "end"}:
                raise LinkerContractError(
                    f"boundary {boundary.symbol} has invalid edge "
                    f"{boundary.edge}"
                )

        semantic_names: set[str] = set()
        valid_points = {"code_start", "code_end", "data_start", "data_end"}
        for boundary in self.semantic_boundaries:
            if not boundary.symbol:
                raise LinkerContractError("semantic boundary requires a symbol")
            if boundary.symbol in semantic_names:
                raise LinkerContractError(
                    f"duplicate semantic boundary symbol: {boundary.symbol}"
                )
            semantic_names.add(boundary.symbol)
            if boundary.point not in valid_points:
                raise LinkerContractError(
                    f"semantic boundary {boundary.symbol} has invalid point "
                    f"{boundary.point}"
                )

        duplicate_names = boundary_names.intersection(semantic_names)
        if duplicate_names:
            raise LinkerContractError(
                "symbol is both section and semantic boundary: "
                + ", ".join(sorted(duplicate_names))
            )

        overlap = boundary_names.union(semantic_names).intersection(self.aliases)
        if overlap:
            raise LinkerContractError(
                "symbol is both alias and boundary: "
                + ", ".join(sorted(overlap))
            )
        for name, target in self.aliases.items():
            if not name or not target:
                raise LinkerContractError("linker aliases require name and target")

    @classmethod
    def from_dict(cls, raw: dict) -> "LinkerContract":
        aliases = {
            str(name): str(target)
            for name, target in raw.get("aliases", {}).items()
        }
        groups = tuple(
            SectionGroup(
                name=str(item["name"]),
                sections=tuple(str(x) for x in item.get("sections", ())),
                align=int(item.get("align", 1)),
                empty_after=(
                    str(item["empty_after"])
                    if item.get("empty_after") is not None
                    else None
                ),
            )
            for item in raw.get("groups", ())
        )
        boundaries = tuple(
            BoundarySymbol(
                symbol=str(item["symbol"]),
                group=str(item["group"]),
                edge=str(item["edge"]),
            )
            for item in raw.get("boundaries", ())
        )
        semantic_boundaries = tuple(
            SemanticBoundary(
                symbol=str(item["symbol"]),
                point=str(item["point"]),
            )
            for item in raw.get("semantic_boundaries", ())
        )
        return cls(aliases, groups, boundaries, semantic_boundaries)

    @classmethod
    def load(cls, path: str | Path) -> "LinkerContract":
        source = Path(path)
        try:
            raw = json.loads(source.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise LinkerContractError(
                f"cannot read linker contract {source}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise LinkerContractError("linker contract root must be an object")
        return cls.from_dict(raw)

    def active_groups(self, sections: set[str]) -> set[str]:
        active: set[str] = set()
        for group in self.groups:
            has_members = any(
                section in sections for section in group.sections
            )
            anchored_empty = (
                not has_members
                and group.empty_after is not None
                and group.empty_after in active
            )
            if has_members or anchored_empty:
                active.add(group.name)
        return active

    def active_boundary_symbols(self, sections: set[str]) -> set[str]:
        active = self.active_groups(sections)
        return {
            boundary.symbol
            for boundary in self.boundaries
            if boundary.group in active
        }

    def defined_symbols(self, sections: set[str]) -> set[str]:
        return (
            set(self.aliases)
            .union(self.active_boundary_symbols(sections))
            .union(boundary.symbol for boundary in self.semantic_boundaries)
        )
