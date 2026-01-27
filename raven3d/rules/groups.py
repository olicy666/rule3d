from __future__ import annotations

from typing import Dict, Iterable, List, Set

# 大类划分
R1: Set[str] = {
    "R1-1",
    "R1-2",
    "R1-3",
    "R1-4",
    "R1-5",
    "R1-6",
    "R1-7",
    "R1-8",
    "R1-9",
    "R1-10",
    "R1-11",
    "R1-12",
    "R1-13",
    "R1-14",
    "R1-15",
    "R1-16",
}
R2: Set[str] = {
    "R2-1",
    "R2-2",
    "R2-3",
    "R2-4",
    "R2-5",
    "R2-6",
    "R2-7",
    "R2-8",
    "R2-9",
    "R2-10",
    "R2-11",
    "R2-12",
}
R3: Set[str] = {
    "R3-1",
    "R3-2",
    "R3-3",
    "R3-5",
    "R3-6",
    "R3-7",
    "R3-8",
    "R3-9",
    "R3-10",
    "R3-11",
}
R4: Set[str] = {
    "R4-1",
    "R4-2",
    "R4-3",
    "R4-4",
    "R4-5",
    "R4-6",
    "R4-7",
    "R4-8",
    "R4-9",
    "R4-10",
}
# 子类划分
ALL_RULES: Set[str] = set().union(R1, R2, R3, R4)

# 预设模式 -> 可选规则集合
MODE_TO_RULES: Dict[str, Set[str]] = {
    "main": ALL_RULES,
    "r1-only": R1,
    "r2-only": R2,
    "r3-only": R3,
    "r4-only": R4,
    "all-minus-r1": ALL_RULES - R1,
    "all-minus-r2": ALL_RULES - R2,
    "all-minus-r3": ALL_RULES - R3,
    "all-minus-r4": ALL_RULES - R4,
}


def list_all_rules() -> List[str]:
    """Return all supported rule IDs sorted by group and index."""
    def key(rule_id: str) -> tuple[int, int, str]:
        if rule_id.startswith("R") and "-" in rule_id:
            group_part, _, idx_part = rule_id.partition("-")
            group_num = group_part[1:]
            if group_num.isdigit() and idx_part.isdigit():
                return (int(group_num), int(idx_part), rule_id)
        return (99, 999, rule_id)

    return sorted(ALL_RULES, key=key)


def list_available_modes() -> List[str]:
    """Return supported mode keys for argparse choices."""
    return sorted(MODE_TO_RULES.keys())


def rules_for_mode(mode: str) -> Set[str]:
    key = mode.lower()
    if key not in MODE_TO_RULES:
        raise ValueError(f"Unsupported mode '{mode}'. Allowed: {', '.join(list_available_modes())}")
    return MODE_TO_RULES[key]


def validate_rule_ids(rule_ids: Iterable[str]) -> Set[str]:
    """Normalize and validate custom rule IDs."""
    normalized = {rid.strip().upper() for rid in rule_ids if rid and rid.strip()}
    if not normalized:
        raise ValueError("Custom rule list is empty.")
    unknown = normalized - ALL_RULES
    if unknown:
        raise ValueError(f"Unsupported rule ids: {', '.join(sorted(unknown))}. Allowed: {', '.join(list_all_rules())}")
    return normalized
