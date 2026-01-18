from __future__ import annotations

from typing import Dict, Iterable, List, Set

# 大类划分（R4 已移除）
R1: Set[str] = {
    "S02",
    "S04",
    "S05",
    "S06",
    "S07",
    "S09",
    "S12",
    "S13",
    "S14",
    "M14",
    "C01",
}
R2: Set[str] = {
    "M02",
    "M03",
    "M04",
    "M06",
    "M07",
    "M09",
    "C10",
    "C11",
}
R3: Set[str] = {
    "M08",
    "M10",
    "M12",
    "C08",
    "C09",
    "C12",
}

# 子类划分
ALL_RULES: Set[str] = set().union(R1, R2, R3)

# 预设模式 -> 可选规则集合
MODE_TO_RULES: Dict[str, Set[str]] = {
    "main": ALL_RULES,
    "r1-only": R1,
    "r2-only": R2,
    "r3-only": R3,
    "all-minus-r1": ALL_RULES - R1,
    "all-minus-r2": ALL_RULES - R2,
    "all-minus-r3": ALL_RULES - R3,
}


def list_all_rules() -> List[str]:
    """Return all supported rule IDs sorted alphabetically."""
    return sorted(ALL_RULES)


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
