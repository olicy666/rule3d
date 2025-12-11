from __future__ import annotations

from typing import Dict, List, Set

# 大类划分
R1: Set[str] = {
    "S01",
    "S02",
    "S03",
    "S04",
    "S05",
    "S06",
    "S07",
    "S08",
    "S09",
    "S10",
    "S11",
    "S12",
    "S13",
    "S14",
    "M14",
    "C01",
    "C02",
    "C03",
}
R2: Set[str] = {
    "M01",
    "M02",
    "M03",
    "M04",
    "M05",
    "M06",
    "M07",
    "M09",
    "C10",
    "C11",
}
R3: Set[str] = {
    "M08",
    "M10",
    "M11",
    "M12",
    "M13",
    "C08",
    "C09",
    "C12",
}
R4: Set[str] = {"C04", "C05", "C06", "C07"}

# 子类划分
R1_SUB: Dict[str, Set[str]] = {
    "r1-1": {"S01", "S02", "S03", "S09", "S12"},
    "r1-2": {"S04", "S05", "S06", "S07", "S13"},
    "r1-3": {"S08", "S10", "S11", "S14", "M14"},
    "r1-4": {"C01", "C02", "C03"},
}

R2_SUB: Dict[str, Set[str]] = {
    "r2-1": {"M01", "M02", "M03", "C10", "C11"},
    "r2-2": {"M04", "M05", "M06", "M07", "M09"},
}

R3_SUB: Dict[str, Set[str]] = {
    "r3-1": {"M08", "C08", "C09"},
    "r3-2": {"M10", "M11", "M12", "M13", "C12"},
}

ALL_RULES: Set[str] = set().union(R1, R2, R3, R4)

# 预设模式 -> 可选规则集合
MODE_TO_RULES: Dict[str, Set[str]] = {
    "main": ALL_RULES,
    "r1-only": R1,
    "r2-only": R2,
    "r3-only": R3,
    "r4-only": R4,
    **R1_SUB,
    **R2_SUB,
    **R3_SUB,
    "all-minus-r1": ALL_RULES - R1,
    "all-minus-r2": ALL_RULES - R2,
    "all-minus-r3": ALL_RULES - R3,
    "all-minus-r4": ALL_RULES - R4,
}


def list_available_modes() -> List[str]:
    """Return supported mode keys for argparse choices."""
    return sorted(MODE_TO_RULES.keys())


def rules_for_mode(mode: str) -> Set[str]:
    key = mode.lower()
    if key not in MODE_TO_RULES:
        raise ValueError(f"Unsupported mode '{mode}'. Allowed: {', '.join(list_available_modes())}")
    return MODE_TO_RULES[key]
