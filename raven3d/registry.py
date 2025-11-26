from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np

from .rules.base import Rule, RuleDifficulty


class RuleRegistry:
    def __init__(self) -> None:
        self._rules: Dict[RuleDifficulty, List[Rule]] = defaultdict(list)

    def register(self, rule: Rule) -> None:
        self._rules[rule.difficulty].append(rule)

    def all_rules(self) -> List[Rule]:
        items: List[Rule] = []
        for lst in self._rules.values():
            items.extend(lst)
        return items

    def rules_by_difficulty(self, difficulty: RuleDifficulty) -> List[Rule]:
        return self._rules.get(difficulty, [])

    def sample_rule(self, difficulty: RuleDifficulty, rng: np.random.Generator) -> Rule:
        rules = self.rules_by_difficulty(difficulty)
        if not rules:
            raise ValueError(f"No rules registered for {difficulty}")
        idx = rng.integers(0, len(rules))
        return rules[idx]
