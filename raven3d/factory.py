from __future__ import annotations

from .registry import RuleRegistry
from .rules import build_simple_rules, build_medium_rules, build_complex_rules


def create_default_registry() -> RuleRegistry:
    registry = RuleRegistry()
    for rule in build_simple_rules() + build_medium_rules() + build_complex_rules():
        registry.register(rule)
    return registry
