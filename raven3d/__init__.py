"""
Raven3D dataset generator package.

The package exposes primitives, scenes, rules, and a dataset generator used
to build three-step point cloud reasoning samples as described in program.md.
"""

from .geometry import (
    Primitive,
    Sphere,
    Cube,
    Cylinder,
    Cone,
    rotation_matrix,
)
from .scene import Scene
from .rules.base import RuleDifficulty, Rule
from .registry import RuleRegistry
from .dataset import DatasetGenerator, GenerationConfig

__all__ = [
    "Primitive",
    "Sphere",
    "Cube",
    "Cylinder",
    "Cone",
    "Scene",
    "RuleDifficulty",
    "Rule",
    "RuleRegistry",
    "DatasetGenerator",
    "GenerationConfig",
    "rotation_matrix",
]
