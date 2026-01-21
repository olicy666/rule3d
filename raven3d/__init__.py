"""
SPIRAL3D:Structured Perception to Intelligent Reasoning And Logic in 3D
dataset generator package.

The package exposes primitives, scenes, rules, and a dataset generator used
to build three-step point cloud reasoning samples as described in program.md.
"""

from .geometry import (
    Primitive,
    Sphere,
    Cube,
    Cylinder,
    Cone,
    TriangularPrism,
    Capsule,
    Torus,
    rotation_matrix,
)
from .scene import Scene, ObjectState
from .rules.base import RuleDifficulty, Rule
from .registry import RuleRegistry
from .dataset import DatasetGenerator, GenerationConfig

__all__ = [
    "Primitive",
    "Sphere",
    "Cube",
    "Cylinder",
    "Cone",
    "TriangularPrism",
    "Capsule",
    "Torus",
    "Scene",
    "ObjectState",
    "RuleDifficulty",
    "Rule",
    "RuleRegistry",
    "DatasetGenerator",
    "GenerationConfig",
    "rotation_matrix",
]
