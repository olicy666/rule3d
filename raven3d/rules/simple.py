from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .base import Rule, RuleDifficulty
from .utils import (
    apply_rotation,
    apply_scale,
    apply_translation,
    primitive_from_config,
    primitive_to_config,
    random_primitive,
)
from ..scene import Scene
from ..geometry import rotation_matrix


@dataclass
class S01ScaleLinear(Rule):
    def __init__(self) -> None:
        super().__init__("S01", RuleDifficulty.SIMPLE, "Scale Linear", "B = k*A, C = k*B")

    def sample_params(self, rng) -> Dict:
        return {"k": float(rng.uniform(0.7, 1.4)), "base": primitive_to_config(random_primitive(rng))}

    def generate_triplet(self, params, rng):
        base = primitive_from_config(params["base"])
        k = params["k"]
        b = apply_scale(base, k)
        c = apply_scale(b, k)
        return Scene([base]), Scene([b]), Scene([c]), {"k": k, "base": params["base"]}


@dataclass
class S02SingleAxisStretch(Rule):
    def __init__(self) -> None:
        super().__init__("S02", RuleDifficulty.SIMPLE, "Single Axis Stretch", "Scale along one axis only")

    def sample_params(self, rng) -> Dict:
        axis = int(rng.integers(0, 3))
        factor = float(rng.uniform(1.2, 1.8))
        return {"axis": axis, "factor": factor, "base": primitive_to_config(random_primitive(rng))}

    def generate_triplet(self, params, rng):
        axis = params["axis"]
        factor = params["factor"]
        base = primitive_from_config(params["base"])
        scale_vec = [1.0, 1.0, 1.0]
        scale_vec[axis] = factor
        b = apply_scale(base, tuple(scale_vec))
        c = apply_scale(b, tuple(scale_vec))
        return Scene([base]), Scene([b]), Scene([c]), params


@dataclass
class S03UniformScaling(Rule):
    def __init__(self) -> None:
        super().__init__("S03", RuleDifficulty.SIMPLE, "Uniform Scaling", "Global uniform scaling A->B->C")

    def sample_params(self, rng) -> Dict:
        k = float(rng.uniform(0.8, 1.5))
        return {"k": k, "base": primitive_to_config(random_primitive(rng))}

    def generate_triplet(self, params, rng):
        base = primitive_from_config(params["base"])
        k = params["k"]
        b = apply_scale(base, k)
        c = apply_scale(b, k)
        return Scene([base]), Scene([b]), Scene([c]), params


@dataclass
class S04FixedAxisRotation(Rule):
    def __init__(self) -> None:
        super().__init__("S04", RuleDifficulty.SIMPLE, "Fixed Axis Rotation", "Rotate around single axis by Δθ")

    def sample_params(self, rng) -> Dict:
        axis = int(rng.integers(0, 3))
        delta = float(rng.uniform(math.pi / 12, math.pi / 6))
        return {"axis": axis, "delta": delta, "base": primitive_to_config(random_primitive(rng))}

    def generate_triplet(self, params, rng):
        axis = params["axis"]
        delta = params["delta"]
        base = primitive_from_config(params["base"])
        rot_vec = np.zeros(3)
        rot_vec[axis] = delta
        b = apply_rotation(base, rot_vec)
        c = apply_rotation(b, rot_vec)
        return Scene([base]), Scene([b]), Scene([c]), params


@dataclass
class S05FullEulerRotation(Rule):
    def __init__(self) -> None:
        super().__init__("S05", RuleDifficulty.SIMPLE, "Full Euler Rotation", "Increment all Euler angles")

    def sample_params(self, rng) -> Dict:
        delta = rng.uniform(math.pi / 18, math.pi / 8, size=3)
        return {"delta": delta.tolist(), "base": primitive_to_config(random_primitive(rng))}

    def generate_triplet(self, params, rng):
        base = primitive_from_config(params["base"])
        delta = np.array(params["delta"])
        b = apply_rotation(base, delta)
        c = apply_rotation(b, delta)
        return Scene([base]), Scene([b]), Scene([c]), params


@dataclass
class S06Translation(Rule):
    def __init__(self) -> None:
        super().__init__("S06", RuleDifficulty.SIMPLE, "Translation", "Translate by constant vector d")

    def sample_params(self, rng) -> Dict:
        delta = rng.uniform(0.2, 0.6, size=3) * rng.choice([-1, 1], size=3)
        return {"delta": delta.tolist(), "base": primitive_to_config(random_primitive(rng))}

    def generate_triplet(self, params, rng):
        base = primitive_from_config(params["base"])
        delta = np.array(params["delta"])
        b = apply_translation(base, delta)
        c = apply_translation(b, delta)
        return Scene([base]), Scene([b]), Scene([c]), params


@dataclass
class S07TwoStepTranslation(Rule):
    def __init__(self) -> None:
        super().__init__("S07", RuleDifficulty.SIMPLE, "Two-step Translation", "Two-step translation with scaled distance")

    def sample_params(self, rng) -> Dict:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction) + 1e-9
        d1 = float(rng.uniform(0.2, 0.6))
        ratio = float(rng.uniform(0.5, 1.5))
        return {
            "direction": direction.tolist(),
            "d1": d1,
            "ratio": ratio,
            "base": primitive_to_config(random_primitive(rng)),
        }

    def generate_triplet(self, params, rng):
        base = primitive_from_config(params["base"])
        direction = np.array(params["direction"])
        d1 = params["d1"]
        ratio = params["ratio"]
        delta1 = direction * d1
        delta2 = direction * d1 * ratio
        b = apply_translation(base, delta1)
        c = apply_translation(b, delta2)
        return Scene([base]), Scene([b]), Scene([c]), params


@dataclass
class S08ShapeSubstitution(Rule):
    def __init__(self) -> None:
        super().__init__("S08", RuleDifficulty.SIMPLE, "Shape Substitution", "Cycle shapes across steps")

    def sample_params(self, rng) -> Dict:
        center = rng.uniform(-0.3, 0.3, size=3)
        rotation = rng.uniform(-math.pi / 8, math.pi / 8, size=3)
        scale = rng.uniform(0.8, 1.2, size=3)
        return {"center": center.tolist(), "rotation": rotation.tolist(), "scale": scale.tolist()}

    def generate_triplet(self, params, rng):
        center = np.array(params["center"])
        rotation = np.array(params["rotation"])
        scale = np.array(params["scale"])
        shapes = ["sphere", "cylinder", "cone", "cube"]
        start_idx = int(rng.integers(0, len(shapes)))

        def make_shape(name: str):
            if name == "sphere":
                from ..geometry import Sphere

                return Sphere(center=center, rotation=rotation, scale=scale, radius=0.45)
            if name == "cylinder":
                from ..geometry import Cylinder

                return Cylinder(center=center, rotation=rotation, scale=scale, radius=0.35, height=0.9)
            if name == "cone":
                from ..geometry import Cone

                return Cone(center=center, rotation=rotation, scale=scale, radius=0.35, height=1.0)
            from ..geometry import Cube

            return Cube(center=center, rotation=rotation, scale=scale, edge_lengths=np.array([0.8, 0.8, 0.8]))

        a_shape = shapes[start_idx % len(shapes)]
        b_shape = shapes[(start_idx + 1) % len(shapes)]
        c_shape = shapes[(start_idx + 2) % len(shapes)]
        a = make_shape(a_shape)
        b = make_shape(b_shape)
        c = make_shape(c_shape)
        return (
            Scene([a]),
            Scene([b]),
            Scene([c]),
            {"sequence": [a_shape, b_shape, c_shape], "center": params["center"], "rotation": params["rotation"]},
        )


@dataclass
class S09ShapeConstantScaleChange(Rule):
    def __init__(self) -> None:
        super().__init__("S09", RuleDifficulty.SIMPLE, "Shape Constant Scale Change", "Shape fixed, per-axis scale changes")

    def sample_params(self, rng) -> Dict:
        base = primitive_to_config(random_primitive(rng))
        delta = rng.uniform(0.8, 1.4, size=3)
        return {"base": base, "delta": delta.tolist()}

    def generate_triplet(self, params, rng):
        base = primitive_from_config(params["base"])
        delta = np.array(params["delta"])
        b = apply_scale(base, tuple(delta))
        c = apply_scale(b, tuple(delta))
        return Scene([base]), Scene([b]), Scene([c]), params


@dataclass
class S10PointDensityChange(Rule):
    def __init__(self) -> None:
        super().__init__("S10", RuleDifficulty.SIMPLE, "Point Density Change", "Apparent density increases")

    def sample_params(self, rng) -> Dict:
        base = primitive_to_config(random_primitive(rng))
        factors = sorted(rng.uniform(0.7, 1.2, size=3), reverse=True)
        return {"base": base, "scale_factors": factors}

    def generate_triplet(self, params, rng):
        base = primitive_from_config(params["base"])
        f1, f2, f3 = params["scale_factors"]
        # Larger scale -> lower apparent density, smaller -> higher density.
        a = apply_scale(base, f1)
        b = apply_scale(base, f2)
        c = apply_scale(base, f3)
        return Scene([a]), Scene([b]), Scene([c]), params


@dataclass
class S11PoseFixedSizeVertical(Rule):
    def __init__(self) -> None:
        super().__init__("S11", RuleDifficulty.SIMPLE, "Pose Fixed Vertical Growth", "Keep pose, grow size along vertical axis")

    def sample_params(self, rng) -> Dict:
        base = primitive_to_config(random_primitive(rng))
        delta = float(rng.uniform(1.2, 1.6))
        return {"base": base, "vertical_factor": delta}

    def generate_triplet(self, params, rng):
        base = primitive_from_config(params["base"])
        factor = params["vertical_factor"]
        a = base
        b = apply_scale(base, (1.0, 1.0, factor))
        c = apply_scale(b, (1.0, 1.0, factor))
        return Scene([a]), Scene([b]), Scene([c]), params


@dataclass
class S12DistanceScaleMatched(Rule):
    def __init__(self) -> None:
        super().__init__("S12", RuleDifficulty.SIMPLE, "Distance-Scale Matched", "Scale while compensating translation to keep centroid")

    def sample_params(self, rng) -> Dict:
        base = primitive_to_config(random_primitive(rng))
        factor = float(rng.uniform(1.2, 1.8))
        return {"base": base, "factor": factor}

    def generate_triplet(self, params, rng):
        base = primitive_from_config(params["base"])
        factor = params["factor"]
        centroid = base.center
        a = base
        b = apply_scale(base, factor)
        # Translate back so centroid stays close.
        delta_b = centroid - rotation_matrix(base.rotation) @ (centroid * factor)
        b.center = b.center + delta_b
        c = apply_scale(b, factor)
        delta_c = centroid - rotation_matrix(b.rotation) @ (centroid * factor)
        c.center = c.center + delta_c
        return Scene([a]), Scene([b]), Scene([c]), params


@dataclass
class S13SinglePrimitiveRotationFixedPoint(Rule):
    def __init__(self) -> None:
        super().__init__("S13", RuleDifficulty.SIMPLE, "Rotation Around Fixed Point", "Rotate around a fixed pivot")

    def sample_params(self, rng) -> Dict:
        base = primitive_to_config(random_primitive(rng))
        pivot = rng.uniform(-0.2, 0.2, size=3)
        delta = rng.uniform(math.pi / 18, math.pi / 10, size=3)
        return {"base": base, "pivot": pivot.tolist(), "delta": delta.tolist()}

    def generate_triplet(self, params, rng):
        base = primitive_from_config(params["base"])
        pivot = np.array(params["pivot"])
        delta = np.array(params["delta"])
        a = base
        b = apply_rotation(base, delta)
        c = apply_rotation(b, delta)
        # Recompute centers so pivot stays fixed.
        def rotate_center(prim, total_delta):
            rot = rotation_matrix(total_delta)
            offset = prim.center - pivot
            return pivot + (rot @ offset)

        b.center = rotate_center(base, delta)
        c.center = rotate_center(b, delta)
        return Scene([a]), Scene([b]), Scene([c]), params


@dataclass
class S14IdentityRule(Rule):
    def __init__(self) -> None:
        super().__init__("S14", RuleDifficulty.SIMPLE, "Identity", "A, B, C identical")

    def sample_params(self, rng) -> Dict:
        return {"base": primitive_to_config(random_primitive(rng))}

    def generate_triplet(self, params, rng):
        base = primitive_from_config(params["base"])
        return Scene([base]), Scene([base]), Scene([base]), params


def build_simple_rules():
    return [
        S01ScaleLinear(),
        S02SingleAxisStretch(),
        S03UniformScaling(),
        S04FixedAxisRotation(),
        S05FullEulerRotation(),
        S06Translation(),
        S07TwoStepTranslation(),
        S08ShapeSubstitution(),
        S09ShapeConstantScaleChange(),
        S10PointDensityChange(),
        S11PoseFixedSizeVertical(),
        S12DistanceScaleMatched(),
        S13SinglePrimitiveRotationFixedPoint(),
        S14IdentityRule(),
    ]
