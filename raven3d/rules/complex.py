from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

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
from ..geometry import Cube, Cylinder, Sphere, Cone, rotation_matrix
from ..scene import Scene


def unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / (np.linalg.norm(v) + 1e-9)


@dataclass
class C01ScaleTranslationCoupled(Rule):
    def __init__(self) -> None:
        super().__init__("C01", RuleDifficulty.COMPLEX, "Scale + Translation Coupled", "Scaling linked with proportional translation")

    def sample_params(self, rng) -> Dict:
        factor = float(rng.uniform(1.1, 1.6))
        direction = unit_vec(rng)
        return {"factor": factor, "direction": direction.tolist(), "base": primitive_to_config(random_primitive(rng))}

    def generate_triplet(self, params, rng):
        base = primitive_from_config(params["base"])
        factor = params["factor"]
        direction = np.array(params["direction"])
        delta = direction * 0.4
        b = apply_translation(apply_scale(base, factor), delta * factor)
        c = apply_translation(apply_scale(b, factor), delta * factor)
        return Scene([base]), Scene([b]), Scene([c]), params


@dataclass
class C02RotationScalingCoupled(Rule):
    def __init__(self) -> None:
        super().__init__("C02", RuleDifficulty.COMPLEX, "Rotation + Scaling Coupled", "Rotation increases with size")

    def sample_params(self, rng) -> Dict:
        factor = float(rng.uniform(1.1, 1.4))
        delta_rot = rng.uniform(math.pi / 20, math.pi / 10, size=3)
        return {"factor": factor, "delta_rot": delta_rot.tolist(), "base": primitive_to_config(random_primitive(rng))}

    def generate_triplet(self, params, rng):
        base = primitive_from_config(params["base"])
        factor = params["factor"]
        delta_rot = np.array(params["delta_rot"])
        b = apply_scale(apply_rotation(base, delta_rot), factor)
        c = apply_scale(apply_rotation(b, delta_rot), factor)
        return Scene([base]), Scene([b]), Scene([c]), params


@dataclass
class C03MultiStepSequence(Rule):
    def __init__(self) -> None:
        super().__init__("C03", RuleDifficulty.COMPLEX, "Multi-step Sequence", "Repeated function across steps")

    def sample_params(self, rng) -> Dict:
        scale = float(rng.uniform(1.05, 1.25))
        translation = rng.uniform(0.1, 0.3, size=3) * rng.choice([-1, 1], size=3)
        rotation = rng.uniform(math.pi / 30, math.pi / 15, size=3)
        base = primitive_to_config(random_primitive(rng))
        return {"scale": scale, "translation": translation.tolist(), "rotation": rotation.tolist(), "base": base}

    def generate_triplet(self, params, rng):
        base = primitive_from_config(params["base"])
        scale = params["scale"]
        translation = np.array(params["translation"])
        rotation = np.array(params["rotation"])
        a = base
        b = apply_translation(apply_rotation(apply_scale(a, scale), rotation), translation)
        c = apply_translation(apply_rotation(apply_scale(b, scale), rotation), translation)
        return Scene([a]), Scene([b]), Scene([c]), params


@dataclass
class C04BooleanSequence(Rule):
    def __init__(self) -> None:
        super().__init__("C04", RuleDifficulty.COMPLEX, "Boolean Sequence", "Objects move from separate to deep intersection")

    def sample_params(self, rng) -> Dict:
        offset = float(rng.uniform(0.6, 1.0))
        return {"offset": offset}

    def generate_triplet(self, params, rng):
        offset = params["offset"]
        cube = Cube(center=np.array([-offset, 0, 0]), edge_lengths=np.array([0.9, 0.9, 0.9]))
        sphere = Sphere(center=np.array([offset, 0, 0]), radius=0.55)
        delta = np.array([offset * 0.8, 0, 0])
        a = Scene([cube, sphere])
        b = Scene([apply_translation(cube, delta * 0.6), apply_translation(sphere, -delta * 0.6)])
        c = Scene([apply_translation(cube, delta), apply_translation(sphere, -delta)])
        return a, b, c, params


@dataclass
class C05HoleTopology(Rule):
    def __init__(self) -> None:
        super().__init__("C05", RuleDifficulty.COMPLEX, "Hole Topology", "Cylinder punches through cube forming hole")

    def sample_params(self, rng) -> Dict:
        radius = float(rng.uniform(0.15, 0.3))
        shift = float(rng.uniform(0.0, 0.3))
        return {"radius": radius, "shift": shift}

    def generate_triplet(self, params, rng):
        radius = params["radius"]
        shift = params["shift"]
        cube = Cube(center=np.zeros(3), edge_lengths=np.array([1.0, 1.0, 1.0]))
        cylinder_a = Cylinder(center=np.array([shift, 0, 0]), radius=radius, height=1.3, rotation=np.array([0, math.pi / 2, 0]))
        cylinder_b = Cylinder(center=np.array([shift, 0, 0]), radius=radius * 1.3, height=1.3, rotation=np.array([0, math.pi / 2, 0]))
        cylinder_c = Cylinder(center=np.array([shift + 0.15, 0, 0]), radius=radius * 1.6, height=1.3, rotation=np.array([0, math.pi / 2, 0]))
        return Scene([cube]), Scene([cube, cylinder_a]), Scene([cube, cylinder_c]), params


@dataclass
class C06Tunnel(Rule):
    def __init__(self) -> None:
        super().__init__("C06", RuleDifficulty.COMPLEX, "Tunnel", "Two cylinders pass through cube with changing layout")

    def sample_params(self, rng) -> Dict:
        angle = float(rng.uniform(0, math.pi / 8))
        return {"angle": angle}

    def generate_triplet(self, params, rng):
        angle = params["angle"]
        cube = Cube(center=np.zeros(3), edge_lengths=np.array([1.2, 1.2, 1.2]))
        cyl1 = Cylinder(center=np.array([-0.2, 0, 0]), radius=0.2, height=1.4, rotation=np.array([0, math.pi / 2, 0]))
        cyl2 = Cylinder(center=np.array([0.2, 0, 0]), radius=0.2, height=1.4, rotation=np.array([angle, 0, math.pi / 2]))
        b_cyl2 = apply_rotation(cyl2, np.array([angle, 0, 0]))
        c_cyl2 = apply_rotation(b_cyl2, np.array([angle, 0, 0]))
        return Scene([cube, cyl1, cyl2]), Scene([cube, cyl1, b_cyl2]), Scene([cube, cyl1, c_cyl2]), params


@dataclass
class C07CrossSection(Rule):
    def __init__(self) -> None:
        super().__init__("C07", RuleDifficulty.COMPLEX, "Cross-section", "Intersection cross-section becomes clearer")

    def sample_params(self, rng) -> Dict:
        return {"offset": float(rng.uniform(0.2, 0.6))}

    def generate_triplet(self, params, rng):
        offset = params["offset"]
        cube = Cube(center=np.zeros(3), edge_lengths=np.array([1.0, 1.0, 1.0]))
        cyl_a = Cylinder(center=np.array([0, 0, offset]), radius=0.35, height=1.4)
        cyl_b = Cylinder(center=np.array([0, 0, offset * 0.4]), radius=0.35, height=1.4)
        cyl_c = Cylinder(center=np.array([0, 0, 0]), radius=0.35, height=1.4)
        return Scene([cube, cyl_a]), Scene([cube, cyl_b]), Scene([cube, cyl_c]), params


@dataclass
class C08SymmetryScale(Rule):
    def __init__(self) -> None:
        super().__init__("C08", RuleDifficulty.COMPLEX, "Symmetry + Scale", "Mirror objects scale together")

    def sample_params(self, rng) -> Dict:
        factor = float(rng.uniform(1.1, 1.5))
        distance = float(rng.uniform(0.5, 1.0))
        return {"factor": factor, "distance": distance}

    def generate_triplet(self, params, rng):
        factor = params["factor"]
        distance = params["distance"]
        left = Cone(center=np.array([-distance, 0, 0]), radius=0.3, height=1.0)
        right = Cone(center=np.array([distance, 0, 0]), radius=0.3, height=1.0, rotation=np.array([0, 0, math.pi]))
        a = Scene([left, right])
        b = Scene([apply_scale(left, factor), apply_scale(right, factor)])
        c = Scene([apply_scale(left, factor * factor), apply_scale(right, factor * factor)])
        return a, b, c, params


@dataclass
class C09RoleExchange(Rule):
    def __init__(self) -> None:
        super().__init__("C09", RuleDifficulty.COMPLEX, "Role Exchange", "Objects swap positions, sizes, and orientations")

    def sample_params(self, rng) -> Dict:
        obj1 = primitive_to_config(random_primitive(rng))
        obj2 = primitive_to_config(random_primitive(rng))
        mid = rng.uniform(-0.3, 0.3, size=3).tolist()
        return {"obj1": obj1, "obj2": obj2, "midpoint": mid}

    def generate_triplet(self, params, rng):
        obj1 = primitive_from_config(params["obj1"])
        obj2 = primitive_from_config(params["obj2"])
        midpoint = np.array(params["midpoint"])
        obj1.center = midpoint + np.array([-0.6, 0, 0])
        obj2.center = midpoint + np.array([0.6, 0, 0])
        b1 = apply_translation(obj1, np.array([0.3, 0, 0]))
        b2 = apply_translation(obj2, np.array([-0.3, 0, 0]))
        c1 = apply_translation(obj2, np.array([-1.2, 0, 0]))
        c2 = apply_translation(obj1, np.array([1.2, 0, 0]))
        return Scene([obj1, obj2]), Scene([b1, b2]), Scene([c1, c2]), params


@dataclass
class C10ProgressiveClashAngle(Rule):
    def __init__(self) -> None:
        super().__init__("C10", RuleDifficulty.COMPLEX, "Progressive Clash Angle", "Cylinders collide with growing angle")

    def sample_params(self, rng) -> Dict:
        base_angle = float(rng.uniform(math.pi / 12, math.pi / 8))
        delta = float(rng.uniform(math.pi / 18, math.pi / 10))
        return {"base_angle": base_angle, "delta": delta}

    def generate_triplet(self, params, rng):
        base_angle = params["base_angle"]
        delta = params["delta"]
        cyl1 = Cylinder(center=np.zeros(3), radius=0.25, height=1.6, rotation=np.array([0, 0, 0]))
        cyl2 = Cylinder(center=np.zeros(3), radius=0.25, height=1.6, rotation=np.array([0, base_angle, 0]))
        b2 = apply_rotation(cyl2, np.array([0, delta, 0]))
        c2 = apply_rotation(b2, np.array([0, delta, 0]))
        return Scene([cyl1, cyl2]), Scene([cyl1, b2]), Scene([cyl1, c2]), params


@dataclass
class C11ContactAreaGrowth(Rule):
    def __init__(self) -> None:
        super().__init__("C11", RuleDifficulty.COMPLEX, "Contact Area Growth", "Contact evolves from point to line to face")

    def sample_params(self, rng) -> Dict:
        tilt = float(rng.uniform(math.pi / 18, math.pi / 12))
        return {"tilt": tilt}

    def generate_triplet(self, params, rng):
        tilt = params["tilt"]
        cube = Cube(center=np.zeros(3), edge_lengths=np.array([1.0, 1.0, 1.0]))
        cyl_point = Cylinder(center=np.array([0, 0, 0.6]), radius=0.15, height=1.2, rotation=np.array([tilt, 0, 0]))
        cyl_line = apply_translation(cyl_point, np.array([0, 0, -0.3]))
        cyl_face = apply_translation(cyl_point, np.array([0, 0, -0.6]))
        return Scene([cube, cyl_point]), Scene([cube, cyl_line]), Scene([cube, cyl_face]), params


@dataclass
class C12MultiObjectConfiguration(Rule):
    def __init__(self) -> None:
        super().__init__("C12", RuleDifficulty.COMPLEX, "Multi-Object Configuration", "Three-object formation scales/rotates together")

    def sample_params(self, rng) -> Dict:
        scale = float(rng.uniform(1.1, 1.5))
        rotation = rng.uniform(math.pi / 30, math.pi / 12, size=3)
        return {"scale": scale, "rotation": rotation.tolist()}

    def generate_triplet(self, params, rng):
        scale = params["scale"]
        rotation = np.array(params["rotation"])
        objs = [
            Sphere(center=np.array([0.6, 0, 0]), radius=0.25),
            Sphere(center=np.array([-0.3, 0.52, 0]), radius=0.25),
            Sphere(center=np.array([-0.3, -0.52, 0]), radius=0.25),
        ]

        def transform(obj, factor, rot):
            new_obj = apply_scale(obj, factor)
            rot_m = rotation_matrix(rot)
            new_obj.center = rot_m @ new_obj.center
            return new_obj

        a = Scene(objs)
        b = Scene([transform(o, scale, rotation) for o in objs])
        c = Scene([transform(o, scale * scale, rotation * 2) for o in objs])
        return a, b, c, params


def build_complex_rules() -> List[Rule]:
    return [
        C01ScaleTranslationCoupled(),
        C02RotationScalingCoupled(),
        C03MultiStepSequence(),
        C04BooleanSequence(),
        C05HoleTopology(),
        C06Tunnel(),
        C07CrossSection(),
        C08SymmetryScale(),
        C09RoleExchange(),
        C10ProgressiveClashAngle(),
        C11ContactAreaGrowth(),
        C12MultiObjectConfiguration(),
    ]
