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


def unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / (np.linalg.norm(v) + 1e-9)


@dataclass
class M01SeparateTouchIntersect(Rule):
    def __init__(self) -> None:
        super().__init__("M01", RuleDifficulty.MEDIUM, "Separate → Touch → Intersect", "Objects move from separate to intersecting")

    def sample_params(self, rng) -> Dict:
        r1, r2 = float(rng.uniform(0.25, 0.45)), float(rng.uniform(0.25, 0.45))
        direction = unit_vector(rng)
        return {"radii": [r1, r2], "direction": direction.tolist()}

    def generate_triplet(self, params, rng):
        r1, r2 = params["radii"]
        direction = np.array(params["direction"])
        gap_a = (r1 + r2) * 1.8
        gap_b = (r1 + r2) * 1.0
        gap_c = (r1 + r2) * 0.6
        center1 = -direction * gap_a / 2
        center2 = direction * gap_a / 2
        obj1 = Sphere(center=center1, radius=r1)
        obj2 = Sphere(center=center2, radius=r2)
        a = Scene([obj1, obj2])
        b = Scene(
            [
                Sphere(center=-direction * gap_b / 2, radius=r1),
                Sphere(center=direction * gap_b / 2, radius=r2),
            ]
        )
        c = Scene(
            [
                Sphere(center=-direction * gap_c / 2, radius=r1),
                Sphere(center=direction * gap_c / 2, radius=r2),
            ]
        )
        return a, b, c, params


@dataclass
class M02IntersectionDepthGrowth(Rule):
    def __init__(self) -> None:
        super().__init__("M02", RuleDifficulty.MEDIUM, "Intersection Depth Growth", "Intersection depth increases along direction")

    def sample_params(self, rng) -> Dict:
        direction = unit_vector(rng)
        offset = float(rng.uniform(0.4, 0.8))
        return {"direction": direction.tolist(), "offset": offset}

    def generate_triplet(self, params, rng):
        direction = np.array(params["direction"])
        offset = params["offset"]
        base1 = Cube(center=-direction * offset, edge_lengths=np.array([0.8, 0.8, 0.8]))
        base2 = Cube(center=direction * offset, edge_lengths=np.array([0.8, 0.8, 0.8]))
        delta = direction * offset * 0.6
        a = Scene([base1, base2])
        b = Scene([apply_translation(base1, delta), base2])
        c = Scene([apply_translation(base1, delta * 2), base2])
        return a, b, c, params


@dataclass
class M03Containment(Rule):
    def __init__(self) -> None:
        super().__init__("M03", RuleDifficulty.MEDIUM, "Containment", "Inner moves inside outer without leaving")

    def sample_params(self, rng) -> Dict:
        outer_size = float(rng.uniform(1.2, 1.6))
        inner_radius = float(rng.uniform(0.25, 0.45))
        return {"outer_size": outer_size, "inner_radius": inner_radius}

    def generate_triplet(self, params, rng):
        outer_size = params["outer_size"]
        inner_radius = params["inner_radius"]
        outer = Cube(center=np.zeros(3), edge_lengths=np.array([outer_size] * 3))
        path = unit_vector(rng) * (outer_size / 4)
        a_inner = Sphere(center=-path, radius=inner_radius)
        b_inner = Sphere(center=np.zeros(3), radius=inner_radius)
        c_inner = Sphere(center=path, radius=inner_radius)
        return Scene([outer, a_inner]), Scene([outer, b_inner]), Scene([outer, c_inner]), params


@dataclass
class M04RelativePosition(Rule):
    def __init__(self) -> None:
        super().__init__("M04", RuleDifficulty.MEDIUM, "Relative Position", "Objects move in one direction keeping ordering")

    def sample_params(self, rng) -> Dict:
        direction = unit_vector(rng)
        step = float(rng.uniform(0.4, 0.8))
        base = primitive_to_config(random_primitive(rng))
        base2 = primitive_to_config(random_primitive(rng))
        return {"direction": direction.tolist(), "step": step, "obj1": base, "obj2": base2}

    def generate_triplet(self, params, rng):
        direction = np.array(params["direction"])
        step = params["step"]
        obj1 = primitive_from_config(params["obj1"])
        obj2 = primitive_from_config(params["obj2"])
        obj1.center = -direction * step
        obj2.center = direction * step
        a = Scene([obj1, obj2])
        b = Scene([apply_translation(obj1, direction * step), apply_translation(obj2, direction * step)])
        c = Scene([apply_translation(obj1, direction * step * 2), apply_translation(obj2, direction * step * 2)])
        return a, b, c, {"direction": params["direction"], "step": step}


@dataclass
class M05CenterOfMassShift(Rule):
    def __init__(self) -> None:
        super().__init__("M05", RuleDifficulty.MEDIUM, "Center-of-Mass Shift", "Relative center-of-mass offset changes linearly")

    def sample_params(self, rng) -> Dict:
        offset = float(rng.uniform(0.3, 0.7))
        direction = unit_vector(rng)
        return {"offset": offset, "direction": direction.tolist()}

    def generate_triplet(self, params, rng):
        direction = np.array(params["direction"])
        offset = params["offset"]
        obj1 = Sphere(center=-direction * offset, radius=0.4)
        obj2 = Cube(center=direction * offset, edge_lengths=np.array([0.6, 0.6, 0.6]))
        delta = direction * offset * 0.6
        a = Scene([obj1, obj2])
        b = Scene([apply_translation(obj1, delta), obj2])
        c = Scene([apply_translation(obj1, delta * 2), obj2])
        return a, b, c, params


@dataclass
class M06ParallelPerpendicular(Rule):
    def __init__(self) -> None:
        super().__init__("M06", RuleDifficulty.MEDIUM, "Parallel/Perpendicular", "Keep axes parallel or perpendicular")

    def sample_params(self, rng) -> Dict:
        base_angle = float(rng.uniform(0, math.pi / 4))
        parallel = bool(rng.choice([True, False]))
        return {"base_angle": base_angle, "mode": "parallel" if parallel else "perpendicular"}

    def generate_triplet(self, params, rng):
        base_angle = params["base_angle"]
        parallel = params["mode"] == "parallel"
        cyl1 = Cylinder(center=np.array([-0.4, 0, 0]), radius=0.25, height=1.2, rotation=np.array([base_angle, 0, 0]))
        if parallel:
            cyl2_rot = np.array([base_angle, 0, 0])
        else:
            cyl2_rot = np.array([base_angle + math.pi / 2, 0, 0])
        cyl2 = Cylinder(center=np.array([0.4, 0, 0]), radius=0.25, height=1.2, rotation=cyl2_rot)
        delta = np.array([0.1, 0.2, 0])
        a = Scene([cyl1, cyl2])
        b = Scene([apply_translation(cyl1, delta), apply_translation(cyl2, delta)])
        c = Scene([apply_translation(cyl1, delta * 2), apply_translation(cyl2, delta * 2)])
        return a, b, c, params


@dataclass
class M07AxisAngleLinear(Rule):
    def __init__(self) -> None:
        super().__init__("M07", RuleDifficulty.MEDIUM, "Axis Angle Linear", "Axis angle changes gradually")

    def sample_params(self, rng) -> Dict:
        delta = float(rng.uniform(math.pi / 18, math.pi / 8))
        base_angle = float(rng.uniform(-math.pi / 6, math.pi / 6))
        return {"delta": delta, "base_angle": base_angle}

    def generate_triplet(self, params, rng):
        delta = params["delta"]
        base_angle = params["base_angle"]
        obj1 = Cylinder(center=np.array([-0.5, 0, 0]), rotation=np.array([0, base_angle, 0]), radius=0.2, height=1.3)
        obj2 = Cylinder(center=np.array([0.5, 0, 0]), rotation=np.array([0, -base_angle, 0]), radius=0.2, height=1.3)
        b1 = apply_rotation(obj1, np.array([0, delta, 0]))
        b2 = apply_rotation(obj2, np.array([0, -delta, 0]))
        c1 = apply_rotation(b1, np.array([0, delta, 0]))
        c2 = apply_rotation(b2, np.array([0, -delta, 0]))
        return Scene([obj1, obj2]), Scene([b1, b2]), Scene([c1, c2]), params


@dataclass
class M08AxesSymmetry(Rule):
    def __init__(self) -> None:
        super().__init__("M08", RuleDifficulty.MEDIUM, "Axes Symmetry", "Maintain mirror symmetry")

    def sample_params(self, rng) -> Dict:
        distance = float(rng.uniform(0.4, 0.8))
        scale = float(rng.uniform(0.8, 1.3))
        return {"distance": distance, "scale": scale}

    def generate_triplet(self, params, rng):
        distance = params["distance"]
        scale = params["scale"]
        obj_l = Cone(center=np.array([-distance, 0, 0]), rotation=np.array([0, 0, 0]), radius=0.3, height=1.0)
        obj_r = Cone(center=np.array([distance, 0, 0]), rotation=np.array([0, 0, math.pi]), radius=0.3, height=1.0)
        a = Scene([obj_l, obj_r])
        b = Scene([apply_scale(obj_l, scale), apply_scale(obj_r, scale)])
        c = Scene([apply_scale(obj_l, scale * scale), apply_scale(obj_r, scale * scale)])
        return a, b, c, params


@dataclass
class M09DistanceLinear(Rule):
    def __init__(self) -> None:
        super().__init__("M09", RuleDifficulty.MEDIUM, "Distance Linear Change", "Distance between objects changes linearly")

    def sample_params(self, rng) -> Dict:
        direction = unit_vector(rng)
        base_dist = float(rng.uniform(0.5, 1.0))
        step = float(rng.uniform(0.2, 0.4))
        return {"direction": direction.tolist(), "base_distance": base_dist, "step": step}

    def generate_triplet(self, params, rng):
        direction = np.array(params["direction"])
        base_dist = params["base_distance"]
        step = params["step"]
        obj1 = Sphere(center=-direction * base_dist / 2, radius=0.35)
        obj2 = Sphere(center=direction * base_dist / 2, radius=0.35)
        a = Scene([obj1, obj2])
        b = Scene([apply_translation(obj1, -direction * step / 2), apply_translation(obj2, direction * step / 2)])
        c = Scene([apply_translation(obj1, -direction * step), apply_translation(obj2, direction * step)])
        return a, b, c, params


@dataclass
class M10PatternMovement(Rule):
    def __init__(self) -> None:
        super().__init__("M10", RuleDifficulty.MEDIUM, "Pattern Movement", "Group moves maintaining formation")

    def sample_params(self, rng) -> Dict:
        direction = unit_vector(rng)
        step = float(rng.uniform(0.3, 0.6))
        return {"direction": direction.tolist(), "step": step}

    def generate_triplet(self, params, rng):
        direction = np.array(params["direction"])
        step = params["step"]
        objs = []
        for i in range(3):
            center = np.array([i * 0.6 - 0.6, 0, 0]) + rng.uniform(-0.1, 0.1, size=3)
            objs.append(Sphere(center=center, radius=0.25))
        a = Scene(objs)
        b = Scene([apply_translation(o, direction * step) for o in objs])
        c = Scene([apply_translation(o, direction * step * 2) for o in objs])
        return a, b, c, params


@dataclass
class M11HierarchyPreserved(Rule):
    def __init__(self) -> None:
        super().__init__("M11", RuleDifficulty.MEDIUM, "Hierarchy Preserved", "Smaller object attached to larger one")

    def sample_params(self, rng) -> Dict:
        offset = rng.uniform(0.3, 0.6, size=3)
        return {"offset": offset.tolist()}

    def generate_triplet(self, params, rng):
        offset = np.array(params["offset"])
        base = Cylinder(center=np.zeros(3), radius=0.35, height=1.2)
        attached = Sphere(center=offset, radius=0.25)
        delta = rng.uniform(-0.1, 0.1, size=3)
        a = Scene([base, attached])
        b = Scene([apply_translation(base, delta), apply_translation(attached, delta)])
        c = Scene([apply_translation(base, delta * 2), apply_translation(attached, delta * 2)])
        return a, b, c, params


@dataclass
class M12CompositeRatio(Rule):
    def __init__(self) -> None:
        super().__init__("M12", RuleDifficulty.MEDIUM, "Composite Ratio", "Size and distance change with a shared ratio")

    def sample_params(self, rng) -> Dict:
        ratio = float(rng.uniform(1.1, 1.5))
        return {"ratio": ratio}

    def generate_triplet(self, params, rng):
        ratio = params["ratio"]
        obj1 = Cube(center=np.array([-0.5, 0, 0]), edge_lengths=np.array([0.6, 0.6, 0.6]))
        obj2 = Cube(center=np.array([0.5, 0, 0]), edge_lengths=np.array([0.6, 0.6, 0.6]))
        a = Scene([obj1, obj2])
        b = Scene([apply_scale(obj1, ratio), apply_translation(apply_scale(obj2, ratio), np.array([ratio * 0.3, 0, 0]))])
        c = Scene([apply_scale(obj1, ratio * ratio), apply_translation(apply_scale(obj2, ratio * ratio), np.array([ratio * 0.6, 0, 0]))])
        return a, b, c, params


@dataclass
class M13Alignment(Rule):
    def __init__(self) -> None:
        super().__init__("M13", RuleDifficulty.MEDIUM, "Alignment", "Face -> edge -> vertex alignment sequence")

    def sample_params(self, rng) -> Dict:
        return {"offset": float(rng.uniform(0.3, 0.7))}

    def generate_triplet(self, params, rng):
        offset = params["offset"]
        cube1 = Cube(center=np.zeros(3), edge_lengths=np.array([0.8, 0.8, 0.8]))
        cube2 = Cube(center=np.array([offset, 0, 0]), edge_lengths=np.array([0.6, 0.6, 0.6]))
        a = Scene([cube1, cube2])
        # Edge align by moving along diagonal on face.
        b = Scene([cube1, Cube(center=np.array([offset, offset, 0]), edge_lengths=cube2.edge_lengths)])
        # Vertex align at full diagonal.
        c = Scene([cube1, Cube(center=np.array([offset, offset, offset]), edge_lengths=cube2.edge_lengths)])
        return a, b, c, params


@dataclass
class M14ShapeFamily(Rule):
    def __init__(self) -> None:
        super().__init__("M14", RuleDifficulty.MEDIUM, "Shape Family Relation", "Morph within a shape family")

    def sample_params(self, rng) -> Dict:
        stretch = float(rng.uniform(1.2, 1.6))
        return {"stretch": stretch}

    def generate_triplet(self, params, rng):
        stretch = params["stretch"]
        sphere = Sphere(center=np.zeros(3), radius=0.45)
        ellipsoid = apply_scale(sphere, (stretch, 1.0, 1.0))
        a = Scene([sphere])
        b = Scene([ellipsoid])
        c = Scene([sphere])
        return a, b, c, params


def build_medium_rules() -> List[Rule]:
    return [
        M01SeparateTouchIntersect(),
        M02IntersectionDepthGrowth(),
        M03Containment(),
        M04RelativePosition(),
        M05CenterOfMassShift(),
        M06ParallelPerpendicular(),
        M07AxisAngleLinear(),
        M08AxesSymmetry(),
        M09DistanceLinear(),
        M10PatternMovement(),
        M11HierarchyPreserved(),
        M12CompositeRatio(),
        M13Alignment(),
        M14ShapeFamily(),
    ]
