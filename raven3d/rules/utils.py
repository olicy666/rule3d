from __future__ import annotations

import copy
import math
from typing import Tuple

import numpy as np

from ..geometry import Sphere, Cube, Cylinder, Cone, Primitive


def random_center(rng: np.random.Generator, low: float = -0.6, high: float = 0.6) -> np.ndarray:
    return rng.uniform(low, high, size=3)


def random_rotation(rng: np.random.Generator, max_angle: float = math.pi / 3) -> np.ndarray:
    return rng.uniform(-max_angle, max_angle, size=3)


def random_scale(rng: np.random.Generator, low: float = 0.7, high: float = 1.3) -> np.ndarray:
    return rng.uniform(low, high, size=3)


def random_primitive(rng: np.random.Generator) -> Primitive:
    shape = rng.choice(["sphere", "cube", "cylinder", "cone"])
    center = random_center(rng)
    rotation = random_rotation(rng)
    scale = random_scale(rng)
    if shape == "sphere":
        radius = float(rng.uniform(0.25, 0.55))
        return Sphere(center=center, rotation=rotation, scale=scale, radius=radius)
    if shape == "cube":
        edges = rng.uniform(0.4, 0.9, size=3)
        return Cube(center=center, rotation=rotation, scale=scale, edge_lengths=edges)
    if shape == "cylinder":
        radius = float(rng.uniform(0.2, 0.5))
        height = float(rng.uniform(0.6, 1.1))
        return Cylinder(center=center, rotation=rotation, scale=scale, radius=radius, height=height)
    radius = float(rng.uniform(0.2, 0.5))
    height = float(rng.uniform(0.7, 1.1))
    return Cone(center=center, rotation=rotation, scale=scale, radius=radius, height=height)


def clone_primitive(primitive: Primitive) -> Primitive:
    return copy.deepcopy(primitive)


def apply_scale(primitive: Primitive, factor: float | Tuple[float, float, float]) -> Primitive:
    new_prim = clone_primitive(primitive)
    factor_arr = np.array(factor if isinstance(factor, tuple) or isinstance(factor, list) else [factor] * 3, dtype=float)
    new_prim.scale = new_prim.scale * factor_arr
    return new_prim


def apply_translation(primitive: Primitive, delta: np.ndarray) -> Primitive:
    new_prim = clone_primitive(primitive)
    new_prim.center = new_prim.center + delta
    return new_prim


def apply_rotation(primitive: Primitive, delta: np.ndarray) -> Primitive:
    new_prim = clone_primitive(primitive)
    new_prim.rotation = new_prim.rotation + delta
    return new_prim


def primitive_to_config(primitive: Primitive) -> dict:
    base = {
        "type": primitive.__class__.__name__,
        "center": primitive.center.tolist(),
        "rotation": primitive.rotation.tolist(),
        "scale": primitive.scale.tolist(),
    }
    if isinstance(primitive, Sphere):
        base["radius"] = primitive.radius
    elif isinstance(primitive, Cube):
        base["edge_lengths"] = primitive.edge_lengths.tolist()
    elif isinstance(primitive, Cylinder):
        base["radius"] = primitive.radius
        base["height"] = primitive.height
    elif isinstance(primitive, Cone):
        base["radius"] = primitive.radius
        base["height"] = primitive.height
    return base


def primitive_from_config(cfg: dict) -> Primitive:
    prim_type = cfg["type"].lower()
    center = np.array(cfg["center"], dtype=float)
    rotation = np.array(cfg.get("rotation", [0, 0, 0]), dtype=float)
    scale = np.array(cfg.get("scale", [1, 1, 1]), dtype=float)
    if prim_type == "sphere":
        return Sphere(center=center, rotation=rotation, scale=scale, radius=float(cfg.get("radius", 0.5)))
    if prim_type == "cube":
        edges = np.array(cfg.get("edge_lengths", [1, 1, 1]), dtype=float)
        return Cube(center=center, rotation=rotation, scale=scale, edge_lengths=edges)
    if prim_type == "cylinder":
        return Cylinder(
            center=center,
            rotation=rotation,
            scale=scale,
            radius=float(cfg.get("radius", 0.5)),
            height=float(cfg.get("height", 1.0)),
        )
    if prim_type == "cone":
        return Cone(
            center=center,
            rotation=rotation,
            scale=scale,
            radius=float(cfg.get("radius", 0.5)),
            height=float(cfg.get("height", 1.0)),
        )
    raise ValueError(f"Unknown primitive type: {prim_type}")
