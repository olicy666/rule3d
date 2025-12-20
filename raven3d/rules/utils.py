from __future__ import annotations

import math
from typing import Iterable, List, Sequence

import numpy as np

from ..geometry import rotation_matrix
from ..scene import ObjectState, Scene

SHAPES = ["cube", "sphere", "cylinder", "cone"]


def random_center(rng: np.random.Generator, low: float = -0.5, high: float = 0.5) -> np.ndarray:
    return rng.uniform(low, high, size=3)


def random_rotation(rng: np.random.Generator, max_angle: float = math.pi / 4) -> np.ndarray:
    return rng.uniform(-max_angle, max_angle, size=3)


def random_scale(rng: np.random.Generator, low: float = 0.7, high: float = 1.3) -> np.ndarray:
    return rng.uniform(low, high, size=3)


def random_density(rng: np.random.Generator) -> float:
    return float(rng.uniform(0.8, 1.2))


def random_object(rng: np.random.Generator, shape: str | None = None) -> ObjectState:
    shape = shape or str(rng.choice(SHAPES))
    return ObjectState(
        shape=shape,
        r=random_scale(rng),
        p=random_center(rng),
        rotation=random_rotation(rng),
        density=random_density(rng),
    )


def init_objects(rng: np.random.Generator, k: int, m: int | None = None) -> List[ObjectState]:
    """Create M objects (2 or 3) ensuring M >= K."""
    if m is None:
        m = max(k, int(rng.integers(2, 4)))
    m = min(max(m, k, 2), 3)
    return [random_object(rng) for _ in range(m)]


def clone_objects(objs: Sequence[ObjectState]) -> List[ObjectState]:
    return [o.copy() for o in objs]


def apply_scale(obj: ObjectState, factor: float | Sequence[float], axis_mask: Sequence[bool] | None = None) -> ObjectState:
    factor_arr = np.array(factor if isinstance(factor, (list, tuple, np.ndarray)) else [factor] * 3, dtype=float)
    new_obj = obj.copy()
    if axis_mask is not None:
        mask = np.array(axis_mask, dtype=bool)
        factor_arr = np.where(mask, factor_arr, 1.0)
    new_obj.r = new_obj.r * factor_arr
    return new_obj


def apply_translation(obj: ObjectState, delta: Sequence[float]) -> ObjectState:
    new_obj = obj.copy()
    new_obj.p = new_obj.p + np.array(delta, dtype=float)
    return new_obj


def apply_rotation(obj: ObjectState, delta: Sequence[float]) -> ObjectState:
    new_obj = obj.copy()
    new_obj.rotation = new_obj.rotation + np.array(delta, dtype=float)
    return new_obj


def apply_density(obj: ObjectState, factor: float) -> ObjectState:
    new_obj = obj.copy()
    new_obj.density = max(new_obj.density * factor, 1e-3)
    return new_obj


def switch_shape(obj: ObjectState, shape: str) -> ObjectState:
    new_obj = obj.copy()
    new_obj.shape = shape
    return new_obj


def size(obj: ObjectState) -> float:
    """Use volume as the size scalar (fixed choice across rules)."""
    return float(np.prod(obj.r))


def aspect_ratio(obj: ObjectState) -> np.ndarray:
    return np.array([obj.r[0] / (obj.r[1] + 1e-6), obj.r[1] / (obj.r[2] + 1e-6)], dtype=float)


def axis(obj: ObjectState, idx: int = 0) -> np.ndarray:
    return rotation_matrix(obj.rotation)[:, idx]


def density(obj: ObjectState) -> float:
    return float(obj.density)


def dist(a: ObjectState, b: ObjectState) -> float:
    return float(np.linalg.norm(a.p - b.p))


def direction(a: ObjectState, b: ObjectState, eps: float = 1e-6) -> np.ndarray:
    delta = b.p - a.p
    norm = np.linalg.norm(delta)
    return delta / (norm + eps)


def ang(a: ObjectState, b: ObjectState) -> float:
    u = axis(a, 0)
    v = axis(b, 0)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.arccos(dot))


def approx_radius(obj: ObjectState, c: float = 0.6) -> float:
    return float(c * np.linalg.norm(obj.r))


def touch(a: ObjectState, b: ObjectState, tau: float = 0.05) -> int:
    return int(dist(a, b) <= approx_radius(a) + approx_radius(b) + tau)


def aabb(obj: ObjectState) -> tuple[np.ndarray, np.ndarray]:
    half = obj.r / 2.0
    return obj.p - half, obj.p + half


def contain(a: ObjectState, b: ObjectState, tau: float = 0.05) -> int:
    amin, amax = aabb(a)
    bmin, bmax = aabb(b)
    return int(np.all(bmin >= amin - tau) and np.all(bmax <= amax + tau))


def centroid(objs: Iterable[ObjectState]) -> np.ndarray:
    pts = np.stack([o.p for o in objs], axis=0)
    return pts.mean(axis=0)


def triangle_area(o1: ObjectState, o2: ObjectState, o3: ObjectState) -> float:
    return float(0.5 * np.linalg.norm(np.cross(o2.p - o1.p, o3.p - o1.p)))


def order_indices_x(objs: Sequence[ObjectState]) -> List[int]:
    return [idx for idx, _ in sorted(enumerate(objs), key=lambda kv: kv[1].p[0])]


def symmetry_flag(objs: Sequence[ObjectState], axis_name: str = "x", tol: float = 0.12) -> int:
    if not objs:
        return 0
    n = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}[axis_name]
    c = centroid(objs)
    flags = []
    for i, obj in enumerate(objs):
        mirror_p = obj.p - 2 * np.dot(obj.p - c, n) * n
        dists = [np.linalg.norm(mirror_p - other.p) for j, other in enumerate(objs) if j != i]
        flags.append(min(dists) <= tol if dists else True)
    return int(all(flags))


def scene_from_objects(objs: Sequence[ObjectState]) -> Scene:
    return Scene(objects=list(objs))


def build_rule_meta(
    rule,
    rule_group: str,
    k_r: int,
    involved_indices: Sequence[int],
    base_attrs_used: Sequence[str],
    derived_funcs_used: Sequence[str],
    pattern_type: str,
    pattern_params: dict,
    v: Sequence[Sequence[float]],
    scenes: Sequence[Scene],
) -> dict:
    return {
        "rule_id": rule.rule_id,
        "rule_group": rule_group,
        "difficulty": rule.difficulty.value,
        "K_R": k_r,
        "M_t": [len(scene.objects) for scene in scenes],
        "involved_indices": list(involved_indices),
        "base_attrs_used": list(base_attrs_used),
        "derived_funcs": list(derived_funcs_used),
        "pattern_type": pattern_type,
        "pattern_params": pattern_params,
        "v": {"v1": list(np.atleast_1d(v[0]).tolist()), "v2": list(np.atleast_1d(v[1]).tolist()), "v3": list(np.atleast_1d(v[2]).tolist())},
        "frames": [{"objects": scene.as_descriptions()} for scene in scenes],
    }
