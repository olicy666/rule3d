from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

from .base import Rule, RuleDifficulty
from .medium import R2_1DistanceGeometric, R2_3DirectionRotate, R3_2OrderingCycle
from .simple import (
    R1_1ScaleArithmetic,
    R1_2FixedAxisRotation,
    R1_4TranslationArithmetic,
    R1_5DensityArithmetic,
    R2_2AnisotropicGeometric,
)
from .utils import (
    aabb,
    ang,
    apply_density,
    apply_rotation,
    apply_scale,
    apply_translation,
    approx_radius,
    aspect_ratio,
    build_rule_meta,
    centroid,
    clone_objects,
    direction,
    dist,
    init_objects,
    order_indices_x,
    random_object,
    scene_from_objects,
    SHAPES,
    size,
    switch_shape,
    symmetry_flag,
    _separate_objects_no_contact,
)
from ..geometry import rotation_matrix
from ..scene import Scene, ObjectState


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / (np.linalg.norm(v) + 1e-9)


def _spread_objects(objs: Sequence[ObjectState], rng) -> None:
    if len(objs) < 2:
        return
    sizes = [np.linalg.norm(o.r) for o in objs]
    avg_size = float(np.mean(sizes)) if sizes else 0.0
    min_sep = max(0.16, 0.25 * avg_size)
    min_sep /= max(1.0, math.sqrt(len(objs)) / 1.5)
    for _ in range(10):
        moved = False
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                delta = objs[j].p - objs[i].p
                dist = float(np.linalg.norm(delta))
                if dist < min_sep:
                    direction_vec = _unit_vector(rng) if dist < 1e-6 else delta / dist
                    shift = 0.5 * (min_sep - dist)
                    objs[i].p = objs[i].p - direction_vec * shift
                    objs[j].p = objs[j].p + direction_vec * shift
                    moved = True
        if not moved:
            break


def _min_pairwise_distance(objs: Sequence[ObjectState]) -> float:
    if len(objs) < 2:
        return float("inf")
    min_dist = float("inf")
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            dist = float(np.linalg.norm(objs[i].p - objs[j].p))
            if dist < min_dist:
                min_dist = dist
    return min_dist


def _all_non_contact(objs: Sequence[ObjectState], gap: float = 0.05) -> bool:
    if len(objs) < 2:
        return True
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            min_dist = approx_radius(objs[i]) + approx_radius(objs[j]) + gap
            if float(np.linalg.norm(objs[i].p - objs[j].p)) < min_dist:
                return False
    return True


@dataclass
class R1_9ScaleRotateCoupled(Rule):
    def __init__(self) -> None:
        super().__init__("R1-9", RuleDifficulty.COMPLEX, "复合位姿缩放", "scale 与 rotation 复合")

    def sample_params(self, rng) -> Dict:
        k = float(rng.uniform(1.15, 1.5))
        delta_rot = rng.uniform(math.pi / 18, math.pi / 10, size=3)
        return {"k": k, "delta_rot": delta_rot.tolist()}

    def generate_triplet(self, params, rng):
        k = params["k"]
        delta_rot = np.array(params["delta_rot"])
        objs = init_objects(rng, 1, m=2)
        non_sphere_shapes = [shape for shape in SHAPES if shape != "sphere"]
        for idx, obj in enumerate(objs):
            if obj.shape == "sphere":
                objs[idx] = switch_shape(obj, str(rng.choice(non_sphere_shapes)))
        involved = [0]
        a_objs = clone_objects(objs)
        b_objs = clone_objects(objs)
        b_objs[0] = apply_scale(apply_rotation(b_objs[0], delta_rot), k)
        c_objs = clone_objects(b_objs)
        c_objs[0] = apply_scale(apply_rotation(c_objs[0], delta_rot), k)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [[size(o), float(np.linalg.norm(o.rotation))] for o in [a_objs[0], b_objs[0], c_objs[0]]]
        meta = build_rule_meta(
            self,
            "R1",
            1,
            involved,
            ["r", "R"],
            ["size(O0)", "axis(O0)"],
            "compound",
            {"k": k, "delta_rot": delta_rot.tolist()},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class C02PiecewiseChange(Rule):
    def __init__(self) -> None:
        super().__init__("C02", RuleDifficulty.COMPLEX, "分段序列", "先变化再保持")

    def sample_params(self, rng) -> Dict:
        delta_ratio = float(rng.uniform(0.2, 0.35))
        return {"delta_ratio": delta_ratio}

    def generate_triplet(self, params, rng):
        delta_ratio = params["delta_ratio"]
        objs = init_objects(rng, 1, m=2)
        involved = [0]
        base_size = size(objs[0])
        delta = base_size * delta_ratio
        target = base_size + delta
        factor = (target / base_size) ** (1 / 3)
        a_objs = clone_objects(objs)
        b_objs = [apply_scale(objs[0], factor)]
        c_objs = [b_objs[0].copy()]
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [base_size, size(b_objs[0]), size(c_objs[0])]
        meta = build_rule_meta(
            self, "R1", 1, involved, ["r"], ["size(O0)"], "piecewise", {"delta": delta}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class C03ConditionalShapeScale(Rule):
    def __init__(self) -> None:
        super().__init__("C03", RuleDifficulty.COMPLEX, "条件触发缩放", "形状变化触发尺度变化")

    def sample_params(self, rng) -> Dict:
        k = float(rng.uniform(1.2, 1.6))
        return {"k": k}

    def generate_triplet(self, params, rng):
        k = params["k"]
        objs = init_objects(rng, 1, m=2)
        involved = [0]
        shape_a = objs[0].shape
        shape_b = rng.choice([s for s in SHAPES if s != shape_a])
        base_size = size(objs[0])
        factor = (base_size * k / base_size) ** (1 / 3)
        b_obj = objs[0].copy()
        b_obj.shape = shape_b
        b_obj = apply_scale(b_obj, factor)
        c_obj = b_obj.copy()
        scenes = [scene_from_objects([objs[0]]), scene_from_objects([b_obj]), scene_from_objects([c_obj])]
        v = [[shape_a, base_size], [shape_b, size(b_obj)], [shape_b, size(c_obj)]]
        meta = build_rule_meta(
            self,
            "R1",
            1,
            involved,
            ["s", "r"],
            ["size(O0)", "s(O0)"],
            "conditional",
            {"k": k},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class R2_6RelativeOrientationInvariant(Rule):
    def __init__(self) -> None:
        super().__init__("R2-6", RuleDifficulty.COMPLEX, "相对姿态保持", "共同旋转保持夹角不变")

    def sample_params(self, rng) -> Dict:
        theta = float(rng.uniform(math.pi / 18, math.pi / 10))
        axis = int(rng.integers(0, 3))
        return {"theta": theta, "axis": axis}

    def generate_triplet(self, params, rng):
        theta = params["theta"]
        axis_idx = params["axis"]
        objs = init_objects(rng, 2, m=2)
        involved = [0, 1]
        delta_vec = np.zeros(3)
        delta_vec[axis_idx] = theta
        a_objs = clone_objects(objs)
        b_objs = [apply_rotation(o, delta_vec) for o in objs]
        c_objs = [apply_rotation(o, delta_vec * 2) for o in objs]
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [ang(*pair) for pair in [[a_objs[0], a_objs[1]], [b_objs[0], b_objs[1]], [c_objs[0], c_objs[1]]]]
        meta = build_rule_meta(
            self,
            "R2",
            2,
            involved,
            ["R"],
            ["ang(0,1)"],
            "rigid-rotation",
            {"axis": axis_idx, "theta": theta},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class R3_7PositionCycle(Rule):
    def __init__(self) -> None:
        super().__init__("R3-7", RuleDifficulty.COMPLEX, "多对象位置轮换", "球体沿结构按步长轮换")

    def sample_params(self, rng) -> Dict:
        count = int(rng.choice([3, 5]))
        direction = "cw" if rng.random() < 0.5 else "ccw"
        step = int(rng.integers(1, count))
        return {"count": count, "direction": direction, "step": step}

    @staticmethod
    def _regular_polygon(count: int, radius: float) -> List[np.ndarray]:
        angles = [math.pi / 2 - i * 2 * math.pi / count for i in range(count)]
        return [np.array([radius * math.cos(a), radius * math.sin(a), 0.0]) for a in angles]

    def _layout_positions(self, count: int) -> tuple[List[np.ndarray], str]:
        if count == 2:
            return [np.array([-0.6, 0.0, 0.0]), np.array([0.6, 0.0, 0.0])], "line"
        if count == 3:
            return self._regular_polygon(3, 0.65), "triangle"
        if count == 4:
            return [
                np.array([-0.7, -0.45, 0.0]),
                np.array([0.7, -0.45, 0.0]),
                np.array([0.7, 0.45, 0.0]),
                np.array([-0.7, 0.45, 0.0]),
            ], "rectangle"
        if count == 5:
            return self._regular_polygon(5, 0.7), "pentagon"
        raise ValueError(f"Unsupported object count {count}")

    def generate_triplet(self, params, rng):
        count = int(params["count"])
        direction = params["direction"]
        step = int(params.get("step", 1))
        if step <= 0:
            step = 1
        if step >= count:
            step = count - 1
        positions, layout_name = self._layout_positions(count)
        base_rot = rng.uniform(-math.pi / 6, math.pi / 6, size=3)
        scale = float(rng.uniform(0.9, 1.1))
        rot = rotation_matrix(base_rot)
        positions = [scale * (rot @ p) for p in positions]
        objs = [random_object(rng, shape="sphere") for _ in range(count)]
        size_factors = self._distinct_size_factors(rng, count)
        density_values = self._distinct_densities(rng, count)
        for obj, factor, density in zip(objs, size_factors, density_values):
            obj.r = obj.r * factor
            obj.density = density
        involved = list(range(count))
        step = step if direction == "cw" else -step

        frame_angles = [float(rng.uniform(-math.pi / 4, math.pi / 4)) for _ in range(3)]
        if max(frame_angles) - min(frame_angles) < math.pi / 12:
            frame_angles[1] = float(frame_angles[1] + math.pi / 6)
        frame_rots = [rotation_matrix(np.array([0.0, 0.0, a])) for a in frame_angles]

        def build_frame(offset: int, rot_m: np.ndarray) -> List:
            arranged = clone_objects(objs)
            for idx, obj in enumerate(arranged):
                obj.p = rot_m @ positions[(idx + offset) % count]
            return arranged

        def index_in_positions(offset: int) -> List[int]:
            return [int((j - offset) % count) for j in range(count)]

        a_objs = build_frame(0, frame_rots[0])
        b_objs = build_frame(step, frame_rots[1])
        c_objs = build_frame(2 * step, frame_rots[2])
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [index_in_positions(0), index_in_positions(step), index_in_positions(2 * step)]
        meta = build_rule_meta(
            self,
            "R3",
            3,
            involved,
            ["p"],
            ["position_cycle"],
            "cyclic",
            {
                "direction": direction,
                "count": count,
                "step": abs(step),
                "layout": layout_name,
                "rotation_euler": base_rot.tolist(),
                "frame_rot_z": frame_angles,
                "scale": scale,
                "size_factors": size_factors,
                "densities": density_values,
            },
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []
        count = len(scene_c.objects)
        if count < 3:
            return [], []
        step = int(meta.get("pattern_params", {}).get("step", 1))
        if step <= 0:
            step = 1
        if step >= count:
            step = count - 1

        center = centroid(scene_c.objects)
        angles = [math.atan2(o.p[1] - center[1], o.p[0] - center[0]) for o in scene_c.objects]
        order = [idx for idx, _ in sorted(enumerate(angles), key=lambda kv: kv[1])]
        positions = [scene_c.objects[i].p.copy() for i in order]

        def shift_assign(offset: int) -> Scene:
            arranged = clone_objects(scene_c.objects)
            for k, obj_idx in enumerate(order):
                arranged[obj_idx].p = positions[(k + offset) % count]
            return scene_from_objects(arranged)

        wrong_step = 1 if step != 1 else 2
        wrong_step = min(wrong_step, count - 1)
        reverse_step = (-step) % count

        def stretch_y(scale: float) -> Scene:
            arranged = clone_objects(scene_c.objects)
            for obj in arranged:
                vec = obj.p - center
                vec[1] *= scale
                obj.p = center + vec
            return scene_from_objects(arranged)

        distractors = [shift_assign(wrong_step)]
        reasons = ["轮换步长错误"]
        if reverse_step != wrong_step:
            distractors.append(shift_assign(reverse_step))
            reasons.append("轮换方向错误")
        else:
            arranged = clone_objects(scene_c.objects)
            if count >= 2:
                a_idx = order[0]
                b_idx = order[1]
                tmp = arranged[a_idx].p.copy()
                arranged[a_idx].p = arranged[b_idx].p.copy()
                arranged[b_idx].p = tmp
            distractors.append(scene_from_objects(arranged))
            reasons.append("位置交换错误")
        distractors.append(stretch_y(1.4))
        reasons.append("结构被拉伸，位置关系破坏")
        return distractors, reasons

    @staticmethod
    def _distinct_size_factors(rng, count: int) -> List[float]:
        factors: List[float] = []
        attempts = 0
        while len(factors) < count and attempts < 200:
            candidate = float(rng.uniform(0.55, 1.75))
            if all(abs(candidate - f) > 0.3 for f in factors):
                factors.append(candidate)
            attempts += 1
        if len(factors) < count:
            base = float(rng.uniform(0.8, 1.25))
            step = 0.3
            factors = [max(base + step * (i - (count - 1) / 2), 0.3) for i in range(count)]
        return factors

    @staticmethod
    def _distinct_densities(rng, count: int) -> List[float]:
        densities: List[float] = []
        attempts = 0
        while len(densities) < count and attempts < 200:
            candidate = float(rng.uniform(0.5, 1.8))
            if all(abs(candidate - d) > 0.3 for d in densities):
                densities.append(candidate)
            attempts += 1
        if len(densities) < count:
            base = float(rng.uniform(0.7, 1.3))
            step = 0.35
            densities = [max(base + step * (i - (count - 1) / 2), 0.25) for i in range(count)]
        return densities


@dataclass
class R3_8DensityShift(Rule):
    def __init__(self) -> None:
        super().__init__("R3-8", RuleDifficulty.COMPLEX, "多对象密度变化", "多对象密度按位置延续增减")

    def sample_params(self, rng) -> Dict:
        count = int(rng.integers(2, 6))
        return {"count": count}

    @staticmethod
    def _regular_polygon(count: int, radius: float) -> List[np.ndarray]:
        angles = [math.pi / 2 - i * 2 * math.pi / count for i in range(count)]
        return [np.array([radius * math.cos(a), radius * math.sin(a), 0.0]) for a in angles]

    def _layout_positions(self, count: int) -> tuple[List[np.ndarray], str]:
        if count == 2:
            return [np.array([-0.6, 0.0, 0.0]), np.array([0.6, 0.0, 0.0])], "line"
        if count == 3:
            return self._regular_polygon(3, 0.65), "triangle"
        if count == 4:
            return [
                np.array([-0.7, -0.45, 0.0]),
                np.array([0.7, -0.45, 0.0]),
                np.array([0.7, 0.45, 0.0]),
                np.array([-0.7, 0.45, 0.0]),
            ], "rectangle"
        if count == 5:
            return self._regular_polygon(5, 0.7), "pentagon"
        raise ValueError(f"Unsupported object count {count}")

    @staticmethod
    def _distinct_densities(rng, count: int) -> List[float]:
        densities: List[float] = []
        attempts = 0
        while len(densities) < count and attempts < 200:
            candidate = float(rng.uniform(0.6, 1.4))
            if all(abs(candidate - d) > 0.08 for d in densities):
                densities.append(candidate)
            attempts += 1
        if len(densities) < count:
            base = float(rng.uniform(0.7, 1.1))
            step = 0.12
            densities = [max(base + step * (i - (count - 1) / 2), 0.2) for i in range(count)]
        return densities

    def generate_triplet(self, params, rng):
        count = int(params["count"])
        positions, layout_name = self._layout_positions(count)
        objs = [random_object(rng) for _ in range(count)]
        involved = list(range(count))
        base_densities = self._distinct_densities(rng, count)

        for obj, pos, den in zip(objs, positions, base_densities):
            obj.p = pos
            obj.density = den
            obj.rotation = obj.rotation + rng.uniform(-math.pi / 18, math.pi / 18, size=3)
            obj.r = obj.r * rng.uniform(0.9, 1.1, size=3)

        factors = []
        for _ in range(count):
            roll = rng.random()
            if roll < 0.4:
                factors.append(float(rng.uniform(1.2, 1.6)))
            elif roll < 0.8:
                factors.append(float(rng.uniform(0.6, 0.85)))
            else:
                factors.append(1.0)
        if all(abs(f - 1.0) < 1e-6 for f in factors):
            idx = int(rng.integers(0, count))
            factors[idx] = float(rng.uniform(1.2, 1.6))

        a_objs = clone_objects(objs)
        b_objs = clone_objects(a_objs)
        for i, f in enumerate(factors):
            b_objs[i] = apply_density(b_objs[i], f)
        c_objs = clone_objects(b_objs)
        for i, f in enumerate(factors):
            c_objs[i] = apply_density(c_objs[i], f)

        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [[obj.density for obj in s.objects] for s in scenes]
        meta = build_rule_meta(
            self,
            "R3",
            3,
            involved,
            ["d"],
            ["den(Oi)"],
            "per-object-scale",
            {"count": count, "layout": layout_name, "factors": factors},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []
        v = meta.get("v", {})
        v2 = v.get("v2")
        v3 = v.get("v3")
        factors = meta.get("pattern_params", {}).get("factors")
        if not (v2 and v3 and factors):
            return [], []
        f = np.array(factors, dtype=float)
        if f.shape[0] != len(scene_c.objects):
            return [], []
        v2_arr = np.array(v2, dtype=float)

        def build_with(densities: np.ndarray) -> Scene:
            objs = clone_objects(scene_c.objects)
            for i, den in enumerate(densities.tolist()):
                obj = objs[i].copy()
                obj.density = max(float(den), 1e-3)
                objs[i] = obj
            return scene_from_objects(objs)

        wrong_single = v2_arr
        wrong_reverse = v2_arr / np.where(np.abs(f) < 1e-6, 1.0, f)
        wrong_over = v2_arr * (f ** 2)
        distractors = [build_with(x) for x in [wrong_single, wrong_reverse, wrong_over]]
        reasons = [
            "密度未继续变化，停留在上一帧",
            "密度变化方向反向",
            "密度变化幅度过大",
        ]
        return distractors, reasons


@dataclass
class R3_9ScaleShift(Rule):
    def __init__(self) -> None:
        super().__init__("R3-9", RuleDifficulty.COMPLEX, "多对象尺度变化", "多对象尺度按位置延续增减")

    def sample_params(self, rng) -> Dict:
        count = int(rng.integers(2, 6))
        return {"count": count}

    @staticmethod
    def _regular_polygon(count: int, radius: float) -> List[np.ndarray]:
        angles = [math.pi / 2 - i * 2 * math.pi / count for i in range(count)]
        return [np.array([radius * math.cos(a), radius * math.sin(a), 0.0]) for a in angles]

    def _layout_positions(self, count: int) -> tuple[List[np.ndarray], str]:
        if count == 2:
            return [np.array([-0.6, 0.0, 0.0]), np.array([0.6, 0.0, 0.0])], "line"
        if count == 3:
            return self._regular_polygon(3, 0.65), "triangle"
        if count == 4:
            return [
                np.array([-0.7, -0.45, 0.0]),
                np.array([0.7, -0.45, 0.0]),
                np.array([0.7, 0.45, 0.0]),
                np.array([-0.7, 0.45, 0.0]),
            ], "rectangle"
        if count == 5:
            return self._regular_polygon(5, 0.7), "pentagon"
        raise ValueError(f"Unsupported object count {count}")

    @staticmethod
    def _distinct_size_factors(rng, count: int) -> List[float]:
        factors: List[float] = []
        attempts = 0
        while len(factors) < count and attempts < 200:
            candidate = float(rng.uniform(0.75, 1.35))
            if all(abs(candidate - f) > 0.12 for f in factors):
                factors.append(candidate)
            attempts += 1
        if len(factors) < count:
            base = float(rng.uniform(0.85, 1.15))
            step = 0.15
            factors = [max(base + step * (i - (count - 1) / 2), 0.3) for i in range(count)]
        return factors

    def generate_triplet(self, params, rng):
        count = int(params["count"])
        positions, layout_name = self._layout_positions(count)
        objs = [random_object(rng) for _ in range(count)]
        involved = list(range(count))
        base_factors = self._distinct_size_factors(rng, count)

        for obj, pos, factor in zip(objs, positions, base_factors):
            obj.p = pos
            obj.r = obj.r * factor
            obj.rotation = obj.rotation + rng.uniform(-math.pi / 18, math.pi / 18, size=3)

        factors = []
        for _ in range(count):
            roll = rng.random()
            if roll < 0.4:
                factors.append(float(rng.uniform(1.2, 1.6)))
            elif roll < 0.8:
                factors.append(float(rng.uniform(0.6, 0.85)))
            else:
                factors.append(1.0)
        if all(abs(f - 1.0) < 1e-6 for f in factors):
            idx = int(rng.integers(0, count))
            factors[idx] = float(rng.uniform(1.2, 1.6))

        a_objs = clone_objects(objs)
        b_objs = clone_objects(a_objs)
        for i, f in enumerate(factors):
            b_objs[i] = apply_scale(b_objs[i], f)
        c_objs = clone_objects(b_objs)
        for i, f in enumerate(factors):
            c_objs[i] = apply_scale(c_objs[i], f)

        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [[size(obj) for obj in s.objects] for s in scenes]
        meta = build_rule_meta(
            self,
            "R3",
            3,
            involved,
            ["r"],
            ["size(Oi)"],
            "per-object-scale",
            {"count": count, "layout": layout_name, "factors": factors},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []
        factors = meta.get("pattern_params", {}).get("factors")
        if not factors:
            return [], []
        f = np.array(factors, dtype=float)
        if f.shape[0] != len(scene_c.objects):
            return [], []
        safe_f = np.where(np.abs(f) < 1e-6, 1.0, f)

        def build_with(scale_factors: np.ndarray) -> Scene:
            objs = clone_objects(scene_c.objects)
            for i, factor in enumerate(scale_factors.tolist()):
                objs[i] = apply_scale(objs[i], float(factor))
            return scene_from_objects(objs)

        step_back = 1.0 / safe_f
        reverse = 1.0 / (safe_f ** 2)
        over = safe_f ** 2
        distractors = [build_with(x) for x in [step_back, reverse, over]]
        reasons = [
            "尺度未继续变化，停留在上一帧",
            "尺度变化方向反向",
            "尺度变化幅度过大",
        ]
        return distractors, reasons


@dataclass
class R3_10ShapeShift(Rule):
    def __init__(self) -> None:
        super().__init__("R3-10", RuleDifficulty.COMPLEX, "多对象形状变化", "多对象形状按位置延续转换")

    def sample_params(self, rng) -> Dict:
        count = int(rng.integers(2, 6))
        return {"count": count}

    @staticmethod
    def _regular_polygon(count: int, radius: float) -> List[np.ndarray]:
        angles = [math.pi / 2 - i * 2 * math.pi / count for i in range(count)]
        return [np.array([radius * math.cos(a), radius * math.sin(a), 0.0]) for a in angles]

    def _layout_positions(self, count: int) -> tuple[List[np.ndarray], str]:
        if count == 2:
            return [np.array([-0.6, 0.0, 0.0]), np.array([0.6, 0.0, 0.0])], "line"
        if count == 3:
            return self._regular_polygon(3, 0.65), "triangle"
        if count == 4:
            return [
                np.array([-0.7, -0.45, 0.0]),
                np.array([0.7, -0.45, 0.0]),
                np.array([0.7, 0.45, 0.0]),
                np.array([-0.7, 0.45, 0.0]),
            ], "rectangle"
        if count == 5:
            base = self._regular_polygon(5, 0.7)
            star_order = [0, 2, 4, 1, 3]
            return [base[i] for i in star_order], "pentagram"
        raise ValueError(f"Unsupported object count {count}")

    @staticmethod
    def _derangement_mapping(rng) -> Dict[str, str]:
        shapes = list(SHAPES)
        while True:
            perm = rng.permutation(shapes)
            if all(a != b for a, b in zip(shapes, perm.tolist())):
                return {a: b for a, b in zip(shapes, perm.tolist())}

    def generate_triplet(self, params, rng):
        count = int(params["count"])
        positions, layout_name = self._layout_positions(count)
        shape_map = self._derangement_mapping(rng)
        a_shapes = rng.choice(SHAPES, size=count, replace=False).tolist()
        b_shapes = [shape_map[s] for s in a_shapes]
        c_shapes = [shape_map[s] for s in b_shapes]
        objs = [random_object(rng, shape=shape) for shape in a_shapes]
        involved = list(range(count))

        for obj, pos in zip(objs, positions):
            obj.p = pos
            obj.rotation = obj.rotation + rng.uniform(-math.pi / 18, math.pi / 18, size=3)
            obj.r = obj.r * rng.uniform(0.9, 1.1, size=3)

        def build_frame(shapes: List[str]) -> List:
            arranged = clone_objects(objs)
            for obj, shape in zip(arranged, shapes):
                obj.shape = shape
            return arranged

        a_objs = build_frame(a_shapes)
        b_objs = build_frame(b_shapes)
        c_objs = build_frame(c_shapes)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [a_shapes, b_shapes, c_shapes]
        meta = build_rule_meta(
            self,
            "R3",
            3,
            involved,
            ["s"],
            ["shape_map"],
            "mapping",
            {"count": count, "layout": layout_name, "map": shape_map},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class R3_11SinePositionShift(Rule):
    def __init__(self) -> None:
        super().__init__("R3-11", RuleDifficulty.COMPLEX, "正弦位置转换", "位置沿 sin 采样点连续滑动")

    @staticmethod
    def _sine_positions() -> tuple[List[np.ndarray], List[int]]:
        angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
        x_vals = np.linspace(-0.8, 0.8, len(angles_deg))
        positions = []
        for idx, deg in enumerate(angles_deg):
            rad = math.radians(deg)
            positions.append(np.array([x_vals[idx], 0.6 * math.sin(rad), 0.0]))
        return positions, angles_deg

    def sample_params(self, rng) -> Dict:
        return {}

    def generate_triplet(self, params, rng):
        positions, angles_deg = self._sine_positions()
        n = len(positions)
        step = 1 if rng.random() < 0.5 else -1
        if step == 1:
            start = int(rng.integers(0, n - 2))
            if start > n - 3:
                start = n - 3
        else:
            start = int(rng.integers(2, n - 1))
            if start < 2:
                start = 2

        idx_a = [start, start + 1]
        idx_b = [start + step, start + 1 + step]
        idx_c = [start + 2 * step, start + 1 + 2 * step]

        objs = init_objects(rng, 2, m=2)
        involved = [0, 1]

        def build_frame(indices: List[int]) -> List:
            arranged = clone_objects(objs)
            for obj, idx in zip(arranged, indices):
                obj.p = positions[int(idx)]
            return arranged

        a_objs = build_frame(idx_a)
        b_objs = build_frame(idx_b)
        c_objs = build_frame(idx_c)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [idx_a, idx_b, idx_c]
        meta = build_rule_meta(
            self,
            "R3",
            2,
            involved,
            ["p"],
            ["sin-pos"],
            "sine-adjacent",
            {"angles_deg": angles_deg, "step": step, "start": start},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if len(scene_c.objects) < 2:
            return [], []
        positions, _angles = self._sine_positions()
        params = meta.get("pattern_params", {})
        step = int(params.get("step", 1))
        v = meta.get("v", {})
        idx_a = v.get("v1")
        idx_b = v.get("v2")
        idx_c = v.get("v3")
        if not (idx_a and idx_b and idx_c):
            return [], []

        def build_from(indices: List[int]) -> Scene:
            objs = clone_objects(scene_c.objects)
            for obj, idx in zip(objs, indices):
                obj.p = positions[int(idx)]
            return scene_from_objects(objs)

        gap_base = int(idx_c[0])
        gap_idx = gap_base + 2 if gap_base + 2 < len(positions) else gap_base - 2
        if gap_idx < 0 or gap_idx >= len(positions):
            gap_idx = int(idx_a[0])
        gap_pair = [gap_base, gap_idx]

        candidates = [idx_b, idx_a, gap_pair]
        distractors = [build_from(pair) for pair in candidates]
        reasons = [
            "位置未继续滑动，停留在上一帧",
            "位置回退到上一阶段",
            "位置不连续，跳过中间点",
        ]
        return distractors, reasons


@dataclass(frozen=True)
class _R4Candidate:
    rule: Rule
    advance_fn: Callable[[Scene, Dict, Dict, Dict], Scene]
    value_fn: Callable[[Scene, Dict, Dict, Dict], Union[Sequence[float], float]]
    state_fn: Callable[[Scene, Scene, Scene, Dict, Dict], Dict] | None = None


def _advance_r1_1(scene_c: Scene, _params: Dict, base_meta: Dict, _state: Dict) -> Scene:
    delta = float(base_meta.get("pattern_params", {}).get("delta", 0.0))
    objs = clone_objects(scene_c.objects)
    cur = size(objs[0])
    target = max(cur + delta, 1e-6)
    scale = (target / cur) ** (1 / 3) if cur > 1e-6 else 1.0
    objs[0] = apply_scale(objs[0], scale)
    return scene_from_objects(objs)


def _advance_r1_2(scene_c: Scene, params: Dict, _base_meta: Dict, _state: Dict) -> Scene:
    axis_idx = int(params.get("axis", 0))
    theta = float(params.get("theta", 0.0))
    delta = np.zeros(3)
    delta[axis_idx] = theta
    objs = clone_objects(scene_c.objects)
    objs[0] = apply_rotation(objs[0], delta)
    return scene_from_objects(objs)


def _advance_r1_4(scene_c: Scene, params: Dict, _base_meta: Dict, _state: Dict) -> Scene:
    delta = np.array(params.get("delta", [0.0, 0.0, 0.0]), dtype=float)
    objs = clone_objects(scene_c.objects)
    objs[0] = apply_translation(objs[0], delta)
    return scene_from_objects(objs)


def _advance_r1_5(scene_c: Scene, _params: Dict, base_meta: Dict, _state: Dict) -> Scene:
    delta = float(base_meta.get("pattern_params", {}).get("delta", 0.0))
    objs = clone_objects(scene_c.objects)
    objs[0].density = max(objs[0].density + delta, 1e-3)
    return scene_from_objects(objs)


def _advance_r2_1(scene_c: Scene, params: Dict, _base_meta: Dict, _state: Dict) -> Scene:
    k = float(params.get("k", 1.0))
    objs = clone_objects(scene_c.objects)
    if len(objs) < 2:
        return scene_from_objects(objs)
    obj0, obj1 = objs[0], objs[1]
    delta = obj1.p - obj0.p
    base_dist = float(np.linalg.norm(delta))
    if base_dist < 1e-6:
        direction_vec = np.array([1.0, 0.0, 0.0])
    else:
        direction_vec = delta / base_dist
    mid = 0.5 * (obj0.p + obj1.p)
    new_dist = base_dist * k
    obj0.p = mid - direction_vec * new_dist / 2
    obj1.p = mid + direction_vec * new_dist / 2
    return scene_from_objects(objs)


def _advance_r2_2(scene_c: Scene, params: Dict, _base_meta: Dict, _state: Dict) -> Scene:
    factor = float(params.get("factor", 1.0))
    axis_idx = int(params.get("axis", 0))
    squeeze = 1.0 / math.sqrt(factor) if factor > 1e-6 else 1.0
    scale = np.ones(3)
    scale[axis_idx] = factor
    for i in range(3):
        if i != axis_idx:
            scale[i] = squeeze
    objs = clone_objects(scene_c.objects)
    objs[0] = apply_scale(objs[0], scale)
    return scene_from_objects(objs)


def _advance_r2_3(scene_c: Scene, params: Dict, _base_meta: Dict, _state: Dict) -> Scene:
    theta = float(params.get("theta", 0.0))
    objs = clone_objects(scene_c.objects)
    if len(objs) < 2:
        return scene_from_objects(objs)
    obj0, obj1 = objs[0], objs[1]
    base_dist = dist(obj0, obj1)
    if base_dist < 1e-6:
        return scene_from_objects(objs)
    base_dir = direction(obj0, obj1)
    rot = np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    new_dir = rot @ base_dir
    obj1.p = obj0.p + new_dir * base_dist
    return scene_from_objects(objs)


def _advance_r3_2(scene_c: Scene, _params: Dict, _base_meta: Dict, _state: Dict) -> Scene:
    perms = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    base_positions = [-0.8, 0.0, 0.8]
    objs = clone_objects(scene_c.objects)
    ordering = order_indices_x(objs)
    try:
        idx = perms.index(ordering)
    except ValueError:
        idx = 0
    next_perm = perms[(idx + 1) % len(perms)]
    for rank, obj_idx in enumerate(next_perm):
        objs[obj_idx].p[0] = base_positions[rank]
    return scene_from_objects(objs)


def _advance_r3_3(scene_c: Scene, params: Dict, _base_meta: Dict, state: Dict) -> Scene:
    k_left = float(params.get("k_left", 1.0))
    k_right = float(params.get("k_right", 1.0))
    base_cent = np.array(state.get("base_cent", centroid(scene_c.objects)), dtype=float)
    objs = []
    for obj in scene_c.objects:
        new_obj = obj.copy()
        delta = new_obj.p - base_cent
        if delta[0] < -1e-6:
            factor = k_left
        elif delta[0] > 1e-6:
            factor = k_right
        else:
            factor = 1.0
        new_obj.p = base_cent + delta * factor
        objs.append(new_obj)
    return scene_from_objects(objs)


def _value_size(scene: Scene, _params: Dict, _base_meta: Dict, _state: Dict) -> float:
    return float(size(scene.objects[0])) if scene.objects else 0.0


def _value_r(scene: Scene, _params: Dict, _base_meta: Dict, _state: Dict) -> Sequence[float]:
    return scene.objects[0].r.tolist() if scene.objects else [0.0, 0.0, 0.0]


def _value_rotation_axis(scene: Scene, params: Dict, _base_meta: Dict, _state: Dict) -> float:
    axis_idx = int(params.get("axis", 0))
    if not scene.objects:
        return 0.0
    return float(scene.objects[0].rotation[axis_idx])


def _value_position(scene: Scene, _params: Dict, _base_meta: Dict, _state: Dict) -> Sequence[float]:
    return scene.objects[0].p.tolist() if scene.objects else [0.0, 0.0, 0.0]


def _value_density(scene: Scene, _params: Dict, _base_meta: Dict, _state: Dict) -> float:
    return float(scene.objects[0].density) if scene.objects else 0.0


def _value_dist(scene: Scene, _params: Dict, _base_meta: Dict, _state: Dict) -> float:
    if len(scene.objects) < 2:
        return 0.0
    return float(dist(scene.objects[0], scene.objects[1]))


def _value_aspect_ratio(scene: Scene, _params: Dict, _base_meta: Dict, _state: Dict) -> Sequence[float]:
    return aspect_ratio(scene.objects[0]).tolist() if scene.objects else [0.0, 0.0]


def _value_direction(scene: Scene, _params: Dict, _base_meta: Dict, _state: Dict) -> Sequence[float]:
    if len(scene.objects) < 2:
        return [0.0, 0.0, 0.0]
    return direction(scene.objects[0], scene.objects[1]).tolist()


def _value_order(scene: Scene, _params: Dict, _base_meta: Dict, _state: Dict) -> Sequence[int]:
    return order_indices_x(scene.objects) if scene.objects else []


def _value_dist_set(scene: Scene, _params: Dict, _base_meta: Dict, _state: Dict) -> Sequence[float]:
    if len(scene.objects) < 3:
        return [0.0, 0.0, 0.0]
    return [
        float(dist(scene.objects[0], scene.objects[1])),
        float(dist(scene.objects[0], scene.objects[2])),
        float(dist(scene.objects[1], scene.objects[2])),
    ]


def _build_r4_candidates() -> List[_R4Candidate]:
    return [
        _R4Candidate(R1_1ScaleArithmetic(), _advance_r1_1, _value_size),
        _R4Candidate(R1_2FixedAxisRotation(), _advance_r1_2, _value_rotation_axis),
        _R4Candidate(R1_4TranslationArithmetic(), _advance_r1_4, _value_position),
        _R4Candidate(R1_5DensityArithmetic(), _advance_r1_5, _value_density),
    ]


@dataclass
class R4_1SkipAhead(Rule):
    def __init__(self) -> None:
        super().__init__("R4-1", RuleDifficulty.COMPLEX, "两步外推预测", "从可外推规则中预测下下帧")
        self._candidates = _build_r4_candidates()
        self._candidate_map = {c.rule.rule_id: c for c in self._candidates}

    def sample_params(self, rng) -> Dict:
        idx = int(rng.integers(0, len(self._candidates)))
        candidate = self._candidates[idx]
        base_params = candidate.rule.sample_params(rng)
        return {"base_rule_id": candidate.rule.rule_id, "base_params": base_params}

    def generate_triplet(self, params, rng):
        base_rule_id = params.get("base_rule_id")
        base_params = params.get("base_params", {})
        candidate = self._candidate_map.get(base_rule_id, self._candidates[0])
        if not base_rule_id:
            base_rule_id = candidate.rule.rule_id
        scene_a, scene_b, scene_c, base_meta = candidate.rule.generate_triplet(base_params, rng)
        state = candidate.state_fn(scene_a, scene_b, scene_c, base_params, base_meta) if candidate.state_fn else {}
        scene_d = candidate.advance_fn(scene_c, base_params, base_meta, state)

        v = [candidate.value_fn(s, base_params, base_meta, state) for s in [scene_a, scene_b, scene_d]]
        base_involved = base_meta.get("involved_indices", list(range(len(scene_a.objects))))
        base_attrs = base_meta.get("base_attrs_used", [])
        derived_funcs = base_meta.get("derived_funcs", [])
        pattern_params = {
            "base_rule_id": base_rule_id,
            "base_rule_group": base_meta.get("rule_group"),
            "base_pattern": base_meta.get("pattern_type"),
            "base_params": base_params,
            "skip": 1,
        }
        meta = build_rule_meta(
            self,
            "R4",
            int(base_meta.get("K_R", len(base_involved))),
            base_involved,
            base_attrs,
            derived_funcs,
            "skip-ahead",
            pattern_params,
            v,
            [scene_a, scene_b, scene_d],
        )
        return scene_a, scene_b, scene_d, meta


@dataclass
class R4_2NodeSplitEvolution(Rule):
    def __init__(self) -> None:
        super().__init__("R4-2", RuleDifficulty.COMPLEX, "节点分裂演化", "节点按 1->2 分裂并延续")

    def sample_params(self, rng) -> Dict:
        count = int(rng.integers(1, 4))
        return {"count": count}

    @staticmethod
    def _split_objects(objs: Sequence, rng) -> List:
        split_scale = 0.5 ** (1.0 / 3.0)
        out = []
        for obj in objs:
            base = obj.copy()
            direction_vec = rng.normal(size=3)
            norm = float(np.linalg.norm(direction_vec))
            if norm < 1e-6:
                direction_vec = np.array([1.0, 0.0, 0.0])
            else:
                direction_vec = direction_vec / norm
            radius = 0.6 * float(np.linalg.norm(base.r))
            offset_mag = max(0.12, radius * 0.35)
            offset = direction_vec * offset_mag
            for sign in (-1.0, 1.0):
                child = base.copy()
                child.r = child.r * split_scale
                child.p = child.p + sign * offset
                child.rotation = rng.uniform(-math.pi / 4, math.pi / 4, size=3)
                out.append(child)
        _separate_objects_no_contact(out, rng, gap=0.25)  # 增大间距，让物体分开得更远
        rng.shuffle(out)
        return out

    def generate_triplet(self, params, rng):
        count = int(params.get("count", 1))
        objs = [random_object(rng) for _ in range(count)]

        a_objs = clone_objects(objs)
        b_objs = self._split_objects(a_objs, rng)
        c_objs = self._split_objects(b_objs, rng)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [len(a_objs), len(b_objs), len(c_objs)]
        involved = list(range(len(c_objs)))
        meta = build_rule_meta(
            self,
            "R4",
            len(involved),
            involved,
            ["s", "d", "r", "p", "R"],
            ["count(O)", "size(Oi)"],
            "split-evolution",
            {"split_factor": 2, "volume_ratio": 0.5, "count_a": len(a_objs)},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []
        objs = clone_objects(scene_c.objects)
        attrs = meta.get("pattern_params", {}).get("infect_attrs", [])
        if len(objs) < 2:
            return [], []

        target_count = max(1, len(objs) // 2)
        keep_indices = rng.choice(len(objs), size=target_count, replace=False)
        wrong_count = [objs[int(i)].copy() for i in keep_indices]

        wrong_shape = clone_objects(objs)
        idx_shape = int(rng.integers(0, len(wrong_shape)))
        shape_choices = [s for s in SHAPES if s != wrong_shape[idx_shape].shape]
        if shape_choices:
            wrong_shape[idx_shape] = switch_shape(wrong_shape[idx_shape], str(rng.choice(shape_choices)))

        wrong_density = clone_objects(objs)
        idx_density = int(rng.integers(0, len(wrong_density)))
        density_factor = float(rng.uniform(0.35, 0.55) if rng.random() < 0.5 else rng.uniform(1.8, 2.6))
        size_factor = float(rng.uniform(0.5, 0.7) if rng.random() < 0.5 else rng.uniform(1.5, 1.9))
        adjusted = apply_scale(wrong_density[idx_density], size_factor)
        adjusted.density = max(adjusted.density * density_factor, 1e-3)
        wrong_density[idx_density] = adjusted

        distractors = [
            scene_from_objects(wrong_count),
            scene_from_objects(wrong_shape),
            scene_from_objects(wrong_density),
        ]
        reasons = [
            "分裂数量不足，未按倍增延续",
            "分裂后形状被错误更换",
            "分裂后密度或体积未保持",
        ]
        return distractors, reasons


@dataclass
class R4_3NodeFusionEvolution(Rule):
    def __init__(self) -> None:
        super().__init__("R4-3", RuleDifficulty.COMPLEX, "节点融合演化", "节点按 2->1 融合并延续")

    def sample_params(self, rng) -> Dict:
        count = int(rng.integers(3, 7))
        return {"count": count}

    @staticmethod
    def _build_shapes(count: int, rng) -> List[str]:
        max_pairs = min(3, count // 2)
        pair_target = int(rng.integers(1, max_pairs + 1))
        dup_count_map = {1: 3, 2: 4, 3: 6}
        dup_count = dup_count_map[pair_target]
        fusion_shape = str(rng.choice(SHAPES))

        shapes = [fusion_shape] * dup_count
        remaining = count - dup_count
        if remaining > 0:
            other_shapes = [s for s in SHAPES if s != fusion_shape]
            if remaining > len(other_shapes):
                extra = rng.choice(other_shapes, size=remaining, replace=True).tolist()
            else:
                extra = rng.choice(other_shapes, size=remaining, replace=False).tolist()
            shapes.extend(extra)
        rng.shuffle(shapes)
        return shapes

    @staticmethod
    def _fuse_objects(objs: Sequence, rng) -> List:
        by_shape: Dict[str, List[int]] = {}
        for idx, obj in enumerate(objs):
            by_shape.setdefault(obj.shape, []).append(idx)

        fused = []
        used = set()
        candidates = [shape for shape, indices in by_shape.items() if len(indices) >= 2]
        if not candidates:
            return [obj.copy() for obj in objs]
        preferred = [shape for shape in candidates if len(by_shape[shape]) >= 3]
        fuse_shape = str(rng.choice(preferred if preferred else candidates))
        indices = by_shape[fuse_shape]
        rng.shuffle(indices)
        if len(indices) < 2:
            return [obj.copy() for obj in objs]
        idx_a, idx_b = sorted([int(indices[0]), int(indices[1])])
        obj_a = objs[idx_a]
        obj_b = objs[idx_b]
        base = obj_a.copy()
        base.r = obj_a.r + obj_b.r
        base.p = (obj_a.p + obj_b.p) / 2.0
        base.rotation = rng.uniform(-math.pi / 4, math.pi / 4, size=3)
        weight_a = float(obj_a.density) * float(obj_a.volume())
        weight_b = float(obj_b.density) * float(obj_b.volume())
        fused_volume = float(base.volume())
        if fused_volume > 1e-6:
            base.density = (weight_a + weight_b) / fused_volume
        else:
            base.density = float(obj_a.density)

        for idx, obj in enumerate(objs):
            if idx == idx_a:
                fused.append(base)
            elif idx == idx_b:
                continue
            else:
                fused.append(obj.copy())

        _separate_objects_no_contact(fused, rng, gap=0.25)  # 增大间距，让物体分开得更远
        return fused

    def generate_triplet(self, params, rng):
        count = max(3, int(params.get("count", 2)))
        objs = [random_object(rng) for _ in range(count)]
        shapes = self._build_shapes(count, rng)
        for i, shape in enumerate(shapes):
            objs[i].shape = shape
        density_by_shape: Dict[str, float] = {}
        for obj in objs:
            if obj.shape not in density_by_shape:
                density_by_shape[obj.shape] = float(rng.uniform(0.8, 1.2))
            obj.density = density_by_shape[obj.shape]

        _separate_objects_no_contact(objs, rng, gap=0.25)  # 增大间距，让物体分开得更远
        a_objs = clone_objects(objs)
        b_objs = self._fuse_objects(a_objs, rng)
        c_objs = self._fuse_objects(b_objs, rng)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [len(a_objs), len(b_objs), len(c_objs)]
        involved = list(range(len(c_objs)))
        meta = build_rule_meta(
            self,
            "R4",
            len(involved),
            involved,
            ["s", "d", "r", "p", "R"],
            ["count(shape)", "size(Oi)"],
            "fusion-evolution",
            {"count_a": len(a_objs)},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []
        objs = clone_objects(scene_c.objects)

        wrong_count = clone_objects(objs)
        dup_idx = int(rng.integers(0, len(wrong_count)))
        dup = wrong_count[dup_idx].copy()
        delta = rng.uniform(0.08, 0.16, size=3) * rng.choice([-1, 1], size=3)
        dup = apply_translation(dup, delta)
        wrong_count.append(dup)
        rng.shuffle(wrong_count)

        wrong_shape = clone_objects(objs)
        idx_shape = int(rng.integers(0, len(wrong_shape)))
        shape_choices = [s for s in SHAPES if s != wrong_shape[idx_shape].shape]
        if shape_choices:
            wrong_shape[idx_shape] = switch_shape(wrong_shape[idx_shape], str(rng.choice(shape_choices)))

        wrong_density = clone_objects(objs)
        idx_density = int(rng.integers(0, len(wrong_density)))
        density_factor = float(rng.uniform(0.35, 0.55) if rng.random() < 0.5 else rng.uniform(1.8, 2.6))
        size_factor = float(rng.uniform(0.5, 0.7) if rng.random() < 0.5 else rng.uniform(1.5, 1.9))
        adjusted = apply_scale(wrong_density[idx_density], size_factor)
        adjusted.density = max(adjusted.density * density_factor, 1e-3)
        wrong_density[idx_density] = adjusted

        distractors = [
            scene_from_objects(wrong_count),
            scene_from_objects(wrong_shape),
            scene_from_objects(wrong_density),
        ]
        reasons = [
            "融合数量不足，仍保留过多对象",
            "融合后形状被错误改变",
            "融合后密度或体积未保持",
        ]
        return distractors, reasons


@dataclass
class R4_4FormationEvolution(Rule):
    _EDGE_TO_FORMATION = {
        6: ("hexagon", 6),
        5: ("star", 5),
        4: ("rectangle", 4),
        3: ("triangle", 3),
        1: ("line", 2),
    }

    def __init__(self) -> None:
        super().__init__("R4-4", RuleDifficulty.COMPLEX, "几何阵型演化", "阵型边数按递减规律演化")

    def sample_params(self, rng) -> Dict:
        return {}

    @classmethod
    def _formation_spec(cls, edge_count: int) -> tuple[str, int]:
        if edge_count not in cls._EDGE_TO_FORMATION:
            raise ValueError(f"Unsupported edge count {edge_count}")
        return cls._EDGE_TO_FORMATION[edge_count]

    @staticmethod
    def _positions_for(formation: str, count: int, rng) -> List[np.ndarray]:
        if formation == "line":
            xs = np.linspace(-0.8, 0.8, count)
            pts = np.stack([xs, np.zeros(count), np.zeros(count)], axis=1)
        else:
            angles = np.linspace(0, 2 * math.pi, count, endpoint=False)
            if formation == "star":
                outer = 0.9
                inner = 0.45
                radii = np.array([outer if i % 2 == 0 else inner for i in range(count)], dtype=float)
            else:
                radii = np.full(count, 0.85, dtype=float)
            pts = np.stack([radii * np.cos(angles), radii * np.sin(angles), np.zeros(count)], axis=1)

        angle = float(rng.uniform(0.0, 2 * math.pi))
        rot = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
        pts = pts @ rot.T
        return [pts[i] for i in range(count)]

    def _build_scene(self, objs: Sequence, edge_count: int, rng) -> Scene:
        formation, count = self._formation_spec(edge_count)
        if len(objs) != count:
            raise ValueError("Object count does not match formation spec.")
        positions = self._positions_for(formation, count, rng)
        if len(positions) >= 2:
            min_scale = 1.0
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    base_dist = float(np.linalg.norm(positions[i] - positions[j]))
                    if base_dist < 1e-6:
                        continue
                    required = approx_radius(objs[i]) + approx_radius(objs[j]) + 0.06
                    min_scale = max(min_scale, required / base_dist)
            if min_scale > 1.0:
                positions = [pos * min_scale for pos in positions]
        arranged = []
        for obj, pos in zip(objs, positions):
            new_obj = obj.copy()
            new_obj.p = np.array(pos, dtype=float)
            new_obj.rotation = rng.uniform(-math.pi / 4, math.pi / 4, size=3)
            arranged.append(new_obj)
        return scene_from_objects(arranged)

    def generate_triplet(self, params, rng):
        edge_a = 6
        edges = list(self._EDGE_TO_FORMATION.keys())
        while True:
            d1 = int(rng.integers(0, 4))
            edge_b = edge_a - d1
            d2 = d1 - 1
            edge_c = edge_b - d2
            if edge_b in edges and edge_c in edges:
                break
        formation_b, count_b = self._formation_spec(edge_b)
        formation_c, count_c = self._formation_spec(edge_c)

        a_objs = [random_object(rng) for _ in range(6)]
        scene_a = self._build_scene(a_objs, edge_a, rng)

        pick_b = rng.choice(len(a_objs), size=count_b, replace=False)
        b_objs = [a_objs[int(i)].copy() for i in pick_b]
        scene_b = self._build_scene(b_objs, edge_b, rng)

        pick_c = rng.choice(len(b_objs), size=count_c, replace=False)
        c_objs = [b_objs[int(i)].copy() for i in pick_c]
        scene_c = self._build_scene(c_objs, edge_c, rng)

        v = [edge_a, edge_b, edge_c]
        involved = list(range(len(scene_c.objects)))
        meta = build_rule_meta(
            self,
            "R4",
            len(involved),
            involved,
            ["p"],
            ["formation_edges", "count(O)"],
            "formation-evolution",
            {
                "edge_a": edge_a,
                "edge_b": edge_b,
                "edge_c": edge_c,
                "reduce_ab": d1,
                "reduce_bc": d2,
                "formation_b": formation_b,
                "formation_c": formation_c,
            },
            v,
            [scene_a, scene_b, scene_c],
        )
        return scene_a, scene_b, scene_c, meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        params = meta.get("pattern_params", {})
        edge_a = int(params.get("edge_a", 6))
        edge_b = int(params.get("edge_b", 5))
        d1 = int(params.get("reduce_ab", 1))
        edges = list(self._EDGE_TO_FORMATION.keys())

        wrong_edge_c = edge_b - d1
        if wrong_edge_c not in edges:
            wrong_edge_c = edge_b
        wrong_count = self._formation_spec(wrong_edge_c)[1]

        base_objs = clone_objects(scene_c.objects)
        if len(base_objs) >= wrong_count:
            pick = rng.choice(len(base_objs), size=wrong_count, replace=False)
            wrong_count_objs = [base_objs[int(i)].copy() for i in pick]
        else:
            wrong_count_objs = base_objs[:]
            while len(wrong_count_objs) < wrong_count:
                wrong_count_objs.append(random_object(rng))
        wrong_count_scene = self._build_scene(wrong_count_objs, wrong_edge_c, rng)

        wrong_shape = clone_objects(scene_c.objects)
        idx_shape = int(rng.integers(0, len(wrong_shape)))
        shape_choices = [s for s in SHAPES if s != wrong_shape[idx_shape].shape]
        if shape_choices:
            wrong_shape[idx_shape] = switch_shape(wrong_shape[idx_shape], str(rng.choice(shape_choices)))
        wrong_shape_scene = scene_from_objects(wrong_shape)

        wrong_density = clone_objects(scene_c.objects)
        idx_density = int(rng.integers(0, len(wrong_density)))
        density_factor = float(rng.uniform(0.35, 0.55) if rng.random() < 0.5 else rng.uniform(1.8, 2.6))
        size_factor = float(rng.uniform(0.5, 0.7) if rng.random() < 0.5 else rng.uniform(1.5, 1.9))
        adjusted = apply_scale(wrong_density[idx_density], size_factor)
        adjusted.density = max(adjusted.density * density_factor, 1e-3)
        wrong_density[idx_density] = adjusted
        wrong_density_scene = scene_from_objects(wrong_density)

        distractors = [
            wrong_count_scene,
            wrong_shape_scene,
            wrong_density_scene,
        ]
        reasons = [
            "阵型边数减少规律错误",
            "形状与阵型演化不匹配",
            "密度或体积变化破坏规则",
        ]
        return distractors, reasons


@dataclass
class R4_5ContactInfection(Rule):
    def __init__(self) -> None:
        super().__init__("R4-5", RuleDifficulty.COMPLEX, "接触式属性传染", "高尺寸物体向低尺寸物体传染属性")

    def sample_params(self, rng) -> Dict:
        count = int(rng.integers(2, 5))
        weights = [0.08, 0.42, 0.32, 0.18]
        attr_count = int(rng.choice([0, 1, 2, 3], p=weights))
        attrs = []
        if attr_count > 0:
            attrs = rng.choice(["r", "d", "R"], size=attr_count, replace=False).tolist()
        return {"count": count, "infect_attrs": attrs}

    @staticmethod
    def _ensure_size_contrast(objs: Sequence, rng) -> None:
        sizes = np.array([size(o) for o in objs], dtype=float)
        if len(sizes) < 2:
            return
        max_idx = int(np.argmax(sizes))
        min_idx = int(np.argmin(sizes))
        if sizes[min_idx] <= 1e-6:
            return
        target_ratio = float(rng.uniform(1.8, 2.4))
        cur_ratio = sizes[max_idx] / sizes[min_idx]
        if cur_ratio < target_ratio:
            scale = (target_ratio / cur_ratio) ** (1 / 3)
            objs[max_idx] = apply_scale(objs[max_idx], scale)

    @staticmethod
    def _ensure_density_contrast(objs: Sequence, rng) -> None:
        if len(objs) < 2:
            return
        sizes = np.array([size(o) for o in objs], dtype=float)
        max_idx = int(np.argmax(sizes))
        min_idx = int(np.argmin(sizes))
        if max_idx == min_idx:
            min_idx = (max_idx + 1) % len(objs)
        low = max(float(objs[min_idx].density) * float(rng.uniform(0.4, 0.7)), 1e-3)
        target_ratio = float(rng.uniform(1.8, 2.4))
        high = max(low * target_ratio, 1e-3)
        objs[min_idx].density = low
        objs[max_idx].density = high

    @staticmethod
    def _infect(scene: Scene, attrs: Sequence[str], rng) -> tuple[Scene, int]:
        objs = clone_objects(scene.objects)
        sizes = np.array([size(o) for o in objs], dtype=float)
        source_idx = int(np.argmax(sizes)) if sizes.size else 0
        source = objs[source_idx]
        for idx, obj in enumerate(objs):
            if idx == source_idx:
                continue
            if size(obj) >= size(source):
                continue
            if "r" in attrs:
                obj.r = source.r.copy()
            if "d" in attrs:
                obj.density = float(source.density)
            if "R" in attrs:
                obj.rotation = source.rotation.copy()
            objs[idx] = obj
        return scene_from_objects(objs), source_idx

    @staticmethod
    def _add_random(scene: Scene, rng, count: int = 2) -> Scene:
        objs = clone_objects(scene.objects)
        for _ in range(count):
            objs.append(random_object(rng))
        _separate_objects_no_contact(objs, rng, gap=0.25)  # 增大间距，让物体分开得更远
        return scene_from_objects(objs)

    def generate_triplet(self, params, rng):
        count = int(params.get("count", 2))
        attrs = list(params.get("infect_attrs", []))
        objs = [random_object(rng) for _ in range(count)]
        self._ensure_size_contrast(objs, rng)
        self._ensure_density_contrast(objs, rng)
        _separate_objects_no_contact(objs, rng, gap=0.25)  # 增大间距，让物体分开得更远
        scene_a = scene_from_objects(objs)

        infected_b, source_b = self._infect(scene_a, attrs, rng)
        scene_b = self._add_random(infected_b, rng, count=2)

        infected_c, source_c = self._infect(scene_b, attrs, rng)
        scene_c = infected_c

        v = [len(scene_a.objects), len(scene_b.objects), len(scene_c.objects)]
        involved = list(range(len(scene_c.objects)))
        meta = build_rule_meta(
            self,
            "R4",
            len(involved),
            involved,
            ["r", "d", "R"],
            ["infect(r)", "infect(d)", "infect(R)", "count(O)"],
            "infection-evolution",
            {
                "infect_attrs": attrs,
                "added_each": [2, 0],
                "source_indices": [source_b, source_c],
            },
            v,
            [scene_a, scene_b, scene_c],
        )
        return scene_a, scene_b, scene_c, meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []
        objs = clone_objects(scene_c.objects)
        attrs = meta.get("pattern_params", {}).get("infect_attrs", [])

        if len(objs) > 2:
            drop = rng.choice(len(objs), size=2, replace=False)
            wrong_count = [obj for i, obj in enumerate(objs) if i not in set(int(x) for x in drop)]
        else:
            wrong_count = objs + [random_object(rng)]
        _separate_objects_no_contact(wrong_count, rng, gap=0.25)  # 增大间距

        wrong_size = clone_objects(objs)
        idx_size = int(rng.integers(0, len(wrong_size)))
        scale = float(rng.uniform(0.5, 0.7) if rng.random() < 0.5 else rng.uniform(1.5, 1.9))
        wrong_size[idx_size] = apply_scale(wrong_size[idx_size], scale)
        _separate_objects_no_contact(wrong_size, rng, gap=0.25)  # 增大间距

        wrong_density = clone_objects(objs)
        idx_density = int(rng.integers(0, len(wrong_density)))
        if "R" in attrs:
            wrong_density[idx_density] = apply_rotation(
                wrong_density[idx_density], rng.uniform(0.25, 0.6, size=3)
            )
        else:
            density_factor = float(rng.uniform(0.35, 0.55) if rng.random() < 0.5 else rng.uniform(1.8, 2.6))
            wrong_density[idx_density] = wrong_density[idx_density].copy()
            wrong_density[idx_density].density = max(wrong_density[idx_density].density * density_factor, 1e-3)
        _separate_objects_no_contact(wrong_density, rng, gap=0.25)  # 增大间距

        distractors = [
            scene_from_objects(wrong_count),
            scene_from_objects(wrong_size),
            scene_from_objects(wrong_density),
        ]
        reasons = [
            "每帧新增数量不符合规则",
            "尺寸传染关系被破坏",
            "位姿或密度传染关系被破坏",
        ]
        return distractors, reasons


@dataclass
class R4_6AdvancedOrbitalRotation(Rule):
    def __init__(self) -> None:
        super().__init__("R4-6", RuleDifficulty.COMPLEX, "进阶行星公转", "公转中心按相邻最大尺寸切换")

    def sample_params(self, rng) -> Dict:
        return {"count": 3}

    @staticmethod
    def _unique_samples(rng, count: int, low: float, high: float, min_gap: float) -> List[float]:
        vals: List[float] = []
        attempts = 0
        while len(vals) < count and attempts < 200:
            candidate = float(rng.uniform(low, high))
            if all(abs(candidate - v) > min_gap for v in vals):
                vals.append(candidate)
            attempts += 1
        if len(vals) < count:
            vals = np.linspace(low, high, count).tolist()
        return vals

    @staticmethod
    def _rotate_about_center(p: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
        vec = p - center
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x = vec[0] * cos_a - vec[1] * sin_a
        y = vec[0] * sin_a + vec[1] * cos_a
        return center + np.array([x, y, vec[2]])

    @classmethod
    def _rotate_step_scene(cls, scene: Scene, center_idx: int, deltas: Sequence[float]) -> Scene:
        objs = clone_objects(scene.objects)
        if not objs:
            return scene_from_objects(objs)
        center_idx = int(center_idx)
        if center_idx < 0 or center_idx >= len(objs):
            return scene_from_objects(objs)
        center_p = objs[center_idx].p.copy()
        for i, obj in enumerate(objs):
            if i == center_idx:
                continue
            angle = float(deltas[i]) if i < len(deltas) else 0.0
            obj.p = cls._rotate_about_center(obj.p, center_p, angle)
            objs[i] = obj
        return scene_from_objects(objs)

    @staticmethod
    def _adjacent_by_radius(order: Sequence[int], center_idx: int) -> List[int]:
        if not order:
            return []
        try:
            pos = order.index(int(center_idx))
        except ValueError:
            return []
        neighbors = []
        if pos > 0:
            neighbors.append(order[pos - 1])
        if pos + 1 < len(order):
            neighbors.append(order[pos + 1])
        return neighbors

    @classmethod
    def _next_center_idx(cls, order: Sequence[int], objs: Sequence[ObjectState], center_idx: int) -> int:
        neighbors = cls._adjacent_by_radius(order, center_idx)
        if not neighbors:
            return int(center_idx)
        if len(neighbors) == 1:
            return int(neighbors[0])
        return int(max(neighbors, key=lambda idx: size(objs[int(idx)])))

    @staticmethod
    def _scene_from_frame(frame: Dict) -> Scene:
        objs = []
        for obj in frame.get("objects", []):
            objs.append(
                ObjectState(
                    shape=obj["shape"],
                    r=np.array(obj["r"], dtype=float),
                    p=np.array(obj["p"], dtype=float),
                    rotation=np.array(obj["rotation_euler"], dtype=float),
                    density=float(obj.get("density", 1.0)),
                )
            )
        return Scene(objects=objs)

    def generate_triplet(self, params, rng):
        count = 3
        gap = 0.3  # 增大间距，确保几何体之间不得有任何接触
        for _ in range(40):
            objs = [random_object(rng) for _ in range(count)]
            # 增加 radii 之间的最小间距，确保初始位置就有足够距离，避免旋转后接触
            radii = self._unique_samples(rng, count, 0.5, 1.4, 0.3)  # 增大最小间距和范围
            angles = [float(rng.uniform(0.0, 2 * math.pi)) for _ in range(count)]
            delta_base = float(rng.uniform(math.pi / 8, math.pi / 4))
            deltas = self._unique_samples(rng, count, delta_base * 0.7, delta_base * 1.3, 0.1)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            deltas = [sign * d for d in deltas]

            for obj, r, ang in zip(objs, radii, angles):
                obj.p = np.array([r * math.cos(ang), r * math.sin(ang), 0.0])
            
            # 确保初始位置没有接触
            _separate_objects_no_contact(objs, rng, gap=gap)

            order = [idx for idx, _ in sorted(enumerate(radii), key=lambda kv: kv[1])]
            center_a = int(order[0])
            center_b = self._next_center_idx(order, objs, center_a)
            center_c = self._next_center_idx(order, objs, center_b)
            center_indices = [center_a, center_b, center_c]

            scene_a = scene_from_objects(clone_objects(objs))
            scene_b = self._rotate_step_scene(scene_a, center_b, deltas)
            scene_c = self._rotate_step_scene(scene_b, center_c, deltas)
            
            # 检查所有帧是否都没有接触（使用更大的 gap 确保安全距离）
            if (
                _all_non_contact(scene_a.objects, gap)
                and _all_non_contact(scene_b.objects, gap)
                and _all_non_contact(scene_c.objects, gap)
            ):
                break
        v = center_indices
        involved = list(range(len(scene_c.objects)))
        meta = build_rule_meta(
            self,
            "R4",
            len(involved),
            involved,
            ["p", "r"],
            ["orbit(theta_i)", "center_idx", "size(Oi)"],
            "orbit-center-shift",
            {
                "center_order": order,
                "center_indices": center_indices,
                "radii": radii,
                "angles": angles,
                "deltas": deltas,
            },
            v,
            [scene_a, scene_b, scene_c],
        )
        return scene_a, scene_b, scene_c, meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        frames = meta.get("frames", [])
        params = meta.get("pattern_params", {})
        deltas = params.get("deltas", [])
        center_indices = params.get("center_indices", [])
        center_order = params.get("center_order", [])

        scene_b = self._scene_from_frame(frames[1]) if len(frames) > 1 else scene_c

        if not center_indices or not deltas or not center_order:
            return [], []

        if len(center_indices) < 3:
            return [], []
        prev_center = center_indices[1]
        correct_center = center_indices[2]
        neighbors = self._adjacent_by_radius(center_order, prev_center)
        wrong_reason = "公转中心未选相邻最大尺寸"
        if len(neighbors) >= 2:
            wrong_center = neighbors[0] if neighbors[0] != correct_center else neighbors[1]
        else:
            non_adjacent = [
                idx for idx in center_order if idx not in neighbors and idx not in (prev_center, correct_center)
            ]
            if non_adjacent:
                wrong_center = non_adjacent[0]
                wrong_reason = "公转中心选择了不相邻物体"
            else:
                wrong_center = correct_center

        wrong_skip = self._rotate_step_scene(scene_b, wrong_center, deltas)
        wrong_hold = self._rotate_step_scene(scene_b, center_indices[1], deltas)
        scaled = [d * 0.3 for d in deltas]
        wrong_angle = self._rotate_step_scene(scene_b, center_indices[2], scaled)

        distractors = [wrong_skip, wrong_hold, wrong_angle]
        reasons = [
            wrong_reason,
            "公转中心未按规则切换",
            "公转角速度不符合规则",
        ]
        return distractors, reasons


@dataclass
class R4_7SymmetryTransform(Rule):
    def __init__(self) -> None:
        super().__init__("R4-7", RuleDifficulty.COMPLEX, "对称转换", "左右镜像动作反向同步")

    def sample_params(self, rng) -> Dict:
        count = int(rng.integers(1, 4))
        action = "translate" if rng.random() < 0.6 else "rotate"
        if action == "translate":
            step = float(rng.uniform(0.2, 0.4))
            directions = [
                np.array([step, 0.0, 0.0]),
                np.array([-step, 0.0, 0.0]),
                np.array([0.0, step, 0.0]),
                np.array([0.0, -step, 0.0]),
            ]
            delta = directions[int(rng.integers(0, len(directions)))]
            return {"count": count, "action": action, "delta": delta.tolist()}
        theta = float(rng.uniform(math.pi / 10, math.pi / 6))
        return {"count": count, "action": action, "theta": theta}

    @staticmethod
    def _scene_from_frame(frame: Dict) -> Scene:
        objs = []
        for obj in frame.get("objects", []):
            objs.append(
                ObjectState(
                    shape=obj["shape"],
                    r=np.array(obj["r"], dtype=float),
                    p=np.array(obj["p"], dtype=float),
                    rotation=np.array(obj["rotation_euler"], dtype=float),
                    density=float(obj.get("density", 1.0)),
                )
            )
        return Scene(objects=objs)

    @staticmethod
    def _apply_action(objs: Sequence, left_idx: Sequence[int], right_idx: Sequence[int], action: str, delta, mirror: bool) -> List:
        out = clone_objects(objs)
        if action == "translate":
            delta_vec = np.array(delta, dtype=float)
            mirror_delta = np.array([-delta_vec[0], -delta_vec[1], delta_vec[2]])
            for idx in left_idx:
                out[idx] = apply_translation(out[idx], delta_vec)
            for idx in right_idx:
                out[idx] = apply_translation(out[idx], mirror_delta if mirror else delta_vec)
        else:
            theta = float(delta)
            left_rot = np.array([0.0, 0.0, theta])
            right_rot = np.array([0.0, 0.0, -theta if mirror else theta])
            for idx in left_idx:
                out[idx] = apply_rotation(out[idx], left_rot)
            for idx in right_idx:
                out[idx] = apply_rotation(out[idx], right_rot)
        return out

    def generate_triplet(self, params, rng):
        count = int(params.get("count", 1))
        action = params.get("action", "translate")
        delta = params.get("delta")
        theta = float(params.get("theta", math.pi / 12))

        left_objs = []
        right_objs = []
        for _ in range(count):
            left = random_object(rng)
            offset = float(rng.uniform(0.35, 0.8))
            y = float(rng.uniform(-0.35, 0.35))
            z = float(rng.uniform(-0.2, 0.2))
            left.p = np.array([-offset, y, z])
            right = left.copy()
            right.p = np.array([offset, y, z])
            left_objs.append(left)
            right_objs.append(right)

        a_objs = left_objs + right_objs
        left_idx = list(range(count))
        right_idx = list(range(count, 2 * count))

        if action == "translate":
            b_objs = self._apply_action(a_objs, left_idx, right_idx, action, delta, mirror=True)
            c_objs = self._apply_action(b_objs, left_idx, right_idx, action, delta, mirror=True)
        else:
            b_objs = self._apply_action(a_objs, left_idx, right_idx, action, theta, mirror=True)
            c_objs = self._apply_action(b_objs, left_idx, right_idx, action, theta, mirror=True)

        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [count, count, count]
        involved = list(range(len(c_objs)))
        meta = build_rule_meta(
            self,
            "R4",
            len(involved),
            involved,
            ["p", "R"],
            ["mirror(p)", "mirror(R)"],
            "mirror-evolution",
            {
                "count": count,
                "action": action,
                "delta": delta,
                "theta": theta,
                "left_indices": left_idx,
                "right_indices": right_idx,
            },
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        params = meta.get("pattern_params", {})
        action = params.get("action", "translate")
        delta = params.get("delta")
        theta = float(params.get("theta", math.pi / 12))
        left_idx = params.get("left_indices", [])
        right_idx = params.get("right_indices", [])
        frames = meta.get("frames", [])
        if not left_idx or not right_idx or not frames:
            return [], []

        scene_b = self._scene_from_frame(frames[1])

        if action == "translate":
            wrong_mirror = self._apply_action(scene_b.objects, left_idx, right_idx, action, delta, mirror=False)
            wrong_hold = clone_objects(scene_b.objects)
            half_delta = (np.array(delta, dtype=float) * 0.5).tolist()
            wrong_step = self._apply_action(scene_b.objects, left_idx, right_idx, action, half_delta, mirror=True)
        else:
            wrong_mirror = self._apply_action(scene_b.objects, left_idx, right_idx, action, theta, mirror=False)
            wrong_hold = clone_objects(scene_b.objects)
            wrong_step = self._apply_action(scene_b.objects, left_idx, right_idx, action, theta * 0.5, mirror=True)

        distractors = [
            scene_from_objects(wrong_mirror),
            scene_from_objects(wrong_hold),
            scene_from_objects(wrong_step),
        ]
        reasons = [
            "左右镜像动作方向未反向",
            "动作未重复延续",
            "动作幅度不一致",
        ]
        return distractors, reasons


@dataclass
class R4_8DampedBounce(Rule):
    def __init__(self) -> None:
        super().__init__("R4-8", RuleDifficulty.COMPLEX, "阻尼弹跳", "弹跳高度按衰减比例递减")

    def sample_params(self, rng) -> Dict:
        count = int(rng.integers(1, 4))
        return {"count": count}

    @staticmethod
    def _sample_positions(rng, count: int) -> List[np.ndarray]:
        min_sep = 0.5
        positions: List[np.ndarray] = []
        attempts = 0
        while len(positions) < count and attempts < 200:
            candidate = np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6)], dtype=float)
            if all(np.linalg.norm(candidate - p) >= min_sep for p in positions):
                positions.append(candidate)
            attempts += 1
            if attempts % 40 == 0 and len(positions) < count:
                min_sep = max(0.3, min_sep * 0.85)
        if len(positions) < count:
            angles = np.linspace(0, 2 * math.pi, count, endpoint=False)
            positions = [np.array([0.6 * math.cos(a), 0.6 * math.sin(a)], dtype=float) for a in angles]
        return positions

    @staticmethod
    def _sample_ratios(rng, count: int) -> List[float]:
        ratios: List[float] = []
        attempts = 0
        while len(ratios) < count and attempts < 200:
            candidate = float(rng.uniform(0.35, 0.7))
            if all(abs(candidate - r) >= 0.06 for r in ratios):
                ratios.append(candidate)
            attempts += 1
        if len(ratios) < count:
            ratios = np.linspace(0.38, 0.68, count).tolist()
        return ratios

    @staticmethod
    def _with_height(obj: ObjectState, h: float) -> ObjectState:
        updated = obj.copy()
        updated.p = np.array([updated.p[0], h, updated.p[2]], dtype=float)
        return updated

    def generate_triplet(self, params, rng):
        count = int(params.get("count", 1))
        count = min(max(count, 1), 3)
        positions = self._sample_positions(rng, count)
        ratios = self._sample_ratios(rng, count)
        heights = [float(rng.uniform(0.9, 1.6)) for _ in range(count)]

        a_objs = []
        for i in range(count):
            base = random_object(rng, shape="sphere")
            base.p = np.array([positions[i][0], heights[i], positions[i][1]], dtype=float)
            a_objs.append(base)

        h2_list = [heights[i] * ratios[i] for i in range(count)]
        h3_list = [h2_list[i] * ratios[i] for i in range(count)]

        b_objs = [self._with_height(obj, h2_list[i]) for i, obj in enumerate(a_objs)]
        c_objs = [self._with_height(obj, h3_list[i]) for i, obj in enumerate(a_objs)]
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [heights, h2_list, h3_list]
        involved = list(range(count))
        meta = build_rule_meta(
            self,
            "R4",
            count,
            involved,
            ["p"],
            ["height(Oi)"],
            "damped-bounce",
            {"count": count, "ratios": ratios},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []
        v = meta.get("v", {})
        h3_list = v.get("v3", [])
        if len(h3_list) != len(scene_c.objects):
            h3_list = [float(obj.p[1]) for obj in scene_c.objects]

        high_factor = float(rng.uniform(1.35, 1.7))
        low_factor = float(rng.uniform(0.55, 0.75))
        wrong_high = [max(h * high_factor, 0.02) for h in h3_list]
        wrong_low = [max(h * low_factor, 0.02) for h in h3_list]

        def build(heights: Sequence[float]) -> Scene:
            objs = clone_objects(scene_c.objects)
            for obj, h in zip(objs, heights):
                obj.p = np.array([obj.p[0], max(float(h), 0.02), obj.p[2]], dtype=float)
            return scene_from_objects(objs)

        wrong_shape_size = clone_objects(scene_c.objects)
        idx = int(rng.integers(0, len(wrong_shape_size)))
        if rng.random() < 0.5:
            shape_choices = [s for s in SHAPES if s != wrong_shape_size[idx].shape]
            if shape_choices:
                wrong_shape_size[idx] = switch_shape(wrong_shape_size[idx], str(rng.choice(shape_choices)))
            shape_size_reason = "形状变化破坏规则"
        else:
            size_factor = float(rng.uniform(0.4, 0.6) if rng.random() < 0.5 else rng.uniform(1.7, 2.2))
            wrong_shape_size[idx] = apply_scale(wrong_shape_size[idx], size_factor)
            shape_size_reason = "尺寸变化破坏规则"

        distractors = [
            build(wrong_high),
            build(wrong_low),
            scene_from_objects(wrong_shape_size),
        ]
        reasons = [
            "衰减比例偏小，弹跳高度过高",
            "衰减比例偏大，弹跳高度过低",
            shape_size_reason,
        ]
        return distractors, reasons


@dataclass
class R4_9SoftBodySqueeze(Rule):
    def __init__(self) -> None:
        super().__init__("R4-9", RuleDifficulty.COMPLEX, "软体挤压", "外部压力导致球体形变")

    def sample_params(self, rng) -> Dict:
        count = int(rng.integers(1, 6))
        return {"count": count}

    @staticmethod
    def _build_base_sphere() -> ObjectState:
        # 球体底部在 y=0，所以球体中心在 y = r[1]
        sphere_radius = 1.6
        sphere = ObjectState(
            shape="sphere",
            r=np.array([1.6, 1.6, 1.6]),
            p=np.array([0.0, sphere_radius, 0.0]),  # 球体中心在 y = r[1]，使底部在 y=0
            rotation=np.zeros(3),
            density=1.0,
        )
        return sphere

    @staticmethod
    def _random_presser(rng) -> ObjectState:
        shape = str(rng.choice(["sphere", "cylinder"]))
        obj = random_object(rng, shape=shape)
        scale = float(rng.uniform(0.75, 1.8))
        obj = apply_scale(obj, scale)
        return obj

    @staticmethod
    def _position_pressers(pressers: Sequence[ObjectState], sphere: ObjectState, rng) -> List[ObjectState]:
        """
        将挤压物体垂直堆叠在球体上方，像积木一样一个叠一个，确保接触但不穿模。
        使用 AABB 边界框来精确计算接触位置，使用较大的重叠确保真正接触。
        """
        placed = []
        # 根据 aabb 函数：half = r / 2.0, min = p - half, max = p + half
        # 所以：物体顶部 y = p[1] + r[1]/2，底部 y = p[1] - r[1]/2
        
        # 计算球体顶部位置（使用 AABB）
        sphere_min, sphere_max = aabb(sphere)
        sphere_top_y = sphere_max[1]
        
        # 当前堆叠的顶部位置
        current_top_y = sphere_top_y
        
        for i, obj in enumerate(pressers):
            # 根据 aabb：物体底部 = p[1] - r[1]/2，顶部 = p[1] + r[1]/2
            # 物体半高 = r[1]/2
            
            # 使用基于物体尺寸的重叠比例，确保真正接触
            obj_half_height = obj.r[1] / 2.0
            obj_height = obj.r[1]
            # 使用物体高度的较大比例作为重叠量，确保不同大小的物体都能明显接触
            # 重叠 50% 的物体高度，确保有非常明显的接触和挤压效果
            # 这样物体底部会深入前一个物体内部一半，确保明显的挤压
            contact_overlap = -obj_height * 0.50  # 重叠物体高度的 50%，确保非常明显的接触和挤压
            
            # 物体底部应该接触前一个物体的顶部（重叠）
            # 物体底部 y = obj_center_y - r[1]/2
            # 我们希望：obj_center_y - r[1]/2 = current_top_y + contact_overlap
            # 所以：obj_center_y = current_top_y + contact_overlap + r[1]/2
            obj_center_y = current_top_y + contact_overlap + obj_half_height
            
            # X 和 Z 位置：可以稍微随机偏移，但保持在球体中心附近
            # 为了更真实，可以让物体稍微偏离中心，但不要太多
            offset_x = float(rng.uniform(-0.10, 0.10))
            offset_z = float(rng.uniform(-0.10, 0.10))
            
            new_obj = obj.copy()
            new_obj.p = np.array([
                sphere.p[0] + offset_x,
                obj_center_y,
                sphere.p[2] + offset_z
            ], dtype=float)
            
            placed.append(new_obj)
            
            # 更新堆叠顶部位置：使用新物体的实际 AABB 顶部
            new_obj_min, new_obj_max = aabb(new_obj)
            current_top_y = new_obj_max[1]
        
        return placed

    @staticmethod
    def _pressure_ratio(pressers: Sequence[ObjectState], sphere: ObjectState) -> float:
        total = float(sum(size(o) for o in pressers))
        base = float(size(sphere))
        if base <= 1e-6:
            return 0.0
        ratio = total / base * 0.18
        return float(np.clip(ratio, 0.05, 0.6))

    @staticmethod
    def _deform_sphere(sphere: ObjectState, ratio: float) -> ObjectState:
        ratio = float(np.clip(ratio, 0.0, 0.7))
        scale = np.array([1.0 + ratio * 0.55, 1.0 - ratio, 1.0 + ratio * 0.55], dtype=float)
        scale[1] = max(scale[1], 0.45)
        return apply_scale(sphere, scale)

    def _build_scene(self, pressers: Sequence[ObjectState], rng) -> tuple[Scene, float]:
        base_sphere = self._build_base_sphere()
        # 先计算压力比例
        ratio = self._pressure_ratio(pressers, base_sphere)
        # 先形变球体，然后基于形变后的球体计算堆叠位置
        deformed = self._deform_sphere(base_sphere, ratio)
        # 基于形变后的球体计算堆叠位置，确保物体接触形变后的球体顶部
        placed = self._position_pressers(pressers, deformed, rng)
        objs = [deformed] + placed
        return scene_from_objects(objs), ratio

    @staticmethod
    def _adjust_count(rng, pressers: List[ObjectState], delta: int) -> List[ObjectState]:
        updated = [p.copy() for p in pressers]
        if delta > 0:
            for _ in range(delta):
                updated.append(R4_9SoftBodySqueeze._random_presser(rng))
        elif delta < 0:
            for _ in range(min(-delta, len(updated) - 1)):
                idx = int(rng.integers(0, len(updated)))
                updated.pop(idx)
        return updated

    def generate_triplet(self, params, rng):
        count = int(params.get("count", 1))
        pressers_a = [self._random_presser(rng) for _ in range(count)]

        delta1_choices = [-2, -1, 1, 2]
        while True:
            delta1 = int(rng.choice(delta1_choices))
            if 1 <= len(pressers_a) + delta1 <= 5:
                break
        pressers_b = self._adjust_count(rng, pressers_a, delta1)

        while True:
            delta2 = int(rng.choice(delta1_choices))
            if 1 <= len(pressers_b) + delta2 <= 5:
                break
        pressers_c = self._adjust_count(rng, pressers_b, delta2)

        scene_a, ratio_a = self._build_scene(pressers_a, rng)
        scene_b, ratio_b = self._build_scene(pressers_b, rng)
        scene_c, ratio_c = self._build_scene(pressers_c, rng)

        v = [ratio_a, ratio_b, ratio_c]
        involved = list(range(len(scene_c.objects)))
        meta = build_rule_meta(
            self,
            "R4",
            len(involved),
            involved,
            ["r", "p"],
            ["deform_ratio", "size(pressers)"],
            "soft-squeeze",
            {
                "ratio_a": ratio_a,
                "ratio_b": ratio_b,
                "ratio_c": ratio_c,
                "counts": [len(pressers_a), len(pressers_b), len(pressers_c)],
            },
            v,
            [scene_a, scene_b, scene_c],
        )
        return scene_a, scene_b, scene_c, meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []
        params = meta.get("pattern_params", {})
        ratio_b = float(params.get("ratio_b", 0.2))
        ratio_c = float(params.get("ratio_c", 0.3))
        pressers = [obj.copy() for obj in scene_c.objects[1:]]

        def build_with_ratio(ratio: float) -> Scene:
            base_sphere = self._build_base_sphere()
            deformed = self._deform_sphere(base_sphere, ratio)
            # 基于形变后的球体计算堆叠位置
            placed = self._position_pressers(pressers, deformed, rng)
            return scene_from_objects([deformed] + placed)

        wrong_hold = build_with_ratio(ratio_b)
        wrong_small = build_with_ratio(ratio_c * 0.6)
        wrong_big = build_with_ratio(min(ratio_c * 1.45, 0.7))

        distractors = [wrong_hold, wrong_small, wrong_big]
        reasons = [
            "形变未延续，停留在上一帧",
            "形变幅度过小",
            "形变幅度过大",
        ]
        return distractors, reasons


@dataclass
class R4_10DominoChain(Rule):
    def __init__(self) -> None:
        super().__init__("R4-10", RuleDifficulty.COMPLEX, "多米诺骨牌", "倾斜-倒下序列沿链传播")

    def sample_params(self, rng) -> Dict:
        count = int(rng.integers(3, 6))
        theta = float(rng.uniform(math.pi / 6, math.pi / 3))
        direction = 1 if rng.random() < 0.5 else -1
        if direction == 1:
            start = int(rng.integers(0, count - 2))
        else:
            start = int(rng.integers(2, count))
        return {"count": count, "theta": theta, "direction": direction, "start": start}

    @staticmethod
    def _tilt_angle(theta: float, direction: int) -> float:
        return -float(direction) * float(theta)

    @classmethod
    def _apply_angles(cls, objs: Sequence[ObjectState], angles: Dict[int, float]) -> List[ObjectState]:
        out = clone_objects(objs)
        for idx, angle in angles.items():
            if idx < 0 or idx >= len(out):
                continue
            obj = out[idx].copy()
            obj.rotation = np.array([0.0, 0.0, float(angle)])
            out[idx] = obj
        return out

    def generate_triplet(self, params, rng):
        count = int(params["count"])
        theta = float(params["theta"])
        direction = int(params["direction"])
        start = int(params["start"])
        spacing = float(rng.uniform(0.55, 0.75))

        base = random_object(rng, shape="cube")
        base.rotation = np.zeros(3)
        base.r = np.array([0.4, 0.9, 0.35], dtype=float)
        base.density = float(rng.uniform(0.9, 1.1))

        objs = []
        for i in range(count):
            obj = base.copy()
            x = (i - (count - 1) / 2.0) * spacing
            obj.p = np.array([x, 0.0, 0.0], dtype=float)
            objs.append(obj)

        next1 = start + direction
        next2 = start + 2 * direction
        tilt = self._tilt_angle(theta, direction)
        fallen = self._tilt_angle(math.pi / 2, direction)

        a_objs = self._apply_angles(objs, {start: tilt})
        b_objs = self._apply_angles(a_objs, {start: fallen, next1: tilt})
        c_objs = self._apply_angles(b_objs, {next1: fallen, next2: tilt})

        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [
            [float(o.rotation[2]) for o in a_objs],
            [float(o.rotation[2]) for o in b_objs],
            [float(o.rotation[2]) for o in c_objs],
        ]
        involved = list(range(len(c_objs)))
        meta = build_rule_meta(
            self,
            "R4",
            len(involved),
            involved,
            ["R"],
            ["domino(angle)"],
            "domino-chain",
            {"count": count, "theta": theta, "direction": direction, "start": start},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        params = meta.get("pattern_params", {})
        count = int(params.get("count", len(scene_c.objects)))
        theta = float(params.get("theta", math.pi / 6))
        direction = int(params.get("direction", 1))
        start = int(params.get("start", 0))
        if len(scene_c.objects) != count:
            count = len(scene_c.objects)

        next1 = start + direction
        next2 = start + 2 * direction
        tilt = self._tilt_angle(theta, direction)
        fallen = self._tilt_angle(math.pi / 2, direction)

        wrong_hold = self._apply_angles(scene_c.objects, {next1: tilt, next2: 0.0})
        wrong_over = self._apply_angles(scene_c.objects, {next1: fallen, next2: fallen})
        wrong_dir = self._apply_angles(scene_c.objects, {next1: fallen, next2: -tilt})

        distractors = [
            scene_from_objects(wrong_hold),
            scene_from_objects(wrong_over),
            scene_from_objects(wrong_dir),
        ]
        reasons = [
            "倒下未延续，停留在倾斜状态",
            "倒下过快，过度推进",
            "倾斜方向错误",
        ]
        return distractors, reasons


def build_complex_rules() -> List[Rule]:
    return [
        R1_9ScaleRotateCoupled(),
        R2_6RelativeOrientationInvariant(),
        R3_7PositionCycle(),
        R3_8DensityShift(),
        R3_9ScaleShift(),
        R3_10ShapeShift(),
        R3_11SinePositionShift(),
        R4_1SkipAhead(),
        R4_2NodeSplitEvolution(),
        R4_3NodeFusionEvolution(),
        R4_4FormationEvolution(),
        R4_5ContactInfection(),
        R4_6AdvancedOrbitalRotation(),
        R4_7SymmetryTransform(),
        R4_8DampedBounce(),
        R4_9SoftBodySqueeze(),
        R4_10DominoChain(),
    ]
