from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .base import Rule, RuleDifficulty
from .utils import (
    ang,
    apply_density,
    apply_rotation,
    apply_scale,
    apply_translation,
    build_rule_meta,
    centroid,
    clone_objects,
    dist,
    init_objects,
    random_object,
    scene_from_objects,
    SHAPES,
    size,
    symmetry_flag,
)
from ..geometry import rotation_matrix
from ..scene import Scene


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / (np.linalg.norm(v) + 1e-9)


@dataclass
class R1_10ScaleRotateCoupled(Rule):
    def __init__(self) -> None:
        super().__init__("R1-10", RuleDifficulty.COMPLEX, "尺度等比+旋转等差", "scale 与 rotation 复合")

    def sample_params(self, rng) -> Dict:
        k = float(rng.uniform(1.15, 1.5))
        delta_rot = rng.uniform(math.pi / 18, math.pi / 10, size=3)
        return {"k": k, "delta_rot": delta_rot.tolist()}

    def generate_triplet(self, params, rng):
        k = params["k"]
        delta_rot = np.array(params["delta_rot"])
        objs = init_objects(rng, 1, m=2)
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
class R3_4SymmetryRigid(Rule):
    def __init__(self) -> None:
        super().__init__("R3-4", RuleDifficulty.COMPLEX, "对称+刚体", "第二帧对称，第三帧刚体变换")

    def sample_params(self, rng) -> Dict:
        theta = float(rng.uniform(math.pi / 6, math.pi / 4))
        translation = rng.uniform(0.4, 0.8, size=3) * rng.choice([-1, 1], size=3)
        return {"theta": theta, "translation": translation.tolist()}

    def generate_triplet(self, params, rng):
        theta = params["theta"]
        translation = np.array(params["translation"])
        obj_count = int(rng.integers(3, 6))
        objs = [random_object(rng) for _ in range(obj_count)]
        involved = list(range(obj_count))

        def symmetric_positions(count: int) -> List[np.ndarray]:
            pair_count = count // 2
            positions: List[np.ndarray] = []
            for _ in range(pair_count):
                x = float(rng.uniform(0.35, 0.65))
                y = float(rng.uniform(-0.3, 0.3))
                z = float(rng.uniform(-0.25, 0.25))
                positions.append(np.array([-x, y, z]))
                positions.append(np.array([x, y, z]))
            if count % 2 == 1:
                positions.append(
                    np.array(
                        [
                            0.0,
                            float(rng.uniform(0.35, 0.55)),
                            float(rng.uniform(-0.1, 0.1)),
                        ]
                    )
                )
            rng.shuffle(positions)
            return positions

        # Frame B: enforce symmetry about the x=0 plane (left-right mirror).
        b_positions = symmetric_positions(obj_count)
        b_objs = clone_objects(objs)
        for obj, pos in zip(b_objs, b_positions):
            obj.p = pos
        # Frame A: mild asymmetry
        a_objs = clone_objects(b_objs)
        idx = int(rng.integers(0, len(a_objs)))
        delta = rng.uniform(0.06, 0.12, size=3) * rng.choice([-1, 1], size=3)
        delta[0] = float(rng.uniform(0.08, 0.15)) * (1 if rng.random() < 0.5 else -1)
        a_objs[idx].p = a_objs[idx].p + delta
        # Frame C: rigid transform from B
        rot = np.array([[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0], [math.sin(theta), 0, math.cos(theta)]])

        def rigid(objs_in: Sequence) -> List:
            out = []
            for o in objs_in:
                new_o = o.copy()
                new_o.p = rot @ new_o.p + translation
                out.append(new_o)
            return out

        c_objs = rigid(b_objs)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [symmetry_flag(s.objects, axis_name="x") for s in scenes]
        meta = build_rule_meta(
            self,
            "R3",
            3,
            involved,
            ["p"],
            ["sym(S)"],
            "discrete+rigid",
            {"theta": theta, "translation": translation.tolist()},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []
        pattern = meta.get("pattern_params", {})
        base_t = np.array(pattern.get("translation", []), dtype=float)
        if base_t.shape != (3,) or not np.all(np.isfinite(base_t)):
            base_t = rng.uniform(0.8, 1.2, size=3) * rng.choice([-1, 1], size=3)
        factors = [1.6, -1.6, 2.2]
        distractors = []
        for i, factor in enumerate(factors):
            objs = clone_objects(scene_c.objects)
            idx = i % len(objs)
            delta = base_t * factor
            objs[idx] = apply_translation(objs[idx], delta)
            distractors.append(scene_from_objects(objs))
        reasons = [
            "单体平移幅度显著偏大",
            "单体平移方向反向",
            "单体平移幅度过大",
        ]
        return distractors, reasons


@dataclass
class R3_5GroupCentroidDistance(Rule):
    def __init__(self) -> None:
        super().__init__("R3-5", RuleDifficulty.COMPLEX, "组间质心距离等差", "两组对象质心距离等差")

    def sample_params(self, rng) -> Dict:
        delta = float(rng.uniform(0.2, 0.4)) * (1 if rng.random() < 0.5 else -1)
        base = float(rng.uniform(0.8, 1.2))
        return {"delta": delta, "base": base}

    def generate_triplet(self, params, rng):
        delta, base = params["delta"], params["base"]
        obj_count = int(rng.integers(3, 6))
        objs = [random_object(rng) for _ in range(obj_count)]
        involved = list(range(obj_count))
        group_a = [0, 1]
        group_b = list(range(2, obj_count))
        direction = _unit_vector(rng)

        def place(distance: float):
            arranged = clone_objects(objs)
            # move group_b along direction
            for idx in group_b:
                arranged[idx].p = arranged[idx].p + direction * distance
            return arranged

        distances = [base, base + delta, base + 2 * delta]
        scenes_objs = [place(d) for d in distances]
        scenes = [scene_from_objects(x) for x in scenes_objs]

        def group_cent(objects: Sequence, indices: List[int]) -> np.ndarray:
            return centroid([objects[i] for i in indices])

        v = []
        for objs_frame in scenes_objs:
            ca = group_cent(objs_frame, group_a)
            cb = group_cent(objs_frame, group_b)
            v.append(float(np.linalg.norm(ca - cb)))
        meta = build_rule_meta(
            self,
            "R3",
            3,
            involved,
            ["p"],
            ["cent(S_a)", "cent(S_b)"],
            "arithmetic",
            {"delta": delta},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class R2_7RigidTransform(Rule):
    def __init__(self) -> None:
        super().__init__("R2-7", RuleDifficulty.COMPLEX, "刚体一致变换", "整体刚体变换, pairwise dist 不变")

    def sample_params(self, rng) -> Dict:
        theta = float(rng.uniform(math.pi / 18, math.pi / 10))
        translation = rng.uniform(0.2, 0.4, size=3) * rng.choice([-1, 1], size=3)
        return {"theta": theta, "translation": translation.tolist()}

    def generate_triplet(self, params, rng):
        theta = params["theta"]
        translation = np.array(params["translation"])
        objs = init_objects(rng, 3, m=3)
        involved = [0, 1, 2]
        rot = np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

        def rigid(objs_in: Sequence, times: int) -> List:
            out = []
            for o in objs_in:
                new_o = o.copy()
                p = new_o.p
                for _ in range(times):
                    p = rot @ p + translation
                new_o.p = p
                out.append(new_o)
            return out

        a_objs = clone_objects(objs)
        b_objs = rigid(objs, 1)
        c_objs = rigid(objs, 2)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]

        def dist_stats(objects: Sequence) -> List[float]:
            return [dist(objects[0], objects[1]), dist(objects[0], objects[2]), dist(objects[1], objects[2])]

        v = [dist_stats(s.objects) for s in scenes]
        meta = build_rule_meta(
            self,
            "R2",
            3,
            involved,
            ["p"],
            ["dist-set"],
            "rigid",
            {"theta": theta, "translation": translation.tolist()},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class R2_8RelativeOrientationInvariant(Rule):
    def __init__(self) -> None:
        super().__init__("R2-8", RuleDifficulty.COMPLEX, "相对姿态保持", "共同旋转保持夹角不变")

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
class R3_6AreaDistanceCoupled(Rule):
    def __init__(self) -> None:
        super().__init__("R3-6", RuleDifficulty.COMPLEX, "面积-边长守恒", "area 与 dist 乘积守恒")

    def sample_params(self, rng) -> Dict:
        factor = float(rng.uniform(1.2, 1.5))
        return {"factor": factor}

    def generate_triplet(self, params, rng):
        factor = params["factor"]
        objs = init_objects(rng, 3, m=3)
        involved = [0, 1, 2]
        # Place base triangle roughly on XY-plane.
        objs[0].p = np.array([-0.6, 0.0, 0])
        objs[1].p = np.array([0.6, 0.0, 0])
        objs[2].p = np.array([0.0, 0.6, 0])

        def stats(objects: Sequence) -> tuple[float, float, float]:
            area = float(0.5 * np.linalg.norm(np.cross(objects[1].p - objects[0].p, objects[2].p - objects[0].p)))
            d12 = dist(objects[0], objects[1])
            return area, d12, area * d12

        area1, d1, const = stats(objs)
        area2 = area1 * factor
        d2 = const / area2
        area3 = area1 / factor
        d3 = const / area3

        def adjust(area_target: float, d_target: float) -> List:
            arranged = clone_objects(objs)
            arranged[1].p = arranged[0].p + np.array([d_target, 0, 0])
            scale_y = area_target * 2 / d_target
            arranged[2].p = np.array([0.0, scale_y, 0.0])
            return arranged

        a_objs = clone_objects(objs)
        b_objs = adjust(area2, d2)
        c_objs = adjust(area3, d3)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [list(stats(s.objects))[:2] for s in scenes]
        meta = build_rule_meta(
            self,
            "R3",
            3,
            involved,
            ["p"],
            ["area(0,1,2)", "dist(0,1)"],
            "coupled",
            {"constant": const},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if len(scene_c.objects) < 3:
            return [], []
        base_objs = scene_c.objects
        p0 = base_objs[0].p.copy()
        d_base = dist(base_objs[0], base_objs[1])
        if d_base <= 1e-6:
            return [], []
        area_base = float(0.5 * np.linalg.norm(np.cross(base_objs[1].p - base_objs[0].p, base_objs[2].p - base_objs[0].p)))

        def adjust(area_target: float, d_target: float) -> Scene:
            arranged = clone_objects(base_objs)
            safe_d = max(float(d_target), 0.05)
            arranged[0].p = p0
            arranged[1].p = p0 + np.array([safe_d, 0.0, 0.0])
            scale_y = max(area_target * 2 / safe_d, 0.04)
            arranged[2].p = np.array([0.0, scale_y, 0.0])
            return scene_from_objects(arranged)

        combos = [
            (area_base * 1.8, d_base),
            (area_base, d_base * 1.6),
            (area_base * 0.6, d_base * 0.6),
        ]
        distractors = [adjust(a, d) for a, d in combos]
        reasons = [
            "面积显著偏大，乘积偏离常数",
            "边长显著偏大，乘积偏离常数",
            "面积与边长同时偏小，乘积偏离常数",
        ]
        return distractors, reasons


@dataclass
class R3_7PositionCycle(Rule):
    def __init__(self) -> None:
        super().__init__("R3-7", RuleDifficulty.COMPLEX, "多对象位置轮换", "不同形状沿结构相邻位置轮换")

    def sample_params(self, rng) -> Dict:
        count = int(rng.integers(2, 6))
        direction = "cw" if rng.random() < 0.5 else "ccw"
        return {"count": count, "direction": direction}

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

    def generate_triplet(self, params, rng):
        count = int(params["count"])
        direction = params["direction"]
        positions, layout_name = self._layout_positions(count)
        rot_angles = rng.uniform(-math.pi / 6, math.pi / 6, size=3)
        scale = float(rng.uniform(0.9, 1.1))
        rot = rotation_matrix(rot_angles)
        positions = [scale * (rot @ p) for p in positions]
        shapes = rng.choice(SHAPES, size=count, replace=False).tolist()
        objs = [random_object(rng, shape=shape) for shape in shapes]
        involved = list(range(count))
        step = 1 if direction == "cw" else -1

        def build_frame(offset: int) -> List:
            arranged = clone_objects(objs)
            for idx, obj in enumerate(arranged):
                obj.p = positions[(idx + offset) % count]
            return arranged

        def shapes_in_positions(offset: int) -> List[str]:
            return [shapes[(j - offset) % count] for j in range(count)]

        a_objs = build_frame(0)
        b_objs = build_frame(step)
        c_objs = build_frame(2 * step)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [shapes_in_positions(0), shapes_in_positions(step), shapes_in_positions(2 * step)]
        meta = build_rule_meta(
            self,
            "R3",
            3,
            involved,
            ["p", "s"],
            ["position_cycle"],
            "cyclic",
            {
                "direction": direction,
                "count": count,
                "layout": layout_name,
                "rotation_euler": rot_angles.tolist(),
                "scale": scale,
            },
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


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
            base = self._regular_polygon(5, 0.7)
            star_order = [0, 2, 4, 1, 3]
            return [base[i] for i in star_order], "pentagram"
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
            base = self._regular_polygon(5, 0.7)
            star_order = [0, 2, 4, 1, 3]
            return [base[i] for i in star_order], "pentagram"
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


def build_complex_rules() -> List[Rule]:
    return [
        R1_10ScaleRotateCoupled(),
        R3_4SymmetryRigid(),
        R3_5GroupCentroidDistance(),
        R2_7RigidTransform(),
        R2_8RelativeOrientationInvariant(),
        R3_6AreaDistanceCoupled(),
        R3_7PositionCycle(),
        R3_8DensityShift(),
        R3_9ScaleShift(),
        R3_10ShapeShift(),
    ]
