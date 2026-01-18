from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .base import Rule, RuleDifficulty
from .utils import (
    ang,
    apply_rotation,
    apply_scale,
    apply_translation,
    build_rule_meta,
    centroid,
    clone_objects,
    dist,
    init_objects,
    scene_from_objects,
    size,
    symmetry_flag,
)
from ..scene import Scene


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / (np.linalg.norm(v) + 1e-9)


@dataclass
class R1_11ScaleRotateCoupled(Rule):
    def __init__(self) -> None:
        super().__init__("R1-11", RuleDifficulty.COMPLEX, "尺度等比+旋转等差", "scale 与 rotation 复合")

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
        shape_options = ["cube", "sphere", "cylinder", "cone"]
        shape_a = objs[0].shape
        shape_b = rng.choice([s for s in shape_options if s != shape_a])
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
        theta = float(rng.uniform(math.pi / 12, math.pi / 8))
        translation = rng.uniform(0.2, 0.4, size=3) * rng.choice([-1, 1], size=3)
        return {"theta": theta, "translation": translation.tolist()}

    def generate_triplet(self, params, rng):
        theta = params["theta"]
        translation = np.array(params["translation"])
        objs = init_objects(rng, 3, m=3)
        involved = [0, 1, 2]
        # Frame A: mild asymmetry
        a_objs = clone_objects(objs)
        a_objs[0].p = np.array([-0.6, 0.2, 0])
        a_objs[1].p = np.array([0.6, -0.1, 0])
        a_objs[2].p = np.array([0.0, 0.45, 0])
        # Frame B: enforce symmetry about y-axis
        b_objs = clone_objects(a_objs)
        b_objs[0].p = np.array([-0.5, 0.0, 0])
        b_objs[1].p = np.array([0.5, 0.0, 0])
        b_objs[2].p = np.array([0.0, 0.45, 0])
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
        v = [symmetry_flag(s.objects, axis_name="y") for s in scenes]
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
        objs = init_objects(rng, 3, m=3)
        involved = [0, 1, 2]
        group_a = [0, 1]
        group_b = [2]
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


def build_complex_rules() -> List[Rule]:
    return [
        R1_11ScaleRotateCoupled(),
        R3_4SymmetryRigid(),
        R3_5GroupCentroidDistance(),
        R2_7RigidTransform(),
        R2_8RelativeOrientationInvariant(),
        R3_6AreaDistanceCoupled(),
    ]
