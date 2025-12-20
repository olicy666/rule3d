from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .base import Rule, RuleDifficulty
from .utils import (
    SHAPES,
    ang,
    apply_rotation,
    apply_scale,
    apply_translation,
    aspect_ratio,
    build_rule_meta,
    centroid,
    clone_objects,
    contain,
    direction,
    dist,
    init_objects,
    order_indices_x,
    scene_from_objects,
    size,
    symmetry_flag,
    touch,
)
from ..scene import Scene


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / (np.linalg.norm(v) + 1e-9)


@dataclass
class M01DistanceArithmetic(Rule):
    def __init__(self) -> None:
        super().__init__("M01", RuleDifficulty.MEDIUM, "成对距离等差", "dist(i,j) 等差")

    def sample_params(self, rng) -> Dict:
        base = float(rng.uniform(0.6, 1.1))
        step = float(rng.uniform(0.15, 0.35)) * (1 if rng.random() < 0.5 else -1)
        return {"base": base, "step": step}

    def generate_triplet(self, params, rng):
        base, step = params["base"], params["step"]
        direction_vec = _unit_vector(rng)
        objs = init_objects(rng, 2, m=2)
        involved = [0, 1]
        def place(distance: float):
            obj_a, obj_b = clone_objects(objs)
            obj_a.p = -direction_vec * distance / 2
            obj_b.p = direction_vec * distance / 2
            return [obj_a, obj_b]

        distances = [base, base + step, base + 2 * step]
        scenes_objs = [place(d) for d in distances]
        scenes = [scene_from_objects(x) for x in scenes_objs]
        v = distances
        meta = build_rule_meta(
            self, "R2", 2, involved, ["p"], ["dist(0,1)"], "arithmetic", {"delta": step}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class M02DistanceGeometric(Rule):
    def __init__(self) -> None:
        super().__init__("M02", RuleDifficulty.MEDIUM, "成对距离等比", "dist(i,j) 等比")

    def sample_params(self, rng) -> Dict:
        k = float(rng.uniform(1.15, 1.6) if rng.random() < 0.5 else rng.uniform(0.65, 0.9))
        base = float(rng.uniform(0.5, 1.0))
        return {"k": k, "base": base}

    def generate_triplet(self, params, rng):
        k, base = params["k"], params["base"]
        direction_vec = _unit_vector(rng)
        objs = init_objects(rng, 2, m=2)
        involved = [0, 1]

        def place(distance: float):
            obj_a, obj_b = clone_objects(objs)
            obj_a.p = -direction_vec * distance / 2
            obj_b.p = direction_vec * distance / 2
            return [obj_a, obj_b]

        distances = [base, base * k, base * k * k]
        scenes_objs = [place(d) for d in distances]
        scenes = [scene_from_objects(x) for x in scenes_objs]
        v = distances
        meta = build_rule_meta(
            self, "R2", 2, involved, ["p"], ["dist(0,1)"], "geometric", {"k": k}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class M03DirectionLocked(Rule):
    def __init__(self) -> None:
        super().__init__("M03", RuleDifficulty.MEDIUM, "方向保持距离变化", "dir 恒定, dist 线性")

    def sample_params(self, rng) -> Dict:
        step = float(rng.uniform(0.1, 0.3)) * (1 if rng.random() < 0.5 else -1)
        base = float(rng.uniform(0.6, 1.0))
        return {"step": step, "base": base}

    def generate_triplet(self, params, rng):
        base, step = params["base"], params["step"]
        direction_vec = _unit_vector(rng)
        objs = init_objects(rng, 2, m=2)
        involved = [0, 1]

        def place(distance: float):
            o0, o1 = clone_objects(objs)
            o0.p = np.zeros(3)
            o1.p = direction_vec * distance
            return [o0, o1]

        distances = [base, base + step, base + 2 * step]
        scenes_objs = [place(d) for d in distances]
        scenes = [scene_from_objects(x) for x in scenes_objs]
        v = distances
        meta = build_rule_meta(
            self,
            "R2",
            2,
            involved,
            ["p"],
            ["dir(0,1)", "dist(0,1)"],
            "direction-locked",
            {"direction": direction_vec.tolist(), "delta": step},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class M04DirectionRotate(Rule):
    def __init__(self) -> None:
        super().__init__("M04", RuleDifficulty.MEDIUM, "方向旋转等差角", "dir 旋转, dist 保持")

    def sample_params(self, rng) -> Dict:
        theta = float(rng.uniform(math.pi / 12, math.pi / 6))
        base = float(rng.uniform(0.7, 1.0))
        return {"theta": theta, "base": base}

    def generate_triplet(self, params, rng):
        theta, base = params["theta"], params["base"]
        objs = init_objects(rng, 2, m=2)
        involved = [0, 1]
        # Use rotation around z axis for direction updates.
        base_dir = _unit_vector(rng)
        base_dir[2] = 0  # keep planar
        base_dir = base_dir / (np.linalg.norm(base_dir) + 1e-9)

        def rotate_dir(angle: float) -> np.ndarray:
            rot = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
            return rot @ base_dir

        dirs = [rotate_dir(0), rotate_dir(theta), rotate_dir(2 * theta)]

        def place(dir_vec: np.ndarray):
            o0, o1 = clone_objects(objs)
            o0.p = np.zeros(3)
            o1.p = dir_vec * base
            return [o0, o1]

        scenes_objs = [place(d) for d in dirs]
        scenes = [scene_from_objects(x) for x in scenes_objs]
        v = [d.tolist() for d in dirs]
        meta = build_rule_meta(
            self,
            "R2",
            2,
            involved,
            ["p"],
            ["dir(0,1)"],
            "angular-arithmetic",
            {"theta": theta},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class M05TouchSequence(Rule):
    def __init__(self) -> None:
        super().__init__("M05", RuleDifficulty.MEDIUM, "接触状态序列", "touch 序列 0→1→1")

    def sample_params(self, rng) -> Dict:
        return {}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 2, m=2)
        involved = [0, 1]
        a_obj0, a_obj1 = clone_objects(objs)
        required = 0.2 + 0.6 * rng.random()
        direction_vec = _unit_vector(rng)
        # Step distances to realize touch states.
        radius_sum = 0.5 * (np.linalg.norm(a_obj0.r) + np.linalg.norm(a_obj1.r))
        dist_far = radius_sum + required
        dist_touch = radius_sum * 1.02

        def place(distance: float):
            o0, o1 = clone_objects(objs)
            o0.p = -direction_vec * distance / 2
            o1.p = direction_vec * distance / 2
            return [o0, o1]

        scenes_objs = [place(dist_far), place(dist_touch), place(dist_touch * 0.95)]
        scenes = [scene_from_objects(x) for x in scenes_objs]
        v = [touch(*objs_pair) for objs_pair in scenes_objs]
        meta = build_rule_meta(
            self,
            "R2",
            2,
            involved,
            ["p", "r"],
            ["touch(0,1)"],
            "discrete",
            {"sequence": [0, 1, 1]},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class M06ContainSequence(Rule):
    def __init__(self) -> None:
        super().__init__("M06", RuleDifficulty.MEDIUM, "包含关系序列", "contain 序列 0→1→0")

    def sample_params(self, rng) -> Dict:
        return {}

    def generate_triplet(self, params, rng):
        outer = init_objects(rng, 1, m=2)[0]
        inner = init_objects(rng, 1, m=2)[1]
        # Make sure outer is bigger.
        outer.r = outer.r * 1.5
        involved = [0, 1]
        offset = rng.uniform(0.1, 0.25, size=3)

        a_objs = [outer.copy(), apply_translation(inner, outer.p + offset - inner.p)]
        b_inner = inner.copy()
        b_inner.p = outer.p
        b_objs = [outer.copy(), b_inner]
        escape = outer.r * 0.8 + offset
        c_objs = [outer.copy(), apply_translation(inner, outer.p + escape - inner.p)]
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [contain(*objs_pair) for objs_pair in [a_objs, b_objs, c_objs]]
        meta = build_rule_meta(
            self,
            "R2",
            2,
            involved,
            ["p", "r"],
            ["contain(0,1)"],
            "discrete",
            {"sequence": [0, 1, 0]},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class M07AngleArithmetic(Rule):
    def __init__(self) -> None:
        super().__init__("M07", RuleDifficulty.MEDIUM, "夹角等差", "轴夹角按等差变化")

    def sample_params(self, rng) -> Dict:
        base_angle = float(rng.uniform(math.pi / 12, math.pi / 6))
        delta = float(rng.uniform(math.pi / 18, math.pi / 10))
        return {"base_angle": base_angle, "delta": delta}

    def generate_triplet(self, params, rng):
        base_angle, delta = params["base_angle"], params["delta"]
        objs = init_objects(rng, 2, m=2)
        involved = [0, 1]
        axis_rot = np.array([0.0, base_angle, 0.0])
        a_objs = clone_objects(objs)
        a_objs[1] = apply_rotation(a_objs[1], axis_rot)
        b_objs = clone_objects(a_objs)
        b_objs[1] = apply_rotation(b_objs[1], np.array([0.0, delta, 0.0]))
        c_objs = clone_objects(b_objs)
        c_objs[1] = apply_rotation(c_objs[1], np.array([0.0, delta, 0.0]))
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [ang(*pair) for pair in [[a_objs[0], a_objs[1]], [b_objs[0], b_objs[1]], [c_objs[0], c_objs[1]]]]
        meta = build_rule_meta(
            self, "R2", 2, involved, ["R"], ["ang(0,1)"], "arithmetic", {"delta": delta}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class M08AreaArithmetic(Rule):
    def __init__(self) -> None:
        super().__init__("M08", RuleDifficulty.MEDIUM, "三对象面积等差", "三点面积等差变化")

    def sample_params(self, rng) -> Dict:
        delta_ratio = float(rng.uniform(0.15, 0.35)) * (1 if rng.random() < 0.5 else -1)
        return {"delta_ratio": delta_ratio}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 3, m=3)
        involved = [0, 1, 2]
        # Fix base edge on x 轴，控制高度即可精准调节面积
        base_len = 1.0
        objs[0].p = np.array([-base_len / 2, 0, 0])
        objs[1].p = np.array([base_len / 2, 0, 0])
        height1 = float(rng.uniform(0.4, 0.8))
        objs[2].p = np.array([0.0, height1, 0.0])
        area1 = 0.5 * base_len * height1
        delta = area1 * params["delta_ratio"]
        area2 = area1 + delta
        area3 = area1 + 2 * delta
        height2 = 2 * area2 / base_len
        height3 = 2 * area3 / base_len

        b_objs = clone_objects(objs)
        b_objs[2].p = np.array([0.0, height2, 0.0])
        c_objs = clone_objects(objs)
        c_objs[2].p = np.array([0.0, height3, 0.0])
        scenes = [scene_from_objects(x) for x in [objs, b_objs, c_objs]]
        v = [self._area(s.objects) for s in scenes]
        meta = build_rule_meta(
            self,
            "R3",
            3,
            involved,
            ["p"],
            ["area(0,1,2)"],
            "arithmetic",
            {"delta": delta},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta
    
    @staticmethod
    def _area(objs: Sequence) -> float:
        return float(0.5 * np.linalg.norm(np.cross(objs[1].p - objs[0].p, objs[2].p - objs[0].p)))


@dataclass
class M09DistanceDifferenceConserved(Rule):
    def __init__(self) -> None:
        super().__init__("M09", RuleDifficulty.MEDIUM, "距离差分守恒", "两对距离差保持常数")

    def sample_params(self, rng) -> Dict:
        constant = float(rng.uniform(0.2, 0.5))
        step = float(rng.uniform(0.1, 0.25))
        return {"constant": constant, "step": step}

    def generate_triplet(self, params, rng):
        c_val, step = params["constant"], params["step"]
        objs = init_objects(rng, 3, m=3)
        involved = [0, 1, 2]
        direction_vec = _unit_vector(rng)
        base_dist = float(rng.uniform(0.7, 1.0))
        d1 = base_dist
        d2 = max(0.3, base_dist - c_val)

        def place(d01: float, d12: float):
            o0, o1, o2 = clone_objects(objs)
            o1.p = np.zeros(3)
            o0.p = -direction_vec * d01
            o2.p = direction_vec * d12
            return [o0, o1, o2]

        scenes_objs = [
            place(d1, d2),
            place(d1 + step, d2 + step),
            place(d1 + 2 * step, d2 + 2 * step),
        ]
        scenes = [scene_from_objects(x) for x in scenes_objs]
        v = [dist(pair[0], pair[1]) - dist(pair[1], pair[2]) for pair in scenes_objs]
        meta = build_rule_meta(
            self,
            "R2",
            3,
            involved,
            ["p"],
            ["dist(0,1)-dist(1,2)"],
            "conservation",
            {"constant": c_val},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class M10OrderingCycle(Rule):
    def __init__(self) -> None:
        super().__init__("M10", RuleDifficulty.MEDIUM, "排序模式循环", "沿 x 轴顺序循环")

    def sample_params(self, rng) -> Dict:
        return {}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 3, m=3)
        involved = [0, 1, 2]
        base_positions = [-0.8, 0.0, 0.8]
        perms = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]

        def assign(ordering: List[int]):
            arranged = clone_objects(objs)
            for rank, obj_idx in enumerate(ordering):
                arranged[obj_idx].p[0] = base_positions[rank]
            return arranged

        scenes_objs = [assign(p) for p in perms]
        scenes = [scene_from_objects(x) for x in scenes_objs]
        v = [order_indices_x(s.objects) for s in scenes]
        meta = build_rule_meta(
            self,
            "R3",
            3,
            involved,
            ["p"],
            ["ord_x(S)"],
            "discrete",
            {"permutation": perms},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class M11CentroidArithmetic(Rule):
    def __init__(self) -> None:
        super().__init__("M11", RuleDifficulty.MEDIUM, "集合质心等差平移", "集合质心等差")

    def sample_params(self, rng) -> Dict:
        delta = rng.uniform(0.2, 0.4, size=3) * rng.choice([-1, 1], size=3)
        return {"delta": delta.tolist()}

    def generate_triplet(self, params, rng):
        delta = np.array(params["delta"])
        objs = init_objects(rng, 3, m=3)
        involved = [0, 1, 2]
        a_objs = clone_objects(objs)
        b_objs = [apply_translation(o, delta) for o in objs]
        c_objs = [apply_translation(o, delta * 2) for o in objs]
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [centroid(s.objects) for s in scenes]
        meta = build_rule_meta(
            self, "R3", 3, involved, ["p"], ["cent(S)"], "arithmetic", {"delta": delta.tolist()}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class M12DistanceVectorGeometric(Rule):
    def __init__(self) -> None:
        super().__init__("M12", RuleDifficulty.MEDIUM, "距离集合等比缩放", "三对距离成等比")

    def sample_params(self, rng) -> Dict:
        k = float(rng.uniform(1.15, 1.5))
        return {"k": k}

    def generate_triplet(self, params, rng):
        k = params["k"]
        objs = init_objects(rng, 3, m=3)
        involved = [0, 1, 2]
        base_cent = centroid(objs)

        def scale_positions(objs_in: Sequence, factor: float):
            scaled = []
            for o in objs_in:
                new_o = o.copy()
                new_o.p = base_cent + (o.p - base_cent) * factor
                scaled.append(new_o)
            return scaled

        a_objs = clone_objects(objs)
        b_objs = scale_positions(objs, k)
        c_objs = scale_positions(b_objs, k)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]

        def dist_vec(objects: Sequence) -> List[float]:
            return [dist(objects[0], objects[1]), dist(objects[0], objects[2]), dist(objects[1], objects[2])]

        v = [dist_vec(s.objects) for s in scenes]
        meta = build_rule_meta(
            self,
            "R3",
            3,
            involved,
            ["p"],
            ["dist-set"],
            "geometric",
            {"k": k},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class M13SymmetrySwitch(Rule):
    def __init__(self) -> None:
        super().__init__("M13", RuleDifficulty.MEDIUM, "对称性开关", "sym 序列 0→1→1")

    def sample_params(self, rng) -> Dict:
        return {}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 2, m=2)
        involved = [0, 1]
        base_offset = rng.uniform(0.5, 0.8)
        a_objs = clone_objects(objs)
        a_objs[0].p = np.array([-base_offset, 0.2, 0])
        a_objs[1].p = np.array([base_offset, -0.1, 0])

        b_objs = clone_objects(objs)
        b_objs[0].p = np.array([-base_offset, 0, 0])
        b_objs[1].p = np.array([base_offset, 0, 0])

        c_objs = clone_objects(b_objs)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [symmetry_flag(s.objects, axis_name="y") for s in scenes]
        meta = build_rule_meta(
            self,
            "R3",
            2,
            involved,
            ["p"],
            ["sym(S)"],
            "discrete",
            {"sequence": [0, 1, 1]},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class M14DualSizeConservation(Rule):
    def __init__(self) -> None:
        super().__init__("M14", RuleDifficulty.MEDIUM, "双对象属性联动", "size 和保持守恒")

    def sample_params(self, rng) -> Dict:
        delta_ratio = float(rng.uniform(0.1, 0.25))
        sign = 1 if rng.random() < 0.5 else -1
        return {"delta_ratio": delta_ratio * sign}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 2, m=2)
        involved = [0, 1]
        s0, s1 = size(objs[0]), size(objs[1])
        total = s0 + s1
        delta = s0 * params["delta_ratio"]

        def resize_pair(base0, base1, delta_val):
            target0 = base0 + delta_val
            target1 = total - target0
            f0 = (target0 / base0) ** (1 / 3)
            f1 = (target1 / base1) ** (1 / 3)
            o0 = apply_scale(objs[0], f0)
            o1 = apply_scale(objs[1], f1)
            return [o0, o1]

        a_objs = clone_objects(objs)
        b_objs = resize_pair(s0, s1, delta)
        c_objs = resize_pair(s0 + delta, s1 - delta, delta)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [[size(o0), size(o1)] for o0, o1 in [a_objs, b_objs, c_objs]]
        meta = build_rule_meta(
            self,
            "R1",
            2,
            involved,
            ["r"],
            ["size(0)", "size(1)"],
            "coupled",
            {"total": total},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


def build_medium_rules() -> List[Rule]:
    return [
        M01DistanceArithmetic(),
        M02DistanceGeometric(),
        M03DirectionLocked(),
        M04DirectionRotate(),
        M05TouchSequence(),
        M06ContainSequence(),
        M07AngleArithmetic(),
        M08AreaArithmetic(),
        M09DistanceDifferenceConserved(),
        M10OrderingCycle(),
        M11CentroidArithmetic(),
        M12DistanceVectorGeometric(),
        M13SymmetrySwitch(),
        M14DualSizeConservation(),
    ]
