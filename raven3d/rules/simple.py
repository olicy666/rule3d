from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .base import Rule, RuleDifficulty
from .utils import (
    SHAPES,
    apply_density,
    apply_rotation,
    apply_scale,
    apply_translation,
    aspect_ratio,
    approx_radius,
    axis,
    build_rule_meta,
    centroid,
    clone_objects,
    dist,
    init_objects,
    place_extras_apart,
    random_object,
    scene_from_objects,
    size,
    switch_shape,
)
from ..scene import Scene


@dataclass
class S01ScaleGeometric(Rule):
    def __init__(self) -> None:
        super().__init__("S01", RuleDifficulty.SIMPLE, "等比统一缩放", "size 按等比增长或缩小")

    def sample_params(self, rng) -> Dict:
        k = float(rng.uniform(1.15, 1.6) if rng.random() < 0.5 else rng.uniform(0.6, 0.85))
        return {"k": k}

    def generate_triplet(self, params, rng):
        k = params["k"]
        objs = init_objects(rng, 1)
        involved = list(range(len(objs)))
        a_objs = clone_objects(objs)
        b_objs = clone_objects(objs)
        b_objs[0] = apply_scale(b_objs[0], k)
        c_objs = clone_objects(b_objs)
        c_objs[0] = apply_scale(c_objs[0], k)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [size(scenes[0].objects[0]), size(scenes[1].objects[0]), size(scenes[2].objects[0])]
        meta = build_rule_meta(
            self, "R1", 1, involved, ["r"], ["size(O0)"], "geometric", {"k": k}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class R1_1ScaleArithmetic(Rule):
    def __init__(self) -> None:
        super().__init__("R1-1", RuleDifficulty.SIMPLE, "等差统一缩放", "size 按等差递进")

    def sample_params(self, rng) -> Dict:
        sign = -1.0 if rng.random() < 0.5 else 1.0
        if sign < 0:
            delta_ratio = float(rng.uniform(0.3, 0.45))
        else:
            delta_ratio = float(rng.uniform(0.3, 0.55))
        return {"delta_ratio": delta_ratio * sign}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 1)
        involved = list(range(len(objs)))
        base_obj = objs[0]
        base_size = size(base_obj)
        delta_ratio = float(params["delta_ratio"])
        if delta_ratio < -0.45:
            delta_ratio = -0.45
        delta = base_size * delta_ratio
        target_b = base_size + delta
        factor_b = (target_b / base_size) ** (1 / 3)
        target_c = target_b + delta
        factor_c = (target_c / target_b) ** (1 / 3)

        a_objs = clone_objects(objs)
        b_objs = clone_objects(objs)
        b_objs[0] = apply_scale(b_objs[0], factor_b)
        c_objs = clone_objects(b_objs)
        c_objs[0] = apply_scale(c_objs[0], factor_c)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [base_size, size(b_objs[0]), size(c_objs[0])]
        meta = build_rule_meta(
            self, "R1", 1, involved, ["r"], ["size(O0)"], "arithmetic", {"delta": delta}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []
        v = meta.get("v", {})
        v2 = v.get("v2", [None])[0]
        v3 = v.get("v3", [scene_c.objects[0].density])[0]
        delta = meta.get("pattern_params", {}).get("delta")
        if delta is None and v2 is not None:
            delta = float(v3) - float(v2)
        if delta is None:
            delta = 0.0
        base_size = float(v2) if v2 is not None else float(v3) - float(delta)

        wrong_deltas = [
            delta * float(rng.uniform(2.0, 3.0)),
            delta * float(rng.uniform(0.15, 0.35)),
            -delta * float(rng.uniform(1.5, 2.5)),
        ]

        def with_size(target_size: float) -> Scene:
            objs = clone_objects(scene_c.objects)
            cur_size = size(objs[0])
            if cur_size > 1e-6:
                safe_target = float(target_size)
                if not math.isfinite(safe_target) or safe_target <= 1e-6:
                    # Keep the target size positive to avoid complex scaling.
                    safe_target = max(cur_size * 0.2, 1e-6)
                scale = (safe_target / cur_size) ** (1 / 3)
                objs[0] = apply_scale(objs[0], scale)
            return scene_from_objects(objs)

        distractors = [with_size(base_size + d) for d in wrong_deltas]
        reasons = [
            "等差步长显著偏大",
            "等差步长显著偏小",
            "等差方向错误",
        ]
        return distractors, reasons


@dataclass
class S03SingleAxisGeometric(Rule):
    def __init__(self) -> None:
        super().__init__("S03", RuleDifficulty.SIMPLE, "单轴比例等比", "仅 x 轴按等比缩放")

    def sample_params(self, rng) -> Dict:
        k = float(rng.uniform(1.1, 1.6) if rng.random() < 0.5 else rng.uniform(0.65, 0.9))
        return {"k": k}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 1)
        involved = [0]
        k = params["k"]
        a_objs = clone_objects(objs)
        b_objs = clone_objects(objs)
        b_objs[0] = apply_scale(b_objs[0], k, axis_mask=[True, False, False])
        c_objs = clone_objects(b_objs)
        c_objs[0] = apply_scale(c_objs[0], k, axis_mask=[True, False, False])
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [aspect_ratio(o) for o in [a_objs[0], b_objs[0], c_objs[0]]]
        meta = build_rule_meta(
            self, "R1", 1, involved, ["r"], ["ar(O0)"], "geometric", {"k": k, "axis": "x"}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class R1_2AxisPermutation(Rule):
    def __init__(self) -> None:
        super().__init__("R1-2", RuleDifficulty.SIMPLE, "尺度轴置换循环", "r_x/r_y/r_z 按固定置换循环")

    def sample_params(self, rng) -> Dict:
        order = rng.choice([[1, 2, 0], [2, 0, 1]])
        return {"order": [int(x) for x in order]}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 1)
        involved = [0]
        base = objs[0].copy()
        base.rotation = np.zeros(3)
        scale = float(np.mean(base.r))
        ratios = np.array([0.6, 1.0, 1.45], dtype=float)
        rng.shuffle(ratios)
        base.r = ratios * scale

        order = params["order"]

        def apply_perm(obj, perm):
            new_obj = obj.copy()
            new_obj.r = new_obj.r[np.array(perm, dtype=int)]
            return new_obj

        a_objs = [base.copy()]
        b_objs = [apply_perm(base, order)]
        c_objs = [apply_perm(b_objs[0], order)]
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [o.r.tolist() for o in [a_objs[0], b_objs[0], c_objs[0]]]
        meta = build_rule_meta(
            self,
            "R1",
            1,
            involved,
            ["r"],
            ["perm(r)"],
            "permute",
            {"order": order},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []

        def with_perm(order: List[int]) -> Scene:
            objs = clone_objects(scene_c.objects)
            r = objs[0].r.copy()
            objs[0].r = r[np.array(order, dtype=int)]
            return scene_from_objects(objs)

        distractors = [
            with_perm([1, 0, 2]),
            with_perm([0, 2, 1]),
            with_perm([2, 1, 0]),
        ]
        reasons = [
            "仅交换两个轴",
            "置换顺序错误",
            "轴顺序完全反向",
        ]
        return distractors, reasons


@dataclass
class R2_2AnisotropicGeometric(Rule):
    def __init__(self) -> None:
        super().__init__("R2-2", RuleDifficulty.SIMPLE, "各向异性等比拉伸", "体积不变的等比拉伸")

    def sample_params(self, rng) -> Dict:
        factor = float(rng.uniform(1.25, 1.7))
        axis_idx = int(rng.integers(0, 3))
        return {"factor": factor, "axis": axis_idx}

    def generate_triplet(self, params, rng):
        count = int(rng.integers(3, 7))
        objs = [random_object(rng) for _ in range(count)]
        involved = [0]
        factor = float(params["factor"])
        axis_idx = int(params["axis"])
        squeeze = 1.0 / math.sqrt(factor)
        scale = np.ones(3)
        scale[axis_idx] = factor
        for i in range(3):
            if i != axis_idx:
                scale[i] = squeeze
        base = objs[0]
        alt = apply_scale(base, scale)
        alt2 = apply_scale(alt, scale)
        max_radius = max(approx_radius(base), approx_radius(alt), approx_radius(alt2))
        place_extras_apart(objs, rng, fixed_count=1, reserved=[(base.p, max_radius)])

        a_objs = clone_objects(objs)
        b_objs = [apply_scale(obj, scale) for obj in a_objs]
        c_objs = [apply_scale(obj, scale) for obj in b_objs]
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [aspect_ratio(o) for o in [a_objs[0], b_objs[0], c_objs[0]]]
        meta = build_rule_meta(
            self,
            "R2",
            1,
            involved,
            ["r"],
            ["ar(O0)"],
            "geometric",
            {"factor": factor, "axis": axis_idx, "volume_const": True},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []
        factor = float(meta.get("pattern_params", {}).get("factor", 1.4))
        axis_idx = int(meta.get("pattern_params", {}).get("axis", 0))
        def with_scale(scale: np.ndarray) -> Scene:
            objs = clone_objects(scene_c.objects)
            objs = [apply_scale(obj, scale) for obj in objs]
            return scene_from_objects(objs)

        squeeze = 1.0 / math.sqrt(factor)
        good_scale = np.ones(3)
        good_scale[axis_idx] = factor
        for i in range(3):
            if i != axis_idx:
                good_scale[i] = squeeze

        bad_skew = good_scale.copy()
        other_axes = [i for i in range(3) if i != axis_idx]
        bad_skew[other_axes[0]] = squeeze * 0.6
        bad_skew[other_axes[1]] = squeeze * 1.4

        bad_nonconst = np.ones(3)
        bad_nonconst[axis_idx] = factor
        bad_nonconst[other_axes[0]] = 1.0
        bad_nonconst[other_axes[1]] = 1.0

        flip_scale = good_scale.copy()
        flip_scale[axis_idx] = 1.0 / factor
        for i in range(3):
            if i != axis_idx:
                flip_scale[i] = math.sqrt(factor)

        distractors = [
            with_scale(bad_skew),
            with_scale(bad_nonconst),
            with_scale(flip_scale),
        ]
        reasons = [
            "非等比拉伸，比例不符",
            "仅单轴放大，体积不守恒",
            "拉伸方向相反，比例不符",
        ]
        return distractors, reasons


@dataclass
class R1_3FixedAxisRotation(Rule):
    def __init__(self) -> None:
        super().__init__("R1-3", RuleDifficulty.SIMPLE, "固定轴旋转", "绕固定轴等差旋转")

    def sample_params(self, rng) -> Dict:
        axis_idx = int(rng.integers(0, 3))
        theta = float(rng.uniform(math.pi / 5, 2 * math.pi / 5))
        return {"axis": axis_idx, "theta": theta}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 1)
        involved = [0]
        axis_idx = params["axis"]
        theta = params["theta"]
        delta = np.zeros(3)
        delta[axis_idx] = theta

        a_objs = clone_objects(objs)
        b_objs = clone_objects(objs)
        b_objs[0] = apply_rotation(b_objs[0], delta)
        c_objs = clone_objects(b_objs)
        c_objs[0] = apply_rotation(c_objs[0], delta)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [o.rotation[axis_idx] for o in [a_objs[0], b_objs[0], c_objs[0]]]
        meta = build_rule_meta(
            self, "R1", 1, involved, ["R"], [f"axis{axis_idx+1}(O0)"], "arithmetic", {"axis": axis_idx, "theta": theta}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []
        axis_idx = int(meta.get("pattern_params", {}).get("axis", 0))
        theta = float(meta.get("pattern_params", {}).get("theta", math.pi / 6))
        delta_vec = np.zeros(3)
        delta_vec[axis_idx] = theta
        alt_axis = (axis_idx + 1) % 3
        alt_delta = np.zeros(3)
        alt_delta[alt_axis] = theta

        base_rot = scene_c.objects[0].rotation - 2 * delta_vec

        def with_rotation(rot_vec: np.ndarray) -> Scene:
            objs = clone_objects(scene_c.objects)
            objs[0].rotation = rot_vec.copy()
            return scene_from_objects(objs)

        distractors = [
            with_rotation(base_rot),
            with_rotation(base_rot + 3 * delta_vec),
            with_rotation(base_rot + 2 * alt_delta),
        ]
        reasons = [
            "旋转幅度不足，未按等差延续",
            "旋转幅度过大，未按等差延续",
            "旋转轴错误",
        ]
        return distractors, reasons


@dataclass
class R1_4RotationDiscrete(Rule):
    def __init__(self) -> None:
        super().__init__("R1-4", RuleDifficulty.SIMPLE, "旋转状态离散循环", "0/90/180 度离散旋转")

    def sample_params(self, rng) -> Dict:
        axis_idx = int(rng.integers(0, 3))
        return {"axis": axis_idx}

    def generate_triplet(self, params, rng):
        axis_idx = params["axis"]
        objs = init_objects(rng, 1)
        non_sphere = [s for s in SHAPES if s != "sphere"]
        objs[0].shape = str(rng.choice(non_sphere))
        involved = [0]
        base = objs[0]
        deltas = [0.0, math.pi / 2, math.pi]
        a_objs = [apply_rotation(base, self._delta_vec(axis_idx, deltas[0]))]
        b_objs = [apply_rotation(base, self._delta_vec(axis_idx, deltas[1]))]
        c_objs = [apply_rotation(base, self._delta_vec(axis_idx, deltas[2]))]
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [o.rotation[axis_idx] for o in [a_objs[0], b_objs[0], c_objs[0]]]
        meta = build_rule_meta(
            self,
            "R1",
            1,
            involved,
            ["R"],
            [f"axis{axis_idx+1}(O0)"],
            "discrete",
            {"states_rad": deltas, "axis": axis_idx},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    @staticmethod
    def _delta_vec(axis_idx: int, angle: float) -> np.ndarray:
        v = np.zeros(3)
        v[axis_idx] = angle
        return v


@dataclass
class R1_5TranslationArithmetic(Rule):
    def __init__(self) -> None:
        super().__init__("R1-5", RuleDifficulty.SIMPLE, "固定向量平移", "等差平移")

    def sample_params(self, rng) -> Dict:
        delta = rng.uniform(0.25, 0.5, size=3) * rng.choice([-1, 1], size=3)
        return {"delta": delta.tolist()}

    def generate_triplet(self, params, rng):
        delta = np.array(params["delta"])
        objs = init_objects(rng, 1)
        involved = [0]
        a_objs = clone_objects(objs)
        b_objs = clone_objects(objs)
        b_objs[0] = apply_translation(b_objs[0], delta)
        c_objs = clone_objects(b_objs)
        c_objs[0] = apply_translation(c_objs[0], delta)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [a_objs[0].p.tolist(), b_objs[0].p.tolist(), c_objs[0].p.tolist()]
        meta = build_rule_meta(
            self, "R1", 1, involved, ["p"], ["p(O0)"], "arithmetic", {"delta": delta.tolist()}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []
        obj = scene_c.objects[0]
        delta = np.array(meta.get("pattern_params", {}).get("delta", [0.0, 0.0, 0.0]), dtype=float)
        if np.linalg.norm(delta) < 1e-6:
            delta = rng.uniform(0.3, 0.6, size=3) * rng.choice([-1, 1], size=3)

        def with_position(offset: np.ndarray) -> Scene:
            objs = clone_objects(scene_c.objects)
            objs[0].p = obj.p + offset
            return scene_from_objects(objs)

        ortho = np.array([delta[1], -delta[0], delta[2]])
        if np.linalg.norm(ortho) < 1e-6:
            ortho = rng.uniform(0.4, 0.8, size=3) * rng.choice([-1, 1], size=3)

        distractors = [
            with_position(-delta),
            with_position(delta * 2.0),
            with_position(ortho),
        ]
        reasons = [
            "平移方向反向",
            "平移幅度过大",
            "平移方向改变",
        ]
        return distractors, reasons


@dataclass
class S08TranslationDiscrete(Rule):
    def __init__(self) -> None:
        super().__init__("S08", RuleDifficulty.SIMPLE, "平移离散开关", "位置在两点间 ABA 切换")

    def sample_params(self, rng) -> Dict:
        delta = rng.uniform(0.3, 0.6, size=3) * rng.choice([-1, 1], size=3)
        return {"delta": delta.tolist()}

    def generate_triplet(self, params, rng):
        delta = np.array(params["delta"])
        objs = init_objects(rng, 1)
        involved = [0]
        base = objs[0]
        alt = apply_translation(base, delta)
        a_objs = [base.copy()]
        b_objs = [alt]
        c_objs = [base.copy()]
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [o.p.tolist() for o in [a_objs[0], b_objs[0], c_objs[0]]]
        meta = build_rule_meta(
            self, "R1", 1, involved, ["p"], ["p(O0)"], "discrete", {"positions": ["c1", "c2", "c1"]}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class R1_6DensityArithmetic(Rule):
    def __init__(self) -> None:
        super().__init__("R1-6", RuleDifficulty.SIMPLE, "密度等差", "density 按等差变化")

    def sample_params(self, rng) -> Dict:
        delta_ratio = float(rng.uniform(3.0, 4.0))
        return {"delta_ratio": delta_ratio}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 2)
        involved = [0]
        a_objs = clone_objects(objs)
        b_objs = clone_objects(objs)
        c_objs = clone_objects(objs)
        base = a_objs[0]
        delta_ratio = float(params["delta_ratio"])
        if delta_ratio < 0:
            min_density = 0.2
            max_neg_ratio = (base.density - min_density) / (2 * base.density)
            if max_neg_ratio < 0.35:
                delta_ratio = abs(delta_ratio)
            else:
                delta_ratio = -min(abs(delta_ratio), max_neg_ratio)
        delta = base.density * delta_ratio
        b_density = base.density + delta
        c_density = b_density + delta
        b_objs[0].density = b_density
        c_objs[0].density = c_density
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [base.density, b_objs[0].density, c_objs[0].density]
        meta = build_rule_meta(
            self, "R1", 1, involved, ["d"], ["den(O0)"], "arithmetic", {"delta": delta}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []
        v = meta.get("v", {})
        v2 = v.get("v2", [None])[0]
        v3 = v.get("v3", [scene_c.objects[0].density])[0]
        delta = meta.get("pattern_params", {}).get("delta")
        if delta is None and v2 is not None:
            delta = float(v3) - float(v2)
        if delta is None:
            delta = 0.0
        base_density = float(v2) if v2 is not None else float(v3) - float(delta)

        wrong_factors = [4.5, 0.6, -0.1]
        correct_density = float(scene_c.objects[0].density)
        min_ratio = 2.0

        def enforce_ratio(value: float) -> float:
            value = max(float(value), 1e-3)
            if correct_density <= 0:
                return value
            ratio = max(value / correct_density, correct_density / value)
            if ratio < min_ratio:
                if value >= correct_density:
                    value = correct_density * min_ratio
                else:
                    value = correct_density / min_ratio
            return max(float(value), 1e-3)

        def with_density(value: float) -> Scene:
            objs = clone_objects(scene_c.objects)
            objs[0].density = max(float(value), 1e-3)
            return scene_from_objects(objs)

        distractors = [with_density(enforce_ratio(base_density + delta * f)) for f in wrong_factors]
        reasons = [
            "密度差值显著偏大",
            "密度差值显著偏小",
            "密度差值方向错误",
        ]
        return distractors, reasons


@dataclass
class S10DensityGeometric(Rule):
    def __init__(self) -> None:
        super().__init__("S10", RuleDifficulty.SIMPLE, "密度等比", "density 按等比变化")

    def sample_params(self, rng) -> Dict:
        k = float(rng.uniform(2.2, 3.0) if rng.random() < 0.5 else rng.uniform(0.3, 0.5))
        return {"k": k}

    def generate_triplet(self, params, rng):
        k = params["k"]
        objs = init_objects(rng, 2)
        involved = [0]
        a_objs = clone_objects(objs)
        b_objs = clone_objects(objs)
        c_objs = clone_objects(objs)
        base = a_objs[0]
        b_objs[0] = apply_density(base, k)
        c_objs[0] = apply_density(b_objs[0], k)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [base.density, b_objs[0].density, c_objs[0].density]
        meta = build_rule_meta(
            self, "R1", 1, involved, ["d"], ["den(O0)"], "geometric", {"k": k}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []
        v = meta.get("v", {})
        v2 = v.get("v2", [None])[0]
        v3 = v.get("v3", [scene_c.objects[0].density])[0]
        k = meta.get("pattern_params", {}).get("k")
        if k is None and v2:
            k = float(v3 / v2) if v2 else None
        if k is None:
            k = 1.0
        base_density = float(v2) if v2 else float(v3) / float(k if k != 0 else 1.0)

        if k >= 1.0:
            wrong_scales = [
                float(rng.uniform(0.4, 0.7)),
                float(rng.uniform(0.9, 1.2)),
                float(rng.uniform(3.2, 4.0)),
            ]
        else:
            wrong_scales = [
                float(rng.uniform(0.1, 0.25)),
                float(rng.uniform(0.8, 1.1)),
                float(rng.uniform(1.6, 2.2)),
            ]

        def with_density(value: float) -> Scene:
            objs = clone_objects(scene_c.objects)
            objs[0].density = max(float(value), 1e-3)
            return scene_from_objects(objs)

        distractors = [with_density(base_density * s) for s in wrong_scales]
        reasons = [
            "密度比例显著偏小",
            "密度比例不符合等比延续",
            "密度比例显著偏大",
        ]
        return distractors, reasons


@dataclass
class S11ShapeABA(Rule):
    def __init__(self) -> None:
        super().__init__("S11", RuleDifficulty.SIMPLE, "形状离散替换 ABA", "形状在两种类型间往返")

    def sample_params(self, rng) -> Dict:
        shape_a, shape_b = rng.choice(SHAPES, size=2, replace=False).tolist()
        return {"shape_a": shape_a, "shape_b": shape_b}

    def generate_triplet(self, params, rng):
        shape_a, shape_b = params["shape_a"], params["shape_b"]
        objs = init_objects(rng, 1)
        involved = [0]
        objs[0].shape = shape_a
        a_objs = clone_objects(objs)
        b_objs = clone_objects(objs)
        b_objs[0] = switch_shape(b_objs[0], shape_b)
        c_objs = clone_objects(objs)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [[shape_a], [shape_b], [shape_a]]
        meta = build_rule_meta(
            self, "R1", 1, involved, ["s"], ["s(O0)"], "discrete", {"shapes": [shape_a, shape_b, shape_a]}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class R1_7ShapeChangeFollow(Rule):
    def __init__(self) -> None:
        super().__init__("R1-7", RuleDifficulty.SIMPLE, "形状变化继承", "A->B 变化的位置在 C 继续变化")

    def sample_params(self, rng) -> Dict:
        return {}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 3, m=3)
        involved = [0, 1, 2]
        shapes_a = rng.choice(SHAPES, size=3, replace=True).tolist()
        while len(set(shapes_a)) == 1:
            shapes_a = rng.choice(SHAPES, size=3, replace=True).tolist()
        for i, shape in enumerate(shapes_a):
            objs[i].shape = shape

        change_idx = int(rng.integers(0, 3))
        shapes_b = list(shapes_a)
        shape_b_choices = [s for s in SHAPES if s != shapes_a[change_idx]]
        shapes_b[change_idx] = str(rng.choice(shape_b_choices))

        shapes_c = list(shapes_b)
        shape_c_choices = [s for s in SHAPES if s != shapes_b[change_idx]]
        shapes_c[change_idx] = str(rng.choice(shape_c_choices))

        a_objs = clone_objects(objs)
        b_objs = clone_objects(objs)
        c_objs = clone_objects(objs)
        for i in range(3):
            b_objs[i] = switch_shape(b_objs[i], shapes_b[i])
            c_objs[i] = switch_shape(c_objs[i], shapes_c[i])
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [shapes_a, shapes_b, shapes_c]
        meta = build_rule_meta(
            self, "R1", 3, involved, ["s", "p"], ["shape_change_mask"], "discrete", {}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if len(scene_c.objects) < 3:
            return [], []
        v = meta.get("v", {})
        shapes_a = v.get("v1")
        shapes_b = v.get("v2")
        if not shapes_a or not shapes_b:
            return [], []
        change_indices = [i for i, (sa, sb) in enumerate(zip(shapes_a, shapes_b)) if sa != sb]
        if not change_indices:
            return [], []
        change_idx = int(change_indices[0])

        shapes_c = [obj.shape for obj in scene_c.objects]

        def with_shapes(shapes: List[str]) -> Scene:
            objs = clone_objects(scene_c.objects)
            for i, shape in enumerate(shapes):
                objs[i] = switch_shape(objs[i], shape)
            return scene_from_objects(objs)

        wrong_no_change = with_shapes(shapes_b)

        wrong_idx = (change_idx + 1) % 3
        shapes_wrong_idx = list(shapes_c)
        alt_shapes = [s for s in SHAPES if s != shapes_wrong_idx[wrong_idx]]
        shapes_wrong_idx[wrong_idx] = str(rng.choice(alt_shapes))
        wrong_change_other = with_shapes(shapes_wrong_idx)

        shapes_extra = list(shapes_c)
        extra_idx = (change_idx + 2) % 3
        extra_choices = [s for s in SHAPES if s != shapes_extra[extra_idx]]
        shapes_extra[extra_idx] = str(rng.choice(extra_choices))
        wrong_extra_change = with_shapes(shapes_extra)

        return [
            wrong_no_change,
            wrong_change_other,
            wrong_extra_change,
        ], [
            "变化位置未延续（C 与 B 相同）",
            "变化位置错误（在其他位置发生变化）",
            "变化位置过多（出现额外变化）",
        ]


@dataclass
class R1_8ScaleCentroidCoupled(Rule):
    def __init__(self) -> None:
        super().__init__("R1-8", RuleDifficulty.SIMPLE, "质心守恒缩放", "缩放并平移以保持参与集合质心不变")

    def sample_params(self, rng) -> Dict:
        k = float(rng.uniform(1.2, 1.6))
        return {"k": k}

    def generate_triplet(self, params, rng):
        k = params["k"]
        m = int(rng.integers(2, 4))
        objs = init_objects(rng, k=m, m=m)
        involved = list(range(len(objs)))
        base_cent = centroid(objs)

        def scale_keep_cent(objs_in: List) -> List:
            scaled = []
            for o in objs_in:
                new_o = apply_scale(o, k)
                new_o.p = base_cent + (o.p - base_cent) / k
                scaled.append(new_o)
            return scaled

        a_objs = clone_objects(objs)
        b_objs = scale_keep_cent(objs)
        c_objs = scale_keep_cent(b_objs)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [centroid(s.objects) for s in scenes]
        meta = build_rule_meta(
            self,
            "R1",
            len(involved),
            involved,
            ["r", "p"],
            ["cent(S)"],
            "coupled",
            {"k": k},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class R1_15InverseDistanceSize(Rule):
    def __init__(self) -> None:
        super().__init__("R1-15", RuleDifficulty.SIMPLE, "距离尺寸倒数", "距离越近尺寸变化越剧烈")

    def sample_params(self, rng) -> Dict:
        k = float(rng.uniform(0.35, 0.65))
        direction = "up" if rng.random() < 0.5 else "down"
        return {"k": k, "direction": direction}

    def generate_triplet(self, params, rng):
        k = float(params["k"])
        direction = params["direction"]
        sign = 1.0 if direction == "up" else -1.0
        objs = init_objects(rng, 3, m=3)
        involved = [0, 1, 2]

        d_close = float(rng.uniform(0.35, 0.55))
        d_far = float(rng.uniform(0.85, 1.15))
        objs[0].p = np.array([0.0, 0.0, 0.0])
        objs[1].p = np.array([d_close, 0.0, 0.0])
        objs[2].p = np.array([-d_far, 0.0, 0.0])

        dists = [0.0, dist(objs[0], objs[1]), dist(objs[0], objs[2])]
        inv = [0.0, 1.0 / (dists[1] + 1e-6), 1.0 / (dists[2] + 1e-6)]
        inv_max = max(inv[1], inv[2])
        ratios = [0.0, inv[1] / inv_max, inv[2] / inv_max]
        size_factors = [1.0] + [1.0 + sign * k * ratios[i] for i in [1, 2]]
        scale_factors = [f ** (1.0 / 3.0) for f in size_factors]

        def scale_step(objs_in: List) -> List:
            scaled = clone_objects(objs_in)
            for idx in [1, 2]:
                scaled[idx] = apply_scale(scaled[idx], scale_factors[idx])
            return scaled

        a_objs = clone_objects(objs)
        b_objs = scale_step(a_objs)
        c_objs = scale_step(b_objs)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [[size(o) for o in s.objects] for s in scenes]
        meta = build_rule_meta(
            self,
            "R1",
            3,
            involved,
            ["r"],
            ["size(Oi)"],
            "inverse-distance",
            {"k": k, "direction": direction, "anchor": 0, "factors": size_factors, "distances": dists},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if len(scene_c.objects) < 3:
            return [], []
        factors = meta.get("pattern_params", {}).get("factors")
        if not factors or len(factors) != len(scene_c.objects):
            return [], []
        factors = np.array(factors, dtype=float)
        sizes_c = np.array([size(o) for o in scene_c.objects], dtype=float)
        base_sizes = sizes_c / np.where(factors == 0, 1.0, factors**2)

        def build_with(target_sizes: np.ndarray) -> Scene:
            objs = clone_objects(scene_c.objects)
            for i, target in enumerate(target_sizes.tolist()):
                cur = max(size(objs[i]), 1e-6)
                safe_target = max(float(target), 1e-6)
                scale = (safe_target / cur) ** (1 / 3)
                objs[i] = apply_scale(objs[i], scale)
            return scene_from_objects(objs)

        wrong_no_change = base_sizes * factors
        wrong_reverse = base_sizes
        swapped = factors.copy()
        swapped[1], swapped[2] = swapped[2], swapped[1]
        wrong_swap = base_sizes * (swapped ** 2)
        distractors = [build_with(x) for x in [wrong_no_change, wrong_reverse, wrong_swap]]
        reasons = [
            "变化未延续（停留在上一帧）",
            "变化方向相反（回到初始）",
            "距离与变化强度对应关系错误",
        ]
        return distractors, reasons


@dataclass
class R1_16InverseDistanceDensity(Rule):
    def __init__(self) -> None:
        super().__init__("R1-16", RuleDifficulty.SIMPLE, "距离密度倒数", "距离越近密度变化越剧烈")

    def sample_params(self, rng) -> Dict:
        k = float(rng.uniform(0.5, 0.8))
        direction = "up" if rng.random() < 0.5 else "down"
        return {"k": k, "direction": direction}

    def generate_triplet(self, params, rng):
        k = float(params["k"])
        direction = params["direction"]
        sign = 1.0 if direction == "up" else -1.0
        objs = init_objects(rng, 3, m=3)
        involved = [0, 1, 2]

        d_close = float(rng.uniform(0.35, 0.55))
        d_far = float(rng.uniform(0.85, 1.15))
        objs[0].p = np.array([0.0, 0.0, 0.0])
        objs[1].p = np.array([d_close, 0.0, 0.0])
        objs[2].p = np.array([-d_far, 0.0, 0.0])

        dists = [0.0, dist(objs[0], objs[1]), dist(objs[0], objs[2])]
        inv = [0.0, 1.0 / (dists[1] + 1e-6), 1.0 / (dists[2] + 1e-6)]
        inv_max = max(inv[1], inv[2])
        ratios = [0.0, inv[1] / inv_max, inv[2] / inv_max]
        density_factors = [1.0] + [max(1.0 + sign * k * ratios[i], 0.2) for i in [1, 2]]

        def density_step(objs_in: List) -> List:
            scaled = clone_objects(objs_in)
            for idx in [1, 2]:
                scaled[idx] = apply_density(scaled[idx], density_factors[idx])
            return scaled

        a_objs = clone_objects(objs)
        b_objs = density_step(a_objs)
        c_objs = density_step(b_objs)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [[o.density for o in s.objects] for s in scenes]
        meta = build_rule_meta(
            self,
            "R1",
            3,
            involved,
            ["d"],
            ["den(Oi)"],
            "inverse-distance",
            {"k": k, "direction": direction, "anchor": 0, "factors": density_factors, "distances": dists},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if len(scene_c.objects) < 3:
            return [], []
        factors = meta.get("pattern_params", {}).get("factors")
        if not factors or len(factors) != len(scene_c.objects):
            return [], []
        factors = np.array(factors, dtype=float)
        dens_c = np.array([o.density for o in scene_c.objects], dtype=float)
        base_dens = dens_c / np.where(factors == 0, 1.0, factors**2)

        def build_with(target_dens: np.ndarray) -> Scene:
            objs = clone_objects(scene_c.objects)
            for i, target in enumerate(target_dens.tolist()):
                objs[i] = objs[i].copy()
                objs[i].density = max(float(target), 1e-3)
            return scene_from_objects(objs)

        wrong_no_change = base_dens * factors
        wrong_reverse = base_dens
        swapped = factors.copy()
        swapped[1], swapped[2] = swapped[2], swapped[1]
        wrong_swap = base_dens * (swapped ** 2)
        distractors = [build_with(x) for x in [wrong_no_change, wrong_reverse, wrong_swap]]
        reasons = [
            "变化未延续（停留在上一帧）",
            "变化方向相反（回到初始）",
            "距离与变化强度对应关系错误",
        ]
        return distractors, reasons


@dataclass
class R1_17ShapeCountArithmetic(Rule):
    def __init__(self) -> None:
        super().__init__("R1-17", RuleDifficulty.SIMPLE, "几何体个数变化", "形状计数等差增加")

    def sample_params(self, rng) -> Dict:
        return {}

    def generate_triplet(self, params, rng):
        shape_count = int(rng.integers(2, 5))
        shapes = rng.choice(SHAPES, size=shape_count, replace=False).tolist()
        max_total = 6

        while True:
            total_a = int(rng.integers(shape_count, max_total + 1))
            max_delta_sum = (max_total - total_a) // 2
            if max_delta_sum >= 1:
                sum_delta = int(rng.integers(1, max_delta_sum + 1))
                break

        counts_a = [1] * shape_count
        for _ in range(total_a - shape_count):
            counts_a[int(rng.integers(0, shape_count))] += 1

        delta_counts = [0] * shape_count
        for _ in range(sum_delta):
            delta_counts[int(rng.integers(0, shape_count))] += 1

        counts_b = [a + d for a, d in zip(counts_a, delta_counts)]
        counts_c = [a + 2 * d for a, d in zip(counts_a, delta_counts)]

        def build_scene(counts: List[int]) -> Scene:
            objs = []
            for shape, num in zip(shapes, counts):
                for _ in range(int(num)):
                    objs.append(random_object(rng, shape=shape))
            rng.shuffle(objs)
            return scene_from_objects(objs)

        scene_a = build_scene(counts_a)
        scene_b = build_scene(counts_b)
        scene_c = build_scene(counts_c)
        scenes = [scene_a, scene_b, scene_c]
        v = [counts_a, counts_b, counts_c]
        involved = list(range(len(scene_c.objects)))
        meta = build_rule_meta(
            self,
            "R1",
            len(involved),
            involved,
            ["s"],
            ["count(shape)"],
            "count-arithmetic",
            {"shapes": shapes, "counts_a": counts_a, "delta_counts": delta_counts},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        params = meta.get("pattern_params", {})
        shapes = params.get("shapes")
        counts_a = params.get("counts_a")
        delta_counts = params.get("delta_counts")
        if not shapes or not counts_a or not delta_counts:
            return [], []
        if len(shapes) != len(counts_a) or len(shapes) != len(delta_counts):
            return [], []

        counts_b = [a + d for a, d in zip(counts_a, delta_counts)]
        sum_delta = int(sum(delta_counts))

        def build_scene(counts: List[int]) -> Scene:
            objs = []
            for shape, num in zip(shapes, counts):
                for _ in range(int(num)):
                    objs.append(random_object(rng, shape=shape))
            rng.shuffle(objs)
            return scene_from_objects(objs)

        delta_perm = list(rng.permutation(delta_counts))
        if delta_perm == delta_counts:
            idx_from = next((i for i, d in enumerate(delta_perm) if d > 0), None)
            if idx_from is not None:
                idx_to = (idx_from + 1) % len(delta_perm)
                delta_perm[idx_from] -= 1
                delta_perm[idx_to] += 1

        wrong_no = counts_b
        wrong_shape = [b + d for b, d in zip(counts_b, delta_perm)]

        if sum_delta > 1:
            delta_under = list(delta_counts)
            idx = int(rng.choice([i for i, d in enumerate(delta_under) if d > 0]))
            delta_under[idx] -= 1
            wrong_step = [b + d for b, d in zip(counts_b, delta_under)]
        else:
            wrong_step = counts_a

        distractors = [
            build_scene(wrong_no),
            build_scene(wrong_shape),
            build_scene(wrong_step),
        ]
        reasons = [
            "数量未继续增加，停留在上一帧",
            "增加形状对应关系错误",
            "增加幅度不一致",
        ]
        return distractors, reasons


def build_simple_rules() -> List[Rule]:
    return [
        R1_1ScaleArithmetic(),
        R1_2AxisPermutation(),
        R2_2AnisotropicGeometric(),
        R1_3FixedAxisRotation(),
        R1_4RotationDiscrete(),
        R1_5TranslationArithmetic(),
        R1_6DensityArithmetic(),
        R1_7ShapeChangeFollow(),
        R1_8ScaleCentroidCoupled(),
        R1_15InverseDistanceSize(),
        R1_16InverseDistanceDensity(),
        R1_17ShapeCountArithmetic(),
    ]
