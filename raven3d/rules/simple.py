from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .base import Rule, RuleDifficulty
from .utils import (
    SHAPES,
    apply_density,
    apply_rotation,
    apply_scale,
    apply_translation,
    aspect_ratio,
    axis,
    build_rule_meta,
    centroid,
    clone_objects,
    init_objects,
    scene_from_objects,
    size,
    switch_shape,
)


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
        involved = [0]
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
class S02ScaleArithmetic(Rule):
    def __init__(self) -> None:
        super().__init__("S02", RuleDifficulty.SIMPLE, "等差统一缩放", "size 按等差递进")

    def sample_params(self, rng) -> Dict:
        delta_ratio = float(rng.uniform(0.15, 0.35))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        return {"delta_ratio": delta_ratio * sign}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 1)
        involved = [0]
        base_obj = objs[0]
        base_size = size(base_obj)
        delta = base_size * params["delta_ratio"]
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
class S04AnisotropicToggle(Rule):
    def __init__(self) -> None:
        super().__init__("S04", RuleDifficulty.SIMPLE, "各向异性比例切换", "aspect ratio 在两种状态 ABA 切换")

    def sample_params(self, rng) -> Dict:
        factor = float(rng.uniform(1.3, 1.8))
        return {"factor": factor}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 1)
        involved = [0]
        factor = params["factor"]
        base = objs[0]
        alt = apply_scale(base, factor, axis_mask=[True, False, False])

        a_objs = [base.copy()]
        b_objs = [alt]
        c_objs = [base.copy()]
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [aspect_ratio(o) for o in [a_objs[0], b_objs[0], c_objs[0]]]
        meta = build_rule_meta(
            self, "R1", 1, involved, ["r"], ["ar(O0)"], "discrete", {"states": ["c1", "c2", "c1"]}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class S05FixedAxisRotation(Rule):
    def __init__(self) -> None:
        super().__init__("S05", RuleDifficulty.SIMPLE, "固定轴旋转", "绕固定轴等差旋转")

    def sample_params(self, rng) -> Dict:
        axis_idx = int(rng.integers(0, 3))
        theta = float(rng.uniform(math.pi / 10, math.pi / 5))
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


@dataclass
class S06RotationDiscrete(Rule):
    def __init__(self) -> None:
        super().__init__("S06", RuleDifficulty.SIMPLE, "旋转状态离散循环", "0/90/180 度离散旋转")

    def sample_params(self, rng) -> Dict:
        axis_idx = int(rng.integers(0, 3))
        return {"axis": axis_idx}

    def generate_triplet(self, params, rng):
        axis_idx = params["axis"]
        objs = init_objects(rng, 1)
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
class S07TranslationArithmetic(Rule):
    def __init__(self) -> None:
        super().__init__("S07", RuleDifficulty.SIMPLE, "固定向量平移", "等差平移")

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
class S09DensityArithmetic(Rule):
    def __init__(self) -> None:
        super().__init__("S09", RuleDifficulty.SIMPLE, "密度等差", "density 按等差变化")

    def sample_params(self, rng) -> Dict:
        delta_ratio = float(rng.uniform(0.35, 0.6))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        return {"delta_ratio": delta_ratio * sign}

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


@dataclass
class S10DensityGeometric(Rule):
    def __init__(self) -> None:
        super().__init__("S10", RuleDifficulty.SIMPLE, "密度等比", "density 按等比变化")

    def sample_params(self, rng) -> Dict:
        k = float(rng.uniform(1.7, 2.2) if rng.random() < 0.5 else rng.uniform(0.5, 0.7))
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
class S12ShapeABC(Rule):
    def __init__(self) -> None:
        super().__init__("S12", RuleDifficulty.SIMPLE, "形状离散替换 ABC", "三步三种不同形状")

    def sample_params(self, rng) -> Dict:
        shapes = rng.choice(SHAPES, size=3, replace=False).tolist()
        return {"shapes": shapes}

    def generate_triplet(self, params, rng):
        shapes = params["shapes"]
        objs = init_objects(rng, 1)
        involved = [0]
        objs[0].shape = shapes[0]
        a_objs = clone_objects(objs)
        b_objs = clone_objects(objs)
        b_objs[0] = switch_shape(b_objs[0], shapes[1])
        c_objs = clone_objects(objs)
        c_objs[0] = switch_shape(c_objs[0], shapes[2])
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [[shapes[0]], [shapes[1]], [shapes[2]]]
        meta = build_rule_meta(
            self, "R1", 1, involved, ["s"], ["s(O0)"], "discrete", {"shapes": shapes}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta


@dataclass
class S13ScaleCentroidCoupled(Rule):
    def __init__(self) -> None:
        super().__init__("S13", RuleDifficulty.SIMPLE, "尺度-位置联动", "缩放并平移以保持参与集合质心不变")

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
class S14Identity(Rule):
    def __init__(self) -> None:
        super().__init__("S14", RuleDifficulty.SIMPLE, "恒等规则", "A/B/C 完全相同")

    def sample_params(self, rng) -> Dict:
        return {}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 2)
        involved = []
        scene = scene_from_objects(objs)
        scenes = [scene, scene_from_objects(clone_objects(objs)), scene_from_objects(clone_objects(objs))]
        v = [["same"], ["same"], ["same"]]
        meta = build_rule_meta(
            self, "R1", 0, involved, ["s", "r", "p", "R", "d"], ["identity"], "constant", {}, v, scenes
        )
        return scenes[0], scenes[1], scenes[2], meta


def build_simple_rules() -> List[Rule]:
    return [
        S01ScaleGeometric(),
        S02ScaleArithmetic(),
        S03SingleAxisGeometric(),
        S04AnisotropicToggle(),
        S05FixedAxisRotation(),
        S06RotationDiscrete(),
        S07TranslationArithmetic(),
        S08TranslationDiscrete(),
        S09DensityArithmetic(),
        S10DensityGeometric(),
        S11ShapeABA(),
        S12ShapeABC(),
        S13ScaleCentroidCoupled(),
        S14Identity(),
    ]
