from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .base import Rule, RuleDifficulty
from .utils import (
    SHAPES,
    ang,
    apply_density,
    apply_rotation,
    apply_scale,
    apply_translation,
    aspect_ratio,
    axis,
    build_rule_meta,
    centroid,
    clone_objects,
    direction,
    dist,
    init_objects,
    order_indices_x,
    random_object,
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
class R2_1DistanceGeometric(Rule):
    def __init__(self) -> None:
        super().__init__("R2-1", RuleDifficulty.MEDIUM, "成对距离等比", "dist(i,j) 等比")

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

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[List[Scene], List[str]]:
        if len(scene_c.objects) < 2:
            return [], []
        obj0, obj1 = scene_c.objects[0], scene_c.objects[1]
        base_dist = dist(obj0, obj1)
        if base_dist < 1e-6:
            return [], []
        base_dir = (obj1.p - obj0.p) / base_dist
        origin = obj0.p.copy()
        wrong_scales = [
            float(rng.uniform(0.35, 0.55)),
            float(rng.uniform(1.6, 2.2)),
            float(rng.uniform(2.4, 3.0)),
        ]

        def place(dir_vec: np.ndarray, distance: float) -> Scene:
            o0, o1 = clone_objects(scene_c.objects)
            o0.p = origin.copy()
            o1.p = origin + dir_vec * distance
            return scene_from_objects([o0, o1])

        alt_dir = _unit_vector(rng)
        distractors = [
            place(base_dir, base_dist * wrong_scales[0]),
            place(base_dir, base_dist * wrong_scales[1]),
            place(alt_dir, base_dist * wrong_scales[2]),
        ]
        reasons = [
            "距离显著偏小，未满足等比延续",
            "距离显著偏大，未满足等比延续",
            "方向改变且距离偏离等比延续",
        ]
        return distractors, reasons


@dataclass
class R2_3DirectionRotate(Rule):
    def __init__(self) -> None:
        super().__init__("R2-3", RuleDifficulty.MEDIUM, "方向旋转等差", "dir 旋转, dist 保持")

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

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[List[Scene], List[str]]:
        if len(scene_c.objects) < 2:
            return [], []
        obj0, obj1 = scene_c.objects[0], scene_c.objects[1]
        base_dist = dist(obj0, obj1)
        if base_dist < 1e-6:
            return [], []
        base_dir = (obj1.p - obj0.p) / base_dist
        origin = obj0.p.copy()
        big_angle = float(rng.uniform(math.pi / 3, math.pi / 2))

        def rotate_dir(angle: float) -> np.ndarray:
            rot = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
            return rot @ base_dir

        def place(dir_vec: np.ndarray, distance: float) -> Scene:
            o0, o1 = clone_objects(scene_c.objects)
            o0.p = origin.copy()
            o1.p = origin + dir_vec * distance
            return scene_from_objects([o0, o1])

        dist_far = base_dist * float(rng.uniform(1.6, 2.1))
        dist_near = base_dist * float(rng.uniform(0.4, 0.7))
        distractors = [
            place(rotate_dir(big_angle), base_dist),
            place(base_dir, dist_far),
            place(rotate_dir(-big_angle), dist_near),
        ]
        reasons = [
            "方向旋转幅度偏离等差规律",
            "距离不保持恒定",
            "方向与距离均偏离规则",
        ]
        return distractors, reasons


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

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[List[Scene], List[str]]:
        if len(scene_c.objects) < 2:
            return [], []
        obj0, obj1 = scene_c.objects[0], scene_c.objects[1]
        touch_threshold = 0.6 * (np.linalg.norm(obj0.r) + np.linalg.norm(obj1.r)) + 0.05
        direction_vec = _unit_vector(rng)
        distances = [touch_threshold + rng.uniform(0.4, 0.8) for _ in range(3)]
        distractors = []
        reasons = []
        for distance in distances:
            o0, o1 = clone_objects(scene_c.objects)
            o0.p = -direction_vec * distance / 2
            o1.p = direction_vec * distance / 2
            distractors.append(scene_from_objects([o0, o1]))
            reasons.append("两物体距离过大，未接触")
        return distractors, reasons


@dataclass
class R2_9AcceleratedRotation(Rule):
    def __init__(self) -> None:
        super().__init__("R2-9", RuleDifficulty.MEDIUM, "加速旋转", "旋转幅度递增")

    def sample_params(self, rng) -> Dict:
        obj_count = int(rng.integers(1, 4))
        axis_idx = int(rng.integers(0, 3))
        sign = 1 if rng.random() < 0.5 else -1
        delta1 = float(rng.uniform(math.pi / 4, math.pi / 2)) * sign
        accel = float(rng.uniform(1.35, 1.75))
        delta2 = delta1 * accel
        return {"count": obj_count, "axis": axis_idx, "delta1": delta1, "delta2": delta2}

    def generate_triplet(self, params, rng):
        count = int(params["count"])
        axis_idx = int(params["axis"])
        delta1 = float(params["delta1"])
        delta2 = float(params["delta2"])
        objs = init_objects(rng, k=count, m=count)
        involved = list(range(len(objs)))

        delta_vec1 = np.zeros(3)
        delta_vec1[axis_idx] = delta1
        delta_vec2 = np.zeros(3)
        delta_vec2[axis_idx] = delta2

        a_objs = clone_objects(objs)
        b_objs = [apply_rotation(o, delta_vec1) for o in a_objs]
        c_objs = [apply_rotation(o, delta_vec2) for o in b_objs]
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]

        def rot_mag(objs_in: List) -> float:
            return float(np.mean([np.linalg.norm(o.rotation) for o in objs_in]))

        v = [rot_mag(a_objs), rot_mag(b_objs), rot_mag(c_objs)]
        meta = build_rule_meta(
            self,
            "R2",
            len(involved),
            involved,
            ["R"],
            ["rot(Oi)"],
            "accelerated-rotation",
            {"axis": axis_idx, "delta1": delta1, "delta2": delta2},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[List[Scene], List[str]]:
        if not scene_c.objects:
            return [], []
        params = meta.get("pattern_params", {})
        axis_idx = int(params.get("axis", 0))
        delta2 = float(params.get("delta2", 0.0))
        if abs(delta2) < 1e-6:
            return [], []

        delta_vec = np.zeros(3)
        delta_vec[axis_idx] = delta2

        def step(objs_in: List, delta: np.ndarray) -> List:
            return [apply_rotation(o, delta) for o in objs_in]

        b_objs = step(clone_objects(scene_c.objects), -delta_vec)
        small = step(clone_objects(b_objs), delta_vec * 0.4)
        same = clone_objects(b_objs)
        reverse = step(clone_objects(b_objs), -delta_vec * 0.6)

        distractors = [
            scene_from_objects(same),
            scene_from_objects(small),
            scene_from_objects(reverse),
        ]
        reasons = [
            "旋转幅度未增大，停留在上一帧",
            "旋转幅度减小，未满足加速",
            "旋转方向反向，未满足加速",
        ]
        return distractors, reasons


@dataclass
class R2_10GeometricFusion(Rule):
    def __init__(self) -> None:
        super().__init__("R2-10", RuleDifficulty.MEDIUM, "几何融合", "等差步伐向中心融合")

    def sample_params(self, rng) -> Dict:
        delta = float(rng.uniform(0.28, 0.38))
        return {"delta": delta}

    def generate_triplet(self, params, rng):
        delta = float(params["delta"])
        objs = init_objects(rng, k=3, m=3)
        involved = [0, 1, 2]

        for obj in objs:
            direction_vec = _unit_vector(rng)
            radius = float(rng.uniform(0.75, 1.1))
            obj.p = direction_vec * radius

        center = centroid(objs)
        scales = [1.0, 1.0 - delta, 1.0 - 2 * delta]

        def place(scale: float) -> List:
            arranged = clone_objects(objs)
            for o in arranged:
                o.p = center + (o.p - center) * scale
            return arranged

        a_objs = place(scales[0])
        b_objs = place(scales[1])
        c_objs = place(scales[2])
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [[float(np.linalg.norm(o.p - center)) for o in s.objects] for s in scenes]
        meta = build_rule_meta(
            self,
            "R2",
            3,
            involved,
            ["p"],
            ["cent(S)"],
            "fusion-arithmetic",
            {"delta": delta, "center": center.tolist(), "scales": scales},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[List[Scene], List[str]]:
        if len(scene_c.objects) < 3:
            return [], []
        params = meta.get("pattern_params", {})
        scales = params.get("scales")
        center = np.array(params.get("center", [0.0, 0.0, 0.0]), dtype=float)
        if not scales or len(scales) != 3:
            return [], []
        s2, s3 = float(scales[1]), float(scales[2])
        if abs(s3) < 1e-6:
            return [], []

        def rescale(objs_in: List, target_scale: float) -> List:
            arranged = clone_objects(objs_in)
            for o in arranged:
                vec = o.p - center
                o.p = center + vec * (target_scale / s3)
            return arranged

        same = rescale(scene_c.objects, s2)
        small = rescale(scene_c.objects, (s2 + s3) / 2)
        overshoot = rescale(scene_c.objects, max(s3 - (s2 - s3), 0.08))
        distractors = [scene_from_objects(x) for x in [same, small, overshoot]]
        reasons = [
            "融合未延续，停留在上一帧",
            "融合步幅过小，未达到等差推进",
            "融合过度，步幅偏大",
        ]
        return distractors, reasons


@dataclass
class R2_11OrbitalRotation(Rule):
    def __init__(self) -> None:
        super().__init__("R2-11", RuleDifficulty.MEDIUM, "行星公转", "多物体绕中心球体等速公转")

    def sample_params(self, rng) -> Dict:
        count = int(rng.integers(2, 5))
        return {"count": count}

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

    def generate_triplet(self, params, rng):
        count = int(params["count"])
        center = np.zeros(3)
        orbit_shapes = [s for s in SHAPES if s != "sphere"]
        shapes = rng.choice(orbit_shapes, size=count, replace=False).tolist()
        orbiters = [random_object(rng, shape=s) for s in shapes]
        center_obj = random_object(rng, shape="sphere")
        center_obj.p = center.copy()

        radii = self._unique_samples(rng, count, 0.7, 1.1, 0.08)
        base_angles = [float(rng.uniform(0, 2 * math.pi)) for _ in range(count)]
        delta_base = float(rng.uniform(math.pi / 4, math.pi / 2))
        delta_vals = self._unique_samples(rng, count, delta_base * 0.7, delta_base * 1.3, 0.2)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        deltas = [sign * d for d in delta_vals]

        def build_frame(step: int) -> List:
            objs = [center_obj.copy()]
            for i, orb in enumerate(orbiters):
                obj = orb.copy()
                angle = base_angles[i] + deltas[i] * step
                obj.p = center + np.array(
                    [radii[i] * math.cos(angle), radii[i] * math.sin(angle), 0.0]
                )
                objs.append(obj)
            return objs

        a_objs = build_frame(0)
        b_objs = build_frame(1)
        c_objs = build_frame(2)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        orbit_indices = list(range(1, count + 1))
        v = []
        for s in scenes:
            angles = []
            for idx in orbit_indices:
                vec = s.objects[idx].p - center
                angles.append(float(math.atan2(vec[1], vec[0])))
            v.append(angles)
        meta = build_rule_meta(
            self,
            "R2",
            len(orbit_indices),
            list(range(len(a_objs))),
            ["p"],
            ["orbit(theta_i)"],
            "orbit-constant",
            {
                "center": center.tolist(),
                "radii": radii,
                "angles": base_angles,
                "deltas": deltas,
                "orbit_indices": orbit_indices,
            },
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[List[Scene], List[str]]:
        if not scene_c.objects:
            return [], []
        params = meta.get("pattern_params", {})
        center = np.array(params.get("center", [0.0, 0.0, 0.0]), dtype=float)
        deltas = params.get("deltas")
        orbit_indices = params.get("orbit_indices")
        if deltas is None or orbit_indices is None:
            return [], []
        if len(deltas) != len(orbit_indices):
            return [], []

        def rotate_step(objs_in: List, scale: float) -> List:
            out = clone_objects(objs_in)
            for idx, delta in zip(orbit_indices, deltas):
                out[idx].p = self._rotate_about_center(out[idx].p, center, delta * scale)
            return out

        b_objs = rotate_step(scene_c.objects, -1.0)
        same = clone_objects(b_objs)
        small = rotate_step(b_objs, 0.4)
        reverse = rotate_step(b_objs, -0.6)
        distractors = [
            scene_from_objects(same),
            scene_from_objects(small),
            scene_from_objects(reverse),
        ]
        reasons = [
            "旋转未延续，停留在上一帧",
            "旋转幅度减小",
            "旋转方向反向",
        ]
        return distractors, reasons


@dataclass
class R2_12PoseShift(Rule):
    def __init__(self) -> None:
        super().__init__("R2-12", RuleDifficulty.MEDIUM, "易位姿态转换", "按形状匹配的姿态变化持续")

    def sample_params(self, rng) -> Dict:
        count = int(rng.integers(2, 6))
        return {"count": count}

    def generate_triplet(self, params, rng):
        count = int(params["count"])
        shape_pool = [s for s in SHAPES if s not in ("sphere", "cylinder")]
        shapes = rng.choice(shape_pool, size=count, replace=False).tolist()
        objs = [random_object(rng, shape=s) for s in shapes]
        involved = list(range(count))

        axes = []
        deltas = []
        for _ in range(count):
            axis_idx = int(rng.integers(0, 3))
            roll = rng.random()
            if roll < 0.4:
                delta = float(rng.uniform(math.pi / 5, math.pi / 3))
            elif roll < 0.8:
                delta = -float(rng.uniform(math.pi / 5, math.pi / 3))
            else:
                delta = 0.0
            axes.append(axis_idx)
            deltas.append(delta)
        if all(abs(d) < 1e-6 for d in deltas):
            idx = int(rng.integers(0, count))
            deltas[idx] = float(rng.uniform(math.pi / 5, math.pi / 3))

        def build_frame(step: int) -> List:
            arranged = clone_objects(objs)
            for i, obj in enumerate(arranged):
                delta_vec = np.zeros(3)
                delta_vec[axes[i]] = deltas[i] * step
                obj.rotation = obj.rotation + delta_vec
                obj.p = rng.uniform(-0.8, 0.8, size=3)
            return arranged

        a_objs = build_frame(0)
        b_objs = build_frame(1)
        c_objs = build_frame(2)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [[float(np.linalg.norm(o.rotation)) for o in s.objects] for s in scenes]
        meta = build_rule_meta(
            self,
            "R2",
            len(involved),
            involved,
            ["R"],
            ["rot(Oi)"],
            "pose-shift",
            {"shapes": shapes, "axes": axes, "deltas": deltas},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[List[Scene], List[str]]:
        if not scene_c.objects:
            return [], []
        params = meta.get("pattern_params", {})
        axes = params.get("axes")
        deltas = params.get("deltas")
        if axes is None or deltas is None:
            return [], []
        if len(axes) != len(scene_c.objects) or len(deltas) != len(scene_c.objects):
            return [], []

        def apply_step(objs_in: List, scale: float) -> List:
            out = clone_objects(objs_in)
            for i, obj in enumerate(out):
                delta_vec = np.zeros(3)
                delta_vec[int(axes[i])] = float(deltas[i]) * scale
                obj.rotation = obj.rotation + delta_vec
            return out

        b_objs = apply_step(scene_c.objects, -1.0)
        same = clone_objects(b_objs)
        small = apply_step(b_objs, 0.4)
        reverse = apply_step(b_objs, -0.6)
        distractors = [
            scene_from_objects(same),
            scene_from_objects(small),
            scene_from_objects(reverse),
        ]
        reasons = [
            "姿态未延续变化，停留在上一帧",
            "姿态变化幅度减小",
            "姿态变化方向反向",
        ]
        return distractors, reasons


@dataclass
class R2_13PositionScaleShift(Rule):
    def __init__(self) -> None:
        super().__init__("R2-13", RuleDifficulty.MEDIUM, "易位尺寸转换", "按形状匹配的尺寸变化持续")

    def sample_params(self, rng) -> Dict:
        count = int(rng.integers(2, 6))
        return {"count": count}

    def generate_triplet(self, params, rng):
        count = int(params["count"])
        shapes = rng.choice(SHAPES, size=count, replace=False).tolist()
        objs = [random_object(rng, shape=s) for s in shapes]
        involved = list(range(count))

        factors = []
        for _ in range(count):
            roll = rng.random()
            if roll < 0.4:
                factors.append(float(rng.uniform(1.35, 1.75)))
            elif roll < 0.8:
                factors.append(float(rng.uniform(0.6, 0.8)))
            else:
                factors.append(1.0)
        if all(abs(f - 1.0) < 1e-6 for f in factors):
            idx = int(rng.integers(0, count))
            factors[idx] = float(rng.uniform(1.35, 1.75))

        def build_frame(step: int) -> List:
            arranged = clone_objects(objs)
            for i, obj in enumerate(arranged):
                scale_factor = factors[i] ** step
                if step > 0:
                    obj = apply_scale(obj, scale_factor)
                obj.p = rng.uniform(-0.8, 0.8, size=3)
                arranged[i] = obj
            return arranged

        a_objs = build_frame(0)
        b_objs = build_frame(1)
        c_objs = build_frame(2)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [[size(o) for o in s.objects] for s in scenes]
        meta = build_rule_meta(
            self,
            "R2",
            len(involved),
            involved,
            ["r"],
            ["size(Oi)"],
            "shape-conditioned",
            {"shapes": shapes, "factors": factors},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[List[Scene], List[str]]:
        if not scene_c.objects:
            return [], []
        params = meta.get("pattern_params", {})
        factors = params.get("factors")
        if factors is None or len(factors) != len(scene_c.objects):
            return [], []

        def apply_power(objs_in: List, power: float) -> List:
            out = clone_objects(objs_in)
            for i, obj in enumerate(out):
                factor = float(factors[i]) ** power
                out[i] = apply_scale(obj, factor)
            return out

        stay = apply_power(scene_c.objects, -1.0)
        smaller = apply_power(scene_c.objects, -0.5)
        reverse = apply_power(scene_c.objects, -2.0)
        distractors = [
            scene_from_objects(stay),
            scene_from_objects(smaller),
            scene_from_objects(reverse),
        ]
        reasons = [
            "尺寸未延续变化，停留在上一帧",
            "尺寸变化幅度减小",
            "尺寸变化方向反向",
        ]
        return distractors, reasons


@dataclass
class R2_14PositionDensityShift(Rule):
    def __init__(self) -> None:
        super().__init__("R2-14", RuleDifficulty.MEDIUM, "易位密度转换", "按形状匹配的密度变化持续")

    def sample_params(self, rng) -> Dict:
        count = int(rng.integers(2, 6))
        return {"count": count}

    def generate_triplet(self, params, rng):
        count = int(params["count"])
        shapes = rng.choice(SHAPES, size=count, replace=False).tolist()
        objs = [random_object(rng, shape=s) for s in shapes]
        involved = list(range(count))

        factors = []
        for _ in range(count):
            roll = rng.random()
            if roll < 0.4:
                factors.append(float(rng.uniform(1.7, 2.4)))
            elif roll < 0.8:
                factors.append(float(rng.uniform(0.4, 0.65)))
            else:
                factors.append(1.0)
        if all(abs(f - 1.0) < 1e-6 for f in factors):
            idx = int(rng.integers(0, count))
            factors[idx] = float(rng.uniform(1.7, 2.4))

        def build_frame(step: int) -> List:
            arranged = clone_objects(objs)
            for i, obj in enumerate(arranged):
                density_factor = factors[i] ** step
                if step > 0:
                    obj = apply_density(obj, density_factor)
                obj.p = rng.uniform(-0.8, 0.8, size=3)
                arranged[i] = obj
            return arranged

        a_objs = build_frame(0)
        b_objs = build_frame(1)
        c_objs = build_frame(2)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [[o.density for o in s.objects] for s in scenes]
        meta = build_rule_meta(
            self,
            "R2",
            len(involved),
            involved,
            ["d"],
            ["den(Oi)"],
            "shape-conditioned",
            {"shapes": shapes, "factors": factors},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[List[Scene], List[str]]:
        if not scene_c.objects:
            return [], []
        params = meta.get("pattern_params", {})
        factors = params.get("factors")
        if factors is None or len(factors) != len(scene_c.objects):
            return [], []

        def apply_power(objs_in: List, power: float) -> List:
            out = clone_objects(objs_in)
            for i, obj in enumerate(out):
                factor = float(factors[i]) ** power
                out[i] = apply_density(obj, factor)
            return out

        stay = apply_power(scene_c.objects, -1.0)
        smaller = apply_power(scene_c.objects, -0.5)
        reverse = apply_power(scene_c.objects, -2.0)
        distractors = [
            scene_from_objects(stay),
            scene_from_objects(smaller),
            scene_from_objects(reverse),
        ]
        reasons = [
            "密度未延续变化，停留在上一帧",
            "密度变化幅度减小",
            "密度变化方向反向",
        ]
        return distractors, reasons


@dataclass
class R2_4ContainRatioArithmetic(Rule):
    def __init__(self) -> None:
        super().__init__("R2-4", RuleDifficulty.MEDIUM, "包含比例等差", "内含比例按等差变化")

    def sample_params(self, rng) -> Dict:
        for _ in range(30):
            base_ratio = float(rng.uniform(0.2, 0.55))
            delta = float(rng.uniform(0.18, 0.28)) * (1 if rng.random() < 0.5 else -1)
            r2 = base_ratio + delta
            r3 = base_ratio + 2 * delta
            if 0.12 <= r2 <= 0.85 and 0.12 <= r3 <= 0.85:
                return {"base_ratio": base_ratio, "delta": delta}
        return {"base_ratio": 0.45, "delta": -0.2}

    def generate_triplet(self, params, rng):
        outer = init_objects(rng, 1, m=2)[0]
        inner = init_objects(rng, 1, m=2)[1]
        outer.r = np.maximum(outer.r, inner.r * 2.2)
        involved = [0, 1]
        base_ratio = float(params["base_ratio"])
        delta = float(params["delta"])
        ratios = [base_ratio, base_ratio + delta, base_ratio + 2 * delta]
        slack = outer.r / 2.0 - inner.r / 2.0
        axis_idx = int(np.argmax(slack))
        direction = 1 if rng.random() < 0.5 else -1

        def place(ratio: float):
            o0, o1 = outer.copy(), inner.copy()
            o0, o1 = self._place_ratio(o0, o1, ratio, axis_idx, direction)
            return [o0, o1]

        scenes_objs = [place(r) for r in ratios]
        scenes = [scene_from_objects(x) for x in scenes_objs]
        v = [self._contain_ratio(*objs_pair) for objs_pair in scenes_objs]
        meta = build_rule_meta(
            self,
            "R2",
            2,
            involved,
            ["p", "r"],
            ["contain_ratio(0,1)"],
            "arithmetic",
            {"delta": delta},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[List[Scene], List[str]]:
        if len(scene_c.objects) < 2:
            return [], []
        outer = scene_c.objects[0].copy()
        inner = scene_c.objects[1].copy()
        target_ratio = meta.get("v", {}).get("v3", [self._contain_ratio(outer, inner)])[0]

        def pick_wrong_ratio():
            delta = float(rng.uniform(0.15, 0.3)) * (1 if rng.random() < 0.5 else -1)
            ratio = float(np.clip(target_ratio + delta, 0.05, 0.95))
            if abs(ratio - target_ratio) < 0.08:
                ratio = float(np.clip(target_ratio - delta, 0.05, 0.95))
            return ratio

        def build_scene(ratio: float, scale_inner: bool = False, tweak_shape: bool = False, tweak_rot: bool = False):
            o0, o1 = outer.copy(), inner.copy()
            if scale_inner:
                o1 = apply_scale(o1, float(rng.uniform(0.7, 0.85)))
                o0.r = np.maximum(o0.r, o1.r * 1.6)
            if tweak_shape:
                shape_options = [s for s in SHAPES if s != o1.shape]
                o1.shape = str(rng.choice(shape_options))
            if tweak_rot:
                o1 = apply_rotation(o1, rng.uniform(-0.4, 0.4, size=3))
            slack = o0.r / 2.0 - o1.r / 2.0
            axis_idx = int(np.argmax(slack))
            direction = 1 if rng.random() < 0.5 else -1
            o0, o1 = self._place_ratio(o0, o1, ratio, axis_idx, direction)
            return scene_from_objects([o0, o1])

        def break_containment():
            o0, o1 = outer.copy(), inner.copy()
            slack = o0.r / 2.0 - o1.r / 2.0
            axis_idx = int(np.argmax(slack))
            direction = 1 if rng.random() < 0.5 else -1
            o1.p = o0.p.copy()
            o1.p[axis_idx] += direction * (slack[axis_idx] + o1.r[axis_idx] * 0.2)
            return scene_from_objects([o0, o1])

        distractors = [
            build_scene(pick_wrong_ratio(), scale_inner=False, tweak_shape=False, tweak_rot=False),
            build_scene(pick_wrong_ratio(), scale_inner=True, tweak_shape=False, tweak_rot=False),
            break_containment(),
        ]
        reasons = [
            "包含比例与等差规律不一致（位置偏移）",
            "缩放内物体导致包含比例偏离等差规律",
            "内物体部分露出，违反包含关系",
        ]
        return distractors, reasons

    @staticmethod
    def _contain_ratio(outer, inner) -> float:
        outer_half = outer.r / 2.0
        inner_half = inner.r / 2.0
        slack = outer_half - inner_half
        if np.any(slack <= 1e-6):
            return 0.0
        offset = inner.p - outer.p
        min_margin = np.minimum(slack - offset, slack + offset)
        ratio_axis = min_margin / slack
        ratio = float(np.min(ratio_axis))
        return max(0.0, min(1.0, ratio))

    @staticmethod
    def _place_ratio(outer, inner, ratio: float, axis_idx: int, direction: int):
        slack = outer.r / 2.0 - inner.r / 2.0
        offset = (1.0 - ratio) * slack[axis_idx]
        inner.p = outer.p.copy()
        inner.p[axis_idx] += direction * offset
        return outer, inner


@dataclass
class R2_5AngleArithmetic(Rule):
    def __init__(self) -> None:
        super().__init__("R2-5", RuleDifficulty.MEDIUM, "夹角等差", "轴夹角按等差变化")

    def sample_params(self, rng) -> Dict:
        base_angle = float(rng.uniform(math.pi / 12, math.pi / 6))
        delta = float(rng.uniform(math.pi / 12, math.pi / 6))
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
class R3_1AreaArithmetic(Rule):
    def __init__(self) -> None:
        super().__init__("R3-1", RuleDifficulty.MEDIUM, "三对象面积等差", "三点面积等差变化")

    def sample_params(self, rng) -> Dict:
        mode = rng.choice(["up", "down", "flat"])
        if mode == "flat":
            return {"delta_ratio": 0.0, "mode": mode}
        if mode == "up":
            delta_ratio = float(rng.uniform(0.35, 0.6))
        else:
            delta_ratio = -float(rng.uniform(0.35, 0.45))
        return {"delta_ratio": delta_ratio, "mode": mode}

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
        delta_ratio = float(params["delta_ratio"])
        delta = area1 * delta_ratio
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
            {"delta": delta, "mode": params.get("mode", "up" if delta_ratio > 0 else ("down" if delta_ratio < 0 else "flat"))},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if len(scene_c.objects) < 3:
            return [], []
        v = meta.get("v", {})
        v1 = v.get("v1")
        v2 = v.get("v2")
        if not (v1 and v2):
            return [], []
        area1 = float(v1[0])
        area2 = float(v2[0])
        delta = area2 - area1
        base_len = float(np.linalg.norm(scene_c.objects[1].p - scene_c.objects[0].p))
        if base_len <= 1e-6:
            return [], []

        def build_scene(area_target: float) -> Scene:
            objs = clone_objects(scene_c.objects)
            height = max(area_target * 2 / base_len, 0.04)
            new_p = objs[2].p.copy()
            sign = 1.0 if new_p[1] >= 0 else -1.0
            new_p[1] = height * sign
            objs[2].p = new_p
            return scene_from_objects(objs)

        min_area = max(area1 * 0.12, 0.02)

        if abs(delta) < 1e-6:
            wrong_areas = [area1 * 1.6, area1 * 0.5, area1 * 2.1]
            reasons = ["面积被放大", "面积被缩小", "面积放大过多"]
        else:
            step = abs(delta)
            wrong_areas = [
                area1,
                area1 + (2 * step if delta < 0 else -2 * step),
                area1 + (3 * step if delta > 0 else -3 * step),
            ]
            reasons = ["面积未变化", "面积变化方向相反", "面积步长过大"]

        wrong_areas = [max(min_area, float(a)) for a in wrong_areas]
        distractors = [build_scene(a) for a in wrong_areas]
        return distractors, reasons

    @staticmethod
    def _area(objs: Sequence) -> float:
        return float(0.5 * np.linalg.norm(np.cross(objs[1].p - objs[0].p, objs[2].p - objs[0].p)))


@dataclass
class R2_6DistanceDifferenceConserved(Rule):
    def __init__(self) -> None:
        super().__init__("R2-6", RuleDifficulty.MEDIUM, "距离差分守恒", "两对距离差保持常数")

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
class R3_2OrderingCycle(Rule):
    def __init__(self) -> None:
        super().__init__("R3-2", RuleDifficulty.MEDIUM, "排序模式循环", "沿 x 轴顺序循环")

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

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[List[Scene], List[str]]:
        base_positions = [-0.8, 0.0, 0.8]
        all_perms = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ]
        correct = order_indices_x(scene_c.objects)
        wrong_perms = [p for p in all_perms if p != correct]
        rng.shuffle(wrong_perms)
        chosen = wrong_perms[:3]
        distractors: List[Scene] = []
        reasons: List[str] = []
        for perm in chosen:
            arranged = clone_objects(scene_c.objects)
            for rank, obj_idx in enumerate(perm):
                arranged[obj_idx].p[0] = base_positions[rank]
            distractors.append(scene_from_objects(arranged))
            reasons.append("x 轴排序未按循环置换")
        return distractors, reasons


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
class R3_3DistanceVectorGeometric(Rule):
    def __init__(self) -> None:
        super().__init__("R3-3", RuleDifficulty.MEDIUM, "距离集合等比缩放", "左右独立等比缩放")

    def sample_params(self, rng) -> Dict:
        mode_left = rng.choice(["up", "down", "flat"], p=[0.45, 0.45, 0.10])
        mode_right = rng.choice(["up", "down", "flat"], p=[0.45, 0.45, 0.10])

        def pick_k(mode: str) -> float:
            if mode == "flat":
                return 1.0
            if mode == "up":
                return float(rng.uniform(1.45, 1.9))
            return float(rng.uniform(0.55, 0.8))

        return {
            "k_left": pick_k(mode_left),
            "k_right": pick_k(mode_right),
            "mode_left": mode_left,
            "mode_right": mode_right,
        }

    def generate_triplet(self, params, rng):
        k_left = float(params["k_left"])
        k_right = float(params["k_right"])
        objs = init_objects(rng, 3, m=3)
        involved = [0, 1, 2]
        base_offset = float(rng.uniform(0.6, 0.9))
        y_offsets = rng.uniform(-0.25, 0.25, size=3)
        z_offsets = rng.uniform(-0.2, 0.2, size=3)
        objs[0].p = np.array([-base_offset, y_offsets[0], z_offsets[0]])
        objs[1].p = np.array([0.0, y_offsets[1], z_offsets[1]])
        objs[2].p = np.array([base_offset, y_offsets[2], z_offsets[2]])
        base_cent = centroid(objs)

        def scale_positions(objs_in: Sequence, left_factor: float, right_factor: float):
            scaled = []
            for o in objs_in:
                new_o = o.copy()
                delta = o.p - base_cent
                if delta[0] < -1e-6:
                    factor = left_factor
                elif delta[0] > 1e-6:
                    factor = right_factor
                else:
                    factor = 1.0
                new_o.p = base_cent + delta * factor
                scaled.append(new_o)
            return scaled

        a_objs = clone_objects(objs)
        b_objs = scale_positions(objs, k_left, k_right)
        c_objs = scale_positions(b_objs, k_left, k_right)
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
            "piecewise",
            {
                "k_left": k_left,
                "k_right": k_right,
                "mode_left": params.get("mode_left"),
                "mode_right": params.get("mode_right"),
            },
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
class R1_9DualSizeConservation(Rule):
    def __init__(self) -> None:
        super().__init__("R1-9", RuleDifficulty.MEDIUM, "双对象属性联动", "size 和保持守恒")

    def sample_params(self, rng) -> Dict:
        delta_ratio = float(rng.uniform(0.4, 0.8))
        sign = 1 if rng.random() < 0.5 else -1
        return {"delta_ratio": delta_ratio * sign}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 2, m=2)
        involved = [0, 1]
        s0, s1 = size(objs[0]), size(objs[1])
        total = s0 + s1
        min_ratio = 0.2
        if min(s0, s1) / total < min_ratio:
            if s0 < s1:
                target_small = (min_ratio / (1.0 - min_ratio)) * s1
                f = (target_small / s0) ** (1 / 3)
                objs[0] = apply_scale(objs[0], f)
            else:
                target_small = (min_ratio / (1.0 - min_ratio)) * s0
                f = (target_small / s1) ** (1 / 3)
                objs[1] = apply_scale(objs[1], f)
            s0, s1 = size(objs[0]), size(objs[1])
            total = s0 + s1

        delta = s0 * params["delta_ratio"]
        min_size = total * min_ratio
        max_pos = (s1 - min_size) / 2.0
        max_neg = (min_size - s0) / 2.0  # negative or zero
        if delta >= 0:
            delta = min(delta, max_pos)
        else:
            delta = max(delta, max_neg)
        min_delta_mag = total * 0.08
        if abs(delta) < min_delta_mag:
            if max_pos >= abs(max_neg):
                delta = max_pos
            else:
                delta = max_neg

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

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if len(scene_c.objects) < 2:
            return [], []
        v = meta.get("v", {})
        v1 = v.get("v1")
        v2 = v.get("v2")
        v3 = v.get("v3")
        if not (v1 and v2 and v3):
            return [], []
        sizes_a = [float(v1[0]), float(v1[1])]
        sizes_b = [float(v2[0]), float(v2[1])]
        sizes_c = [float(v3[0]), float(v3[1])]
        total = float(sum(sizes_a))
        min_size = total * 0.2

        def apply_sizes(target_sizes: List[float]) -> Scene:
            objs = clone_objects(scene_c.objects)
            for i in range(2):
                cur = size(objs[i])
                target = max(float(target_sizes[i]), min_size)
                if cur > 1e-6:
                    f = (target / cur) ** (1 / 3)
                    objs[i] = apply_scale(objs[i], f)
            return scene_from_objects(objs)

        swap_sizes = [sizes_c[1], sizes_c[0]]
        if abs(swap_sizes[0] - swap_sizes[1]) < total * 0.15:
            swap_sizes = [sizes_c[0] * 1.3, sizes_c[1] * 0.7]
        both_up = [sizes_c[0] * 1.35, sizes_c[1] * 1.35]

        distractors = [
            apply_sizes(sizes_b),
            apply_sizes(swap_sizes),
            apply_sizes(both_up),
        ]
        reasons = [
            "未按联动延续（停留在前一步）",
            "双对象比例关系错误",
            "双对象同时变大，守恒被破坏",
        ]
        return distractors, reasons


@dataclass
class R1_11AttributeSwap(Rule):
    def __init__(self) -> None:
        super().__init__("R1-11", RuleDifficulty.MEDIUM, "尺度/姿态交替互换", "尺寸与姿态交替互换（形状固定）")

    def sample_params(self, rng) -> Dict:
        first_attr = "r" if rng.random() < 0.5 else "R"
        return {"first_attr": first_attr}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 2, m=2)
        involved = [0, 1]
        base_offset = float(rng.uniform(0.35, 0.6))
        base_y = float(rng.uniform(-0.2, 0.2))
        base_z = float(rng.uniform(-0.2, 0.2))
        objs[0].p = np.array([-base_offset, base_y, base_z])
        objs[1].p = np.array([base_offset, -base_y, -base_z])
        allowed_shapes = [s for s in SHAPES if s not in ("sphere", "ellipsoid")]
        if not allowed_shapes:
            allowed_shapes = [s for s in SHAPES if s != "sphere"]
        shape = str(rng.choice(allowed_shapes))
        objs[0].shape = shape
        objs[1].shape = shape
        size_a = size(objs[0])
        size_b = size(objs[1])
        min_ratio = 2.2
        ratio = max(size_a, size_b) / max(min(size_a, size_b), 1e-6)
        if ratio < min_ratio:
            target_ratio = float(rng.uniform(2.6, 3.8))
            if size_a >= size_b:
                factor = (size_a * target_ratio / size_b) ** (1 / 3)
                objs[1] = apply_scale(objs[1], factor)
            else:
                factor = (size_b * target_ratio / size_a) ** (1 / 3)
                objs[0] = apply_scale(objs[0], factor)
        min_rot_diff = 0.9
        if float(np.linalg.norm(objs[0].rotation - objs[1].rotation)) < min_rot_diff:
            delta_rot = rng.uniform(math.pi / 3, math.pi / 2, size=3)
            delta_rot = delta_rot * rng.choice([-1, 1], size=3)
            objs[1] = apply_rotation(objs[1], delta_rot)

        first_attr = params.get("first_attr", "r")
        second_attr = "R" if first_attr == "r" else "r"

        def swap_attr(src, attr: str):
            swapped = clone_objects(src)
            if attr == "r":
                swapped[0].r, swapped[1].r = swapped[1].r.copy(), swapped[0].r.copy()
            elif attr == "R":
                swapped[0].rotation, swapped[1].rotation = swapped[1].rotation.copy(), swapped[0].rotation.copy()
            else:
                raise ValueError(f"Unsupported attr '{attr}'")
            return swapped

        a_objs = clone_objects(objs)
        b_objs = swap_attr(a_objs, first_attr)
        c_objs = swap_attr(b_objs, second_attr)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]

        def snapshot(src):
            return [
                size(src[0]),
                size(src[1]),
                float(np.linalg.norm(src[0].rotation)),
                float(np.linalg.norm(src[1].rotation)),
            ]

        v = [
            snapshot(a_objs),
            snapshot(b_objs),
            snapshot(c_objs),
        ]
        meta = build_rule_meta(
            self,
            "R1",
            2,
            involved,
            ["r", "R"],
            ["size(Oi)", "axis(Oi)"],
            "swap",
            {"first_attr": first_attr, "second_attr": second_attr},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if len(scene_c.objects) < 2:
            return [], []
        first_attr = meta.get("pattern_params", {}).get("first_attr", "r")
        second_attr = "R" if first_attr == "r" else "r"

        def swap_attr(src, attr: str):
            swapped = clone_objects(src)
            if attr == "r":
                swapped[0].r, swapped[1].r = swapped[1].r.copy(), swapped[0].r.copy()
            elif attr == "R":
                swapped[0].rotation, swapped[1].rotation = swapped[1].rotation.copy(), swapped[0].rotation.copy()
            else:
                raise ValueError(f"Unsupported attr '{attr}'")
            return swapped

        objs = clone_objects(scene_c.objects)
        only_first = swap_attr(objs, second_attr)
        only_second = swap_attr(objs, first_attr)
        none = swap_attr(only_first, first_attr)

        distractors = [
            scene_from_objects(only_first),
            scene_from_objects(only_second),
            scene_from_objects(none),
        ]
        reasons = ["仅交换第一步属性", "仅交换第二步属性", "两次交换都被撤销"]
        return distractors, reasons


@dataclass
class R1_12OrientationFollowMotion(Rule):
    def __init__(self) -> None:
        super().__init__("R1-12", RuleDifficulty.MEDIUM, "主轴对齐位移", "移动方向与主轴一致")

    def sample_params(self, rng) -> Dict:
        step = float(rng.uniform(0.25, 0.4))
        return {"step": step}

    def generate_triplet(self, params, rng):
        obj = init_objects(rng, 1, m=2)[0]
        involved = [0]
        r = obj.r.copy()
        if r.max() / (r.min() + 1e-6) < 1.2:
            r[0] *= 1.45
            obj = obj.copy()
            obj.r = r
        move_dir = axis(obj, 0)
        step = params["step"]
        a_objs = [obj.copy()]
        b_objs = [apply_translation(a_objs[0], move_dir * step)]
        c_objs = [apply_translation(b_objs[0], move_dir * step)]
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [move_dir.tolist(), move_dir.tolist(), move_dir.tolist()]
        meta = build_rule_meta(
            self,
            "R1",
            1,
            involved,
            ["p", "R"],
            ["axis(O0)", "dir"],
            "coupled",
            {"step": step},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if not scene_c.objects:
            return [], []
        step = float(meta.get("pattern_params", {}).get("step", 0.3))
        base = scene_c.objects[0]
        axis0 = axis(base, 0)
        axis1 = axis(base, 1)

        wrong_axis = [apply_translation(base, axis1 * step)]
        back_step = [apply_translation(base, -axis0 * step)]
        wrong_rot = [apply_rotation(base, rng.uniform(0.4, 0.7, size=3))]

        distractors = [
            scene_from_objects(wrong_axis),
            scene_from_objects(back_step),
            scene_from_objects(wrong_rot),
        ]
        reasons = ["位移方向偏离主轴", "位移方向反向", "姿态被错误旋转"]
        return distractors, reasons


@dataclass
class R1_13DensitySizeCoupled(Rule):
    def __init__(self) -> None:
        super().__init__("R1-13", RuleDifficulty.MEDIUM, "密度驱动缩放", "密度越大缩放越大，密度越小缩放越小")

    def sample_params(self, rng) -> Dict:
        scale_up = float(rng.uniform(3.0, 4.2))
        scale_down = float(rng.uniform(0.55, 0.75))
        factors = [scale_up, scale_down] if rng.random() < 0.5 else [scale_down, scale_up]
        return {"density_factors": factors}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 2, m=2)
        involved = [0, 1]
        base_offset = float(rng.uniform(0.35, 0.6))
        base_y = float(rng.uniform(-0.15, 0.15))
        base_z = float(rng.uniform(-0.15, 0.15))
        objs[0].p = np.array([-base_offset, base_y, base_z])
        objs[1].p = np.array([base_offset, -base_y, -base_z])

        factors = [float(f) for f in params.get("density_factors", [3.2, 0.6])]
        if len(factors) != 2:
            factors = [3.2, 0.6]
        base_low = float(rng.uniform(0.7, 0.9))
        base_high = float(rng.uniform(1.4, 1.7))
        if factors[0] >= factors[1]:
            objs[0].density = base_high
            objs[1].density = base_low
        else:
            objs[0].density = base_low
            objs[1].density = base_high
        scale_factors = [f ** (1.0 / 3.0) for f in factors]

        def step_objs(src):
            stepped = []
            for obj, den_factor, sc_factor in zip(src, factors, scale_factors):
                updated = apply_density(obj, den_factor)
                updated = apply_scale(updated, sc_factor)
                stepped.append(updated)
            return stepped

        a_objs = clone_objects(objs)
        b_objs = step_objs(a_objs)
        c_objs = step_objs(b_objs)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]

        def den_size(objects):
            return [float(objects[0].density), size(objects[0]), float(objects[1].density), size(objects[1])]

        v = [den_size(s.objects) for s in scenes]
        meta = build_rule_meta(
            self,
            "R1",
            2,
            involved,
            ["d", "r"],
            ["den(Oi)", "size(Oi)"],
            "coupled",
            {"density_factors": factors},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if len(scene_c.objects) < 2:
            return [], []
        objs = clone_objects(scene_c.objects)
        params = meta.get("pattern_params", {})
        factors = params.get("density_factors")
        if factors is None or len(factors) != 2:
            factors = [1.3, 0.75]
        scale_factors = [float(f) ** (1.0 / 3.0) for f in factors]

        wrong_scale = clone_objects(objs)
        for i, f in enumerate(scale_factors):
            wrong_scale[i] = apply_scale(wrong_scale[i], 1.0 / f)

        wrong_density = clone_objects(objs)
        idx = int(rng.integers(0, len(wrong_density)))
        wrong_density[idx] = apply_density(wrong_density[idx], 1.0 / float(factors[idx]))

        swap_sizes = clone_objects(objs)
        swap_sizes[0].r, swap_sizes[1].r = swap_sizes[1].r.copy(), swap_sizes[0].r.copy()

        distractors = [
            scene_from_objects(wrong_scale),
            scene_from_objects(wrong_density),
            scene_from_objects(swap_sizes),
        ]
        reasons = ["尺寸未随密度延续变化", "密度变化方向错误", "尺寸与密度对应关系被打乱"]
        return distractors, reasons


@dataclass
class R1_14MirrorDensityComplement(Rule):
    def __init__(self) -> None:
        super().__init__("R1-14", RuleDifficulty.MEDIUM, "镜像密度缩放", "镜像位置下密度一增一减")

    def sample_params(self, rng) -> Dict:
        scale_up = float(rng.uniform(1.8, 2.6))
        scale_down = float(rng.uniform(0.35, 0.6))
        if rng.random() < 0.5:
            factors = [scale_up, scale_down]
        else:
            factors = [scale_down, scale_up]
        return {"density_factors": factors}

    def generate_triplet(self, params, rng):
        objs = init_objects(rng, 2, m=2)
        involved = [0, 1]
        base_offset = float(rng.uniform(0.35, 0.55))
        base_y = float(rng.uniform(-0.2, 0.2))
        base_z = float(rng.uniform(-0.2, 0.2))
        shape = str(rng.choice(SHAPES))
        rot = rng.uniform(-0.6, 0.6, size=3)

        left = objs[0].copy()
        right = objs[1].copy()
        left.shape = shape
        right.shape = shape
        left.p = np.array([-base_offset, base_y, base_z])
        right.p = np.array([base_offset, base_y, base_z])
        left.rotation = rot
        right.rotation = rot * np.array([1.0, -1.0, -1.0])

        factors = [float(f) for f in params.get("density_factors", [2.0, 0.5])]
        if len(factors) != 2:
            factors = [2.0, 0.5]

        def step_pair(src):
            out = []
            for obj, factor in zip(src, factors):
                out.append(apply_density(obj, factor))
            return out

        a_objs = [left, right]
        b_objs = step_pair(a_objs)
        c_objs = step_pair(b_objs)
        scenes = [scene_from_objects(x) for x in [a_objs, b_objs, c_objs]]
        v = [[float(o0.density), float(o1.density)] for o0, o1 in [a_objs, b_objs, c_objs]]
        meta = build_rule_meta(
            self,
            "R1",
            2,
            involved,
            ["p", "d", "R"],
            ["mirror(x)", "den(Oi)"],
            "coupled",
            {"density_factors": factors},
            v,
            scenes,
        )
        return scenes[0], scenes[1], scenes[2], meta

    def make_distractors(self, scene_c: Scene, rng, meta: Dict) -> Tuple[list[Scene], list[str]]:
        if len(scene_c.objects) < 2:
            return [], []
        objs = clone_objects(scene_c.objects)
        params = meta.get("pattern_params", {})
        factors = params.get("density_factors")
        if factors is None or len(factors) != 2:
            factors = [2.0, 0.5]

        stay = clone_objects(objs)
        for i, factor in enumerate(factors):
            stay[i] = apply_density(stay[i], 1.0 / float(factor))

        reverse = clone_objects(objs)
        for i, factor in enumerate(factors):
            reverse[i] = apply_density(reverse[i], 1.0 / float(factor) ** 2)

        break_mirror = clone_objects(objs)
        break_mirror[1] = apply_translation(break_mirror[1], np.array([0.12, 0.0, 0.0]))

        distractors = [
            scene_from_objects(stay),
            scene_from_objects(reverse),
            scene_from_objects(break_mirror),
        ]
        reasons = ["密度未延续变化，停留在上一帧", "密度变化方向反向", "镜像关系被破坏"]
        return distractors, reasons


def build_medium_rules() -> List[Rule]:
    return [
        R2_1DistanceGeometric(),
        R2_3DirectionRotate(),
        R2_9AcceleratedRotation(),
        R2_10GeometricFusion(),
        R2_11OrbitalRotation(),
        R2_12PoseShift(),
        R2_13PositionScaleShift(),
        R2_14PositionDensityShift(),
        R2_4ContainRatioArithmetic(),
        R2_5AngleArithmetic(),
        R3_1AreaArithmetic(),
        R2_6DistanceDifferenceConserved(),
        R3_2OrderingCycle(),
        R3_3DistanceVectorGeometric(),
        R1_9DualSizeConservation(),
        R1_11AttributeSwap(),
        R1_12OrientationFollowMotion(),
        R1_13DensitySizeCoupled(),
        R1_14MirrorDensityComplement(),
    ]
