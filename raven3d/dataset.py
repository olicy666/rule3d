from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from .io import write_meta, write_ply, ensure_dir
from .geometry import Sphere, Cube, Cylinder, Cone, Primitive
from .registry import RuleRegistry
from .rules.base import RuleDifficulty
from .scene import Scene


@dataclass
class GenerationConfig:
    n_points: int = 4096
    difficulty_probs: Dict[RuleDifficulty, float] = field(
        default_factory=lambda: {
            RuleDifficulty.SIMPLE: 0.7,
            RuleDifficulty.MEDIUM: 0.2,
            RuleDifficulty.COMPLEX: 0.1,
        }
    )
    rule_filter: Set[str] | None = None


class DatasetGenerator:
    def __init__(self, registry: RuleRegistry, config: GenerationConfig | None = None, seed: int | None = None) -> None:
        self.registry = registry
        self.config = config or GenerationConfig()
        self.rule_filter = set(self.config.rule_filter) if self.config.rule_filter else None
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            np.random.seed(seed)
        self._rule_pools = self._build_rule_pools()

    def _build_rule_pools(self) -> Dict[RuleDifficulty, List]:
        pools: Dict[RuleDifficulty, List] = {}
        for difficulty in RuleDifficulty:
            rules = self.registry.rules_by_difficulty(difficulty)
            if self.rule_filter is not None:
                rules = [r for r in rules if r.rule_id in self.rule_filter]
            pools[difficulty] = rules
        if sum(len(v) for v in pools.values()) == 0:
            raise ValueError("No rules available after applying the current mode filter.")
        return pools

    def _sample_difficulty(self) -> RuleDifficulty:
        difficulties = [d for d in RuleDifficulty if self._rule_pools.get(d)]
        probs = np.array([self.config.difficulty_probs.get(d, 0.0) for d in difficulties], dtype=float)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(len(difficulties)) / len(difficulties)
        idx = self.rng.choice(len(difficulties), p=probs)
        return difficulties[int(idx)]

    def _sample_rule(self):
        difficulty = self._sample_difficulty()
        rules = self._rule_pools[difficulty]
        idx = self.rng.integers(0, len(rules))
        return rules[int(idx)]

    def _swap_shape_preserve_transform(self, prim: Primitive) -> Primitive:
        """Replace primitive type but keep center/rotation/scale."""
        center = prim.center.copy()
        rotation = prim.rotation.copy()
        scale = prim.scale.copy()
        shapes = [Sphere, Cube, Cylinder, Cone]
        # Choose a different shape class.
        choices = [cls for cls in shapes if not isinstance(prim, cls)]
        cls = choices[int(self.rng.integers(0, len(choices)))]
        if cls is Sphere:
            new_prim = Sphere(center=center, rotation=rotation, scale=scale, radius=float(self.rng.uniform(0.3, 0.6)))
        elif cls is Cube:
            edges = self.rng.uniform(0.4, 0.9, size=3)
            new_prim = Cube(center=center, rotation=rotation, scale=scale, edge_lengths=edges)
        elif cls is Cylinder:
            new_prim = Cylinder(
                center=center,
                rotation=rotation,
                scale=scale,
                radius=float(self.rng.uniform(0.25, 0.55)),
                height=float(self.rng.uniform(0.6, 1.1)),
            )
        else:  # Cone
            new_prim = Cone(
                center=center,
                rotation=rotation,
                scale=scale,
                radius=float(self.rng.uniform(0.25, 0.55)),
                height=float(self.rng.uniform(0.7, 1.1)),
            )
        return new_prim

    def _perturb_scene(self, scene: Scene) -> Tuple[Scene, str]:
        """Create a distractor by altering one property of the correct scene."""
        new_scene = scene.copy()
        if not new_scene.primitives:
            return new_scene, "场景为空"
        prim_idx = int(self.rng.integers(0, len(new_scene.primitives)))
        prim = new_scene.primitives[prim_idx]
        perturb_types = ["scale", "translation", "rotation", "shape"]
        kind = str(self.rng.choice(perturb_types))
        if kind == "scale":
            factor = float(self.rng.uniform(0.6, 0.9) if self.rng.random() < 0.5 else self.rng.uniform(1.1, 1.4))
            prim.scale = prim.scale * factor
            reason = "尺度比例被打破，不再符合原规则的大小递进"
        elif kind == "translation":
            delta = self.rng.uniform(0.2, 0.5, size=3) * self.rng.choice([-1, 1], size=3)
            prim.center = prim.center + delta
            reason = "位置偏移破坏了原规则的平移或对齐关系"
        elif kind == "rotation":
            delta = self.rng.uniform(0.2, 0.5, size=3)
            prim.rotation = prim.rotation + delta
            reason = "姿态变化打破了原规则的旋转模式"
        else:  # shape
            new_scene.primitives[prim_idx] = self._swap_shape_preserve_transform(prim)
            reason = "形状被替换，违背了原规则的形状/拓扑约束"
        return new_scene, reason

    def generate_sample(self, output_root: Path, sample_index: int, correct_idx: int | None = None) -> Dict:
        output_root = Path(output_root)
        # Primary rule to produce the reference A/B and the correct continuation.
        rule = self._sample_rule()
        params = rule.sample_params(self.rng)
        scene_a, scene_b, scene_c, meta_params = rule.generate_triplet(params, self.rng)

        pts_a = scene_a.sample_point_cloud(self.config.n_points)
        pts_b = scene_b.sample_point_cloud(self.config.n_points)
        correct_pts = scene_c.sample_point_cloud(self.config.n_points)

        # Option labels correspond to output filenames 3.ply ~ 6.ply
        option_labels = ["A", "B", "C", "D"]
        candidate_files = ["3.ply", "4.ply", "5.ply", "6.ply"]
        if correct_idx is None:
            correct_idx = int(self.rng.integers(0, len(option_labels)))
        candidate_points = [None] * len(option_labels)
        candidate_reasons = [""] * len(option_labels)
        candidate_points[correct_idx] = correct_pts
        candidate_reasons[correct_idx] = "符合原规则的正确延续"

        for i in range(len(candidate_points)):
            if candidate_points[i] is not None:
                continue
            distractor_scene, reason = self._perturb_scene(scene_c)
            candidate_points[i] = distractor_scene.sample_point_cloud(self.config.n_points)
            candidate_reasons[i] = reason

        sample_dir = output_root / f"sample_{sample_index:06d}"
        ensure_dir(sample_dir)
        write_ply(sample_dir / "1.ply", pts_a)
        write_ply(sample_dir / "2.ply", pts_b)

        for fname, pts in zip(candidate_files, candidate_points):
            write_ply(sample_dir / fname, pts)

        def rel(path: Path) -> str:
            try:
                return path.relative_to(output_root).as_posix()
            except ValueError:
                return path.as_posix()

        entry = {
            "id": f"sample-{sample_index + 1:02d}",
            "ref1_path": rel(sample_dir / "1.ply"),
            "ref2_path": rel(sample_dir / "2.ply"),
            "cand1_path": rel(sample_dir / "3.ply"),
            "cand2_path": rel(sample_dir / "4.ply"),
            "cand3_path": rel(sample_dir / "5.ply"),
            "cand4_path": rel(sample_dir / "6.ply"),
            "gt_option": option_labels[correct_idx],  # A/B/C/D 对应 3/4/5/6.ply
            "point_count": self.config.n_points,
            "cand1_reason": candidate_reasons[0],
            "cand2_reason": candidate_reasons[1],
            "cand3_reason": candidate_reasons[2],
            "cand4_reason": candidate_reasons[3],
            "rule_id": rule.rule_id,
        }
        write_meta(sample_dir / "meta.json", entry)
        return entry

    def generate_dataset(self, output_root: str | Path, num_samples: int) -> None:
        output_root = Path(output_root)
        ensure_dir(output_root)
        all_entries = []
        option_count = 4
        base = num_samples // option_count
        remainder = num_samples % option_count
        correct_indices = []
        for i in range(option_count):
            correct_indices.extend([i] * (base + (1 if i < remainder else 0)))
        self.rng.shuffle(correct_indices)

        for idx in range(num_samples):
            correct_idx = int(correct_indices[idx]) if correct_indices else None
            entry = self.generate_sample(output_root, idx, correct_idx=correct_idx)
            all_entries.append(entry)
        # Dataset-level manifest that follows the requested list-of-question format.
        write_meta(output_root / "meta.json", all_entries)
