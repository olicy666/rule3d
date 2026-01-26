from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from .io import write_meta, write_ply, ensure_dir
from .registry import RuleRegistry
from .rules.base import RuleDifficulty
from .scene import Scene
from .rules.utils import apply_rotation, apply_scale, apply_translation, random_object


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
        self._all_rules = [rule for rules in self._rule_pools.values() for rule in rules]

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
        idx = self.rng.integers(0, len(self._all_rules))
        return self._all_rules[int(idx)]

    def _perturb_scene(self, scene: Scene, meta: Dict) -> Tuple[Scene, str]:
        """Create a distractor by altering one participating object while keeping others intact."""
        new_scene = scene.copy()
        if not new_scene.objects:
            return new_scene, "场景为空"
        involved = meta.get("involved_indices") or list(range(len(new_scene.objects)))
        base_attrs = meta.get("base_attrs_used", [])
        derived_funcs = meta.get("derived_funcs", [])
        pattern = meta.get("pattern_type", "模式变换")
        attr_map = {"r": "尺度", "p": "位置", "R": "姿态", "d": "密度", "s": "形状"}
        attr_desc = "、".join(attr_map.get(a, a) for a in base_attrs) if base_attrs else "属性"
        derived_desc = "、".join(derived_funcs) if derived_funcs else attr_desc
        idx = int(self.rng.choice(involved))
        obj = new_scene.objects[idx]
        reason = "扰动"
        if "r" in base_attrs:
            factor = float(self.rng.uniform(0.65, 0.85) if self.rng.random() < 0.5 else self.rng.uniform(1.15, 1.35))
            new_scene.objects[idx] = apply_scale(obj, factor)
            reason = f"修改{attr_desc}，派生 {derived_desc} 不再满足 {pattern} 模式变换（尺度被缩放）"
        elif "p" in base_attrs:
            delta = self.rng.uniform(0.15, 0.35, size=3) * self.rng.choice([-1, 1], size=3)
            new_scene.objects[idx] = apply_translation(obj, delta)
            reason = f"平移参与对象，派生 {derived_desc} 不再符合 {pattern} 模式变换"
        elif "R" in base_attrs:
            delta = self.rng.uniform(0.2, 0.5, size=3)
            new_scene.objects[idx] = apply_rotation(obj, delta)
            reason = f"改变姿态，派生 {derived_desc} 与 {pattern} 模式变换失配"
        elif "d" in base_attrs:
            factor = float(self.rng.uniform(0.6, 0.85) if self.rng.random() < 0.5 else self.rng.uniform(1.2, 1.4))
            new_scene.objects[idx] = obj.copy()
            new_scene.objects[idx].density = max(obj.density * factor, 1e-3)
            reason = f"调整密度，导致 {derived_desc} 破坏 {pattern} 模式变换"
        elif "s" in base_attrs:
            alt_shape = random_object(self.rng).shape
            new_scene.objects[idx] = obj.copy()
            new_scene.objects[idx].shape = alt_shape
            reason = f"替换形状，派生 {derived_desc} 不再满足 {pattern} 模式变换"
        else:
            # fallback: small translation
            delta = self.rng.uniform(0.15, 0.3, size=3) * self.rng.choice([-1, 1], size=3)
            new_scene.objects[idx] = apply_translation(obj, delta)
            reason = f"引入位移扰动，破坏 {pattern} 模式变换"
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
        color_map = {
            "1.ply": (31, 119, 180),  # deep blue
            "2.ply": (255, 127, 14),  # vivid orange
            "3.ply": (44, 160, 44),  # forest green
            "4.ply": (214, 39, 40),  # brick red
            "5.ply": (148, 103, 189),  # muted purple
            "6.ply": (140, 86, 75),  # cocoa brown
        }
        if correct_idx is None:
            correct_idx = int(self.rng.integers(0, len(option_labels)))
        candidate_points = [None] * len(option_labels)
        candidate_reasons = [""] * len(option_labels)
        candidate_points[correct_idx] = correct_pts
        candidate_reasons[correct_idx] = "符合原规则的正确延续"

        distractor_scenes, distractor_reasons = rule.make_distractors(scene_c, self.rng, meta_params)
        if len(distractor_scenes) != 3:
            distractor_scenes = []
            distractor_reasons = []
            for _ in range(len(option_labels) - 1):
                sc, rs = self._perturb_scene(scene_c, meta_params)
                distractor_scenes.append(sc)
                distractor_reasons.append(rs)

        empty_indices = [i for i, pts in enumerate(candidate_points) if pts is None]
        self.rng.shuffle(empty_indices)
        for d_idx, slot_idx in enumerate(empty_indices):
            distractor_scene = distractor_scenes[d_idx]
            reason = distractor_reasons[d_idx] if d_idx < len(distractor_reasons) else "扰动候选"
            candidate_points[slot_idx] = distractor_scene.sample_point_cloud(self.config.n_points)
            candidate_reasons[slot_idx] = reason

        sample_dir = output_root / f"sample_{sample_index:06d}"
        ensure_dir(sample_dir)
        write_ply(sample_dir / "1.ply", pts_a, color=color_map["1.ply"])
        write_ply(sample_dir / "2.ply", pts_b, color=color_map["2.ply"])

        for fname, pts in zip(candidate_files, candidate_points):
            write_ply(sample_dir / fname, pts, color=color_map.get(fname))

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
            "rule_meta": meta_params,
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
