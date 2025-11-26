from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .io import write_meta, write_ply, ensure_dir
from .registry import RuleRegistry
from .rules.base import RuleDifficulty


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


class DatasetGenerator:
    def __init__(self, registry: RuleRegistry, config: GenerationConfig | None = None, seed: int | None = None) -> None:
        self.registry = registry
        self.config = config or GenerationConfig()
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            np.random.seed(seed)

    def _sample_difficulty(self) -> RuleDifficulty:
        difficulties = list(self.config.difficulty_probs.keys())
        probs = np.array(list(self.config.difficulty_probs.values()), dtype=float)
        probs = probs / probs.sum()
        idx = self.rng.choice(len(difficulties), p=probs)
        return difficulties[int(idx)]

    def generate_sample(self, output_root: str | Path, sample_index: int) -> Tuple[str, Dict]:
        difficulty = self._sample_difficulty()
        rule = self.registry.sample_rule(difficulty, self.rng)
        params = rule.sample_params(self.rng)
        scene_a, scene_b, scene_c, meta_params = rule.generate_triplet(params, self.rng)

        pts_a = scene_a.sample_point_cloud(self.config.n_points)
        pts_b = scene_b.sample_point_cloud(self.config.n_points)
        pts_c = scene_c.sample_point_cloud(self.config.n_points)

        sample_dir = Path(output_root) / f"sample_{sample_index:06d}"
        ensure_dir(sample_dir)
        write_ply(sample_dir / "A.ply", pts_a)
        write_ply(sample_dir / "B.ply", pts_b)
        write_ply(sample_dir / "C.ply", pts_c)

        meta = {
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "difficulty": rule.difficulty.value,
            "description": rule.description,
            "params": meta_params,
            "point_count": self.config.n_points,
        }
        write_meta(sample_dir / "meta.json", meta)
        return str(sample_dir), meta

    def generate_dataset(self, output_root: str | Path, num_samples: int) -> None:
        output_root = Path(output_root)
        ensure_dir(output_root)
        for idx in range(num_samples):
            self.generate_sample(output_root, idx)
