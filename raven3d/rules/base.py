from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from ..scene import Scene


class RuleDifficulty(str, enum.Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class Rule:
    rule_id: str
    difficulty: RuleDifficulty
    name: str
    description: str

    def sample_params(self, rng) -> Dict[str, Any]:
        raise NotImplementedError

    def generate_triplet(self, params: Dict[str, Any], rng) -> Tuple[Scene, Scene, Scene, Dict[str, Any]]:
        """
        Returns A, B, C scenes and meta parameters for meta.json.
        """
        raise NotImplementedError

    def make_distractors(self, scene_c: Scene, rng, meta: Dict[str, Any]) -> Tuple[list[Scene], list[str]]:
        """
        Optional hook: return (distractors, reasons) where distractors length is 3.
        Default implementation is empty; DatasetGenerator will fall back to a generic perturbation.
        """
        return [], []
