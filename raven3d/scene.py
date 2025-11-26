from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from .geometry import Primitive


@dataclass
class Scene:
    primitives: List[Primitive] = field(default_factory=list)

    def add(self, primitive: Primitive) -> None:
        self.primitives.append(primitive)

    def copy(self) -> "Scene":
        return Scene(primitives=copy.deepcopy(self.primitives))

    def surface_areas(self) -> np.ndarray:
        return np.array([p.surface_area() for p in self.primitives], dtype=float)

    def sample_point_cloud(self, n_points: int = 4096, shuffle: bool = True) -> np.ndarray:
        if not self.primitives:
            return np.zeros((n_points, 3))
        areas = self.surface_areas()
        total = areas.sum()
        probs = areas / total if total > 0 else np.full(len(areas), 1 / len(areas))
        counts = np.random.multinomial(n_points, probs)
        pts = []
        for prim, count in zip(self.primitives, counts):
            if count <= 0:
                continue
            pts.append(prim.sample_surface(count))
        cloud = np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))
        if shuffle and len(cloud) > 0:
            np.random.shuffle(cloud)
        return cloud

    def as_descriptions(self) -> List[dict]:
        descs = []
        for prim in self.primitives:
            descs.append(
                {
                    "type": prim.__class__.__name__.lower(),
                    "center": prim.center.tolist(),
                    "rotation": prim.rotation.tolist(),
                    "scale": prim.scale.tolist(),
                    **{
                        k: float(v) if isinstance(v, (int, float)) else (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in prim.__dict__.items()
                        if k not in {"center", "rotation", "scale"}
                    },
                }
            )
        return descs
