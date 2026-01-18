from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .geometry import Primitive, Sphere, Cube, Cylinder, Cone, rotation_matrix


@dataclass
class ObjectState:
    """Mathematical object definition used throughout rule generation."""

    shape: str
    r: np.ndarray  # axis-aligned scale / size vector
    p: np.ndarray  # translation (centroid)
    rotation: np.ndarray  # Euler angles (radians)
    density: float = 1.0  # sampling weight
    color: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))

    def copy(self) -> "ObjectState":
        return ObjectState(
            shape=self.shape,
            r=self.r.copy(),
            p=self.p.copy(),
            rotation=self.rotation.copy(),
            density=float(self.density),
            color=self.color.copy(),
        )

    def rotation_matrix(self) -> np.ndarray:
        return rotation_matrix(self.rotation)

    def volume(self) -> float:
        return float(np.prod(self.r))

    def to_primitive(self) -> Primitive:
        """Convert the abstract object into a renderable primitive."""
        base_kwargs = {
            "center": self.p.copy(),
            "rotation": self.rotation.copy(),
            "scale": self.r.copy(),
        }
        shape = self.shape.lower()
        if shape == "sphere":
            return Sphere(radius=0.4, **base_kwargs)
        if shape == "cube":
            return Cube(edge_lengths=np.array([0.8, 0.8, 0.8]), **base_kwargs)
        if shape == "cylinder":
            return Cylinder(radius=0.35, height=1.0, **base_kwargs)
        if shape == "cone":
            return Cone(radius=0.35, height=1.0, **base_kwargs)
        raise ValueError(f"Unsupported shape '{self.shape}'")

    def as_dict(self) -> dict:
        rot_mat = self.rotation_matrix()
        return {
            "shape": self.shape,
            "r": self.r.tolist(),
            "p": self.p.tolist(),
            "R": rot_mat.tolist(),
            "rotation_euler": self.rotation.tolist(),
            "density": float(self.density),
            "color": self.color.tolist(),
            "volume": self.volume(),
        }


@dataclass
class Scene:
    objects: List[ObjectState] = field(default_factory=list)

    def add(self, obj: ObjectState) -> None:
        self.objects.append(obj)

    def copy(self) -> "Scene":
        return Scene(objects=[o.copy() for o in self.objects])

    def sample_point_cloud(self, n_points: int = 4096, shuffle: bool = True) -> np.ndarray:
        if not self.objects:
            return np.zeros((n_points, 3))
        weights = np.array([max(obj.density * obj.volume(), 1e-6) for obj in self.objects], dtype=float)
        probs = weights / weights.sum() if weights.sum() > 0 else np.full(len(weights), 1.0 / len(weights))
        counts = np.random.multinomial(n_points, probs)
        pts = []
        for obj, count in zip(self.objects, counts):
            if count <= 0:
                continue
            pts.append(obj.to_primitive().sample_surface(int(count)))
        cloud = np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))
        if shuffle and len(cloud) > 0:
            np.random.shuffle(cloud)
        return cloud

    def as_descriptions(self) -> List[dict]:
        return [obj.as_dict() for obj in self.objects]
