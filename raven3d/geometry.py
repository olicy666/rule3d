from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np


def rotation_matrix(euler_radians: np.ndarray) -> np.ndarray:
    """Compute XYZ Euler rotation matrix."""
    x, y, z = euler_radians
    cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
    sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def random_unit_vectors(n: int) -> np.ndarray:
    """Sample n random unit vectors uniformly on the sphere."""
    vec = np.random.normal(size=(n, 3))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
    return vec


@dataclass
class Primitive:
    center: np.ndarray
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))

    def surface_area(self) -> float:
        raise NotImplementedError

    def sample_surface(self, n_points: int) -> np.ndarray:
        local_pts = self._sample_local_surface(n_points)
        rot = rotation_matrix(self.rotation)
        scaled = local_pts * self.scale
        rotated = scaled @ rot.T
        return rotated + self.center

    def _sample_local_surface(self, n_points: int) -> np.ndarray:
        raise NotImplementedError


@dataclass
class Sphere(Primitive):
    radius: float = 0.5

    def surface_area(self) -> float:
        avg_scale = float(np.mean(self.scale))
        return 4 * math.pi * (self.radius * avg_scale) ** 2

    def _sample_local_surface(self, n_points: int) -> np.ndarray:
        dirs = random_unit_vectors(n_points)
        return dirs * self.radius


@dataclass
class Cube(Primitive):
    edge_lengths: np.ndarray = field(default_factory=lambda: np.ones(3))

    def surface_area(self) -> float:
        lx, ly, lz = self.edge_lengths * self.scale
        return 2 * (lx * ly + ly * lz + lx * lz)

    def _sample_local_surface(self, n_points: int) -> np.ndarray:
        lx, ly, lz = self.edge_lengths / 2.0
        faces = np.array(
            [
                ly * lz,  # +x
                ly * lz,  # -x
                lx * lz,  # +y
                lx * lz,  # -y
                lx * ly,  # +z
                lx * ly,  # -z
            ],
            dtype=float,
        )
        probs = faces / faces.sum()
        face_choices = np.random.choice(6, size=n_points, p=probs)
        pts = np.zeros((n_points, 3))
        for i, face in enumerate(face_choices):
            u = np.random.uniform(-1, 1, size=2)
            if face == 0:  # +x
                pts[i] = [lx, u[0] * ly, u[1] * lz]
            elif face == 1:  # -x
                pts[i] = [-lx, u[0] * ly, u[1] * lz]
            elif face == 2:  # +y
                pts[i] = [u[0] * lx, ly, u[1] * lz]
            elif face == 3:  # -y
                pts[i] = [u[0] * lx, -ly, u[1] * lz]
            elif face == 4:  # +z
                pts[i] = [u[0] * lx, u[1] * ly, lz]
            else:  # -z
                pts[i] = [u[0] * lx, u[1] * ly, -lz]
        return pts


@dataclass
class TriangularPrism(Primitive):
    side_length: float = 0.8
    height: float = 1.0

    def surface_area(self) -> float:
        s = self.side_length
        tri_h = s * math.sqrt(3) / 2.0
        tri_area = 0.5 * s * tri_h
        return 2 * tri_area + 3 * s * self.height

    def _sample_local_surface(self, n_points: int) -> np.ndarray:
        s = self.side_length
        tri_h = s * math.sqrt(3) / 2.0
        v0 = np.array([-s / 2.0, -tri_h / 3.0, 0.0])
        v1 = np.array([s / 2.0, -tri_h / 3.0, 0.0])
        v2 = np.array([0.0, 2.0 * tri_h / 3.0, 0.0])

        tri_area = 0.5 * s * tri_h
        side_area = s * self.height
        probs = np.array([tri_area, tri_area, side_area, side_area, side_area], dtype=float)
        probs = probs / probs.sum()
        face_choices = np.random.choice(5, size=n_points, p=probs)
        pts = np.zeros((n_points, 3))
        z_top = self.height / 2.0
        z_bottom = -self.height / 2.0

        for i, face in enumerate(face_choices):
            if face in (0, 1):
                r1, r2 = np.random.uniform(size=2)
                if r1 + r2 > 1.0:
                    r1 = 1.0 - r1
                    r2 = 1.0 - r2
                p = v0 + r1 * (v1 - v0) + r2 * (v2 - v0)
                p[2] = z_top if face == 0 else z_bottom
                pts[i] = p
            else:
                if face == 2:
                    a, b = v0, v1
                elif face == 3:
                    a, b = v1, v2
                else:
                    a, b = v2, v0
                t = np.random.uniform()
                z = np.random.uniform(z_bottom, z_top)
                p = a + t * (b - a)
                pts[i] = [p[0], p[1], z]
        return pts


@dataclass
class Cylinder(Primitive):
    radius: float = 0.5
    height: float = 1.0

    def surface_area(self) -> float:
        sx, sy, sz = self.scale
        # Approximate radius scaling by mean of x/y scale.
        scale_r = float((sx + sy) / 2.0)
        return 2 * math.pi * (self.radius * scale_r) * (self.height * sz) + 2 * math.pi * (self.radius * scale_r) ** 2

    def _sample_local_surface(self, n_points: int) -> np.ndarray:
        side_area = 2 * math.pi * self.radius * self.height
        cap_area = 2 * math.pi * self.radius**2
        probs = np.array([side_area, cap_area])
        probs = probs / probs.sum()
        side_count = np.random.binomial(n_points, probs[0])
        cap_count = n_points - side_count

        pts: List[np.ndarray] = []
        if side_count > 0:
            theta = np.random.uniform(0, 2 * math.pi, size=side_count)
            z = np.random.uniform(-self.height / 2.0, self.height / 2.0, size=side_count)
            pts.append(np.stack([self.radius * np.cos(theta), self.radius * np.sin(theta), z], axis=1))
        if cap_count > 0:
            # Split evenly between top and bottom caps.
            top = cap_count // 2
            bottom = cap_count - top
            for count, z_val in [(top, self.height / 2.0), (bottom, -self.height / 2.0)]:
                if count == 0:
                    continue
                r = self.radius * np.sqrt(np.random.uniform(size=count))
                theta = np.random.uniform(0, 2 * math.pi, size=count)
                pts.append(np.stack([r * np.cos(theta), r * np.sin(theta), np.full(count, z_val)], axis=1))
        return np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))


@dataclass
class Cone(Primitive):
    radius: float = 0.5
    height: float = 1.0

    def surface_area(self) -> float:
        sx, sy, sz = self.scale
        scale_r = float((sx + sy) / 2.0)
        slant = math.sqrt((self.radius * scale_r) ** 2 + (self.height * sz) ** 2)
        return math.pi * self.radius * scale_r * slant + math.pi * (self.radius * scale_r) ** 2

    def _sample_local_surface(self, n_points: int) -> np.ndarray:
        base_area = math.pi * self.radius**2
        slant_area = math.pi * self.radius * math.sqrt(self.radius**2 + self.height**2)
        probs = np.array([slant_area, base_area])
        probs = probs / probs.sum()
        side_count = np.random.binomial(n_points, probs[0])
        base_count = n_points - side_count

        pts: List[np.ndarray] = []
        if side_count > 0:
            # Approximate uniform sampling on lateral surface.
            u = np.random.uniform(size=side_count)
            h = np.random.uniform(size=side_count)
            theta = 2 * math.pi * h
            r = self.radius * (1 - u)
            z = -self.height / 2.0 + self.height * u
            pts.append(np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1))
        if base_count > 0:
            r = self.radius * np.sqrt(np.random.uniform(size=base_count))
            theta = np.random.uniform(0, 2 * math.pi, size=base_count)
            z = np.full(base_count, -self.height / 2.0)
            pts.append(np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1))
        return np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))


@dataclass
class Capsule(Primitive):
    radius: float = 0.35
    height: float = 0.7

    def surface_area(self) -> float:
        side_area = 2 * math.pi * self.radius * self.height
        cap_area = 4 * math.pi * self.radius**2
        return side_area + cap_area

    def _sample_local_surface(self, n_points: int) -> np.ndarray:
        side_area = 2 * math.pi * self.radius * self.height
        cap_area = 4 * math.pi * self.radius**2
        probs = np.array([side_area, cap_area], dtype=float)
        probs = probs / probs.sum()
        side_count = np.random.binomial(n_points, probs[0])
        cap_count = n_points - side_count

        pts: List[np.ndarray] = []
        if side_count > 0:
            theta = np.random.uniform(0, 2 * math.pi, size=side_count)
            z = np.random.uniform(-self.height / 2.0, self.height / 2.0, size=side_count)
            pts.append(np.stack([self.radius * np.cos(theta), self.radius * np.sin(theta), z], axis=1))
        if cap_count > 0:
            half = cap_count // 2
            counts = [(half, 1.0), (cap_count - half, -1.0)]
            for count, sign in counts:
                if count == 0:
                    continue
                dirs = random_unit_vectors(count)
                dirs[:, 2] = np.abs(dirs[:, 2]) * sign
                cap_center = np.array([0.0, 0.0, sign * self.height / 2.0])
                pts.append(dirs * self.radius + cap_center)
        return np.concatenate(pts, axis=0) if pts else np.zeros((0, 3))


@dataclass
class Torus(Primitive):
    major_radius: float = 0.5
    minor_radius: float = 0.2

    def surface_area(self) -> float:
        return 4 * math.pi**2 * self.major_radius * self.minor_radius

    def _sample_local_surface(self, n_points: int) -> np.ndarray:
        theta = np.random.uniform(0, 2 * math.pi, size=n_points)
        phi = np.random.uniform(0, 2 * math.pi, size=n_points)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x = (self.major_radius + self.minor_radius * cos_phi) * cos_theta
        y = (self.major_radius + self.minor_radius * cos_phi) * sin_theta
        z = self.minor_radius * sin_phi
        return np.stack([x, y, z], axis=1)
