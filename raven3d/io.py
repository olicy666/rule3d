from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_ply(path: str | os.PathLike, points: np.ndarray, color: np.ndarray | tuple | list | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    count = points.shape[0]
    color_block: np.ndarray | None = None
    if color is not None:
        color_block = np.asarray(color, dtype=np.uint8)
        if color_block.ndim == 1:
            # Broadcast a single RGB value to all points.
            color_block = np.broadcast_to(color_block, (count, 3)).astype(np.uint8)
        elif color_block.shape == (count, 3):
            color_block = color_block.astype(np.uint8)
        else:
            raise ValueError("color must be length-3 RGB or shape (N,3) array matching points")
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {count}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if color_block is not None:
        header.extend(
            [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        )
    header.append("end_header")

    with path.open("w") as f:
        f.write("\n".join(header) + "\n")
        if color_block is None:
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            return
        for p, c in zip(points, color_block):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_meta(path: str | os.PathLike, meta: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
