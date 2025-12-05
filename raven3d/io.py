from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_ply(path: str | os.PathLike, points: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    count = points.shape[0]
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {count}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    with path.open("w") as f:
        f.write("\n".join(header) + "\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def write_meta(path: str | os.PathLike, meta: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
