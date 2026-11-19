"""Scene parameters shared by the emitter, the force stack, and the colliders."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _vec(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float64)


@dataclass
class Scene:
    integrator: str = "euler"
    dimension: str = "3d"
    gravity: float = -16.0
    drag_linear: float = 0.15
    drag_quad: float = 0.03
    wind: np.ndarray = field(default_factory=lambda: _vec(0, 0, 0))
    vortex: float = 0.0
    well: float = 0.0
    well_center: np.ndarray = field(default_factory=lambda: _vec(0, 5, 0))
    restitution: float = 0.45
    friction: float = 0.28
    ground: bool = True
    box: bool = False
    sphere: bool = False
    sphere_center: np.ndarray = field(default_factory=lambda: _vec(0, 3.2, 0))
    sphere_radius: float = 1.7
    emission: str = "continuous"
    rate: float = 340.0
    burst_count: int = 640
    burst_every: float = 1.7
    life: tuple[float, float] = (1.8, 2.7)
    size: tuple[float, float] = (0.10, 0.20)
    mass: tuple[float, float] = (0.75, 1.15)
    speed: float = 11.0
    spread: float = 0.2
    cone: float = 0.4
    origin: np.ndarray = field(default_factory=lambda: _vec(0, 0.2, 0))
    direction: np.ndarray = field(default_factory=lambda: _vec(0, 1, 0))
    shape: str = "point"
    velocity_mode: str = "cone"
    color0: np.ndarray = field(default_factory=lambda: _vec(0.78, 0.93, 1.0, 0.95))
    color1: np.ndarray = field(default_factory=lambda: _vec(0.16, 0.45, 0.58, 0.0))
    pointer_strength: float = 70.0
    pointer_rate: float = 420.0

    def copy(self) -> Scene:
        twin = Scene()
        for key, value in self.__dict__.items():
            setattr(twin, key, value.copy() if isinstance(value, np.ndarray) else value)
        return twin
