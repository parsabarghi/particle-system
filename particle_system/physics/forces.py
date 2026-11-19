"""Environmental forces.

Each force adds acceleration into ``store.acc``. Gravity is already an
acceleration. Drag, wind, the vortex, and attractors are forces, so they are
scaled by inverse mass: heavier particles drift less in the wind.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from particle_system.model.store import ParticleStore
from particle_system.scene import Scene


class ForceField(Protocol):
    def accumulate(self, store: ParticleStore, scene: Scene) -> None: ...


class Gravity:
    def accumulate(self, store: ParticleStore, scene: Scene) -> None:
        n = store.count
        if n == 0 or scene.gravity == 0.0:
            return
        store.acc[:n, 1] += scene.gravity


class Drag:
    """F = -(k1 v + k2 |v| v)."""

    def accumulate(self, store: ParticleStore, scene: Scene) -> None:
        n = store.count
        if n == 0:
            return
        k1, k2 = scene.drag_linear, scene.drag_quad
        if k1 == 0.0 and k2 == 0.0:
            return
        vel = store.vel[:n]
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        force = -(k1 * vel + k2 * speed * vel)
        store.acc[:n] += force * store.inv_mass[:n, None]


class Wind:
    def accumulate(self, store: ParticleStore, scene: Scene) -> None:
        n = store.count
        if n == 0 or not np.any(scene.wind):
            return
        store.acc[:n] += scene.wind * store.inv_mass[:n, None]


class Vortex:
    """Right-handed swirl about the world Y axis. Tangent = Ŷ × r_perp."""

    def accumulate(self, store: ParticleStore, scene: Scene) -> None:
        n = store.count
        if n == 0 or scene.vortex == 0.0:
            return
        radial = store.pos[:n, [0, 2]]
        r2 = np.sum(radial * radial, axis=1, keepdims=True)
        scale = (scene.vortex / (r2 + 1.35)) * store.inv_mass[:n, None]
        # (rz, -rx) in the XZ plane.
        store.acc[:n, 0] += store.pos[:n, 2] * scale[:, 0]
        store.acc[:n, 2] -= store.pos[:n, 0] * scale[:, 0]


class PointAttractor:
    """Softened point force. strength > 0 attracts, < 0 repels.

    F = strength * r̂ / (r² + ε), with r the vector from the particle to the point.
    """

    def __init__(self, point: np.ndarray, strength: float) -> None:
        self.point = np.asarray(point, dtype=np.float64)
        self.strength = strength

    def accumulate(self, store: ParticleStore, scene: Scene) -> None:
        del scene
        n = store.count
        if n == 0 or self.strength == 0.0:
            return
        delta = self.point - store.pos[:n]
        r2 = np.sum(delta * delta, axis=1, keepdims=True)
        r = np.sqrt(np.maximum(r2, 1e-12))
        mag = (self.strength / (r2 + 0.45)) * store.inv_mass[:n, None]
        store.acc[:n] += (delta / r) * mag


class Well:
    """Central well. F = strength * r̂ / (r² + ε) toward ``scene.well_center``."""

    def accumulate(self, store: ParticleStore, scene: Scene) -> None:
        n = store.count
        if n == 0 or scene.well == 0.0:
            return
        delta = scene.well_center - store.pos[:n]
        if scene.dimension == "2d":
            delta = delta.copy()
            delta[:, 2] = 0.0
        r2 = np.sum(delta * delta, axis=1, keepdims=True)
        r = np.sqrt(np.maximum(r2, 1e-12))
        mag = (scene.well / (r2 + 2.2)) * store.inv_mass[:n, None]
        store.acc[:n] += (delta / r) * mag
