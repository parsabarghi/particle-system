"""Structure-of-arrays particle buffer.

Alive particles occupy the prefix ``[:count]``. That layout is what lets the
force and collision passes stay in NumPy instead of walking Python objects.
"""

from __future__ import annotations

import numpy as np


class ParticleStore:
    def __init__(self, capacity: int = 12_000) -> None:
        self.capacity = capacity
        self.count = 0
        self.pos = np.zeros((capacity, 3), dtype=np.float64)
        self.vel = np.zeros((capacity, 3), dtype=np.float64)
        self.acc = np.zeros((capacity, 3), dtype=np.float64)
        self.prev = np.zeros((capacity, 3), dtype=np.float64)
        self.mass = np.ones(capacity, dtype=np.float64)
        self.inv_mass = np.ones(capacity, dtype=np.float64)
        self.age = np.zeros(capacity, dtype=np.float64)
        self.life = np.ones(capacity, dtype=np.float64)
        self.size = np.ones(capacity, dtype=np.float64)
        self.color0 = np.ones((capacity, 4), dtype=np.float64)
        self.color1 = np.ones((capacity, 4), dtype=np.float64)
        self.hit = np.zeros(capacity, dtype=bool)

    def clear(self) -> None:
        self.count = 0

    def spawn_batch(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        mass: np.ndarray,
        life: np.ndarray,
        size: np.ndarray,
        color0: np.ndarray,
        color1: np.ndarray,
        dt: float,
    ) -> None:
        """Append a batch. If the pool is full, drop the oldest prefix first."""
        n = int(pos.shape[0])
        if n <= 0:
            return
        if n > self.capacity:
            start = n - self.capacity
            pos, vel, mass = pos[start:], vel[start:], mass[start:]
            life, size = life[start:], size[start:]
            color0, color1 = color0[start:], color1[start:]
            n = self.capacity
            self.count = 0
        elif self.count + n > self.capacity:
            self._drop_prefix(self.count + n - self.capacity)

        i0 = self.count
        i1 = i0 + n
        self.pos[i0:i1] = pos
        self.vel[i0:i1] = vel
        self.acc[i0:i1] = 0.0
        self.prev[i0:i1] = pos - vel * dt
        self.mass[i0:i1] = mass
        self.inv_mass[i0:i1] = np.divide(1.0, mass, out=np.zeros(n), where=mass > 1e-8)
        self.age[i0:i1] = 0.0
        self.life[i0:i1] = life
        self.size[i0:i1] = size
        self.color0[i0:i1] = color0
        self.color1[i0:i1] = color1
        self.hit[i0:i1] = False
        self.count = i1

    def kill_dead(self) -> None:
        """Compact the prefix so dead particles disappear without leaving holes."""
        n = self.count
        if n == 0:
            return
        alive = self.age[:n] < self.life[:n]
        kept = int(np.count_nonzero(alive))
        if kept == n:
            return
        self.pos[:kept] = self.pos[:n][alive]
        self.vel[:kept] = self.vel[:n][alive]
        self.acc[:kept] = self.acc[:n][alive]
        self.prev[:kept] = self.prev[:n][alive]
        self.mass[:kept] = self.mass[:n][alive]
        self.inv_mass[:kept] = self.inv_mass[:n][alive]
        self.age[:kept] = self.age[:n][alive]
        self.life[:kept] = self.life[:n][alive]
        self.size[:kept] = self.size[:n][alive]
        self.color0[:kept] = self.color0[:n][alive]
        self.color1[:kept] = self.color1[:n][alive]
        self.hit[:kept] = self.hit[:n][alive]
        self.count = kept

    def _drop_prefix(self, drop: int) -> None:
        keep = self.count - drop
        if keep <= 0:
            self.count = 0
            return
        self.pos[:keep] = self.pos[drop : self.count]
        self.vel[:keep] = self.vel[drop : self.count]
        self.prev[:keep] = self.prev[drop : self.count]
        self.mass[:keep] = self.mass[drop : self.count]
        self.inv_mass[:keep] = self.inv_mass[drop : self.count]
        self.age[:keep] = self.age[drop : self.count]
        self.life[:keep] = self.life[drop : self.count]
        self.size[:keep] = self.size[drop : self.count]
        self.color0[:keep] = self.color0[drop : self.count]
        self.color1[:keep] = self.color1[drop : self.count]
        self.count = keep
