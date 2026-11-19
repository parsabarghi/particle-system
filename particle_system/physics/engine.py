"""Fixed-step physics orchestrator.

Order each step:
    emit → clear acceleration → forces → integrate → collide → planar lock → age.
The renderer is not allowed to write back into the store.
"""

from __future__ import annotations

import numpy as np

from particle_system.emit.manager import EmitterManager
from particle_system.model.store import ParticleStore
from particle_system.physics.colliders import resolve_box, resolve_ground, resolve_sphere
from particle_system.physics.forces import Drag, Gravity, PointAttractor, Vortex, Well, Wind
from particle_system.physics.integrators import (
    integrate_euler,
    integrate_verlet,
    resync_hits,
    seed_history,
)
from particle_system.scene import Scene

MAX_SPEED = 52.0


class PhysicsEngine:
    def __init__(self, scene: Scene | None = None) -> None:
        self.scene = scene or Scene()
        self.store = ParticleStore()
        self.emitter = EmitterManager()
        self.pointer_down = False
        self.pointer_mode = "spawn"
        self.pointer = np.zeros(3, dtype=np.float64)
        self._spawn_carry = 0.0
        self._primed_integrator: str | None = None
        self.forces = [Gravity(), Drag(), Wind(), Vortex(), Well()]

    def reset(self) -> None:
        self.store.clear()
        self.emitter.reset()
        self._spawn_carry = 0.0
        self._primed_integrator = None

    def warm(self, steps: int = 40, dt: float = 1.0 / 90.0) -> None:
        for _ in range(steps):
            self.step(dt)

    def step(self, dt: float) -> None:
        scene = self.scene
        store = self.store
        if self._primed_integrator != scene.integrator:
            if scene.integrator == "verlet":
                seed_history(store, dt)
            self._primed_integrator = scene.integrator

        self.emitter.update(store, scene, dt)
        self._emit_from_pointer(dt)

        n = store.count
        if n:
            store.acc[:n] = 0.0
            store.hit[:n] = False

        for force in self.forces:
            force.accumulate(store, scene)
        self._pointer_force()

        if scene.integrator == "verlet":
            integrate_verlet(store, dt)
        else:
            integrate_euler(store, dt)

        resolve_sphere(store, scene, dt)
        resolve_box(store, scene, dt)
        resolve_ground(store, scene, dt)

        n = store.count
        if scene.dimension == "2d" and n:
            store.pos[:n, 2] = 0.0
            store.vel[:n, 2] = 0.0
            store.acc[:n, 2] = 0.0
            store.prev[:n, 2] = 0.0
        self._limit_speed()
        if scene.integrator == "verlet":
            resync_hits(store, dt)

        if store.count:
            store.age[: store.count] += dt
            store.kill_dead()

    def _emit_from_pointer(self, dt: float) -> None:
        if not self.pointer_down or self.pointer_mode != "spawn":
            self._spawn_carry = 0.0
            return
        self._spawn_carry += self.scene.pointer_rate * dt
        count = int(self._spawn_carry)
        if count <= 0:
            return
        self._spawn_carry -= count
        self.emitter.emit_at(self.store, self.scene, count, self.pointer, dt)

    def _pointer_force(self) -> None:
        if not self.pointer_down:
            return
        if self.pointer_mode == "attract":
            strength = self.scene.pointer_strength
        elif self.pointer_mode == "repel":
            strength = -self.scene.pointer_strength
        else:
            return
        PointAttractor(self.pointer, strength).accumulate(self.store, self.scene)

    def _limit_speed(self) -> None:
        n = self.store.count
        if n == 0:
            return
        vel = self.store.vel[:n]
        speed = np.linalg.norm(vel, axis=1)
        too_fast = speed > MAX_SPEED
        if not np.any(too_fast):
            return
        scale = np.ones(n)
        scale[too_fast] = MAX_SPEED / speed[too_fast]
        vel *= scale[:, None]
