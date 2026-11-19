"""Continuous and burst emission into a ParticleStore."""

from __future__ import annotations

import numpy as np

from particle_system.model.store import ParticleStore
from particle_system.scene import Scene

BOX_X = (-7.7, 7.7)
BOX_Z = (-7.7, 7.7)
RAIN_Y = 11.75


class EmitterManager:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()
        self.rate_accum = 0.0
        self.burst_accum = 0.0
        self.primed = True

    def reset(self) -> None:
        self.rate_accum = 0.0
        self.burst_accum = 0.0
        self.primed = True

    def burst_now(self) -> None:
        self.primed = True

    def update(self, store: ParticleStore, scene: Scene, dt: float) -> None:
        if scene.emission == "continuous":
            self.rate_accum += max(0.0, scene.rate) * dt
            count = int(self.rate_accum)
            if count > 0:
                self.rate_accum -= count
                self.emit(store, scene, min(count, 2000), dt)
            return
        self.burst_accum += dt
        if self.primed or (scene.burst_every > 0 and self.burst_accum >= scene.burst_every):
            self.burst_accum = 0.0
            self.primed = False
            self.emit(store, scene, scene.burst_count, dt)

    def emit_at(
        self,
        store: ParticleStore,
        scene: Scene,
        count: int,
        point: np.ndarray,
        dt: float,
    ) -> None:
        if count <= 0:
            return
        origin = np.broadcast_to(np.asarray(point, dtype=np.float64), (count, 3)).copy()
        origin += self.rng.uniform(-0.12, 0.12, size=origin.shape)
        if scene.dimension == "2d":
            origin[:, 2] = 0.0
        self._write(store, scene, origin, dt)

    def emit(self, store: ParticleStore, scene: Scene, count: int, dt: float) -> None:
        if count <= 0:
            return
        pos = self._positions(scene, count)
        self._write(store, scene, pos, dt)

    def _write(self, store: ParticleStore, scene: Scene, pos: np.ndarray, dt: float) -> None:
        count = pos.shape[0]
        vel = self._velocities(scene, pos)
        mass = self.rng.uniform(scene.mass[0], scene.mass[1], size=count)
        life = self.rng.uniform(scene.life[0], max(scene.life[0], scene.life[1]), size=count)
        size = self.rng.uniform(scene.size[0], max(scene.size[0], scene.size[1]), size=count)
        color0 = np.broadcast_to(scene.color0, (count, 4)).copy()
        color1 = np.broadcast_to(scene.color1, (count, 4)).copy()
        if scene.dimension == "2d":
            pos = pos.copy()
            vel = vel.copy()
            pos[:, 2] = 0.0
            vel[:, 2] = 0.0
        store.spawn_batch(pos, vel, mass, life, size, color0, color1, dt)

    def _positions(self, scene: Scene, count: int) -> np.ndarray:
        rng = self.rng
        origin = scene.origin
        if scene.shape == "rain":
            pos = np.column_stack(
                (
                    rng.uniform(BOX_X[0], BOX_X[1], size=count),
                    np.full(count, RAIN_Y),
                    rng.uniform(BOX_Z[0], BOX_Z[1], size=count),
                )
            )
        elif scene.shape == "point":
            pos = np.broadcast_to(origin, (count, 3)).copy()
        elif scene.shape == "volume":
            pos = origin + rng.uniform(-scene.spread, scene.spread, size=(count, 3))
            pos[:, 1] = origin[1] + rng.uniform(-scene.spread * 0.4, scene.spread * 0.4, size=count)
        else:
            ang = rng.uniform(0.0, 2.0 * np.pi, size=count)
            radius = scene.spread if scene.shape == "ring" else scene.spread * np.sqrt(rng.random(count))
            if scene.dimension == "2d":
                pos = np.column_stack(
                    (
                        origin[0] + np.cos(ang) * radius,
                        origin[1] + np.sin(ang) * radius,
                        np.zeros(count),
                    )
                )
            else:
                pos = np.column_stack(
                    (
                        origin[0] + np.cos(ang) * radius,
                        np.full(count, origin[1]),
                        origin[2] + np.sin(ang) * radius,
                    )
                )
        if scene.dimension == "2d":
            pos[:, 2] = 0.0
        return pos

    def _velocities(self, scene: Scene, pos: np.ndarray) -> np.ndarray:
        rng = self.rng
        count = pos.shape[0]
        speed = scene.speed * rng.uniform(0.75, 1.15, size=count)
        mode = scene.velocity_mode
        if mode == "drift":
            vel = np.column_stack(
                (
                    rng.uniform(-0.45, 0.45, size=count),
                    -speed,
                    rng.uniform(-0.45, 0.45, size=count),
                )
            )
        elif mode == "sphere":
            vel = rng.normal(size=(count, 3))
            vel /= np.maximum(np.linalg.norm(vel, axis=1, keepdims=True), 1e-8)
            vel *= speed[:, None]
        elif mode == "tangential":
            if scene.dimension == "2d":
                rel = pos[:, :2] - scene.origin[:2]
                length = np.maximum(np.linalg.norm(rel, axis=1), 1e-8)
                vel = np.column_stack((-rel[:, 1] / length * speed, rel[:, 0] / length * speed, np.zeros(count)))
            else:
                rel = pos[:, [0, 2]] - scene.origin[[0, 2]]
                length = np.maximum(np.linalg.norm(rel, axis=1), 1e-8)
                # Ŷ × r = (rz, 0, -rx)
                vel = np.column_stack(
                    (
                        rel[:, 1] / length * speed,
                        rng.uniform(-0.15, 0.15, size=count),
                        -rel[:, 0] / length * speed,
                    )
                )
        else:
            dirs = _cone(scene.direction, scene.cone, count, rng)
            vel = dirs * speed[:, None]
        if scene.dimension == "2d":
            vel[:, 2] = 0.0
        return vel


def _cone(direction: np.ndarray, angle: float, count: int, rng: np.random.Generator) -> np.ndarray:
    forward = np.asarray(direction, dtype=np.float64)
    forward = forward / max(float(np.linalg.norm(forward)), 1e-8)
    helper = np.array([1.0, 0.0, 0.0] if abs(forward[1]) >= 0.9 else [0.0, 1.0, 0.0])
    right = np.cross(forward, helper)
    right /= max(float(np.linalg.norm(right)), 1e-8)
    up = np.cross(right, forward)
    theta = angle * np.sqrt(rng.random(count))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=count)
    st, ct = np.sin(theta), np.cos(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    offset = right * (cp * st)[:, None] + up * (sp * st)[:, None]
    return forward * ct[:, None] + offset
