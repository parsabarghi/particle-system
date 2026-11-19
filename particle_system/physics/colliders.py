"""Analytical collision response.

Ground plane, axis-aligned box, and sphere. The outward unit normal ``n``
reflects an approaching velocity:

    v ← v − (1+e) v_n n          when v_n = v · n < 0

Friction then removes tangential speed up to μ (1+e) |v_n| (an impulse, so it
does not depend on the frame rate) and applies a small exponential slide damping
while the contact persists. Impacts slower than ``SLEEP`` are captured (e = 0)
so resting piles do not chatter.
"""

from __future__ import annotations

import numpy as np

from particle_system.model.store import ParticleStore
from particle_system.scene import Scene

SLEEP = 0.55
BOX_MIN = np.array([-8.0, 0.0, -8.0])
BOX_MAX = np.array([8.0, 12.0, 8.0])


def _respond(
    store: ParticleStore,
    mask: np.ndarray,
    normals: np.ndarray,
    restitution: float,
    friction: float,
    dt: float,
) -> None:
    n = store.count
    if n == 0 or not np.any(mask):
        return
    vel = store.vel[:n]
    vn = np.sum(vel * normals, axis=1)
    approaching = mask & (vn < 0.0)
    e_use = np.where(approaching & (vn > -SLEEP), 0.0, restitution)
    bounce = np.where(approaching, -(1.0 + e_use) * vn, 0.0)
    vel += bounce[:, None] * normals

    vn2 = np.sum(vel * normals, axis=1)
    tang = vel - vn2[:, None] * normals
    ts = np.linalg.norm(tang, axis=1)
    max_f = np.where(approaching, friction * (1.0 + e_use) * (-vn), 0.0)
    scale = np.ones(n, dtype=np.float64)
    blocked = approaching & (ts <= max_f) & (max_f > 0.0) & (ts > 1e-6)
    partial = approaching & (ts > max_f) & (max_f > 0.0)
    scale = np.where(blocked, 0.0, scale)
    scale = np.where(partial, (ts - max_f) / np.maximum(ts, 1e-8), scale)
    tang *= scale[:, None]
    vel[:] = vn2[:, None] * normals + tang

    vn3 = np.sum(vel * normals, axis=1)
    tang = vel - vn3[:, None] * normals
    damp = np.where(mask, np.exp(-friction * 1.4 * dt), 1.0)
    tang *= damp[:, None]
    nvel = vn3.copy()
    nvel = np.where(mask & (np.abs(nvel) < SLEEP), 0.0, nvel)
    vel[:] = nvel[:, None] * normals + tang
    store.hit[:n] |= mask


def _plane(n: int, normal: np.ndarray) -> np.ndarray:
    return np.repeat(normal[None, :], n, axis=0)


def resolve_ground(store: ParticleStore, scene: Scene, dt: float) -> None:
    n = store.count
    if n == 0 or not scene.ground:
        return
    mask = store.pos[:n, 1] < 0.0
    if not np.any(mask):
        return
    store.pos[:n, 1] = np.where(mask, 0.0, store.pos[:n, 1])
    _respond(store, mask, _plane(n, np.array([0.0, 1.0, 0.0])), scene.restitution, scene.friction, dt)


def resolve_box(store: ParticleStore, scene: Scene, dt: float) -> None:
    n = store.count
    if n == 0 or not scene.box:
        return
    pos = store.pos[:n]
    e, mu = scene.restitution, scene.friction
    faces = (
        (pos[:, 0] < BOX_MIN[0], np.array([1.0, 0.0, 0.0]), 0, BOX_MIN[0]),
        (pos[:, 0] > BOX_MAX[0], np.array([-1.0, 0.0, 0.0]), 0, BOX_MAX[0]),
        (pos[:, 1] < BOX_MIN[1], np.array([0.0, 1.0, 0.0]), 1, BOX_MIN[1]),
        (pos[:, 1] > BOX_MAX[1], np.array([0.0, -1.0, 0.0]), 1, BOX_MAX[1]),
    )
    extra = ()
    if scene.dimension != "2d":
        extra = (
            (pos[:, 2] < BOX_MIN[2], np.array([0.0, 0.0, 1.0]), 2, BOX_MIN[2]),
            (pos[:, 2] > BOX_MAX[2], np.array([0.0, 0.0, -1.0]), 2, BOX_MAX[2]),
        )
    for mask, normal, axis, limit in (*faces, *extra):
        if not np.any(mask):
            continue
        store.pos[:n, axis] = np.where(mask, limit, store.pos[:n, axis])
        _respond(store, mask, _plane(n, normal), e, mu, dt)


def resolve_sphere(store: ParticleStore, scene: Scene, dt: float) -> None:
    n = store.count
    if n == 0 or not scene.sphere:
        return
    rel = store.pos[:n] - scene.sphere_center
    if scene.dimension == "2d":
        rel = rel.copy()
        rel[:, 2] = 0.0
    d2 = np.sum(rel * rel, axis=1)
    radius = scene.sphere_radius
    mask = (d2 < radius * radius) & (d2 > 1e-8)
    if not np.any(mask):
        return
    dist = np.sqrt(np.maximum(d2, 1e-12))
    normals = rel / dist[:, None]
    pushed = scene.sphere_center + normals * radius
    store.pos[:n] = np.where(mask[:, None], pushed, store.pos[:n])
    _respond(store, mask, normals, scene.restitution, scene.friction, dt)
