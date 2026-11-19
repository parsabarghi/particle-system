"""Numerical integrators.

Semi-implicit Euler
    v ← v + a Δt
    x ← x + v Δt
    The position step consumes the updated velocity. That is what keeps a
    stiff drag term from blowing up the way explicit Euler does.

Position Verlet
    x_new = 2x − x_prev + a Δt²
    v     = (x_new − x_prev) / (2 Δt)
    Previous positions are seeded at birth as x − v Δt. Velocity-dependent
    forces such as drag are one substep late, which is expected for this form.
"""

from __future__ import annotations

import numpy as np

from particle_system.model.store import ParticleStore


def integrate_euler(store: ParticleStore, dt: float) -> None:
    n = store.count
    if n == 0:
        return
    store.vel[:n] += store.acc[:n] * dt
    store.pos[:n] += store.vel[:n] * dt


def integrate_verlet(store: ParticleStore, dt: float) -> None:
    n = store.count
    if n == 0:
        return
    pos = store.pos[:n]
    prev = store.prev[:n]
    new = 2.0 * pos - prev + store.acc[:n] * (dt * dt)
    store.vel[:n] = (new - prev) / (2.0 * dt)
    store.prev[:n] = pos
    store.pos[:n] = new


def seed_history(store: ParticleStore, dt: float) -> None:
    n = store.count
    if n == 0:
        return
    store.prev[:n] = store.pos[:n] - store.vel[:n] * dt


def resync_hits(store: ParticleStore, dt: float) -> None:
    """Rebuild x_prev only for particles a collider just moved.

    Applying this to every particle each frame collapses Verlet into a
    half-step Euler, so the mask matters.
    """
    n = store.count
    if n == 0:
        return
    hit = store.hit[:n]
    if not np.any(hit):
        return
    prev = store.prev[:n]
    prev[hit] = store.pos[:n][hit] - store.vel[:n][hit] * dt
