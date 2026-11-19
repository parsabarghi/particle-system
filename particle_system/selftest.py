"""Headless checks for integration, contacts, emission, and recycling.

    python -m particle_system.selftest
"""

from __future__ import annotations

import numpy as np

from particle_system.emit.manager import EmitterManager
from particle_system.model.store import ParticleStore
from particle_system.physics.colliders import resolve_box, resolve_ground, resolve_sphere
from particle_system.physics.engine import PhysicsEngine
from particle_system.physics.forces import Drag, Gravity, Wind
from particle_system.physics.integrators import integrate_euler, integrate_verlet
from particle_system.scene import Scene


def _check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"  ok  {name}")


def test_euler_gravity() -> None:
    store = ParticleStore(8)
    dt = 1.0 / 120.0
    steps = 120
    g = -9.8
    _spawn_one(store, y=10.0, dt=dt)
    for _ in range(steps):
        store.acc[:1] = (0.0, g, 0.0)
        integrate_euler(store, dt)
    # Semi-implicit Euler from rest: y = y0 + g dt² n(n+1)/2
    expected = 10.0 + g * dt * dt * steps * (steps + 1) / 2.0
    _check("euler gravity matches the closed form", abs(float(store.pos[0, 1]) - expected) < 1e-8)


def test_verlet_falls() -> None:
    store = ParticleStore(4)
    dt = 1.0 / 60.0
    _spawn_one(store, y=5.0, dt=dt)
    for _ in range(30):
        store.acc[:1] = (0.0, -10.0, 0.0)
        integrate_verlet(store, dt)
    _check("verlet moves downward under gravity", float(store.pos[0, 1]) < 5.0)
    _check("verlet gains downward speed", float(store.vel[0, 1]) < 0.0)


def test_drag_slows() -> None:
    store = ParticleStore(4)
    _spawn_one(store, y=0.0, vx=12.0, dt=1 / 60)
    scene = Scene(drag_linear=1.5, drag_quad=0.2, gravity=0.0)
    drag = Drag()
    for _ in range(40):
        store.acc[:1] = 0.0
        drag.accumulate(store, scene)
        integrate_euler(store, 1 / 60)
    _check("drag reduces horizontal speed", abs(float(store.vel[0, 0])) < 12.0)


def test_heavy_wind() -> None:
    light = ParticleStore(2)
    heavy = ParticleStore(2)
    _spawn_one(light, mass=0.4, dt=1 / 60)
    _spawn_one(heavy, mass=4.0, dt=1 / 60)
    scene = Scene(gravity=0.0, drag_linear=0.0, drag_quad=0.0)
    scene.wind[:] = (8.0, 0.0, 0.0)
    wind = Wind()
    wind.accumulate(light, scene)
    wind.accumulate(heavy, scene)
    _check(
        "wind acceleration scales with inverse mass",
        float(light.acc[0, 0]) > float(heavy.acc[0, 0]) * 5,
    )


def test_ground_bounce() -> None:
    scene = Scene(gravity=-20.0, restitution=0.6, friction=0.2, ground=True, drag_linear=0.0, drag_quad=0.0)
    engine = PhysicsEngine(scene)
    engine.emitter.rate_accum = 0
    engine.scene.rate = 0
    engine.scene.emission = "continuous"
    _spawn_one(engine.store, y=2.0, vy=-1.0, dt=1 / 90)
    for _ in range(180):
        engine.step(1 / 90)
    _check("ground keeps the particle above the plane", float(engine.store.pos[0, 1]) >= -1e-6)
    _check("captured or rebounding vertical speed is non-negative", float(engine.store.vel[0, 1]) >= -1e-6)


def test_box_contains() -> None:
    scene = Scene(gravity=0.0, box=True, ground=False, restitution=0.8, friction=0.0, drag_linear=0.0)
    engine = PhysicsEngine(scene)
    engine.scene.rate = 0
    _spawn_one(engine.store, x=0.0, y=6.0, vx=40.0, dt=1 / 90)
    for _ in range(90):
        engine.step(1 / 90)
    x = float(engine.store.pos[0, 0])
    _check("box walls contain x", -8.0 - 1e-6 <= x <= 8.0 + 1e-6)


def test_sphere_pushes_out() -> None:
    store = ParticleStore(4)
    scene = Scene(sphere=True, restitution=0.5, friction=0.0)
    scene.sphere_center[:] = (0.0, 3.0, 0.0)
    scene.sphere_radius = 1.5
    _spawn_one(store, x=0.1, y=3.0, z=0.0, dt=1 / 60)
    resolve_sphere(store, scene, 1 / 60)
    rel = store.pos[0] - scene.sphere_center
    _check("sphere pushes an interior point onto the shell", abs(float(np.linalg.norm(rel)) - 1.5) < 1e-6)


def test_emitter_and_recycle() -> None:
    store = ParticleStore(32)
    scene = Scene(rate=100.0, emission="continuous", life=(0.4, 0.4), shape="point")
    emitter = EmitterManager(np.random.default_rng(0))
    emitter.update(store, scene, 0.5)
    _check("continuous emitter respects the rate accumulator", store.count == 50 or store.count == 32)
    # 0.5 * 100 = 50, capacity 32, so the pool recycles down to 32.
    _check("pool never exceeds capacity", store.count <= store.capacity)
    store.age[: store.count] = store.life[: store.count]
    store.kill_dead()
    _check("expired particles are removed", store.count == 0)


def test_engine_burst_dies() -> None:
    scene = Scene(emission="burst", burst_count=25, burst_every=10.0, life=(0.2, 0.2), gravity=0.0, ground=False)
    engine = PhysicsEngine(scene)
    engine.step(1 / 60)
    born = engine.store.count
    _check("a primed burst emits immediately", born == 25)
    for _ in range(30):
        engine.step(1 / 60)
    _check("short-lived particles are recycled", engine.store.count == 0)


def _spawn_one(
    store: ParticleStore,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
    mass: float = 1.0,
    dt: float = 1 / 60,
) -> None:
    store.spawn_batch(
        np.array([[x, y, z]]),
        np.array([[vx, vy, vz]]),
        np.array([mass]),
        np.array([10.0]),
        np.array([0.1]),
        np.ones((1, 4)),
        np.ones((1, 4)),
        dt,
    )


def main() -> None:
    print("Aether self-test")
    test_euler_gravity()
    test_verlet_falls()
    test_drag_slows()
    test_heavy_wind()
    test_ground_bounce()
    test_box_contains()
    test_sphere_pushes_out()
    test_emitter_and_recycle()
    test_engine_burst_dies()
    # Silence unused-import warnings in case a checker imports these.
    assert Gravity is not None and resolve_ground is not None and resolve_box is not None
    print("all checks passed")


if __name__ == "__main__":
    main()
