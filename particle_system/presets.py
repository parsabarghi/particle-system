"""Named scenes. Each one is a full Scene so switching presets cannot leak state."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from particle_system.scene import Scene


def _color(*rgba: float) -> np.ndarray:
    return np.array(rgba, dtype=np.float64)


def fountain() -> Scene:
    return Scene()


def embers() -> Scene:
    scene = Scene(
        gravity=-3.4,
        drag_linear=0.7,
        drag_quad=0.1,
        restitution=0.18,
        friction=0.45,
        rate=280,
        life=(1.3, 2.5),
        size=(0.08, 0.22),
        speed=4.4,
        spread=1.5,
        cone=0.75,
        shape="disk",
    )
    scene.wind[:] = (1.6, 0.0, 0.0)
    scene.color0[:] = _color(1, 0.78, 0.38, 1)
    scene.color1[:] = _color(0.55, 0.08, 0.03, 0)
    return scene


def snow() -> Scene:
    scene = Scene(
        gravity=-3.2,
        drag_linear=1.15,
        drag_quad=0.15,
        restitution=0.05,
        friction=0.92,
        box=True,
        rate=160,
        life=(7.0, 10.0),
        size=(0.06, 0.12),
        mass=(0.35, 0.7),
        speed=0.8,
        shape="rain",
        velocity_mode="drift",
    )
    scene.wind[:] = (1.8, 0.0, 0.0)
    scene.color0[:] = _color(0.96, 0.97, 1, 0.92)
    scene.color1[:] = _color(0.72, 0.8, 0.9, 0)
    return scene


def vortex() -> Scene:
    scene = Scene(
        gravity=0.0,
        drag_linear=0.08,
        drag_quad=0.0,
        vortex=42.0,
        ground=False,
        rate=220,
        life=(4.0, 7.0),
        size=(0.07, 0.16),
        speed=1.1,
        spread=4.4,
        cone=1.5,
        shape="disk",
    )
    scene.origin[:] = (0.0, 5.0, 0.0)
    scene.color0[:] = _color(0.5, 0.82, 1, 0.9)
    scene.color1[:] = _color(0.95, 0.7, 0.32, 0)
    return scene


def rain() -> Scene:
    scene = Scene(
        gravity=-26.0,
        drag_linear=0.04,
        drag_quad=0.01,
        restitution=0.02,
        friction=0.15,
        box=True,
        rate=480,
        life=(0.65, 1.05),
        size=(0.045, 0.08),
        speed=1.5,
        shape="rain",
        velocity_mode="drift",
    )
    scene.color0[:] = _color(0.62, 0.78, 1, 0.75)
    scene.color1[:] = _color(0.35, 0.5, 0.75, 0)
    return scene


def burst() -> Scene:
    scene = Scene(
        gravity=-11.0,
        drag_linear=0.04,
        drag_quad=0.01,
        restitution=0.78,
        friction=0.12,
        box=True,
        emission="burst",
        burst_count=700,
        burst_every=1.65,
        life=(1.5, 2.3),
        speed=9.5,
        shape="point",
        velocity_mode="sphere",
    )
    scene.origin[:] = (0.0, 5.5, 0.0)
    scene.color0[:] = _color(1, 0.94, 0.82, 1)
    scene.color1[:] = _color(0.92, 0.28, 0.42, 0)
    return scene


def smoke() -> Scene:
    scene = Scene(
        gravity=-0.55,
        drag_linear=0.5,
        drag_quad=0.05,
        ground=False,
        rate=64,
        life=(3.6, 5.8),
        size=(0.5, 0.95),
        mass=(0.3, 0.55),
        speed=1.7,
        spread=0.7,
        cone=0.4,
        shape="disk",
    )
    scene.color0[:] = _color(0.86, 0.84, 0.78, 0.16)
    scene.color1[:] = _color(0.5, 0.52, 0.54, 0)
    return scene


def well() -> Scene:
    scene = Scene(
        gravity=0.0,
        drag_linear=0.015,
        drag_quad=0.0,
        well=78.0,
        ground=False,
        rate=110,
        life=(8.0, 12.0),
        size=(0.07, 0.14),
        speed=4.15,
        spread=3.7,
        shape="ring",
        velocity_mode="tangential",
    )
    scene.origin[:] = (0.0, 5.0, 0.0)
    scene.well_center[:] = (0.0, 5.0, 0.0)
    scene.color0[:] = _color(0.98, 0.9, 0.7, 0.95)
    scene.color1[:] = _color(0.35, 0.62, 0.82, 0)
    return scene


PRESETS: dict[str, Callable[[], Scene]] = {
    "fountain": fountain,
    "embers": embers,
    "snow": snow,
    "vortex": vortex,
    "rain": rain,
    "burst": burst,
    "smoke": smoke,
    "well": well,
}

PRESET_KEYS = list(PRESETS)
