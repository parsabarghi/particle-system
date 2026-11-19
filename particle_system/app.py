"""Interactive Pygame front end.

Run from the unzipped folder:

    python -m particle_system

Left drag spawns, right drag attracts, Shift+right drag repels.
"""

from __future__ import annotations

import numpy as np
import pygame

from particle_system.physics.engine import PhysicsEngine
from particle_system.presets import PRESET_KEYS, PRESETS
from particle_system.render.pygame_view import PygameRenderer
from particle_system.scene import Scene


HELP = "1-8 presets   Tab 2D/3D   E integrator   Space pause   left spawn   right attract   shift+right repel   Esc quit"


def _pick_2d(sx: float, sy: float, width: int, height: int) -> np.ndarray:
    scale = min(width / 20.0, height / 15.2)
    ox, oy = width / 2.0, height * 0.86
    return np.array([(sx - ox) / scale, (oy - sy) / scale, 0.0], dtype=np.float64)


def _pick_3d(renderer: PygameRenderer, sx: float, sy: float, width: int, height: int, plane_y: float) -> np.ndarray:
    eye, forward, right, up = renderer._camera()
    focal = min(width, height) * 1.05
    ndc_x = (sx - width / 2.0) / focal
    ndc_y = (height / 2.0 - sy) / focal
    direction = forward + right * ndc_x + up * ndc_y
    direction = direction / (float(np.linalg.norm(direction)) or 1.0)
    if abs(float(direction[1])) < 1e-4:
        return np.array([0.0, plane_y, 0.0])
    t = (plane_y - float(eye[1])) / float(direction[1])
    if t <= 0:
        return np.array([0.0, plane_y, 0.0])
    point = eye + direction * t
    point[0] = float(np.clip(point[0], -9, 9))
    point[1] = plane_y
    point[2] = float(np.clip(point[2], -9, 9))
    return point


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((1200, 720), pygame.RESIZABLE)
    pygame.display.set_caption("Aether — Particle Dynamics Laboratory")
    clock = pygame.time.Clock()
    renderer = PygameRenderer()
    scene: Scene = PRESETS["fountain"]()
    engine = PhysicsEngine(scene)
    engine.warm(48)
    paused = False
    accum = 0.0
    step = 1.0 / 90.0
    running = True

    while running:
        frame_dt = min(clock.tick(60) / 1000.0, 0.05)
        width, height = screen.get_size()
        shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_e:
                    scene.integrator = "verlet" if scene.integrator == "euler" else "euler"
                elif event.key == pygame.K_TAB:
                    scene.dimension = "2d" if scene.dimension == "2d" else "2d"
                # elif event.key == pygame.K_TAB:
                #     scene.dimension = "3d" if scene.dimension == "2d" else "3d"
                elif pygame.K_1 <= event.key <= pygame.K_8:
                    index = event.key - pygame.K_1
                    scene = PRESETS[PRESET_KEYS[index]]()
                    engine.scene = scene
                    engine.reset()
                    engine.warm(48)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
                _press(engine, renderer, event.pos, width, height, event.button, shift, spawn_burst=True)
            elif event.type == pygame.MOUSEBUTTONUP and event.button in (1, 3):
                engine.pointer_down = False
            elif event.type == pygame.MOUSEMOTION and engine.pointer_down:
                _move(engine, renderer, event.pos, width, height)

        if not paused:
            accum += frame_dt
            guard = 0
            while accum >= step and guard < 5:
                engine.step(step)
                accum -= step
                guard += 1
            if guard == 5:
                accum = 0.0
        elif scene.dimension == "3d":
            renderer.yaw += frame_dt * 0.16
        if scene.dimension == "3d" and not engine.pointer_down and not paused:
            renderer.yaw += frame_dt * 0.16

        renderer.draw(screen, engine, clock.get_fps(), HELP)
        pygame.display.flip()

    pygame.quit()


def _pointer_mode(button: int, shift: bool) -> str:
    if button == 3:
        return "repel" if shift else "attract"
    return "spawn"


def _world(engine: PhysicsEngine, renderer: PygameRenderer, pos: tuple[int, int], width: int, height: int) -> np.ndarray:
    plane_y = max(1.2, float(engine.scene.origin[1]))
    if engine.scene.dimension == "2d":
        return _pick_2d(pos[0], pos[1], width, height)
    return _pick_3d(renderer, pos[0], pos[1], width, height, plane_y)


def _press(
    engine: PhysicsEngine,
    renderer: PygameRenderer,
    pos: tuple[int, int],
    width: int,
    height: int,
    button: int,
    shift: bool,
    spawn_burst: bool,
) -> None:
    world = _world(engine, renderer, pos, width, height)
    engine.pointer[:] = world
    engine.pointer_mode = _pointer_mode(button, shift)
    engine.pointer_down = True
    if spawn_burst and engine.pointer_mode == "spawn":
        engine.emitter.emit_at(engine.store, engine.scene, 18, world, 1.0 / 90.0)


def _move(engine: PhysicsEngine, renderer: PygameRenderer, pos: tuple[int, int], width: int, height: int) -> None:
    engine.pointer[:] = _world(engine, renderer, pos, width, height)
