"""Software renderer. Pygame draws circles; it does not integrate the simulation.

2D is an orthographic Y-up view. 3D is a CPU perspective projection of the same
store. No shader, no GPU buffer.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from particle_system.physics.engine import PhysicsEngine
from particle_system.scene import Scene

INK = (16, 20, 24)
MIST = (147, 160, 170)
BRASS = (224, 164, 90)
PAPER = (243, 234, 220)


class PygameRenderer:
    def __init__(self) -> None:
        self.yaw = 0.72
        self.pitch = 0.38
        self.distance = 22.0
        self.target = np.array([0.0, 4.6, 0.0])
        self.font: pygame.font.Font | None = None

    def draw(
        self,
        screen: pygame.Surface,
        engine: PhysicsEngine,
        fps: float,
        help_line: str,
    ) -> None:
        scene = engine.scene
        screen.fill(INK)
        width, height = screen.get_size()
        if self.font is None:
            self.font = pygame.font.SysFont("sans", 18)
        if scene.dimension == "2d":
            self._draw_2d(screen, engine, width, height)
        else:
            self._draw_3d(screen, engine, width, height)
        label = self.font.render(
            f"{fps:5.1f} fps    {engine.store.count:5d} alive    {scene.integrator}    {scene.dimension}",
            True,
            PAPER,
        )
        screen.blit(label, (16, 14))
        hint = self.font.render(help_line, True, MIST)
        screen.blit(hint, (16, height - 32))

    def _scale_2d(self, width: int, height: int) -> tuple[float, float, float]:
        scale = min(width / 20.0, height / 15.2)
        return scale, width / 2.0, height * 0.86

    def _to_screen_2d(self, x: float, y: float, width: int, height: int) -> tuple[int, int]:
        scale, ox, oy = self._scale_2d(width, height)
        return int(ox + x * scale), int(oy - y * scale)

    def _draw_2d(self, screen: pygame.Surface, engine: PhysicsEngine, width: int, height: int) -> None:
        scene = engine.scene
        for x in range(-8, 9, 2):
            pygame.draw.line(
                screen,
                _dim(MIST, 70),
                self._to_screen_2d(x, 0, width, height),
                self._to_screen_2d(x, 12, width, height),
                1,
            )
        for y in range(0, 13, 2):
            pygame.draw.line(
                screen,
                _dim(MIST, 70),
                self._to_screen_2d(-8, y, width, height),
                self._to_screen_2d(8, y, width, height),
                1,
            )
        if scene.ground:
            pygame.draw.line(
                screen,
                BRASS,
                self._to_screen_2d(-10, 0, width, height),
                self._to_screen_2d(10, 0, width, height),
                2,
            )
        if scene.box:
            rect_a = self._to_screen_2d(-8, 12, width, height)
            rect_b = self._to_screen_2d(8, 0, width, height)
            pygame.draw.rect(screen, _dim(PAPER, 90), pygame.Rect(rect_a, (rect_b[0] - rect_a[0], rect_b[1] - rect_a[1])), 1)
        if scene.sphere:
            center = self._to_screen_2d(float(scene.sphere_center[0]), float(scene.sphere_center[1]), width, height)
            scale, _, _ = self._scale_2d(width, height)
            pygame.draw.circle(screen, BRASS, center, max(2, int(scene.sphere_radius * scale)), 1)
        self._paint(screen, engine, width, height, None)

    def _camera(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        sy, cy = math.sin(self.yaw), math.cos(self.yaw)
        eye = self.target + np.array([self.distance * cp * sy, self.distance * sp, self.distance * cp * cy])
        forward = _unit(self.target - eye)
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(right) < 1e-4:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = _unit(right)
        up = _unit(np.cross(right, forward))
        return eye, forward, right, up

    def _project(
        self,
        points: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eye, forward, right, up = self._camera()
        rel = points - eye
        cx = rel @ right
        cy = rel @ up
        cz = rel @ forward
        focal = min(width, height) * 1.05
        sx = width / 2.0 + (cx / np.maximum(cz, 1e-3)) * focal
        sy = height / 2.0 - (cy / np.maximum(cz, 1e-3)) * focal
        return sx, sy, cz

    def _draw_3d(self, screen: pygame.Surface, engine: PhysicsEngine, width: int, height: int) -> None:
        scene = engine.scene
        ground = []
        for x in range(-8, 9, 2):
            ground.append([[x, 0, -8], [x, 0, 8]])
        for z in range(-8, 9, 2):
            ground.append([[-8, 0, z], [8, 0, z]])
        self._lines(screen, ground, width, height, _dim(MIST, 70))
        if scene.ground:
            self._lines(screen, [[[-8, 0, 0], [8, 0, 0]], [[0, 0, -8], [0, 0, 8]]], width, height, BRASS)
        if scene.box:
            self._lines(screen, _box_edges(), width, height, _dim(PAPER, 90))
        if scene.sphere:
            self._lines(screen, _sphere_loops(scene.sphere_center, scene.sphere_radius), width, height, BRASS)
        self._paint(screen, engine, width, height, self._project)

    def _lines(
        self,
        screen: pygame.Surface,
        segments: list,
        width: int,
        height: int,
        color: tuple[int, int, int],
    ) -> None:
        if not segments:
            return
        pts = np.array(segments, dtype=np.float64).reshape(-1, 3)
        sx, sy, cz = self._project(pts, width, height)
        sx = sx.reshape(-1, 2)
        sy = sy.reshape(-1, 2)
        cz = cz.reshape(-1, 2)
        for i in range(sx.shape[0]):
            if cz[i, 0] < 0.25 or cz[i, 1] < 0.25:
                continue
            pygame.draw.line(screen, color, (int(sx[i, 0]), int(sy[i, 0])), (int(sx[i, 1]), int(sy[i, 1])), 1)

    def _paint(self, screen: pygame.Surface, engine: PhysicsEngine, width: int, height: int, project) -> None:
        store = engine.store
        n = store.count
        if n == 0:
            return
        pos = store.pos[:n]
        age = store.age[:n]
        life = np.maximum(store.life[:n], 1e-4)
        t = np.clip(age / life, 0.0, 1.0)
        u = t * t * (3.0 - 2.0 * t)
        fade = np.ones(n)
        fade = np.where(t < 0.08, t / 0.08, fade)
        fade = np.where(t > 0.7, np.maximum(0.0, (1.0 - t) / 0.3), fade)
        rgba = store.color0[:n] + (store.color1[:n] - store.color0[:n]) * u[:, None]
        rgba[:, 3] *= fade
        if project is None:
            scale, ox, oy = self._scale_2d(width, height)
            sx = ox + pos[:, 0] * scale
            sy = oy - pos[:, 1] * scale
            radius = np.maximum(1.0, store.size[:n] * scale)
        else:
            sx, sy, cz = project(pos, width, height)
            visible = cz > 0.25
            focal = min(width, height) * 1.05
            radius = np.maximum(1.0, store.size[:n] * focal / np.maximum(cz, 0.25))
            sx, sy, rgba, radius = sx[visible], sy[visible], rgba[visible], radius[visible]
        # Draw a capped subset of the largest-on-screen marks if the batch is huge.
        order = np.argsort(radius) if radius.size > 2500 else np.arange(radius.size)
        if order.size > 2500:
            order = order[-2500:]
        for i in order:
            if rgba[i, 3] < 0.02:
                continue
            color = (
                int(np.clip(rgba[i, 0] * 255, 0, 255)),
                int(np.clip(rgba[i, 1] * 255, 0, 255)),
                int(np.clip(rgba[i, 2] * 255, 0, 255)),
            )
            pygame.draw.circle(screen, color, (int(sx[i]), int(sy[i])), int(radius[i]))


def _unit(v: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(v)) or 1.0
    return v / length


def _dim(color: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    return tuple(max(0, c * amount // 255) for c in color)  # type: ignore[return-value]


def _box_edges() -> list[list[list[float]]]:
    c = [
        [-8, 0, -8],
        [8, 0, -8],
        [8, 0, 8],
        [-8, 0, 8],
        [-8, 12, -8],
        [8, 12, -8],
        [8, 12, 8],
        [-8, 12, 8],
    ]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [[c[a], c[b]] for a, b in pairs]


def _sphere_loops(center: np.ndarray, radius: float, steps: int = 28) -> list[list[list[float]]]:
    segments: list[list[list[float]]] = []
    angles = np.linspace(0.0, 2.0 * math.pi, steps, endpoint=False)
    rings = []
    for ang in angles:
        rings.append(
            [
                [center[0] + math.cos(ang) * radius, center[1], center[2] + math.sin(ang) * radius],
                [center[0] + math.cos(ang) * radius, center[1] + math.sin(ang) * radius, center[2]],
                [center[0], center[1] + math.sin(ang) * radius, center[2] + math.cos(ang) * radius],
            ]
        )
    # Connect successive samples on each of the three rings (columns).
    count = len(rings)
    for ring in range(3):
        for i in range(count):
            j = (i + 1) % count
            segments.append([rings[i][ring], rings[j][ring]])
    return segments
