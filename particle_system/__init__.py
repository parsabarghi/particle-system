"""Aether — a modular CPU particle dynamics system for graphics coursework.

Subsystems
----------
model.store.ParticleStore
    Structure-of-arrays buffer. Birth, aging, and recycling.
physics.engine.PhysicsEngine
    Fixed-step orchestrator.
physics.forces / physics.integrators / physics.colliders
    Environmental forces, Euler or Verlet, analytical contacts.
emit.manager.EmitterManager
    Continuous and burst emission.
render.pygame_view.PygameRenderer
    Software 2D / 3D view. Drawing never writes simulation state.
"""

__version__ = "1.0.0"
