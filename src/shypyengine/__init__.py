"""ShyPyEngine: a modular computational physics engine.

Sub-packages
------------
core            Shared constants, base classes, and configuration.
electrostatics  Coulomb's law, electric fields/potentials, Laplace/Poisson solver.
brownian        Random walk, Gaussian steps, MSD, diffusion equation.
driftdiffusion  Combines electrostatics (drift) and brownian (diffusion).
visualization   Shared plotting utilities.
utils           Shared, dependency-free helper functions.

No simulation logic is implemented yet. This package currently exists as
scaffolding — see ROADMAP.md for the implementation plan.
"""

__version__ = "0.0.0"
