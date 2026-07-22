# ShyPyEngine

A computational physics engine written in Python, focused on electrostatics,
Brownian motion, and drift-diffusion simulation.

ShyPyEngine implements two independent physics modules, electric field
simulation and Brownian motion, and combines them into a single
drift-diffusion simulation engine that models how charged particles move
under an electric field in the presence of thermal noise.

**Status:** Pre-alpha. Phase 0 (repository infrastructure), Phase 1
(electrostatics), and Phase 2 (Brownian motion) are complete. Phase 3
(drift-diffusion integration) is next. See [ROADMAP.md](ROADMAP.md) for
the full plan and [CHANGELOG.md](CHANGELOG.md) for what has actually
shipped.

**Maintainer:** This project is developed and maintained by Shaurya Sharma.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Example Code](#example-code)
- [Features](#features)
- [Numerical Methods](#numerical-methods)
- [Package API](#package-api)
- [Repository Structure](#repository-structure)
- [Current Limitations](#current-limitations)
- [Version Compatibility](#version-compatibility)
- [Development Workflow](#development-workflow)
- [Team](#team)
- [Contributing](#contributing)
- [Citation](#citation)
- [Roadmap](#roadmap)
- [License](#license)

## Installation

Requires Python 3.11 or later.

```bash
git clone https://github.com/Shasha-2025/ShyPyEngine.git
cd ShyPyEngine

python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

Verify the install:

```bash
python -c "import shypyengine; print(shypyengine.__name__)"
pytest
```

## Quick Start

```python
from shypyengine.brownian import RandomWalk, mean_squared_displacement

# 2D Brownian motion, diffusion coefficient D = 1.0, time step dt = 0.01
walk = RandomWalk(n_dim=2, diffusion_coefficient=1.0, dt=0.01, seed=0)
trajectories = walk.generate(n_steps=1000, n_particles=100)

msd = mean_squared_displacement(trajectories)
print(msd[-1])
```

Run the bundled demo, which generates trajectories, computes MSD, recovers
the diffusion coefficient, and solves the 1D diffusion equation:

```bash
python examples/brownian_demo.py
```

It writes `brownian_trajectories.png`, `brownian_msd.png`, and
`brownian_diffusion_profiles.png` to the working directory.

Electrostatics quick start:

```python
import numpy as np
from shypyengine.electrostatics import PointCharge, electric_field, electric_potential

charge = PointCharge(charge=1e-6, position=np.array([0.0, 0.0]))
point = np.array([2.0, 0.0])

print(electric_field([charge], point))      # [E_x, E_y], in V/m
print(electric_potential([charge], point))  # V, in volts
```

Run the electrostatics demo, which validates the point-charge functions
against Coulomb's law, solves the Poisson/Laplace equation for a parallel-
plate capacitor, and plots a dipole's field and potential:

```bash
python examples/electrostatics_demo.py
```

It writes `electrostatics_dipole.png`, `electrostatics_equipotentials.png`,
and `electrostatics_parallel_plates.png` to the working directory.

## Example Code

Estimate a diffusion coefficient from simulated trajectories and compare it
to the value used to generate them:

```python
from shypyengine.brownian import RandomWalk, mean_squared_displacement, estimate_diffusion_coefficient

true_d = 2.5
walk = RandomWalk(n_dim=3, diffusion_coefficient=true_d, dt=0.01, seed=7)
trajectories = walk.generate(n_steps=2000, n_particles=2000)
time = walk.time_axis(n_steps=2000)

msd = mean_squared_displacement(trajectories)
estimated_d = estimate_diffusion_coefficient(time, msd, n_dim=3)

print(f"true D = {true_d}, estimated D = {estimated_d:.4f}")
```

Solve the 1D diffusion equation for a Gaussian pulse:

```python
from shypyengine.brownian import DiffusionSolver1D

solver = DiffusionSolver1D(diffusion_coefficient=1.0, dx=0.1, dt=0.001, n_grid=101)
c0 = solver.initial_gaussian(mean=solver.x[50], sigma=0.3)
history = solver.solve(c0, n_steps=500)  # shape (501, 101)
```

Solve the Poisson equation for a parallel-plate capacitor and compare the
resulting field to the analytical result:

```python
import numpy as np
from shypyengine.electrostatics import BoundaryCondition, PoissonSolver2D, electric_field_from_potential

v0 = 10.0
nx, ny, dx, dy = 61, 21, 0.05, 0.05
solver = PoissonSolver2D(nx=nx, ny=ny, dx=dx, dy=dy)
rho = np.zeros((ny, nx))

V = solver.solve(
    rho,
    left=BoundaryCondition("dirichlet", v0),
    right=BoundaryCondition("dirichlet", -v0),
    top=BoundaryCondition("neumann"),
    bottom=BoundaryCondition("neumann"),
)
Ex, Ey = electric_field_from_potential(V, dx, dy)

plate_separation = (nx - 1) * dx
print(f"analytical E = {2 * v0 / plate_separation:.4f}, simulated E = {Ex[1:-1, 1:-1].mean():.4f}")
```

## Features

Implemented (Phase 0, Phase 1, and Phase 2):

- Point-charge electrostatics: Coulomb's law, and electric field/potential
  via superposition, fully vectorized over arbitrary point grids
- Finite-difference solver for the 2D electrostatic Poisson/Laplace
  equation, with Dirichlet and zero-flux Neumann boundary conditions
- Equipotential line extraction from a potential grid
- 1D, 2D, and 3D Brownian motion via a Gaussian random walk generator
- Mean squared displacement (MSD) calculation over an ensemble of particles
- Diffusion coefficient estimation from simulated trajectories
- Explicit finite-difference solver for the 1D diffusion equation
- Matplotlib plotting helpers for field lines, potential maps,
  equipotentials, trajectories, MSD curves, and diffusion profiles
- GitHub Actions CI running lint, tests, and a package build check on every
  push and pull request

Not yet implemented (see [Roadmap](#roadmap)):

- Drift-diffusion integration combining electrostatics and Brownian motion
- Interactive Plotly visualization
- 3D grid-based Poisson/Laplace solver (only the point-charge functions
  support 3D directly today)
- GUI, AI assistant, and gamification layers

## Numerical Methods

See also [docs/electrostatics.md](docs/electrostatics.md) for the full
electrostatics background (Coulomb's law, field/potential superposition,
the Poisson/Laplace equation, boundary conditions, and analytical
validation cases). The summary below focuses on Brownian motion and the
1D diffusion solver.

### Brownian motion

Trajectories are generated with a discrete-time Gaussian random walk. At
each step, the displacement along every spatial axis is drawn independently
from a normal distribution with mean 0 and variance `2 * D * dt`, where `D`
is the diffusion coefficient and `dt` is the time step. This discretization
is the direct stochastic analogue of the diffusion equation
`∂c/∂t = D * ∇²c`, so trajectories generated this way converge to that PDE
in the continuum limit.

Mean squared displacement is computed as `MSD(t) = <|x(t) - x(0)|²>`,
averaged over the particle ensemble. For free diffusion in `n` dimensions
the analytical result is `MSD(t) = 2 * n * D * t`, which is a straight line
through the origin. `estimate_diffusion_coefficient` recovers `D` with an
ordinary least squares fit of that line, forced through the origin since
`MSD(0) = 0` by construction.

### Diffusion equation solver

`DiffusionSolver1D` solves `∂c/∂t = D * ∂²c/∂x²` on a uniform 1D grid using
an explicit forward-time, centered-space (FTCS) finite-difference scheme:

```
c[i, n+1] = c[i, n] + alpha * (c[i+1, n] - 2*c[i, n] + c[i-1, n])
alpha = D * dt / dx^2
```

**Boundary conditions:** zero-flux (reflective, Neumann) at both ends of
the domain. Mass is conserved to within floating-point and truncation
error, which the test suite checks directly.

**Stability:** the explicit scheme is conditionally stable and requires
`alpha = D * dt / dx² <= 0.5`. `DiffusionSolver1D` checks this at
construction time and raises `InvalidParameterError` if it is violated,
rather than silently producing an unstable, oscillating solution.

**Complexity:** each time step costs `O(n_grid)` (a single vectorized pass
over the grid), so solving for `n_steps` costs `O(n_steps * n_grid)` time
and `O(n_steps * n_grid)` memory, since the full time history is stored.
The random walk generator costs `O(n_steps * n_particles * n_dim)` for both
time and memory, for the same reason.

### Validation methodology

The random walk implementation is validated by generating a large ensemble
of trajectories (2000 particles, 2000 steps) at a known diffusion
coefficient, computing MSD, and fitting `D` back out. The test suite checks
that the recovered value is within 10% of the true value in 1D, 2D, and 3D.
The diffusion solver is validated by checking that total mass is conserved
under reflective boundaries and that an initially concentrated Gaussian
pulse flattens out over time, which is the expected qualitative behavior of
a diffusion process.

### Mathematical assumptions

- Particles do not interact with each other (no excluded volume, no
  particle-particle forces).
- The diffusion coefficient is constant in space and time.
- The random walk uses Gaussian, uncorrelated steps (no memory between
  steps).
- The 1D diffusion solver assumes a fixed grid spacing and reflective
  boundaries; absorbing or periodic boundaries are not implemented.

## Package API

```
shypyengine.core
    BOLTZMANN_CONSTANT, COULOMB_CONSTANT, VACUUM_PERMITTIVITY
    ShyPyEngineError, InvalidParameterError

shypyengine.electrostatics
    PointCharge(charge, position)
        .n_dim -> int

    coulomb_force(q1, q2, r1, r2, k=COULOMB_CONSTANT) -> ndarray
    electric_field(charges, points, k=COULOMB_CONSTANT) -> ndarray
    electric_potential(charges, points, k=COULOMB_CONSTANT) -> ndarray

    BoundaryCondition(kind, value=0.0)  # kind: "dirichlet" | "neumann"
    PoissonSolver2D(nx, ny, dx, dy)
        .solve(rho, left, right, top, bottom, epsilon0=VACUUM_PERMITTIVITY) -> ndarray
        .x, .y, .X, .Y

    electric_field_from_potential(V, dx, dy) -> (ndarray, ndarray)
    extract_equipotential_lines(X, Y, V, levels) -> dict[float, list[ndarray]]

shypyengine.brownian
    RandomWalk(n_dim, diffusion_coefficient, dt, seed=None)
        .generate(n_steps, n_particles=1, initial_positions=None) -> ndarray
        .time_axis(n_steps) -> ndarray
        .step_std -> float

    mean_squared_displacement(trajectories) -> ndarray
    estimate_diffusion_coefficient(time, msd, n_dim) -> float

    DiffusionSolver1D(diffusion_coefficient, dx, dt, n_grid)
        .initial_gaussian(mean, sigma, amplitude=1.0) -> ndarray
        .step(c) -> ndarray
        .solve(c0, n_steps) -> ndarray

shypyengine.visualization
    plot_field_lines(X, Y, Ex, Ey, ax=None, density=1.2)
    plot_potential_map(X, Y, V, ax=None, cmap="RdBu_r")
    plot_equipotentials(equipotential_lines, ax=None)
    plot_trajectories_2d(trajectories, ax=None)
    plot_msd(time, msd, ax=None)
    plot_diffusion_profiles(x, history, n_curves=5, ax=None)
```

Every public function and class has a NumPy-style docstring; run
`help(shypyengine.electrostatics.PoissonSolver2D)` or
`help(shypyengine.brownian.RandomWalk)` for full parameter documentation.

## Repository Structure

```
ShyPyEngine/
├── src/shypyengine/       # Installable package (all engine source code)
│   ├── core/               # Shared base classes, constants, exceptions
│   ├── electrostatics/     # Electrostatics module (Phase 1)
│   ├── brownian/           # Brownian motion module
│   ├── driftdiffusion/     # Drift-diffusion integration (not yet implemented)
│   ├── visualization/      # Plotting helpers
│   └── utils/               # Shared utilities (I/O, validation, math helpers)
├── tests/                  # Unit and integration tests (pytest)
├── docs/                   # Design docs, numerical methods, architecture notes
├── examples/                # Runnable example scripts
├── notebooks/               # Jupyter notebooks for exploration/demos
├── assets/                  # Images, diagrams, sample data
├── scripts/                  # Dev/maintenance scripts
├── .github/                  # Issue templates, PR template, CI workflow
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── ROADMAP.md
├── CODE_OF_CONDUCT.md
├── LICENSE
├── requirements.txt
└── pyproject.toml
```

See [docs/architecture.md](docs/architecture.md) for how the modules relate
to each other.

## Current Limitations

- Only a 2D grid Poisson/Laplace solver exists; 3D grid-based
  electrostatics is not implemented (though the point-charge field and
  potential functions do support 3D directly).
- No dielectric materials (spatially varying permittivity) or conductors
  with induced (as opposed to prescribed) charge distributions.
- The Poisson/Laplace solver only supports Dirichlet and zero-flux
  Neumann boundary conditions; general nonzero-flux Neumann and periodic
  boundaries are not implemented.
- Only 1D diffusion has a PDE solver; 2D/3D diffusion is currently only
  available through the particle-based random walk, not a grid solver.
- No drift-diffusion integration yet (Phase 3): electrostatics and
  Brownian motion are implemented as independent modules and are not yet
  combined.
- No absorbing or periodic boundary conditions for the diffusion solver,
  only reflective.
- No interacting-particle physics (collisions, excluded volume, or
  particle-particle forces).
- No GPU or parallel execution; simulations run single-threaded with NumPy.
- Test coverage covers the `brownian`, `electrostatics`, and
  `visualization` modules; `core` and the unimplemented modules do not
  yet have dedicated tests.

## Version Compatibility

| ShyPyEngine | Python | NumPy | Notes |
|---|---|---|---|
| 0.1.0 (current) | 3.11, 3.12 | >=1.26 | Tested with NumPy 2.x; `np.trapezoid` is used where available with a fallback to `np.trapz` for older NumPy. |

CI (`.github/workflows/ci.yml`) runs the test suite against Python 3.11 and
3.12 on every push and pull request.

## Development Workflow

ShyPyEngine follows a protected `main` / integration `develop` / feature
branch workflow:

- `main`: protected, always stable. Only updated via merge from `develop`.
- `develop`: integration branch where completed features land before a
  release.
- Feature branches: one per unit of work, branched from `develop` and
  merged back via pull request, for example `feature/github-setup`,
  `feature/brownian-motion`, `feature/electrostatics`,
  `feature/drift-diffusion`, `feature/visualization`, `feature/testing`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for full branching, commit, and PR
conventions.

## Team

ShyPyEngine was originally planned as a five-person team project. It is now
developed and maintained by one person:

| Role | Focus |
|---|---|
| M4 | Sole maintainer: physics implementation, testing, and infrastructure |

## Contributing

This is currently a solo project, but contributions, issues, and discussion
are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before opening a pull request.

## Citation

If you use ShyPyEngine in academic or research work, cite the repository
directly:

```bibtex
@software{shypyengine,
  title  = {ShyPyEngine: A Computational Physics Engine for Electrostatics,
            Brownian Motion, and Drift-Diffusion Simulation},
  author = {{ShyPyEngine Contributors}},
  year   = {2026},
  url    = {https://github.com/Shasha-2025/ShyPyEngine}
}
```

There is no associated paper at this stage of the project.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full phase-by-phase plan, from
repository infrastructure through electrostatics, Brownian motion,
drift-diffusion, visualization, testing, a GUI, an AI assistant, and
gamification.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
