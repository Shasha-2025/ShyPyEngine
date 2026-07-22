# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- Repository scaffolding, folder hierarchy, and core documentation.
- Python packaging (`pyproject.toml`, `requirements.txt`).
- Numerical methods documentation scaffold.
- GitHub Actions CI workflow (`.github/workflows/ci.yml`) running ruff lint,
  pytest with coverage, and a package build check on every push/PR.
- Issue templates (bug report, feature request, documentation) and a pull
  request template.
- `shypyengine.core`: shared physical constants and exception types.
- `shypyengine.brownian`: `RandomWalk` (1D/2D/3D Gaussian random walk),
  `mean_squared_displacement`, `estimate_diffusion_coefficient`, and
  `DiffusionSolver1D` (explicit FTCS finite-difference solver).
- `shypyengine.electrostatics`: `PointCharge`, `coulomb_force`,
  `electric_field`, and `electric_potential` (Coulomb's law and
  superposition, vectorized over arbitrary point grids); `PoissonSolver2D`
  and `BoundaryCondition` (5-point finite-difference Poisson/Laplace
  solver with Dirichlet and zero-flux Neumann boundary conditions, solved
  directly via `scipy.sparse.linalg.spsolve`); `electric_field_from_potential`;
  `extract_equipotential_lines` (marching-squares extraction via
  `contourpy`).
- `shypyengine.visualization`: static Matplotlib plots for trajectories, MSD,
  diffusion profiles, field lines, potential maps, and equipotential overlays.
- `tests/test_brownian.py`: unit tests plus validation of recovered
  diffusion coefficients against the analytical `MSD(t) = 2*n_dim*D*t`
  scaling, in 1D, 2D, and 3D.
- `tests/test_electrostatics.py`: unit tests plus validation against three
  analytical configurations (point charge, an approximated infinite line
  charge, and a parallel-plate capacitor), and a method-of-manufactured-
  solutions test confirming the Poisson solver's error scales as O(dx²)
  under grid refinement.
- `tests/test_visualization.py` and `tests/test_electrostatics_visualization.py`:
  smoke tests for all plotting helpers.
- `examples/brownian_demo.py` and `examples/electrostatics_demo.py`:
  runnable end-to-end demos.
- `docs/numerical_methods.md`: full derivations (Poisson/Laplace equation
  from Gauss's law, the finite-difference stencil via Taylor expansion,
  von Neumann stability analysis for the diffusion solver, and the
  random-walk-to-diffusion-equation derivation) plus an error analysis
  section (truncation vs. discretization error, floating-point precision,
  boundary condition accuracy).
- `docs/electrostatics.md`: usage-oriented Phase 1 physics/numerics
  background, cross-linked with `docs/numerical_methods.md`.
- `docs/architecture.md`: documentation map explaining how the docs relate.

### Changed
- Project restructured for solo maintenance (originally scoped for a
  5-person team); README and ROADMAP updated accordingly.
- Removed all individual maintainer naming from documentation and package
  metadata; the project is now described purely as a solo-maintained
  effort with no personal identifiers.
- Phase 0, Phase 1, and Phase 2 marked complete in ROADMAP.md.
- Rewrote README.md with Installation, Quick Start, Example Code, Features,
  Numerical Methods, Package API, Current Limitations, Version Compatibility,
  Citation, and Roadmap sections; Phase 1 additions folded into each.
- Standardized terminology across all docs: "Brownian motion" in prose,
  "Brownian Motion" in headings only; same rule for "Electrostatics"; "AI
  assistant" used consistently instead of "AI research assistant".
- Expanded docs/architecture.md with a solver architecture section and a
  documentation map.
- Docstrings across `brownian`, `electrostatics`, and `visualization`
  standardized to full NumPy style, including `Raises` sections for every
  function/method that raises `InvalidParameterError`.
- `shypyengine.visualization` now raises `InvalidParameterError` (not a
  bare `ValueError`) on invalid input, for consistency with the rest of
  the exception hierarchy.
- `contourpy` added as an explicit direct dependency in `pyproject.toml`
  and `requirements.txt` (previously only an implicit transitive
  dependency of `matplotlib`).
- Added `black` and `isort` as dev dependencies, with matching config
  sections in `pyproject.toml` (99-character line length, black-compatible
  isort profile); CI now checks formatting and import order in addition
  to `ruff` lint. `flake8` removed from dev dependencies as redundant with
  `ruff`.

### Fixed
- `PoissonSolver2D.solve` now explicitly rejects an all-Neumann boundary
  configuration (`InvalidParameterError`) instead of silently returning
  `NaN` from the resulting singular linear system.

