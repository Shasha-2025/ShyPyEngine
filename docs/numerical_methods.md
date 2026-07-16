# Numerical Methods

This document tracks the numerical methods used in ShyPyEngine, the
assumptions behind each, and the validation strategy used to confirm
correctness. It should be updated whenever a method is implemented or
changed (see [CONTRIBUTING.md](../CONTRIBUTING.md)).

## Electrostatics

### Coulomb's Law / Superposition
- **Method:** Direct pairwise summation of point-charge contributions.
- **Complexity:** O(N²) for N charges — acceptable for small N; revisit
  (e.g., tree methods) if scaling becomes a bottleneck.
- **Validation:** Compare field/potential of a single charge and of a dipole
  against closed-form analytical expressions.

### Electric Field & Potential on a Grid
- **Method:** Evaluate closed-form expressions on a discretized 2D/3D grid.
- **Validation:** Symmetry checks (e.g., field magnitude constant at fixed
  radius from a point charge); potential decreasing with 1/r.

### Equipotential Lines
- **Method:** Contour extraction from the potential grid (e.g., via
  `matplotlib.pyplot.contour` or `skimage.measure.find_contours`).
- **Validation:** Visual inspection against known symmetric cases
  (point charge → circles; dipole → characteristic figure-eight pattern).

### Laplace / Poisson Equation — Finite Difference Solver
- **Method:** Discretize the domain into a grid and solve
  ∇²V = -ρ/ε₀ (Poisson) or ∇²V = 0 (Laplace) using finite differences
  (e.g., 5-point stencil in 2D) with an iterative solver (Jacobi,
  Gauss-Seidel, or SOR) or a direct sparse linear solve.
- **Boundary conditions:** Dirichlet (fixed potential) and/or Neumann
  (fixed field) — to be finalized during implementation.
- **Validation:** Compare against analytical solutions for simple geometries
  (parallel plate capacitor, charge in a grounded box) and check convergence
  as grid resolution increases.

## Brownian Motion

### Random Walk / Gaussian Steps
- **Method:** At each timestep, draw displacement steps from a Gaussian
  distribution with variance set by the diffusion coefficient and timestep
  (`step ~ N(0, 2*D*dt)` per dimension).
- **Validation:** Step distribution histogram matches the expected Gaussian;
  no directional bias over many trials.

### Mean Squared Displacement (MSD)
- **Method:** Compute ⟨|x(t) - x(0)|²⟩ averaged over an ensemble of
  trajectories.
- **Validation:** MSD should scale linearly with time (`MSD = 2*d*D*t` for
  `d` dimensions), matching the theoretical diffusion coefficient `D`.

### Diffusion Equation
- **Method:** Finite-difference solution of ∂c/∂t = D∇²c on a grid.
- **Validation:** Compare against the analytical Gaussian spreading solution
  for a point-source initial condition.

## Drift-Diffusion Integration

- **Method:** Combine a deterministic drift term (from the electrostatic
  field, `v_drift = μ*E`) with the stochastic Brownian diffusion term at
  each timestep (Euler–Maruyama integration).
- **Validation:** Recover pure diffusion when the field is zero; recover
  pure drift when temperature/diffusion coefficient is zero; compare
  ensemble behavior against known drift-diffusion analytical limits
  (e.g., steady-state Boltzmann distribution in a potential well).

## Conventions

- All solvers should accept explicit units or work in a single consistent
  unit system (SI by default) documented in `src/shypyengine/core`.
- Numerical parameters (grid resolution, timestep, tolerance) should have
  sensible defaults but always be overridable by the caller.
- Every solver should expose a way to inspect convergence (iteration count,
  residual) for debugging and testing.

## Status

| Method | Implemented | Tested | Notes |
|---|---|---|---|
| Coulomb's law / superposition | ❌ | ❌ | |
| E-field / potential on grid | ❌ | ❌ | |
| Equipotential extraction | ❌ | ❌ | |
| Laplace/Poisson FD solver | ❌ | ❌ | |
| Random walk / Gaussian steps | ❌ | ❌ | |
| MSD calculation | ❌ | ❌ | |
| Diffusion equation solver | ❌ | ❌ | |
| Drift-diffusion integration | ❌ | ❌ | |
