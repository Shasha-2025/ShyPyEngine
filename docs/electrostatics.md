# Electrostatics Module

`shypyengine.electrostatics` implements point-charge electrostatics via
Coulomb's law and superposition, plus a finite-difference solver for the
electrostatic Poisson and Laplace equations on a 2D grid. This document
covers the physics and *usage* of the module; for full mathematical
derivations (Poisson/Laplace from Gauss's law, the finite-difference
stencil via Taylor expansion, truncation/discretization error analysis,
and stability/conditioning) see
[docs/numerical_methods.md](numerical_methods.md). See the
[Package API](../README.md#package-api) section of the README for the
function/class signatures, and `examples/electrostatics_demo.py` for a
runnable walkthrough.

## Physical background

### Coulomb's law

The force between two point charges `q1` and `q2` separated by a
displacement vector `r = r1 - r2` is

```
F = k * q1 * q2 * r / |r|^3
```

where `k` is Coulomb's constant (`shypyengine.core.constants.COULOMB_CONSTANT`,
≈ 8.988e9 N·m²/C²). The force is repulsive (points along `+r`, away from
`q2`) when `q1` and `q2` have the same sign, and attractive otherwise.
Implemented in `coulomb_force`.

### Electric field via superposition

The electric field at a point `r` due to a collection of point charges
`{qi}` at positions `{ri}` is the vector sum of each charge's individual
field:

```
E(r) = sum_i [ k * qi * (r - ri) / |r - ri|^3 ]
```

This is the field a unit positive test charge at `r` would feel. Implemented
in `electric_field`, fully vectorized: `points` may be a single point, shape
`(n_dim,)`, or an arbitrarily-shaped grid, shape `(..., n_dim)` (for example
a 2D meshgrid of shape `(ny, nx, 2)`).

### Electric potential via superposition

The scalar electric potential at `r` is the corresponding sum of each
charge's `1/r` contribution:

```
V(r) = sum_i [ k * qi / |r - ri| ]
```

`E` and `V` are related by `E = -grad(V)`; `electric_field_from_potential`
recovers `E` numerically from a potential grid via central differences.
Implemented in `electric_potential`, with the same vectorization behavior
as `electric_field`.

**Singularities:** both functions raise `InvalidParameterError` if a
field/evaluation point coincides with a charge, where the `1/r` (potential)
or `1/r²` (field) expressions are undefined, rather than silently
returning `inf` or `nan`.

### Poisson and Laplace equations

Coulomb's law and superposition describe the field of point charges
directly, but many problems are more naturally posed in terms of a
continuous charge density `rho(r)` and boundary conditions on a domain's
edges (for example, conductors held at fixed potentials). That is the
electrostatic Poisson equation:

```
nabla^2 V = -rho / epsilon_0
```

where `epsilon_0` is the vacuum permittivity
(`shypyengine.core.constants.VACUUM_PERMITTIVITY`, ≈ 8.854e-12 F/m). In a
charge-free region (`rho = 0`), this reduces to the Laplace equation:

```
nabla^2 V = 0
```

`PoissonSolver2D` solves both (the Laplace equation is simply the `rho = 0`
case) on a uniform 2D grid.

### Boundary conditions

`PoissonSolver2D.solve` takes an independent `BoundaryCondition` for each
of the four grid edges (left, right, top, bottom):

- **Dirichlet**: fixes the potential to a given value along the edge
  (scalar, or an array matching the edge's length for a spatially varying
  boundary, e.g. one charged plate). Physically: a conductor held at a
  known potential. Implemented exactly (the boundary row of the linear
  system is an identity equation), so a Dirichlet boundary introduces no
  discretization error of its own.
- **Neumann**: enforces a zero normal derivative (`dV/dn = 0`) along the
  edge — an insulating boundary, or a symmetry plane where nothing crosses.
  Only the zero-flux case is supported; a general nonzero-flux Neumann
  condition is not implemented. Implemented via a first-order one-sided
  (mirror) approximation, `V[boundary] = V[nearest interior neighbor]`,
  which has local truncation error `O(dx)` — one order lower than the
  `O(dx²)` interior stencil. See
  [docs/numerical_methods.md](numerical_methods.md#boundary-conditions)
  for the full accuracy discussion.

**Well-posedness:** boundary conditions must fix the potential's additive
constant somewhere. A configuration with Neumann conditions on *all four*
edges has no unique solution (`V` and `V + C` both satisfy it for any
constant `C`), and the underlying linear system is singular.
`PoissonSolver2D.solve` detects this and raises `InvalidParameterError`
rather than returning the `NaN` values a singular solve would otherwise
produce silently — at least one edge must be Dirichlet.

Corner nodes belong to two edges simultaneously; by convention the
left/right boundary condition takes precedence over top/bottom at corners
(see `PoissonSolver2D.solve`'s docstring for the exact rule).

## Numerical method: finite differences

### Discretization

`PoissonSolver2D` uses the standard second-order, 5-point finite-difference
stencil for the 2D Laplacian on a uniform grid with spacing `dx`, `dy`:

```
(V[i+1,j] - 2*V[i,j] + V[i-1,j]) / dx^2
    + (V[i,j+1] - 2*V[i,j] + V[i,j-1]) / dy^2 = -rho[i,j] / epsilon_0
```

Unlike the time-dependent diffusion equation solved elsewhere in this
project (see `docs/numerical_methods.md`), the Poisson/Laplace equation is
a linear boundary value problem with no time axis, so there is nothing to
step forward. `PoissonSolver2D` assembles the discretized equations (one
per grid node, including the boundary conditions) as a sparse linear
system `A x = b` and solves it directly with
`scipy.sparse.linalg.spsolve`, rather than iterating (e.g. Jacobi or
Gauss-Seidel relaxation) to convergence. This is exact up to the
discretization error and floating-point solver tolerance — there is no
separate iteration-convergence error to reason about.

### Equipotential line extraction

Equipotential lines are the level curves of `V(x, y)`: every point on a
given curve shares the same potential, and by construction such a curve is
always perpendicular to the local electric field. `extract_equipotential_lines`
extracts these via the marching-squares algorithm, using `contourpy` (the
same contour-computation library Matplotlib uses internally) directly,
without going through a plotting call. This keeps line *extraction*
independent of line *rendering*: `shypyengine.visualization.plot_equipotentials`
handles the latter.

### Complexity

For an `nx` by `ny` grid, matrix assembly and the sparse direct solve are
both `O(nx * ny)` in the sizes involved (the coefficient matrix has 5
nonzero entries per row, i.e. `O(nx * ny)` nonzeros total); the practical
cost of `spsolve` depends on the resulting sparse matrix's fill-in during
factorization, which is well-behaved for the banded structure produced by
a 2D grid Laplacian.

## Validation methodology

Three configurations with known analytical solutions are used to validate
the implementation (see `tests/test_electrostatics.py`):

1. **Point charge**: `electric_field` and `electric_potential` are checked
   directly against Coulomb's law, `|E| = k*q/r²` and `V = k*q/r`, at an
   arbitrary point.
2. **Infinite line charge**: approximated by a long, finely-discretized
   chain of point charges with linear charge density `lambda`, evaluated
   perpendicular to its midpoint (far from the ends, where the
   finite-length approximation breaks down). The computed field is
   checked against the analytical infinite-line result,
   `E = 2 * k * lambda / r`.
3. **Parallel-plate capacitor**: `PoissonSolver2D` is configured with
   Dirichlet boundaries at `+V0` and `-V0` on the left/right edges and
   Neumann (zero-flux) boundaries on top/bottom, approximating two
   infinite charged plates. In the charge-free gap, this must produce a
   linear potential profile and a uniform field of magnitude
   `E = 2*V0 / d` (with `d` the plate separation), matching the standard
   parallel-plate capacitor result (fringing fields are not modeled).

## Mathematical assumptions

- Point charges are idealized as dimensionless; the field and potential
  are undefined exactly at a charge's location.
- The 2D Poisson/Laplace solver assumes a fixed, uniform rectangular grid
  and only supports Dirichlet and zero-flux Neumann boundary conditions
  (no periodic or nonzero-flux Neumann boundaries).
- `epsilon_0` and `k` are treated as vacuum values; no dielectric medium
  (relative permittivity) is modeled.
- The parallel-plate validation neglects fringing fields, consistent with
  the standard infinite-plate approximation.

## Current limitations

- No 3D grid Poisson/Laplace solver; only the point-charge functions
  (`electric_field`, `electric_potential`) support 3D directly.
- No dielectric materials (spatially varying permittivity).
- No conductors with induced (as opposed to prescribed) charge
  distributions.
- Equipotential extraction depends on `contourpy`; extremely coarse grids
  may produce jagged or missing contours for closely spaced levels.
- A configuration with Neumann boundaries on all four edges is rejected
  (the underlying linear system is singular); at least one edge must be
  Dirichlet to pin down the potential's additive constant.
