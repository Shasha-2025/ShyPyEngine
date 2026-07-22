# Numerical Methods

This document derives the mathematics behind every numerical method
ShyPyEngine implements, and analyzes the error each introduces. It is the
project's academic/derivation reference; for usage-oriented documentation
(function signatures, worked examples, boundary condition options) see
[docs/electrostatics.md](electrostatics.md) for Phase 1 and the
"Numerical Methods" section of [README.md](../README.md) for Phase 2.

## Contents

- [Electrostatics: the Poisson and Laplace equations](#electrostatics-the-poisson-and-laplace-equations)
- [The finite-difference stencil](#the-finite-difference-stencil)
- [Stability and conditioning](#stability-and-conditioning)
- [Brownian motion: from random walk to the diffusion equation](#brownian-motion-from-random-walk-to-the-diffusion-equation)
- [Error analysis and numerical precision](#error-analysis-and-numerical-precision)
- [Boundary conditions](#boundary-conditions)

---

## Electrostatics: the Poisson and Laplace equations

### Derivation from Gauss's law

Gauss's law, one of Maxwell's equations, states in differential form:

```
nabla . E = rho / epsilon_0
```

where `E` is the electric field, `rho` is the charge density, and
`epsilon_0` is the vacuum permittivity. Electrostatics is curl-free
(`nabla x E = 0`, since there is no time-varying magnetic field), which
means `E` can always be written as the gradient of a scalar potential:

```
E = -nabla V
```

Substituting into Gauss's law:

```
nabla . (-nabla V) = rho / epsilon_0
-nabla^2 V = rho / epsilon_0
nabla^2 V = -rho / epsilon_0
```

This is the **Poisson equation**. In a charge-free region (`rho = 0`), it
reduces to the **Laplace equation**:

```
nabla^2 V = 0
```

`shypyengine.electrostatics.poisson.PoissonSolver2D` solves the 2D form of
both (the Laplace equation is simply the `rho = 0` case of `.solve()`).

### Coulomb's law as the point-charge solution

For a single point charge `q` at the origin, the spherically symmetric
solution of the Poisson equation (with a delta-function source) is:

```
V(r) = k * q / r,   k = 1 / (4 * pi * epsilon_0)
```

`E = -nabla V` then gives `E(r) = k * q * r_hat / r^2`, Coulomb's law.
Because the Poisson equation is linear, the potential and field of any
collection of point charges is the sum of each charge's individual
contribution -- this is the superposition principle implemented in
`shypyengine.electrostatics.point_charges.electric_field` and
`.electric_potential`.

---

## The finite-difference stencil

### Derivation via Taylor expansion

To discretize `nabla^2 V` on a grid, expand `V` in a Taylor series around
a grid point `(x_i, y_j)` along the x-direction, with grid spacing `dx`:

```
V(x_i + dx, y_j) = V(x_i, y_j) + dx * V_x + (dx^2/2) * V_xx + (dx^3/6) * V_xxx + (dx^4/24) * V_xxxx + O(dx^5)
V(x_i - dx, y_j) = V(x_i, y_j) - dx * V_x + (dx^2/2) * V_xx - (dx^3/6) * V_xxx + (dx^4/24) * V_xxxx + O(dx^5)
```

where `V_x`, `V_xx`, ... denote partial derivatives evaluated at
`(x_i, y_j)`. Adding these two expansions cancels the odd-order terms
(`V_x`, `V_xxx`):

```
V(x_i+dx, y_j) + V(x_i-dx, y_j) = 2*V(x_i,y_j) + dx^2 * V_xx + (dx^4/12) * V_xxxx + O(dx^6)
```

Solving for `V_xx`:

```
V_xx = [V(x_i+dx,y_j) - 2*V(x_i,y_j) + V(x_i-dx,y_j)] / dx^2 - (dx^2/12)*V_xxxx + O(dx^4)
```

The first term is the **central difference approximation**; the
`(dx^2/12) * V_xxxx` term (and everything after it) is the **truncation
error** discarded by using only the finite-difference approximation. This
is what makes the scheme **second-order accurate**: the leading-order
error term is `O(dx^2)`, not `O(dx)`.

The same derivation along y gives the analogous approximation for `V_yy`.
Adding both and applying the Poisson equation gives the **5-point
stencil** implemented in `PoissonSolver2D.solve`:

```
[V(i+1,j) - 2*V(i,j) + V(i-1,j)] / dx^2 + [V(i,j+1) - 2*V(i,j) + V(i,j-1)] / dy^2 = -rho(i,j) / epsilon_0
```

with local truncation error `O(dx^2 + dy^2)`.

### Assembly as a linear system

Unlike a time-dependent PDE, the Poisson/Laplace equation has no time
axis to step forward -- it is a boundary value problem. Writing the
stencil equation at every grid node (with boundary nodes replaced by their
boundary condition equation instead) produces one linear equation per
node, i.e. a sparse linear system `A x = b` with `nx * ny` unknowns.
`PoissonSolver2D` assembles this system directly and solves it with
`scipy.sparse.linalg.spsolve` (sparse LU factorization), which is exact
up to floating-point round-off -- there is no iterative convergence
tolerance to tune, unlike relaxation methods (Jacobi, Gauss-Seidel, SOR).

---

## Stability and conditioning

"Stability" in the sense of time-stepping schemes (whether errors grow or
decay as a simulation advances) does not apply to `PoissonSolver2D`,
since there is no time axis -- the linear system is solved once, directly.
The relevant numerical-quality question instead is the **conditioning**
of the assembled matrix `A`: how much a small perturbation in `b` (e.g.
floating-point round-off) can be amplified in the solution `x`. For the
standard 5-point Laplacian on a grid of `N = nx * ny` nodes, the condition
number of `A` grows as `O(N)` (equivalently `O(1/dx^2)`) as the grid is
refined, which is the standard, well-understood behavior of discretized
elliptic operators. `scipy`'s sparse LU factorization handles this
robustly for the grid sizes this project targets (see
[Error analysis and numerical precision](#error-analysis-and-numerical-precision)).

Time-stepping stability *does* apply to `DiffusionSolver1D` (Phase 2),
which uses an explicit FTCS (Forward-Time, Centered-Space) scheme for the
time-dependent diffusion equation. A von Neumann stability analysis
proceeds by substituting a Fourier mode `c(x,t) = xi^n * exp(i*k*x)` into
the discretized update

```
c[i, n+1] = c[i, n] + alpha * (c[i+1,n] - 2*c[i,n] + c[i-1,n]),   alpha = D*dt/dx^2
```

which gives an amplification factor

```
xi(k) = 1 - 4*alpha*sin^2(k*dx/2)
```

For the scheme to be stable, every Fourier mode must not grow in
magnitude: `|xi(k)| <= 1` for all `k`. Since `sin^2` ranges over `[0,1]`,
the binding constraint is at `sin^2(k*dx/2) = 1`, giving `xi = 1 - 4*alpha`.
Requiring `-1 <= 1 - 4*alpha` gives:

```
alpha = D * dt / dx^2 <= 1/2
```

`DiffusionSolver1D.__init__` checks this condition at construction time
and raises `InvalidParameterError` if it is violated, rather than
silently producing an unstable, oscillating solution.

---

## Brownian motion: from random walk to the diffusion equation

`shypyengine.brownian.random_walk.RandomWalk` generates trajectories
directly as a discrete-time stochastic process, without solving a PDE.
The connection between that process and the diffusion equation
`DiffusionSolver1D` solves is not a coincidence -- it is a standard
result, derived here for a 1D random walk (the same argument extends
independently to each dimension).

### Setup: the master equation

Let `p(x, t)` be the probability density of a particle's position at time
`t`. In a discrete-time random walk with time step `dt`, a particle at
position `x` at time `t + dt` must have arrived there from some position
`x - s` at time `t`, having taken a step of size `s`. If steps are drawn
from a distribution `phi(s)` independent of position and time, this gives
the **master equation**:

```
p(x, t + dt) = integral[ p(x - s, t) * phi(s) ds ]
```

### Taylor expansion of the master equation

Expand both sides in Taylor series, assuming `dt` and typical step sizes
`s` are small. On the left:

```
p(x, t+dt) = p(x,t) + dt * dp/dt + O(dt^2)
```

On the right, expand `p(x - s, t)` in `s` around `x`:

```
p(x - s, t) = p(x,t) - s * dp/dx + (s^2/2) * d^2p/dx^2 - O(s^3)
```

Substituting and integrating term by term against `phi(s)`:

- `integral[ phi(s) ds ] = 1` (phi is a probability distribution)
- `integral[ s * phi(s) ds ] = <s> = 0` (zero-mean steps, as
  `RandomWalk` uses: `gaussian_displacement` has mean 0 per axis)
- `integral[ s^2 * phi(s) ds ] = <s^2> = 2*D*dt` (this is the *definition*
  of the diffusion coefficient `D` for the step distribution used --
  see below)

gives:

```
p(x,t) + dt * dp/dt + O(dt^2) = p(x,t) + (2*D*dt/2) * d^2p/dx^2 + O(s^3, dt^2)
```

Canceling `p(x,t)` from both sides and dividing by `dt`:

```
dp/dt = D * d^2p/dx^2 + O(dt, s^2)
```

which is exactly the 1D diffusion equation `∂c/∂t = D * ∂²c/∂x²` in the
continuum limit `dt -> 0`. The higher-dimensional case follows the same
argument applied independently along each axis, giving
`∂c/∂t = D * nabla^2 c`.

### Why the step variance must be `2*D*dt`

The derivation above assumed `<s^2> = 2*D*dt`; this is not an extra
assumption but the condition that makes the random walk converge to a
diffusion process with coefficient `D` at all. `RandomWalk` enforces this
directly: each Cartesian component of a step is drawn from
`N(0, 2*D*dt)` (implemented in `RandomWalk.step_std`, which returns
`sqrt(2*D*dt)`), which is exactly the variance the master-equation
derivation above requires.

### Mean squared displacement

For an unbiased random walk (`<s> = 0`) in `n` dimensions with independent
axes, each of variance `2*D*dt` per step, after `N` steps (total elapsed
time `t = N*dt`) the position variance along each axis is `N * 2*D*dt =
2*D*t`, and summing over `n` independent axes:

```
MSD(t) = <|x(t) - x(0)|^2> = 2 * n * D * t
```

This is the analytical target `estimate_diffusion_coefficient` fits
against (a line through the origin, slope `2*n*D`), and the scaling
`tests/test_brownian.py` validates the simulated trajectories against.

---

## Error analysis and numerical precision

### Discretization error vs. truncation error

These two terms are related but distinct:

- **Truncation error** is the error introduced *per grid point* by
  replacing an exact derivative with a finite-difference approximation --
  the `O(dx^2)` term dropped in the Taylor expansion above. It is a
  *local* quantity: the error made in a single stencil equation.
- **Discretization error** (also called *global error*) is the
  difference between the exact continuous solution and the numerical
  solution *of the full discretized system*, after solving it. It
  accumulates the local truncation errors from every grid point, combined
  through the linear solve (for Poisson) or through repeated time
  stepping (for the diffusion equation).

For a well-posed, stable, second-order-accurate scheme, standard
numerical analysis (the Lax equivalence theorem, informally) guarantees
that global discretization error is also `O(dx^2)`: halving the grid
spacing should reduce the error by roughly a factor of 4. This is not
just a theoretical claim in this repository -- it is checked empirically
by `tests/test_electrostatics.py::TestPoissonSolverConvergence`, using
the method of manufactured solutions: an exact potential
`V = sin(pi*x)*sin(pi*y)` is chosen, the charge density that produces it
is derived analytically, the Poisson equation is solved numerically for
several grid resolutions, and the maximum error is confirmed to shrink by
approximately 4x each time the grid spacing is halved.

### Floating-point precision

All solvers use IEEE 754 double precision (`float64`, NumPy's default),
giving roughly 15-17 significant decimal digits of representable
precision. Two places floating-point behavior specifically matters in
this codebase:

- **Near-singular field/potential evaluation**: `electric_field` and
  `electric_potential` raise `InvalidParameterError` when a field point
  is within `1e-12` (length units) of a charge, rather than letting the
  `1/r^2` or `1/r` terms silently blow up to `inf` or lose precision to
  catastrophic cancellation as `r -> 0`.
- **Sparse linear solve conditioning**: as grid resolution increases, the
  condition number of the assembled Poisson matrix grows (see
  [Stability and conditioning](#stability-and-conditioning)), which can
  eventually erode the accuracy gains from a finer grid, since round-off
  error accumulates faster than truncation error shrinks. In practice,
  for the grid sizes this project targets (up to a few hundred points per
  axis), truncation error dominates and this is not a practical concern
  -- the convergence test above would fail to show the expected 4x error
  reduction once round-off dominated, and it does not.

No mixed-precision or extended-precision arithmetic is used anywhere in
the codebase.

---

## Boundary conditions

Both solvers in this project (`PoissonSolver2D` for electrostatics,
`DiffusionSolver1D` for diffusion) require boundary conditions to close
the system of equations -- without them, the interior stencil equations
alone do not have a unique solution.

### Dirichlet boundary conditions

A Dirichlet condition fixes the solution's *value* on the boundary:
`V = V_boundary` on the edge. In `PoissonSolver2D`, this is implemented
*exactly*: the corresponding row of the linear system is replaced with
the identity equation `V[boundary node] = V_boundary`, so a Dirichlet
boundary introduces no discretization error of its own -- the boundary
value is satisfied to floating-point precision, not merely approximated.
Physically, this represents a conductor held at a known, fixed potential
(e.g. a charged plate connected to a battery).

### Neumann boundary conditions

A Neumann condition fixes the solution's *normal derivative* at the
boundary. This project supports only the **zero-flux** case,
`dV/dn = 0`, implemented via a first-order one-sided (mirror) condition:
the boundary node is set equal to its nearest interior neighbor,
`V[boundary] = V[boundary + 1]`. This is a first-order-accurate
approximation to the zero-derivative condition (its local truncation
error is `O(dx)`, one order lower than the interior stencil's `O(dx^2)`),
which in principle could reduce global accuracy near a Neumann edge for
very coarse grids. Physically, a zero-flux Neumann boundary represents an
insulating wall, or a symmetry plane across which nothing flows (used,
for example, to approximate translational symmetry along the plates in
the parallel-plate capacitor validation case).

A general nonzero-flux Neumann condition (`dV/dn = g(x)` for `g != 0`) is
not implemented.

### Well-posedness: why not all-Neumann

A Poisson/Laplace problem with Neumann conditions on *every* boundary has
no unique solution: if `V(x,y)` solves it, so does `V(x,y) + C` for any
constant `C`, since adding a constant does not change any derivative. The
discretized linear system is correspondingly singular in this case.
`PoissonSolver2D.solve` detects this configuration (all four boundaries
Neumann) and raises `InvalidParameterError` explicitly, rather than
silently returning the `NaN` values a singular sparse solve would
otherwise produce. At least one Dirichlet boundary is required to pin
down the additive constant and make the solution unique.

### Reflective (Neumann) boundaries in the diffusion solver

`DiffusionSolver1D` uses the same zero-flux mirror approach at both ends
of its 1D grid, interpreted physically as reflective walls: no
concentration crosses the boundary, so total mass is conserved (checked
directly in `tests/test_brownian.py`).
