# Architecture

ShyPyEngine is organized as an installable package under `src/shypyengine/`,
with one subpackage per physics module:

- `core/`: shared base classes, physical constants, and configuration used
  across all modules.
- `electrostatics/`: Coulomb's law, field/potential computation, Laplace/
  Poisson finite-difference solver (Phase 1).
- `brownian/`: random walk generation, Gaussian step processes, mean
  squared displacement, and a finite-difference diffusion solver (Phase 2).
- `driftdiffusion/`: combines the electrostatic drift term with Brownian
  diffusion into a unified particle/ensemble simulation (Phase 3).
- `visualization/`: shared plotting/rendering helpers used by every module
  (static Matplotlib plots now, interactive Plotly views in Phase 4).
- `utils/`: I/O, validation, and shared math helpers.

Each module is independent and testable in isolation; `driftdiffusion`
composes `electrostatics` and `brownian` rather than duplicating logic.

## Solver Architecture

Each physics module follows the same pattern: a stateless generator or
solver class holds simulation parameters at construction time, validates
them immediately (raising `InvalidParameterError` on invalid input rather
than failing later inside a simulation loop), and exposes plain methods
that take and return NumPy arrays. No solver reads from or writes to global
state, and no solver depends on another solver's internals.

For `brownian`, this looks like:

- `RandomWalk` holds `n_dim`, `diffusion_coefficient`, and `dt`, and
  generates full trajectory arrays on demand via `generate()`.
- `DiffusionSolver1D` holds the grid and time-stepping parameters, exposes
  a single-step `step()` method, and a `solve()` method that repeatedly
  calls `step()` and collects the time history.

This keeps each solver usable both as a one-shot batch call (`solve`,
`generate`) and, where needed, as an incremental stepper (`step`) for
interactive or animated use in a future GUI.

## Documentation Map

Project documentation is split by purpose, to avoid duplicating the same
material across files:

- **README.md**: entry point -- installation, quick start, feature
  overview, and a condensed summary of each module's numerical methods.
- **docs/architecture.md** (this file): how the modules relate to each
  other and the shared design patterns solvers follow.
- **docs/numerical_methods.md**: the project's derivations reference --
  full mathematical derivations (Poisson/Laplace from Gauss's law, the
  finite-difference stencil via Taylor expansion, stability analysis, the
  random-walk-to-diffusion-equation derivation) and error analysis
  (truncation vs. discretization error, floating-point precision,
  boundary condition accuracy).
- **docs/electrostatics.md**: usage-oriented background for Phase 1
  specifically -- the same physics summarized at a level meant to be read
  alongside the API, with pointers into `numerical_methods.md` for the
  full derivations.
