# Architecture

This document describes how ShyPyEngine's modules are organized and how they
are intended to interact. It should be updated whenever a module is added or
module boundaries change.

## Guiding Principles

- **Modularity** — Each physics domain (electrostatics, Brownian motion,
  drift-diffusion) is a self-contained sub-package with its own models,
  solvers, and tests. Modules should be usable independently.
- **Beginner-friendly** — Public APIs favor clarity over cleverness.
  Function and variable names mirror standard physics notation, with
  docstrings explaining the underlying equations.
- **Scalability** — The package layout leaves room for future additions
  (GUI, API layer, AI assistant, gamification) without restructuring the
  physics core.

## Package Layout

```
src/shypyengine/
├── core/            # Shared constants, base classes, unit handling, config
├── electrostatics/  # Coulomb's law, fields, potentials, Laplace/Poisson solver
├── brownian/        # Random walk, Gaussian steps, MSD, diffusion equation
├── driftdiffusion/  # Combines electrostatics + brownian into one engine
├── visualization/   # Shared plotting utilities (Matplotlib/Plotly)
└── utils/           # I/O, validation, math helpers used across modules
```

## Module Responsibilities

### `core`
Owns physical constants (e.g., ε₀, k_B), a consistent unit system, shared
base classes (e.g., a `Field` or `Grid` abstraction), and configuration
handling. Other modules depend on `core`; `core` depends on nothing else in
the package.

### `electrostatics`
Computes electric fields and potentials from charge configurations, extracts
equipotential lines, and solves the Laplace/Poisson equations on a grid.
Exposes a `Field` object that the `driftdiffusion` module can query for a
drift velocity at any point.

### `brownian`
Generates stochastic particle trajectories, computes MSD, and solves the
diffusion equation directly. Exposes a step-generation interface that the
`driftdiffusion` module reuses for its noise term.

### `driftdiffusion`
Depends on both `electrostatics` and `brownian`. Combines the deterministic
drift term from an electrostatic field with the stochastic term from the
Brownian module into a single time-stepped simulation.

### `visualization`
Provides shared, reusable plotting functions (field maps, trajectories,
MSD curves, animations) so that individual modules don't duplicate plotting
code. Depends on `core` only, not on the physics modules, to avoid circular
imports — physics modules pass plain arrays/objects to `visualization`
functions.

### `utils`
Small, dependency-free helpers (input validation, file I/O, array helpers)
usable by any module.

## Data Flow (Target State)

```
 charge configuration
        │
        ▼
  electrostatics.Field  ──────────┐
                                   │  drift velocity at (x, y[, z])
                                   ▼
 brownian.Trajectory ───────► driftdiffusion.Simulation ───► visualization
   (noise term)                                                   │
                                                                   ▼
                                                            plots / animations
```

## Future Layers (not yet started)

- **API layer (`FastAPI`)** — will wrap the `driftdiffusion` engine behind
  HTTP endpoints for the GUI to call. Lives outside `src/shypyengine` in a
  future `api/` package to keep the physics core UI-agnostic.
- **GUI (`React` / `Streamlit`/`PyQt` prototype)** — a separate front-end
  project that consumes the API layer; does not import the physics package
  directly in production.
- **AI assistant** — will call into the public `shypyengine` API to explain
  results and suggest parameters; treated as a consumer of the package, not
  a dependency of it.
- **Gamification** — built on top of the simulation + AI assistant layers.

## Import Rules

To avoid circular dependencies as the project grows:

- `core` → no internal dependencies
- `utils` → may depend on `core`
- `electrostatics`, `brownian` → may depend on `core`, `utils`
- `driftdiffusion` → may depend on `core`, `utils`, `electrostatics`, `brownian`
- `visualization` → may depend on `core`, `utils` only (accepts data, not
  module-specific objects, where practical)
