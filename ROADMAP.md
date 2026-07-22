# Roadmap

This roadmap tracks ShyPyEngine's progress from repository scaffolding through
a full physics engine with visualization, testing, a GUI, an AI assistant,
and gamification. Phases are sequential but may overlap once a phase's core
deliverables are stable.

> **Note:** Originally scoped for a 5-person team, ShyPyEngine is now
> developed and maintained solo by **M4**. All phases below are executed by
> a single contributor; timelines and scope have been adjusted accordingly.

## Phase 0: Repository & Infrastructure (complete)

- [x] Initialize repository, branch strategy, GitHub Desktop setup
- [x] Define team roles and workflow
- [x] Create full folder hierarchy
- [x] Write core documentation (README, CONTRIBUTING, CHANGELOG, ROADMAP,
      LICENSE, CODE_OF_CONDUCT)
- [x] Configure GitHub Actions (lint + test on push/PR)
- [x] Add issue templates and pull request template
- [x] Begin numerical-methods documentation (`docs/numerical_methods.md`)
- [x] Set up packaging (`pyproject.toml`, `requirements.txt`)

## Phase 1: Electrostatics Module (complete)

- [x] Coulomb's law (point charges, superposition)
- [x] Electric field computation on a grid
- [x] Electric potential computation
- [x] Equipotential line extraction
- [x] Finite-difference Laplace/Poisson solver
- [x] Analytical-solution validation tests (point charge, infinite line,
      parallel plates)
- [x] Basic static plots (field lines, potential maps)

## Phase 2: Brownian Motion Module (complete)

- [x] Random walk generator (1D, 2D, 3D)
- [x] Gaussian step process
- [x] Mean squared displacement (MSD) calculation
- [x] Diffusion equation solver (finite difference)
- [x] Validation against theoretical diffusion coefficient scaling
- [x] Basic trajectory plots

## Phase 3: Drift-Diffusion Integration

- [ ] Combine electrostatic drift term with Brownian diffusion term
- [ ] Unified particle/ensemble simulation loop
- [ ] Parameter sweep utilities (field strength, temperature, particle count)
- [ ] Validation against known drift-diffusion analytical limits

## Phase 4: Visualization

- [ ] Interactive Plotly views for fields, potentials, and trajectories
- [ ] Animated particle motion under drift-diffusion
- [ ] Exportable figures/animations for reports and notebooks

## Phase 5: Testing & Quality

- [ ] Full unit test coverage for all modules
- [ ] Integration tests across module boundaries
- [ ] Performance/regression benchmarks
- [ ] Continuous integration gate required for merges into `develop`/`main`

## Phase 6: GUI

- [ ] Prototype interactive UI (Streamlit or PyQt) for parameter exploration
- [ ] FastAPI backend exposing simulation endpoints
- [ ] React frontend consuming the API

## Phase 7: AI Assistant

- [ ] In-app assistant to help users set up simulations and interpret results
- [ ] Guided explanations of underlying physics and numerical methods

## Phase 8: Gamification

- [ ] Challenge/scenario mode (e.g., "tune the field to match this trajectory")
- [ ] Progress tracking and learning-oriented feedback

---

## Long-Term Flow

```
Electrostatics Solver
        │
        ▼
  Brownian Engine
        │
        ▼
Drift-Diffusion Integration
        │
        ▼
     Visualization
        │
        ▼
       Testing
        │
        ▼
         GUI
        │
        ▼
   AI Assistant
        │
        ▼
    Gamification
```

Status will be updated as phases progress. See [CHANGELOG.md](CHANGELOG.md)
for a detailed history of completed work.
