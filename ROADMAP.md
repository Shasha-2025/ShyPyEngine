# Roadmap

This roadmap tracks ShyPyEngine's progress from repository scaffolding through
a full physics engine with visualization, testing, GUI, AI assistance, and
gamification. Phases are sequential but may overlap once a phase's core
deliverables are stable.

## Phase 0 — Repository & Infrastructure (current phase)

- [x] Initialize repository, branch strategy, GitHub Desktop setup
- [x] Define team roles and workflow
- [x] Create full folder hierarchy
- [x] Write core documentation (README, CONTRIBUTING, CHANGELOG, ROADMAP,
      LICENSE, CODE_OF_CONDUCT)
- [ ] Configure GitHub Actions (lint + test on push/PR)
- [ ] Add issue templates and pull request template
- [x] Begin numerical-methods documentation (`docs/numerical_methods.md`)
- [x] Set up packaging (`pyproject.toml`, `requirements.txt`)

## Phase 1 — Electrostatics Module

- [ ] Coulomb's law (point charges, superposition)
- [ ] Electric field computation on a grid
- [ ] Electric potential computation
- [ ] Equipotential line extraction
- [ ] Finite-difference Laplace/Poisson solver
- [ ] Analytical-solution validation tests (point charge, parallel plates)
- [ ] Basic static plots (field lines, potential maps)

## Phase 2 — Brownian Motion Module

- [ ] Random walk generator (1D, 2D, 3D)
- [ ] Gaussian step process
- [ ] Mean squared displacement (MSD) calculation
- [ ] Diffusion equation solver (finite difference)
- [ ] Validation against theoretical diffusion coefficient scaling
- [ ] Basic trajectory plots

## Phase 3 — Drift-Diffusion Integration

- [x] Combine electrostatic drift term with Brownian diffusion term
- [x] Unified particle/ensemble simulation loop
- [ ] Parameter sweep utilities (field strength, temperature, particle count)
- [ ] Validation against known drift-diffusion analytical limits

## Phase 4 — Visualization

- [ ] Interactive Plotly views for fields, potentials, and trajectories
- [ ] Animated particle motion under drift-diffusion
- [ ] Exportable figures/animations for reports and notebooks

## Phase 5 — Testing & Quality

- [ ] Full unit test coverage for all modules
- [ ] Integration tests across module boundaries
- [ ] Performance/regression benchmarks
- [ ] Continuous integration gate required for merges into `develop`/`main`

## Phase 6 — GUI

- [ ] Prototype interactive UI (Streamlit or PyQt) for parameter exploration
- [ ] FastAPI backend exposing simulation endpoints
- [ ] React frontend consuming the API

## Phase 7 — AI Assistant

- [ ] In-app assistant to help users set up simulations and interpret results
- [ ] Guided explanations of underlying physics and numerical methods

## Phase 8 — Gamification

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
