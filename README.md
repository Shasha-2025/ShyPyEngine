# ShyPyEngine

A modular, beginner-friendly computational physics engine written in Python.

ShyPyEngine starts with two independent physics modules — **electrostatic field
simulation** and **Brownian motion** — and combines them into a unified
**drift-diffusion** simulation engine. The project is designed from day one to be
easy to extend with future additions such as an interactive GUI, an AI research
assistant, and gamified learning modes.

> **Status:** 🚧 Pre-alpha. Repository scaffolding and documentation phase.
> No simulation code has been implemented yet — see [ROADMAP.md](ROADMAP.md).

---

## Table of Contents

- [Project Goal](#project-goal)
- [Planned Physics Modules](#planned-physics-modules)
- [Repository Structure](#repository-structure)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Team](#team)
- [Contributing](#contributing)
- [License](#license)

## Project Goal

Build a modular computational physics engine, starting with:

1. **Electrostatics** — Coulomb's law, electric fields, potentials, equipotential
   lines, and numerical solutions to the Laplace/Poisson equations via finite
   difference methods.
2. **Brownian Motion** — random walks, Gaussian step processes, mean squared
   displacement (MSD), and the diffusion equation.
3. **Drift-Diffusion Integration** — combining the electrostatic drift term with
   the Brownian diffusion term into a single simulation engine, the way charged
   particles actually behave in a field with thermal noise.

Later phases add visualization tooling, a test suite, a GUI, an AI assistant for
guided exploration, and gamification for learning.

## Planned Physics Modules

| Module | Description | Status |
|---|---|---|
| `electrostatics` | Coulomb's law, E-field, potential, equipotentials, Laplace/Poisson finite-difference solver | Not started |
| `brownian` | Random walk, Gaussian steps, MSD, diffusion equation | Not started |
| `driftdiffusion` | Combines electrostatics (drift) + Brownian motion (diffusion) | Not started |
| `visualization` | Plotting and interactive views for all modules | Not started |

See [docs/architecture.md](docs/architecture.md) and
[docs/numerical_methods.md](docs/numerical_methods.md) for design notes.

## Repository Structure

```
ShyPyEngine/
├── src/shypyengine/       # Installable package (all engine source code)
│   ├── core/               # Shared base classes, constants, config
│   ├── electrostatics/     # Electrostatics module
│   ├── brownian/           # Brownian motion module
│   ├── driftdiffusion/     # Drift-diffusion integration module
│   ├── visualization/      # Plotting / rendering helpers
│   └── utils/               # Shared utilities (I/O, validation, math helpers)
├── tests/                  # Unit and integration tests (pytest)
├── docs/                   # Design docs, numerical methods, architecture notes
├── examples/                # Runnable example scripts
├── notebooks/               # Jupyter notebooks for exploration/demos
├── assets/                  # Images, diagrams, sample data
├── scripts/                  # Dev/maintenance scripts (setup, linting, etc.)
├── .github/                  # Issue templates, PR template, CI workflows
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── ROADMAP.md
├── CODE_OF_CONDUCT.md
├── LICENSE
├── requirements.txt
└── pyproject.toml
```

## Technology Stack

**Current**
- Python 3.11+
- NumPy, SciPy — numerical computation
- Matplotlib, Plotly — visualization
- Pandas — data handling
- VS Code, Git, GitHub — development environment

**Planned (future phases)**
- FastAPI — backend/API layer
- React — frontend GUI
- Streamlit or PyQt — early interactive UI prototyping

## Getting Started

> These steps describe the intended setup once initial scaffolding is merged.

```bash
# Clone the repository
git clone https://github.com/<org>/ShyPyEngine.git
cd ShyPyEngine

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies (editable install once src/ has a package)
pip install -r requirements.txt
pip install -e .

# Run the test suite
pytest
```

## Development Workflow

ShyPyEngine follows a protected `main` / integration `develop` / feature-branch
workflow:

- **`main`** — protected, always deployable/stable. Merges only from `develop`
  via reviewed pull requests.
- **`develop`** — integration branch where completed features land before a
  release.
- **Feature branches** — one per unit of work, branched from `develop`:
  `electrostatics`, `brownian-motion`, `drift-diffusion`, `visualization`,
  `documentation`, `testing`, `research`, `github-setup`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for full branching, commit, and PR
conventions.

## Team

| Role | Focus |
|---|---|
| M1 | Physics research |
| M2 | Simulation engine |
| M3 | Visualization |
| M4 | Repository / testing / infrastructure |
| M5 | UI/UX and future planning |

## Contributing

Contributions, issues, and discussion are welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
before opening a pull request.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
