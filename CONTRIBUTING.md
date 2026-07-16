# Contributing to ShyPyEngine

Thanks for your interest in contributing! ShyPyEngine is a learning friendly
project, so these guidelines aim to be clear enough for first time
contributors while keeping the codebase consistent as it grows.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Branching Strategy](#branching-strategy)
- [Setting Up Your Environment](#setting-up-your-environment)
- [Commit Message Conventions](#commit-message-conventions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Requirements](#documentation-requirements)

## Code of Conduct

By participating in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Branching Strategy

ShyPyEngine uses a three-tier branching model:

```
main            (protected, always stable)
 └── develop     (integration branch)
      └── feature/<name>   (short lived, one topic each)
```

- **`main`** — Never commit directly. Only updated via reviewed, approved pull
  requests from `develop` (typically at release points).
- **`develop`** — The default branch for ongoing work. Feature branches merge
  here first.
- **Feature branches** — Branch from `develop`, named after the area of work:
  - `feature/electrostatics`
  - `feature/brownian-motion`
  - `feature/drift-diffusion`
  - `feature/visualization`
  - `feature/documentation`
  - `feature/testing`
  - `feature/research`
  - `feature/github-setup`

  Create a new branch for anything not already covered by these:
  `git checkout -b feature/<short-description> develop`

## Setting Up Your Environment

```bash
git clone https://github.com/<org>/ShyPyEngine.git
cd ShyPyEngine
git checkout develop

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .[dev]
pre-commit install   # if/when pre-commit hooks are added
```

## Commit Message Conventions

We use a simplified [Conventional Commits](https://www.conventionalcommits.org/)
style:

```
<type>(<scope>): <short summary>

[optional body]
[optional footer]
```

**Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`, `perf`

**Examples:**
```
feat(electrostatics): add finite difference Poisson solver
docs(readme): clarify installation steps
test(brownian): add MSD convergence test
fix(core): correct units in Coulomb constant
```

Keep the summary line under 72 characters and written in the imperative mood
("add", not "added"/"adds").

## Pull Request Process

1. Branch from `develop` using the naming convention above.
2. Keep PRs focused — one feature or fix per PR.
3. Ensure all tests pass locally (`pytest`) before opening the PR.
4. Fill out the PR template completely, including a description of what
   changed and why.
5. Link any related issues (`Closes #123`).
6. Request review from at least one other team member.
7. Address review feedback with additional commits (avoid force-pushing over
   review history until approval).
8. A maintainer merges once approved and CI passes. Squash merge into
   `develop` is preferred to keep history readable.

## Coding Standards

- **Style:** Follow [PEP 8](https://peps.python.org/pep-0008/). Format code
  with `black` and lint with `ruff` (or `flake8`) before committing.
- **Type hints:** Use type hints on public functions and classes.
- **Docstrings:** Use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)
  for all public modules, classes, and functions — physics code should explain
  the underlying equation or method, not just the parameters.
- **Naming:** Prefer descriptive names that match standard physics notation
  where helpful (e.g., `electric_field`, not `ef`), with a comment mapping to
  symbols (e.g., `# E = q / (4*pi*eps0*r**2)`).
- **Modularity:** New physics functionality belongs in its own module under
  `src/shypyengine/<module>/`, with shared code factored into `core/` or
  `utils/`.

## Testing Requirements

- Every new function that performs a calculation should have at least one
  corresponding test in `tests/`.
- Use `pytest`. Name test files `test_<module>.py` and test functions
  `test_<behavior>()`.
- Where possible, validate numerical results against known analytical
  solutions (e.g., point charge field, free diffusion MSD scaling).
- Run the full suite before opening a PR:
  ```bash
  pytest --cov=src/shypyengine
  ```

## Documentation Requirements

- Update `docs/numerical_methods.md` when adding or changing a numerical
  method.
- Update `docs/architecture.md` when adding a new module or changing how
  modules interact.
- Update `CHANGELOG.md` under an "Unreleased" section for any user-facing
  change.
- Add or update a usage example in `examples/` for new public facing
  features.

Thank you, 
Building ShyPyEngine: where curiosity simulates the real world ! 🎉
