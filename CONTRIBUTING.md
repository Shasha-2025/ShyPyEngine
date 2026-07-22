# Contributing to ShyPyEngine

ShyPyEngine is currently maintained as a solo project. This document
describes the workflow used to keep development organized even as a
one-person effort, and the conventions any future contributor should follow.

## Branching Strategy

- `main`: protected, always stable/deployable. Only updated via merge from
  `develop`.
- `develop`: integration branch. Feature branches merge here first.
- `feature/*`: one branch per unit of work, e.g. `feature/github-setup`,
  `feature/brownian-motion`.

Workflow for every change:

```bash
git checkout develop
git pull
git checkout -b feature/<name>
# ... do the work, commit ...
git checkout develop
git merge --no-ff feature/<name>
# once develop is stable:
git checkout main
git merge --no-ff develop
```

## Commit Messages

Use short, imperative commit messages, e.g. `Add Brownian random walk
generator`, `Fix MSD normalization bug`.

## Code Style

- Follow PEP 8.
- Formatting is enforced with [black](https://black.readthedocs.io/) and
  import ordering with [isort](https://pycqa.github.io/isort/) (both
  configured for a 99-character line length in `pyproject.toml`); linting
  is enforced with [ruff](https://docs.astral.sh/ruff/). Before opening a
  PR, run:

  ```bash
  black --line-length 99 src tests examples
  isort --profile black --line-length 99 src tests examples
  ruff check src tests examples
  ```

  CI checks all three (`black --check`, `isort --check-only`, and
  `ruff check`) and fails the build if any of them would change something.
- Add type hints on public functions.
- Every new module gets a matching test file under `tests/`.
- Docstrings (NumPy style) on all public classes and functions.

## Tests

Run the full suite before merging:

```bash
pytest
```

CI (`.github/workflows/ci.yml`) runs lint + tests automatically on every push
and pull request.

## Issues & Pull Requests

Use the templates under `.github/ISSUE_TEMPLATE/` and
`.github/pull_request_template.md` when opening issues or PRs.
