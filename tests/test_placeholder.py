"""Placeholder test to confirm the test suite and CI pipeline are wired up.

Replace/remove this once real tests exist for the electrostatics, brownian,
and drift-diffusion modules (see CONTRIBUTING.md for testing requirements).
"""

import shypyengine


def test_package_imports():
    """The package should import cleanly and expose a version string."""
    assert hasattr(shypyengine, "__version__")
    assert isinstance(shypyengine.__version__, str)
