"""Shared base classes, constants, and configuration used across modules."""

from shypyengine.core.constants import (
    BOLTZMANN_CONSTANT,
    COULOMB_CONSTANT,
    VACUUM_PERMITTIVITY,
)
from shypyengine.core.exceptions import InvalidParameterError, ShyPyEngineError

__all__ = [
    "BOLTZMANN_CONSTANT",
    "COULOMB_CONSTANT",
    "VACUUM_PERMITTIVITY",
    "InvalidParameterError",
    "ShyPyEngineError",
]
