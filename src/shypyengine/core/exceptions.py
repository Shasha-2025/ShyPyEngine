"""Shared exception types for ShyPyEngine."""

from __future__ import annotations


class ShyPyEngineError(Exception):
    """Base exception for all ShyPyEngine errors."""


class InvalidParameterError(ShyPyEngineError):
    """Raised when a simulation parameter is invalid (e.g. negative time step)."""
