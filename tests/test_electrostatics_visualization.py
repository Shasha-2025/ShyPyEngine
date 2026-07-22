"""Tests for the electrostatics plotting helpers.

Smoke tests, in the same spirit as ``tests/test_visualization.py``: verify
each function runs without error, returns a populated Axes, and raises the
project's own exception type on invalid input.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pytest

from shypyengine.core.exceptions import InvalidParameterError
from shypyengine.electrostatics.equipotential import extract_equipotential_lines
from shypyengine.electrostatics.point_charges import (
    PointCharge,
    electric_field,
    electric_potential,
)
from shypyengine.visualization import plot_equipotentials, plot_field_lines, plot_potential_map


def _make_grid(n=14, extent=3.0):
    x = np.linspace(-extent, extent, n)
    y = np.linspace(-extent, extent, n)
    return np.meshgrid(x, y)


class TestPlotFieldLines:
    def test_returns_axes(self):
        X, Y = _make_grid()
        points = np.stack([X, Y], axis=-1)
        charge = PointCharge(charge=1e-6, position=np.array([0.5, 0.5]))
        field = electric_field([charge], points)
        Ex, Ey = field[..., 0], field[..., 1]

        ax = plot_field_lines(X, Y, Ex, Ey)

        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_rejects_mismatched_shapes(self):
        X, Y = _make_grid()
        with pytest.raises(InvalidParameterError):
            plot_field_lines(X, Y, np.zeros((3, 3)), np.zeros((3, 3)))


class TestPlotPotentialMap:
    def test_returns_axes(self):
        X, Y = _make_grid()
        points = np.stack([X, Y], axis=-1)
        charge = PointCharge(charge=1e-6, position=np.array([0.0, 0.0]))
        V = electric_potential([charge], points)

        ax = plot_potential_map(X, Y, V)

        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_rejects_mismatched_shapes(self):
        X, Y = _make_grid()
        with pytest.raises(InvalidParameterError):
            plot_potential_map(X, Y, np.zeros((3, 3)))


class TestPlotEquipotentials:
    def test_returns_axes_with_lines(self):
        X, Y = _make_grid(n=120)
        points = np.stack([X, Y], axis=-1)
        charge = PointCharge(charge=1e-6, position=np.array([0.0, 0.0]))
        V = electric_potential([charge], points)
        lines = extract_equipotential_lines(X, Y, V, levels=[5000.0, 10000.0])

        ax = plot_equipotentials(lines)

        assert isinstance(ax, plt.Axes)
        assert len(ax.lines) >= 2
        plt.close(ax.figure)

    def test_handles_empty_lines_dict(self):
        ax = plot_equipotentials({})
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)
