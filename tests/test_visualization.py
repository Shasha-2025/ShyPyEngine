"""Tests for the Brownian motion plotting helpers.

These are smoke tests: they don't assert on pixel output, but they do
verify that each function runs without error, returns a populated
Axes, and raises the project's own exception type (not a generic
ValueError) on invalid input.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import matplotlib.pyplot as plt
import pytest

from shypyengine.brownian.diffusion import DiffusionSolver1D
from shypyengine.brownian.msd import mean_squared_displacement
from shypyengine.brownian.random_walk import RandomWalk
from shypyengine.core.exceptions import InvalidParameterError
from shypyengine.visualization import plot_diffusion_profiles, plot_msd, plot_trajectories_2d


class TestPlotTrajectories2D:
    def test_returns_axes_with_one_line_per_particle(self):
        walk = RandomWalk(n_dim=2, diffusion_coefficient=1.0, dt=0.01, seed=0)
        trajectories = walk.generate(n_steps=50, n_particles=4)

        ax = plot_trajectories_2d(trajectories)

        assert isinstance(ax, plt.Axes)
        assert len(ax.lines) == 4
        plt.close(ax.figure)

    def test_rejects_non_2d_trajectories(self):
        walk = RandomWalk(n_dim=3, diffusion_coefficient=1.0, dt=0.01, seed=0)
        trajectories = walk.generate(n_steps=10, n_particles=2)

        with pytest.raises(InvalidParameterError):
            plot_trajectories_2d(trajectories)

    def test_draws_on_provided_axes(self):
        walk = RandomWalk(n_dim=2, diffusion_coefficient=1.0, dt=0.01, seed=0)
        trajectories = walk.generate(n_steps=20, n_particles=2)
        fig, ax = plt.subplots()

        returned_ax = plot_trajectories_2d(trajectories, ax=ax)

        assert returned_ax is ax
        plt.close(fig)


class TestPlotMSD:
    def test_returns_axes_with_data(self):
        walk = RandomWalk(n_dim=2, diffusion_coefficient=1.0, dt=0.01, seed=0)
        trajectories = walk.generate(n_steps=100, n_particles=50)
        time = walk.time_axis(n_steps=100)
        msd = mean_squared_displacement(trajectories)

        ax = plot_msd(time, msd)

        assert isinstance(ax, plt.Axes)
        assert len(ax.lines) == 2  # simulated MSD + linear fit
        plt.close(ax.figure)


class TestPlotDiffusionProfiles:
    def test_returns_axes_with_requested_number_of_curves(self):
        solver = DiffusionSolver1D(diffusion_coefficient=1.0, dx=0.1, dt=0.001, n_grid=51)
        c0 = solver.initial_gaussian(mean=solver.x[25], sigma=0.3)
        history = solver.solve(c0, n_steps=100)

        ax = plot_diffusion_profiles(solver.x, history, n_curves=4)

        assert isinstance(ax, plt.Axes)
        assert len(ax.lines) == 4
        plt.close(ax.figure)

    def test_caps_n_curves_at_available_steps(self):
        solver = DiffusionSolver1D(diffusion_coefficient=1.0, dx=0.1, dt=0.001, n_grid=51)
        c0 = solver.initial_gaussian(mean=solver.x[25], sigma=0.3)
        history = solver.solve(c0, n_steps=2)  # only 3 snapshots exist

        ax = plot_diffusion_profiles(solver.x, history, n_curves=10)

        assert len(ax.lines) == 3
        plt.close(ax.figure)
