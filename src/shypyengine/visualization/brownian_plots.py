"""Basic static (Matplotlib) plots for the Brownian motion module.

Interactive Plotly views are planned for Phase 4 (see ROADMAP.md); these
helpers cover the "basic trajectory plots" deliverable for Phase 2.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from shypyengine.core.exceptions import InvalidParameterError


def plot_trajectories_2d(trajectories: np.ndarray, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot 2D particle trajectories.

    Parameters
    ----------
    trajectories : np.ndarray of shape (n_steps + 1, n_particles, 2)
        Particle positions over time, as returned by
        :meth:`shypyengine.brownian.random_walk.RandomWalk.generate`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the trajectories were drawn on.

    Raises
    ------
    InvalidParameterError
        If ``trajectories`` is not 2-dimensional (i.e. the last axis
        does not have length 2).
    """
    if trajectories.shape[-1] != 2:
        raise InvalidParameterError("plot_trajectories_2d requires 2D trajectories.")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    n_particles = trajectories.shape[1]
    for p in range(n_particles):
        ax.plot(trajectories[:, p, 0], trajectories[:, p, 1], linewidth=0.8, alpha=0.8)
        ax.scatter(*trajectories[0, p], color="green", s=15, zorder=3)
        ax.scatter(*trajectories[-1, p], color="red", s=15, zorder=3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Brownian trajectories (n={n_particles})")
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def plot_msd(time: np.ndarray, msd: np.ndarray, ax: plt.Axes | None = None) -> plt.Axes:
    """Plot MSD vs. time, alongside the linear fit implied by the data.

    Parameters
    ----------
    time : np.ndarray
        Time values, as returned by
        :meth:`shypyengine.brownian.random_walk.RandomWalk.time_axis`.
    msd : np.ndarray
        Mean squared displacement at each time value, as returned by
        :func:`shypyengine.brownian.msd.mean_squared_displacement`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the MSD curve was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.plot(time, msd, "o", markersize=3, label="Simulated MSD")
    slope = np.polyfit(time, msd, 1)[0]
    ax.plot(time, slope * time, "--", label=f"Linear fit (slope={slope:.4g})")
    ax.set_xlabel("time")
    ax.set_ylabel("MSD")
    ax.set_title("Mean Squared Displacement vs. Time")
    ax.legend()
    return ax


def plot_diffusion_profiles(
    x: np.ndarray, history: np.ndarray, n_curves: int = 5, ax: plt.Axes | None = None
) -> plt.Axes:
    """Plot snapshots of a 1D diffusion concentration profile over time.

    Parameters
    ----------
    x : np.ndarray of shape (n_grid,)
        Spatial grid, as exposed by
        :attr:`shypyengine.brownian.diffusion.DiffusionSolver1D.x`.
    history : np.ndarray of shape (n_steps + 1, n_grid)
        Concentration profile at each time step, as returned by
        :meth:`shypyengine.brownian.diffusion.DiffusionSolver1D.solve`.
    n_curves : int, optional
        Number of snapshots to plot, evenly spaced across the time
        history. Capped at ``history.shape[0]`` if fewer steps exist.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the profile snapshots were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    n_steps = history.shape[0]
    indices = np.linspace(0, n_steps - 1, min(n_curves, n_steps)).astype(int)
    for i in indices:
        ax.plot(x, history[i], label=f"step {i}")

    ax.set_xlabel("x")
    ax.set_ylabel("concentration")
    ax.set_title("Diffusion profile evolution")
    ax.legend(fontsize=8)
    return ax
