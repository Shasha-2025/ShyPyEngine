"""Mean squared displacement (MSD) calculation and diffusion coefficient
estimation for Brownian motion trajectories.
"""

from __future__ import annotations

import numpy as np

from shypyengine.core.exceptions import InvalidParameterError


def mean_squared_displacement(trajectories: np.ndarray) -> np.ndarray:
    """Compute the ensemble-averaged MSD at each time step relative to t=0.

    Parameters
    ----------
    trajectories : np.ndarray of shape (n_steps + 1, n_particles, n_dim)
        Particle positions over time, as returned by
        :meth:`shypyengine.brownian.random_walk.RandomWalk.generate`.

    Returns
    -------
    np.ndarray of shape (n_steps + 1,)
        MSD(t) = < |x(t) - x(0)|^2 >, averaged over all particles.

    Raises
    ------
    InvalidParameterError
        If ``trajectories`` does not have exactly 3 dimensions
        (steps, particles, spatial coordinates).
    """
    trajectories = np.asarray(trajectories, dtype=float)
    if trajectories.ndim != 3:
        raise InvalidParameterError(
            "trajectories must have shape (n_steps + 1, n_particles, n_dim)."
        )

    displacements = trajectories - trajectories[0]
    squared_norms = np.sum(displacements**2, axis=-1)  # (n_steps+1, n_particles)
    return squared_norms.mean(axis=1)


def estimate_diffusion_coefficient(time: np.ndarray, msd: np.ndarray, n_dim: int) -> float:
    """Estimate the diffusion coefficient from an MSD-vs-time curve.

    Fits ``MSD(t) = 2 * n_dim * D * t`` via linear least squares (forcing
    the fit through the origin, since MSD(0) = 0 by construction).

    Parameters
    ----------
    time : np.ndarray
        Time values corresponding to each MSD sample.
    msd : np.ndarray
        Mean squared displacement at each time value.
    n_dim : int
        Number of spatial dimensions the trajectories live in.

    Returns
    -------
    float
        Estimated diffusion coefficient D.

    Raises
    ------
    InvalidParameterError
        If ``time`` and ``msd`` have different shapes, if ``n_dim`` is
        not 1, 2, or 3, or if ``time`` is all zeros (degenerate fit).
    """
    time = np.asarray(time, dtype=float)
    msd = np.asarray(msd, dtype=float)
    if time.shape != msd.shape:
        raise InvalidParameterError("time and msd must have the same shape.")
    if n_dim not in (1, 2, 3):
        raise InvalidParameterError("n_dim must be 1, 2, or 3.")

    # Least-squares slope through the origin: slope = sum(t*msd) / sum(t*t)
    denom = np.sum(time * time)
    if denom == 0:
        raise InvalidParameterError("time array must contain nonzero values.")
    slope = np.sum(time * msd) / denom
    return float(slope / (2.0 * n_dim))
