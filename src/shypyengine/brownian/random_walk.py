"""Random walk and Gaussian step process generation for Brownian motion.

The random walk is discretized so that, in the continuum limit, it recovers
the diffusion equation ``∂c/∂t = D * ∇²c``. Each Cartesian component of the
displacement at every step is drawn independently from
``N(0, 2 * D * dt)``, which gives the correct mean squared displacement
scaling ``MSD(t) = 2 * n_dim * D * t`` for free diffusion.
"""

from __future__ import annotations

import numpy as np

from shypyengine.core.exceptions import InvalidParameterError


class RandomWalk:
    """Generate Brownian-motion trajectories via a Gaussian random walk.

    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions (1, 2, or 3).
    diffusion_coefficient : float
        Diffusion coefficient ``D`` (length^2 / time). Must be positive.
    dt : float
        Time step size. Must be positive.
    seed : int or None, optional
        Seed for the random number generator, for reproducibility.

    Raises
    ------
    InvalidParameterError
        If ``n_dim`` is not 1, 2, or 3, or if ``diffusion_coefficient``
        or ``dt`` is not strictly positive.

    Examples
    --------
    >>> walk = RandomWalk(n_dim=2, diffusion_coefficient=1.0, dt=0.01, seed=0)
    >>> trajectories = walk.generate(n_steps=1000, n_particles=50)
    >>> trajectories.shape
    (51, 50, 2)
    """

    def __init__(
        self,
        n_dim: int,
        diffusion_coefficient: float,
        dt: float,
        seed: int | None = None,
    ) -> None:
        if n_dim not in (1, 2, 3):
            raise InvalidParameterError("n_dim must be 1, 2, or 3.")
        if diffusion_coefficient <= 0:
            raise InvalidParameterError("diffusion_coefficient must be positive.")
        if dt <= 0:
            raise InvalidParameterError("dt must be positive.")

        self.n_dim = n_dim
        self.diffusion_coefficient = diffusion_coefficient
        self.dt = dt
        self._rng = np.random.default_rng(seed)

    @property
    def step_std(self) -> float:
        """Standard deviation of a single Gaussian step, per dimension.

        Returns
        -------
        float
            ``sqrt(2 * diffusion_coefficient * dt)``.
        """
        return float(np.sqrt(2.0 * self.diffusion_coefficient * self.dt))

    def generate(
        self,
        n_steps: int,
        n_particles: int = 1,
        initial_positions: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate random-walk trajectories.

        Parameters
        ----------
        n_steps : int
            Number of time steps to simulate (must be >= 1).
        n_particles : int, optional
            Number of independent particles/trajectories to generate.
        initial_positions : np.ndarray of shape (n_particles, n_dim), optional
            Starting positions. Defaults to the origin for every particle.

        Returns
        -------
        np.ndarray of shape (n_steps + 1, n_particles, n_dim)
            Particle positions at each time step, including the initial
            position at index 0.

        Raises
        ------
        InvalidParameterError
            If ``n_steps < 1``, ``n_particles < 1``, or
            ``initial_positions`` does not have shape
            ``(n_particles, n_dim)``.
        """
        if n_steps < 1:
            raise InvalidParameterError("n_steps must be >= 1.")
        if n_particles < 1:
            raise InvalidParameterError("n_particles must be >= 1.")

        if initial_positions is None:
            initial_positions = np.zeros((n_particles, self.n_dim))
        else:
            initial_positions = np.asarray(initial_positions, dtype=float)
            if initial_positions.shape != (n_particles, self.n_dim):
                raise InvalidParameterError(
                    f"initial_positions must have shape " f"({n_particles}, {self.n_dim})."
                )

        steps = self._rng.normal(
            loc=0.0,
            scale=self.step_std,
            size=(n_steps, n_particles, self.n_dim),
        )

        trajectories = np.empty((n_steps + 1, n_particles, self.n_dim))
        trajectories[0] = initial_positions
        trajectories[1:] = initial_positions + np.cumsum(steps, axis=0)
        return trajectories

    def time_axis(self, n_steps: int) -> np.ndarray:
        """Return the time values corresponding to ``generate``'s output.

        Parameters
        ----------
        n_steps : int
            Number of time steps, matching the value passed to
            :meth:`generate`.

        Returns
        -------
        np.ndarray of shape (n_steps + 1,)
            Time values ``[0, dt, 2*dt, ..., n_steps*dt]``.
        """
        return np.arange(n_steps + 1) * self.dt
