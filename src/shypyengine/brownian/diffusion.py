"""Finite-difference solver for the 1D diffusion equation.

Solves ``∂c/∂t = D * ∂²c/∂x²`` on a uniform grid using an explicit
forward-time, centered-space (FTCS) scheme, subject to the stability
condition ``D * dt / dx**2 <= 0.5``.
"""

from __future__ import annotations

import numpy as np

from shypyengine.core.exceptions import InvalidParameterError


class DiffusionSolver1D:
    """Explicit finite-difference solver for the 1D diffusion equation.

    Parameters
    ----------
    diffusion_coefficient : float
        Diffusion coefficient D. Must be positive.
    dx : float
        Grid spacing. Must be positive.
    dt : float
        Time step. Must be positive and satisfy the FTCS stability
        condition ``D * dt / dx**2 <= 0.5``.
    n_grid : int
        Number of grid points. Must be >= 3.

    Raises
    ------
    InvalidParameterError
        If ``diffusion_coefficient``, ``dx``, or ``dt`` is not
        strictly positive, if ``n_grid < 3``, or if the FTCS
        stability condition ``D * dt / dx**2 <= 0.5`` is violated.
    """

    def __init__(
        self,
        diffusion_coefficient: float,
        dx: float,
        dt: float,
        n_grid: int,
    ) -> None:
        if diffusion_coefficient <= 0:
            raise InvalidParameterError("diffusion_coefficient must be positive.")
        if dx <= 0 or dt <= 0:
            raise InvalidParameterError("dx and dt must be positive.")
        if n_grid < 3:
            raise InvalidParameterError("n_grid must be >= 3.")

        self.diffusion_coefficient = diffusion_coefficient
        self.dx = dx
        self.dt = dt
        self.n_grid = n_grid

        self.alpha = diffusion_coefficient * dt / dx**2
        if self.alpha > 0.5:
            raise InvalidParameterError(
                "Stability condition violated: D * dt / dx**2 = "
                f"{self.alpha:.4f} > 0.5. Reduce dt or increase dx."
            )

        self.x = np.linspace(0.0, (n_grid - 1) * dx, n_grid)

    def initial_gaussian(self, mean: float, sigma: float, amplitude: float = 1.0) -> np.ndarray:
        """Build a Gaussian pulse as an initial concentration profile.

        Parameters
        ----------
        mean : float
            Center of the pulse, in the same units as ``self.x``.
        sigma : float
            Standard deviation (width) of the pulse. Must be positive
            for a well-defined pulse shape.
        amplitude : float, optional
            Peak concentration value at ``x = mean``.

        Returns
        -------
        np.ndarray of shape (n_grid,)
            Concentration profile evaluated on ``self.x``.
        """
        return amplitude * np.exp(-((self.x - mean) ** 2) / (2.0 * sigma**2))

    def step(self, c: np.ndarray) -> np.ndarray:
        """Advance the concentration profile ``c`` by one time step.

        Uses zero-flux (reflective) boundary conditions at both ends.

        Parameters
        ----------
        c : np.ndarray of shape (n_grid,)
            Concentration profile at the current time step.

        Returns
        -------
        np.ndarray of shape (n_grid,)
            Concentration profile after one FTCS update.

        Raises
        ------
        InvalidParameterError
            If ``c`` does not have shape ``(n_grid,)``.
        """
        c = np.asarray(c, dtype=float)
        if c.shape != (self.n_grid,):
            raise InvalidParameterError(f"c must have shape ({self.n_grid},).")

        c_new = c.copy()
        c_new[1:-1] = c[1:-1] + self.alpha * (c[2:] - 2.0 * c[1:-1] + c[:-2])
        # Reflective (Neumann, zero-flux) boundaries
        c_new[0] = c_new[1]
        c_new[-1] = c_new[-2]
        return c_new

    def solve(self, c0: np.ndarray, n_steps: int) -> np.ndarray:
        """Run the solver for ``n_steps`` and return the full time history.

        Parameters
        ----------
        c0 : np.ndarray of shape (n_grid,)
            Initial concentration profile.
        n_steps : int
            Number of time steps to advance.

        Returns
        -------
        np.ndarray of shape (n_steps + 1, n_grid)
            Concentration profile at each time step, including the initial
            profile at index 0.

        Raises
        ------
        InvalidParameterError
            If ``n_steps < 1``.
        """
        if n_steps < 1:
            raise InvalidParameterError("n_steps must be >= 1.")

        history = np.empty((n_steps + 1, self.n_grid))
        history[0] = np.asarray(c0, dtype=float)
        for i in range(n_steps):
            history[i + 1] = self.step(history[i])
        return history
