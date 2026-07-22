"""Finite-difference solver for the 2D electrostatic Poisson equation
``nabla^2 V = -rho / epsilon_0``, and the charge-free special case, the
Laplace equation ``nabla^2 V = 0``.

Uses the standard 5-point finite-difference stencil on a uniform
rectangular grid, assembled as a sparse linear system and solved directly
(via ``scipy.sparse.linalg.spsolve``) rather than iterated to convergence,
since the discretized Poisson/Laplace equation is a linear boundary value
problem, not an initial value problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from shypyengine.core.constants import VACUUM_PERMITTIVITY
from shypyengine.core.exceptions import InvalidParameterError

_VALID_KINDS = ("dirichlet", "neumann")


@dataclass
class BoundaryCondition:
    """Boundary condition applied along one edge of the solver grid.

    Parameters
    ----------
    kind : {"dirichlet", "neumann"}
        ``"dirichlet"`` fixes the potential to ``value`` along the edge.
        ``"neumann"`` enforces a zero-flux (zero normal derivative,
        insulating/symmetric) boundary; ``value`` is ignored in that case.
        General nonzero-flux Neumann conditions are not supported.
    value : float or np.ndarray, optional
        Potential value(s) for a Dirichlet edge. Either a scalar
        (applied uniformly along the edge) or a 1D array matching the
        edge's length. Ignored for Neumann edges.

    Raises
    ------
    InvalidParameterError
        If ``kind`` is not ``"dirichlet"`` or ``"neumann"``.
    """

    kind: Literal["dirichlet", "neumann"]
    value: float | np.ndarray = 0.0

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise InvalidParameterError(f"kind must be one of {_VALID_KINDS}, got {self.kind!r}.")


def _edge_value(bc: BoundaryCondition, index: int, expected_length: int, edge_name: str) -> float:
    """Resolve a (possibly array-valued) Dirichlet boundary value at a
    single position along an edge.
    """
    if np.isscalar(bc.value):
        return float(bc.value)
    value_array = np.asarray(bc.value, dtype=float)
    if value_array.shape != (expected_length,):
        raise InvalidParameterError(
            f"{edge_name} boundary value array must have shape "
            f"({expected_length},), got {value_array.shape}."
        )
    return float(value_array[index])


class PoissonSolver2D:
    """Explicit finite-difference solver for the 2D Poisson/Laplace
    equation on a uniform rectangular grid.

    Parameters
    ----------
    nx, ny : int
        Number of grid points along x and y. Each must be >= 3.
    dx, dy : float
        Grid spacing along x and y. Each must be positive.

    Attributes
    ----------
    x, y : np.ndarray
        1D coordinate arrays along x (length ``nx``) and y (length ``ny``).
    X, Y : np.ndarray
        2D coordinate meshgrids, each of shape ``(ny, nx)``, matching the
        shape of ``rho`` and the returned potential.

    Raises
    ------
    InvalidParameterError
        If ``nx`` or ``ny`` is less than 3, or if ``dx`` or ``dy`` is not
        strictly positive.
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float) -> None:
        if nx < 3 or ny < 3:
            raise InvalidParameterError("nx and ny must each be >= 3.")
        if dx <= 0 or dy <= 0:
            raise InvalidParameterError("dx and dy must be positive.")

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.x = np.arange(nx) * dx
        self.y = np.arange(ny) * dy
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def _index(self, j: int, i: int) -> int:
        """Flatten a (row, col) = (y-index, x-index) pair to a 1D index."""
        return j * self.nx + i

    def solve(
        self,
        rho: np.ndarray,
        left: BoundaryCondition,
        right: BoundaryCondition,
        top: BoundaryCondition,
        bottom: BoundaryCondition,
        epsilon0: float = VACUUM_PERMITTIVITY,
    ) -> np.ndarray:
        """Solve for the electrostatic potential given a charge density
        and boundary conditions on all four edges.

        Solves ``nabla^2 V = -rho / epsilon0`` (the Laplace equation when
        ``rho`` is all zeros) with the 5-point finite-difference stencil.

        Parameters
        ----------
        rho : np.ndarray of shape (ny, nx)
            Charge density at each grid point, in C/m^3 (2D solver, so
            technically C/m^2 per unit depth). Pass an all-zero array to
            solve the charge-free Laplace equation.
        left, right : BoundaryCondition
            Boundary condition along the ``x = x_min`` / ``x = x_max``
            edges. Dirichlet array values, if given, must have length
            ``ny``.
        top, bottom : BoundaryCondition
            Boundary condition along the ``y = y_min`` / ``y = y_max``
            edges. Dirichlet array values, if given, must have length
            ``nx``.
        epsilon0 : float, optional
            Vacuum permittivity. Defaults to the physical value in
            :mod:`shypyengine.core.constants`.

        Returns
        -------
        np.ndarray of shape (ny, nx)
            The electrostatic potential at each grid point.

        Raises
        ------
        InvalidParameterError
            If ``rho`` does not have shape ``(ny, nx)``, or if all four
            boundaries are Neumann (the resulting linear system is
            singular; see Notes).

        Notes
        -----
        Corner nodes belong to two edges at once. Precedence is
        left/right first, then top/bottom: a corner takes its condition
        from the left or right edge, since that pass runs first and
        claims every row (including the corner rows) before the
        top/bottom pass runs.
        """
        nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy
        rho = np.asarray(rho, dtype=float)
        if rho.shape != (ny, nx):
            raise InvalidParameterError(f"rho must have shape ({ny}, {nx}), got {rho.shape}.")
        if all(bc.kind == "neumann" for bc in (left, right, top, bottom)):
            raise InvalidParameterError(
                "All four boundaries are Neumann (zero-flux); the potential is then "
                "only defined up to an arbitrary additive constant, and the "
                "discretized system is singular. Fix the potential at at least one "
                "boundary with a Dirichlet condition."
            )

        n = nx * ny
        A = sp.lil_matrix((n, n))
        b = np.zeros(n)
        assigned = np.zeros((ny, nx), dtype=bool)

        # Left edge (i = 0) and right edge (i = nx - 1): assigned first,
        # so they take precedence at the four corners.
        for j in range(ny):
            for i, bc, neighbor_i in ((0, left, 1), (nx - 1, right, nx - 2)):
                k = self._index(j, i)
                if bc.kind == "dirichlet":
                    A[k, k] = 1.0
                    b[k] = _edge_value(bc, j, ny, "left/right")
                else:  # zero-flux Neumann: mirror the nearest interior node
                    A[k, k] = 1.0
                    A[k, self._index(j, neighbor_i)] = -1.0
                    b[k] = 0.0
                assigned[j, i] = True

        # Top edge (j = 0) and bottom edge (j = ny - 1), skipping corners
        # already claimed by the left/right pass above.
        for i in range(nx):
            for j, bc, neighbor_j in ((0, top, 1), (ny - 1, bottom, ny - 2)):
                if assigned[j, i]:
                    continue
                k = self._index(j, i)
                if bc.kind == "dirichlet":
                    A[k, k] = 1.0
                    b[k] = _edge_value(bc, i, nx, "top/bottom")
                else:
                    A[k, k] = 1.0
                    A[k, self._index(neighbor_j, i)] = -1.0
                    b[k] = 0.0
                assigned[j, i] = True

        # Interior nodes: standard 5-point Poisson stencil.
        inv_dx2 = 1.0 / dx**2
        inv_dy2 = 1.0 / dy**2
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                k = self._index(j, i)
                A[k, k] = -2.0 * inv_dx2 - 2.0 * inv_dy2
                A[k, self._index(j, i + 1)] = inv_dx2
                A[k, self._index(j, i - 1)] = inv_dx2
                A[k, self._index(j + 1, i)] = inv_dy2
                A[k, self._index(j - 1, i)] = inv_dy2
                b[k] = -rho[j, i] / epsilon0

        V = spla.spsolve(A.tocsr(), b)
        return V.reshape(ny, nx)


def electric_field_from_potential(
    V: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the electric field from a potential grid via ``E = -grad(V)``.

    Uses second-order central differences in the interior and
    second-order one-sided differences at the domain edges (see
    ``numpy.gradient``).

    Parameters
    ----------
    V : np.ndarray of shape (ny, nx)
        Electrostatic potential grid, as returned by
        :meth:`PoissonSolver2D.solve`.
    dx, dy : float
        Grid spacing along x and y, matching the grid ``V`` was computed on.

    Returns
    -------
    Ex, Ey : np.ndarray, each of shape (ny, nx)
        Electric field components along x and y.
    """
    V = np.asarray(V, dtype=float)
    dVdy, dVdx = np.gradient(V, dy, dx)
    return -dVdx, -dVdy
