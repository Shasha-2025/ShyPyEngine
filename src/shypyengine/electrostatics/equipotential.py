"""Equipotential line extraction from a 2D potential grid.

Equipotential lines are the level curves of ``V(x, y)``: every point on a
given curve has the same electric potential. They are always perpendicular
to the local electric field. Extraction uses the marching-squares
algorithm via ``contourpy`` (the same contouring engine Matplotlib uses
internally), so this module has no plotting dependency of its own.
"""

from __future__ import annotations

import contourpy
import numpy as np

from shypyengine.core.exceptions import InvalidParameterError


def extract_equipotential_lines(
    X: np.ndarray,
    Y: np.ndarray,
    V: np.ndarray,
    levels: list[float],
) -> dict[float, list[np.ndarray]]:
    """Extract equipotential lines (level curves of ``V``) at the given
    potential values.

    Parameters
    ----------
    X, Y : np.ndarray of shape (ny, nx)
        Coordinate meshgrids, e.g. ``PoissonSolver2D.X`` / ``.Y``.
    V : np.ndarray of shape (ny, nx)
        Potential grid, e.g. as returned by :meth:`PoissonSolver2D.solve`
        or :func:`shypyengine.electrostatics.point_charges.electric_potential`
        evaluated on a grid.
    levels : list of float
        Potential values at which to extract equipotential lines.

    Returns
    -------
    dict mapping float -> list of np.ndarray
        For each requested level, a list of polylines (each of shape
        ``(n_vertices, 2)``, columns are x, y) tracing that equipotential.
        A level may produce zero, one, or several disjoint polylines
        depending on the shape of ``V``. Levels outside the range of
        ``V`` produce an empty list.

    Raises
    ------
    InvalidParameterError
        If ``X``, ``Y``, and ``V`` do not all share the same shape, or if
        ``levels`` is empty.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    V = np.asarray(V, dtype=float)
    if not (X.shape == Y.shape == V.shape):
        raise InvalidParameterError(
            f"X, Y, and V must all share the same shape; got {X.shape}, " f"{Y.shape}, {V.shape}."
        )
    if len(levels) == 0:
        raise InvalidParameterError("levels must contain at least one value.")

    generator = contourpy.contour_generator(x=X, y=Y, z=V)
    return {level: generator.lines(level) for level in levels}
