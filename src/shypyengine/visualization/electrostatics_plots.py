"""Basic static (Matplotlib) plots for the electrostatics module: field
lines, potential maps, and equipotential overlays.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from shypyengine.core.exceptions import InvalidParameterError


def plot_field_lines(
    X: np.ndarray,
    Y: np.ndarray,
    Ex: np.ndarray,
    Ey: np.ndarray,
    ax: plt.Axes | None = None,
    density: float = 1.2,
) -> plt.Axes:
    """Plot electric field lines as a streamplot.

    Parameters
    ----------
    X, Y : np.ndarray of shape (ny, nx)
        Coordinate meshgrids.
    Ex, Ey : np.ndarray of shape (ny, nx)
        Electric field components, e.g. from
        :func:`shypyengine.electrostatics.point_charges.electric_field`
        evaluated on a grid, or
        :func:`shypyengine.electrostatics.poisson.electric_field_from_potential`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if omitted.
    density : float, optional
        Streamline density, passed through to ``Axes.streamplot``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the field lines were drawn on.

    Raises
    ------
    InvalidParameterError
        If ``X``, ``Y``, ``Ex``, and ``Ey`` do not all share the same shape.
    """
    if not (X.shape == Y.shape == Ex.shape == Ey.shape):
        raise InvalidParameterError("X, Y, Ex, and Ey must all share the same shape.")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    magnitude = np.sqrt(Ex**2 + Ey**2)
    ax.streamplot(X, Y, Ex, Ey, color=magnitude, cmap="viridis", density=density)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Electric Field Lines")
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_potential_map(
    X: np.ndarray,
    Y: np.ndarray,
    V: np.ndarray,
    ax: plt.Axes | None = None,
    cmap: str = "RdBu_r",
) -> plt.Axes:
    """Plot the electric potential as a filled contour (heat) map.

    Parameters
    ----------
    X, Y : np.ndarray of shape (ny, nx)
        Coordinate meshgrids.
    V : np.ndarray of shape (ny, nx)
        Potential grid.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if omitted.
    cmap : str, optional
        Matplotlib colormap name. Defaults to a diverging colormap, which
        is a natural fit for potentials that change sign.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the potential map was drawn on.

    Raises
    ------
    InvalidParameterError
        If ``X``, ``Y``, and ``V`` do not all share the same shape.
    """
    if not (X.shape == Y.shape == V.shape):
        raise InvalidParameterError("X, Y, and V must all share the same shape.")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    mesh = ax.contourf(X, Y, V, levels=50, cmap=cmap)
    ax.figure.colorbar(mesh, ax=ax, label="Potential (V)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Electric Potential Map")
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_equipotentials(
    equipotential_lines: dict[float, list[np.ndarray]],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay equipotential lines on an axes.

    Parameters
    ----------
    equipotential_lines : dict mapping float -> list of np.ndarray
        Output of
        :func:`shypyengine.electrostatics.equipotential.extract_equipotential_lines`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created if omitted (useful
        for overlaying on top of :func:`plot_potential_map` or
        :func:`plot_field_lines` by passing the same ``ax``).

    Returns
    -------
    matplotlib.axes.Axes
        The axes the equipotential lines were drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    for level, polylines in equipotential_lines.items():
        for i, line in enumerate(polylines):
            # Only label the first polyline per level to avoid a
            # duplicate legend entry per disjoint segment.
            label = f"V = {level:.3g}" if i == 0 else None
            ax.plot(line[:, 0], line[:, 1], color="black", linewidth=0.8, label=label)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Equipotential Lines")
    ax.set_aspect("equal", adjustable="box")
    if equipotential_lines:
        ax.legend(fontsize=7, loc="upper right")
    return ax
