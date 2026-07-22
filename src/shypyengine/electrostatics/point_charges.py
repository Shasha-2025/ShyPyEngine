"""Point-charge electrostatics: Coulomb's law, electric field and potential
via superposition.

All functions accept charge positions and field points in 2D or 3D
Cartesian coordinates and are fully vectorized over arbitrary point grids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shypyengine.core.constants import COULOMB_CONSTANT
from shypyengine.core.exceptions import InvalidParameterError

#: Minimum separation (in the same length units as positions) below which
#: a field point is considered coincident with a charge, and therefore the
#: 1/r^2 (field) or 1/r (potential) singularity is undefined.
_MIN_SEPARATION = 1e-12


@dataclass(frozen=True)
class PointCharge:
    """A single point charge.

    Parameters
    ----------
    charge : float
        Charge magnitude, in coulombs. May be positive or negative.
    position : np.ndarray
        Position of the charge, shape ``(n_dim,)`` with ``n_dim`` in
        ``{2, 3}``.

    Raises
    ------
    InvalidParameterError
        If ``position`` is not a 1D array of length 2 or 3.
    """

    charge: float
    position: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float)
        if position.ndim != 1 or position.shape[0] not in (2, 3):
            raise InvalidParameterError(
                "PointCharge.position must be a 1D array of length 2 or 3, "
                f"got shape {position.shape}."
            )
        # dataclass is frozen, so bypass __setattr__ to normalize the dtype
        object.__setattr__(self, "position", position)

    @property
    def n_dim(self) -> int:
        """Spatial dimensionality of this charge's position (2 or 3)."""
        return self.position.shape[0]


def _validate_charges(charges: list[PointCharge]) -> int:
    """Check that all charges share the same dimensionality and return it."""
    if len(charges) == 0:
        raise InvalidParameterError("charges must contain at least one PointCharge.")
    n_dim = charges[0].n_dim
    for c in charges:
        if c.n_dim != n_dim:
            raise InvalidParameterError(
                "All charges must have the same dimensionality; "
                f"got both {n_dim} and {c.n_dim}."
            )
    return n_dim


def _prepare_points(points: np.ndarray, n_dim: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Validate ``points`` against ``n_dim`` and return it reshaped to
    ``(n_points, n_dim)`` along with the original leading shape.
    """
    points = np.asarray(points, dtype=float)
    if points.shape[-1] != n_dim:
        raise InvalidParameterError(
            f"points must have last axis of length {n_dim} to match the "
            f"charges' dimensionality, got shape {points.shape}."
        )
    leading_shape = points.shape[:-1]
    return points.reshape(-1, n_dim), leading_shape


def coulomb_force(
    q1: float,
    q2: float,
    r1: np.ndarray,
    r2: np.ndarray,
    k: float = COULOMB_CONSTANT,
) -> np.ndarray:
    """Coulomb's law: the force on charge ``q1`` at ``r1`` due to charge
    ``q2`` at ``r2``.

    ``F = k * q1 * q2 * (r1 - r2) / |r1 - r2|^3``

    The result points away from ``r2`` (repulsive) when ``q1`` and ``q2``
    have the same sign, and toward ``r2`` (attractive) otherwise.

    Parameters
    ----------
    q1, q2 : float
        Charge magnitudes, in coulombs.
    r1, r2 : np.ndarray
        Positions of charge 1 and charge 2, each shape ``(n_dim,)``.
    k : float, optional
        Coulomb's constant. Defaults to the physical value in
        :mod:`shypyengine.core.constants`.

    Returns
    -------
    np.ndarray of shape (n_dim,)
        Force vector on charge 1, in newtons.

    Raises
    ------
    InvalidParameterError
        If ``r1`` and ``r2`` are separated by less than ``1e-12`` (in the
        given length units), where the force is undefined.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    displacement = r1 - r2
    distance = np.linalg.norm(displacement)
    if distance < _MIN_SEPARATION:
        raise InvalidParameterError(
            "coulomb_force is undefined when r1 and r2 coincide " f"(distance = {distance:.3e})."
        )
    return k * q1 * q2 * displacement / distance**3


def electric_field(
    charges: list[PointCharge],
    points: np.ndarray,
    k: float = COULOMB_CONSTANT,
) -> np.ndarray:
    """Electric field at one or more points via superposition of point
    charges.

    ``E(r) = sum_i [ k * qi * (r - ri) / |r - ri|^3 ]``

    Parameters
    ----------
    charges : list of PointCharge
        Source charges. Must all share the same dimensionality (2 or 3).
    points : np.ndarray of shape (..., n_dim)
        Field point(s) at which to evaluate E. Any number of leading
        dimensions is supported (e.g. a single point of shape ``(n_dim,)``,
        or a 2D grid of shape ``(ny, nx, n_dim)``).
    k : float, optional
        Coulomb's constant. Defaults to the physical value in
        :mod:`shypyengine.core.constants`.

    Returns
    -------
    np.ndarray of shape (..., n_dim)
        Electric field vector at each input point, matching the leading
        shape of ``points``.

    Raises
    ------
    InvalidParameterError
        If ``charges`` is empty, charges have inconsistent dimensionality,
        ``points`` does not match that dimensionality, or a field point
        coincides with a charge (the field is singular there).
    """
    n_dim = _validate_charges(charges)
    flat_points, leading_shape = _prepare_points(points, n_dim)

    field = np.zeros_like(flat_points)
    for c in charges:
        displacement = flat_points - c.position  # (n_points, n_dim)
        distance = np.linalg.norm(displacement, axis=-1)  # (n_points,)
        if np.any(distance < _MIN_SEPARATION):
            raise InvalidParameterError(
                "electric_field is undefined at a point coincident with a "
                "charge (the 1/r^2 singularity)."
            )
        field += k * c.charge * displacement / distance[:, np.newaxis] ** 3

    return field.reshape(*leading_shape, n_dim)


def electric_potential(
    charges: list[PointCharge],
    points: np.ndarray,
    k: float = COULOMB_CONSTANT,
) -> np.ndarray:
    """Electric potential at one or more points via superposition of point
    charges.

    ``V(r) = sum_i [ k * qi / |r - ri| ]``

    Parameters
    ----------
    charges : list of PointCharge
        Source charges. Must all share the same dimensionality (2 or 3).
    points : np.ndarray of shape (..., n_dim)
        Field point(s) at which to evaluate V. Any number of leading
        dimensions is supported, as in :func:`electric_field`.
    k : float, optional
        Coulomb's constant. Defaults to the physical value in
        :mod:`shypyengine.core.constants`.

    Returns
    -------
    np.ndarray of shape (...)
        Electric potential at each input point, matching the leading
        shape of ``points`` (with the trailing ``n_dim`` axis removed).

    Raises
    ------
    InvalidParameterError
        If ``charges`` is empty, charges have inconsistent dimensionality,
        ``points`` does not match that dimensionality, or a field point
        coincides with a charge (the potential is singular there).
    """
    n_dim = _validate_charges(charges)
    flat_points, leading_shape = _prepare_points(points, n_dim)

    potential = np.zeros(flat_points.shape[0])
    for c in charges:
        distance = np.linalg.norm(flat_points - c.position, axis=-1)
        if np.any(distance < _MIN_SEPARATION):
            raise InvalidParameterError(
                "electric_potential is undefined at a point coincident "
                "with a charge (the 1/r singularity)."
            )
        potential += k * c.charge / distance

    return potential.reshape(leading_shape)
