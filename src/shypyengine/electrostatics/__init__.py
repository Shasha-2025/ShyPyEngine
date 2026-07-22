"""Electrostatics module: Coulomb's law, field/potential superposition,
the finite-difference Poisson/Laplace solver, and equipotential
extraction.
"""

from shypyengine.electrostatics.equipotential import extract_equipotential_lines
from shypyengine.electrostatics.point_charges import (
    PointCharge,
    coulomb_force,
    electric_field,
    electric_potential,
)
from shypyengine.electrostatics.poisson import (
    BoundaryCondition,
    PoissonSolver2D,
    electric_field_from_potential,
)

__all__ = [
    "BoundaryCondition",
    "PointCharge",
    "PoissonSolver2D",
    "coulomb_force",
    "electric_field",
    "electric_field_from_potential",
    "electric_potential",
    "extract_equipotential_lines",
]
