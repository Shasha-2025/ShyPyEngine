"""Shared physical constants used across ShyPyEngine modules.

Values are in SI units unless otherwise noted.
"""

from __future__ import annotations

#: Boltzmann constant, J/K
BOLTZMANN_CONSTANT: float = 1.380649e-23

#: Vacuum permittivity, F/m (used by the electrostatics module)
VACUUM_PERMITTIVITY: float = 8.8541878128e-12

#: Coulomb's constant, N*m^2/C^2 (used by the electrostatics module)
COULOMB_CONSTANT: float = 8.9875517923e9
