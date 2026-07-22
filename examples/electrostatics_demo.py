"""Demo: point-charge field/potential, the finite-difference Poisson/Laplace
solver, equipotential extraction, and validation against three analytical
configurations (point charge, infinite line, parallel plates).

Run with:
    python examples/electrostatics_demo.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from shypyengine.core.constants import COULOMB_CONSTANT
from shypyengine.electrostatics import (
    BoundaryCondition,
    PointCharge,
    PoissonSolver2D,
    electric_field,
    electric_field_from_potential,
    electric_potential,
    extract_equipotential_lines,
)
from shypyengine.visualization import plot_equipotentials, plot_field_lines, plot_potential_map


def point_charge_dipole_demo() -> None:
    """Field, potential, and equipotential lines for a +/- charge dipole."""
    charges = [
        PointCharge(charge=1e-6, position=np.array([-1.0, 0.0])),
        PointCharge(charge=-1e-6, position=np.array([1.0, 0.0])),
    ]

    x = np.linspace(-4, 4, 120)  # even count: avoids landing exactly on a charge
    y = np.linspace(-4, 4, 120)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X, Y], axis=-1)

    field = electric_field(charges, points)
    Ex, Ey = field[..., 0], field[..., 1]
    V = electric_potential(charges, points)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_potential_map(X, Y, V, ax=ax1)
    plot_field_lines(X, Y, Ex, Ey, ax=ax2)
    fig.tight_layout()
    fig.savefig("electrostatics_dipole.png", dpi=150)

    lines = extract_equipotential_lines(X, Y, V, levels=[-2000, -500, 500, 2000])
    fig2, ax = plt.subplots(figsize=(6, 6))
    plot_potential_map(X, Y, V, ax=ax)
    plot_equipotentials(lines, ax=ax)
    fig2.tight_layout()
    fig2.savefig("electrostatics_equipotentials.png", dpi=150)

    print("Saved electrostatics_dipole.png and electrostatics_equipotentials.png")


def parallel_plate_capacitor_demo() -> None:
    """Solve the Laplace equation between two charged plates and check the
    result against the analytical parallel-plate capacitor field.
    """
    v0 = 10.0
    nx, ny = 61, 21
    dx = dy = 0.05
    solver = PoissonSolver2D(nx=nx, ny=ny, dx=dx, dy=dy)
    rho = np.zeros((ny, nx))

    V = solver.solve(
        rho,
        left=BoundaryCondition("dirichlet", v0),
        right=BoundaryCondition("dirichlet", -v0),
        top=BoundaryCondition("neumann"),
        bottom=BoundaryCondition("neumann"),
    )
    Ex, Ey = electric_field_from_potential(V, dx, dy)

    plate_separation = (nx - 1) * dx
    analytical_field = (2 * v0) / plate_separation
    simulated_field = Ex[1:-1, 1:-1].mean()
    print(
        f"Parallel plates: analytical E = {analytical_field:.4f} V/m, "
        f"simulated E = {simulated_field:.4f} V/m"
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_potential_map(solver.X, solver.Y, V, ax=ax)
    fig.tight_layout()
    fig.savefig("electrostatics_parallel_plates.png", dpi=150)
    print("Saved electrostatics_parallel_plates.png")


def point_charge_analytical_check() -> None:
    """Confirm the field/potential functions reproduce Coulomb's law."""
    q = 5e-6
    charge = PointCharge(charge=q, position=np.array([0.0, 0.0]))
    r = 2.0
    point = np.array([r, 0.0])

    field_magnitude = np.linalg.norm(electric_field([charge], point))
    potential = electric_potential([charge], point)

    print(
        f"Point charge: analytical |E| = {COULOMB_CONSTANT * q / r**2:.4f} V/m, "
        f"computed |E| = {field_magnitude:.4f} V/m"
    )
    print(
        f"Point charge: analytical V = {COULOMB_CONSTANT * q / r:.4f} V, "
        f"computed V = {potential:.4f} V"
    )


def main() -> None:
    point_charge_analytical_check()
    point_charge_dipole_demo()
    parallel_plate_capacitor_demo()


if __name__ == "__main__":
    main()
