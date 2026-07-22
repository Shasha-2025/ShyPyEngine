"""Tests for the electrostatics module (Phase 1).

Covers Coulomb's law, field/potential superposition, the finite-difference
Poisson/Laplace solver, and equipotential extraction. Includes analytical
validation against three textbook configurations: a single point charge,
an (approximated) infinite line charge, and a parallel-plate capacitor.
"""

from __future__ import annotations

import numpy as np
import pytest

from shypyengine.core.constants import COULOMB_CONSTANT
from shypyengine.core.exceptions import InvalidParameterError
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

# ---------------------------------------------------------------------------
# Coulomb's law
# ---------------------------------------------------------------------------


class TestCoulombForce:
    def test_like_charges_repel(self):
        force = coulomb_force(q1=1e-6, q2=1e-6, r1=np.array([1.0, 0.0]), r2=np.array([0.0, 0.0]))
        # Force on q1 should point away from q2, i.e. in +x.
        assert force[0] > 0
        assert force[1] == pytest.approx(0.0, abs=1e-20)

    def test_opposite_charges_attract(self):
        force = coulomb_force(q1=1e-6, q2=-1e-6, r1=np.array([1.0, 0.0]), r2=np.array([0.0, 0.0]))
        assert force[0] < 0

    def test_matches_analytical_magnitude(self):
        q1, q2 = 2e-6, 3e-6
        r = 0.5
        force = coulomb_force(q1, q2, r1=np.array([r, 0.0]), r2=np.array([0.0, 0.0]))
        expected_magnitude = COULOMB_CONSTANT * abs(q1 * q2) / r**2
        assert np.linalg.norm(force) == pytest.approx(expected_magnitude, rel=1e-9)

    def test_rejects_coincident_points(self):
        with pytest.raises(InvalidParameterError):
            coulomb_force(1.0, 1.0, r1=np.array([0.0, 0.0]), r2=np.array([0.0, 0.0]))

    def test_newtons_third_law(self):
        r1, r2 = np.array([1.0, 2.0]), np.array([-3.0, 0.5])
        f_on_1 = coulomb_force(1e-6, -2e-6, r1, r2)
        f_on_2 = coulomb_force(-2e-6, 1e-6, r2, r1)
        np.testing.assert_allclose(f_on_1, -f_on_2)


# ---------------------------------------------------------------------------
# PointCharge validation
# ---------------------------------------------------------------------------


class TestPointCharge:
    def test_rejects_bad_position_shape(self):
        with pytest.raises(InvalidParameterError):
            PointCharge(charge=1.0, position=np.array([[1.0, 2.0]]))

    def test_rejects_wrong_dimensionality(self):
        with pytest.raises(InvalidParameterError):
            PointCharge(charge=1.0, position=np.array([1.0, 2.0, 3.0, 4.0]))

    def test_n_dim_property(self):
        c2 = PointCharge(charge=1.0, position=np.array([1.0, 2.0]))
        c3 = PointCharge(charge=1.0, position=np.array([1.0, 2.0, 3.0]))
        assert c2.n_dim == 2
        assert c3.n_dim == 3


# ---------------------------------------------------------------------------
# Analytical validation: single point charge
# ---------------------------------------------------------------------------


class TestPointChargeAnalytical:
    def test_field_matches_coulombs_law(self):
        q = 5e-6
        charge = PointCharge(charge=q, position=np.array([0.0, 0.0]))
        r = 2.0
        point = np.array([r, 0.0])

        field = electric_field([charge], point)
        expected_magnitude = COULOMB_CONSTANT * q / r**2

        assert np.linalg.norm(field) == pytest.approx(expected_magnitude, rel=1e-9)
        # Field of a positive charge points radially outward.
        np.testing.assert_allclose(field / np.linalg.norm(field), [1.0, 0.0], atol=1e-9)

    def test_potential_matches_analytical_formula(self):
        q = 5e-6
        charge = PointCharge(charge=q, position=np.array([0.0, 0.0]))
        r = 3.0
        point = np.array([0.0, r])

        potential = electric_potential([charge], point)
        expected = COULOMB_CONSTANT * q / r

        assert potential == pytest.approx(expected, rel=1e-9)

    def test_field_and_potential_vectorize_over_grid(self):
        charge = PointCharge(charge=1e-6, position=np.array([0.0, 0.0]))
        x = np.linspace(1, 5, 4)
        y = np.linspace(1, 5, 4)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X, Y], axis=-1)  # (4, 4, 2)

        field = electric_field([charge], points)
        potential = electric_potential([charge], points)

        assert field.shape == (4, 4, 2)
        assert potential.shape == (4, 4)

    def test_field_undefined_at_charge_location(self):
        charge = PointCharge(charge=1.0, position=np.array([0.0, 0.0]))
        with pytest.raises(InvalidParameterError):
            electric_field([charge], np.array([0.0, 0.0]))

    def test_rejects_empty_charge_list(self):
        with pytest.raises(InvalidParameterError):
            electric_field([], np.array([1.0, 0.0]))

    def test_rejects_mismatched_point_dimensionality(self):
        charge = PointCharge(charge=1.0, position=np.array([0.0, 0.0]))
        with pytest.raises(InvalidParameterError):
            electric_field([charge], np.array([1.0, 0.0, 0.0]))

    def test_rejects_charges_with_inconsistent_dimensionality(self):
        c2 = PointCharge(charge=1.0, position=np.array([0.0, 0.0]))
        c3 = PointCharge(charge=1.0, position=np.array([0.0, 0.0, 0.0]))
        with pytest.raises(InvalidParameterError):
            electric_field([c2, c3], np.array([1.0, 1.0]))

    def test_potential_undefined_at_charge_location(self):
        charge = PointCharge(charge=1.0, position=np.array([0.0, 0.0]))
        with pytest.raises(InvalidParameterError):
            electric_potential([charge], np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# Analytical validation: infinite line charge (approximated by a long,
# finite chain of point charges, evaluated near its midpoint)
# ---------------------------------------------------------------------------


class TestInfiniteLineChargeApproximation:
    def test_field_matches_infinite_line_scaling_near_midpoint(self):
        # Approximate an infinite line with linear charge density lambda
        # along the x-axis using many closely spaced point charges, and
        # evaluate the field perpendicular to the midpoint, where end
        # effects are negligible. Analytical result: E = 2 k lambda / r.
        linear_density = 1e-9  # C/m
        half_length = 50.0  # m; long compared to the evaluation distance
        n_charges = 4001
        positions = np.linspace(-half_length, half_length, n_charges)
        segment_length = positions[1] - positions[0]
        charge_per_segment = linear_density * segment_length

        charges = [
            PointCharge(charge=charge_per_segment, position=np.array([x, 0.0])) for x in positions
        ]

        r = 0.5  # m; small compared to half_length, so end effects are negligible
        point = np.array([0.0, r])
        field = electric_field(charges, point)

        expected_magnitude = 2 * COULOMB_CONSTANT * linear_density / r
        assert np.linalg.norm(field) == pytest.approx(expected_magnitude, rel=1e-3)
        # Field should point purely radially outward (along +y) by symmetry.
        assert field[0] == pytest.approx(0.0, abs=1e-6 * expected_magnitude)
        assert field[1] > 0


# ---------------------------------------------------------------------------
# PoissonSolver2D
# ---------------------------------------------------------------------------


class TestPoissonSolver2D:
    def test_rejects_too_small_grid(self):
        with pytest.raises(InvalidParameterError):
            PoissonSolver2D(nx=2, ny=5, dx=0.1, dy=0.1)

    def test_rejects_nonpositive_spacing(self):
        with pytest.raises(InvalidParameterError):
            PoissonSolver2D(nx=5, ny=5, dx=0.0, dy=0.1)

    def test_rejects_wrong_rho_shape(self):
        solver = PoissonSolver2D(nx=5, ny=5, dx=0.1, dy=0.1)
        with pytest.raises(InvalidParameterError):
            solver.solve(
                rho=np.zeros((4, 4)),
                left=BoundaryCondition("dirichlet", 0.0),
                right=BoundaryCondition("dirichlet", 0.0),
                top=BoundaryCondition("dirichlet", 0.0),
                bottom=BoundaryCondition("dirichlet", 0.0),
            )

    def test_rejects_invalid_boundary_kind(self):
        with pytest.raises(InvalidParameterError):
            BoundaryCondition(kind="robin", value=0.0)

    def test_dirichlet_box_with_no_charge_is_flat_when_boundaries_equal(self):
        # Laplace equation with the same potential on all four edges must
        # give that constant potential everywhere (uniqueness of the
        # Laplace equation's solution).
        solver = PoissonSolver2D(nx=11, ny=11, dx=0.1, dy=0.1)
        rho = np.zeros((11, 11))
        bc = BoundaryCondition("dirichlet", 7.0)
        V = solver.solve(rho, left=bc, right=bc, top=bc, bottom=bc)
        np.testing.assert_allclose(V, 7.0, atol=1e-9)

    def test_array_valued_dirichlet_boundary(self):
        # A left edge whose potential varies along y should be reproduced
        # exactly in the solution's left column (a Dirichlet condition is
        # an exact constraint, not something the solver approximates).
        nx, ny = 9, 6
        solver = PoissonSolver2D(nx=nx, ny=ny, dx=0.1, dy=0.1)
        rho = np.zeros((ny, nx))
        left_profile = np.linspace(0.0, 5.0, ny)
        V = solver.solve(
            rho,
            left=BoundaryCondition("dirichlet", left_profile),
            right=BoundaryCondition("dirichlet", 0.0),
            top=BoundaryCondition("dirichlet", 0.0),
            bottom=BoundaryCondition("dirichlet", 0.0),
        )
        np.testing.assert_allclose(V[:, 0], left_profile, atol=1e-9)

    def test_rejects_mismatched_array_boundary_length(self):
        solver = PoissonSolver2D(nx=9, ny=6, dx=0.1, dy=0.1)
        rho = np.zeros((6, 9))
        with pytest.raises(InvalidParameterError):
            solver.solve(
                rho,
                left=BoundaryCondition("dirichlet", np.zeros(3)),  # wrong length
                right=BoundaryCondition("dirichlet", 0.0),
                top=BoundaryCondition("dirichlet", 0.0),
                bottom=BoundaryCondition("dirichlet", 0.0),
            )

    def test_neumann_on_left_and_right_mirrors_interior_at_boundary(self):
        # With Neumann (zero-flux) on every edge, the problem is
        # under-determined up to an additive constant, but the boundary
        # values must still equal their nearest interior neighbor by
        # construction. Fix the mean with one Dirichlet corner-adjacent
        # point instead by checking left/right mirroring directly via a
        # simple, well-posed setup: Dirichlet top/bottom, Neumann left/right.
        nx, ny = 7, 9
        solver = PoissonSolver2D(nx=nx, ny=ny, dx=0.1, dy=0.1)
        rho = np.zeros((ny, nx))
        V = solver.solve(
            rho,
            left=BoundaryCondition("neumann"),
            right=BoundaryCondition("neumann"),
            top=BoundaryCondition("dirichlet", 3.0),
            bottom=BoundaryCondition("dirichlet", -3.0),
        )
        # Zero-flux left/right means the left and right columns equal
        # their nearest interior neighbor column.
        np.testing.assert_allclose(V[:, 0], V[:, 1], atol=1e-9)
        np.testing.assert_allclose(V[:, -1], V[:, -2], atol=1e-9)

    def test_rejects_all_neumann_boundaries(self):
        # A fully insulated box has no unique potential (only defined up
        # to an arbitrary additive constant); the discretized system is
        # singular. The solver must reject this explicitly rather than
        # silently returning NaN from a singular linear solve.
        solver = PoissonSolver2D(nx=5, ny=5, dx=0.1, dy=0.1)
        rho = np.zeros((5, 5))
        neumann = BoundaryCondition("neumann")
        with pytest.raises(InvalidParameterError):
            solver.solve(rho, left=neumann, right=neumann, top=neumann, bottom=neumann)

    def test_minimum_grid_size_solves_without_error(self):
        # nx = ny = 3 is the smallest permitted grid (a single interior
        # node surrounded by boundary on all sides).
        solver = PoissonSolver2D(nx=3, ny=3, dx=0.1, dy=0.1)
        rho = np.zeros((3, 3))
        v0 = 2.0
        V = solver.solve(
            rho,
            left=BoundaryCondition("dirichlet", v0),
            right=BoundaryCondition("dirichlet", v0),
            top=BoundaryCondition("dirichlet", v0),
            bottom=BoundaryCondition("dirichlet", v0),
        )
        np.testing.assert_allclose(V, v0, atol=1e-9)

    def test_asymmetric_grid_nx_not_equal_ny(self):
        # Regression guard against accidentally transposing nx/ny inside
        # the solver, which would only surface on a non-square grid.
        nx, ny = 15, 6
        solver = PoissonSolver2D(nx=nx, ny=ny, dx=0.1, dy=0.2)
        assert solver.X.shape == (ny, nx)
        assert solver.Y.shape == (ny, nx)
        rho = np.zeros((ny, nx))
        v0 = 4.0
        V = solver.solve(
            rho,
            left=BoundaryCondition("dirichlet", v0),
            right=BoundaryCondition("dirichlet", v0),
            top=BoundaryCondition("dirichlet", v0),
            bottom=BoundaryCondition("dirichlet", v0),
        )
        assert V.shape == (ny, nx)
        np.testing.assert_allclose(V, v0, atol=1e-9)


# ---------------------------------------------------------------------------
# Analytical validation: parallel-plate capacitor
# ---------------------------------------------------------------------------


class TestParallelPlateCapacitorAnalytical:
    def test_uniform_field_between_plates(self):
        # Two infinite parallel plates (approximated by Dirichlet
        # boundaries on the left/right edges, with Neumann/insulating
        # top and bottom to enforce translational symmetry along y) held
        # at +V0 and -V0 give, in the charge-free gap, a linear potential
        # and a uniform field E = (V_left - V_right) / plate_separation.
        v0 = 10.0
        nx, ny = 41, 5
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

        plate_separation = (nx - 1) * dx
        expected_field_magnitude = (2 * v0) / plate_separation

        Ex, Ey = electric_field_from_potential(V, dx, dy)

        # Field should be uniform and point in +x (from the high-potential
        # left plate to the low-potential right plate), with negligible
        # y-component away from the top/bottom edges.
        interior = Ex[1:-1, 1:-1]
        assert interior.std() < 0.02 * expected_field_magnitude
        np.testing.assert_allclose(interior.mean(), expected_field_magnitude, rtol=0.02)
        np.testing.assert_allclose(Ey[1:-1, 1:-1], 0.0, atol=1e-8 * expected_field_magnitude)

    def test_potential_is_linear_between_plates(self):
        v0 = 5.0
        nx, ny = 21, 5
        dx = dy = 0.1
        solver = PoissonSolver2D(nx=nx, ny=ny, dx=dx, dy=dy)
        rho = np.zeros((ny, nx))

        V = solver.solve(
            rho,
            left=BoundaryCondition("dirichlet", v0),
            right=BoundaryCondition("dirichlet", -v0),
            top=BoundaryCondition("neumann"),
            bottom=BoundaryCondition("neumann"),
        )

        expected_profile = np.linspace(v0, -v0, nx)
        for row in V[1:-1]:
            np.testing.assert_allclose(row, expected_profile, atol=1e-9)


# ---------------------------------------------------------------------------
# Method-of-manufactured-solutions convergence test
# ---------------------------------------------------------------------------


class TestPoissonSolverConvergence:
    """Method-of-manufactured-solutions test: pick a known analytical
    potential, derive the charge density that produces it, solve
    numerically, and check that the error shrinks as O(dx^2) under grid
    refinement -- the accuracy the finite-difference stencil is
    theoretically expected to have (see docs/numerical_methods.md).
    """

    @staticmethod
    def _manufactured_solution_error(n: int) -> float:
        from shypyengine.core.constants import VACUUM_PERMITTIVITY

        dx = dy = 1.0 / (n - 1)
        solver = PoissonSolver2D(nx=n, ny=n, dx=dx, dy=dy)
        X, Y = solver.X, solver.Y

        # V = sin(pi x) sin(pi y) vanishes on all four edges of [0,1]x[0,1],
        # so a simple V=0 Dirichlet boundary is exact for this solution.
        v_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
        rho = 2 * VACUUM_PERMITTIVITY * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)

        zero = BoundaryCondition("dirichlet", 0.0)
        v_numerical = solver.solve(rho, left=zero, right=zero, top=zero, bottom=zero)
        return float(np.max(np.abs(v_numerical - v_exact)))

    def test_error_scales_quadratically_with_grid_spacing(self):
        errors = [self._manufactured_solution_error(n) for n in (11, 21, 41, 81)]
        # Halving dx should reduce the max error by a factor of ~4 for a
        # second-order-accurate scheme. Allow generous tolerance (3.5-4.5)
        # since this is a numerical, not exact, convergence rate.
        for e_coarse, e_fine in zip(errors, errors[1:]):
            ratio = e_coarse / e_fine
            assert 3.5 < ratio < 4.5, f"expected ~4x error reduction, got {ratio:.2f}x"


# ---------------------------------------------------------------------------
# Equipotential line extraction
# ---------------------------------------------------------------------------


class TestEquipotentialExtraction:
    def test_extracts_circular_contours_around_point_charge(self):
        charge = PointCharge(charge=1e-6, position=np.array([0.0, 0.0]))
        # Even point count deliberately avoids a grid point landing exactly
        # on the origin, where the potential is singular.
        x = np.linspace(-3, 3, 120)
        y = np.linspace(-3, 3, 120)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X, Y], axis=-1)
        V = electric_potential([charge], points)

        target_r = 1.5
        level = COULOMB_CONSTANT * 1e-6 / target_r
        lines = extract_equipotential_lines(X, Y, V, levels=[level])

        assert level in lines
        assert len(lines[level]) >= 1
        polyline = lines[level][0]
        radii = np.linalg.norm(polyline, axis=1)
        # Equipotentials of a point charge are circles at r = kq/V.
        np.testing.assert_allclose(radii, target_r, rtol=0.05)

    def test_rejects_mismatched_shapes(self):
        X = np.zeros((5, 5))
        Y = np.zeros((5, 5))
        V = np.zeros((4, 4))
        with pytest.raises(InvalidParameterError):
            extract_equipotential_lines(X, Y, V, levels=[1.0])

    def test_rejects_empty_levels(self):
        X = np.zeros((5, 5))
        Y = np.zeros((5, 5))
        V = np.zeros((5, 5))
        with pytest.raises(InvalidParameterError):
            extract_equipotential_lines(X, Y, V, levels=[])
