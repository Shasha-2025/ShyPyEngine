"""Tests for the Brownian motion module (Phase 2).

Covers the random walk generator, MSD calculation, diffusion-coefficient
recovery against the known analytical scaling MSD(t) = 2*n_dim*D*t, and the
finite-difference diffusion equation solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from shypyengine.brownian.diffusion import DiffusionSolver1D
from shypyengine.brownian.msd import estimate_diffusion_coefficient, mean_squared_displacement
from shypyengine.brownian.random_walk import RandomWalk
from shypyengine.core.exceptions import InvalidParameterError

# numpy >= 2.0 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# RandomWalk
# ---------------------------------------------------------------------------


class TestRandomWalk:
    def test_output_shape(self):
        walk = RandomWalk(n_dim=3, diffusion_coefficient=1.0, dt=0.01, seed=0)
        traj = walk.generate(n_steps=100, n_particles=20)
        assert traj.shape == (101, 20, 3)

    def test_starts_at_origin_by_default(self):
        walk = RandomWalk(n_dim=2, diffusion_coefficient=0.5, dt=0.01, seed=1)
        traj = walk.generate(n_steps=50, n_particles=10)
        np.testing.assert_allclose(traj[0], np.zeros((10, 2)))

    def test_custom_initial_positions(self):
        walk = RandomWalk(n_dim=1, diffusion_coefficient=1.0, dt=0.1, seed=2)
        init = np.array([[5.0], [-5.0]])
        traj = walk.generate(n_steps=10, n_particles=2, initial_positions=init)
        np.testing.assert_allclose(traj[0], init)

    def test_reproducible_with_seed(self):
        w1 = RandomWalk(n_dim=2, diffusion_coefficient=1.0, dt=0.01, seed=42)
        w2 = RandomWalk(n_dim=2, diffusion_coefficient=1.0, dt=0.01, seed=42)
        t1 = w1.generate(n_steps=100, n_particles=5)
        t2 = w2.generate(n_steps=100, n_particles=5)
        np.testing.assert_allclose(t1, t2)

    def test_time_axis(self):
        walk = RandomWalk(n_dim=1, diffusion_coefficient=1.0, dt=0.5, seed=0)
        t = walk.time_axis(n_steps=4)
        np.testing.assert_allclose(t, [0.0, 0.5, 1.0, 1.5, 2.0])

    @pytest.mark.parametrize("bad_dim", [0, 4, -1])
    def test_rejects_invalid_n_dim(self, bad_dim):
        with pytest.raises(InvalidParameterError):
            RandomWalk(n_dim=bad_dim, diffusion_coefficient=1.0, dt=0.01)

    def test_rejects_nonpositive_diffusion_coefficient(self):
        with pytest.raises(InvalidParameterError):
            RandomWalk(n_dim=1, diffusion_coefficient=0.0, dt=0.01)

    def test_rejects_nonpositive_dt(self):
        with pytest.raises(InvalidParameterError):
            RandomWalk(n_dim=1, diffusion_coefficient=1.0, dt=0.0)

    def test_rejects_n_steps_less_than_one(self):
        walk = RandomWalk(n_dim=2, diffusion_coefficient=1.0, dt=0.01, seed=0)
        with pytest.raises(InvalidParameterError):
            walk.generate(n_steps=0, n_particles=5)

    def test_rejects_n_particles_less_than_one(self):
        walk = RandomWalk(n_dim=2, diffusion_coefficient=1.0, dt=0.01, seed=0)
        with pytest.raises(InvalidParameterError):
            walk.generate(n_steps=10, n_particles=0)

    def test_rejects_mismatched_initial_positions_shape(self):
        walk = RandomWalk(n_dim=2, diffusion_coefficient=1.0, dt=0.01, seed=0)
        wrong_shape_init = np.zeros((3, 1))  # n_particles=5 expected, dim mismatch too
        with pytest.raises(InvalidParameterError):
            walk.generate(n_steps=10, n_particles=5, initial_positions=wrong_shape_init)


# ---------------------------------------------------------------------------
# MSD + diffusion coefficient recovery (validation against analytical result)
# ---------------------------------------------------------------------------


class TestMSDAndDiffusionRecovery:
    def test_msd_zero_at_t0(self):
        walk = RandomWalk(n_dim=2, diffusion_coefficient=1.0, dt=0.01, seed=0)
        traj = walk.generate(n_steps=200, n_particles=500)
        msd = mean_squared_displacement(traj)
        assert msd[0] == pytest.approx(0.0)

    def test_msd_is_monotonically_increasing_on_average(self):
        walk = RandomWalk(n_dim=1, diffusion_coefficient=1.0, dt=0.01, seed=0)
        traj = walk.generate(n_steps=500, n_particles=1000)
        msd = mean_squared_displacement(traj)
        # Not strictly monotonic sample-to-sample, but the overall trend
        # from start to end must be a large increase.
        assert msd[-1] > msd[len(msd) // 2] > msd[0]

    @pytest.mark.parametrize("n_dim", [1, 2, 3])
    def test_diffusion_coefficient_recovered_within_tolerance(self, n_dim):
        true_d = 2.5
        dt = 0.01
        walk = RandomWalk(n_dim=n_dim, diffusion_coefficient=true_d, dt=dt, seed=7)
        traj = walk.generate(n_steps=2000, n_particles=2000)
        time = walk.time_axis(n_steps=2000)
        msd = mean_squared_displacement(traj)

        estimated_d = estimate_diffusion_coefficient(time, msd, n_dim=n_dim)

        # Large ensemble (2000 particles) over many steps should recover D
        # to within 10%.
        assert estimated_d == pytest.approx(true_d, rel=0.1)

    def test_msd_shape_matches_mismatch_raises(self):
        with pytest.raises(InvalidParameterError):
            estimate_diffusion_coefficient(np.array([0.0, 1.0]), np.array([0.0]), n_dim=1)

    def test_mean_squared_displacement_rejects_wrong_ndim(self):
        # trajectories must be (n_steps+1, n_particles, n_dim), i.e. ndim == 3
        flat_trajectories = np.zeros((10, 3))
        with pytest.raises(InvalidParameterError):
            mean_squared_displacement(flat_trajectories)

    def test_estimate_diffusion_coefficient_rejects_invalid_n_dim(self):
        time = np.array([0.0, 1.0, 2.0])
        msd = np.array([0.0, 2.0, 4.0])
        with pytest.raises(InvalidParameterError):
            estimate_diffusion_coefficient(time, msd, n_dim=4)

    def test_estimate_diffusion_coefficient_rejects_all_zero_time(self):
        time = np.zeros(5)
        msd = np.zeros(5)
        with pytest.raises(InvalidParameterError):
            estimate_diffusion_coefficient(time, msd, n_dim=1)


# ---------------------------------------------------------------------------
# DiffusionSolver1D
# ---------------------------------------------------------------------------


class TestDiffusionSolver1D:
    def test_rejects_unstable_parameters(self):
        # alpha = D * dt / dx**2 must be <= 0.5
        with pytest.raises(InvalidParameterError):
            DiffusionSolver1D(diffusion_coefficient=1.0, dx=0.1, dt=1.0, n_grid=50)

    def test_mass_conservation_with_reflective_boundaries(self):
        solver = DiffusionSolver1D(diffusion_coefficient=1.0, dx=0.1, dt=0.001, n_grid=101)
        c0 = solver.initial_gaussian(mean=solver.x[50], sigma=0.5)
        history = solver.solve(c0, n_steps=200)

        initial_mass = _trapz(history[0], solver.x)
        final_mass = _trapz(history[-1], solver.x)
        assert final_mass == pytest.approx(initial_mass, rel=1e-3)

    def test_gaussian_spreads_out_over_time(self):
        solver = DiffusionSolver1D(diffusion_coefficient=1.0, dx=0.1, dt=0.001, n_grid=101)
        c0 = solver.initial_gaussian(mean=solver.x[50], sigma=0.3)
        history = solver.solve(c0, n_steps=500)

        # The peak concentration should decrease as the pulse spreads.
        assert history[-1].max() < history[0].max()

    def test_output_shape(self):
        solver = DiffusionSolver1D(diffusion_coefficient=0.5, dx=0.2, dt=0.01, n_grid=30)
        c0 = solver.initial_gaussian(mean=solver.x[15], sigma=1.0)
        history = solver.solve(c0, n_steps=10)
        assert history.shape == (11, 30)

    def test_rejects_nonpositive_diffusion_coefficient(self):
        with pytest.raises(InvalidParameterError):
            DiffusionSolver1D(diffusion_coefficient=0.0, dx=0.1, dt=0.001, n_grid=50)

    def test_rejects_nonpositive_dx_or_dt(self):
        with pytest.raises(InvalidParameterError):
            DiffusionSolver1D(diffusion_coefficient=1.0, dx=0.0, dt=0.001, n_grid=50)
        with pytest.raises(InvalidParameterError):
            DiffusionSolver1D(diffusion_coefficient=1.0, dx=0.1, dt=0.0, n_grid=50)

    def test_rejects_too_few_grid_points(self):
        with pytest.raises(InvalidParameterError):
            DiffusionSolver1D(diffusion_coefficient=1.0, dx=0.1, dt=0.001, n_grid=2)

    def test_step_rejects_wrong_shape_profile(self):
        solver = DiffusionSolver1D(diffusion_coefficient=1.0, dx=0.1, dt=0.001, n_grid=50)
        with pytest.raises(InvalidParameterError):
            solver.step(np.zeros(10))

    def test_solve_rejects_n_steps_less_than_one(self):
        solver = DiffusionSolver1D(diffusion_coefficient=1.0, dx=0.1, dt=0.001, n_grid=50)
        c0 = solver.initial_gaussian(mean=solver.x[25], sigma=0.5)
        with pytest.raises(InvalidParameterError):
            solver.solve(c0, n_steps=0)

    def test_variance_growth_matches_analytical_diffusion(self):
        # Regression test against the analytical result for free diffusion:
        # variance of the concentration profile grows as sigma^2(t) = sigma_0^2 + 2*D*t.
        d = 0.8
        sigma0 = 0.4
        solver = DiffusionSolver1D(diffusion_coefficient=d, dx=0.05, dt=0.0005, n_grid=401)
        c0 = solver.initial_gaussian(mean=solver.x[200], sigma=sigma0)
        n_steps = 400
        history = solver.solve(c0, n_steps=n_steps)
        elapsed_time = n_steps * solver.dt

        def variance(profile):
            mass = _trapz(profile, solver.x)
            mean_x = _trapz(solver.x * profile, solver.x) / mass
            return _trapz(((solver.x - mean_x) ** 2) * profile, solver.x) / mass

        var0 = variance(history[0])
        var_final = variance(history[-1])
        expected_var_final = sigma0**2 + 2 * d * elapsed_time

        assert var0 == pytest.approx(sigma0**2, rel=0.05)
        assert var_final == pytest.approx(expected_var_final, rel=0.05)
