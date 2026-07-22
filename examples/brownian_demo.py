"""Demo: generate Brownian trajectories, compute MSD, recover D, and solve
the 1D diffusion equation, with plots for each.

Run with:
    python examples/brownian_demo.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from shypyengine.brownian import (
    DiffusionSolver1D,
    RandomWalk,
    estimate_diffusion_coefficient,
    mean_squared_displacement,
)
from shypyengine.visualization import plot_diffusion_profiles, plot_msd, plot_trajectories_2d


def main() -> None:
    # 1. Random walk in 2D + trajectory plot
    walk = RandomWalk(n_dim=2, diffusion_coefficient=1.0, dt=0.01, seed=0)
    trajectories = walk.generate(n_steps=2000, n_particles=15)

    fig1, ax1 = plt.subplots(figsize=(6, 6))
    plot_trajectories_2d(trajectories, ax=ax1)
    fig1.tight_layout()
    fig1.savefig("brownian_trajectories.png", dpi=150)

    # 2. MSD + diffusion coefficient recovery
    ensemble_walk = RandomWalk(n_dim=2, diffusion_coefficient=1.0, dt=0.01, seed=1)
    ensemble_traj = ensemble_walk.generate(n_steps=2000, n_particles=1000)
    time = ensemble_walk.time_axis(n_steps=2000)
    msd = mean_squared_displacement(ensemble_traj)
    estimated_d = estimate_diffusion_coefficient(time, msd, n_dim=2)
    print(f"True D = 1.0, estimated D = {estimated_d:.4f}")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    plot_msd(time, msd, ax=ax2)
    fig2.tight_layout()
    fig2.savefig("brownian_msd.png", dpi=150)

    # 3. 1D diffusion equation solver
    solver = DiffusionSolver1D(diffusion_coefficient=1.0, dx=0.1, dt=0.001, n_grid=101)
    c0 = solver.initial_gaussian(mean=solver.x[50], sigma=0.3)
    history = solver.solve(c0, n_steps=500)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    plot_diffusion_profiles(solver.x, history, n_curves=5, ax=ax3)
    fig3.tight_layout()
    fig3.savefig("brownian_diffusion_profiles.png", dpi=150)

    print("Saved: brownian_trajectories.png, brownian_msd.png, brownian_diffusion_profiles.png")


if __name__ == "__main__":
    main()
