import time

import numpy as np
from scipy.integrate import solve_ivp

from include import cts, analitical, IC
from include import plotter
from include.optimizer import optimize_case_I


"""
Case I optimization script for the Earth-Saturn transfer problem.

This script:
1. Optimizes the initial angle theta and ignition delta-v.
2. Propagates the optimal trajectory until it reaches R_B.
3. Prints the optimal delta-v budget.
4. Generates the required plots for the optimal trajectory.

Numerical tolerance validation is performed separately in err_check.py.
Case II is not implemented here (for teh moment).
"""


print("Initializing simulation, pls wait ...")

# Tolerances selected from err_check.py
atol = np.array([1e-2, 1e-2, 1e-6, 1e-6])
rtol = 1e-9

# Reference tolerances used only for plotting error curves.
# The tolerance validation itself is done in err_check.py.
atol_ref = np.array([1e-6, 1e-6, 1e-10, 1e-10])
rtol_ref = 1e-12

nstep_opt = 750
nstep_plot = 4000

t0 = 0.0
tf = float(analitical.T_transfer)


def R(t, R_orb, frec):
    return np.array([
        R_orb * np.cos(frec * t - IC.delta0),
        R_orb * np.sin(frec * t - IC.delta0),
    ])


def F(t, Y):
    x, y, vx, vy = Y

    r = np.hypot(x, y)
    mu_r3 = cts.mu_sun / (r**3)

    Rx, Ry = R(t, cts.R_orb_A, cts.frec_A)
    Rm = np.hypot(Rx, Ry)
    Rm3 = 1.0 / (Rm**3)

    dx = x - Rx
    dy = y - Ry
    dm = np.hypot(dx, dy)
    dm3 = 1.0 / (dm**3)

    ax = (-x * mu_r3) - cts.mu_earth * (dx * dm3 + Rx * Rm3)
    ay = (-y * mu_r3) - cts.mu_earth * (dy * dm3 + Ry * Rm3)

    return np.array([vx, vy, ax, ay])


def reach_RB(t, Y):
    return np.hypot(Y[0], Y[1]) - cts.R_orb_B


reach_RB.terminal = True
reach_RB.direction = 1


print("\nStarting optimization...")
t_opt_start = time.time()

best = optimize_case_I(
    F=F,
    nstep=nstep_opt,
    atol=atol,
    rtol=rtol,
    tf=tf,
    t0=t0,
    n_grid=10,
    n_refines=2,
)

t_opt_end = time.time()

if best is None:
    print("No valid (theta, dv_ign) reached R_B in the explored grids.")
    raise SystemExit(1)


theta_opt = float(best["theta"])
dv_ign_opt = float(best["dv_ign"])
dv_fin_opt = float(best["dv_fin"])
dv_tot_opt = float(best["dv_tot"])
t_fin_opt = float(best["t_fin"])

print("\n--- OPTIMUM (Case I) ---")
print("theta_opt (rad) =", theta_opt)
print("dv_ign_opt (km/s) =", dv_ign_opt)
print("dv_fin_opt (km/s) =", dv_fin_opt)
print("dv_tot_opt (km/s) =", dv_tot_opt)
print("t_fin_opt (years) =", t_fin_opt / (365.25 * 24 * 3600))
print("optimizer runtime (s) =", t_opt_end - t_opt_start)


# Build optimal initial condition
baseY0, t_hat_theta = IC.ICtoY0(
    IC.rho0,
    theta0=theta_opt,
    delta0=IC.delta0,
)

V_ign = dv_ign_opt * t_hat_theta
Y0 = baseY0.copy()
Y0[2:4] += V_ign


# Propagate optimal trajectory until R_B
dt_event = (tf - t0) / nstep_opt

t_sim_start = time.time()

sol = solve_ivp(
    F,
    (t0, tf),
    Y0,
    method="DOP853",
    atol=atol,
    rtol=rtol,
    events=reach_RB,
    max_step=dt_event,
)

t_sim_end = time.time()

if len(sol.t_events[0]) == 0:
    rmax = np.hypot(sol.y[0], sol.y[1]).max()

    print("\nDid NOT reach R_B with optimal parameters.")
    print("r_max =", rmax, "km")
    print("missing =", cts.R_orb_B - rmax, "km")

    raise SystemExit(1)


t_hit = float(sol.t_events[0][0])
y_hit = sol.y_events[0][0]
r_hit = float(np.hypot(y_hit[0], y_hit[1]))

print("\n--- HIT (Case I optimum) ---")
print("t_hit (years) =", t_hit / (365.25 * 24 * 3600))
print("r_hit (km) =", r_hit)
print("r_hit - R_B (km) =", r_hit - float(cts.R_orb_B))
print("sim runtime (s) =", t_sim_end - t_sim_start)


# Plots of the optimal trajectory
t_plot = np.linspace(t0, t_hit, nstep_plot + 1, endpoint=True)
dt_plot = (t_hit - t0) / nstep_plot

sol_plot = solve_ivp(
    F,
    (t0, t_hit),
    Y0,
    t_eval=t_plot,
    method="DOP853",
    atol=atol,
    rtol=rtol,
)

sol_plot_ref = solve_ivp(
    F,
    (t0, t_hit),
    Y0,
    t_eval=t_plot,
    method="DOP853",
    atol=atol_ref,
    rtol=rtol_ref,
)

print("\nPlotting optimum trajectory and errors...")

plotter.plot_solution(sol_plot.t, sol_plot.y, sol_plot_ref.y)
plotter.plot2D(sol_plot.t, dt_plot, sol_plot.y, R)