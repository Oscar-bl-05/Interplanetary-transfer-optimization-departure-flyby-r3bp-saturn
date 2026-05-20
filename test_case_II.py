import time

import numpy as np
from scipy.integrate import solve_ivp

from include import cts, analitical, IC
from include import plotter


"""
Case II test script for the Earth-Saturn transfer problem.

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
tf = float(analitical.T_transfer_case_II)


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

print("\nInitializing case II simulation")

t_case_II_start = time.time()

theta_II = float(analitical.theta_0II)
dv_ign_II = float(analitical.deltaV_ignII)

# Initial condition
baseY0, t_hat_theta = IC.ICtoY0(
    IC.rho0,
    theta0=theta_II,
    delta0=IC.delta0,
)

V_ign = dv_ign_II * t_hat_theta
Y0 = baseY0.copy()
Y0[2:4] += V_ign

dt_event = (tf - t0) / nstep_opt

t_sim_II_start = time.time()

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

t_sim_II_end = time.time()

print("\n--- CASE II checks (without optimize) ---")
print("solver success =", sol.success)
print("solver message =", sol.message)
print("sim runtime (s) =", t_sim_II_end - t_sim_II_start)
print("total runtime (s) =", t_sim_II_end - t_case_II_start)

if len(sol.t_events[0]) == 0:
    rmax = np.hypot(sol.y[0], sol.y[1]).max()

    print("\nDid NOT reach R_B with non-optimized Case II parameters.")
    print("r_max =", rmax, "km")
    print("R_B =", float(cts.R_orb_B), "km")
    print("missing =", float(cts.R_orb_B) - rmax, "km")

else:
    t_hit = float(sol.t_events[0][0])
    y_hit = sol.y_events[0][0]
    r_hit = float(np.hypot(y_hit[0], y_hit[1]))

    print("\n--- HIT (without optimize) ---")
    print("t_hit (years) =", t_hit / (365.25 * 24 * 3600))
    print("r_hit (km) =", r_hit)
    print("r_hit - R_B (km) =", r_hit - float(cts.R_orb_B))
