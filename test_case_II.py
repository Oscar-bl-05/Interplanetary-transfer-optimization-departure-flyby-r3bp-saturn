import time

import numpy as np
from scipy.integrate import solve_ivp

from include import cts, analitical, IC
from include import plotter


# Case II test script for the Earth-Saturn transfer problem.

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

earth_x = cts.R_orb_A * np.cos(cts.frec_A * sol.t - IC.delta0)
earth_y = cts.R_orb_A * np.sin(cts.frec_A * sol.t - IC.delta0)

earth_distance = np.hypot(sol.y[0] - earth_x, sol.y[1] - earth_y)

after_departure = sol.t > 0.5 * cts.year_to_s

earth_distance_after_departure = earth_distance[after_departure]
time_after_departure = sol.t[after_departure]

closest_index = earth_distance_after_departure.argmin()

minimum_earth_distance = earth_distance_after_departure[closest_index]
time_of_closest_earth_return = time_after_departure[closest_index]

print("\n--- CASE II INITIAL GUESS ---")
print("theta_II (rad) =", theta_II)
print("dv_ign_II (km/s) =", dv_ign_II)
print("v_infII (km/s) =", float(analitical.v_infII))
print("deltaV_ignI analytical (km/s) =", float(analitical.deltaV_ignI))
print("dv_ign_II < deltaV_ignI: ", dv_ign_II < float(analitical.deltaV_ignI))
print("T_resonance (years) =", float(analitical.T_resonance) / (365.25 * 24 * 3600))
print("desired_a (km) =", float(analitical.desired_a))
print("desired_R_max (km) =", float(analitical.desired_R_max))
print("tf_case_II (years) =", tf / (365.25 * 24 * 3600))

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

print("\n--- EARTH RETURN CHECK ---")
print("minimum Earth distance after departure (km) =", minimum_earth_distance)
print("time of closest Earth return (years) =", time_of_closest_earth_return / cts.year_to_s)
print("Earth SOI radius (km) =", cts.earth_SOI_radius)
print("inside Earth SOI ? =", minimum_earth_distance < cts.earth_SOI_radius)
