import time
import numpy as np
from scipy.integrate import solve_ivp

from include import cts, analitical, IC
from include import optimizer


print("Initializing simulation, pls wait ...")

# Tolerances selected from err_check.py
atol = np.array([1e-2, 1e-2, 1e-6, 1e-6])
rtol = 1e-9

nstep_opt = 750
t0 = 0.0


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


def check_initial_guess(resonance):
    tf = resonance["T_transfer_case_II"]

    theta_II = resonance["theta_0II"]
    dv_ign_II = resonance["deltaV_ignII"]

    baseY0, t_hat_theta = IC.ICtoY0(
        IC.rho0,
        theta0=theta_II,
        delta0=IC.delta0,
    )

    V_ign = dv_ign_II * t_hat_theta

    Y0 = baseY0.copy()
    Y0[2:4] += V_ign

    def reach_RB(t, Y):
        return np.hypot(Y[0], Y[1]) - cts.R_orb_B

    reach_RB.terminal = True
    reach_RB.direction = 1

    dt_event = (tf - t0) / nstep_opt

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
    print("resonance =", str(resonance["n"]) + ":" + str(resonance["n_earth"]))
    print("theta_II (rad) =", theta_II)
    print("dv_ign_II (km/s) =", dv_ign_II)
    print("v_infII (km/s) =", resonance["v_infII"])
    print("deltaV_ignI analytical (km/s) =", float(analitical.deltaV_ignI))
    print("dv_ign_II < deltaV_ignI: ", dv_ign_II < float(analitical.deltaV_ignI))
    print("T_resonance (years) =", resonance["T_resonance"]/cts.year_to_s)
    print("desired_a (km) =", resonance["desired_a"])
    print("desired_R_max (km) =", resonance["desired_R_max"])
    print("tf_case_II (years) =", tf/cts.year_to_s)

    print("\n--- CASE II checks (without optimize) ---")
    print("solver success =", sol.success)
    print("solver message =", sol.message)

    if len(sol.t_events[0]) == 0:
        rmax = np.hypot(sol.y[0], sol.y[1]).max()

        print("\nDid NOT reach R_B with non-optimized Case II parameters.")
        print("r_max =", int(rmax//1000000), "Gm")
        print("R_B =", int(float(cts.R_orb_B)//1000000), "Gm")
        print("missing =", int((float(cts.R_orb_B) - rmax)//1000000), "Gm")

    else:
        t_hit = float(sol.t_events[0][0])
        y_hit = sol.y_events[0][0]
        r_hit = float(np.hypot(y_hit[0], y_hit[1]))

        print("\n--- HIT (without optimize) ---")
        print("t_hit (years) =", t_hit/cts.year_to_s)
        print("r_hit (km) =", r_hit)
        print("r_hit - R_B (km) =", r_hit - float(cts.R_orb_B))

    print("\n--- EARTH RETURN CHECK ---")
    print("minimum Earth distance after departure (km) =", minimum_earth_distance)
    print("time of closest Earth return (years) =", time_of_closest_earth_return/cts.year_to_s)
    print("Earth SOI radius (km) =", cts.earth_SOI_radius)
    print("inside Earth SOI ? =", minimum_earth_distance < cts.earth_SOI_radius)
    print("above Earth surface ? =", minimum_earth_distance > cts.R_Earth)


print("\nInitializing case II simulation")

t_start = time.time()

# Se enseña el caso 1:12 porque es el relevante para Tierra-Saturno,
# pero la optimización real barre 2..12.
resonance_12 = analitical.resonance_case_II_estimate(n=1, n_earth=12)

check_initial_guess(resonance_12)

print("\n--- CASE II RESONANCE SWEEP ---")

t_sweep_start = time.time()

best_caseII, sweep_results = optimizer.optimize_case_II_resonance_sweep(
    F=F,
    nstep=nstep_opt,
    atol=atol,
    rtol=rtol,
    t0=t0,
    n=1,
    n_earth_values=[12],
    n_grid_deltav=7,
    n_grid_theta=7,
    n_refines=0,
    min_flyby_altitude_km=300.0,
    max_flyby_altitude_km=350000.0
)

t_sweep_end = time.time()

print("\nCase II sweep time =", t_sweep_end - t_sweep_start)

if best_caseII is None:
    print("\nNo valid Case II solution found.")

else:
    print("\n--- OPTIMUM (Case II) ---")
    print("resonance =", str(best_caseII["resonance_n"]) + ":" + str(best_caseII["resonance_n_earth"]))
    print("theta_opt_II (rad) =", float(best_caseII["theta"]))
    print("dv_ign_opt_II (km/s) =", float(best_caseII["dv_ign"]))
    print("dv_fin_opt_II (km/s) =", float(best_caseII["dv_fin"]))
    print("dv_tot_opt_II (km/s) =", float(best_caseII["dv_tot"]))
    print("t_fin_opt_II (years) =", float(best_caseII["t_fin"]) / cts.year_to_s)

    print("\n--- EARTH FLYBY CHECK ---")
    print("t_SOI_in (years) =", float(best_caseII["t_SOI_in"]) / cts.year_to_s)
    print("t_SOI_out (years) =", float(best_caseII["t_SOI_out"]) / cts.year_to_s)
    print("minimum Earth distance after departure (km) =", float(best_caseII["MED"]))
    print("minimum altitude over Earth (km) =", float(best_caseII["minimum_altitude"]))
    print("time of closest Earth return (years) =", float(best_caseII["t_MED"]) / cts.year_to_s)
    print("Earth SOI radius (km) =", cts.earth_SOI_radius)
    print("inside Earth SOI ? =", best_caseII["MED"] < cts.earth_SOI_radius)
    print("above Earth surface ? =", best_caseII["MED"] > cts.R_Earth)
    print("dv_ign_II < deltaV_ignI ? =", best_caseII["dv_ign"] < float(analitical.deltaV_ignI))

    print("\n--- FLYBY ENERGY CHECK ---")
    print("heliocentric energy before (km^2/s^2) =", best_caseII["energy_before"])
    print("heliocentric energy after (km^2/s^2) =", best_caseII["energy_after"])
    print("energy gain (km^2/s^2) =", best_caseII["energy_gain"])
    print("post-flyby aphelion (Gm) =", best_caseII["r_apo_after"]/1000000)

print("\nTotal runtime =", time.time() - t_start)