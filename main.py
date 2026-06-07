import time

import numpy as np
from scipy.integrate import solve_ivp

from include import cts, analitical, IC, plotter
from include.optimizer import optimize_case_I, optimize_case_II_resonance_sweep


print("Initializing simulation, pls wait ...")

# Tolerances selected from err_check.py
atol = np.array([1e-2, 1e-2, 1e-6, 1e-6])
rtol = 1e-9

# Reference tolerances used only for plotting error curves.
# The tolerance validation itself is done in err_check.py.
atol_ref = np.array([1e-6, 1e-6, 1e-10, 1e-10])
rtol_ref = 1e-12

nstep_opt = 800
nstep_opt2 = 800*4
nstep_plot = 4000

opt1_n_grid = 10 #number of values to scan in the case I optimization
opt1_n_refines = 1 #number of refinements in case I optimization

min_flyby_altitude_km = 300 # minimum desired altitude above Earth's surface in flyby to prevent aerodynamic drag

n_earth_values=[12] #list of resonances (years) values to test ### to speed up code execution only the value 12 is selected, but in reality the optimization was done with 2...12
opt2_n_grid_deltav=20 #number of deltaV values to scan in the case II optimization
opt2_n_grid_theta=10 #number of theta values to scan in the case II optimization
opt2_n_refines=0 #number of refinements in case II optimization

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

def reach_RB(t, Y):# detecta la llegada a punto lagrange de planeta B
    return np.hypot(Y[0], Y[1]) - cts.R_orb_B
reach_RB.terminal = True
reach_RB.direction = 1


#### #### CASE I #### ####

if input("Run case I optimization (Y/n):\n") in ["Y","y"]:

    print("\nStarting case I optimization...")
    print(f"Estimated time = {opt1_n_grid**2*opt1_n_refines*0.085} s")
    t0 = 0.0
    tf1 = float(analitical.T_transfer_case_I) 

    t_opt1_start = time.time()

    best = optimize_case_I(
        F=F,
        nstep=nstep_opt,
        atol=atol,
        rtol=rtol,
        tf=tf1,
        t0=t0,
        n_grid=opt1_n_grid,
        n_refines=opt1_n_refines,
    )

    t_opt1_end = time.time()

    if best is None:
        print("No valid (theta, dv_ign) reached R_B in the explored grids.")
        raise SystemExit(1)


    theta_opt1 = float(best["theta"])
    dv_ign_opt1 = float(best["dv_ign"])
    dv_fin_opt1 = float(best["dv_fin"])
    dv_tot_opt1 = float(best["dv_tot"])
    t_fin_opt1 = float(best["t_fin"])

    print("\n--- OPTIMUM (Case I) ---")
    print("theta_opt (rad) =", theta_opt1)
    print("dv_ign_opt (km/s) =", dv_ign_opt1)
    print("dv_fin_opt (km/s) =", dv_fin_opt1)
    print("dv_tot_opt (km/s) =", dv_tot_opt1)
    print("t_fin_opt (years) =", t_fin_opt1 / cts.year2seconds)
    print("optimizer runtime (s) =", t_opt1_end - t_opt1_start)


    # Testing Case I solution
    baseY0, t_hat_theta = IC.ICtoY0(
        IC.rho0,
        theta0=theta_opt1,
        delta0=IC.delta0,
    )

    V_ign = dv_ign_opt1 * t_hat_theta
    Y0 = baseY0.copy()
    Y0[2:4] += V_ign

    dt_event = (tf1 - t0) / nstep_opt

    t_sim_start = time.time()

    sol = solve_ivp(
        F,
        (t0, tf1),
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
    print("t_hit (years) =", t_hit / cts.year2seconds)
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

    print("\nGenerating plots for case I...\n-------------------X-------------------")

    plotter.plot_solution(sol_plot.t, sol_plot.y, sol_plot_ref.y)
    plotter.plot2D(sol_plot.t, dt_plot, sol_plot.y, R)
    plotter.plot_heliocentric_trajectory(sol_plot.t, sol_plot.y, R, title="Case I optimum - heliocentric trajectory")
    plotter.plot_geocentric_trajectory(sol_plot.t, sol_plot.y, R, title="Case I optimum - Earth-centered trajectory")
    plotter.plot_distances(sol_plot.t, sol_plot.y, R, title="Case I optimum",)
    plotter.plot_orbital_elements(sol_plot.t, sol_plot.y, R, center="sun", title="Case I optimum - heliocentric elements")
    plotter.plot_orbital_elements(sol_plot.t, sol_plot.y, R, center="earth", title="Case I optimum - Earth-relative departure elements", time_window=(t0, 0.20 * cts.year2seconds))

#### #### CASE II #### ####

def check_initial_guess(resonance):
    tf2 = resonance["T_transfer_case_II"]

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

    dt_event = (tf2 - t0) / nstep_opt

    sol = solve_ivp(
        F,
        (t0, tf2),
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

    after_departure = sol.t > 0.5 * cts.year2seconds

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
    print("T_resonance (years) =", resonance["T_resonance"]/cts.year2seconds)
    print("desired_a (km) =", resonance["desired_a"])
    print("desired_R_max (km) =", resonance["desired_R_max"])
    print("tf_case_II (years) =", tf2/cts.year2seconds)

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
        print("t_hit (years) =", t_hit/cts.year2seconds)
        print("r_hit (Gm) =", r_hit//1000000)
        print("r_hit - R_B (Gm) =", (r_hit - float(cts.R_orb_B))//1000000)

    print("\n--- EARTH RETURN CHECK ---")
    print("minimum Earth distance after departure (km) =", minimum_earth_distance)
    print("time of closest Earth return (years) =", time_of_closest_earth_return/cts.year2seconds)
    print("Earth SOI radius (km) =", cts.earth_SOI_radius)
    print("inside Earth SOI ? =", minimum_earth_distance < cts.earth_SOI_radius)
    print("above Earth surface ? =", minimum_earth_distance > cts.R_Earth)

if input("Run case II optimization (Y/n):\n") in ["Y","y"]:

    t_opt2_start = time.time()

    print("\nInitializing case II simulation...")
    t0 = 0.0
    tf2 = float(analitical.T_transfer_case_II)

    # The case 1:12 is shown simply as a relevant sample to showcase the code
    resonance_12 = analitical.resonance_case_II_estimate(n=1, n_earth=12)
    check_initial_guess(resonance_12)

    print("\n--- CASE II RESONANCE SWEEP ---")

    t_sweep_start = time.time()

    best_caseII, sweep_results = optimize_case_II_resonance_sweep(
        F=F,
        nstep=nstep_opt2,
        atol=atol,
        rtol=rtol,
        t0=t0,
        n=1,
        n_earth_values=[12],
        n_grid_deltav=opt2_n_grid_deltav,
        n_grid_theta=opt2_n_grid_theta,
        n_refines=opt2_n_refines,
        min_flyby_altitude_km=min_flyby_altitude_km,
        max_flyby_altitude_km=cts.earth_SOI_radius
    )

    t_sweep_end = time.time()

    print("\nCase II sweep time =", t_sweep_end - t_sweep_start, "s")

    if best_caseII is None:
        print("\nNo valid Case II solution found.")

    else:
        theta_opt2 = float(best_caseII["theta"])
        dv_ign_opt2 = float(best_caseII["dv_ign"])
        dv_fin_opt2 = float(best_caseII["dv_fin"])
        dv_tot_opt2 = float(best_caseII["dv_tot"])
        t_fin_opt2 = float(best_caseII["t_fin"])

        resonance_n_opt2 = int(best_caseII["resonance_n"])
        resonance_n_earth_opt2 = int(best_caseII["resonance_n_earth"])

        t_SOI_in_opt2 = float(best_caseII["t_SOI_in"])
        t_SOI_out_opt2 = float(best_caseII["t_SOI_out"])
        MED_opt2 = float(best_caseII["MED"])
        minimum_altitude_opt2 = float(best_caseII["minimum_altitude"])
        t_MED_opt2 = float(best_caseII["t_MED"])

        energy_before_opt2 = float(best_caseII["energy_before"])
        energy_after_opt2 = float(best_caseII["energy_after"])
        energy_gain_opt2 = float(best_caseII["energy_gain"])
        r_apo_after_opt2 = float(best_caseII["r_apo_after"])


        print("\n--- OPTIMUM (Case II) ---")
        print("resonance =", f"{resonance_n_opt2}:{resonance_n_earth_opt2}")
        print("theta_opt2 (rad) =", theta_opt2)
        print("dv_ign_opt2 (km/s) =", dv_ign_opt2)
        print("dv_fin_opt2 (km/s) =", dv_fin_opt2)
        print("dv_tot_opt2 (km/s) =", dv_tot_opt2)
        print("t_fin_opt2 (years) =", t_fin_opt2 / cts.year2seconds)

        print("\n--- CASE II flyby checks ---")
        print("t_SOI_in (years) =", t_SOI_in_opt2 / cts.year2seconds)
        print("t_SOI_out (years) =", t_SOI_out_opt2 / cts.year2seconds)
        print("minimum Earth distance MED (km) =", MED_opt2)
        print("minimum flyby altitude (km) =", minimum_altitude_opt2)
        print("t_MED (years) =", t_MED_opt2 / cts.year2seconds)

        print("\n--- CASE II energy checks ---")
        print("energy_before (km^2/s^2) =", energy_before_opt2)
        print("energy_after (km^2/s^2) =", energy_after_opt2)
        print("energy_gain (km^2/s^2) =", energy_gain_opt2)
        print("r_apo_after (km) =", r_apo_after_opt2)

        # Build optimal initial condition for case II
        baseY0_2, t_hat_theta2 = IC.ICtoY0(
            IC.rho0,
            theta0=theta_opt2,
            delta0=IC.delta0,
        )

        V_ign2 = dv_ign_opt2 * t_hat_theta2

        Y0_2 = baseY0_2.copy()
        Y0_2[2:4] += V_ign2

        # Reconstruct complete optimal Case II trajectory
        t_hit2 = t_fin_opt2
        y_hit2 = best_caseII["y_fin"]
        r_hit2 = float(np.hypot(y_hit2[0], y_hit2[1]))

        print("\n--- HIT (Case II optimum) ---")
        print("t_hit2 (years) =", t_hit2 / cts.year2seconds)
        print("r_hit2 (km) =", r_hit2)
        print("r_hit2 - R_B (km) =", r_hit2 - float(cts.R_orb_B))

        print("\nGenerating plots for case II...")

        # Full time vector for the complete plotted trajectory.
        t_plot2 = np.linspace(t0, t_hit2, nstep_plot + 1, endpoint=True)
        dt_plot2 = (t_hit2 - t0) / nstep_plot

        mask_pre2 = t_plot2 <= t_SOI_out_opt2
        mask_post2 = t_plot2 > t_SOI_out_opt2

        t_plot2_pre = t_plot2[mask_pre2]
        t_plot2_post = t_plot2[mask_post2]

        # ---------- Nominal trajectory ----------
        sol_pre2 = solve_ivp(
            F,
            (t0, t_SOI_out_opt2),
            Y0_2,
            t_eval=t_plot2_pre,
            method="DOP853",
            atol=atol,
            rtol=rtol,
        )

        sol_post2 = solve_ivp(
            F,
            (t_SOI_out_opt2, t_hit2),
            best_caseII["Y_SOI_out"],
            t_eval=t_plot2_post,
            method="DOP853",
            atol=atol,
            rtol=rtol,
        )

        Y_plot2 = np.zeros((4, len(t_plot2)))
        Y_plot2[:, mask_pre2] = sol_pre2.y
        Y_plot2[:, mask_post2] = sol_post2.y

        # ---------- Reference trajectory ----------
        sol_pre2_ref = solve_ivp(
            F,
            (t0, t_SOI_out_opt2),
            Y0_2,
            t_eval=t_plot2_pre,
            method="DOP853",
            atol=atol_ref,
            rtol=rtol_ref,
        )

        sol_post2_ref = solve_ivp(
            F,
            (t_SOI_out_opt2, t_hit2),
            best_caseII["Y_SOI_out"],
            t_eval=t_plot2_post,
            method="DOP853",
            atol=atol_ref,
            rtol=rtol_ref,
        )

        Y_plot2_ref = np.zeros((4, len(t_plot2)))
        Y_plot2_ref[:, mask_pre2] = sol_pre2_ref.y
        Y_plot2_ref[:, mask_post2] = sol_post2_ref.y

        plotter.plot_solution(t_plot2, Y_plot2, Y_plot2_ref)
        plotter.plot2D(t_plot2, dt_plot2, Y_plot2, R)
        plotter.plot_heliocentric_trajectory(t_plot2, Y_plot2, R, title="Case II optimum - heliocentric trajectory", t_SOI_in=t_SOI_in_opt2, t_SOI_out=t_SOI_out_opt2, t_MED=t_MED_opt2)
        plotter.plot_geocentric_trajectory(t_plot2, Y_plot2, R, title="Case II optimum - Earth-centered trajectory", t_SOI_in=t_SOI_in_opt2, t_SOI_out=t_SOI_out_opt2, t_MED=t_MED_opt2)
        plotter.plot2D_caseII(t_plot2, Y_plot2, R, t_SOI_in_opt2, t_SOI_out_opt2, t_MED_opt2)
        plotter.plot_distances(t_plot2, Y_plot2, R, title="Case II optimum", t_SOI_in=t_SOI_in_opt2, t_SOI_out=t_SOI_out_opt2, t_MED=t_MED_opt2)
        plotter.plot_orbital_elements(t_plot2, Y_plot2, R, center="sun", title="Case II optimum - heliocentric elements", t_SOI_in=t_SOI_in_opt2, t_SOI_out=t_SOI_out_opt2, t_MED=t_MED_opt2)
        plotter.plot_orbital_elements(t_plot2, Y_plot2, R, center="earth", title="Case II optimum - Earth-relative flyby elements", t_SOI_in=t_SOI_in_opt2, t_SOI_out=t_SOI_out_opt2, t_MED=t_MED_opt2, time_window=(t_SOI_in_opt2 - 0.15 * cts.year2seconds, t_SOI_out_opt2 + 0.15 * cts.year2seconds))

        # Case I vs Case II
        dv_saving = dv_tot_opt1 - dv_tot_opt2
        relative_saving = 100.0 * dv_saving / dv_tot_opt1
        extra_time = t_fin_opt2 - t_fin_opt1

        print("\n--- FINAL COMPARISON: CASE I vs CASE II ---")
        print("dv_tot_I (km/s) =", dv_tot_opt1)
        print("dv_tot_II (km/s) =", dv_tot_opt2)
        print("dv saving I-II (m/s) =", 1000.0 * dv_saving)
        print("relative saving (%) =", relative_saving)
        print("t_fin_I (years) =", t_fin_opt1 / cts.year2seconds)
        print("t_fin_II (years) =", t_fin_opt2 / cts.year2seconds)
        print("extra time Case II - Case I (years) =", extra_time / cts.year2seconds)
        print("dv_ign_II < dv_ign_I optimized ? =", dv_ign_opt2 < dv_ign_opt1)
        print("dv_tot_II < dv_tot_I optimized ? =", dv_tot_opt2 < dv_tot_opt1)
