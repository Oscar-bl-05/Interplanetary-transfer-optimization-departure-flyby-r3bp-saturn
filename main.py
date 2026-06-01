"""Main entry point for the Earth-Saturn transfer exercise.

The script can run Case I, Case II, or both.  Case II follows the
professor's resonant-PCA strategy and then checks the full numerical
constraints: second Earth SOI pass, no Earth impact, arrival at R_B and
minimum total delta-v.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from include import IC, cts, analitical, dynamics, plotter
from include.optimizer import (
    optimize_case_I,
    optimize_case_II_guided,
    optimize_case_II_resonance_sweep,
)

# Tolerances selected from err_check.py
ATOL = np.array([1e-2, 1e-2, 1e-6, 1e-6])
RTOL = 1e-9

# Reference tolerances used only for plotting error curves.
ATOL_REF = np.array([1e-6, 1e-6, 1e-10, 1e-10])
RTOL_REF = 1e-12

NSTEP_CASE_I_OPT = 750
NSTEP_CASE_II_OPT = 1500
NSTEP_PLOT = 4000


def parse_args():
    parser = argparse.ArgumentParser(description="Earth-Saturn transfer optimizer: Case I and Case II.")

    parser.add_argument(
        "--case",
        choices=("I", "II", "both"),
        default="both",
        help="Which case to run. Default: both."
    )

    # Plot controls
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Open interactive Matplotlib windows. Usually not needed; saved PNGs are opened by default with the OS viewer.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Do not generate plots after the optimization.",
    )
    parser.add_argument(
        "--open-plots",
        dest="open_plots",
        action="store_true",
        default=True,
        help="Open saved PNG figures with the operating-system image viewer. Enabled by default.",
    )
    parser.add_argument(
        "--no-open-plots",
        dest="open_plots",
        action="store_false",
        help="Save plots but do not open them automatically.",
    )
    parser.add_argument(
        "--plots-dir",
        default="figures",
        help="Directory where PNG figures are saved when plots are generated.",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Show the interactive trajectory animation. Only useful together with --show-plots.",
    )
    parser.add_argument(
        "--skip-animations",
        action="store_true",
        help="Do not save heliocentric GIF animations. By default, final animations are saved when plots are generated.",
    )
    parser.add_argument(
        "--animation-frames",
        type=int,
        default=180,
        help="Number of frames in saved heliocentric GIF animations. Default: 180.",
    )
    parser.add_argument(
        "--plot-case-ii",
        action="store_true",
        help="Deprecated compatibility flag. Case-II diagnostic plots are saved by default unless --no-case-ii-plots or --skip-plots is used.",
    )
    parser.add_argument(
        "--no-case-ii-plots",
        action="store_true",
        help="Do not save Case-II diagnostic/optimized plots.",
    )

    # Case II controls
    parser.add_argument(
        "--case-ii-mode",
        choices=("diagnostic", "single", "sweep"),
        default="sweep",
        help=(
            "Case II mode. 'diagnostic' prints the resonant PCA table and propagates the selected guess. "
            "'single' optimizes one resonance. 'sweep' optimizes a range of resonances. Default: sweep."
        ),
    )
    parser.add_argument("--n", type=int, default=1, help="Spacecraft revolutions in the resonance n*T(a)=n_earth*T_Earth for diagnostic/single modes.")
    parser.add_argument("--n-max", type=int, default=1, help="Maximum spacecraft resonance integer n used in sweep mode. Use --n-max 2 or higher to test long fractional resonances such as 2:25.")
    parser.add_argument("--n-earth", type=int, default=12, help="Selected Earth resonance integer for diagnostic/single modes. Default: 12 for Earth-Saturn; use --n-earth 2 to reproduce the professor's first suggested test.")
    parser.add_argument("--n-earth-min", type=int, default=12, help="Minimum n_earth for Case II sweep. Default: 12 for the automatic Earth-Saturn solution; the diagnostic table still shows 2...12, and you can pass --n-earth-min 2 to reproduce a wider sweep.")
    parser.add_argument("--n-earth-max", type=int, default=12, help="Maximum n_earth for Case II sweep. Use 25 or 26 together with --n-max 2 to test longer fractional resonances such as 2:25.")
    parser.add_argument("--case-ii-dv-limit-mode", choices=("pca", "optimized"), default="optimized", help="Upper bound for Case-II ignition delta-v. 'optimized' uses the numerical Case-I dv_ign when available; 'pca' uses the analytical PCA value. Default: optimized.")
    parser.add_argument("--case-ii-dv-limit-value", type=float, default=None, help="Optional explicit Case-II ignition upper limit [km/s]. Overrides --case-ii-dv-limit-mode.")
    parser.add_argument("--case-ii-grid-theta", type=int, default=13, help="Number of theta grid points for Case II optimization. Default is a guided automatic sweep; increase to 35-60 for final runs.")
    parser.add_argument("--case-ii-grid-dv", type=int, default=9, help="Number of delta-v grid points for Case II optimization. Default is a guided automatic sweep; increase to 20-35 for final runs.")
    parser.add_argument("--case-ii-refines", type=int, default=0, help="Number of Case II grid refinements. Default 1 refines the narrow flyby window; use 2 for final runs.")
    parser.add_argument("--case-ii-nstep", type=int, default=600, help="Step-control parameter for Case II max_step. Increase to 1500-2500 for final validation.")
    parser.add_argument("--case-ii-clearance", type=float, default=0.0, help="Extra hard safety margin above Earth radius [km]; normally keep 0 because min/max flyby altitude are handled separately.")
    parser.add_argument("--case-ii-min-altitude", type=float, default=300.0, help="Minimum allowed altitude during the second Earth flyby [km]. Default: 300 km.")
    parser.add_argument("--case-ii-max-altitude", type=float, default=350000.0, help="Maximum allowed altitude during the second Earth flyby [km]. Default: 350000 km to reject weak SOI-grazing flybys.")
    parser.add_argument("--case-ii-objective", choices=("fuel", "energy", "balanced"), default="balanced", help="Objective inside the close-flyby window: fuel=min dv_tot, energy=max dE/aphelion, balanced=max dE among candidates within 50 m/s of the fuel-best close flyby. Default: balanced.")
    parser.add_argument("--verbose-case-ii", action="store_true", help="Print rejection reasons inside the Case II search.")

    return parser.parse_args()


def propagate_case_I(theta, dv_ign, t0, tf, atol, rtol, nstep):
    """Propagate a Case I candidate until R_B."""
    Y0 = dynamics.apply_ignition_delta_v(theta, dv_ign)
    reach_RB = dynamics.make_reach_radius_event(cts.R_orb_B, direction=1, terminal=True)
    dt_event = (tf - t0) / nstep

    return solve_ivp(
        dynamics.F,
        (t0, tf),
        Y0,
        method="DOP853",
        atol=atol,
        rtol=rtol,
        events=reach_RB,
        max_step=dt_event,
    )


def validate_reach_RB(label, theta, dv_ign, tf, nstep=1200):
    """Compare selected tolerances against a tighter reference at the R_B event."""
    print(f"\n--- NUMERICAL VALIDATION ({label}) ---")
    Y0 = dynamics.apply_ignition_delta_v(theta, dv_ign)
    reach_RB_fast = dynamics.make_reach_radius_event(cts.R_orb_B, direction=1, terminal=True)
    reach_RB_ref = dynamics.make_reach_radius_event(cts.R_orb_B, direction=1, terminal=True)
    max_step = min(float(tf) / max(1, int(nstep)), 20.0 * cts.DAY_TO_S)

    sol_fast = solve_ivp(
        dynamics.F,
        (0.0, float(tf)),
        Y0,
        method="DOP853",
        atol=ATOL,
        rtol=RTOL,
        events=reach_RB_fast,
        max_step=max_step,
    )
    sol_ref = solve_ivp(
        dynamics.F,
        (0.0, float(tf)),
        Y0,
        method="DOP853",
        atol=ATOL_REF,
        rtol=RTOL_REF,
        events=reach_RB_ref,
        max_step=max_step,
    )

    if len(sol_fast.t_events[0]) == 0 or len(sol_ref.t_events[0]) == 0:
        print("Validation failed: R_B event was not detected in fast and/or reference propagation.")
        print("fast success/message:", sol_fast.success, sol_fast.message)
        print("ref  success/message:", sol_ref.success, sol_ref.message)
        return None

    y_fast = sol_fast.y_events[0][0]
    y_ref = sol_ref.y_events[0][0]
    t_fast = float(sol_fast.t_events[0][0])
    t_ref = float(sol_ref.t_events[0][0])

    pos_err_km = float(np.linalg.norm(y_fast[0:2] - y_ref[0:2]))
    vel_err_ms = float(1000.0 * np.linalg.norm(y_fast[2:4] - y_ref[2:4]))
    t_err_days = float(abs(t_fast - t_ref) / cts.DAY_TO_S)

    print("t_fast years =", t_fast / cts.YEAR_TO_S)
    print("t_ref years  =", t_ref / cts.YEAR_TO_S)
    print("time error [days] =", t_err_days)
    print("position error at R_B [km] =", pos_err_km)
    print("velocity error at R_B [m/s] =", vel_err_ms)
    print("position criterion < 1000 km =", pos_err_km < 1000.0)
    print("velocity criterion < 10 m/s =", vel_err_ms < 10.0)

    return {
        "t_fast": t_fast,
        "t_ref": t_ref,
        "position_error_km": pos_err_km,
        "velocity_error_ms": vel_err_ms,
        "time_error_days": t_err_days,
    }


def validate_case_II_postflyby(best, nstep=1200):
    """Validate the post-flyby Case-II leg against a tighter reference solution."""
    print("\n--- NUMERICAL VALIDATION (Case II post-flyby leg) ---")
    if best is None or "y_soi_out" not in best or "t_soi_out" not in best:
        print("Validation skipped: best Case-II candidate does not contain SOI-exit state.")
        return None

    t0 = float(best["t_soi_out"])
    tf = float(best["t_fin"] + 0.5 * cts.YEAR_TO_S)
    y0 = np.asarray(best["y_soi_out"], dtype=float)
    max_step = min((tf - t0) / max(1, int(nstep)), 5.0 * cts.DAY_TO_S)

    reach_fast = dynamics.make_reach_radius_event(cts.R_orb_B, direction=1, terminal=True)
    impact_fast = dynamics.make_earth_impact_event(direction=-1, terminal=True)
    reach_ref = dynamics.make_reach_radius_event(cts.R_orb_B, direction=1, terminal=True)
    impact_ref = dynamics.make_earth_impact_event(direction=-1, terminal=True)

    sol_fast = solve_ivp(
        dynamics.F,
        (t0, tf),
        y0,
        method="DOP853",
        atol=ATOL,
        rtol=RTOL,
        events=(reach_fast, impact_fast),
        max_step=max_step,
    )
    sol_ref = solve_ivp(
        dynamics.F,
        (t0, tf),
        y0,
        method="DOP853",
        atol=ATOL_REF,
        rtol=RTOL_REF,
        events=(reach_ref, impact_ref),
        max_step=max_step,
    )

    if len(sol_fast.t_events[1]) > 0 or len(sol_ref.t_events[1]) > 0:
        print("Validation failed: Earth impact detected during post-flyby leg.")
        return None
    if len(sol_fast.t_events[0]) == 0 or len(sol_ref.t_events[0]) == 0:
        print("Validation failed: R_B event missing in fast and/or reference post-flyby propagation.")
        print("fast success/message:", sol_fast.success, sol_fast.message)
        print("ref  success/message:", sol_ref.success, sol_ref.message)
        return None

    y_fast = sol_fast.y_events[0][0]
    y_ref = sol_ref.y_events[0][0]
    t_fast = float(sol_fast.t_events[0][0])
    t_ref = float(sol_ref.t_events[0][0])
    pos_err_km = float(np.linalg.norm(y_fast[0:2] - y_ref[0:2]))
    vel_err_ms = float(1000.0 * np.linalg.norm(y_fast[2:4] - y_ref[2:4]))
    t_err_days = float(abs(t_fast - t_ref) / cts.DAY_TO_S)

    print("t_fast years =", t_fast / cts.YEAR_TO_S)
    print("t_ref years  =", t_ref / cts.YEAR_TO_S)
    print("time error [days] =", t_err_days)
    print("position error at R_B [km] =", pos_err_km)
    print("velocity error at R_B [m/s] =", vel_err_ms)
    print("position criterion < 1000 km =", pos_err_km < 1000.0)
    print("velocity criterion < 10 m/s =", vel_err_ms < 10.0)

    return {
        "t_fast": t_fast,
        "t_ref": t_ref,
        "position_error_km": pos_err_km,
        "velocity_error_ms": vel_err_ms,
        "time_error_days": t_err_days,
    }


def merged_case_II_solution(best):
    """Concatenate the launch-to-SOI screening arc and the post-flyby R_B arc."""
    if best is None:
        return None, None
    screen_sol = best.get("screen_sol")
    post_sol = best.get("sol")
    if screen_sol is None or post_sol is None:
        return None, None

    t_cut = float(best.get("t_soi_out", screen_sol.t[-1]))
    mask1 = screen_sol.t <= t_cut
    t1 = screen_sol.t[mask1]
    y1 = screen_sol.y[:, mask1]
    t2 = post_sol.t
    y2 = post_sol.y

    if len(t1) == 0:
        return t2, y2
    if len(t2) == 0:
        return t1, y1
    if t2[0] <= t1[-1] + 1e-6:
        t = np.concatenate([t1, t2[1:]])
        y = np.hstack([y1, y2[:, 1:]])
    else:
        t = np.concatenate([t1, t2])
        y = np.hstack([y1, y2])
    return t, y


def run_case_I():
    """Optimize and validate Case I. Returns the best candidate and plot data."""
    print("Initializing Case I simulation...")

    t0 = 0.0
    tf = float(analitical.T_transfer_case_I)

    print("\nStarting Case I optimization...")
    t_opt_start = time.time()

    best = optimize_case_I(
        F=dynamics.F,
        nstep=NSTEP_CASE_I_OPT,
        atol=ATOL,
        rtol=RTOL,
        tf=tf,
        t0=t0,
        n_grid=10,
        n_refines=2,
    )

    t_opt_end = time.time()

    if best is None:
        raise RuntimeError("No valid Case I candidate reached R_B in the explored grids.")

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
    print("t_fin_opt (years) =", t_fin_opt / cts.YEAR_TO_S)
    print("optimizer runtime (s) =", t_opt_end - t_opt_start)

    pca_dv_tot = float(analitical.deltaV_ignI + analitical.deltaV_finI)
    pca_rel_err = 100.0 * (pca_dv_tot - dv_tot_opt) / dv_tot_opt

    print("\n--- PCA COMPARISON (Case I) ---")
    print("PCA dv_ignI (km/s) =", float(analitical.deltaV_ignI))
    print("PCA dv_finI (km/s) =", float(analitical.deltaV_finI))
    print("PCA dv_totI (km/s) =", pca_dv_tot)
    print("relative error vs optimized total (%) =", pca_rel_err)

    sol = propagate_case_I(
        theta=theta_opt,
        dv_ign=dv_ign_opt,
        t0=t0,
        tf=tf,
        atol=ATOL,
        rtol=RTOL,
        nstep=NSTEP_CASE_I_OPT,
    )

    if len(sol.t_events[0]) == 0:
        rmax = np.hypot(sol.y[0], sol.y[1]).max()
        print("\nDid NOT reach R_B with optimal parameters.")
        print("r_max =", rmax, "km")
        print("missing =", cts.R_orb_B - rmax, "km")
        raise RuntimeError("Case I optimum failed final validation.")

    t_hit = float(sol.t_events[0][0])
    y_hit = sol.y_events[0][0]
    r_hit = float(np.hypot(y_hit[0], y_hit[1]))

    print("\n--- HIT (Case I optimum) ---")
    print("t_hit (years) =", t_hit / cts.YEAR_TO_S)
    print("r_hit (km) =", r_hit)
    print("r_hit - R_B (km) =", r_hit - float(cts.R_orb_B))

    validation = validate_reach_RB(
        "Case I optimum",
        theta_opt,
        dv_ign_opt,
        tf=max(tf, t_hit + 0.1 * cts.YEAR_TO_S),
        nstep=NSTEP_CASE_I_OPT,
    )

    Y0 = dynamics.apply_ignition_delta_v(theta_opt, dv_ign_opt)
    t_plot = np.linspace(t0, t_hit, NSTEP_PLOT + 1, endpoint=True)

    sol_plot = solve_ivp(
        dynamics.F,
        (t0, t_hit),
        Y0,
        t_eval=t_plot,
        method="DOP853",
        atol=ATOL,
        rtol=RTOL,
    )

    sol_plot_ref = solve_ivp(
        dynamics.F,
        (t0, t_hit),
        Y0,
        t_eval=t_plot,
        method="DOP853",
        atol=ATOL_REF,
        rtol=RTOL_REF,
    )

    return {
        "best": best,
        "sol_plot": sol_plot,
        "sol_plot_ref": sol_plot_ref,
        "t_hit": t_hit,
        "validation": validation,
    }



def print_initial_geometry_check():
    """Check that the initial Earth/spacecraft phase is not artificially blocking Case II."""
    theta_demo = -0.35
    Y0_demo = dynamics.apply_ignition_delta_v(theta_demo, float(analitical.deltaV_ignI))
    earth_v = IC.earth_velocity_at_t0()
    rel_v_after = Y0_demo[2:4] - earth_v

    print("\n--- INITIAL GEOMETRY CHECK ---")
    print("Earth starts at (R_A, 0) and moves in +y; this is only a phase convention.")
    print("For an exterior transfer the useful theta interval is -pi/2 < theta < 0.")
    print("Example theta =", theta_demo)
    print("Earth heliocentric velocity [km/s] =", earth_v)
    print("spacecraft relative velocity after burn [km/s] =", rel_v_after)
    print("heliocentric vy is larger than Earth's vy =", Y0_demo[3] > earth_v[1])
    print("Waiting before departure rotates the circular Sun-Earth-spacecraft geometry. Since the numerical target is only r = R_B, not a fixed Saturn phase, waiting is not an independent degree of freedom; the departure point around Earth is already theta.")

def print_resonance_table(n=1, n_earth_min=2, n_earth_max=13):
    """Print PCA estimates for several Earth-return resonances."""
    print("\n--- CASE II PCA RESONANCE TABLE ---")
    print("condition: n*T(a) = n_earth*T_Earth")
    print("target R_B (Gm) =", float(cts.R_orb_B) / 1e6)
    print("Case-I PCA dv_ign (km/s) =", float(analitical.deltaV_ignI))
    print()
    print(
        " n  nE | T(a)[yr] | dv_ign[km/s] | theta[rad] | "
        "r_apo first[Gm] | max apo after ideal flyby[Gm] | usable"
    )
    print("-" * 106)

    for est in analitical.resonance_table(n=n, n_earth_min=n_earth_min, n_earth_max=n_earth_max):
        max_ra = analitical.pca_max_aphelion_after_earth_flyby(est["v_inf"])
        usable = (est["dv_ign"] < float(analitical.deltaV_ignI)) and (max_ra >= float(cts.R_orb_B))
        max_ra_txt = "inf" if not np.isfinite(max_ra) else f"{max_ra/1e6:10.1f}"
        print(
            f"{est['n']:2d} {est['n_earth']:3d} | "
            f"{est['T_resonance']/cts.YEAR_TO_S:8.3f} | "
            f"{est['dv_ign']:12.6f} | "
            f"{est['theta']:10.6f} | "
            f"{est['r_apo']/1e6:16.1f} | "
            f"{max_ra_txt:>28} | "
            f"{usable}"
        )

    print(
        "\nNote: 'max apo after ideal flyby' is only a PCA upper bound. "
        "If it is below R_B, that resonance is energetically very unlikely "
        "to reach Saturn with only the two required impulses."
    )


def propagate_case_II_guess(n, n_earth):
    """Propagate the PCA resonant guess for one Case-II resonance."""
    est = analitical.resonance_case_II_estimate(n=n, n_earth=n_earth)
    t0 = 0.0
    tf = float(est["t_sim"])
    dt_event = min((tf - t0) / NSTEP_CASE_II_OPT, 20.0 * cts.DAY_TO_S)

    Y0 = dynamics.apply_ignition_delta_v(est["theta"], est["dv_ign"])

    reach_RB = dynamics.make_reach_radius_event(cts.R_orb_B, direction=1, terminal=True)
    impact_Earth = dynamics.make_earth_impact_event(terminal=True)
    enter_SOI = dynamics.make_earth_soi_event(direction=-1, terminal=False)
    exit_SOI = dynamics.make_earth_soi_event(direction=1, terminal=False)

    sol = solve_ivp(
        dynamics.F,
        (t0, tf),
        Y0,
        method="DOP853",
        atol=ATOL,
        rtol=RTOL,
        events=(reach_RB, impact_Earth, enter_SOI, exit_SOI),
        max_step=dt_event,
        dense_output=True,
    )
    return est, sol


def print_case_II_guess_diagnostics(est, sol, runtime):
    """Print diagnostics for the PCA Case-II guess."""
    reached_RB = len(sol.t_events[0]) != 0
    impacted_Earth = len(sol.t_events[1]) != 0

    med_sampled, t_med_sampled = dynamics.minimum_earth_distance_after_departure(sol)
    rmax = float(np.hypot(sol.y[0], sol.y[1]).max())

    soi_entries = [float(t) for t in sol.t_events[2] if float(t) > 0.5 * cts.YEAR_TO_S]
    soi_exits = [float(t) for t in sol.t_events[3] if float(t) > 0.5 * cts.YEAR_TO_S]

    max_ra = analitical.pca_max_aphelion_after_earth_flyby(est["v_inf"])

    print("\n--- CASE II ANALYTICAL RESONANT GUESS ---")
    print("n =", est["n"], "| n_earth =", est["n_earth"])
    print("T(a) (years) =", est["T_resonance"] / cts.YEAR_TO_S)
    print("theta_guess (rad) =", est["theta"])
    print("dv_ign_guess (km/s) =", est["dv_ign"])
    print("dv_escape_min (km/s) =", float(IC.escape_delta_v_min()))
    print("deltaV_ignI PCA (km/s) =", float(analitical.deltaV_ignI))
    print("dv_escape_min < dv_ign < deltaV_ignI =", IC.escape_delta_v_min() < est["dv_ign"] < float(analitical.deltaV_ignI))
    print("v_inf guess (km/s) =", est["v_inf"])
    print("first ellipse a (Gm) =", est["a"] / 1e6)
    print("first ellipse aphelion (Gm) =", est["r_apo"] / 1e6)
    print("PCA max aphelion after ideal flyby (Gm) =", max_ra / 1e6 if np.isfinite(max_ra) else np.inf)
    print("t_simulation (years) =", est["t_sim"] / cts.YEAR_TO_S)

    print("\n--- CASE II INTEGRATION CHECK ---")
    print("solver success =", sol.success)
    print("solver message =", sol.message)
    print("runtime (s) =", runtime)
    print("reached_R_B =", reached_RB)
    print("impacted_Earth =", impacted_Earth)

    if reached_RB:
        t_hit = float(sol.t_events[0][0])
        y_hit = sol.y_events[0][0]
        r_hit = float(np.hypot(y_hit[0], y_hit[1]))
        print("\n--- CASE II R_B HIT ---")
        print("t_hit (years) =", t_hit / cts.YEAR_TO_S)
        print("r_hit (km) =", r_hit)
        print("r_hit - R_B (km) =", r_hit - float(cts.R_orb_B))
    else:
        print("\n--- CASE II R_B MISS ---")
        print("r_max (Gm) =", rmax / 1e6)
        print("R_B (Gm) =", float(cts.R_orb_B) / 1e6)
        print("missing (Gm) =", (float(cts.R_orb_B) - rmax) / 1e6)

    if impacted_Earth:
        print("\n--- CASE II EARTH IMPACT ---")
        print("impact time (years) =", float(sol.t_events[1][0]) / cts.YEAR_TO_S)

    print("\n--- CASE II SOI RETURN CHECK ---")
    print("SOI entries after departure (years) =", [t / cts.YEAR_TO_S for t in soi_entries])
    print("SOI exits after departure (years) =", [t / cts.YEAR_TO_S for t in soi_exits])

    print("\n--- CASE II EARTH DISTANCE CHECK ---")
    print("minimum sampled Earth distance after departure (km) =", med_sampled)
    print("time of sampled closest Earth return (years) =", t_med_sampled / cts.YEAR_TO_S)
    print("Earth radius (km) =", cts.R_Earth)
    print("Earth SOI radius (km) =", cts.earth_SOI_radius)
    print("inside Earth SOI =", med_sampled < cts.earth_SOI_radius)
    print("above Earth surface =", med_sampled > cts.R_Earth)


def print_case_II_best(best):
    """Print a valid optimized Case-II candidate, or explain that none was found."""
    if best is None:
        print("\nNo valid constrained Case-II solution found.")
        return

    print("\n--- BEST VALID CASE II ---")
    print("resonance n =", best.get("resonance_n"))
    print("resonance n_earth =", best.get("resonance_n_earth"))
    print("theta =", best["theta"])
    print("dv_ign =", best["dv_ign"])
    print("dv_fin =", best["dv_fin"])
    print("dv_tot =", best["dv_tot"])
    print("t_fin years =", best["t_fin"] / cts.YEAR_TO_S)
    print("t_soi_in years =", best["t_soi_in"] / cts.YEAR_TO_S)
    print("t_soi_out years =", best["t_soi_out"] / cts.YEAR_TO_S)
    print("MED km =", best["MED"])
    print("minimum altitude km =", best["minimum_altitude"])
    print("selected objective =", best.get("selected_objective"))
    print("number of valid close-flyby candidates checked =", len(best.get("valid_candidates", [])))
    print("energy before flyby [km^2/s^2] =", best.get("energy_before"))
    print("energy after flyby  [km^2/s^2] =", best.get("energy_after"))
    print("delta energy        [km^2/s^2] =", best.get("delta_energy"))
    print("speed before flyby [km/s] =", best.get("speed_before"))
    print("speed after flyby  [km/s] =", best.get("speed_after"))
    print("delta speed        [km/s] =", best.get("delta_speed"))
    print("post-flyby osculating aphelion [Gm] =", best.get("r_apo_after") / 1e6 if best.get("r_apo_after") is not None else None)
    print("dv_ign < Case-I PCA dv_ign =", best["dv_ign"] < float(analitical.deltaV_ignI))

    fuel = best.get("best_fuel_candidate")
    energy = best.get("best_energy_candidate")
    if fuel is not None and energy is not None:
        print("\n--- CASE II CLOSE-FLYBY INTERNAL TRADE-OFF ---")
        print("fuel-best close flyby:   dv_tot =", fuel["dv_tot"], "dE =", fuel.get("delta_energy"), "altitude =", fuel.get("minimum_altitude"), "r_apo_after[Gm] =", fuel.get("r_apo_after") / 1e6)
        print("energy-best close flyby: dv_tot =", energy["dv_tot"], "dE =", energy.get("delta_energy"), "altitude =", energy.get("minimum_altitude"), "r_apo_after[Gm] =", energy.get("r_apo_after") / 1e6)
def run_case_II(args, case_I_data=None):
    """Run Case II diagnostics or optimization from main.py."""
    print("\nInitializing Case II simulation...")
    print_initial_geometry_check()

    if args.case_ii_dv_limit_value is not None:
        dv_ign_upper_limit = float(args.case_ii_dv_limit_value)
        dv_limit_label = "explicit"
    elif args.case_ii_dv_limit_mode == "optimized" and case_I_data is not None:
        dv_ign_upper_limit = float(case_I_data["best"]["dv_ign"])
        dv_limit_label = "Case-I optimized dv_ign"
    else:
        dv_ign_upper_limit = float(analitical.deltaV_ignI)
        dv_limit_label = "Case-I PCA dv_ign"

    print("Case-II dv_ign upper limit mode =", dv_limit_label)
    print("Case-II dv_ign upper limit [km/s] =", dv_ign_upper_limit)
    # Always show the full professor-style resonance table from nE=2,
    # even if the numerical sweep is restricted to the energetically useful
    # high resonances by default.
    print_resonance_table(
        n=args.n,
        n_earth_min=2,
        n_earth_max=max(args.n_earth_max, args.n_earth),
    )

    t_start = time.time()
    est, sol = propagate_case_II_guess(args.n, args.n_earth)
    runtime = time.time() - t_start
    print_case_II_guess_diagnostics(est, sol, runtime)

    case_II_data = {
        "mode": args.case_ii_mode,
        "guess_est": est,
        "guess_sol": sol,
        "best": None,
        "sweep_results": None,
    }

    if args.case_ii_mode == "diagnostic":
        print(
            "\nCase II mode is diagnostic only. "
            "This propagates the PCA resonant guess; it is not the optimized Case-II trajectory. "
            "Use --case-ii-mode single or --case-ii-mode sweep to run numerical optimization."
        )
        return case_II_data

    store_full_solution = (not args.skip_plots and not args.no_case_ii_plots) or args.plot_case_ii

    if args.case_ii_mode == "single":
        print("\n--- CASE II OPTIMIZATION FOR SELECTED RESONANCE ---")
        print("close-flyby altitude window [km] =", args.case_ii_min_altitude, "to", args.case_ii_max_altitude)
        print("close-flyby objective =", args.case_ii_objective)
        print("n =", args.n, "n_earth =", args.n_earth)
        print("PCA dv_ign guess =", est["dv_ign"])
        print("PCA theta guess =", est["theta"])
        print("tf years =", est["t_sim"] / cts.YEAR_TO_S)

        best = optimize_case_II_guided(
            F=dynamics.F,
            nstep=args.case_ii_nstep,
            atol=ATOL,
            rtol=RTOL,
            tf=float(est["t_sim"]),
            n_grid_deltav=args.case_ii_grid_dv,
            n_grid_theta=args.case_ii_grid_theta,
            n_refines=args.case_ii_refines,
            resonance_n=args.n,
            resonance_n_earth=args.n_earth,
            earth_clearance_km=args.case_ii_clearance,
            min_flyby_altitude_km=args.case_ii_min_altitude,
            max_flyby_altitude_km=args.case_ii_max_altitude,
            case_ii_objective=args.case_ii_objective,
            dv_ign_upper_limit=dv_ign_upper_limit,
            store_full_solution=store_full_solution,
            verbose=args.verbose_case_ii,
        )
        print_case_II_best(best)
        if best is not None:
            case_II_data["validation"] = validate_case_II_postflyby(best, nstep=args.case_ii_nstep)
        case_II_data["best"] = best
        return case_II_data

    print("\n--- CASE II RESONANCE SWEEP ---")
    print("close-flyby altitude window [km] =", args.case_ii_min_altitude, "to", args.case_ii_max_altitude)
    print("close-flyby objective =", args.case_ii_objective)
    t0 = time.time()
    best = None
    results = []
    for n_val in range(int(args.n), int(args.n_max) + 1):
        nE_min = max(int(args.n_earth_min), n_val + 1)
        nE_max = int(args.n_earth_max)
        if nE_min > nE_max:
            continue
        cand_best, cand_results = optimize_case_II_resonance_sweep(
            F=dynamics.F,
            nstep=args.case_ii_nstep,
            atol=ATOL,
            rtol=RTOL,
            n=n_val,
            n_earth_values=range(nE_min, nE_max + 1),
            n_grid_deltav=args.case_ii_grid_dv,
            n_grid_theta=args.case_ii_grid_theta,
            n_refines=args.case_ii_refines,
            earth_clearance_km=args.case_ii_clearance,
            min_flyby_altitude_km=args.case_ii_min_altitude,
            max_flyby_altitude_km=args.case_ii_max_altitude,
            case_ii_objective=args.case_ii_objective,
            dv_ign_upper_limit=dv_ign_upper_limit,
            store_full_solution=store_full_solution,
            verbose=args.verbose_case_ii,
        )
        results.extend(cand_results)
        if cand_best is not None and (best is None or cand_best["dv_tot"] < best["dv_tot"]):
            best = cand_best
    print("sweep runtime (s) =", time.time() - t0)

    print("\n--- CASE II SWEEP SUMMARY ---")
    for row in results:
        cand = row["best"]
        reason = row["skipped_reason"]
        if cand is None:
            print(f"n={row['n']} nE={row['n_earth']}: no valid candidate", (f"({reason})" if reason else ""))
        else:
            print(
                f"n={row['n']} nE={row['n_earth']}: "
                f"dv_tot={cand['dv_tot']:.6f}, "
                f"dv_ign={cand['dv_ign']:.6f}, "
                f"dv_fin={cand['dv_fin']:.6f}, "
                f"MED={cand['MED']:.1f} km, "
                f"alt={cand.get('minimum_altitude', float('nan')):.1f} km, "
                f"dE={cand.get('delta_energy', float('nan')):.3f}, "
                f"r_apo_after={cand.get('r_apo_after', float('nan'))/1e6:.1f} Gm, "
                f"t_fin={cand['t_fin']/cts.YEAR_TO_S:.3f} yr"
            )

    print_case_II_best(best)
    if best is not None:
        case_II_data["validation"] = validate_case_II_postflyby(best, nstep=args.case_ii_nstep)
    case_II_data["best"] = best
    case_II_data["sweep_results"] = results
    return case_II_data

def plot_case_I(case_I_data, plots_dir, show_plots, animate, save_animations=True, animation_frames=180):
    """Save/show all Case I plots requested by the guide."""
    print("\nGenerating Case I plots...")
    plotter.plot_solution(
        case_I_data["sol_plot"].t,
        case_I_data["sol_plot"].y,
        case_I_data["sol_plot_ref"].y,
        save_dir=plots_dir,
        prefix="case_I",
        show=show_plots,
        block=False,
        close=not show_plots,
    )

    plotter.plot_trajectory_2d_static(
        case_I_data["sol_plot"].t,
        case_I_data["sol_plot"].y,
        save_path=plots_dir / "case_I_heliocentric_trajectory.png",
        title="Case I heliocentric trajectory",
        show=show_plots,
        block=False,
        close=not show_plots,
    )
    plotter.plot_radial_and_earth_distance(
        case_I_data["sol_plot"].t,
        case_I_data["sol_plot"].y,
        save_path=plots_dir / "case_I_distances.png",
        title="Case I distances: heliocentric radius and Earth-relative distance",
        show=show_plots,
        block=False,
        close=not show_plots,
    )
    plotter.plot_geocentric_trajectory_static(
        case_I_data["sol_plot"].t,
        case_I_data["sol_plot"].y,
        save_path=plots_dir / "case_I_geocentric_trajectory.png",
        title="Case I Earth-relative trajectory",
        show=show_plots,
        block=False,
        close=not show_plots,
    )
    plotter.plot_orbital_elements(
        case_I_data["sol_plot"].t,
        case_I_data["sol_plot"].y,
        center="sun",
        save_path=plots_dir / "case_I_orbital_elements_sun.png",
        title="Case I osculating orbital elements relative to Sun",
        show=show_plots,
        block=False,
        close=not show_plots,
    )
    plotter.plot_orbital_elements(
        case_I_data["sol_plot"].t,
        case_I_data["sol_plot"].y,
        center="earth",
        save_path=plots_dir / "case_I_orbital_elements_earth.png",
        title="Case I osculating orbital elements relative to Earth",
        show=show_plots,
        block=False,
        close=not show_plots,
    )

    if save_animations:
        print("Saving Case I heliocentric animation GIF...")
        plotter.save_heliocentric_animation(
            case_I_data["sol_plot"].t,
            case_I_data["sol_plot"].y,
            save_path=plots_dir / "case_I_heliocentric_animation.gif",
            title="Case I heliocentric trajectory animation",
            n_frames=animation_frames,
        )

    if animate:
        if not show_plots:
            print("Animation requested but --show-plots was not provided; skipping animation.")
        else:
            print("Opening Case I trajectory animation window...")
            plotter.plot2D(case_I_data["sol_plot"].t, case_I_data["sol_plot"].y, show=True, block=True)


def plot_case_II(case_II_data, plots_dir, show_plots, save_animations=True, animation_frames=180):
    """Save/show all Case-II diagnostic plots and optimized plots requested by the guide."""
    if case_II_data is None:
        return

    guess_sol = case_II_data.get("guess_sol")
    guess_est = case_II_data.get("guess_est")

    if case_II_data.get("mode") == "diagnostic" and guess_sol is not None and guess_sol.y.shape[1] >= 2:
        suffix = f"n{guess_est['n']}_nE{guess_est['n_earth']}" if guess_est is not None else "guess"
        print("\nGenerating Case II diagnostic plots...")
        plotter.plot_state_variables(
            guess_sol.t,
            guess_sol.y,
            save_path=plots_dir / f"case_II_diagnostic_{suffix}_state_variables.png",
            title=f"Case II diagnostic PCA guess ({suffix})",
            show=show_plots,
            block=False,
            close=not show_plots,
        )
        plotter.plot_trajectory_2d_static(
            guess_sol.t,
            guess_sol.y,
            save_path=plots_dir / f"case_II_diagnostic_{suffix}_heliocentric_trajectory.png",
            title=f"Case II diagnostic heliocentric trajectory ({suffix})",
            show=show_plots,
            block=False,
            close=not show_plots,
        )
        plotter.plot_radial_and_earth_distance(
            guess_sol.t,
            guess_sol.y,
            save_path=plots_dir / f"case_II_diagnostic_{suffix}_distances.png",
            title=f"Case II diagnostic distances ({suffix})",
            show=show_plots,
            block=False,
            close=not show_plots,
        )
        plotter.plot_geocentric_trajectory_static(
            guess_sol.t,
            guess_sol.y,
            save_path=plots_dir / f"case_II_diagnostic_{suffix}_geocentric_trajectory_full.png",
            title=f"Case II diagnostic Earth-relative trajectory, full scale ({suffix})",
            show=show_plots,
            block=False,
            close=not show_plots,
        )
        plotter.plot_earth_encounter_zoom(
            guess_sol.t,
            guess_sol.y,
            save_path=plots_dir / f"case_II_diagnostic_{suffix}_earth_encounter_zoom.png",
            title=f"Case II diagnostic Earth-encounter zoom ({suffix})",
            show=show_plots,
            block=False,
            close=not show_plots,
        )
        plotter.plot_case_II_energy_diagnostics(
            guess_sol.t,
            guess_sol.y,
            save_path=plots_dir / f"case_II_diagnostic_{suffix}_energy.png",
            title=f"Case II diagnostic energy/aphelion ({suffix})",
            show=show_plots,
            block=False,
            close=not show_plots,
        )

    best_case_II = case_II_data.get("best")
    if best_case_II is None:
        print("\nCase II optimized plots skipped: no valid optimized Case-II solution was found.")
        return

    t_merged, y_merged = merged_case_II_solution(best_case_II)
    if t_merged is None or y_merged is None or y_merged.shape[1] < 2:
        print("\nCase II optimized plots skipped: no usable launch-to-arrival solution could be reconstructed.")
        return

    print("\nGenerating Case II optimized plots...")
    print("  - optimized state variables")
    plotter.plot_state_variables(
        t_merged,
        y_merged,
        save_path=plots_dir / "case_II_optimized_state_variables.png",
        title="Case II optimized dynamical variables",
        show=show_plots,
        block=False,
        close=not show_plots,
    )
    print("  - optimized heliocentric trajectory")
    plotter.plot_trajectory_2d_static(
        t_merged,
        y_merged,
        save_path=plots_dir / "case_II_optimized_heliocentric_trajectory.png",
        title="Case II optimized heliocentric trajectory",
        show=show_plots,
        block=False,
        close=not show_plots,
    )
    print("  - optimized distances")
    plotter.plot_radial_and_earth_distance(
        t_merged,
        y_merged,
        save_path=plots_dir / "case_II_optimized_distances.png",
        title="Case II optimized distances",
        show=show_plots,
        block=False,
        close=not show_plots,
    )
    print("  - optimized geocentric trajectory")
    plotter.plot_geocentric_trajectory_static(
        t_merged,
        y_merged,
        save_path=plots_dir / "case_II_optimized_geocentric_trajectory_full.png",
        title="Case II optimized Earth-relative trajectory, full scale",
        show=show_plots,
        block=False,
        close=not show_plots,
    )
    print("  - optimized Earth encounter zoom")
    plotter.plot_earth_encounter_zoom(
        t_merged,
        y_merged,
        save_path=plots_dir / "case_II_optimized_earth_encounter_zoom.png",
        title="Case II optimized Earth flyby zoom",
        show=show_plots,
        block=False,
        close=not show_plots,
        center_time=best_case_II.get("t_MED"),
    )
    print("  - optimized energy diagnostics")
    plotter.plot_case_II_energy_diagnostics(
        t_merged,
        y_merged,
        save_path=plots_dir / "case_II_optimized_energy.png",
        title="Case II optimized energy/aphelion diagnostics",
        show=show_plots,
        block=False,
        close=not show_plots,
    )
    print("  - optimized orbital elements, Sun")
    plotter.plot_orbital_elements(
        t_merged,
        y_merged,
        center="sun",
        save_path=plots_dir / "case_II_optimized_orbital_elements_sun.png",
        title="Case II optimized osculating orbital elements relative to Sun",
        show=show_plots,
        block=False,
        close=not show_plots,
    )
    print("  - optimized orbital elements, Earth")
    plotter.plot_orbital_elements(
        t_merged,
        y_merged,
        center="earth",
        save_path=plots_dir / "case_II_optimized_orbital_elements_earth.png",
        title="Case II optimized osculating orbital elements relative to Earth",
        show=show_plots,
        block=False,
        close=not show_plots,
    )
    if save_animations:
        print("Saving Case II heliocentric animation GIF...")
        plotter.save_heliocentric_animation(
            t_merged,
            y_merged,
            save_path=plots_dir / "case_II_optimized_heliocentric_animation.gif",
            title="Case II optimized heliocentric trajectory animation",
            n_frames=animation_frames,
        )

def print_case_comparison(case_I_data, case_II_data):
    """Compare Case I and optimized Case II if both are available."""
    if case_I_data is None or case_II_data is None:
        return

    best_case_II = case_II_data.get("best") if isinstance(case_II_data, dict) else case_II_data
    if best_case_II is None:
        return

    dv_I = float(case_I_data["best"]["dv_tot"])
    dv_II = float(best_case_II["dv_tot"])
    diff = dv_II - dv_I
    rel = 100.0 * diff / dv_I

    print("\n--- CASE I vs CASE II COMPARISON ---")
    print("Case I dv_tot (km/s) =", dv_I)
    print("Case II dv_tot (km/s) =", dv_II)
    print("Case II - Case I (km/s) =", diff)
    print("relative difference (%) =", rel)



def plot_comparison(case_I_data, case_II_data, plots_dir, show_plots):
    """Save a compact Case-I/Case-II delta-v comparison figure."""
    if case_I_data is None or case_II_data is None:
        return
    best_case_II = case_II_data.get("best") if isinstance(case_II_data, dict) else None
    if best_case_II is None:
        return
    print("\nGenerating Case I vs Case II comparison plot...")
    plotter.plot_comparison_summary(
        case_I_data["best"],
        best_case_II,
        save_path=plots_dir / "case_I_vs_case_II_delta_v_comparison.png",
        show=show_plots,
        block=False,
        close=not show_plots,
    )


def open_recent_plot_files(plots_dir, since_ts, max_files=16):
    """Open recently generated PNG/GIF figures with the OS viewer without blocking Python.

    This avoids Matplotlib/VS Code temporary plot files and opens the real .png
    files saved in the figures directory.
    """
    plots_dir = Path(plots_dir)
    files = []
    for path in sorted(list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.gif"))):
        try:
            if path.stat().st_mtime >= since_ts - 1.0:
                files.append(path.resolve())
        except OSError:
            continue

    if not files:
        print("\nNo newly generated PNG/GIF plots found to open.")
        return

    if len(files) > max_files:
        print(f"\nOpening first {max_files} of {len(files)} generated PNG/GIF figures. All files remain saved in {plots_dir.resolve()}.")
        files = files[:max_files]
    else:
        print(f"\nOpening {len(files)} generated PNG/GIF figure(s) with the system viewer...")

    for path in files:
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            print(f"Could not open {path}: {exc}")

def main():
    args = parse_args()

    case_I_data = None
    case_II_data = None

    try:
        if args.case in ("I", "both"):
            case_I_data = run_case_I()

        if args.case in ("II", "both"):
            case_II_data = run_case_II(args, case_I_data=case_I_data)

        print_case_comparison(case_I_data, case_II_data)

        if args.skip_plots:
            print("\nPlot generation skipped by user request.")
            return

        plots_dir = Path(args.plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        print("\nFigures directory:", plots_dir.resolve())
        plot_start_ts = time.time()

        if case_I_data is not None:
            plot_case_I(
                case_I_data,
                plots_dir,
                args.show_plots,
                args.animate,
                save_animations=not args.skip_animations,
                animation_frames=args.animation_frames,
            )

        if case_II_data is not None and not args.no_case_ii_plots:
            plot_case_II(
                case_II_data,
                plots_dir,
                args.show_plots,
                save_animations=not args.skip_animations,
                animation_frames=args.animation_frames,
            )

        plot_comparison(case_I_data, case_II_data, plots_dir, args.show_plots)

        if args.open_plots and not args.show_plots:
            open_recent_plot_files(plots_dir, since_ts=plot_start_ts)

        # Make sure batch runs terminate cleanly even after saving many Matplotlib figures.
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass

    except KeyboardInterrupt:
        raise SystemExit("\nExecution interrupted by user.")


if __name__ == "__main__":
    main()
