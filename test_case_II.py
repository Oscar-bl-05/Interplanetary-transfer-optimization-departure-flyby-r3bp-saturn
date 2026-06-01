"""Case-II diagnostics and resonance-based optimizer entry point.

This follows the professor's PCA recommendation:

    n*T(a) = n_earth*T_Earth,     n_earth > n >= 1

The default diagnostic tests n=1, n_earth=2, because that is the first case
explicitly suggested.  Use --sweep to try several n_earth values numerically.
"""

import argparse
import time

import numpy as np
from scipy.integrate import solve_ivp

from include import IC, analitical, cts, dynamics
from include.optimizer import optimize_case_II, optimize_case_II_resonance_sweep

ATOL = np.array([1e-2, 1e-2, 1e-6, 1e-6])
RTOL = 1e-9

NSTEP_OPT = 1500


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
    dt_event = min((tf - t0) / NSTEP_OPT, 20.0 * cts.DAY_TO_S)

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


def print_case_II_diagnostics(est, sol, runtime):
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

    print("\n--- INTEGRATION CHECK ---")
    print("solver success =", sol.success)
    print("solver message =", sol.message)
    print("runtime (s) =", runtime)
    print("reached_R_B =", reached_RB)
    print("impacted_Earth =", impacted_Earth)

    if reached_RB:
        t_hit = float(sol.t_events[0][0])
        y_hit = sol.y_events[0][0]
        r_hit = float(np.hypot(y_hit[0], y_hit[1]))
        print("\n--- R_B HIT ---")
        print("t_hit (years) =", t_hit / cts.YEAR_TO_S)
        print("r_hit (km) =", r_hit)
        print("r_hit - R_B (km) =", r_hit - float(cts.R_orb_B))
    else:
        print("\n--- R_B MISS ---")
        print("r_max (Gm) =", rmax / 1e6)
        print("R_B (Gm) =", float(cts.R_orb_B) / 1e6)
        print("missing (Gm) =", (float(cts.R_orb_B) - rmax) / 1e6)

    if impacted_Earth:
        print("\n--- EARTH IMPACT ---")
        print("impact time (years) =", float(sol.t_events[1][0]) / cts.YEAR_TO_S)

    print("\n--- SOI RETURN CHECK ---")
    print("SOI entries after departure (years) =", [t / cts.YEAR_TO_S for t in soi_entries])
    print("SOI exits after departure (years) =", [t / cts.YEAR_TO_S for t in soi_exits])

    print("\n--- EARTH DISTANCE CHECK ---")
    print("minimum sampled Earth distance after departure (km) =", med_sampled)
    print("time of sampled closest Earth return (years) =", t_med_sampled / cts.YEAR_TO_S)
    print("Earth radius (km) =", cts.R_Earth)
    print("Earth SOI radius (km) =", cts.earth_SOI_radius)
    print("inside Earth SOI =", med_sampled < cts.earth_SOI_radius)
    print("above Earth surface =", med_sampled > cts.R_Earth)


def run_selected_resonance_optimization(args):
    est = analitical.resonance_case_II_estimate(n=args.n, n_earth=args.n_earth)
    print("\n--- CASE II OPTIMIZATION FOR SELECTED RESONANCE ---")
    print("n =", args.n, "n_earth =", args.n_earth)
    print("PCA dv_ign guess =", est["dv_ign"])
    print("PCA theta guess =", est["theta"])
    print("tf years =", est["t_sim"] / cts.YEAR_TO_S)

    best = optimize_case_II(
        F=dynamics.F,
        nstep=args.nstep,
        atol=ATOL,
        rtol=RTOL,
        tf=float(est["t_sim"]),
        n_grid_deltav=args.n_grid_dv,
        n_grid_theta=args.n_grid_theta,
        n_refines=args.n_refines,
        resonance_n=args.n,
        resonance_n_earth=args.n_earth,
        earth_clearance_km=args.clearance,
        store_full_solution=True,
        verbose=args.verbose,
    )

    print_case_II_best(best)


def run_resonance_sweep(args):
    print("\n--- CASE II RESONANCE SWEEP ---")
    t0 = time.time()
    best, results = optimize_case_II_resonance_sweep(
        F=dynamics.F,
        nstep=args.nstep,
        atol=ATOL,
        rtol=RTOL,
        n=args.n,
        n_earth_values=range(args.n_earth_min, args.n_earth_max + 1),
        n_grid_deltav=args.n_grid_dv,
        n_grid_theta=args.n_grid_theta,
        n_refines=args.n_refines,
        earth_clearance_km=args.clearance,
        store_full_solution=False,
        verbose=args.verbose,
    )
    print("sweep runtime (s) =", time.time() - t0)

    print("\n--- SWEEP SUMMARY ---")
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
                f"t_fin={cand['t_fin']/cts.YEAR_TO_S:.3f} yr"
            )

    print_case_II_best(best)


def print_case_II_best(best):
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
    print("dv_ign < Case-I PCA dv_ign =", best["dv_ign"] < float(analitical.deltaV_ignI))


def parse_args():
    parser = argparse.ArgumentParser(description="Case-II resonance diagnostics and optimizer")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--n-earth", type=int, default=2)
    parser.add_argument("--n-earth-min", type=int, default=2)
    parser.add_argument("--n-earth-max", type=int, default=12)
    parser.add_argument("--n-grid-theta", type=int, default=18)
    parser.add_argument("--n-grid-dv", type=int, default=12)
    parser.add_argument("--n-refines", type=int, default=1)
    parser.add_argument("--nstep", type=int, default=NSTEP_OPT)
    parser.add_argument("--clearance", type=float, default=0.0, help="Minimum altitude margin over Earth radius [km]")
    parser.add_argument("--single-opt", action="store_true", help="Optimize only the selected resonance")
    parser.add_argument("--sweep", action="store_true", help="Optimize all resonances in the selected range")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Initializing Case II diagnostic...")
    print_resonance_table(n=args.n, n_earth_min=args.n_earth_min, n_earth_max=max(args.n_earth_max, args.n_earth))

    t_start = time.time()
    est, sol = propagate_case_II_guess(args.n, args.n_earth)
    runtime = time.time() - t_start
    print_case_II_diagnostics(est, sol, runtime)

    if args.single_opt:
        run_selected_resonance_optimization(args)

    if args.sweep:
        run_resonance_sweep(args)


if __name__ == "__main__":
    main()
