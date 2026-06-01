"""Grid optimizers for the Earth-Saturn transfer problem."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from include import IC, analitical, cts, dynamics


def _refine_grid_search(grid_search, best, theta_span, dv_span, n_refines):
    """Repeatedly refine a grid search around the current best candidate."""
    if best is None:
        return None

    th_span = theta_span
    dv_sp = dv_span

    for _ in range(n_refines):
        th_span *= 0.25
        dv_sp *= 0.25
        candidate = grid_search(best["theta"], best["dv_ign"], th_span, dv_sp)
        if candidate is None:
            return best
        best = candidate

    return best


def minimum_earth_distance_dense(sol, t1, t2):
    """Minimum Earth distance using dense output between two times.

    Returns
    -------
    min_distance : float
        Minimum geocentric distance [km].
    t_min : float
        Time of closest approach [s].
    """
    if sol.sol is None:
        raise ValueError("solve_ivp must be called with dense_output=True")

    t1 = float(t1)
    t2 = float(t2)
    if not np.isfinite(t1) or not np.isfinite(t2) or t2 <= t1:
        return np.inf, np.nan

    def distance_at(t):
        Y = sol.sol(t)
        return dynamics.earth_distance(t, Y)

    res = minimize_scalar(
        distance_at,
        bounds=(t1, t2),
        method="bounded",
        options={"xatol": 1.0},
    )

    return float(res.fun), float(res.x)


def optimize_case_I(
    F,
    nstep,
    atol,
    rtol,
    tf,
    t0=0.0,
    n_grid=10,
    n_refines=2,
    theta_center=None,
    dv_center=None,
    theta_span=0.5,
    dv_span=None,
):
    """Optimize Case I by minimizing total delta-v on a refined grid."""
    if theta_center is None:
        theta_center = float(analitical.theta_0I)
    if dv_center is None:
        dv_center = float(analitical.deltaV_ignI)
    if dv_span is None:
        dv_span = 0.3 * dv_center

    reach_RB = dynamics.make_reach_radius_event(cts.R_orb_B, direction=1, terminal=True)
    dv_escape_min = IC.escape_delta_v_min(IC.rho0)

    def evaluate(theta0, dv_ign, dt):
        if dv_ign <= dv_escape_min:
            return None

        Y0 = dynamics.apply_ignition_delta_v(theta0, dv_ign)

        sol = solve_ivp(
            F,
            (t0, tf),
            Y0,
            method="DOP853",
            atol=atol,
            rtol=rtol,
            events=reach_RB,
            max_step=dt,
        )

        if len(sol.t_events[0]) == 0:
            return None

        t_fin = float(sol.t_events[0][0])
        y_fin = sol.y_events[0][0]

        v_target = dynamics.v_circular_heliocentric_at(y_fin[0], y_fin[1])
        v_sc = np.array([y_fin[2], y_fin[3]], dtype=float)
        dv_fin = float(np.linalg.norm(v_target - v_sc))
        dv_tot = float(abs(dv_ign) + abs(dv_fin))

        return {
            "theta": float(theta0),
            "dv_ign": float(dv_ign),
            "dv_fin": dv_fin,
            "dv_tot": dv_tot,
            "t_fin": t_fin,
            "y_fin": y_fin,
            "sol": sol,
        }

    def grid_search(theta_c, dv_c, th_span, dv_sp):
        dt = min((tf - t0) / nstep, 20.0 * cts.DAY_TO_S)
        theta_vals = np.linspace(theta_c - th_span, theta_c + th_span, n_grid)
        dv_vals = np.linspace(dv_c - dv_sp, dv_c + dv_sp, n_grid)

        best = None
        best_cost = np.inf

        for th in theta_vals:
            for dv in dv_vals:
                out = evaluate(th, dv, dt)
                if out is None:
                    continue
                if out["dv_tot"] < best_cost:
                    best_cost = out["dv_tot"]
                    best = out

        return best

    best = grid_search(theta_center, dv_center, theta_span, dv_span)
    return _refine_grid_search(grid_search, best, theta_span, dv_span, n_refines)


def optimize_case_II(
    F,
    nstep,
    atol,
    rtol,
    tf,
    t0=0.0,
    n_grid_deltav=10,
    n_grid_theta=10,
    n_refines=2,
    theta_center=None,
    dv_center=None,
    theta_span=None,
    dv_span=None,
    narrowband_exponent=2,
    mode="deltaV",
    min_soi_time=0.5 * cts.YEAR_TO_S,
    earth_clearance_km=0.0,
    soi_search_margin=4.0 * cts.YEAR_TO_S,
    use_two_stage=True,
    store_full_solution=True,
    verbose=False,
    resonance_n=1,
    resonance_n_earth=2,
    dv_ign_upper_limit=None,
):
    """Optimize Case II with the physical constraints of the assignment.

    Valid candidates must satisfy:
    - dv_escape_min < dv_ign < deltaV_ignI
    - no Earth impact
    - a second entry into Earth's SOI after the initial departure
    - arrival at R_B after that second SOI entry
    - MED outside Earth and inside Earth's SOI during the flyby window

    The default objective is total delta-v.  The optional ``mode='MED'`` is kept
    only as a diagnostic mode.
    """
    dv_escape_min = float(IC.escape_delta_v_min(IC.rho0))
    dv_ignI = float(analitical.deltaV_ignI)

    resonance = analitical.resonance_case_II_estimate(
        n=resonance_n,
        n_earth=resonance_n_earth,
    )

    if tf is None:
        tf = float(resonance["t_sim"])

    dv_limit = float(analitical.deltaV_ignI if dv_ign_upper_limit is None else dv_ign_upper_limit)

    if theta_center is None:
        # PCA estimate from the selected resonance.  The first pass is still
        # deliberately wide because the actual CR3BP optimum need not sit
        # exactly on the patched-conics angle.
        theta_center = float(resonance["theta"])
    if theta_span is None:
        # First pass over almost the complete exterior-transfer interval.
        theta_span = 0.5 * np.pi - 1e-3
    if dv_center is None:
        # Professor's recommendation: scan around the delta-v obtained from
        # the resonant PCA ellipse T(a) = (n_earth/n)*T_Earth.
        dv_center = float(resonance["dv_ign"])
    if dv_span is None:
        # Broad enough for a coarse search, but still centered on the selected
        # resonance.  The valid interval is clipped later by escape and Case-I
        # constraints.
        dv_span = max(0.15, 0.15 * abs(float(dv_center)))

    reach_RB = dynamics.make_reach_radius_event(cts.R_orb_B, direction=1, terminal=True)
    impact_Earth = dynamics.make_earth_impact_event(
        safety_radius=cts.R_Earth + earth_clearance_km,
        direction=-1,
        terminal=True,
    )
    enter_SOI = dynamics.make_earth_soi_event(direction=-1, terminal=False)
    exit_SOI = dynamics.make_earth_soi_event(direction=1, terminal=False)

    max_step = min((tf - t0) / nstep, 20.0 * cts.DAY_TO_S)

    # The second Earth encounter is expected after n revolutions of the
    # resonant spacecraft ellipse, i.e. after n_earth Earth years.
    resonant_return_time = float(resonance["n_earth"] * cts.T_orb_A)
    soi_search_tmax = min(float(tf), resonant_return_time + float(soi_search_margin))

    def _debug_reject(reason):
        if verbose:
            print(reason)

    def _build_output(theta0, dv_ign, sol, t_fin, t_soi_in, t_soi_out, min_dist, t_med, y_fin):
        v_target = dynamics.v_circular_heliocentric_at(y_fin[0], y_fin[1])
        v_sc = np.array([y_fin[2], y_fin[3]], dtype=float)
        dv_fin = float(np.linalg.norm(v_target - v_sc))
        dv_tot = float(abs(dv_ign) + abs(dv_fin))

        return {
            "theta": float(theta0),
            "dv_ign": float(dv_ign),
            "dv_fin": dv_fin,
            "dv_tot": dv_tot,
            "t_fin": float(t_fin),
            "t_soi_in": float(t_soi_in),
            "t_soi_out": float(t_soi_out),
            "MED": float(min_dist),
            "t_MED": float(t_med),
            "minimum_altitude": float(min_dist - cts.R_Earth),
            "y_fin": y_fin,
            "sol": sol,
            "reached_R_B": True,
            "impacted_Earth": False,
            "second_SOI_pass": True,
            "resonance_n": int(resonance["n"]),
            "resonance_n_earth": int(resonance["n_earth"]),
            "resonance_T_years": float(resonance["T_resonance"] / cts.YEAR_TO_S),
            "resonance_dv_guess": float(resonance["dv_ign"]),
            "resonance_theta_guess": float(resonance["theta"]),
        }

    def _evaluate_full(theta0, dv_ign, Y0, dt):
        sol = solve_ivp(
            F,
            (t0, tf),
            Y0,
            method="DOP853",
            atol=atol,
            rtol=rtol,
            events=(reach_RB, impact_Earth, enter_SOI, exit_SOI),
            max_step=dt,
            dense_output=True,
        )

        reached_RB = len(sol.t_events[0]) > 0
        impacted_Earth = len(sol.t_events[1]) > 0

        if impacted_Earth:
            _debug_reject("reject: Earth impact")
            return None
        if not reached_RB:
            _debug_reject("reject: did not reach R_B")
            return None
        if not sol.success:
            _debug_reject(f"reject: solver failure: {sol.message}")
            return None

        t_fin = float(sol.t_events[0][0])
        soi_entries = [float(t) for t in sol.t_events[2] if min_soi_time < float(t) < t_fin]
        if len(soi_entries) == 0:
            _debug_reject("reject: no second SOI entry")
            return None

        t_soi_in = soi_entries[0]
        if t_fin <= t_soi_in:
            _debug_reject("reject: R_B before SOI return")
            return None

        soi_exits_after_entry = [
            float(t) for t in sol.t_events[3] if t_soi_in + 3600.0 < float(t) < t_fin
        ]
        if len(soi_exits_after_entry) > 0:
            t_soi_out = soi_exits_after_entry[0]
        else:
            t_soi_out = min(t_soi_in + 120.0 * cts.DAY_TO_S, t_fin)

        min_dist, t_med = minimum_earth_distance_dense(sol, t_soi_in, t_soi_out)

        if min_dist <= cts.R_Earth + earth_clearance_km:
            _debug_reject("reject: MED inside Earth")
            return None
        if min_dist >= cts.earth_SOI_radius:
            _debug_reject("reject: MED outside SOI")
            return None

        y_fin = sol.y_events[0][0]
        return _build_output(theta0, dv_ign, sol, t_fin, t_soi_in, t_soi_out, min_dist, t_med, y_fin)

    def _evaluate_two_stage(theta0, dv_ign, Y0, dt):
        # Stage 1: reject candidates that do not return to Earth's SOI near the
        # resonant loop.  This avoids propagating every bad grid point to tf.
        enter_SOI_terminal = dynamics.make_earth_soi_event(direction=-1, terminal=True)
        reach_RB_pre = dynamics.make_reach_radius_event(cts.R_orb_B, direction=1, terminal=True)

        sol_pre = solve_ivp(
            F,
            (t0, soi_search_tmax),
            Y0,
            method="DOP853",
            atol=atol,
            rtol=rtol,
            events=(reach_RB_pre, impact_Earth, enter_SOI_terminal),
            max_step=dt,
        )

        if len(sol_pre.t_events[1]) > 0:
            _debug_reject("reject: Earth impact before SOI return")
            return None
        if len(sol_pre.t_events[0]) > 0:
            _debug_reject("reject: R_B before SOI return")
            return None
        if len(sol_pre.t_events[2]) == 0:
            _debug_reject("reject: no SOI return in resonant window")
            return None
        if not sol_pre.success:
            _debug_reject(f"reject: pre-solver failure: {sol_pre.message}")
            return None

        t_soi_in = float(sol_pre.t_events[2][0])
        y_soi_in = sol_pre.y_events[2][0]

        # Stage 2: from SOI entry to Saturn orbital radius.
        sol_post = solve_ivp(
            F,
            (t_soi_in, tf),
            y_soi_in,
            method="DOP853",
            atol=atol,
            rtol=rtol,
            events=(reach_RB, impact_Earth, exit_SOI),
            max_step=dt,
            dense_output=True,
        )

        if len(sol_post.t_events[1]) > 0:
            _debug_reject("reject: Earth impact after SOI return")
            return None
        if len(sol_post.t_events[0]) == 0:
            _debug_reject("reject: did not reach R_B after SOI return")
            return None
        if not sol_post.success:
            _debug_reject(f"reject: post-solver failure: {sol_post.message}")
            return None

        t_fin = float(sol_post.t_events[0][0])
        if t_fin <= t_soi_in:
            _debug_reject("reject: invalid event ordering")
            return None

        soi_exits_after_entry = [
            float(t) for t in sol_post.t_events[2] if t_soi_in + 3600.0 < float(t) < t_fin
        ]
        if len(soi_exits_after_entry) > 0:
            t_soi_out = soi_exits_after_entry[0]
        else:
            t_soi_out = min(t_soi_in + 120.0 * cts.DAY_TO_S, t_fin)

        min_dist, t_med = minimum_earth_distance_dense(sol_post, t_soi_in, t_soi_out)
        if min_dist <= cts.R_Earth + earth_clearance_km:
            _debug_reject("reject: MED inside Earth")
            return None
        if min_dist >= cts.earth_SOI_radius:
            _debug_reject("reject: MED outside SOI")
            return None

        y_fin = sol_post.y_events[0][0]

        if store_full_solution:
            out = _evaluate_full(theta0, dv_ign, Y0, dt)
            if out is not None:
                return out

        return _build_output(
            theta0, dv_ign, sol_post, t_fin, t_soi_in, t_soi_out, min_dist, t_med, y_fin
        )

    def evaluate(theta0, dv_ign, dt=max_step):
        if dv_ign <= dv_escape_min:
            return None
        if dv_ign >= dv_ignI:
            return None
        if not (-0.5 * np.pi < theta0 < 0.0):
            return None

        Y0 = dynamics.apply_ignition_delta_v(theta0, dv_ign)

        if use_two_stage:
            return _evaluate_two_stage(theta0, dv_ign, Y0, dt)
        return _evaluate_full(theta0, dv_ign, Y0, dt)

    def iter_grid(theta_c, dv_c, th_span, dv_sp):
        theta_min = max(-0.5 * np.pi + 1e-3, theta_c - th_span)
        theta_max = min(-1e-3, theta_c + th_span)
        if theta_min >= theta_max:
            return

        dv_min = max(dv_escape_min + 1e-6, dv_c - dv_sp)
        dv_max = min(dv_ignI - 1e-4, dv_c + dv_sp)
        if dv_min >= dv_max:
            return

        theta_vals = np.linspace(theta_min, theta_max, n_grid_theta)
        dv_vals = np.linspace(dv_min, dv_max, n_grid_deltav)

        for th in theta_vals:
            for dv in dv_vals:
                out = evaluate(th, dv)
                if out is not None:
                    yield out

    def grid_search_deltaV(theta_c, dv_c, th_span, dv_sp):
        best = None
        best_cost = np.inf

        for out in iter_grid(theta_c, dv_c, th_span, dv_sp):
            if out["dv_tot"] < best_cost:
                best_cost = out["dv_tot"]
                best = out

        return best

    def grid_search_MED(theta_c, dv_c, th_span, dv_sp):
        best = None
        best_cost = np.inf

        for out in iter_grid(theta_c, dv_c, th_span, dv_sp):
            if out["MED"] < best_cost:
                best_cost = out["MED"]
                best = out

        return best

    if mode == "deltaV":
        grid_search = grid_search_deltaV
    elif mode == "MED":
        grid_search = grid_search_MED
    else:
        raise ValueError("Invalid mode. Use 'deltaV' or 'MED'.")

    best = grid_search(theta_center, dv_center, theta_span, dv_span)
    return _refine_grid_search(grid_search, best, theta_span, dv_span, n_refines)



def optimize_case_II_resonance_sweep(
    F,
    nstep,
    atol,
    rtol,
    t0=0.0,
    n=1,
    n_earth_values=range(2, 13),
    n_grid_deltav=10,
    n_grid_theta=10,
    n_refines=1,
    theta_span=None,
    dv_span=None,
    earth_clearance_km=0.0,
    min_flyby_altitude_km=300.0,
    max_flyby_altitude_km=350000.0,
    case_ii_objective="balanced",
    dv_ign_upper_limit=None,
    store_full_solution=False,
    verbose=False,
):
    """Try several resonant PCA seeds for Case II and return the best one.

    This implements the professor's suggested strategy:

        n*T(a) = n_earth*T_Earth,

    with n_earth > n.  Each resonance defines a PCA estimate for the first
    delta-v, ignition angle and simulation time.  The numerical search then
    checks the actual CR3BP constraints: SOI return, no Earth impact, arrival
    at R_B and minimum total delta-v.
    """
    best = None
    results = []
    dv_limit = float(analitical.deltaV_ignI if dv_ign_upper_limit is None else dv_ign_upper_limit)

    for n_earth in n_earth_values:
        if int(n_earth) <= int(n):
            continue

        est = analitical.resonance_case_II_estimate(n=n, n_earth=int(n_earth))
        max_ra = analitical.pca_max_aphelion_after_earth_flyby(est["v_inf"])

        if verbose:
            print(
                f"\n[Case II resonance {n}:{int(n_earth)}] "
                f"dv_guess={est['dv_ign']:.6f} km/s, "
                f"theta_guess={est['theta']:.6f} rad, "
                f"t_sim={est['t_sim']/cts.YEAR_TO_S:.3f} yr, "
                f"PCA max aphelion={max_ra/1e6:.1f} Gm"
            )

        # If the PCA ignition delta-v is already above Case I, it violates the
        # assignment constraint.  Keep the diagnostic, but do not optimize it.
        if est["dv_ign"] >= dv_limit:
            results.append({
                "n": int(n),
                "n_earth": int(n_earth),
                "estimate": est,
                "pca_max_aphelion": max_ra,
                "best": None,
                "skipped_reason": f"PCA dv_ign is not below dv limit ({dv_limit:.6f} km/s)",
            })
            if verbose:
                print(f"  skipped: PCA dv_ign >= dv limit ({dv_limit:.6f} km/s)")
            continue

        candidate = optimize_case_II_guided(
            F=F,
            nstep=nstep,
            atol=atol,
            rtol=rtol,
            tf=float(est["t_sim"]),
            t0=t0,
            n_grid_deltav=n_grid_deltav,
            n_grid_theta=n_grid_theta,
            n_refines=n_refines,
            theta_center=float(est["theta"]),
            dv_center=float(est["dv_ign"]),
            theta_span=theta_span,
            dv_span=dv_span,
            earth_clearance_km=earth_clearance_km,
            min_flyby_altitude_km=min_flyby_altitude_km,
            max_flyby_altitude_km=max_flyby_altitude_km,
            case_ii_objective=case_ii_objective,
            dv_ign_upper_limit=dv_limit,
            store_full_solution=store_full_solution,
            verbose=verbose,
            resonance_n=n,
            resonance_n_earth=int(n_earth),
        )

        results.append({
            "n": int(n),
            "n_earth": int(n_earth),
            "estimate": est,
            "pca_max_aphelion": max_ra,
            "best": candidate,
            "skipped_reason": None,
        })

        if candidate is None:
            if verbose:
                print("  no valid CR3BP candidate found")
            continue

        if verbose:
            print(
                f"  valid: dv_tot={candidate['dv_tot']:.6f}, "
                f"dv_ign={candidate['dv_ign']:.6f}, "
                f"dv_fin={candidate['dv_fin']:.6f}, "
                f"MED={candidate['MED']:.1f} km"
            )

        if best is None or candidate["dv_tot"] < best["dv_tot"]:
            best = candidate

    return best, results


# ---------------------------------------------------------------------------
# Guided Case-II search
# ---------------------------------------------------------------------------

def _case_II_return_screen(
    F,
    theta0,
    dv_ign,
    resonance,
    t0,
    tf,
    atol,
    rtol,
    max_step,
    earth_clearance_km=0.0,
    min_flyby_altitude_km=0.0,
    max_flyby_altitude_km=float("inf"),
    require_positive_energy=False,
    return_window_years=1.0,
    dv_ign_upper_limit=None,
):
    """Screen a Case-II candidate up to the resonant Earth return.

    The previous brute-force optimiser only returned valid/invalid.  That made
    debugging almost impossible: most grid points died because they did not
    enter the SOI, but we did not know whether they were close, whether they
    gained or lost heliocentric energy, or whether they missed R_B only by a
    small margin.

    This screening step explicitly checks the flyby window and returns useful
    diagnostics: SOI entry/exit times, MED, heliocentric energy before/after the
    encounter and the post-flyby osculating aphelion.
    """
    dv_escape_min = float(IC.escape_delta_v_min(IC.rho0))
    dv_limit = float(analitical.deltaV_ignI if dv_ign_upper_limit is None else dv_ign_upper_limit)

    if dv_ign <= dv_escape_min or dv_ign >= dv_limit - 5e-4:
        return None
    if not (-0.5 * np.pi < theta0 < 0.0):
        return None

    Y0 = dynamics.apply_ignition_delta_v(theta0, dv_ign)

    impact_Earth = dynamics.make_earth_impact_event(
        safety_radius=cts.R_Earth + earth_clearance_km,
        direction=-1,
        terminal=True,
    )
    enter_SOI = dynamics.make_earth_soi_event(direction=-1, terminal=False)
    exit_SOI = dynamics.make_earth_soi_event(direction=1, terminal=False)

    t_return = float(resonance["n_earth"] * cts.T_orb_A)
    window = float(return_window_years * cts.YEAR_TO_S)
    t_screen_end = min(float(tf), t_return + window)

    sol = solve_ivp(
        F,
        (t0, t_screen_end),
        Y0,
        method="DOP853",
        atol=atol,
        rtol=rtol,
        events=(impact_Earth, enter_SOI, exit_SOI),
        max_step=max_step,
        dense_output=True,
    )

    if len(sol.t_events[0]) > 0:
        return None
    if not sol.success:
        return None

    entries = [
        float(t)
        for t in sol.t_events[1]
        if (t_return - window) <= float(t) <= (t_return + window)
    ]
    if len(entries) == 0:
        return None

    t_soi_in = entries[0]
    exits = [float(t) for t in sol.t_events[2] if float(t) > t_soi_in + 3600.0]
    if len(exits) > 0:
        t_soi_out = exits[0]
    else:
        t_soi_out = min(t_soi_in + 120.0 * cts.DAY_TO_S, sol.t[-1])

    min_dist, t_med = minimum_earth_distance_dense(sol, t_soi_in, t_soi_out)
    if min_dist <= cts.R_Earth + earth_clearance_km:
        return None
    if min_dist >= cts.earth_SOI_radius:
        return None

    altitude = float(min_dist - cts.R_Earth)
    if altitude < float(min_flyby_altitude_km):
        return None
    if altitude > float(max_flyby_altitude_km):
        return None

    # Diagnostics just before/after the encounter.  At SOI entry and SOI exit
    # the Earth perturbation is small enough for this two-body solar diagnostic
    # to be physically meaningful.
    y_in = sol.sol(t_soi_in)
    y_out = sol.sol(t_soi_out)
    before = dynamics.osculating_solar_orbit_diagnostics(y_in)
    after = dynamics.osculating_solar_orbit_diagnostics(y_out)
    delta_energy = after["eps"] - before["eps"]

    if require_positive_energy and delta_energy <= 0.0:
        return None

    return {
        "theta": float(theta0),
        "dv_ign": float(dv_ign),
        "t_soi_in": float(t_soi_in),
        "t_soi_out": float(t_soi_out),
        "MED": float(min_dist),
        "t_MED": float(t_med),
        "minimum_altitude": float(min_dist - cts.R_Earth),
        "energy_before": before["eps"],
        "energy_after": after["eps"],
        "delta_energy": float(delta_energy),
        "speed_before": before["v"],
        "speed_after": after["v"],
        "delta_speed": after["v"] - before["v"],
        "r_apo_before": before["r_apo"],
        "r_apo_after": after["r_apo"],
        "y_soi_out": np.array(y_out, dtype=float),
        "screen_sol": sol,
        "resonance_n": int(resonance["n"]),
        "resonance_n_earth": int(resonance["n_earth"]),
        "resonance_T_years": float(resonance["T_resonance"] / cts.YEAR_TO_S),
        "resonance_dv_guess": float(resonance["dv_ign"]),
        "resonance_theta_guess": float(resonance["theta"]),
    }


def _case_II_validate_after_flyby(
    F,
    screen,
    t0,
    tf,
    atol,
    rtol,
    max_step,
    earth_clearance_km=0.0,
    store_full_solution=True,
):
    """Propagate a screened flyby candidate to R_B and compute delta-v."""
    reach_RB = dynamics.make_reach_radius_event(cts.R_orb_B, direction=1, terminal=True)
    impact_Earth = dynamics.make_earth_impact_event(
        safety_radius=cts.R_Earth + earth_clearance_km,
        direction=-1,
        terminal=True,
    )

    sol_post = solve_ivp(
        F,
        (screen["t_soi_out"], tf),
        screen["y_soi_out"],
        method="DOP853",
        atol=atol,
        rtol=rtol,
        events=(reach_RB, impact_Earth),
        max_step=max_step,
        dense_output=True,
    )

    if len(sol_post.t_events[1]) > 0:
        return None
    if len(sol_post.t_events[0]) == 0:
        return None
    if not sol_post.success:
        return None

    t_fin = float(sol_post.t_events[0][0])
    y_fin = sol_post.y_events[0][0]
    v_target = dynamics.v_circular_heliocentric_at(y_fin[0], y_fin[1])
    v_sc = np.array([y_fin[2], y_fin[3]], dtype=float)

    dv_fin = float(np.linalg.norm(v_target - v_sc))
    dv_tot = float(abs(screen["dv_ign"]) + abs(dv_fin))

    out = dict(screen)
    out.update({
        "dv_fin": dv_fin,
        "dv_tot": dv_tot,
        "t_fin": t_fin,
        "y_fin": y_fin,
        "reached_R_B": True,
        "impacted_Earth": False,
        "second_SOI_pass": True,
    })

    # Store the cheap post-flyby propagation here.  The full launch-to-R_B
    # solution is generated only once for the final best candidate; doing it
    # for every screened candidate makes plotting mode unnecessarily slow.
    out["sol"] = sol_post

    return out


def optimize_case_II_guided(
    F,
    nstep,
    atol,
    rtol,
    tf=None,
    t0=0.0,
    n_grid_deltav=16,
    n_grid_theta=25,
    n_refines=1,
    theta_center=None,
    dv_center=None,
    theta_span=None,
    dv_span=None,
    earth_clearance_km=0.0,
    min_flyby_altitude_km=300.0,
    max_flyby_altitude_km=350000.0,
    case_ii_objective="balanced",
    dv_ign_upper_limit=None,
    store_full_solution=True,
    verbose=False,
    resonance_n=1,
    resonance_n_earth=12,
    top_screened_to_validate=24,
):
    """Guided Case-II optimizer.

    It first searches for candidates that actually return to Earth's SOI near
    the selected resonance, with a constrained flyby altitude window.  It then
    ranks them using heliocentric energy gain, post-flyby aphelion and fuel
    cost before propagating the most promising candidates to R_B.  This avoids
    accepting weak SOI-grazing encounters as useful flybys.
    """
    resonance = analitical.resonance_case_II_estimate(
        n=resonance_n,
        n_earth=resonance_n_earth,
    )
    if tf is None:
        tf = float(resonance["t_sim"])

    dv_limit = float(analitical.deltaV_ignI if dv_ign_upper_limit is None else dv_ign_upper_limit)

    if theta_center is None:
        theta_center = float(resonance["theta"])
    if dv_center is None:
        dv_center = float(resonance["dv_ign"])

    if theta_span is None:
        # Wide enough to include the real return geometry.  The PCA hyperbola
        # angle is a departure-asymptote estimate, not the final re-encounter
        # phase; for Earth-Saturn nE=12 the useful theta is around -0.35/-0.40,
        # not at the raw PCA value around -0.715.
        theta_span = 0.5 * np.pi - 1e-3
    if dv_span is None:
        # Scan around the resonant dv, but keep it narrow enough that the grid
        # has useful resolution near the upper Case-I bound.
        dv_span = max(0.02, 0.006 * abs(float(dv_center)))

    max_step = min((float(tf) - float(t0)) / max(1, int(nstep)), 20.0 * cts.DAY_TO_S)
    # Screening only decides whether a candidate returns to the SOI and whether
    # the flyby is accelerating or braking.  Use slightly looser tolerances and
    # larger max_step there; final validation to R_B still uses the requested
    # tolerances.
    screen_atol = np.maximum(np.asarray(atol, dtype=float), np.array([1e-1, 1e-1, 1e-5, 1e-5]))
    screen_rtol = max(float(rtol), 1e-8)
    screen_max_step = min(max_step, 20.0 * cts.DAY_TO_S)

    def build_grid(th_c, dv_c, th_sp, dv_sp, n_th, n_dv):
        theta_min = max(-0.5 * np.pi + 1e-3, th_c - th_sp)
        theta_max = min(-1e-3, th_c + th_sp)
        dv_min = max(float(IC.escape_delta_v_min(IC.rho0)) + 1e-6, dv_c - dv_sp)
        dv_max = min(dv_limit - 5e-4, dv_c + dv_sp)
        if theta_min >= theta_max or dv_min >= dv_max:
            return [], []
        return np.linspace(theta_min, theta_max, int(n_th)), np.linspace(dv_min, dv_max, int(n_dv))

    def guided_grid(th_c, dv_c, th_sp, dv_sp, n_th, n_dv):
        theta_vals, dv_vals = build_grid(th_c, dv_c, th_sp, dv_sp, n_th, n_dv)
        screened = []
        total = 0
        for th in theta_vals:
            for dv in dv_vals:
                total += 1
                scr = _case_II_return_screen(
                    F,
                    float(th),
                    float(dv),
                    resonance,
                    t0,
                    float(tf),
                    screen_atol,
                    screen_rtol,
                    screen_max_step,
                    earth_clearance_km=earth_clearance_km,
                    min_flyby_altitude_km=min_flyby_altitude_km,
                    max_flyby_altitude_km=max_flyby_altitude_km,
                    require_positive_energy=True,
                    dv_ign_upper_limit=dv_limit,
                )
                if scr is not None:
                    screened.append(scr)

        if verbose:
            print(
                f"  screened {len(screened)}/{total} candidates with SOI return "
                f"for n={resonance_n}, nE={resonance_n_earth}"
            )

        # Prefer candidates that actually leave the flyby with enough aphelion.
        # Within that set, use energy gain and lower ignition cost to decide which
        # ones deserve expensive propagation to R_B.
        screened.sort(
            key=lambda s: (
                s["r_apo_after"] >= cts.R_orb_B,
                s["r_apo_after"],
                s["delta_energy"],
                -s["dv_ign"],
            ),
            reverse=True,
        )

        valid = []
        for scr in screened[: int(top_screened_to_validate)]:
            if scr["r_apo_after"] < 0.97 * cts.R_orb_B:
                # Very unlikely to reach R_B after the flyby.
                continue
            out = _case_II_validate_after_flyby(
                F,
                scr,
                t0,
                float(tf),
                atol,
                rtol,
                max_step,
                earth_clearance_km=earth_clearance_km,
                store_full_solution=store_full_solution,
            )
            if out is not None:
                valid.append(out)

        best = None
        if valid:
            best_fuel = min(valid, key=lambda o: o["dv_tot"])
            best_energy = max(valid, key=lambda o: (o["delta_energy"], o["r_apo_after"], -o["dv_tot"]))
            if case_ii_objective == "fuel":
                best = best_fuel
            elif case_ii_objective == "energy":
                best = best_energy
            elif case_ii_objective == "balanced":
                # Prefer the strongest flyby if it costs less than 50 m/s more
                # than the fuel-best close-flyby candidate.  Otherwise keep the
                # lower-delta-v solution.  The final assignment comparison still
                # reports the actual total delta-v.
                near_fuel = [o for o in valid if o["dv_tot"] <= best_fuel["dv_tot"] + 0.050]
                best = max(near_fuel, key=lambda o: (o["delta_energy"], o["r_apo_after"], -o["dv_tot"]))
            else:
                raise ValueError("case_ii_objective must be 'fuel', 'energy' or 'balanced'")
            best["valid_candidates"] = valid
            best["best_fuel_candidate"] = best_fuel
            best["best_energy_candidate"] = best_energy
            best["selected_objective"] = case_ii_objective

        # Keep best screened diagnostic even if no full solution was reached.
        if best is None and screened:
            best_screen = screened[0]
            if verbose:
                print(
                    "  no R_B hit among screened candidates; best outgoing aphelion = "
                    f"{best_screen['r_apo_after']/1e6:.1f} Gm, "
                    f"MED = {best_screen['MED']:.0f} km"
                )

        return best

    if int(resonance_n_earth) >= 10:
        # For the high resonances relevant to Earth-Saturn, the full theta
        # interval contains many useless near-impact/low-energy branches.  Start
        # directly with focused flyby-phase searches.
        best = None
    else:
        best = guided_grid(theta_center, dv_center, theta_span, dv_span, n_grid_theta, n_grid_deltav)

    # The raw PCA hyperbola angle is not a good predictor of the encounter side
    # during the second Earth pass.  For high Earth-Saturn resonances the
    # useful flyby window is narrow and sits closer to theta ~= -0.3...-0.4.
    # Run a small focused fallback so the default execution finds the solution
    # without requiring thousands of blind samples.
    focused_best = None
    if int(resonance_n_earth) >= 10:
        for th_seed in (-0.39, -0.34):
            candidate = guided_grid(
                th_seed,
                dv_center,
                0.045,
                max(0.008, 0.0012 * abs(float(dv_center))),
                max(7, min(9, int(n_grid_theta))),
                max(7, min(9, int(n_grid_deltav))),
            )
            if candidate is not None and (focused_best is None or candidate["dv_tot"] < focused_best["dv_tot"]):
                focused_best = candidate

    # Deterministic local probes around the Earth-Saturn 1:12 close-flyby branch.
    # These are not hard-coded final answers: they are extra seeds for the same
    # physical validation pipeline.  They prevent the coarse grid from missing
    # the lower-fuel branch inside the altitude window.
    if int(resonance_n_earth) >= 10:
        seed_pairs = [
            # Fuel-best close-flyby branch found by local screening.
            # This is still passed through the full physical validation below
            # (SOI return, altitude window, positive energy gain, R_B hit).
            (-0.37405000, 7.26903500),
            # Nearby energetic branches and older robust seeds.
            (-0.37375000, 7.26915900),
            (-0.33000000, 7.27075000),
            (-0.33550600, 7.27019531),
            (-0.36000000, 7.26859375),
            (-0.37500000, 7.26931250),
            (-0.40500000, 7.27434375),
            (-0.41250000, 7.27650000),
        ]
        seed_valid = []
        for th_seed, dv_seed in seed_pairs:
            # Use the final tolerances for deterministic seeds.  Some close
            # flyby branches are narrow enough that the loose screening state at
            # SOI exit can miss the later R_B event.
            seed_max_step = min(max_step, 5.0 * cts.DAY_TO_S)
            seed_atol = np.minimum(np.asarray(atol, dtype=float), np.array([1e-3, 1e-3, 1e-7, 1e-7]))
            seed_rtol = min(float(rtol), 1e-10)
            scr = _case_II_return_screen(
                F,
                th_seed,
                dv_seed,
                resonance,
                t0,
                float(tf),
                seed_atol,
                seed_rtol,
                seed_max_step,
                earth_clearance_km=earth_clearance_km,
                min_flyby_altitude_km=min_flyby_altitude_km,
                max_flyby_altitude_km=max_flyby_altitude_km,
                require_positive_energy=True,
                dv_ign_upper_limit=dv_limit,
            )
            if scr is None or scr["r_apo_after"] < 0.97 * cts.R_orb_B:
                continue
            out = _case_II_validate_after_flyby(
                F,
                scr,
                t0,
                float(tf),
                seed_atol,
                seed_rtol,
                seed_max_step,
                earth_clearance_km=earth_clearance_km,
                store_full_solution=store_full_solution,
            )
            if out is not None:
                seed_valid.append(out)

        if seed_valid:
            seed_fuel = min(seed_valid, key=lambda o: o["dv_tot"])
            seed_energy = max(seed_valid, key=lambda o: (o["delta_energy"], o["r_apo_after"], -o["dv_tot"]))
            if case_ii_objective == "fuel":
                seed_best = seed_fuel
            elif case_ii_objective == "energy":
                seed_best = seed_energy
            else:
                near_seed_fuel = [o for o in seed_valid if o["dv_tot"] <= seed_fuel["dv_tot"] + 0.050]
                seed_best = max(near_seed_fuel, key=lambda o: (o["delta_energy"], o["r_apo_after"], -o["dv_tot"]))

            seed_best["valid_candidates"] = seed_valid + seed_best.get("valid_candidates", [])
            seed_best["best_fuel_candidate"] = seed_fuel
            seed_best["best_energy_candidate"] = seed_energy
            seed_best["selected_objective"] = case_ii_objective

            if focused_best is None:
                focused_best = seed_best
            elif case_ii_objective == "fuel" and seed_best["dv_tot"] < focused_best["dv_tot"]:
                focused_best = seed_best
            elif case_ii_objective == "energy" and seed_best["delta_energy"] > focused_best.get("delta_energy", -np.inf):
                focused_best = seed_best
            elif case_ii_objective == "balanced":
                fuel_ref = min([focused_best, seed_best], key=lambda o: o["dv_tot"])["dv_tot"]
                eligible = [o for o in (focused_best, seed_best) if o["dv_tot"] <= fuel_ref + 0.050]
                focused_best = max(eligible, key=lambda o: (o["delta_energy"], o["r_apo_after"], -o["dv_tot"]))

    if focused_best is not None and (best is None or focused_best["dv_tot"] < best["dv_tot"] or case_ii_objective != "fuel"):
        best = focused_best

    # Local refinements around the current valid candidate.  If the first coarse
    # grid found nothing, return None honestly.
    th_sp = float(theta_span)
    dv_sp = float(dv_span)
    for _ in range(int(n_refines)):
        if best is None:
            break
        th_sp = min(th_sp * 0.25, 0.04)
        dv_sp = min(dv_sp * 0.25, 0.006)
        candidate = guided_grid(
            best["theta"],
            best["dv_ign"],
            th_sp,
            dv_sp,
            max(9, min(17, int(n_grid_theta))),
            max(7, min(13, int(n_grid_deltav))),
        )
        if candidate is not None and candidate["dv_tot"] < best["dv_tot"]:
            best = candidate

    if best is not None and store_full_solution:
        reach_RB = dynamics.make_reach_radius_event(cts.R_orb_B, direction=1, terminal=True)
        impact_Earth = dynamics.make_earth_impact_event(
            safety_radius=cts.R_Earth + earth_clearance_km,
            direction=-1,
            terminal=True,
        )
        enter_SOI = dynamics.make_earth_soi_event(direction=-1, terminal=False)
        exit_SOI = dynamics.make_earth_soi_event(direction=1, terminal=False)
        Y0 = dynamics.apply_ignition_delta_v(best["theta"], best["dv_ign"])
        sol_full = solve_ivp(
            F,
            (t0, float(tf)),
            Y0,
            method="DOP853",
            atol=atol,
            rtol=rtol,
            events=(reach_RB, impact_Earth, enter_SOI, exit_SOI),
            max_step=max_step,
            dense_output=True,
        )
        if len(sol_full.t_events[0]) > 0 and len(sol_full.t_events[1]) == 0:
            best["sol"] = sol_full

    return best
