import numpy as np
from scipy.integrate import solve_ivp

from include import cts, analitical, IC, plotter


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
    dv_span=None
):
    if theta_center is None:
        theta_center = float(analitical.theta_0I)

    if dv_center is None:
        dv_center = float(analitical.deltaV_ignI)

    if dv_span is None:
        dv_span = 0.3 * dv_center

    def reach_RB(t, Y):
        # detecta la llegada a punto lagrange de planeta B
        return np.hypot(Y[0], Y[1]) - cts.R_orb_B

    reach_RB.terminal = True
    reach_RB.direction = 1

    def v_circ_at_r(x, y):
        r = np.hypot(x, y)
        vmod = np.sqrt(cts.mu_sun / r)
        t_hat = np.array([-y / r, x / r])

        return vmod * t_hat

    def evaluate(theta0, dv_ign, dt):
        baseY0, t_hat_theta = IC.ICtoY0(
            IC.rho0,
            theta0=theta0,
            delta0=IC.delta0
        )

        V_ign = dv_ign * t_hat_theta
        Y0 = baseY0 + np.array([0.0, 0.0, V_ign[0], V_ign[1]])

        sol = solve_ivp(
            F,
            (t0, tf),
            Y0,
            method="DOP853",
            atol=atol,
            rtol=rtol,
            events=reach_RB,
            max_step=dt
        )

        if len(sol.t_events[0]) == 0:
            return None

        t_fin = sol.t_events[0][0]
        y_fin = sol.y_events[0][0]

        v_target = v_circ_at_r(y_fin[0], y_fin[1])
        v_sc = np.array([y_fin[2], y_fin[3]])

        dv_fin = np.linalg.norm(v_target - v_sc)
        dv_tot = abs(dv_ign) + abs(dv_fin)

        return {
            "theta": theta0,
            "dv_ign": dv_ign,
            "dv_fin": dv_fin,
            "dv_tot": dv_tot,
            "t_fin": t_fin,
            "y_fin": y_fin,
            "sol": sol
        }

    def grid_search(theta_c, dv_c, th_span, dv_sp):
        dt = (tf - t0) / nstep

        theta_vals = np.linspace(theta_c - th_span, theta_c + th_span, n_grid)
        dv_vals = np.linspace(dv_c - dv_sp, dv_c + dv_sp, n_grid)

        best = None
        best_cost = np.inf

        for th in theta_vals:
            for dv in dv_vals:
                if dv <= 0:
                    continue

                out = evaluate(th, dv, dt)

                if out is None:
                    continue

                if out["dv_tot"] < best_cost:
                    best_cost = out["dv_tot"]
                    best = out

        return best

    best = grid_search(theta_center, dv_center, theta_span, dv_span)

    if best is None:
        return None

    th_span = theta_span
    dv_sp = dv_span

    for _ in range(n_refines):
        th_span *= 0.25
        dv_sp *= 0.25

        best = grid_search(best["theta"], best["dv_ign"], th_span, dv_sp)

        if best is None:
            return None

    return best

def earth_position(t):
    return np.array([
        cts.R_orb_A * np.cos(cts.frec_A * t - IC.delta0),
        cts.R_orb_A * np.sin(cts.frec_A * t - IC.delta0)
    ])


def earth_distance(t, Y):
    R_earth = earth_position(t)

    return np.hypot(Y[0] - R_earth[0], Y[1] - R_earth[1])


def v_circ_at_r(x, y):
    r = np.hypot(x, y)
    vmod = np.sqrt(cts.mu_sun / r)
    t_hat = np.array([-y/r, x/r])

    return vmod * t_hat


def solar_energy(Y):
    x, y, vx, vy = Y

    r = np.hypot(x, y)
    v2 = vx*vx + vy*vy

    return 0.5*v2 - cts.mu_sun/r


def solar_aphelion(Y):
    x, y, vx, vy = Y

    r_vec = np.array([x, y])
    v_vec = np.array([vx, vy])

    r = np.linalg.norm(r_vec)
    v2 = np.dot(v_vec, v_vec)

    energy = 0.5*v2 - cts.mu_sun/r

    if energy >= 0:
        return np.inf

    h = x*vy - y*vx
    e = np.sqrt(1 + 2*energy*h*h/(cts.mu_sun*cts.mu_sun))
    a = -cts.mu_sun/(2*energy)

    return a*(1 + e)


def initial_state(theta0, dv_ign):
    baseY0, t_hat_theta = IC.ICtoY0(
        IC.rho0,
        theta0=theta0,
        delta0=IC.delta0
    )

    V_ign = dv_ign * t_hat_theta

    Y0 = baseY0.copy()
    Y0[2:4] += V_ign

    return Y0


def make_reach_RB_event():
    def reach_RB(t, Y):
        return np.hypot(Y[0], Y[1]) - cts.R_orb_B

    reach_RB.terminal = True
    reach_RB.direction = 1

    return reach_RB


def make_earth_impact_event(earth_clearance_km=0.0):
    def impact_earth(t, Y):
        return earth_distance(t, Y) - (cts.R_Earth + earth_clearance_km)

    impact_earth.terminal = True
    impact_earth.direction = -1

    return impact_earth


def make_enter_SOI_event():
    def enter_SOI(t, Y):
        return earth_distance(t, Y) - cts.earth_SOI_radius

    enter_SOI.terminal = False
    enter_SOI.direction = -1

    return enter_SOI


def make_exit_SOI_event():
    def exit_SOI(t, Y):
        return earth_distance(t, Y) - cts.earth_SOI_radius

    exit_SOI.terminal = False
    exit_SOI.direction = 1

    return exit_SOI


def minimum_earth_distance(sol, t1, t2):
    t_eval = np.linspace(t1, t2, 300)

    Y = sol.sol(t_eval)

    earth_x = cts.R_orb_A * np.cos(cts.frec_A*t_eval - IC.delta0)
    earth_y = cts.R_orb_A * np.sin(cts.frec_A*t_eval - IC.delta0)

    d = np.hypot(Y[0] - earth_x, Y[1] - earth_y)

    i_min = d.argmin()

    return float(d[i_min]), float(t_eval[i_min])


def case_II_screen_flyby(
    F,
    theta0,
    dv_ign,
    resonance,
    t0,
    atol,
    rtol,
    max_step,
    earth_clearance_km=0.0,
    min_flyby_altitude_km=300.0,
    max_flyby_altitude_km=350000.0,
    return_window_years=1.0
):
    dv_escape_min = (np.sqrt(2) - 1) * np.sqrt(cts.mu_earth/IC.rho0)

    if dv_ign <= dv_escape_min:
        return None

    if dv_ign >= float(analitical.deltaV_ignI):
        return None

    if theta0 <= -0.5*np.pi or theta0 >= 0:
        return None

    Y0 = initial_state(theta0, dv_ign)

    impact_earth = make_earth_impact_event(earth_clearance_km)
    enter_SOI = make_enter_SOI_event()
    exit_SOI = make_exit_SOI_event()

    t_return = float(resonance["n_earth"] * cts.T_orb_A)
    window = float(return_window_years * cts.year_to_s)

    tf_screen = t_return + window

    sol = solve_ivp(
        F,
        (t0, tf_screen),
        Y0,
        method="DOP853",
        atol=atol,
        rtol=rtol,
        events=(impact_earth, enter_SOI, exit_SOI),
        max_step=max_step,
        dense_output=True
    )

    if len(sol.t_events[0]) > 0:
        return None

    if not sol.success:
        return None

    soi_entries = [
        float(t) for t in sol.t_events[1]
        if (t_return - window) <= float(t) <= (t_return + window)
    ]

    if len(soi_entries) == 0:
        return None

    t_SOI_in = soi_entries[0]

    soi_exits = [
        float(t) for t in sol.t_events[2]
        if float(t) > t_SOI_in + 3600.0
    ]

    if len(soi_exits) > 0:
        t_SOI_out = soi_exits[0]
    else:
        t_SOI_out = min(t_SOI_in + 120*24*3600.0, sol.t[-1])

    MED, t_MED = minimum_earth_distance(sol, t_SOI_in, t_SOI_out)

    altitude = MED - cts.R_Earth

    if altitude < min_flyby_altitude_km:
        return None

    if altitude > max_flyby_altitude_km:
        return None

    if MED >= cts.earth_SOI_radius:
        return None

    Y_before = sol.sol(t_SOI_in)
    Y_after = sol.sol(t_SOI_out)

    energy_before = solar_energy(Y_before)
    energy_after = solar_energy(Y_after)
    energy_gain = energy_after - energy_before

    if energy_gain <= -1e-6:
        return None

    r_apo_after = solar_aphelion(Y_after)

    return {
        "theta": float(theta0),
        "dv_ign": float(dv_ign),
        "t_SOI_in": float(t_SOI_in),
        "t_SOI_out": float(t_SOI_out),
        "MED": float(MED),
        "t_MED": float(t_MED),
        "minimum_altitude": float(altitude),
        "energy_before": float(energy_before),
        "energy_after": float(energy_after),
        "energy_gain": float(energy_gain),
        "r_apo_after": float(r_apo_after),
        "Y_SOI_out": np.array(Y_after, dtype=float),
        "resonance_n": int(resonance["n"]),
        "resonance_n_earth": int(resonance["n_earth"])
    }


def case_II_validate_to_RB(
    F,
    screen,
    tf,
    atol,
    rtol,
    max_step,
    earth_clearance_km=0.0
):
    reach_RB = make_reach_RB_event()
    impact_earth = make_earth_impact_event(earth_clearance_km)

    sol = solve_ivp(
        F,
        (screen["t_SOI_out"], tf),
        screen["Y_SOI_out"],
        method="DOP853",
        atol=atol,
        rtol=rtol,
        events=(reach_RB, impact_earth),
        max_step=max_step,
        dense_output=True
    )

    if len(sol.t_events[1]) > 0:
        return None

    if len(sol.t_events[0]) == 0:
        return None

    if not sol.success:
        return None

    t_fin = float(sol.t_events[0][0])
    y_fin = sol.y_events[0][0]

    v_target = v_circ_at_r(y_fin[0], y_fin[1])
    v_sc = np.array([y_fin[2], y_fin[3]])

    dv_fin = np.linalg.norm(v_target - v_sc)
    dv_tot = abs(screen["dv_ign"]) + abs(dv_fin)

    out = dict(screen)

    out.update({
        "dv_fin": float(dv_fin),
        "dv_tot": float(dv_tot),
        "t_fin": float(t_fin),
        "y_fin": y_fin,
        "sol": sol,
        "reached_R_B": True,
        "second_SOI_pass": True
    })

    return out

def optimize_case_II_guided(
    F,
    nstep,
    atol,
    rtol,
    resonance,
    t0=0.0,
    n_grid_deltav=11,
    n_grid_theta=11,
    n_refines=1,
    theta_span=None,
    dv_span=None,
    min_flyby_altitude_km=300.0,
    max_flyby_altitude_km=350000.0
):
    tf = resonance["T_transfer_case_II"]

    max_step = min((tf - t0)/nstep, 20*24*3600.0)

    if theta_span is None:
        theta_span = 0.5*np.pi - 1e-3

    if dv_span is None:
        dv_span = max(0.02, 0.006*abs(resonance["deltaV_ignII"]))

    def grid_search(theta_center, dv_center, th_span, dv_sp):
        theta_min = max(-0.5*np.pi + 1e-3, theta_center - th_span)
        theta_max = min(-1e-3, theta_center + th_span)

        dv_escape_min = (np.sqrt(2) - 1) * np.sqrt(cts.mu_earth/IC.rho0)

        dv_min = max(dv_escape_min + 1e-6, dv_center - dv_sp)
        dv_max = min(float(analitical.deltaV_ignI) - 5e-4, dv_center + dv_sp)

        if theta_min >= theta_max or dv_min >= dv_max:
            return None

        theta_values = np.linspace(theta_min, theta_max, n_grid_theta)
        dv_values = np.linspace(dv_min, dv_max, n_grid_deltav)

        screens = []

        for theta0 in theta_values:
            for dv_ign in dv_values:
                screen = case_II_screen_flyby(
                    F,
                    theta0,
                    dv_ign,
                    resonance,
                    t0,
                    atol,
                    rtol,
                    max_step,
                    min_flyby_altitude_km=min_flyby_altitude_km,
                    max_flyby_altitude_km=max_flyby_altitude_km
                )

                if screen is not None:
                    screens.append(screen)

        if len(screens) == 0:
            return None

        screens.sort(
            key=lambda s: (
                s["r_apo_after"] >= cts.R_orb_B,
                s["r_apo_after"],
                s["energy_gain"],
                -s["dv_ign"]
            ),
            reverse=True
        )

        valid = []

        for screen in screens[:20]:
            if screen["r_apo_after"] < 0.97*cts.R_orb_B:
                continue

            out = case_II_validate_to_RB(
                F,
                screen,
                tf,
                atol,
                rtol,
                max_step
            )

            if out is not None:
                valid.append(out)

        if len(valid) == 0:
            return None

        return min(valid, key=lambda o: o["dv_tot"])

    best = None

    # Resonancias bajas: búsqueda alrededor de la estimación analítica.
    if resonance["n_earth"] < 10:
        return grid_search(
            resonance["theta_0II"],
            resonance["deltaV_ignII"],
            theta_span,
            dv_span
        )

    # Resonancias altas Tierra-Saturno:
    # la estimación analítica da bien el dv, pero no centra bien theta.
    seeds = [
        (-0.39000, resonance["deltaV_ignII"]),
        (-0.34000, resonance["deltaV_ignII"]),
        (-0.37405, 7.26903500),
        (-0.37375, 7.26915900),
        (-0.33000, 7.27075000),
        (-0.33550, 7.27019531),
        (-0.36000, 7.26859375),
        (-0.37500, 7.26931250),
        (-0.40500, 7.27434375),
        (-0.41250, 7.27650000),
    ]

    seed_valid = []

    seed_max_step = min(max_step, 5.0*24*3600.0)

    seed_atol = np.minimum(
        np.asarray(atol, dtype=float),
        np.array([1e-3, 1e-3, 1e-7, 1e-7])
    )

    seed_rtol = min(float(rtol), 1e-10)

    for theta_seed, dv_seed in seeds:
        screen = case_II_screen_flyby(
            F,
            theta_seed,
            dv_seed,
            resonance,
            t0,
            seed_atol,
            seed_rtol,
            seed_max_step,
            min_flyby_altitude_km=min_flyby_altitude_km,
            max_flyby_altitude_km=max_flyby_altitude_km
        )

        if screen is None:
            continue

        if screen["r_apo_after"] < 0.97*cts.R_orb_B:
            continue

        out = case_II_validate_to_RB(
            F,
            screen,
            tf,
            seed_atol,
            seed_rtol,
            seed_max_step
        )

        if out is not None:
            seed_valid.append(out)

    if len(seed_valid) > 0:
        best = min(seed_valid, key=lambda o: o["dv_tot"])

    if best is None:
        # Fallback: búsqueda local alrededor de las semillas.
        for theta_seed, dv_seed in seeds:
            candidate = grid_search(
                theta_seed,
                dv_seed,
                0.045,
                max(0.008, 0.0012*abs(dv_seed))
            )

            if candidate is None:
                continue

            if best is None or candidate["dv_tot"] < best["dv_tot"]:
                best = candidate

    if best is None:
        return None

    # Refinamiento opcional. Para pruebas rápidas, pon n_refines=0 en test_case_II.py.
    for _ in range(n_refines):
        candidate = grid_search(
            best["theta"],
            best["dv_ign"],
            0.010,
            0.0015
        )

        if candidate is not None and candidate["dv_tot"] < best["dv_tot"]:
            best = candidate

    return best

def optimize_case_II_resonance_sweep(
    F,
    nstep,
    atol,
    rtol,
    t0=0.0,
    n=1,
    n_earth_values=range(2, 13),
    n_grid_deltav=11,
    n_grid_theta=11,
    n_refines=1,
    min_flyby_altitude_km=300.0,
    max_flyby_altitude_km=350000.0
):
    best = None
    results = []

    for n_earth in n_earth_values:
        resonance = analitical.resonance_case_II_estimate(n=n, n_earth=n_earth)

        print(
            "\nTesting resonance",
            str(n) + ":" + str(n_earth),
            "| dv_guess =",
            resonance["deltaV_ignII"],
            "| theta_guess =",
            resonance["theta_0II"],
            "| tf years =",
            resonance["T_transfer_case_II"]/cts.year_to_s
        )

        if resonance["deltaV_ignII"] >= float(analitical.deltaV_ignI):
            print("Skipped: dv_ignII >= dv_ignI")
            results.append({
                "resonance": resonance,
                "best": None
            })
            continue

        candidate = optimize_case_II_guided(
            F,
            nstep,
            atol,
            rtol,
            resonance,
            t0=t0,
            n_grid_deltav=n_grid_deltav,
            n_grid_theta=n_grid_theta,
            n_refines=n_refines,
            min_flyby_altitude_km=min_flyby_altitude_km,
            max_flyby_altitude_km=max_flyby_altitude_km
        )

        results.append({
            "resonance": resonance,
            "best": candidate
        })

        if candidate is None:
            print("No valid solution")
            continue

        print(
            "Valid solution:",
            "dv_tot =",
            candidate["dv_tot"],
            "| dv_ign =",
            candidate["dv_ign"],
            "| dv_fin =",
            candidate["dv_fin"],
            "| MED =",
            candidate["MED"],
            "| altitude =",
            candidate["minimum_altitude"]
        )

        if best is None or candidate["dv_tot"] < best["dv_tot"]:
            best = candidate

    return best, results


def optimize_case_II(
    F,
    nstep,
    atol,
    rtol,
    tf=None,
    t0=0.0,
    n_grid_deltav=11,
    n_grid_theta=11,
    n_refines=1,
    theta_center=None,
    dv_center=None,
    theta_span=None,
    dv_span=None,
    narrowband_exponent=2,
    mode="deltaV"
):
    # Wrapper retrocompatible:
    # si el código viejo llama a optimize_case_II, usa el barrido resonante.
    best, _ = optimize_case_II_resonance_sweep(
        F,
        nstep,
        atol,
        rtol,
        t0=t0,
        n=1,
        n_earth_values=range(2, 13),
        n_grid_deltav=n_grid_deltav,
        n_grid_theta=n_grid_theta,
        n_refines=n_refines
    )

    return best