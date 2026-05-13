import numpy as np
from scipy.integrate import solve_ivp
from include import cts, analitical, IC

def optimize_case_I(F, nstep, atol, rtol, tf, t0=0.0, n_grid=10, n_refines=2,
                    theta_center=None, dv_center=None, theta_span=0.5, dv_span=None):
    if theta_center is None:
        theta_center = float(analitical.theta_0I)
    if dv_center is None:
        dv_center = float(analitical.deltaV_ignI)
    if dv_span is None:
        dv_span = 0.3 * dv_center

    def reach_RB(t, Y):
        return np.hypot(Y[0], Y[1]) - cts.R_orb_B
    reach_RB.terminal = True
    reach_RB.direction = 1

    def v_circ_at_r(x, y):
        r = np.hypot(x, y)
        vmod = np.sqrt(cts.mu_sun / r)
        t_hat = np.array([-y / r, x / r])
        return vmod * t_hat

    def evaluate(theta0, dv_ign, dt):
        baseY0, t_hat_theta = IC.ICtoY0(IC.rho0, theta0=theta0, delta0=IC.delta0)
        V_ign = dv_ign * t_hat_theta
        Y0 = baseY0 + np.array([0.0, 0.0, V_ign[0], V_ign[1]])

        sol = solve_ivp(
            F, (t0, tf), Y0,
            method="DOP853",
            atol=atol, rtol=rtol,
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