"""Tolerance check for the Earth-Saturn Case I trajectory."""

import numpy as np
from scipy.integrate import solve_ivp

from include import cts, analitical, dynamics

# Known Case I candidate that reaches R_B.
theta_test = -0.4814677511186809
dv_test = 7.322992225754916


def hit_state(atol, rtol):
    t0 = 0.0
    tf = float(analitical.T_transfer_case_I)
    dt = (tf - t0) / 160.0

    Y0 = dynamics.apply_ignition_delta_v(theta_test, dv_test)
    reach_RB = dynamics.make_reach_radius_event(cts.R_orb_B, direction=1, terminal=True)

    sol = solve_ivp(
        dynamics.F,
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

    return float(sol.t_events[0][0]), sol.y_events[0][0]


def main():
    atol_ref = np.array([1e-6, 1e-6, 1e-10, 1e-10])
    rtol_ref = 1e-12

    ref = hit_state(atol_ref, rtol_ref)
    if ref is None:
        raise SystemExit("No reach R_B even in reference run. Increase tf or review test candidate.")

    t_ref, y_ref = ref

    candidates = [
        (np.array([1e-2, 1e-2, 1e-6, 1e-6]), 1e-9),
        (np.array([1e-1, 1e-1, 1e-5, 1e-5]), 1e-8),
        (np.array([5e-1, 5e-1, 5e-5, 5e-5]), 1e-7),
        (np.array([1.0, 1.0, 1e-4, 1e-4]), 1e-6),
    ]

    print("Reference hit time (years):", t_ref / cts.YEAR_TO_S)

    best = None

    for atol, rtol in candidates:
        out = hit_state(atol, rtol)
        if out is None:
            print("atol", atol, "rtol", rtol, "-> did not reach R_B")
            continue

        _, y = out

        pos_err_km = np.hypot(y[0] - y_ref[0], y[1] - y_ref[1])
        vel_err_ms = 1000.0 * np.hypot(y[2] - y_ref[2], y[3] - y_ref[3])

        ok = (pos_err_km <= 1000.0) and (vel_err_ms <= 10.0)
        print(
            "atol", atol,
            "and rtol", rtol,
            "pos_err_km =", pos_err_km,
            "| vel_err_m/s =", vel_err_ms,
            "| OK =", ok,
        )

        if ok:
            best = (atol, rtol)

    print("\nSelected tolerances:", best)


if __name__ == "__main__":
    main()
