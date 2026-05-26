import numpy as np
from scipy.integrate import solve_ivp
from include import cts, analitical, IC

k_test = 1  # asegurar que llega a R_B
theta_test = -0.4814677511186809
dv_test = 7.322992225754916

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
    Rm3 = 1.0 / (Rm*Rm*Rm)

    dx = x - Rx
    dy = y - Ry
    dm = np.hypot(dx, dy)
    dm3 = 1.0 / (dm*dm*dm)

    ax = (-x * mu_r3) - cts.mu_earth * (dx * dm3 + Rx * Rm3)
    ay = (-y * mu_r3) - cts.mu_earth * (dy * dm3 + Ry * Rm3)

    return np.array([vx, vy, ax, ay])

def reach_RB(t, Y):
    return np.hypot(Y[0], Y[1]) - cts.R_orb_B

reach_RB.terminal = True
reach_RB.direction = 1

def hit_state(atol, rtol):
    t0 = 0.0
    tf = float(analitical.T_transfer_case_I)
    dt = (tf - t0) / 160.0

    baseY0, t_hat_theta = IC.ICtoY0(
    IC.rho0,
    theta0=theta_test,
    delta0=IC.delta0
)

    V_ign = dv_test * t_hat_theta
    Y0 = baseY0.copy()
    Y0[2:4] += V_ign

    sol = solve_ivp(
        F, (t0, tf), Y0,
        method="DOP853",
        atol=atol, rtol=rtol,
        events=reach_RB,
        max_step=dt
    )

    if len(sol.t_events[0]) == 0:
        return None

    return sol.t_events[0][0], sol.y_events[0][0]

atol_ref = np.array([1e-6, 1e-6, 1e-10, 1e-10])
rtol_ref = 1e-12
ref = hit_state(atol_ref, rtol_ref)
if ref is None:
    raise SystemExit("No reach R_B even in reference run (increase k_test or tf).")

t_ref, y_ref = ref

candidates = [
    (np.array([1e-2, 1e-2, 1e-6, 1e-6]), 1e-9),
    (np.array([1e-1, 1e-1, 1e-5, 1e-5]), 1e-8),
    (np.array([5e-1, 5e-1, 5e-5, 5e-5]), 1e-7),
    (np.array([1.0, 1.0, 1e-4, 1e-4]), 1e-6),
]

print("Reference hit time (years):", t_ref / (365.25 * 24 * 3600))

best = None

for atol, rtol in candidates:
    out = hit_state(atol, rtol)
    if out is None:
        print("atol", atol, "rtol", rtol, "-> did not reach R_B")
        continue

    t, y = out

    pos_err_km = np.hypot(y[0] - y_ref[0], y[1] - y_ref[1])
    vel_err_ms = 1000.0 * np.hypot(y[2] - y_ref[2], y[3] - y_ref[3])

    ok = (pos_err_km <= 1000.0) and (vel_err_ms <= 10.0)
    print("atol", atol, " and rtol", rtol, " pos_err_km =", pos_err_km, "| vel_err_m/s =", vel_err_ms, "| OK =", ok)

    if ok:
        best = (atol, rtol)

print("\nSelected tolerances:", best)