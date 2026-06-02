import numpy as np
from scipy.integrate import solve_ivp
from include import cts, analitical, IC
from include.optimizer import optimize_case_II_resonance_sweep

### CASE I ###
print("\n### CASE I ###\n")
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


### CASE II ###

print("\n\n### CASE II ###")

# Same Case II setup used in main.py
t0_2 = 0.0
tf2 = float(analitical.T_transfer_case_II)

nstep_opt2 = 800 * 4

min_flyby_altitude_km = 300
opt2_n_grid_deltav = 20
opt2_n_grid_theta = 10
opt2_n_refines = 0

caseII_nominal_atol = np.array([1e-2, 1e-2, 1e-6, 1e-6])
caseII_nominal_rtol = 1e-9

print("\nSearching Case II optimum used for error check...")

best_caseII, sweep_results = optimize_case_II_resonance_sweep(
    F=F,
    nstep=nstep_opt2,
    atol=caseII_nominal_atol,
    rtol=caseII_nominal_rtol,
    t0=t0_2,
    n=1,
    n_earth_values=[12],
    n_grid_deltav=opt2_n_grid_deltav,
    n_grid_theta=opt2_n_grid_theta,
    n_refines=opt2_n_refines,
    min_flyby_altitude_km=min_flyby_altitude_km,
    max_flyby_altitude_km=cts.earth_SOI_radius,
)

if best_caseII is None:
    raise SystemExit("No valid Case II solution found for error check.")

theta_test2 = float(best_caseII["theta"])
dv_test2 = float(best_caseII["dv_ign"])
dv_fin_test2 = float(best_caseII["dv_fin"])
dv_tot_test2 = float(best_caseII["dv_tot"])

t_SOI_in_test2 = float(best_caseII["t_SOI_in"])
t_SOI_out_test2 = float(best_caseII["t_SOI_out"])
t_MED_test2 = float(best_caseII["t_MED"])
t_fin_test2 = float(best_caseII["t_fin"])

Y_SOI_out_test2 = np.array(best_caseII["Y_SOI_out"], dtype=float)

baseY0_2, t_hat_theta2 = IC.ICtoY0(
    IC.rho0,
    theta0=theta_test2,
    delta0=IC.delta0,
)

V_ign2 = dv_test2 * t_hat_theta2

Y0_2 = baseY0_2.copy()
Y0_2[2:4] += V_ign2

print("\nCase II optimum used for error check:")
print("theta_test2 =", theta_test2)
print("dv_ign_test2 (km/s) =", dv_test2)
print("dv_fin_test2 (km/s) =", dv_fin_test2)
print("dv_tot_test2 (km/s) =", dv_tot_test2)
print("t_SOI_in_test2 (years) =", t_SOI_in_test2 / cts.year2seconds)
print("t_SOI_out_test2 (years) =", t_SOI_out_test2 / cts.year2seconds)
print("t_MED_test2 (years) =", t_MED_test2 / cts.year2seconds)
print("t_fin_test2 (years) =", t_fin_test2 / cts.year2seconds)


def caseII_pre_flyby_state(atol, rtol):
    """
    Arc 1:
        t0 -> t_SOI_out

    This checks the pre-flyby propagation up to the same SOI exit time
    used by the validated Case II solution.
    """

    if t_SOI_out_test2 <= t0_2:
        raise ValueError(
            "Invalid Case II pre-flyby interval: "
            f"t0 = {t0_2}, t_SOI_out = {t_SOI_out_test2}"
        )

    dt = (t_SOI_out_test2 - t0_2) / 4000.0

    sol = solve_ivp(
        F,
        (t0_2, t_SOI_out_test2),
        Y0_2,
        method="DOP853",
        atol=atol,
        rtol=rtol,
        max_step=dt,
    )

    if not sol.success:
        return None

    return sol.t[-1], sol.y[:, -1]


def caseII_post_flyby_hit_state(atol, rtol):
    # t_SOI_out -> R_B

    t_bound = t_fin_test2 + 0.50 * cts.year2seconds

    if t_bound <= t_SOI_out_test2:
        raise ValueError(
            "Invalid Case II post-flyby interval: "
            f"t_SOI_out = {t_SOI_out_test2}, t_bound = {t_bound}"
        )

    dt = (t_bound - t_SOI_out_test2) / 4000.0

    sol = solve_ivp(
        F,
        (t_SOI_out_test2, t_bound),
        Y_SOI_out_test2,
        method="DOP853",
        atol=atol,
        rtol=rtol,
        events=reach_RB,
        max_step=dt,
    )

    if len(sol.t_events[0]) == 0:
        return None

    return sol.t_events[0][0], sol.y_events[0][0]


print("\nCase II time interval checks:")
print("tf2 analytical upper bound (years) =", tf2 / cts.year2seconds)
print("t_SOI_out_test2 (years) =", t_SOI_out_test2 / cts.year2seconds)
print("t_fin_test2 (years) =", t_fin_test2 / cts.year2seconds)
print("post-flyby validation upper bound (years) =", (t_fin_test2 + 0.50 * cts.year2seconds) / cts.year2seconds)

pre_ref = caseII_pre_flyby_state(atol_ref, rtol_ref)
post_ref = caseII_post_flyby_hit_state(atol_ref, rtol_ref)

if pre_ref is None:
    raise SystemExit("Case II reference pre-flyby arc failed.")

if post_ref is None:
    raise SystemExit("Case II reference post-flyby arc did not reach R_B.")

t_pre_ref, y_pre_ref = pre_ref
t_post_ref, y_post_ref = post_ref

print("\nCase II reference:")
print("reference SOI-out time (years):", t_pre_ref / cts.year2seconds)
print("reference R_B hit time (years):", t_post_ref / cts.year2seconds)

best_caseII_tol = None

for atol, rtol in candidates:
    pre_out = caseII_pre_flyby_state(atol, rtol)
    post_out = caseII_post_flyby_hit_state(atol, rtol)

    if pre_out is None:
        print("atol", atol, "rtol", rtol, "-> pre-flyby arc failed")
        continue

    if post_out is None:
        print("atol", atol, "rtol", rtol, "-> post-flyby arc did not reach R_B")
        continue

    t_pre, y_pre = pre_out
    t_post, y_post = post_out

    pre_pos_err_km = np.hypot(y_pre[0] - y_pre_ref[0], y_pre[1] - y_pre_ref[1])
    pre_vel_err_ms = 1000.0 * np.hypot(y_pre[2] - y_pre_ref[2], y_pre[3] - y_pre_ref[3])

    post_pos_err_km = np.hypot(y_post[0] - y_post_ref[0], y_post[1] - y_post_ref[1])
    post_vel_err_ms = 1000.0 * np.hypot(y_post[2] - y_post_ref[2], y_post[3] - y_post_ref[3])

    hit_time_err_s = abs(t_post - t_post_ref)

    ok_pre = (pre_pos_err_km <= 1000.0) and (pre_vel_err_ms <= 10.0)
    ok_post = (post_pos_err_km <= 1000.0) and (post_vel_err_ms <= 10.0)
    ok = ok_pre and ok_post

    print(
        "atol", atol,
        "and rtol", rtol,
        "| pre_pos_err_km =", pre_pos_err_km,
        "| pre_vel_err_m/s =", pre_vel_err_ms,
        "| post_pos_err_km =", post_pos_err_km,
        "| post_vel_err_m/s =", post_vel_err_ms,
        "| hit_time_err_s =", hit_time_err_s,
        "| OK =", ok,
    )

    if ok:
        best_caseII_tol = (atol, rtol)

print("\nSelected Case II tolerances:", best_caseII_tol)