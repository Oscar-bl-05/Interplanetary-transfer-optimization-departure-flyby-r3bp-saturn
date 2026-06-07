import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from include import cts, analitical, IC

# Case I optimum
theta_test = -0.4814677511186809
dv_test = 7.322992225754916

# Case II optimum
theta_test2 = -0.37405
dv_test2 = 7.269035


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


def enter_SOI(t, Y):
    Rx, Ry = R(t, cts.R_orb_A, cts.frec_A)
    return np.hypot(Y[0] - Rx, Y[1] - Ry) - cts.earth_SOI_radius


enter_SOI.terminal = False
enter_SOI.direction = -1


def exit_SOI(t, Y):
    Rx, Ry = R(t, cts.R_orb_A, cts.frec_A)
    return np.hypot(Y[0] - Rx, Y[1] - Ry) - cts.earth_SOI_radius


exit_SOI.terminal = False
exit_SOI.direction = 1


def impact_earth(t, Y):
    Rx, Ry = R(t, cts.R_orb_A, cts.frec_A)
    return np.hypot(Y[0] - Rx, Y[1] - Ry) - cts.R_Earth


impact_earth.terminal = True
impact_earth.direction = -1

def initial_state(theta0, dv_ign):
    baseY0, t_hat_theta = IC.ICtoY0(
        IC.rho0,
        theta0=theta0,
        delta0=IC.delta0,
    )

    V_ign = dv_ign * t_hat_theta

    Y0 = baseY0.copy()
    Y0[2:4] += V_ign

    return Y0


def earth_distance_at(t, Y):
    Rx, Ry = R(t, cts.R_orb_A, cts.frec_A)
    return np.hypot(Y[0] - Rx, Y[1] - Ry)


def closest_earth_approach(sol, t_SOI_in, t_SOI_out):
    def distance_to_earth(t):
        return earth_distance_at(t, sol.sol(t))

    res = minimize_scalar(
        distance_to_earth,
        bounds=(t_SOI_in, t_SOI_out),
        method="bounded",
    )

    return float(res.fun), float(res.x)


def hit_state_caseI(atol, rtol):
    Y0 = initial_state(theta_test, dv_test)
    tf = float(analitical.T_transfer_case_I)
    dt = (tf - 0.0) / 3500.0

    sol = solve_ivp(
        F,
        (0.0, tf),
        Y0,
        method="DOP853",
        atol=atol,
        rtol=rtol,
        events=reach_RB,
        max_step=dt,
        dense_output=True,
    )

    if len(sol.t_events[0]) == 0:
        return None

    return sol.t_events[0][0], sol.y_events[0][0]


def caseII_reference(atol, rtol):
    Y0 = initial_state(theta_test2, dv_test2)
    tf = 22.0 * cts.year2seconds
    dt = min((tf - 0.0) / 7000.0, 2.0*24.0*3600.0)

    sol = solve_ivp(
        F,
        (0.0, tf),
        Y0,
        method="DOP853",
        atol=atol,
        rtol=rtol,
        events=(reach_RB, impact_earth, enter_SOI, exit_SOI),
        max_step=dt,
        dense_output=True,
    )

    if len(sol.t_events[1]) > 0:
        return None

    if len(sol.t_events[0]) == 0:
        return None

    entries = [float(t) for t in sol.t_events[2] if float(t) > 0.5*cts.year2seconds]
    exits = []

    if len(entries) > 0:
        exits = [float(t) for t in sol.t_events[3] if float(t) > entries[0]]

    if len(entries) == 0 or len(exits) == 0:
        return None

    t_SOI_in = entries[0]
    t_SOI_out = exits[0]

    MED, t_MED = closest_earth_approach(sol, t_SOI_in, t_SOI_out)

    return {
        "t_hit": float(sol.t_events[0][0]),
        "y_hit": sol.y_events[0][0],
        "t_SOI_in": float(t_SOI_in),
        "t_SOI_out": float(t_SOI_out),
        "Y_SOI_out": sol.sol(t_SOI_out),
        "MED": float(MED),
        "t_MED": float(t_MED),
    }


def caseII_pre_state(atol, rtol, t_SOI_out):
    Y0 = initial_state(theta_test2, dv_test2)
    dt = max((t_SOI_out - 0.0) / 5000.0, 1.0)

    sol = solve_ivp(
        F,
        (0.0, t_SOI_out),
        Y0,
        method="DOP853",
        atol=atol,
        rtol=rtol,
        max_step=dt,
    )

    if not sol.success:
        return None

    return sol.y[:, -1]


def caseII_post_hit(atol, rtol, t_SOI_out, Y_SOI_out, t_bound):
    dt = max((t_bound - t_SOI_out) / 5000.0, 1.0)

    sol = solve_ivp(
        F,
        (t_SOI_out, t_bound),
        Y_SOI_out,
        method="DOP853",
        atol=atol,
        rtol=rtol,
        events=reach_RB,
        max_step=dt,
    )

    if len(sol.t_events[0]) == 0:
        return None

    return sol.t_events[0][0], sol.y_events[0][0]


def print_error_line(atol, rtol, y, y_ref):
    pos_err_km = np.hypot(y[0] - y_ref[0], y[1] - y_ref[1])
    vel_err_ms = 1000.0 * np.hypot(y[2] - y_ref[2], y[3] - y_ref[3])

    ok = (pos_err_km <= 1000.0) and (vel_err_ms <= 10.0)

    print(
        "atol", atol,
        "and rtol", rtol,
        "pos_err_km =", pos_err_km,
        "| vel_err_m/s =", vel_err_ms,
        "| OK =", ok
    )

    return ok, pos_err_km, vel_err_ms


# Error check setup

atol_ref = np.array([1e-6, 1e-6, 1e-10, 1e-10])
rtol_ref = 1e-12

candidates = [
    (np.array([1e-4, 1e-4, 1e-8, 1e-8]), 1e-11),
    (np.array([1e-3, 1e-3, 1e-7, 1e-7]), 1e-10),
    (np.array([1e-2, 1e-2, 1e-6, 1e-6]), 1e-9),
    (np.array([1e-1, 1e-1, 1e-5, 1e-5]), 1e-8),
    (np.array([5e-1, 5e-1, 5e-5, 5e-5]), 1e-7),
    (np.array([1.0, 1.0, 1e-4, 1e-4]), 1e-6),
]

# CASE I

print("\n### CASE I ###\n")

ref = hit_state_caseI(atol_ref, rtol_ref)

if ref is None:
    raise SystemExit("No reach R_B in Case I reference run.")

t_ref, y_ref = ref

print("Reference hit time (years):", t_ref / cts.year2seconds)

best = None

for atol, rtol in candidates:
    out = hit_state_caseI(atol, rtol)

    if out is None:
        print("atol", atol, "rtol", rtol, "-> did not reach R_B")
        continue

    t, y = out
    ok, pos_err, vel_err = print_error_line(atol, rtol, y, y_ref)

    if ok:
        best = (atol, rtol)

print("\nSelected Case I tolerances:", best)

# CASE II

print("\n\n### CASE II ###\n")
print("Using fixed optimized trajectory, not reoptimizing inside err_check.py")
print("theta_test2 =", theta_test2)
print("dv_test2 =", dv_test2)

ref2 = caseII_reference(atol_ref, rtol_ref)

if ref2 is None:
    raise SystemExit("No valid Case II reference trajectory. Check theta_test2/dv_test2.")

print("Reference hit time (years):", ref2["t_hit"] / cts.year2seconds)
print("Reference SOI entry (years):", ref2["t_SOI_in"] / cts.year2seconds)
print("Reference SOI exit (years):", ref2["t_SOI_out"] / cts.year2seconds)
print("Reference minimum flyby altitude (km):", ref2["MED"] - cts.R_Earth)

# En Case II se revisan dos cosas:
# 1) error acumulado hasta la salida de la SOI;
# 2) error del tramo post-flyby hasta R_B, empezando desde el mismo estado de SOI-out.
# Esto evita volver a optimizar dentro del error check y a la vez controla el tramo sensible del flyby.

best_caseII = None

t_bound_post = ref2["t_hit"] + 0.10*cts.year2seconds
post_ref = caseII_post_hit(
    atol_ref,
    rtol_ref,
    ref2["t_SOI_out"],
    ref2["Y_SOI_out"],
    t_bound_post,
)

if post_ref is None:
    raise SystemExit("Case II post-flyby reference did not reach R_B.")

t_post_ref, y_post_ref = post_ref

for atol, rtol in candidates:
    pre_state = caseII_pre_state(
        atol,
        rtol,
        ref2["t_SOI_out"],
    )

    if pre_state is None:
        print("atol", atol, "rtol", rtol, "-> pre-flyby arc failed")
        continue

    pre_pos_err_km = np.hypot(pre_state[0] - ref2["Y_SOI_out"][0], pre_state[1] - ref2["Y_SOI_out"][1])
    pre_vel_err_ms = 1000.0*np.hypot(pre_state[2] - ref2["Y_SOI_out"][2], pre_state[3] - ref2["Y_SOI_out"][3])

    post = caseII_post_hit(
        atol,
        rtol,
        ref2["t_SOI_out"],
        ref2["Y_SOI_out"],
        t_bound_post,
    )

    if post is None:
        print("atol", atol, "rtol", rtol, "-> post-flyby arc did not reach R_B")
        continue

    t_post, y_post = post

    post_pos_err_km = np.hypot(y_post[0] - y_post_ref[0], y_post[1] - y_post_ref[1])
    post_vel_err_ms = 1000.0*np.hypot(y_post[2] - y_post_ref[2], y_post[3] - y_post_ref[3])

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
        "| OK =", ok,
    )

    if ok:
        best_caseII = (atol, rtol)

print("\nSelected Case II tolerances:", best_caseII)
