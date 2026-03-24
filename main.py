import numpy as np
import time
from scipy.integrate import solve_ivp
from include import cts, analitical, IC
from include import plotter

print("Initializing simulation, pls wait ...")

nstep = int(1.6e2)
atol = np.array([1e0, 1e0, 1e-4, 1e-4])  # km, km, km/s, km/s
rtol = 1e-6

def R(t, R_orb, frec):
    return [
        R_orb * np.cos(frec * t - IC.delta0),
        R_orb * np.sin(frec * t - IC.delta0),
    ]

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

t0 = 0.0
tf = float(analitical.T_transfer)
dt = (tf - t0) / nstep
t = np.linspace(t0, tf, nstep + 1, endpoint=True)

V_ign = analitical.deltaV_ignI * IC.t_hat_theta
Y0 = IC.Y0 + np.array([0.0, 0.0, V_ign[0], V_ign[1]])

t1 = time.time()

sol = solve_ivp(
    F, (t0, tf), Y0,
    t_eval=t,
    method="DOP853",
    atol=atol, rtol=rtol
)

t2 = time.time()

r = np.hypot(sol.y[0], sol.y[1])
print("runtime =", t2 - t1)
print("r_max =", r.max(), "target =", cts.R_orb_B)

plotter.plot2D(sol.t, dt, sol.y, R)