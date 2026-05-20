import time

import numpy as np
from scipy.integrate import solve_ivp

from include import cts, analitical, IC
from include import plotter


"""
Case II test script for the Earth-Saturn transfer problem.

"""


print("Initializing simulation, pls wait ...")

# Tolerances selected from err_check.py
atol = np.array([1e-2, 1e-2, 1e-6, 1e-6])
rtol = 1e-9

# Reference tolerances used only for plotting error curves.
# The tolerance validation itself is done in err_check.py.
atol_ref = np.array([1e-6, 1e-6, 1e-10, 1e-10])
rtol_ref = 1e-12

nstep_opt = 750
nstep_plot = 4000

t0 = 0.0
tf = float(analitical.T_transfer_case_II)


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
    Rm3 = 1.0 / (Rm**3)

    dx = x - Rx
    dy = y - Ry
    dm = np.hypot(dx, dy)
    dm3 = 1.0 / (dm**3)

    ax = (-x * mu_r3) - cts.mu_earth * (dx * dm3 + Rx * Rm3)
    ay = (-y * mu_r3) - cts.mu_earth * (dy * dm3 + Ry * Rm3)

    return np.array([vx, vy, ax, ay])


def reach_RB(t, Y):
    return np.hypot(Y[0], Y[1]) - cts.R_orb_B


reach_RB.terminal = True
reach_RB.direction = 1

print("\nInitializing case II simulation")

t_simII_start = time.time()

best_caseII = optimize_case_II(
    F=F,
    nstep=nstep_opt,
    atol=atol,
    rtol=rtol,
    tf=tf,
    t0=t0,
    n_grid_deltav=200,
    n_grid_theta=10,
    n_refines=1,
)

t_simII_end = time.time()

print("Case two optimization time =", t_simII_end-t_simII_start)
if best_caseII != None:
    print("Found best ii")
    print(best_caseII)
else:
    print("No valid solutions found")
