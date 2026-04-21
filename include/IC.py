## INITIAL CONDITIONS ##
from . import cts
import numpy as np

rho0 = cts.R_Earth * 1.1
theta0 = 0 * cts.deg2rad
delta0 = 0 * cts.deg2rad

def ICtoY0(rho0, theta0=0, delta0=0):
    v0_mod = np.sqrt(cts.mu_earth / rho0)
    v_earth_0_mod = np.sqrt(cts.mu_sun / cts.R_orb_A)

    r0 = np.array([
        cts.R_orb_A + rho0 * np.cos(theta0),
        rho0 * np.sin(theta0)
    ])

    t_hat_theta = np.array([
        -np.sin(theta0),
        np.cos(theta0)
    ])

    t_hat_delta = np.array([
        -np.sin(delta0),
        np.cos(delta0)
    ])

    v_rel = v0_mod * t_hat_theta
    v_earth_0 = v_earth_0_mod * t_hat_delta
    v0 = v_rel + v_earth_0

    return np.array([r0[0], r0[1], v0[0], v0[1]]), t_hat_theta

Y0, t_hat_theta = ICtoY0(rho0, theta0, delta0)