import numpy as np

from include import cts, IC


def compute_planar_orbital_elements(t, Y, R_f=None, center="sun"):

    # elements:
    # e: eccentricity
    # p: semi-latus rectum [km]
    # rp: periapsis radius [km]
    # omega: argument of periapsis [rad]

    if center == "sun":
        x = Y[0, :]
        y = Y[1, :]
        vx = Y[2, :]
        vy = Y[3, :]
        mu = cts.mu_sun

    elif center in ["earth", "planetA"]:
        if R_f is None:
            raise ValueError("R_f must be provided when center='earth'.")

        earth_pos = np.array([R_f(ti, cts.R_orb_A, cts.frec_A) for ti in t])
        earth_x = earth_pos[:, 0]
        earth_y = earth_pos[:, 1]

        earth_vx = -cts.R_orb_A * cts.frec_A * np.sin(cts.frec_A * t - IC.delta0)
        earth_vy =  cts.R_orb_A * cts.frec_A * np.cos(cts.frec_A * t - IC.delta0)

        x = Y[0, :] - earth_x
        y = Y[1, :] - earth_y
        vx = Y[2, :] - earth_vx
        vy = Y[3, :] - earth_vy
        mu = cts.mu_earth

    else:
        raise ValueError("center must be 'sun' or 'earth'.")

    r = np.hypot(x, y)

    # Specific angular momentum, z component
    h = x * vy - y * vx

    # Eccentricity vector in 2D:
    # e_vec = (v x h_vec) / mu - r_vec / r
    e_x = (vy * h) / mu - x / r
    e_y = (-vx * h) / mu - y / r

    e = np.hypot(e_x, e_y)

    # Semi-latus rectum
    p = h**2 / mu

    # Periapsis radius
    rp = p / (1.0 + e)

    # Argument of periapsis
    omega = np.unwrap(np.arctan2(e_y, e_x))

    return {
        "e": e,
        "p": p,
        "rp": rp,
        "omega": omega,
    }