"""Shared dynamics and trajectory utility functions."""

import numpy as np

from . import cts, IC


def planet_position(t, R_orb, frec, delta=IC.delta0):
    """Circular heliocentric position of a planet [km]."""
    return np.array([
        R_orb * np.cos(frec * t - delta),
        R_orb * np.sin(frec * t - delta),
    ], dtype=float)


def earth_position(t):
    """Heliocentric position of Earth [km]."""
    return planet_position(t, cts.R_orb_A, cts.frec_A)


def saturn_position(t):
    """Heliocentric position of Saturn in the circular approximation [km]."""
    return planet_position(t, cts.R_orb_B, cts.frec_B)


def F(t, Y):
    """Restricted three-body equations: Sun + Earth acting on spacecraft.

    State vector Y = [x, y, vx, vy], in km and km/s.
    """
    x, y, vx, vy = Y

    r = np.hypot(x, y)
    mu_sun_r3 = cts.mu_sun / (r ** 3)

    Rx, Ry = earth_position(t)
    Rm = np.hypot(Rx, Ry)
    Rm3_inv = 1.0 / (Rm ** 3)

    dx = x - Rx
    dy = y - Ry
    dm = np.hypot(dx, dy)
    dm3_inv = 1.0 / (dm ** 3)

    ax = -x * mu_sun_r3 - cts.mu_earth * (dx * dm3_inv + Rx * Rm3_inv)
    ay = -y * mu_sun_r3 - cts.mu_earth * (dy * dm3_inv + Ry * Rm3_inv)

    return np.array([vx, vy, ax, ay], dtype=float)


def make_reach_radius_event(radius, direction=1, terminal=True):
    """Create a solve_ivp event for heliocentric radius crossing."""
    def event(t, Y):
        return np.hypot(Y[0], Y[1]) - radius

    event.terminal = terminal
    event.direction = direction
    return event


def make_earth_impact_event(safety_radius=cts.R_Earth, direction=-1, terminal=True):
    """Create a solve_ivp event for Earth impact.

    The event triggers when |r - R_Earth(t)| = safety_radius.
    """
    def event(t, Y):
        Rx, Ry = earth_position(t)
        return np.hypot(Y[0] - Rx, Y[1] - Ry) - safety_radius

    event.terminal = terminal
    event.direction = direction
    return event


def make_earth_soi_event(direction=-1, terminal=False):
    """Create a solve_ivp event for crossing Earth's sphere of influence."""
    def event(t, Y):
        Rx, Ry = earth_position(t)
        return np.hypot(Y[0] - Rx, Y[1] - Ry) - cts.earth_SOI_radius

    event.terminal = terminal
    event.direction = direction
    return event


def v_circular_heliocentric_at(x, y):
    """Circular heliocentric velocity at position (x, y) [km/s]."""
    r = np.hypot(x, y)
    vmod = np.sqrt(cts.mu_sun / r)
    t_hat = np.array([-y / r, x / r], dtype=float)
    return vmod * t_hat


def apply_ignition_delta_v(theta0, dv_ign, rho0=IC.rho0, delta0=IC.delta0):
    """Build initial state just after the first impulsive burn."""
    baseY0, t_hat_theta = IC.ICtoY0(rho0=rho0, theta0=theta0, delta0=delta0)
    Y0 = baseY0.copy()
    Y0[2:4] += float(dv_ign) * t_hat_theta
    return Y0


def earth_distance(t, Y):
    """Distance from spacecraft to Earth for a propagated solution [km]."""
    earth_x = cts.R_orb_A * np.cos(cts.frec_A * t - IC.delta0)
    earth_y = cts.R_orb_A * np.sin(cts.frec_A * t - IC.delta0)
    return np.hypot(Y[0] - earth_x, Y[1] - earth_y)


def minimum_earth_distance_after_departure(sol, min_time=0.5 * cts.YEAR_TO_S):
    """Minimum Earth distance after the initial departure phase.

    Returns (minimum_distance_km, time_of_minimum_s). If the propagated
    solution has no samples after min_time, returns (inf, nan).
    """
    mask = sol.t > min_time
    if not np.any(mask):
        return np.inf, np.nan

    d = earth_distance(sol.t[mask], sol.y[:, mask])
    idx = int(np.argmin(d))
    return float(d[idx]), float(sol.t[mask][idx])


def heliocentric_specific_energy(Y):
    """Specific heliocentric mechanical energy [km^2/s^2].

    This ignores Earth potential and is intended as a diagnostic of the
    osculating solar orbit before/after the flyby.
    """
    r = np.hypot(Y[0], Y[1])
    v = np.hypot(Y[2], Y[3])
    return float(0.5 * v * v - cts.mu_sun / r)


def osculating_solar_orbit_diagnostics(Y):
    """Return simple two-body solar-orbit diagnostics for state Y.

    Returns a dictionary with heliocentric radius, speed, specific energy,
    semimajor axis and aphelion radius. If the osculating orbit is unbound,
    a and r_apo are returned as infinity.
    """
    r = float(np.hypot(Y[0], Y[1]))
    v = float(np.hypot(Y[2], Y[3]))
    eps = float(0.5 * v * v - cts.mu_sun / r)

    if eps >= 0.0:
        a = float("inf")
        r_apo = float("inf")
    else:
        a = float(-cts.mu_sun / (2.0 * eps))
        r_apo = float(2.0 * a - r)

    return {
        "r": r,
        "v": v,
        "eps": eps,
        "a": a,
        "r_apo": r_apo,
    }


def planet_velocity(t, R_orb, frec, delta=IC.delta0):
    """Circular heliocentric velocity of a planet [km/s]."""
    return np.array([
        -R_orb * frec * np.sin(frec * t - delta),
        R_orb * frec * np.cos(frec * t - delta),
    ], dtype=float)


def earth_velocity(t):
    """Heliocentric velocity of Earth [km/s]."""
    return planet_velocity(t, cts.R_orb_A, cts.frec_A)


def relative_to_earth_state(t, Y):
    """Return state relative to Earth [x_rel, y_rel, vx_rel, vy_rel]."""
    rE = earth_position(t)
    vE = earth_velocity(t)
    return np.array([Y[0] - rE[0], Y[1] - rE[1], Y[2] - vE[0], Y[3] - vE[1]], dtype=float)


def planar_orbital_elements_from_state(Y, mu):
    """Planar osculating orbital elements from a 2D Cartesian state.

    Parameters
    ----------
    Y : array-like
        State [x, y, vx, vy] in km and km/s relative to the chosen central body.
    mu : float
        Gravitational parameter of the chosen central body [km^3/s^2].

    Returns
    -------
    dict
        e, a, p, rp, omega, energy, h.  Angles are in radians.
        Hyperbolic/parabolic cases keep finite p and rp where possible and use
        a=inf for non-negative specific energy.
    """
    x, y, vx, vy = np.asarray(Y, dtype=float)
    r_vec = np.array([x, y], dtype=float)
    v_vec = np.array([vx, vy], dtype=float)
    r = float(np.hypot(x, y))
    v2 = float(np.dot(v_vec, v_vec))

    if r <= 0.0 or mu <= 0.0:
        return {"e": np.nan, "a": np.nan, "p": np.nan, "rp": np.nan, "omega": np.nan, "energy": np.nan, "h": np.nan}

    h = float(x * vy - y * vx)
    energy = float(0.5 * v2 - mu / r)

    e_vec = ((v2 - mu / r) * r_vec - float(np.dot(r_vec, v_vec)) * v_vec) / mu
    e = float(np.linalg.norm(e_vec))

    if abs(energy) > 1e-14:
        a = float(-mu / (2.0 * energy))
    else:
        a = float("inf")

    p = float((h * h) / mu)
    rp = float(p / (1.0 + e)) if np.isfinite(e) and (1.0 + e) != 0.0 else np.nan
    omega = float(np.arctan2(e_vec[1], e_vec[0])) if e > 1e-12 else 0.0

    return {"e": e, "a": a, "p": p, "rp": rp, "omega": omega, "energy": energy, "h": h}


def planar_orbital_elements_series(t, Y, center="sun"):
    """Compute planar osculating elements along a propagated solution.

    center='sun' uses heliocentric states and mu_sun.
    center='earth' uses Earth-relative states and mu_earth.
    """
    t = np.asarray(t, dtype=float)
    Y = np.asarray(Y, dtype=float)

    e = np.empty_like(t, dtype=float)
    a = np.empty_like(t, dtype=float)
    p = np.empty_like(t, dtype=float)
    rp = np.empty_like(t, dtype=float)
    omega = np.empty_like(t, dtype=float)
    energy = np.empty_like(t, dtype=float)

    if center == "sun":
        mu = cts.mu_sun
        for i in range(len(t)):
            elems = planar_orbital_elements_from_state(Y[:, i], mu)
            e[i] = elems["e"]
            a[i] = elems["a"]
            p[i] = elems["p"]
            rp[i] = elems["rp"]
            omega[i] = elems["omega"]
            energy[i] = elems["energy"]
    elif center == "earth":
        mu = cts.mu_earth
        for i, ti in enumerate(t):
            rel = relative_to_earth_state(ti, Y[:, i])
            elems = planar_orbital_elements_from_state(rel, mu)
            e[i] = elems["e"]
            a[i] = elems["a"]
            p[i] = elems["p"]
            rp[i] = elems["rp"]
            omega[i] = elems["omega"]
            energy[i] = elems["energy"]
    else:
        raise ValueError("center must be 'sun' or 'earth'")

    omega = np.unwrap(omega)
    return {"e": e, "a": a, "p": p, "rp": rp, "omega": omega, "energy": energy}
