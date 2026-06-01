"""Initial conditions for the spacecraft.

The spacecraft starts in a circular orbit around planet A (Earth), at radius
rho0 = 1.1 * R_Earth, and then receives an impulsive delta-v parallel to its
relative tangential velocity.
"""

import numpy as np

from . import cts

# Default initial relative distance from Earth [km]
rho0 = 1.1 * cts.R_Earth

# Default angular parameters [rad]
theta0 = 0.0
delta0 = 0.0


def tangential_unit(theta):
    """Tangential unit vector for polar angle theta in the orbital plane."""
    return np.array([-np.sin(theta), np.cos(theta)], dtype=float)


def earth_velocity_at_t0(delta=delta0):
    """Earth heliocentric velocity at t = 0 [km/s]."""
    v_earth_0_mod = np.sqrt(cts.mu_sun / cts.R_orb_A)
    return v_earth_0_mod * tangential_unit(delta)


def circular_relative_speed(rho=rho0):
    """Circular speed around Earth at relative radius rho [km/s]."""
    return np.sqrt(cts.mu_earth / rho)


def escape_delta_v_min(rho=rho0):
    """Minimum impulsive delta-v required to escape from circular orbit [km/s]."""
    return (np.sqrt(2.0) - 1.0) * circular_relative_speed(rho)


def ICtoY0(rho0=rho0, theta0=theta0, delta0=delta0):
    """Return heliocentric initial state before the impulsive burn.

    Returns
    -------
    Y0 : ndarray, shape (4,)
        State vector [x, y, vx, vy] in km and km/s.
    t_hat_theta : ndarray, shape (2,)
        Unit vector parallel to the initial relative velocity. The burn is
        applied along this direction for an exterior transfer.
    """
    theta0 = float(theta0)
    rho0 = float(rho0)

    r0 = np.array([
        cts.R_orb_A + rho0 * np.cos(theta0),
        rho0 * np.sin(theta0),
    ], dtype=float)

    t_hat_theta = tangential_unit(theta0)

    v_rel = circular_relative_speed(rho0) * t_hat_theta
    v_earth_0 = earth_velocity_at_t0(delta0)
    v0 = v_rel + v_earth_0

    return np.array([r0[0], r0[1], v0[0], v0[1]], dtype=float), t_hat_theta


def initial_conditions(theta0):
    """Backwards-compatible wrapper around ICtoY0()."""
    return ICtoY0(rho0=rho0, theta0=theta0, delta0=delta0)


# Backwards-compatible module-level defaults.
Y0, t_hat_theta = ICtoY0(rho0=rho0, theta0=theta0, delta0=delta0)
