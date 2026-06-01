"""Analytical/PCA estimates for the Earth-Saturn transfer problem.

The numerical propagation is always the source of truth.  This module only
builds physically meaningful search ranges for the optimizers.
"""

from numpy import arcsin, power, sqrt

from . import IC, cts


# ---------------------------------------------------------------------------
# Global simulation-time safety factors
# ---------------------------------------------------------------------------

# Case I used 1.25 in the original work and in the tolerance check.
transfertime_safety_factor = 1.25

# Professor's recommendation for Case II: simulate a bit longer than
# t_total ~= n_earth*T_Earth + T_HohmannDirect.  Use 1.2 by default.
caseII_time_safety_factor = 1.20


# ---------------------------------------------------------------------------
# Direct Hohmann Earth-Saturn estimates
# ---------------------------------------------------------------------------

# Semiperiod of the Hohmann ellipse between the circular orbits of A and B.
T_HOHMANN_HALF = cts.pi * sqrt(
    (cts.R_orb_A + cts.R_orb_B) ** 3 / (8.0 * cts.mu_sun)
)

# Case I: one Hohmann semiperiod, with safety factor.
T_transfer_case_I = transfertime_safety_factor * T_HOHMANN_HALF

# First delta-v of a pure heliocentric Hohmann transfer from A to B.
coc = cts.R_orb_B / (cts.R_orb_A * (cts.R_orb_A + cts.R_orb_B))
deltaV_1H = sqrt(2.0 * cts.mu_sun * coc) - sqrt(cts.mu_sun / cts.R_orb_A)

# Ignition delta-v from the initial circular parking orbit around Earth, using
# the usual patched-conics energy equation.
deltaV_ignI = sqrt(deltaV_1H * deltaV_1H + 2.0 * cts.mu_earth / IC.rho0) - sqrt(
    cts.mu_earth / IC.rho0
)

# Second Hohmann delta-v, used as analytical estimate of final insertion.
coc = cts.R_orb_A / (cts.R_orb_B * (cts.R_orb_A + cts.R_orb_B))
deltaV_2H = sqrt(cts.mu_sun / cts.R_orb_B) - sqrt(2.0 * cts.mu_sun * coc)
deltaV_finI = abs(deltaV_2H)

# Case-I theta estimate: v_inf ~= first Hohmann delta-v.
e_I = 1.0 + (IC.rho0 * deltaV_1H * deltaV_1H) / cts.mu_earth
theta_0I = -2.0 * arcsin(1.0 / e_I)


# ---------------------------------------------------------------------------
# Case II resonant-PCA estimates, following the professor's suggestion
# ---------------------------------------------------------------------------

# Default first test requested by the professor: n = 1, n_earth = 2.
default_resonance_n = 1
default_resonance_n_earth = 2


def resonance_case_II_estimate(n=default_resonance_n, n_earth=default_resonance_n_earth):
    """Return a PCA estimate for a Case-II Earth-return resonance.

    The resonance condition is

        n*T(a) = n_earth*T_Earth,

    with n_earth > n >= 1.  Therefore

        T(a) = (n_earth/n)*T_Earth.

    The first heliocentric ellipse is assumed to start at perihelion near
    Earth's orbit.  Its perihelion speed gives the outgoing v_inf with respect
    to Earth, and this v_inf is converted into the required ignition delta-v
    from the circular parking orbit.
    """
    n = int(n)
    n_earth = int(n_earth)
    if n < 1:
        raise ValueError("n must be >= 1")
    if n_earth <= n:
        raise ValueError("n_earth must be strictly greater than n")

    T_resonance = (n_earth / n) * cts.T_orb_A

    # Kepler's third law: T = 2*pi*sqrt(a^3/mu).
    a = power(cts.mu_sun * (T_resonance / (2.0 * cts.pi)) ** 2, 1.0 / 3.0)

    # If the first point is approximately perihelion at Earth's orbit:
    r_peri = cts.R_orb_A
    r_apo = 2.0 * a - r_peri

    v_perihelion = sqrt(cts.mu_sun * (2.0 / r_peri - 1.0 / a))
    v_earth = sqrt(cts.mu_sun / cts.R_orb_A)
    v_inf = v_perihelion - v_earth

    dv_ign = sqrt(v_inf * v_inf + 2.0 * cts.mu_earth / IC.rho0) - sqrt(
        cts.mu_earth / IC.rho0
    )

    e_hyp = 1.0 + (IC.rho0 * v_inf * v_inf) / cts.mu_earth
    theta = -2.0 * arcsin(1.0 / e_hyp)

    # Professor's suggested simulation time:
    # t_total ~= n_earth*T_Earth + T_HohmannDirect.
    t_total_approx = n_earth * cts.T_orb_A + T_HOHMANN_HALF
    t_sim = caseII_time_safety_factor * t_total_approx

    return {
        "n": n,
        "n_earth": n_earth,
        "T_resonance": float(T_resonance),
        "a": float(a),
        "r_apo": float(r_apo),
        "v_perihelion": float(v_perihelion),
        "v_earth": float(v_earth),
        "v_inf": float(v_inf),
        "dv_ign": float(dv_ign),
        "theta": float(theta),
        "t_total_approx": float(t_total_approx),
        "t_sim": float(t_sim),
        "dv_ign_below_case_I": bool(dv_ign < deltaV_ignI),
    }


def pca_max_aphelion_after_earth_flyby(v_inf):
    """Upper PCA estimate of heliocentric aphelion after an ideal Earth flyby.

    A flyby conserves the magnitude of v_inf in the planetocentric frame and
    mainly rotates that vector.  Therefore, the most optimistic heliocentric
    speed just after Earth encounter is roughly V_Earth + v_inf.  This gives an
    upper bound for the reachable aphelion in the patched-conics picture.

    Returns infinity if the optimistic post-flyby state is heliocentrically
    unbound.
    """
    v_max = sqrt(cts.mu_sun / cts.R_orb_A) + float(v_inf)
    eps = 0.5 * v_max * v_max - cts.mu_sun / cts.R_orb_A
    if eps >= 0.0:
        return float("inf")
    a = -cts.mu_sun / (2.0 * eps)
    return float(2.0 * a - cts.R_orb_A)


def resonance_table(n=1, n_earth_min=2, n_earth_max=12):
    """Return PCA estimates for a range of Earth-return resonances."""
    return [
        resonance_case_II_estimate(n=n, n_earth=n_earth)
        for n_earth in range(int(n_earth_min), int(n_earth_max) + 1)
        if int(n_earth) > int(n)
    ]


# Backwards-compatible module-level defaults: professor's first recommended test.
_default_caseII = resonance_case_II_estimate(
    n=default_resonance_n,
    n_earth=default_resonance_n_earth,
)

resonance_n = _default_caseII["n"]
resonance_n_earth = _default_caseII["n_earth"]
caseII_resonance_earth_years = resonance_n_earth
T_resonance = _default_caseII["T_resonance"]
desired_a = _default_caseII["a"]
desired_R_max = _default_caseII["r_apo"]
v_perihelion_resonance = _default_caseII["v_perihelion"]
v_earth = _default_caseII["v_earth"]
v_infII = _default_caseII["v_inf"]
deltaV_ignII = _default_caseII["dv_ign"]
theta_0II = _default_caseII["theta"]

# Case II default time also follows the professor's formula for the default
# resonance.  Optimizer calls can override tf when scanning other n_earth.
T_transfer_case_II = _default_caseII["t_sim"]

# Alias retrocompatible used by older scripts.
completely_not_pulled_out_of_my_ass_value = deltaV_ignII
