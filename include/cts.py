"""Physical constants for the Earth-Saturn transfer problem.

Units used throughout the project:
- distance: km
- time: s
- velocity: km/s
- gravitational parameter: km^3/s^2
"""

from math import pi

# Unit conversions
deg2rad = pi / 180.0
DAY_TO_S = 24.0 * 3600.0
YEAR_TO_S = 365.25 * DAY_TO_S

# Standard gravitational parameters [km^3/s^2]
mu_sun = 1.32712440018e11
mu_earth = 3.986004418e5
mu_saturn = 3.7931187e7

# Planet A: Earth
R_orb_A = 1.496e8        # Mean heliocentric orbital radius [km]
T_orb_A = 3.156e7        # Orbital period [s]
R_Earth = 6371.0         # Mean radius [km]
frec_A = 2.0 * pi / T_orb_A

# Planet B: Saturn
R_orb_B = 1.4267254e9    # Mean heliocentric orbital radius [km]
T_orb_B = 9.29e8         # Orbital period [s]
frec_B = 2.0 * pi / T_orb_B

# Earth sphere of influence radius [km]
earth_SOI_radius = R_orb_A * (mu_earth / mu_sun) ** (2.0 / 5.0)

# Backwards-compatible alias used by older scripts.
year_to_s = YEAR_TO_S
