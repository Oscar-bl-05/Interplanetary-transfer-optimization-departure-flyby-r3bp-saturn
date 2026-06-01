from math import pi

deg2rad = pi/180
year2seconds = int(365.25 * 24 * 3600)

# Parámetros gravitacionales (km^3/s^2)
mu_sun = int(1.32712440018e11)
mu_earth = 3.986004418e5
mu_saturn = int(3.7931187e7)

# Parámetros orbitales
R_orb_A = int(1.496e8)        # km (radio orbital medio Tierra-Sol)
T_orb_A = int(3.156e7)        # s
R_Earth = 6371                # km

frec_A = 2*pi/T_orb_A

R_orb_B = int(1.4267254e9)    # km (radio orbital medio Saturno)
T_orb_B = int(9.29e8)         # s
frec_B = 2*pi/T_orb_B

earth_SOI_radius = R_orb_A * (mu_earth / mu_sun)**(2.0 / 5.0) # Radio esfera de influencia terrestre
