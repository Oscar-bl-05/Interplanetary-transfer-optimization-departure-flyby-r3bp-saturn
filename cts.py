import math

pi = math.pi

deg2rad = math.pi/180

# Parámetros gravitacionales (km^3/s^2)
mu_sun = int(1.32712440018e11)
mu_earth = int(3.986004418e5)
mu_saturn = int(3.7931187e7)

# Parámetros orbitales
R_orb_A = int(1.496e8)        # km (radio orbital medio Tierra-Sol)
T_orb_A = int(3.156e7)        # s
Rt = 6371                # km

frec = 2*math.pi/T_orb_A

R_orb_B = int(1.4267254e9 )   # km (radio orbital medio Saturno)
T_orb_B = int(9.29e8)         # s

# Órbita inicial LEO
rho0 = Rt + 400          # km (400 km de altitud)
v0_mod = math.sqrt(mu_earth / rho0)
theta0 = -90 * deg2rad


