import numpy as np

# Parámetros gravitacionales (km^3/s^2)
mu_sun = 1.32712440018e11
mu_earth = 3.986004418e5
mu_saturn = 3.7931187e7

# Parámetros orbitales
R_orb_A = 1.496e8        # km (radio orbital medio Tierra-Sol)
T_orb_A = 3.156e7        # s
Rt = 6371                # km

frec = 2*np.pi/T_orb_A

R_orb_B = 1.4267254e9    # km (radio orbital medio Saturno)
T_orb_B = 9.29e8         # s

# Órbita inicial LEO
rho0 = Rt + 400          # km (400 km de altitud)
v0_mod = np.sqrt(mu_earth / rho0)
theta0 = np.deg2rad(-90)

# Cts 5.1(Parámetros de integración)
coc = R_orb_B/(R_orb_A*(R_orb_A+R_orb_B))
sec = mu_sun/R_orb_A
deltaV_1H = np.sqrt(2*mu_sun*coc) - np.sqrt(sec)
deltaV_ignI = np.sqrt(deltaV_1H*deltaV_1H + 2*mu_earth/rho0)
