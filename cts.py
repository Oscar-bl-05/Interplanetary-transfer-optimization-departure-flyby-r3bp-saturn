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

# Cts 5.1(Parámetros de integración)
coc = R_orb_B/(R_orb_A*(R_orb_A+R_orb_B))
sec = mu_sun/R_orb_A
deltaV_1H = math.sqrt(2*mu_sun*coc) - math.sqrt(sec)
deltaV_ignI = math.sqrt(deltaV_1H*deltaV_1H + 2*mu_earth/rho0)
deltaV_1H = np.sqrt(2*mu_sun*coc1) - np.sqrt(mu_sun/R_orb_A)
deltaV_ignI = np.sqrt(deltaV_1H*deltaV_1H + 2*mu_earth/rho0)

coc = R_orb_A/(R_orb_B*(R_orb_A+R_orb_B))
deltaV_2H = np.sqrt((mu_sun/R_orb_B)) - np.sqrt(2*mu_sun*coc)
deltaV_finI = abs(deltaV_2H) # Not sure

# Theta de ignición
# Caso I: vinf = deltaV_1H
e = 1 + (rho0*deltaV_1H*deltaV_1H)/mu_earth
theta_0I = -2 * np.arcsin(1/e)
# Caso II: vinf esta contenido entre [0, deltaV_1H]
e = 1 + (rho0*0.8*0.8*deltaV_1H*deltaV_1H)/mu_earth # Asumimos un vinf del 80% de deltaV_1H
theta_0II = -2 * np.arcsin(1/e)

