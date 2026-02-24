import numpy as np

# Parámetros gravitacionales (km^3/s^2)
mu_sun = 1.32712440018e11
mu_earth = 3.986004418e5
mu_saturn = 3.7931187e7

# Parámetros orbitales (km y s)
R_orb_A = 1.496e8        # km (radio orbital medio Tierra-Sol)
T_orb_A = 3.156e7        # s
Rt = 6371                # km

R_orb_B = 1.4267254e9    # km (radio orbital medio Saturno)
T_orb_B = 9.29e8         # s

# Órbita inicial LEO
rho0 = Rt + 400          # km (400 km de altitud)

v0_mod = np.sqrt(mu_earth / rho0)

theta0 = np.deg2rad(-90)

# Posición heliocéntrica inicial (km)
r0 = np.array([
    R_orb_A + rho0*np.cos(theta0),
    rho0*np.sin(theta0)
])

# Velocidad relativa a la Tierra (km/s)
t_hat = np.array([
    -np.sin(theta0),
     np.cos(theta0)
])

v_rel = v0_mod * t_hat

# Velocidad orbital de la Tierra alrededor del Sol (km/s)
v_earth_0 = np.array([
    0,
    np.sqrt(mu_sun / R_orb_A)
])

# Velocidad total heliocéntrica (km/s)
v0 = v_rel + v_earth_0

# Función de variación de velocidad (debemos aumentar por lo que usaremos el +)
v
