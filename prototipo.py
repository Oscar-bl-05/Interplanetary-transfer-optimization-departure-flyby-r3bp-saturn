from cts import *

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
