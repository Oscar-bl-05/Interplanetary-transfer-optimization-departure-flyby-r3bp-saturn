## INITIAL CONDITIONS ##
from . import cts
import numpy as np

# Órbita inicial LEO
rho0 = cts.R_Earth + 400          # km (400 km de altitud)
v0_mod = np.sqrt(cts.mu_earth / rho0)
theta0 = -90 * cts.deg2rad

# Posición heliocéntrica inicial (km)
r0 = np.array([
    cts.R_orb_A + rho0*np.cos(theta0),
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
    np.sqrt(cts.mu_sun / cts.R_orb_A)
])

# Velocidad total heliocéntrica (km/s)
v0 = v_rel + v_earth_0

# Función de variación de velocidad (debemos aumentar por lo que usaremos el +)



# Y inicial
#vector Y: x,y,v_x,v_y

Y0 = np.array([r0[0],r0[1], v0[0], v0[1]])

