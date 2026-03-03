## INITIAL CONDITIONS ##
import cts
import numpy as np

# Posición heliocéntrica inicial (km)
r0 = np.array([
    cts.R_orb_A + cts.rho0*np.cos(cts.theta0),
    cts.rho0*np.sin(cts.theta0)
])

# Velocidad relativa a la Tierra (km/s)
t_hat = np.array([
    -np.sin(cts.theta0),
     np.cos(cts.theta0)
])

v_rel = cts.v0_mod * t_hat

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

