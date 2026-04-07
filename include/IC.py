## INITIAL CONDITIONS ##
from . import cts
import numpy as np

# Órbita inicial LEO
rho0 = cts.R_Earth + 400          # km (400 km de altitud)
theta0 = 0 * cts.deg2rad
delta0 = 0 * cts.deg2rad

def ICtoY0(rho0, theta0 = 0, delta0 = 0):
    v0_mod = np.sqrt(cts.mu_earth / rho0)
    v_earth_0_mod = np.sqrt(cts.mu_sun / cts.R_orb_A)

    # Posición heliocéntrica inicial (km)
    r0 = np.array([
        cts.R_orb_A + rho0*np.cos(theta0),
        rho0*np.sin(theta0)
    ])

    # Velocidad relativa a la Tierra (km/s)
    t_hat_theta = np.array([
        -np.sin(theta0),
        np.cos(theta0)
    ])
    t_hat_delta = np.array([
        -np.sin(delta0),
        np.cos(delta0)
    ])

    v_rel = v0_mod * t_hat_theta

    # Velocidad orbital de la Tierra alrededor del Sol (km/s)
    v_earth_0 = v_earth_0_mod * t_hat_delta
    
    # Velocidad total heliocéntrica (km/s)
    v0 = v_rel + v_earth_0
    # Y inicial
    #vector Y: x,y,v_x,v_y
    return np.array([r0[0],r0[1], v0[0], v0[1]]), t_hat_theta

Y0, t_hat_theta = ICtoY0(rho0, theta0, delta0)
rho0 = cts.R_Earth * 1.1

delta0 = 0 * cts.deg2rad
t_hat_delta = np.array([
    -np.sin(delta0),
    np.cos(delta0)
])

def initial_conditions(theta0):

    v0_mod = np.sqrt(cts.mu_earth / rho0)
    v_earth_0_mod = np.sqrt(cts.mu_sun / cts.R_orb_A)

    # Posición heliocéntrica inicial (km)
    r0 = np.array([
        cts.R_orb_A + rho0*np.cos(theta0),
        rho0*np.sin(theta0)
    ])



    # Velocidad relativa a la Tierra (km/s)
    t_hat_theta = np.array([
        -np.sin(theta0),
        np.cos(theta0)
    ])

    v_rel = v0_mod * t_hat_theta

    # Velocidad orbital de la Tierra alrededor del Sol (km/s)
    v_earth_0 = v_earth_0_mod * t_hat_delta
    
    # Velocidad total heliocéntrica (km/s)
    v0 = v_rel + v_earth_0

    # Función de variación de velocidad (debemos aumentar por lo que usaremos el +)
        #creo que está en analitical

    # Y inicial
    #vector Y: x,y,v_x,v_y

    Y0 = np.array([r0[0],r0[1], v0[0], v0[1]])

    return Y0, t_hat_theta

