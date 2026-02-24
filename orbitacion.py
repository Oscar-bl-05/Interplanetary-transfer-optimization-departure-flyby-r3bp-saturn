import numpy as np



# Parámetros
mu_sun = int(1.32712440018e11) # km3/s2
mu_earth = int(3.986004418e5) # km3/s2
mu_saturn = int(3.7931187e7) # km2/s2

R_orb_Earth = 149600000 #km
R_orb_Saturn = 1434000000 #km
T_orb_Earth = int(3.156e7) #s


# Posición del planeta (A)
def R(t):
    frec = 2*np.pi/T_orb_Earth
    
    x_pos = R_orb_Earth*np.cos(frec*t)
    y_pos = R_orb_Earth*np.sin(frec*t)
    #devuelve la posición en x e y como lista
    return [x_pos,y_pos]

    

#función F

def F(t, Y):

    FF = np.zeros_like(Y)        # inicializa FF de manera que sea un array con el mismo numero de componentes que Y, poniendo todas las componentes =0 de entrada
    r = np.sqrt(Y[0]**2+Y[1]**2)
    mu_r3 = -mu_sun/(r*r*r)
    [Rt_x, Rt_y] = R(t) # obtiene la posición del planeta
    Rt = np.sqrt(Rt_x*Rt_x+ Rt_y*Rt_y)
    Rt_3 = 1/(Rt*Rt*Rt)

    rel_pos_x = Y[0]-Rt_x
    rel_pos_y = Y[1]-Rt_y
    rel_pos_mod = np.sqrt(rel_pos_x*rel_pos_x+rel_pos_y*rel_pos_y)
    rel_pos_mod_3 = 1/(rel_pos_mod*rel_pos_mod*rel_pos_mod)

    FF[0] = Y[2]
    FF[1] = Y[3]
    FF[2] = (-Y[0]*mu_r3) - mu_earth*(rel_pos_x*rel_pos_mod_3 + Rt_x*Rt_3)
    FF[3] = (-Y[1]*mu_r3) - mu_earth*(rel_pos_y*rel_pos_mod_3 + Rt_y*Rt_3)

    return FF

#vector Y: x,y,v_x,v_y
