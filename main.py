import numpy as np
import time
from scipy.integrate import solve_ivp
from include import cts, analitical, IC

nstep = 200 #mas steps para mayor precision
tola = 1e-14
tolr = 1e-12

# Posición del planeta (A)
def R(t):
    x_pos = cts.R_orb_A*np.cos(cts.frec*t)
    y_pos = cts.R_orb_A*np.sin(cts.frec*t)
    #devuelve la posición en x e y como lista
    return [x_pos,y_pos]

#función F
def F(t, Y):

    FF = np.zeros_like(Y)        # inicializa FF de manera que sea un array con el mismo numero de componentes que Y, poniendo todas las componentes =0 de entrada
    y0 = Y[0]
    y1 = Y[1]
    r = np.sqrt(y0*y0+y1*y1)
    mu_r3 = -cts.mu_sun/(r*r*r)
    [Rt_x, Rt_y] = R(t) # obtiene la posición del planeta
    Rt = np.sqrt(Rt_x*Rt_x+ Rt_y*Rt_y)
    Rt_3 = 1/(Rt*Rt*Rt)

    rel_pos_x = y0-Rt_x
    rel_pos_y = y1-Rt_y
    rel_pos_mod = np.sqrt(rel_pos_x*rel_pos_x+rel_pos_y*rel_pos_y)
    rel_pos_mod_3 = 1/(rel_pos_mod*rel_pos_mod*rel_pos_mod)

    FF[0] = Y[2]
    FF[1] = Y[3]
    FF[2] = (-y0*mu_r3) - cts.mu_earth*(rel_pos_x*rel_pos_mod_3 + Rt_x*Rt_3)
    FF[3] = (-y1*mu_r3) - cts.mu_earth*(rel_pos_y*rel_pos_mod_3 + Rt_y*Rt_3)

    return FF

#vector Y: x,y,v_x,v_y

t1 = time.time()

t0 = 0
tf = analitical.T_transfer

dt = (tf-t0)/nstep

t = np.linspace(t0,tf,nstep+1, endpoint = True) 
sol = solve_ivp(F, (t0,tf), IC.Y0, t_eval=t, method='DOP853', atol=tola, rtol=tolr)   

Y = sol.y
t2 = time.time()        
print('tiempo de ejec. =',t2-t1)
