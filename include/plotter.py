import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from . import cts

def plot2D(t, dt, Y, R_f):

    x1 = Y[0, :]
    y1 = Y[1, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()

    def animate(i):
        ax.clear()
        ax.grid()

        ax.plot(x1[:i], y1[:i], '-', markersize=1, color='red', label="nave")
        ax.plot(x1[i], y1[i], 'o', markersize=5, color='red')

        ax.set_aspect('equal', adjustable='box')

        ax.set_title('Simulación Órbita')
        ax.text(0.02, 0.95, f'time = {t[i]:.1f} s', transform=ax.transAxes)

        # plottear tierra, saturno y sol 
        [RtA_x, RtA_y] = R_f(i*dt, cts.R_orb_A, cts.frec_A)
        ax.plot(RtA_x, RtA_y, 'o', markersize=6, color='blue', label='Tierra')

        [RtB_x, RtB_y] = R_f(i*dt, -cts.R_orb_B, cts.frec_B)
        ax.plot(RtB_x, RtB_y, 'o', markersize=6, color='brown', label='Saturno')


        ax.plot(0, 0, 'o', markersize=8, color='orange', label='Sol')

        return []

    # Animacion de la evolucion temporal
    ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=50) # np.arange(len(t)),

    plt.show()

def plot_solution(t, Y, Y_ref):
      
    t_yr = t / (365.25 * 24 * 3600)

    var_labels = ['$Y_0 = x$', '$Y_1 = y$', '$Y_2 = v_x$', '$Y_3 = v_y$']
    var_units  = ['km', 'km', 'km/s', 'km/s']
    err_labels = ['Error on $Y_0 = x$', 'Error on $Y_1 = y$', 'Error on $Y_2 = v_x$', 'Error on $Y_3 = v_y$']

    # Variables
    plt.subplots(nrows=2, ncols=4, squeeze=True)
    for i in range(4): 
        plt.subplot(2,4,i+1)   
        plt.plot(t_yr, Y[i, :], color='black', label=var_labels[i])
        plt.xlabel('time (years)')
        plt.ylabel(var_units[i])
        plt.title('Solution for the dynamical variable ' + var_labels[i], fontsize=14, color='gray')
        plt.legend()

    # Errores
    for i in range(4):
        plt.subplot(2,4,i+5) 
        plt.plot(t_yr, Y[i, :] - Y_ref[i, :], color='black', label=err_labels[i])
        plt.xlabel('time (years)')
        plt.ylabel(var_units[i])
        plt.title('Error on the solution for ' + var_labels[i], fontsize=12, color='gray')
        plt.legend()

    plt.show()