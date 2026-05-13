import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from . import cts

def plot2D(t, dt, Y, R_f): # Plot 2D animation

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

def plot_solution(t, Y, Y_ref): # Plot variables and errors
      
    t_yr = t / (365.25 * 24 * 3600)

    var_labels = ['$x$', '$y$', '$v_x$', '$v_y$']
    var_units  = ['km', 'km', 'km/s', 'km/s']
    err_labels = ['$x - x_{ref}$', '$y - y_{ref}$', '$v_x - v_{x,ref}$', '$v_y - v_{y,ref}$']

    # Dynamic variables
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), constrained_layout=True)
    axes = axes.flatten()

    for i in range(4):
        axes[i].plot(t_yr, Y[i, :], color='black', label=var_labels[i])
        axes[i].set_xlabel('Time [years]')
        axes[i].set_ylabel(var_units[i])
        axes[i].set_title(var_labels[i])
        axes[i].grid(True)
        axes[i].legend()

    fig.suptitle('Dynamical variables', fontsize=14)

    # Numerical errors
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), constrained_layout=True)
    axes = axes.flatten()

    for i in range(4):
        axes[i].plot(t_yr, Y[i, :] - Y_ref[i, :], color='black', label=err_labels[i])
        axes[i].set_xlabel('Time [years]')
        axes[i].set_ylabel(var_units[i])
        axes[i].set_title(err_labels[i])
        axes[i].grid(True)
        axes[i].legend()

    fig.suptitle('Numerical error with respect to reference solution', fontsize=14)

    plt.show()