import numpy as np
from . import cts
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plotsim(sol, R_fun, dt):

    t = sol.t
    Y = sol.y

    var_labels = [' $Y_0 = X$ ', ' $Y_1 = Y$ ', ' $Y_2 = Vx$ ', ' $Y_3 = Vy$ ']
    var_units = ['km', 'km', 'km/s', 'km/s']

    # Plots de variables dinámicas
    for i in range(len(var_labels)):
        plt.figure()
        plt.plot(t, Y[i, :], color='black', label=var_labels[i])
        plt.xlabel('time (s)')
        plt.ylabel(var_units[i])
        plt.title(r"Solution for the dynamical variable " + var_labels[i],
                  fontsize=14, color='gray')
        plt.legend()
        plt.grid()

    # Trayectoria de la nave
    x1 = Y[0, :]
    y1 = Y[1, :]

    # Trayectoria de la Tierra
    Rx = cts.R_orb_A * np.cos(cts.frec * t)
    Ry = cts.R_orb_A * np.sin(cts.frec * t)

    fig, ax = plt.subplots()

    # Escala del dibujo
    max_range = max(
        np.max(np.abs(x1)),
        np.max(np.abs(y1)),
        np.max(np.abs(Rx)),
        np.max(np.abs(Ry))
    )

    # Para que la Tierra se vea: radio exagerado solo en visualización
    R_plot = 3000 * cts.Rt

    def animate(i):
        ax.clear()
        ax.grid()

        # Sol
        ax.plot(0, 0, 'o', markersize=8, color='orange', label='Sun')

        # Órbita de la Tierra completa
        ax.plot(Rx, Ry, '--', color='blue', alpha=0.5, label='Earth orbit')

        # Tierra en su posición actual
        earth = plt.Circle((Rx[i], Ry[i]), R_plot, color='blue', alpha=0.6)
        ax.add_patch(earth)

        # Trayectoria de la nave
        ax.plot(x1[:i+1], y1[:i+1], '-', color='red', label='Spacecraft')
        ax.plot(x1[i], y1[i], 'o', color='red', markersize=5)

        ax.set_xlim(-1.1 * max_range, 1.1 * max_range)
        ax.set_ylim(-1.1 * max_range, 1.1 * max_range)
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_title('Orbital simulation (2D)')
        ax.text(0.02, 0.95, f'time = {t[i]:.1f} s', transform=ax.transAxes)

        if i == 0:
            ax.legend()

        return []

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(t),
        interval=50,
        blit=False
    )

    plt.show()