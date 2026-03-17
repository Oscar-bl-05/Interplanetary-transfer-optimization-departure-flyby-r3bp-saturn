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

        # plottear tierra, sol saturno 
        [RtA_x, RtA_y] = R_f(i*dt, cts.R_orb_A, cts.frec_A)
        ax.plot(RtA_x, RtA_y, 'o', markersize=6, color='blue', label='Tierra')

        [RtB_x, RtB_y] = R_f(i*dt, cts.R_orb_B, cts.frec_B)
        ax.plot(RtB_x, RtB_y, 'o', markersize=6, color='brown', label='Saturno')


        ax.plot(0, 0, 'o', markersize=8, color='orange', label='Sol')

        return []

    # Animacion de la evolucion temporal
    ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=50) # np.arange(len(t)),

    plt.show()