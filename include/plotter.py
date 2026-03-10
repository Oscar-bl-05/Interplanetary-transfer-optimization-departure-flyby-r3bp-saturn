import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot2D(t, dt, Y):

    x1 = Y[0, :]
    y1 = Y[1, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()

    def animate(i):
        ax.clear()
        ax.grid()

        #ax.plot_surface(x2, y2, color='yellow', alpha=0.4)

        ax.plot(x1[:i], y1[:i], '-', markersize=1, color='red')
        ax.plot(x1[i], y1[i], 'o', markersize=5, color="red")

        ax.set_aspect('equal', adjustable='box')

        time_template = 'time = %.01fs'  # prints running simulation time
        time_text = ax.text(0.05, 0.9, 0.05, '', transform=ax.transAxes)

        time_text.set_text(time_template % (i*dt))
        return time_text

    # Animacion de la evolucion temporal
    ani = animation.FuncAnimation(fig, animate, np.arange(len(t)), interval=1)

    plt.show()