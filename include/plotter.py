#Importación de librerías
import numpy as np
from . import cts
import matplotlib.pyplot as plt
import matplotlib.animation as animation

nstep = 200 #mas steps para mayor precision
tola = 1e-14
tolr = 1e-12

#####################################################
#  Introduce an array-like form as in this example: #
#---------------------------------------------------#
#[t,Y]=[t0,t1,...;[Y0_0,Y0_1,...;Y1_0,Y1_1,...;...]]#
#####################################################
def plotsim(sol, R_fun, dt):

    t = sol.t
    Y = sol.y
    var_labels = [' $Y_0 = X$ ', ' $Y_1 = Y$ ', ' $Y_2 = Vx$ ', ' $Y_3 = Vy$ ']
    var_units = ['km', 'km', 'km/s', 'km/s']
    var_fig = [' Y_0 ', ' Y_1 ', ' Y_2 ', ' Y_3 ']

    # Estas lineas generan el plot de las variables dinamicas
    for i in range(len(var_labels)):
        plt.figure()
        plt.plot(t, Y[i,:], color = 'black', label=var_labels[i]) 
        plt.xlabel('time (s)')
        plt.ylabel(var_units[i])
    #    plt.title(r"Evolución temporal de la variable dinámica"+var_labels[i], fontsize = 12, color = 'gray')
        plt.title(r"Solution for the dynamical variable "+var_labels[i], fontsize = 14, color = 'gray')
        plt.legend()


    #ANIMATION:
    x1 = Y[0, :]
    y1 = Y[1, :]

    theta = np.arange(0, 2*np.pi+0.1, 0.1)
    phi = np.arange(0, np.pi+0.1, 0.1)
    # 
    theta, phi = np.meshgrid(theta, phi)
    #
    [Rx,Ry] = R_fun(t)
    #
    x2 = Rx + cts.Rt * np.cos(theta) * np.sin(phi)
    y2 = Ry + cts.Rt * np.sin(theta) * np.sin(phi)
    z2 = cts.Rt * np.cos(phi)

    #x3 = cts.Rsat * np.cos(theta) * np.sin(phi)
    #y3 = cts.Rsat * np.sin(theta) * np.sin(phi)
    #z3 = cts.Rsat * np.cos(phi)

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, projection='2d')
    ax.grid()
    # 
    # 
    def animate(i):
        ax.clear()
        ax.grid()
    # 
        ax.plot_surface(x2, y2, z2, color='yellow', alpha=0.4)
    # 
        ax.plot(x1[:i], y1[:i], '-', markersize=1, color='red')
        ax.plot(x1[i], y1[i], 'o', markersize=5, color="red")
    # 
        ax.set_aspect('equal', adjustable='box')
    # 
        time_template = 'time = %.01fs'  # prints running simulation time
        time_text = ax.text(0.05, 0.9, 0.05, '', transform=ax.transAxes)
    # 
        time_text.set_text(time_template % (i*dt))
        return time_text
    # 
    # # Animacion de la evolucion temporal
    ani = animation.FuncAnimation(fig, animate, np.arange(len(t)), interval=0.01)
    # 
    plt.show()